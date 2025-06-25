
#include "spx_importer_wav.h"

#include "core/config/project_settings.h"
#include "core/io/marshalls.h"
#include "scene/resources/audio_stream_wav.h"

#include "core/io/resource_importer.h"

#define DR_WAV_IMPLEMENTATION
#include "core/extension/spx_importer_dr_wav.h"

const float TRIM_DB_LIMIT = -50;
const int TRIM_FADE_OUT_FRAMES = 500;


static const char *get_format_name(drwav_uint16 formatTag) {
	switch (formatTag) {
		case 1:
			return "PCM";
		case 2:
			return "Microsoft ADPCM";
		case 3:
			return "IEEE Float";
		case 6:
			return "A-Law";
		case 7:
			return "μ-Law";
		case 17:
			return "IMA ADPCM";
		case 20:
			return "Yamaha ADPCM";
		case 49:
			return "GSM 6.10";
		case 50:
			return "G.721 ADPCM";
		case 64:
			return "G.722 ADPCM";
		case 65:
			return "G.723 ADPCM";
		case 80:
			return "MPEG";
		case 85:
			return "MPEG Layer 3";
		case 65534:
			return "Extensible";
		default:
			return "Unknown";
	}
}
Error SpxImporterWav::import_asset(Ref<AudioStreamWAV> &sample, const String &p_source_file) {
	List<ResourceImporter::ImportOption> r_options;
	// load default option
	r_options.push_back(ResourceImporter::ImportOption(PropertyInfo(Variant::BOOL, "force/8_bit"), false));
	r_options.push_back(ResourceImporter::ImportOption(PropertyInfo(Variant::BOOL, "force/mono"), false));
	r_options.push_back(ResourceImporter::ImportOption(PropertyInfo(Variant::BOOL, "force/max_rate", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), false));
	r_options.push_back(ResourceImporter::ImportOption(PropertyInfo(Variant::FLOAT, "force/max_rate_hz", PROPERTY_HINT_RANGE, "11025,192000,1,exp"), 44100));
	r_options.push_back(ResourceImporter::ImportOption(PropertyInfo(Variant::BOOL, "edit/trim"), false));
	r_options.push_back(ResourceImporter::ImportOption(PropertyInfo(Variant::BOOL, "edit/normalize"), false));
	r_options.push_back(ResourceImporter::ImportOption(PropertyInfo(Variant::INT, "edit/loop_mode", PROPERTY_HINT_ENUM, "Detect From WAV,Disabled,Forward,Ping-Pong,Backward", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), 0));
	r_options.push_back(ResourceImporter::ImportOption(PropertyInfo(Variant::INT, "edit/loop_begin"), 0));
	r_options.push_back(ResourceImporter::ImportOption(PropertyInfo(Variant::INT, "edit/loop_end"), -1));
	r_options.push_back(ResourceImporter::ImportOption(PropertyInfo(Variant::INT, "compress/mode", PROPERTY_HINT_ENUM, "Disabled,RAM (Ima-ADPCM)"), 0));
	HashMap<StringName, Variant> options_map;
	for (const ResourceImporter::ImportOption &E : r_options) {
		options_map[E.option.name] = E.default_value;
	}
	return import_asset(sample, p_source_file, options_map);
}
Error SpxImporterWav::import_asset(Ref<AudioStreamWAV> &sample, const String &p_source_file, const HashMap<StringName, Variant> &p_options) {
	// Read file using dr_wav
	String actual_path = ProjectSettings::get_singleton()->globalize_path(p_source_file);

	// print_line("Starting to read WAV file: " + actual_path);

	drwav wav;
	if (!drwav_init_file(&wav, actual_path.utf8().get_data(), NULL)) {
		ERR_FAIL_V_MSG(ERR_CANT_OPEN, "Cannot open WAV file '" + actual_path + "'.");
	}

	// Get audio format information
	int format_channels = wav.channels;
	int format_freq = wav.sampleRate;
	int format_bits = wav.bitsPerSample;
	drwav_uint64 frames = wav.totalPCMFrameCount;
	bool is16 = format_bits != 8;

	// Print audio format information
	// print_line("Audio format information:");
	// print_line("  Sample rate: " + String::num_int64(wav.sampleRate) + " Hz");
	// print_line("  Channels: " + String::num_int64(wav.channels));
	// print_line("  Bit depth: " + String::num_int64(wav.bitsPerSample) + " bits");
	// print_line("  Total frames: " + String::num_int64(wav.totalPCMFrameCount));
	// print_line("  Data chunk size: " + String::num_int64(wav.dataChunkDataSize) + " bytes");
	// print_line("  Format tag: " + String::num_int64(wav.translatedFormatTag) + " (" + String(get_format_name(wav.translatedFormatTag)) + ")");
	// print_line("  Container type: " + String::num_int64(wav.container));

	// Allocate buffer
	drwav_uint64 bufferSize = frames * format_channels;
	Vector<float> data;
	data.resize(bufferSize);

	// Select appropriate reading method based on format
	drwav_uint64 framesRead = 0;

	switch (wav.translatedFormatTag) {
		case 1: // PCM
		case 3: // IEEE Float
		case 6: // A-Law
		case 7: // μ-Law
		{
			if (wav.bitsPerSample == 16) {
				Vector<int16_t> temp_data;
				temp_data.resize(bufferSize);
				framesRead = drwav_read_pcm_frames_s16(&wav, frames, temp_data.ptrw());
				for (int i = 0; i < framesRead * format_channels; i++) {
					data.write[i] = temp_data[i] / 32768.f;
				}
			} else if (wav.bitsPerSample == 32) {
				Vector<float> temp_data;
				temp_data.resize(bufferSize);
				framesRead = drwav_read_pcm_frames_f32(&wav, frames, temp_data.ptrw());
				data = temp_data;
			} else {
				Vector<uint8_t> temp_data;
				temp_data.resize(bufferSize);
				size_t bytesRead = drwav_read_raw(&wav, bufferSize, temp_data.ptrw());
				framesRead = bytesRead / format_channels;
				for (int i = 0; i < framesRead * format_channels; i++) {
					data.write[i] = (temp_data[i] - 128) / 128.f;
				}
			}
			break;
		}

		case 2: // Microsoft ADPCM
		case 17: // IMA ADPCM
		case 20: // Yamaha ADPCM
		case 50: // G.721 ADPCM
		case 64: // G.722 ADPCM
		case 65: // G.723 ADPCM
		{
			Vector<int16_t> temp_data;
			temp_data.resize(bufferSize);
			framesRead = drwav_read_pcm_frames_s16(&wav, frames, temp_data.ptrw());
			// print_line("ADPCM frames read: " + String::num_int64(framesRead));
			for (int i = 0; i < framesRead * format_channels; i++) {
				data.write[i] = temp_data[i] / 32768.f;
			}
			break;
		}

		case 49: // GSM 6.10
		case 80: // MPEG
		case 85: // MPEG Layer 3
		{
			WARN_PRINT("Warning: This format (" + String(get_format_name(wav.translatedFormatTag)) + ") may not be fully supported");
			// Try raw reading
			Vector<uint8_t> temp_data;
			temp_data.resize(wav.dataChunkDataSize);
			size_t bytesRead = drwav_read_raw(&wav, wav.dataChunkDataSize, temp_data.ptrw());
			if (bytesRead > 0) {
				// Convert raw data to float format
				for (int i = 0; i < bytesRead; i++) {
					data.write[i] = temp_data[i] / 128.f;
				}
				framesRead = bytesRead / format_channels;
			}
			break;
		}

		default:
			ERR_FAIL_V_MSG(ERR_INVALID_DATA, "Unsupported format: " + String(get_format_name(wav.translatedFormatTag)));
	}

	// Release dr_wav resources
	drwav_uninit(&wav);

	// if (framesRead == 0) {
	// 	ERR_FAIL_V_MSG(ERR_INVALID_DATA, "No data read, format may be unsupported or file may be corrupted");
	// }

	// Apply conversion options
	bool limit_rate = p_options["force/max_rate"];
	int limit_rate_hz = p_options["force/max_rate_hz"];
	if (limit_rate && format_freq > limit_rate_hz && format_freq > 0 && framesRead > 0) {
		// Resampling code remains unchanged
		int new_data_frames = (int)(framesRead * (float)limit_rate_hz / (float)format_freq);
		Vector<float> new_data;
		new_data.resize(new_data_frames * format_channels);
		for (int c = 0; c < format_channels; c++) {
			float frac = .0f;
			int ipos = 0;

			for (int i = 0; i < new_data_frames; i++) {
				float mu = frac;
				float y0 = data[MAX(0, ipos - 1) * format_channels + c];
				float y1 = data[ipos * format_channels + c];
				float y2 = data[MIN(framesRead - 1, ipos + 1) * format_channels + c];
				float y3 = data[MIN(framesRead - 1, ipos + 2) * format_channels + c];

				float mu2 = mu * mu;
				float a0 = y3 - y2 - y0 + y1;
				float a1 = y0 - y1 - a0;
				float a2 = y2 - y0;
				float a3 = y1;

				float res = (a0 * mu * mu2 + a1 * mu2 + a2 * mu + a3);
				new_data.write[i * format_channels + c] = res;

				frac += (float)format_freq / (float)limit_rate_hz;
				int tpos = (int)Math::floor(frac);
				ipos += tpos;
				frac -= tpos;
			}
		}

		data = new_data;
		format_freq = limit_rate_hz;
		framesRead = new_data_frames;
	}

	// Apply other conversion options
	bool normalize = p_options["edit/normalize"];
	if (normalize) {
		float max = 0;
		for (int i = 0; i < data.size(); i++) {
			float amp = Math::abs(data[i]);
			if (amp > max) {
				max = amp;
			}
		}

		if (max > 0) {
			float mult = 1.0 / max;
			for (int i = 0; i < data.size(); i++) {
				data.write[i] *= mult;
			}
		}
	}

	bool trim = p_options["edit/trim"];
	if (trim && format_channels > 0) {
		int first = 0;
		int last = (framesRead / format_channels) - 1;
		bool found = false;
		float limit = Math::db_to_linear(TRIM_DB_LIMIT);

		for (int i = 0; i < data.size() / format_channels; i++) {
			float ampChannelSum = 0;
			for (int j = 0; j < format_channels; j++) {
				ampChannelSum += Math::abs(data[(i * format_channels) + j]);
			}

			float amp = Math::abs(ampChannelSum / (float)format_channels);

			if (!found && amp > limit) {
				first = i;
				found = true;
			}

			if (found && amp > limit) {
				last = i;
			}
		}

		if (first < last) {
			Vector<float> new_data;
			new_data.resize((last - first) * format_channels);
			for (int i = first; i < last; i++) {
				float fadeOutMult = 1;

				if (last - i < TRIM_FADE_OUT_FRAMES) {
					fadeOutMult = ((float)(last - i - 1) / (float)TRIM_FADE_OUT_FRAMES);
				}

				for (int j = 0; j < format_channels; j++) {
					new_data.write[((i - first) * format_channels) + j] = data[(i * format_channels) + j] * fadeOutMult;
				}
			}

			data = new_data;
			framesRead = data.size() / format_channels;
		}
	}

	// Handle loop mode
	AudioStreamWAV::LoopMode loop_mode = AudioStreamWAV::LOOP_DISABLED;
	int loop_begin = 0;
	int loop_end = 0;

	int import_loop_mode = p_options["edit/loop_mode"];
	if (import_loop_mode >= 2) {
		loop_mode = (AudioStreamWAV::LoopMode)(import_loop_mode - 1);
		loop_begin = p_options["edit/loop_begin"];
		loop_end = p_options["edit/loop_end"];
		if (loop_begin < 0) {
			loop_begin = CLAMP(loop_begin + framesRead + 1, 0, framesRead);
		}
		if (loop_end < 0) {
			loop_end = CLAMP(loop_end + framesRead + 1, 0, framesRead);
		}
	}

	// Handle compression and channel options
	int compression = p_options["compress/mode"];
	bool force_mono = p_options["force/mono"];
	bool force_8_bit = p_options["force/8_bit"];

	if (force_mono && format_channels == 2) {
		Vector<float> new_data;
		new_data.resize(data.size() / 2);
		for (int i = 0; i < framesRead; i++) {
			new_data.write[i] = (data[i * 2 + 0] + data[i * 2 + 1]) / 2.0;
		}
		data = new_data;
		format_channels = 1;
	}

	if (force_8_bit) {
		is16 = false;
	}

	// Prepare output data
	Vector<uint8_t> dst_data;
	AudioStreamWAV::Format dst_format;

	if (compression == 1) {
		dst_format = AudioStreamWAV::FORMAT_IMA_ADPCM;
		if (format_channels == 1) {
			_compress_ima_adpcm(data, dst_data);
		} else {
			Vector<float> left;
			Vector<float> right;
			int tframes = data.size() / 2;
			left.resize(tframes);
			right.resize(tframes);

			for (int i = 0; i < tframes; i++) {
				left.write[i] = data[i * 2 + 0];
				right.write[i] = data[i * 2 + 1];
			}

			Vector<uint8_t> bleft;
			Vector<uint8_t> bright;
			_compress_ima_adpcm(left, bleft);
			_compress_ima_adpcm(right, bright);

			int dl = bleft.size();
			dst_data.resize(dl * 2);
			uint8_t *w = dst_data.ptrw();
			const uint8_t *rl = bleft.ptr();
			const uint8_t *rr = bright.ptr();

			for (int i = 0; i < dl; i++) {
				w[i * 2 + 0] = rl[i];
				w[i * 2 + 1] = rr[i];
			}
		}
	} else {
		dst_format = is16 ? AudioStreamWAV::FORMAT_16_BITS : AudioStreamWAV::FORMAT_8_BITS;
		dst_data.resize(data.size() * (is16 ? 2 : 1));
		uint8_t *w = dst_data.ptrw();

		int ds = data.size();
		for (int i = 0; i < ds; i++) {
			if (is16) {
				int16_t v = CLAMP(data[i] * 32768, -32768, 32767);
				encode_uint16(v, &w[i * 2]);
			} else {
				int8_t v = CLAMP(data[i] * 128, -128, 127);
				w[i] = v;
			}
		}
	}

	// Set output samples
	sample.instantiate();
	sample->set_data(dst_data);
	sample->set_format(dst_format);
	sample->set_mix_rate(format_freq);
	sample->set_loop_mode(loop_mode);
	sample->set_loop_begin(loop_begin);
	sample->set_loop_end(loop_end);
	sample->set_stereo(format_channels == 2);

	return OK;
}
