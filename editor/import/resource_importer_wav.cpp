/*************************************************************************/
/*  resource_importer_wav.cpp                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "resource_importer_wav.h"

#include "core/io/marshalls.h"
#include "core/io/resource_saver.h"
#include "core/os/file_access.h"
#include "scene/resources/audio_stream_sample.h"

const float TRIM_DB_LIMIT = -50;
const int TRIM_FADE_OUT_FRAMES = 500;

String ResourceImporterWAV::get_importer_name() const {
	return "wav";
}

String ResourceImporterWAV::get_visible_name() const {
	return "Microsoft WAV";
}
void ResourceImporterWAV::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("wav");
}
String ResourceImporterWAV::get_save_extension() const {
	return "sample";
}

String ResourceImporterWAV::get_resource_type() const {
	return "AudioStreamSample";
}

bool ResourceImporterWAV::get_option_visibility(const String &p_option, const Map<StringName, Variant> &p_options) const {
	if (p_option == "force/max_rate_hz" && !bool(p_options["force/max_rate"])) {
		return false;
	}

	return true;
}

int ResourceImporterWAV::get_preset_count() const {
	return 0;
}
String ResourceImporterWAV::get_preset_name(int p_idx) const {
	return String();
}

void ResourceImporterWAV::get_import_options(List<ImportOption> *r_options, int p_preset) const {
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "force/8_bit"), false));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "force/mono"), false));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "force/max_rate", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), false));
	r_options->push_back(ImportOption(PropertyInfo(Variant::REAL, "force/max_rate_hz", PROPERTY_HINT_EXP_RANGE, "11025,192000,1"), 44100));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "edit/trim"), false));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "edit/normalize"), false));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "edit/loop"), false));
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "compress/mode", PROPERTY_HINT_ENUM, "Disabled,RAM (Ima-ADPCM)"), 0));
}

Error ResourceImporterWAV::import(const String &p_source_file, const String &p_save_path, const Map<StringName, Variant> &p_options, List<String> *r_platform_variants, List<String> *r_gen_files, Variant *r_metadata) {
	/* STEP 1, READ WAVE FILE */

	Error err;
	FileAccess *file = FileAccess::open(p_source_file, FileAccess::READ, &err);

	ERR_FAIL_COND_V_MSG(err != OK, ERR_CANT_OPEN, "Cannot open file '" + p_source_file + "'.");

	/* CHECK RIFF */
	char riff[5];
	riff[4] = 0;
	file->get_buffer((uint8_t *)&riff, 4); //RIFF

	if (riff[0] != 'R' || riff[1] != 'I' || riff[2] != 'F' || riff[3] != 'F') {
		file->close();
		memdelete(file);
		ERR_FAIL_V(ERR_FILE_UNRECOGNIZED);
	}

	/* GET FILESIZE */
	file->get_32(); // filesize

	/* CHECK WAVE */

	char wave[4];

	file->get_buffer((uint8_t *)&wave, 4); //RIFF

	if (wave[0] != 'W' || wave[1] != 'A' || wave[2] != 'V' || wave[3] != 'E') {
		file->close();
		memdelete(file);
		ERR_FAIL_V_MSG(ERR_FILE_UNRECOGNIZED, "Not a WAV file (no WAVE RIFF header).");
	}

	int format_bits = 0;
	int format_channels = 0;

	AudioStreamSample::LoopMode loop = AudioStreamSample::LOOP_DISABLED;
	uint16_t compression_code = 1;
	bool format_found = false;
	bool data_found = false;
	int format_freq = 0;
	int loop_begin = 0;
	int loop_end = 0;
	int frames = 0;

	Vector<float> data;

	while (!file->eof_reached()) {
		/* chunk */
		char chunkID[4];
		file->get_buffer((uint8_t *)&chunkID, 4); //RIFF

		/* chunk size */
		uint32_t chunksize = file->get_32();
		uint32_t file_pos = file->get_position(); //save file pos, so we can skip to next chunk safely

		if (file->eof_reached()) {
			//ERR_PRINT("EOF REACH");
			break;
		}

		if (chunkID[0] == 'f' && chunkID[1] == 'm' && chunkID[2] == 't' && chunkID[3] == ' ' && !format_found) {
			/* IS FORMAT CHUNK */

			//Issue: #7755 : Not a bug - usage of other formats (format codes) are unsupported in current importer version.
			//Consider revision for engine version 3.0
			compression_code = file->get_16();
			if (compression_code != 1 && compression_code != 3) {
				file->close();
				memdelete(file);
				ERR_FAIL_V_MSG(ERR_INVALID_DATA, "Format not supported for WAVE file (not PCM). Save WAVE files as uncompressed PCM instead.");
			}

			format_channels = file->get_16();
			if (format_channels != 1 && format_channels != 2) {
				file->close();
				memdelete(file);
				ERR_FAIL_V_MSG(ERR_INVALID_DATA, "Format not supported for WAVE file (not stereo or mono).");
			}

			format_freq = file->get_32(); //sampling rate

			file->get_32(); // average bits/second (unused)
			file->get_16(); // block align (unused)
			format_bits = file->get_16(); // bits per sample

			if (format_bits % 8 || format_bits == 0) {
				file->close();
				memdelete(file);
				ERR_FAIL_V_MSG(ERR_INVALID_DATA, "Invalid amount of bits in the sample (should be one of 8, 16, 24 or 32).");
			}

			/* Don't need anything else, continue */
			format_found = true;
		}

		if (chunkID[0] == 'd' && chunkID[1] == 'a' && chunkID[2] == 't' && chunkID[3] == 'a' && !data_found) {
			/* IS DATA CHUNK */
			data_found = true;

			if (!format_found) {
				ERR_PRINT("'data' chunk before 'format' chunk found.");
				break;
			}

			frames = chunksize;

			if (format_channels == 0) {
				file->close();
				memdelete(file);
				ERR_FAIL_COND_V(format_channels == 0, ERR_INVALID_DATA);
			}
			frames /= format_channels;
			frames /= (format_bits >> 3);

			/*print_line("chunksize: "+itos(chunksize));
			print_line("channels: "+itos(format_channels));
			print_line("bits: "+itos(format_bits));
			*/

			data.resize(frames * format_channels);

			if (format_bits == 8) {
				for (int i = 0; i < frames * format_channels; i++) {
					// 8 bit samples are UNSIGNED

					data.write[i] = int8_t(file->get_8() - 128) / 128.f;
				}
			} else if (format_bits == 32 && compression_code == 3) {
				for (int i = 0; i < frames * format_channels; i++) {
					//32 bit IEEE Float

					data.write[i] = file->get_float();
				}
			} else if (format_bits == 16) {
				for (int i = 0; i < frames * format_channels; i++) {
					//16 bit SIGNED

					data.write[i] = int16_t(file->get_16()) / 32768.f;
				}
			} else {
				for (int i = 0; i < frames * format_channels; i++) {
					//16+ bits samples are SIGNED
					// if sample is > 16 bits, just read extra bytes

					uint32_t s = 0;
					for (int b = 0; b < (format_bits >> 3); b++) {
						s |= ((uint32_t)file->get_8()) << (b * 8);
					}
					s <<= (32 - format_bits);

					data.write[i] = (int32_t(s) >> 16) / 32768.f;
				}
			}

			if (file->eof_reached()) {
				file->close();
				memdelete(file);
				ERR_FAIL_V_MSG(ERR_FILE_CORRUPT, "Premature end of file.");
			}
		}

		if (chunkID[0] == 's' && chunkID[1] == 'm' && chunkID[2] == 'p' && chunkID[3] == 'l') {
			//loop point info!

			/**
			 *	Consider exploring next document:
			 *		http://www-mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/Docs/RIFFNEW.pdf
			 *	Especially on page:
			 *		16 - 17
			 *	Timestamp:
			 *		22:38 06.07.2017 GMT
			 **/

			for (int i = 0; i < 10; i++) {
				file->get_32(); // i wish to know why should i do this... no doc!
			}

			// only read 0x00 (loop forward), 0x01 (loop ping-pong) and 0x02 (loop backward)
			// Skip anything else because it's not supported, reserved for future uses or sampler specific
			// from https://sites.google.com/site/musicgapi/technical-documents/wav-file-format#smpl (loop type values table)
			int loop_type = file->get_32();
			if (loop_type == 0x00 || loop_type == 0x01 || loop_type == 0x02) {
				if (loop_type == 0x00) {
					loop = AudioStreamSample::LOOP_FORWARD;
				} else if (loop_type == 0x01) {
					loop = AudioStreamSample::LOOP_PING_PONG;
				} else if (loop_type == 0x02) {
					loop = AudioStreamSample::LOOP_BACKWARD;
				}
				loop_begin = file->get_32();
				loop_end = file->get_32();
			}
		}
		file->seek(file_pos + chunksize);
	}

	file->close();
	memdelete(file);

	// STEP 2, APPLY CONVERSIONS

	bool is16 = format_bits != 8;
	int rate = format_freq;

	/*
	print_line("Input Sample: ");
	print_line("\tframes: " + itos(frames));
	print_line("\tformat_channels: " + itos(format_channels));
	print_line("\t16bits: " + itos(is16));
	print_line("\trate: " + itos(rate));
	print_line("\tloop: " + itos(loop));
	print_line("\tloop begin: " + itos(loop_begin));
	print_line("\tloop end: " + itos(loop_end));
	*/

	//apply frequency limit

	bool limit_rate = p_options["force/max_rate"];
	int limit_rate_hz = p_options["force/max_rate_hz"];
	if (limit_rate && rate > limit_rate_hz && rate > 0 && frames > 0) {
		// resample!
		int new_data_frames = (int)(frames * (float)limit_rate_hz / (float)rate);

		Vector<float> new_data;
		new_data.resize(new_data_frames * format_channels);
		for (int c = 0; c < format_channels; c++) {
			float frac = .0f;
			int ipos = 0;

			for (int i = 0; i < new_data_frames; i++) {
				//simple cubic interpolation should be enough.

				float mu = frac;

				float y0 = data[MAX(0, ipos - 1) * format_channels + c];
				float y1 = data[ipos * format_channels + c];
				float y2 = data[MIN(frames - 1, ipos + 1) * format_channels + c];
				float y3 = data[MIN(frames - 1, ipos + 2) * format_channels + c];

				float mu2 = mu * mu;
				float a0 = y3 - y2 - y0 + y1;
				float a1 = y0 - y1 - a0;
				float a2 = y2 - y0;
				float a3 = y1;

				float res = (a0 * mu * mu2 + a1 * mu2 + a2 * mu + a3);

				new_data.write[i * format_channels + c] = res;

				// update position and always keep fractional part within ]0...1]
				// in order to avoid 32bit floating point precision errors

				frac += (float)rate / (float)limit_rate_hz;
				int tpos = (int)Math::floor(frac);
				ipos += tpos;
				frac -= tpos;
			}
		}

		if (loop) {
			loop_begin = (int)(loop_begin * (float)new_data_frames / (float)frames);
			loop_end = (int)(loop_end * (float)new_data_frames / (float)frames);
		}

		data = new_data;
		rate = limit_rate_hz;
		frames = new_data_frames;
	}

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

	if (trim && !loop && format_channels > 0) {
		int first = 0;
		int last = (frames / format_channels) - 1;
		bool found = false;
		float limit = Math::db2linear(TRIM_DB_LIMIT);

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
			frames = data.size() / format_channels;
		}
	}

	bool make_loop = p_options["edit/loop"];

	if (make_loop && !loop) {
		loop = AudioStreamSample::LOOP_FORWARD;
		loop_begin = 0;
		loop_end = frames;
	}

	int compression = p_options["compress/mode"];
	bool force_mono = p_options["force/mono"];

	if (force_mono && format_channels == 2) {
		Vector<float> new_data;
		new_data.resize(data.size() / 2);
		for (int i = 0; i < frames; i++) {
			new_data.write[i] = (data[i * 2 + 0] + data[i * 2 + 1]) / 2.0;
		}

		data = new_data;
		format_channels = 1;
	}

	bool force_8_bit = p_options["force/8_bit"];
	if (force_8_bit) {
		is16 = false;
	}

	PoolVector<uint8_t> dst_data;
	AudioStreamSample::Format dst_format;

	if (compression == 1) {
		dst_format = AudioStreamSample::FORMAT_IMA_ADPCM;
		if (format_channels == 1) {
			_compress_ima_adpcm(data, dst_data);
		} else {
			//byte interleave
			Vector<float> left;
			Vector<float> right;

			int tframes = data.size() / 2;
			left.resize(tframes);
			right.resize(tframes);

			for (int i = 0; i < tframes; i++) {
				left.write[i] = data[i * 2 + 0];
				right.write[i] = data[i * 2 + 1];
			}

			PoolVector<uint8_t> bleft;
			PoolVector<uint8_t> bright;

			_compress_ima_adpcm(left, bleft);
			_compress_ima_adpcm(right, bright);

			int dl = bleft.size();
			dst_data.resize(dl * 2);

			PoolVector<uint8_t>::Write w = dst_data.write();
			PoolVector<uint8_t>::Read rl = bleft.read();
			PoolVector<uint8_t>::Read rr = bright.read();

			for (int i = 0; i < dl; i++) {
				w[i * 2 + 0] = rl[i];
				w[i * 2 + 1] = rr[i];
			}
		}

	} else {
		dst_format = is16 ? AudioStreamSample::FORMAT_16_BITS : AudioStreamSample::FORMAT_8_BITS;
		dst_data.resize(data.size() * (is16 ? 2 : 1));
		{
			PoolVector<uint8_t>::Write w = dst_data.write();

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
	}

	Ref<AudioStreamSample> sample;
	sample.instance();
	sample->set_data(dst_data);
	sample->set_format(dst_format);
	sample->set_mix_rate(rate);
	sample->set_loop_mode(loop);
	sample->set_loop_begin(loop_begin);
	sample->set_loop_end(loop_end);
	sample->set_stereo(format_channels == 2);

	ResourceSaver::save(p_save_path + ".sample", sample);

	return OK;
}

ResourceImporterWAV::ResourceImporterWAV() {
}
