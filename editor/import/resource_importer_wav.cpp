/**************************************************************************/
/*  resource_importer_wav.cpp                                             */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "resource_importer_wav.h"

#include "core/io/file_access.h"
#include "core/io/marshalls.h"
#include "core/io/resource_saver.h"
#include "scene/resources/audio_stream_wav.h"

#define DRWAV_IMPLEMENTATION
#define DR_WAV_NO_STDIO
#define DR_WAV_LIBSNDFILE_COMPAT
#define DRWAV_MALLOC(sz) memalloc(sz)
#define DRWAV_REALLOC(p, sz) memrealloc(p, sz)
#define DRWAV_FREE(p) memfree(p)

#include "thirdparty/dr_libs/dr_wav.h"

const float TRIM_DB_LIMIT = -50;
const int TRIM_FADE_OUT_FRAMES = 500;

String ResourceImporterWAV::get_importer_name() const {
	return "wav";
}

String ResourceImporterWAV::get_visible_name() const {
	return "Microsoft WAV/Apple AIFF";
}

void ResourceImporterWAV::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("wav");
	p_extensions->push_back("wave");
	p_extensions->push_back("aif");
	p_extensions->push_back("aiff");
	p_extensions->push_back("aifc");
}

String ResourceImporterWAV::get_save_extension() const {
	return "sample";
}

String ResourceImporterWAV::get_resource_type() const {
	return "AudioStreamWAV";
}

bool ResourceImporterWAV::get_option_visibility(const String &p_path, const String &p_option, const HashMap<StringName, Variant> &p_options) const {
	if (p_option == "force/max_rate_hz" && !bool(p_options["force/max_rate"])) {
		return false;
	}

	// Don't show begin/end loop points if loop mode is auto-detected or disabled.
	if ((int)p_options["edit/loop_mode"] < 2 && (p_option == "edit/loop_begin" || p_option == "edit/loop_end")) {
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

void ResourceImporterWAV::get_import_options(const String &p_path, List<ImportOption> *r_options, int p_preset) const {
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "force/8_bit"), false));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "force/mono"), false));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "force/max_rate", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), false));
	r_options->push_back(ImportOption(PropertyInfo(Variant::FLOAT, "force/max_rate_hz", PROPERTY_HINT_RANGE, "11025,192000,1,exp"), 44100));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "edit/trim"), false));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "edit/normalize"), false));
	// Keep the `edit/loop_mode` enum in sync with AudioStreamWAV::LoopMode (note: +1 offset due to "Detect From WAV").
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "edit/loop_mode", PROPERTY_HINT_ENUM, "Detect From WAV,Disabled,Forward,Ping-Pong,Backward", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), 0));
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "edit/loop_begin"), 0));
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "edit/loop_end"), -1));
	// Quite OK Audio is lightweight enough and supports virtually every significant AudioStreamWAV feature.
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "compress/mode", PROPERTY_HINT_ENUM, "PCM (Uncompressed),IMA ADPCM,Quite OK Audio"), 2));
}

Error ResourceImporterWAV::import(const String &p_source_file, const String &p_save_path, const HashMap<StringName, Variant> &p_options, List<String> *r_platform_variants, List<String> *r_gen_files, Variant *r_metadata) {
	// STEP 1, READ FILE

	Error err;
	Vector<uint8_t> file = FileAccess::get_file_as_bytes(p_source_file, &err);

	ERR_FAIL_COND_V_MSG(err != OK, ERR_CANT_OPEN, "Cannot open file '" + p_source_file + "'.");

	drwav wav;
	if (!drwav_init_memory_with_metadata(&wav, file.ptr(), file.size(), DRWAV_WITH_METADATA, nullptr)) {
		ERR_FAIL_V_MSG(ERR_FILE_UNRECOGNIZED, "Could not read file '" + p_source_file + "'. Invalid/corrupted data or unsupported format.");
	}

	if (wav.totalPCMFrameCount > INT32_MAX) {
		ERR_FAIL_V_MSG(ERR_FILE_UNRECOGNIZED, "Could not read file '" + p_source_file + "'. Audio data exceeds maximum size of 2,147,483,647 frames.");
	}

	int format_bits = wav.bitsPerSample;
	int format_channels = wav.channels;
	int format_freq = wav.sampleRate;
	int frames = wav.totalPCMFrameCount;

	int import_loop_mode = p_options["edit/loop_mode"];
	int loop_begin = 0;
	int loop_end = 0;
	AudioStreamWAV::LoopMode loop_mode = AudioStreamWAV::LOOP_DISABLED;
	if (import_loop_mode == 0) {
		for (uint32_t meta = 0; meta < wav.metadataCount; meta++) {
			drwav_metadata md = wav.pMetadata[meta];
			if (md.type == drwav_metadata_type_smpl && md.data.smpl.sampleLoopCount) {
				drwav_smpl_loop loop = md.data.smpl.pLoops[0];
				if (loop.type == drwav_smpl_loop_type_forward)
					loop_mode = AudioStreamWAV::LOOP_FORWARD;
				else if (loop.type == drwav_smpl_loop_type_pingpong)
					loop_mode = AudioStreamWAV::LOOP_PINGPONG;
				else if (loop.type == drwav_smpl_loop_type_backward)
					loop_mode = AudioStreamWAV::LOOP_BACKWARD;
				loop_begin = loop.firstSampleByteOffset;
				loop_end = loop.lastSampleByteOffset;
			}
		}
	}

	Vector<float> data;
	data.resize(wav.totalPCMFrameCount * wav.channels);
	drwav_read_pcm_frames_f32(&wav, wav.totalPCMFrameCount, data.ptrw());

	drwav_uninit(&wav);
	file.clear();

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
				// Cubic interpolation should be enough.

				float y0 = data[MAX(0, ipos - 1) * format_channels + c];
				float y1 = data[ipos * format_channels + c];
				float y2 = data[MIN(frames - 1, ipos + 1) * format_channels + c];
				float y3 = data[MIN(frames - 1, ipos + 2) * format_channels + c];

				new_data.write[i * format_channels + c] = Math::cubic_interpolate(y1, y2, y0, y3, frac);

				// update position and always keep fractional part within ]0...1]
				// in order to avoid 32bit floating point precision errors

				frac += (float)rate / (float)limit_rate_hz;
				int tpos = (int)Math::floor(frac);
				ipos += tpos;
				frac -= tpos;
			}
		}

		if (loop_mode) {
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

	if (trim && (loop_mode == AudioStreamWAV::LOOP_DISABLED) && format_channels > 0) {
		int first = 0;
		int last = (frames / format_channels) - 1;
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
			frames = data.size() / format_channels;
		}
	}

	if (import_loop_mode >= 2) {
		loop_mode = (AudioStreamWAV::LoopMode)(import_loop_mode - 1);
		loop_begin = p_options["edit/loop_begin"];
		loop_end = p_options["edit/loop_end"];
		// Wrap around to max frames, so `-1` can be used to select the end, etc.
		if (loop_begin < 0) {
			loop_begin = CLAMP(loop_begin + frames, 0, frames - 1);
		}
		if (loop_end < 0) {
			loop_end = CLAMP(loop_end + frames, 0, frames - 1);
		}
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

	Vector<uint8_t> pcm_data;
	AudioStreamWAV::Format dst_format;

	if (compression == 1) {
		dst_format = AudioStreamWAV::FORMAT_IMA_ADPCM;
		if (format_channels == 1) {
			_compress_ima_adpcm(data, pcm_data);
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

			Vector<uint8_t> bleft;
			Vector<uint8_t> bright;

			_compress_ima_adpcm(left, bleft);
			_compress_ima_adpcm(right, bright);

			int dl = bleft.size();
			pcm_data.resize(dl * 2);

			uint8_t *w = pcm_data.ptrw();
			const uint8_t *rl = bleft.ptr();
			const uint8_t *rr = bright.ptr();

			for (int i = 0; i < dl; i++) {
				w[i * 2 + 0] = rl[i];
				w[i * 2 + 1] = rr[i];
			}
		}

	} else {
		dst_format = is16 ? AudioStreamWAV::FORMAT_16_BITS : AudioStreamWAV::FORMAT_8_BITS;
		bool enforce16 = is16 || compression == 2;
		pcm_data.resize(data.size() * (enforce16 ? 2 : 1));
		{
			uint8_t *w = pcm_data.ptrw();

			int ds = data.size();
			for (int i = 0; i < ds; i++) {
				if (enforce16) {
					int16_t v = CLAMP(data[i] * 32768, -32768, 32767);
					encode_uint16(v, &w[i * 2]);
				} else {
					int8_t v = CLAMP(data[i] * 128, -128, 127);
					w[i] = v;
				}
			}
		}
	}

	Vector<uint8_t> dst_data;
	if (compression == 2) {
		dst_format = AudioStreamWAV::FORMAT_QOA;
		qoa_desc desc = {};
		uint32_t qoa_len = 0;

		desc.samplerate = rate;
		desc.samples = frames;
		desc.channels = format_channels;

		void *encoded = qoa_encode((short *)pcm_data.ptr(), &desc, &qoa_len);
		if (encoded) {
			dst_data.resize(qoa_len);
			memcpy(dst_data.ptrw(), encoded, qoa_len);
			QOA_FREE(encoded);
		}
	} else {
		dst_data = pcm_data;
	}

	Ref<AudioStreamWAV> sample;
	sample.instantiate();
	sample->set_data(dst_data);
	sample->set_format(dst_format);
	sample->set_mix_rate(rate);
	sample->set_loop_mode(loop_mode);
	sample->set_loop_begin(loop_begin);
	sample->set_loop_end(loop_end);
	sample->set_stereo(format_channels == 2);

	ResourceSaver::save(sample, p_save_path + ".sample");

	return OK;
}

ResourceImporterWAV::ResourceImporterWAV() {
}
