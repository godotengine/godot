/*************************************************************************/
/*  resource_importer_wav.cpp                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

	err = AudioStreamSample::parse_wave_data_header(file);
	if (err != OK) {
		file->close();
		memdelete(file);
		return err;
	}

	AudioStreamSample::WaveData wave_data;

	err = AudioStreamSample::parse_wave_data_body(file, &wave_data);

	file->close();
	memdelete(file);

	if (err != OK) {
		return err;
	}

	// STEP 2, APPLY CONVERSIONS

	bool is16 = wave_data.format_bits != 8;
	int rate = wave_data.format_freq;

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
	if (limit_rate && rate > limit_rate_hz && rate > 0 && wave_data.frames > 0) {
		// resample!
		int new_data_frames = (int)(wave_data.frames * (float)limit_rate_hz / (float)rate);

		Vector<float> new_data;
		new_data.resize(new_data_frames * wave_data.format_channels);
		for (int c = 0; c < wave_data.format_channels; c++) {

			float frac = .0f;
			int ipos = 0;

			for (int i = 0; i < new_data_frames; i++) {

				//simple cubic interpolation should be enough.

				float mu = frac;

				float y0 = wave_data.data[MAX(0, ipos - 1) * wave_data.format_channels + c];
				float y1 = wave_data.data[ipos * wave_data.format_channels + c];
				float y2 = wave_data.data[MIN(wave_data.frames - 1, ipos + 1) * wave_data.format_channels + c];
				float y3 = wave_data.data[MIN(wave_data.frames - 1, ipos + 2) * wave_data.format_channels + c];

				float mu2 = mu * mu;
				float a0 = y3 - y2 - y0 + y1;
				float a1 = y0 - y1 - a0;
				float a2 = y2 - y0;
				float a3 = y1;

				float res = (a0 * mu * mu2 + a1 * mu2 + a2 * mu + a3);

				new_data.write[i * wave_data.format_channels + c] = res;

				// update position and always keep fractional part within ]0...1]
				// in order to avoid 32bit floating point precision errors

				frac += (float)rate / (float)limit_rate_hz;
				int tpos = (int)Math::floor(frac);
				ipos += tpos;
				frac -= tpos;
			}
		}

		if (wave_data.loop) {
			wave_data.loop_begin = (int)(wave_data.loop_begin * (float)new_data_frames / (float)wave_data.frames);
			wave_data.loop_end = (int)(wave_data.loop_end * (float)new_data_frames / (float)wave_data.frames);
		}

		wave_data.data = new_data;
		rate = limit_rate_hz;
		wave_data.frames = new_data_frames;
	}

	bool normalize = p_options["edit/normalize"];

	if (normalize) {

		float max = 0;
		for (int i = 0; i < wave_data.data.size(); i++) {

			float amp = Math::abs(wave_data.data[i]);
			if (amp > max)
				max = amp;
		}

		if (max > 0) {

			float mult = 1.0 / max;
			for (int i = 0; i < wave_data.data.size(); i++) {

				wave_data.data.write[i] *= mult;
			}
		}
	}

	bool trim = p_options["edit/trim"];

	if (trim && !wave_data.loop && wave_data.format_channels > 0) {

		int first = 0;
		int last = (wave_data.frames / wave_data.format_channels) - 1;
		bool found = false;
		float limit = Math::db2linear(TRIM_DB_LIMIT);

		for (int i = 0; i < wave_data.data.size() / wave_data.format_channels; i++) {
			float ampChannelSum = 0;
			for (int j = 0; j < wave_data.format_channels; j++) {
				ampChannelSum += Math::abs(wave_data.data[(i * wave_data.format_channels) + j]);
			}

			float amp = Math::abs(ampChannelSum / (float)wave_data.format_channels);

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
			new_data.resize((last - first) * wave_data.format_channels);
			for (int i = first; i < last; i++) {

				float fadeOutMult = 1;

				if (last - i < TRIM_FADE_OUT_FRAMES) {
					fadeOutMult = ((float)(last - i - 1) / (float)TRIM_FADE_OUT_FRAMES);
				}

				for (int j = 0; j < wave_data.format_channels; j++) {
					new_data.write[((i - first) * wave_data.format_channels) + j] = wave_data.data[(i * wave_data.format_channels) + j] * fadeOutMult;
				}
			}

			wave_data.data = new_data;
			wave_data.frames = wave_data.data.size() / wave_data.format_channels;
		}
	}

	bool make_loop = p_options["edit/loop"];

	if (make_loop && !wave_data.loop) {

		wave_data.loop = AudioStreamSample::LOOP_FORWARD;
		wave_data.loop_begin = 0;
		wave_data.loop_end = wave_data.frames;
	}

	int compression = p_options["compress/mode"];
	bool force_mono = p_options["force/mono"];

	if (force_mono && wave_data.format_channels == 2) {

		Vector<float> new_data;
		new_data.resize(wave_data.data.size() / 2);
		for (int i = 0; i < wave_data.frames; i++) {
			new_data.write[i] = (wave_data.data[i * 2 + 0] + wave_data.data[i * 2 + 1]) / 2.0;
		}

		wave_data.data = new_data;
		wave_data.format_channels = 1;
	}

	bool force_8_bit = p_options["force/8_bit"];
	if (force_8_bit) {

		is16 = false;
	}

	PoolVector<uint8_t> dst_data;
	AudioStreamSample::Format dst_format;

	if (compression == 1) {

		dst_format = AudioStreamSample::FORMAT_IMA_ADPCM;
		if (wave_data.format_channels == 1) {
			_compress_ima_adpcm(wave_data.data, dst_data);
		} else {

			//byte interleave
			Vector<float> left;
			Vector<float> right;

			int tframes = wave_data.data.size() / 2;
			left.resize(tframes);
			right.resize(tframes);

			for (int i = 0; i < tframes; i++) {
				left.write[i] = wave_data.data[i * 2 + 0];
				right.write[i] = wave_data.data[i * 2 + 1];
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
		dst_data.resize(wave_data.data.size() * (is16 ? 2 : 1));
		{
			PoolVector<uint8_t>::Write w = dst_data.write();

			int ds = wave_data.data.size();
			for (int i = 0; i < ds; i++) {

				if (is16) {
					int16_t v = CLAMP(wave_data.data[i] * 32768, -32768, 32767);
					encode_uint16(v, &w[i * 2]);
				} else {
					int8_t v = CLAMP(wave_data.data[i] * 128, -128, 127);
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
	sample->set_loop_mode(wave_data.loop);
	sample->set_loop_begin(wave_data.loop_begin);
	sample->set_loop_end(wave_data.loop_end);
	sample->set_stereo(wave_data.format_channels == 2);

	ResourceSaver::save(p_save_path + ".sample", sample);

	return OK;
}

ResourceImporterWAV::ResourceImporterWAV() {
}
