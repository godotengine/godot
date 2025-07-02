/**************************************************************************/
/*  register_types.cpp                                                    */
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

#include "register_types.h"

#include "core/io/marshalls.h"
#include "scene/resources/audio_stream_wav.h"

#define DR_WAV_IMPLEMENTATION
#define DR_WAV_NO_STDIO
#define DR_WAV_LIBSNDFILE_COMPAT
#include "thirdparty/dr_libs/dr_bridge.h"
#include "thirdparty/dr_libs/dr_wav.h"

const float TRIM_DB_LIMIT = -50;
const int TRIM_FADE_OUT_FRAMES = 500;

static Ref<AudioStreamWAV> load(drwav &p_wav, const Dictionary &p_options) {
	// STEP 1, READ_DATA

	uint16_t format_bits = p_wav.bitsPerSample;
	uint32_t format_freq = p_wav.sampleRate;
	int64_t frames = p_wav.totalPCMFrameCount;
	uint16_t format_channels = p_wav.channels;

	if (format_channels != 1 && format_channels != 2) {
		ERR_FAIL_V_MSG(Ref<AudioStreamWAV>(), "Unsupported number of channels: " + itos(format_channels) + ". Only mono (1) and stereo (2) are supported.");
	}

	int import_loop_mode = p_options["edit/loop_mode"];
	int64_t loop_begin = 0;
	int64_t loop_end = 0;
	AudioStreamWAV::LoopMode loop_mode = AudioStreamWAV::LoopMode::LOOP_DISABLED;

	Dictionary tag_dictionary;
	for (uint32_t meta = 0; meta < p_wav.metadataCount; ++meta) {
		drwav_metadata md = p_wav.pMetadata[meta];
		if (import_loop_mode == 0 && md.type == drwav_metadata_type_smpl && md.data.smpl.sampleLoopCount > 0) { // Loop info
			drwav_smpl_loop loop = md.data.smpl.pLoops[0];
			loop_mode = (AudioStreamWAV::LoopMode)(loop.type + 1);
			loop_begin = loop.firstSampleOffset;
			loop_end = loop.lastSampleOffset;
		} else if (md.type & drwav_metadata_type_list_all_info_strings) { // List info metadata
			String tag;
			switch (md.type) {
				case drwav_metadata_type_list_info_software:
					tag = "software";
					break;
				case drwav_metadata_type_list_info_copyright:
					tag = "copyright";
					break;
				case drwav_metadata_type_list_info_title:
					tag = "title";
					break;
				case drwav_metadata_type_list_info_artist:
					tag = "artist";
					break;
				case drwav_metadata_type_list_info_comment:
					tag = "comments";
					break;
				case drwav_metadata_type_list_info_date:
					tag = "date";
					break;
				case drwav_metadata_type_list_info_genre:
					tag = "genre";
					break;
				case drwav_metadata_type_list_info_album:
					tag = "album";
					break;
				case drwav_metadata_type_list_info_tracknumber:
					tag = "tracknumber";
					break;
				case drwav_metadata_type_list_info_location:
					tag = "location";
					break;
				case drwav_metadata_type_list_info_organization:
					tag = "organization";
					break;
				case drwav_metadata_type_list_info_keywords:
					tag = "keywords";
					break;
				case drwav_metadata_type_list_info_medium:
					tag = "medium";
					break;
				case drwav_metadata_type_list_info_description:
					tag = "description";
					break;
				default: // Unsupported or unknown tag.
					break;
			}

			// The data is always an ASCII string. ASCII is a subset of UTF-8.
			String tag_value;
			tag_value.append_utf8(md.data.infoText.pString, md.data.infoText.stringLength);

			tag_dictionary[tag] = tag_value;
		}
	}

	Vector<float> data;
	ERR_FAIL_COND_V(data.resize(frames * format_channels), Ref<AudioStreamWAV>());
	drwav_read_pcm_frames_f32(&p_wav, frames, data.ptrw());
	drwav_uninit(&p_wav);

	// STEP 2, APPLY CONVERSIONS

	bool is16 = format_bits != 8;
	uint32_t rate = format_freq;

	//apply frequency limit

	bool limit_rate = p_options["force/max_rate"];
	uint32_t limit_rate_hz = p_options["force/max_rate_hz"];
	if (limit_rate && rate > limit_rate_hz && rate > 0 && frames > 0) {
		// resample!
		int64_t new_data_frames = (int64_t)(frames * (float)limit_rate_hz / (float)rate);

		Vector<float> new_data;
		ERR_FAIL_COND_V(new_data.resize(new_data_frames * format_channels), Ref<AudioStreamWAV>());
		for (uint16_t c = 0; c < format_channels; c++) {
			float frac = 0.0;
			int64_t ipos = 0;

			for (int64_t i = 0; i < new_data_frames; i++) {
				// Cubic interpolation should be enough.

				float y0 = data[MAX(0, ipos - 1) * format_channels + c];
				float y1 = data[ipos * format_channels + c];
				float y2 = data[MIN(frames - 1, ipos + 1) * format_channels + c];
				float y3 = data[MIN(frames - 1, ipos + 2) * format_channels + c];

				new_data.write[i * format_channels + c] = Math::cubic_interpolate(y1, y2, y0, y3, frac);

				// update position and always keep fractional part within ]0...1]
				// in order to avoid 32bit floating point precision errors

				frac += (float)rate / (float)limit_rate_hz;
				int64_t tpos = (int64_t)Math::floor(frac);
				ipos += tpos;
				frac -= tpos;
			}
		}

		if (loop_mode) {
			loop_begin = (int64_t)(loop_begin * (float)new_data_frames / (float)frames);
			loop_end = (int64_t)(loop_end * (float)new_data_frames / (float)frames);
		}

		data = new_data;
		rate = limit_rate_hz;
		frames = new_data_frames;
	}

	bool normalize = p_options["edit/normalize"];

	if (normalize) {
		float max = 0.0;
		for (int64_t i = 0; i < data.size(); i++) {
			float amp = Math::abs(data[i]);
			if (amp > max) {
				max = amp;
			}
		}

		if (max > 0) {
			float mult = 1.0 / max;
			for (int64_t i = 0; i < data.size(); i++) {
				data.write[i] *= mult;
			}
		}
	}

	bool trim = p_options["edit/trim"];

	if (trim && (loop_mode == AudioStreamWAV::LOOP_DISABLED) && format_channels > 0) {
		int64_t first = 0;
		int64_t last = (frames / format_channels) - 1;
		bool found = false;
		float limit = Math::db_to_linear(TRIM_DB_LIMIT);

		for (int64_t i = 0; i < data.size() / format_channels; i++) {
			float amp_channel_sum = 0.0;
			for (uint16_t j = 0; j < format_channels; j++) {
				amp_channel_sum += Math::abs(data[(i * format_channels) + j]);
			}

			float amp = Math::abs(amp_channel_sum / (float)format_channels);

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
			ERR_FAIL_COND_V(new_data.resize((last - first) * format_channels), Ref<AudioStreamWAV>());
			for (int64_t i = first; i < last; i++) {
				float fade_out_mult = 1.0;

				if (last - i < TRIM_FADE_OUT_FRAMES) {
					fade_out_mult = ((float)(last - i - 1) / (float)TRIM_FADE_OUT_FRAMES);
				}

				for (uint16_t j = 0; j < format_channels; j++) {
					new_data.write[((i - first) * format_channels) + j] = data[(i * format_channels) + j] * fade_out_mult;
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
		ERR_FAIL_COND_V(new_data.resize(data.size() / 2), Ref<AudioStreamWAV>());
		for (int64_t i = 0; i < frames; i++) {
			new_data.write[i] = (data[i * 2 + 0] + data[i * 2 + 1]) / 2.0;
		}

		data = new_data;
		format_channels = 1;
	}

	bool force_8_bit = p_options["force/8_bit"];
	if (force_8_bit) {
		is16 = false;
	}

	Vector<uint8_t> dst_data;
	AudioStreamWAV::Format dst_format;

	if (compression == 1) {
		dst_format = AudioStreamWAV::FORMAT_IMA_ADPCM;
		if (format_channels == 1) {
			AudioStreamWAV::_compress_ima_adpcm(data, dst_data);
		} else {
			//byte interleave
			Vector<float> left;
			Vector<float> right;

			int64_t tframes = data.size() / 2;
			ERR_FAIL_COND_V(left.resize(tframes), Ref<AudioStreamWAV>());
			ERR_FAIL_COND_V(right.resize(tframes), Ref<AudioStreamWAV>());

			for (int i = 0; i < tframes; i++) {
				left.write[i] = data[i * 2 + 0];
				right.write[i] = data[i * 2 + 1];
			}

			Vector<uint8_t> bleft;
			Vector<uint8_t> bright;

			AudioStreamWAV::_compress_ima_adpcm(left, bleft);
			AudioStreamWAV::_compress_ima_adpcm(right, bright);

			int64_t dl = bleft.size();
			ERR_FAIL_COND_V(dst_data.resize(dl * 2), Ref<AudioStreamWAV>());

			uint8_t *w = dst_data.ptrw();
			const uint8_t *rl = bleft.ptr();
			const uint8_t *rr = bright.ptr();

			for (int64_t i = 0; i < dl; i++) {
				w[i * 2 + 0] = rl[i];
				w[i * 2 + 1] = rr[i];
			}
		}

	} else if (compression == 2) {
		dst_format = AudioStreamWAV::FORMAT_QOA;

		qoa_desc desc = {};
		desc.samplerate = rate;
		desc.samples = frames;
		desc.channels = format_channels;

		AudioStreamWAV::_compress_qoa(data, dst_data, &desc);
	} else {
		dst_format = is16 ? AudioStreamWAV::FORMAT_16_BITS : AudioStreamWAV::FORMAT_8_BITS;
		ERR_FAIL_COND_V(dst_data.resize(data.size() * (is16 ? 2 : 1)), Ref<AudioStreamWAV>());
		{
			uint8_t *w = dst_data.ptrw();

			int64_t ds = data.size();
			for (int64_t i = 0; i < ds; i++) {
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

	Ref<AudioStreamWAV> sample;
	sample.instantiate();
	sample->set_data(dst_data);
	sample->set_format(dst_format);
	sample->set_mix_rate(rate);
	sample->set_loop_mode(loop_mode);
	sample->set_loop_begin(loop_begin);
	sample->set_loop_end(loop_end);
	sample->set_stereo(format_channels == 2);
	sample->set_tags(tag_dictionary);

	return sample;
}

drwav_allocation_callbacks *alloc_calls = (drwav_allocation_callbacks *)&dr_alloc_calls;

static Ref<AudioStreamWAV> load_from_buffer(const Vector<uint8_t> &p_stream_data, const Dictionary &p_options) {
	drwav wav;

	if (!drwav_init_memory_with_metadata(&wav, p_stream_data.ptr(), p_stream_data.size(), DRWAV_WITH_METADATA, alloc_calls)) {
		ERR_FAIL_V_MSG(Ref<AudioStreamWAV>(), "Audio data is invalid, corrupted, or an unsupported format.");
	}

	return load(wav, p_options);
}

static Ref<AudioStreamWAV> load_from_file(const String &p_path, const Dictionary &p_options) {
	Error err;

	Ref<FileAccess> file = FileAccess::open(p_path, FileAccess::READ, &err);
	ERR_FAIL_COND_V_MSG(err != OK, Ref<AudioStreamWAV>(), vformat("Cannot open file '%s'.", p_path));

	drwav_read_proc read_fa = (drwav_read_proc)dr_read_fileaccess;
	drwav_seek_proc seek_fa = (drwav_seek_proc)dr_seek_fileaccess<drwav_seek_origin>;
	drwav_tell_proc tell_fa = (drwav_tell_proc)dr_tell_fileaccess;
	drwav wav;

	if (!drwav_init_with_metadata(&wav, read_fa, seek_fa, tell_fa, &file, DRWAV_WITH_METADATA, alloc_calls)) {
		ERR_FAIL_V_MSG(Ref<AudioStreamWAV>(), "Cannot read data from file '" + p_path + "'. Data is invalid, corrupted, or an unsupported format.");
	}

	return load(wav, p_options);
}

void initialize_wav_loader_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}

	AudioStreamWAV::load_from_buffer_func = load_from_buffer;
	AudioStreamWAV::load_from_file_func = load_from_file;
}

void uninitialize_wav_loader_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}

	AudioStreamWAV::load_from_buffer_func = nullptr;
	AudioStreamWAV::load_from_file_func = nullptr;
}
