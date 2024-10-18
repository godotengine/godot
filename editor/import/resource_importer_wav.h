/**************************************************************************/
/*  resource_importer_wav.h                                               */
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

#ifndef RESOURCE_IMPORTER_WAV_H
#define RESOURCE_IMPORTER_WAV_H

#include "core/io/resource_importer.h"
#include "thirdparty/misc/qoa.h"

class ResourceImporterWAV : public ResourceImporter {
	GDCLASS(ResourceImporterWAV, ResourceImporter);

public:
	virtual String get_importer_name() const override;
	virtual String get_visible_name() const override;
	virtual void get_recognized_extensions(List<String> *p_extensions) const override;
	virtual String get_save_extension() const override;
	virtual String get_resource_type() const override;

	virtual int get_preset_count() const override;
	virtual String get_preset_name(int p_idx) const override;

	virtual void get_import_options(const String &p_path, List<ImportOption> *r_options, int p_preset = 0) const override;
	virtual bool get_option_visibility(const String &p_path, const String &p_option, const HashMap<StringName, Variant> &p_options) const override;

	static void _compress_ima_adpcm(const Vector<float> &p_data, Vector<uint8_t> &dst_data) {
		static const int16_t _ima_adpcm_step_table[89] = {
			7, 8, 9, 10, 11, 12, 13, 14, 16, 17,
			19, 21, 23, 25, 28, 31, 34, 37, 41, 45,
			50, 55, 60, 66, 73, 80, 88, 97, 107, 118,
			130, 143, 157, 173, 190, 209, 230, 253, 279, 307,
			337, 371, 408, 449, 494, 544, 598, 658, 724, 796,
			876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066,
			2272, 2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358,
			5894, 6484, 7132, 7845, 8630, 9493, 10442, 11487, 12635, 13899,
			15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767
		};

		static const int8_t _ima_adpcm_index_table[16] = {
			-1, -1, -1, -1, 2, 4, 6, 8,
			-1, -1, -1, -1, 2, 4, 6, 8
		};

		int datalen = p_data.size();
		int datamax = datalen;
		if (datalen & 1) {
			datalen++;
		}

		dst_data.resize(datalen / 2 + 4);
		uint8_t *w = dst_data.ptrw();

		int i, step_idx = 0, prev = 0;
		uint8_t *out = w;
		const float *in = p_data.ptr();

		// Initial value is zero.
		*(out++) = 0;
		*(out++) = 0;
		// Table index initial value.
		*(out++) = 0;
		// Unused.
		*(out++) = 0;

		for (i = 0; i < datalen; i++) {
			int step, diff, vpdiff, mask;
			uint8_t nibble;
			int16_t xm_sample;

			if (i >= datamax) {
				xm_sample = 0;
			} else {
				xm_sample = CLAMP(in[i] * 32767.0, -32768, 32767);
			}

			diff = (int)xm_sample - prev;

			nibble = 0;
			step = _ima_adpcm_step_table[step_idx];
			vpdiff = step >> 3;
			if (diff < 0) {
				nibble = 8;
				diff = -diff;
			}
			mask = 4;
			while (mask) {
				if (diff >= step) {
					nibble |= mask;
					diff -= step;
					vpdiff += step;
				}

				step >>= 1;
				mask >>= 1;
			}

			if (nibble & 8) {
				prev -= vpdiff;
			} else {
				prev += vpdiff;
			}

			prev = CLAMP(prev, -32768, 32767);

			step_idx += _ima_adpcm_index_table[nibble];
			step_idx = CLAMP(step_idx, 0, 88);

			if (i & 1) {
				*out |= nibble << 4;
				out++;
			} else {
				*out = nibble;
			}
		}
	}

	static void _compress_qoa(const Vector<float> &p_data, Vector<uint8_t> &dst_data, qoa_desc *p_desc) {
		uint32_t frames_len = (p_desc->samples + QOA_FRAME_LEN - 1) / QOA_FRAME_LEN * (QOA_LMS_LEN * 4 * p_desc->channels + 8);
		uint32_t slices_len = (p_desc->samples + QOA_SLICE_LEN - 1) / QOA_SLICE_LEN * 8 * p_desc->channels;
		dst_data.resize(8 + frames_len + slices_len);

		for (uint32_t c = 0; c < p_desc->channels; c++) {
			memset(p_desc->lms[c].history, 0, sizeof(p_desc->lms[c].history));
			memset(p_desc->lms[c].weights, 0, sizeof(p_desc->lms[c].weights));
			p_desc->lms[c].weights[2] = -(1 << 13);
			p_desc->lms[c].weights[3] = (1 << 14);
		}

		LocalVector<int16_t> data16;
		data16.resize(QOA_FRAME_LEN * p_desc->channels);

		uint8_t *dst_ptr = dst_data.ptrw();
		dst_ptr += qoa_encode_header(p_desc, dst_data.ptrw());

		uint32_t frame_len = QOA_FRAME_LEN;
		for (uint32_t s = 0; s < p_desc->samples; s += frame_len) {
			frame_len = MIN(frame_len, p_desc->samples - s);
			for (uint32_t i = 0; i < frame_len * p_desc->channels; i++) {
				data16[i] = CLAMP(p_data[s * p_desc->channels + i] * 32767.0, -32768, 32767);
			}
			dst_ptr += qoa_encode_frame(data16.ptr(), p_desc, frame_len, dst_ptr);
		}
	}

	virtual Error import(const String &p_source_file, const String &p_save_path, const HashMap<StringName, Variant> &p_options, List<String> *r_platform_variants, List<String> *r_gen_files = nullptr, Variant *r_metadata = nullptr) override;

	ResourceImporterWAV();
};

#endif // RESOURCE_IMPORTER_WAV_H
