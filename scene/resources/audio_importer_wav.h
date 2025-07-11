#ifndef AUDIO_IMPORTER_WAV_H
#define AUDIO_IMPORTER_WAV_H

#include "core/io/resource_importer.h"
#include "scene/resources/audio_stream_wav.h"

class AudioStreamWAV;

class AudioImporterWav {
public:
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

	static Error import_asset(Ref<AudioStreamWAV> &sample, const String &p_source_file);
	static Error import_asset(Ref<AudioStreamWAV> &sample, const String &p_source_file, const HashMap<StringName, Variant> &p_options);
};

#endif // AUDIO_IMPORTER_WAV_H
