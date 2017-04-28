/*************************************************************************/
/*  resource_importer_wav.cpp                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "io/marshalls.h"
#include "io/resource_saver.h"
#include "os/file_access.h"
#include "scene/resources/audio_stream_sample.h"

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
	return "smp";
}

String ResourceImporterWAV::get_resource_type() const {

	return "AudioStreamSample";
}

bool ResourceImporterWAV::get_option_visibility(const String &p_option, const Map<StringName, Variant> &p_options) const {

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
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "force/max_rate"), false));
	r_options->push_back(ImportOption(PropertyInfo(Variant::REAL, "force/max_rate_hz", PROPERTY_HINT_EXP_RANGE, "11025,192000,1"), 44100));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "edit/trim"), true));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "edit/normalize"), true));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "edit/loop"), false));
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "compress/mode", PROPERTY_HINT_ENUM, "Disabled,RAM (Ima-ADPCM)"), 0));
}

Error ResourceImporterWAV::import(const String &p_source_file, const String &p_save_path, const Map<StringName, Variant> &p_options, List<String> *r_platform_variants, List<String> *r_gen_files) {

	/* STEP 1, READ WAVE FILE */

	Error err;
	FileAccess *file = FileAccess::open(p_source_file, FileAccess::READ, &err);

	ERR_FAIL_COND_V(err != OK, ERR_CANT_OPEN);

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
	uint32_t filesize = file->get_32();

	/* CHECK WAVE */

	char wave[4];

	file->get_buffer((uint8_t *)&wave, 4); //RIFF

	if (wave[0] != 'W' || wave[1] != 'A' || wave[2] != 'V' || wave[3] != 'E') {

		file->close();
		memdelete(file);
		ERR_EXPLAIN("Not a WAV file (no WAVE RIFF Header)")
		ERR_FAIL_V(ERR_FILE_UNRECOGNIZED);
	}

	int format_bits = 0;
	int format_channels = 0;

	AudioStreamSample::LoopMode loop = AudioStreamSample::LOOP_DISABLED;
	bool format_found = false;
	bool data_found = false;
	int format_freq = 0;
	int loop_begin = 0;
	int loop_end = 0;
	int frames;

	Vector<float> data;

	while (!file->eof_reached()) {

		/* chunk */
		char chunkID[4];
		file->get_buffer((uint8_t *)&chunkID, 4); //RIFF

		/* chunk size */
		uint32_t chunksize = file->get_32();
		uint32_t file_pos = file->get_pos(); //save file pos, so we can skip to next chunk safely

		if (file->eof_reached()) {

			//ERR_PRINT("EOF REACH");
			break;
		}

		if (chunkID[0] == 'f' && chunkID[1] == 'm' && chunkID[2] == 't' && chunkID[3] == ' ' && !format_found) {
			/* IS FORMAT CHUNK */

			uint16_t compression_code = file->get_16();

			if (compression_code != 1) {
				ERR_PRINT("Format not supported for WAVE file (not PCM). Save WAVE files as uncompressed PCM instead.");
				break;
			}

			format_channels = file->get_16();
			if (format_channels != 1 && format_channels != 2) {

				ERR_PRINT("Format not supported for WAVE file (not stereo or mono)");
				break;
			}

			format_freq = file->get_32(); //sampling rate

			file->get_32(); // average bits/second (unused)
			file->get_16(); // block align (unused)
			format_bits = file->get_16(); // bits per sample

			if (format_bits % 8) {

				ERR_PRINT("Strange number of bits in sample (not 8,16,24,32)");
				break;
			}

			/* Don't need anything else, continue */
			format_found = true;
		}

		if (chunkID[0] == 'd' && chunkID[1] == 'a' && chunkID[2] == 't' && chunkID[3] == 'a' && !data_found) {
			/* IS FORMAT CHUNK */
			data_found = true;

			if (!format_found) {
				ERR_PRINT("'data' chunk before 'format' chunk found.");
				break;
			}

			frames = chunksize;

			frames /= format_channels;
			frames /= (format_bits >> 3);

			/*print_line("chunksize: "+itos(chunksize));
			print_line("channels: "+itos(format_channels));
			print_line("bits: "+itos(format_bits));
*/

			int len = frames;
			if (format_channels == 2)
				len *= 2;
			if (format_bits > 8)
				len *= 2;

			data.resize(frames * format_channels);

			for (int i = 0; i < frames; i++) {

				for (int c = 0; c < format_channels; c++) {

					if (format_bits == 8) {
						// 8 bit samples are UNSIGNED

						uint8_t s = file->get_8();
						s -= 128;
						int8_t *sp = (int8_t *)&s;

						data[i * format_channels + c] = float(*sp) / 128.0;

					} else {
						//16+ bits samples are SIGNED
						// if sample is > 16 bits, just read extra bytes

						uint32_t s = 0;
						for (int b = 0; b < (format_bits >> 3); b++) {

							s |= ((uint32_t)file->get_8()) << (b * 8);
						}
						s <<= (32 - format_bits);
						int32_t ss = s;

						data[i * format_channels + c] = (ss >> 16) / 32768.0;
					}
				}
			}

			if (file->eof_reached()) {
				file->close();
				memdelete(file);
				ERR_EXPLAIN("Premature end of file.");
				ERR_FAIL_V(ERR_FILE_CORRUPT);
			}
		}

		if (chunkID[0] == 's' && chunkID[1] == 'm' && chunkID[2] == 'p' && chunkID[3] == 'l') {
			//loop point info!

			for (int i = 0; i < 10; i++)
				file->get_32(); // i wish to know why should i do this... no doc!

			loop = file->get_32() ? AudioStreamSample::LOOP_PING_PONG : AudioStreamSample::LOOP_FORWARD;
			loop_begin = file->get_32();
			loop_end = file->get_32();
		}
		file->seek(file_pos + chunksize);
	}

	file->close();
	memdelete(file);

	// STEP 2, APPLY CONVERSIONS

	bool is16 = format_bits != 8;
	int rate = format_freq;

	print_line("Input Sample: ");
	print_line("\tframes: " + itos(frames));
	print_line("\tformat_channels: " + itos(format_channels));
	print_line("\t16bits: " + itos(is16));
	print_line("\trate: " + itos(rate));
	print_line("\tloop: " + itos(loop));
	print_line("\tloop begin: " + itos(loop_begin));
	print_line("\tloop end: " + itos(loop_end));

	//apply frequency limit

	bool limit_rate = p_options["force/max_rate"];
	int limit_rate_hz = p_options["force/max_rate_hz"];
	if (limit_rate && rate > limit_rate_hz) {
		//resampleeee!!!
		int new_data_frames = frames * limit_rate_hz / rate;
		Vector<float> new_data;
		new_data.resize(new_data_frames * format_channels);
		for (int c = 0; c < format_channels; c++) {

			for (int i = 0; i < new_data_frames; i++) {

				//simple cubic interpolation should be enough.
				float pos = float(i) * frames / new_data_frames;
				float mu = pos - Math::floor(pos);
				int ipos = int(Math::floor(pos));

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

				new_data[i * format_channels + c] = res;
			}
		}

		if (loop) {

			loop_begin = loop_begin * new_data_frames / frames;
			loop_end = loop_end * new_data_frames / frames;
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
			if (amp > max)
				max = amp;
		}

		if (max > 0) {

			float mult = 1.0 / max;
			for (int i = 0; i < data.size(); i++) {

				data[i] *= mult;
			}
		}
	}

	bool trim = p_options["edit/trim"];

	if (trim && !loop) {

		int first = 0;
		int last = (frames * format_channels) - 1;
		bool found = false;
		float limit = Math::db2linear((float)-30);
		for (int i = 0; i < data.size(); i++) {
			float amp = Math::abs(data[i]);

			if (!found && amp > limit) {
				first = i;
				found = true;
			}

			if (found && amp > limit) {
				last = i;
			}
		}

		first /= format_channels;
		last /= format_channels;

		if (first < last) {

			Vector<float> new_data;
			new_data.resize((last - first + 1) * format_channels);
			for (int i = first * format_channels; i <= last * format_channels; i++) {
				new_data[i - first * format_channels] = data[i];
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
			new_data[i] = (data[i * 2 + 0] + data[i * 2 + 1]) / 2.0;
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
				left[i] = data[i * 2 + 0];
				right[i] = data[i * 2 + 1];
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

		//print_line("compressing ima-adpcm, resulting buffersize is "+itos(dst_data.size())+" from "+itos(data.size()));

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

	ResourceSaver::save(p_save_path + ".smp", sample);

	return OK;
}

void ResourceImporterWAV::_compress_ima_adpcm(const Vector<float> &p_data, PoolVector<uint8_t> &dst_data) {

	/*p_sample_data->data = (void*)malloc(len);
	xm_s8 *dataptr=(xm_s8*)p_sample_data->data;*/

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
	if (datalen & 1)
		datalen++;

	dst_data.resize(datalen / 2 + 4);
	PoolVector<uint8_t>::Write w = dst_data.write();

	int i, step_idx = 0, prev = 0;
	uint8_t *out = w.ptr();
	//int16_t xm_prev=0;
	const float *in = p_data.ptr();

	/* initial value is zero */
	*(out++) = 0;
	*(out++) = 0;
	/* Table index initial value */
	*(out++) = 0;
	/* unused */
	*(out++) = 0;

	for (i = 0; i < datalen; i++) {
		int step, diff, vpdiff, mask;
		uint8_t nibble;
		int16_t xm_sample;

		if (i >= datamax)
			xm_sample = 0;
		else {

			xm_sample = CLAMP(in[i] * 32767.0, -32768, 32767);
			/*
			if (xm_sample==32767 || xm_sample==-32768)
				printf("clippy!\n",xm_sample);
			*/
		}

		//xm_sample=xm_sample+xm_prev;
		//xm_prev=xm_sample;

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
		};

		if (nibble & 8)
			prev -= vpdiff;
		else
			prev += vpdiff;

		if (prev > 32767) {
			//printf("%i,xms %i, prev %i,diff %i, vpdiff %i, clip up %i\n",i,xm_sample,prev,diff,vpdiff,prev);
			prev = 32767;
		} else if (prev < -32768) {
			//printf("%i,xms %i, prev %i,diff %i, vpdiff %i, clip down %i\n",i,xm_sample,prev,diff,vpdiff,prev);
			prev = -32768;
		}

		step_idx += _ima_adpcm_index_table[nibble];
		if (step_idx < 0)
			step_idx = 0;
		else if (step_idx > 88)
			step_idx = 88;

		if (i & 1) {
			*out |= nibble << 4;
			out++;
		} else {
			*out = nibble;
		}
		/*dataptr[i]=prev>>8;*/
	}
}

ResourceImporterWAV::ResourceImporterWAV() {
}
