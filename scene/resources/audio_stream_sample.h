/*************************************************************************/
/*  audio_stream_sample.h                                                */
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

#ifndef AUDIO_STREAM_SAMPLE_H
#define AUDIO_STREAM_SAMPLE_H

#include "servers/audio/audio_stream.h"

class AudioStreamSample;

class AudioStreamPlaybackSample : public AudioStreamPlayback {

	GDCLASS(AudioStreamPlaybackSample, AudioStreamPlayback);
	enum {
		MIX_FRAC_BITS = 13,
		MIX_FRAC_LEN = (1 << MIX_FRAC_BITS),
		MIX_FRAC_MASK = MIX_FRAC_LEN - 1,
	};

	struct IMA_ADPCM_State {

		int16_t step_index;
		int32_t predictor;
		/* values at loop point */
		int16_t loop_step_index;
		int32_t loop_predictor;
		int32_t last_nibble;
		int32_t loop_pos;
		int32_t window_ofs;
	} ima_adpcm[2];

	int64_t offset;
	int sign;
	bool active;
	friend class AudioStreamSample;
	Ref<AudioStreamSample> base;

	template <class Depth, bool is_stereo, bool is_ima_adpcm>
	void do_resample(const Depth *p_src, AudioFrame *p_dst, int64_t &offset, int32_t &increment, uint32_t amount, IMA_ADPCM_State *ima_adpcm);

public:
	virtual void start(float p_from_pos = 0.0);
	virtual void stop();
	virtual bool is_playing() const;

	virtual int get_loop_count() const; //times it looped

	virtual float get_playback_position() const;
	virtual void seek(float p_time);

	virtual void mix(AudioFrame *p_buffer, float p_rate_scale, int p_frames);

	AudioStreamPlaybackSample();
};

class AudioStreamSample : public AudioStream {
	GDCLASS(AudioStreamSample, AudioStream);
	RES_BASE_EXTENSION("sample")

public:
	enum Format {
		FORMAT_8_BITS,
		FORMAT_16_BITS,
		FORMAT_IMA_ADPCM
	};

	enum LoopMode {
		LOOP_DISABLED,
		LOOP_FORWARD,
		LOOP_PING_PONG,
		LOOP_BACKWARD
	};

	struct WaveData {
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
	};

	static Error parse_wave_data_header(FileAccess *p_file) {
		/* CHECK RIFF */
		char riff[5];
		riff[4] = 0;
		p_file->get_buffer((uint8_t *)&riff, 4); //RIFF

		if (riff[0] != 'R' || riff[1] != 'I' || riff[2] != 'F' || riff[3] != 'F') {

			ERR_FAIL_V(ERR_FILE_UNRECOGNIZED);
		}

		/* GET FILESIZE */
		p_file->get_32(); // filesize

		/* CHECK WAVE */

		char wave[4];

		p_file->get_buffer((uint8_t *)&wave, 4); //RIFF

		if (wave[0] != 'W' || wave[1] != 'A' || wave[2] != 'V' || wave[3] != 'E') {

			ERR_FAIL_V_MSG(ERR_FILE_UNRECOGNIZED, "Not a WAV file (no WAVE RIFF header).");
		}

		return OK;
	}

	static Error parse_wave_data_body(FileAccess *p_file, WaveData *p_wave_data) {

		while (!p_file->eof_reached()) {

			/* chunk */
			char chunkID[4];
			p_file->get_buffer((uint8_t *)&chunkID, 4); //RIFF

			/* chunk size */
			uint32_t chunksize = p_file->get_32();
			uint32_t file_pos = p_file->get_position(); //save file pos, so we can skip to next chunk safely

			if (p_file->eof_reached()) {

				//ERR_PRINT("EOF REACH");
				break;
			}

			if (chunkID[0] == 'f' && chunkID[1] == 'm' && chunkID[2] == 't' && chunkID[3] == ' ' && !p_wave_data->format_found) {
				/* IS FORMAT CHUNK */

				//Issue: #7755 : Not a bug - usage of other formats (format codes) are unsupported in current importer version.
				//Consider revision for engine version 3.0
				p_wave_data->compression_code = p_file->get_16();
				if (p_wave_data->compression_code != 1 && p_wave_data->compression_code != 3) {
					ERR_FAIL_V_MSG(ERR_INVALID_DATA, "Format not supported for WAVE file (not PCM). Save WAVE files as uncompressed PCM instead.");
				}

				p_wave_data->format_channels = p_file->get_16();
				if (p_wave_data->format_channels != 1 && p_wave_data->format_channels != 2) {
					ERR_FAIL_V_MSG(ERR_INVALID_DATA, "Format not supported for WAVE file (not stereo or mono).");
				}

				p_wave_data->format_freq = p_file->get_32(); //sampling rate

				p_file->get_32(); // average bits/second (unused)
				p_file->get_16(); // block align (unused)
				p_wave_data->format_bits = p_file->get_16(); // bits per sample

				if (p_wave_data->format_bits % 8 || p_wave_data->format_bits == 0) {
					ERR_FAIL_V_MSG(ERR_INVALID_DATA, "Invalid amount of bits in the sample (should be one of 8, 16, 24 or 32).");
				}

				/* Don't need anything else, continue */
				p_wave_data->format_found = true;
			}

			if (chunkID[0] == 'd' && chunkID[1] == 'a' && chunkID[2] == 't' && chunkID[3] == 'a' && !p_wave_data->data_found) {
				/* IS DATA CHUNK */
				p_wave_data->data_found = true;

				if (!p_wave_data->format_found) {
					ERR_PRINT("'data' chunk before 'format' chunk found.");
					break;
				}

				p_wave_data->frames = chunksize;

				if (p_wave_data->format_channels == 0) {
					p_file->close();
					ERR_FAIL_COND_V(p_wave_data->format_channels == 0, ERR_INVALID_DATA);
				}
				p_wave_data->frames /= p_wave_data->format_channels;
				p_wave_data->frames /= (p_wave_data->format_bits >> 3);

				/*print_line("chunksize: "+itos(chunksize));
				print_line("channels: "+itos(format_channels));
				print_line("bits: "+itos(format_bits));
				*/

				p_wave_data->data.resize(p_wave_data->frames * p_wave_data->format_channels);

				if (p_wave_data->format_bits == 8) {
					for (int i = 0; i < p_wave_data->frames * p_wave_data->format_channels; i++) {
						// 8 bit samples are UNSIGNED

						p_wave_data->data.write[i] = int8_t(p_file->get_8() - 128) / 128.f;
					}
				} else if (p_wave_data->format_bits == 32 && p_wave_data->compression_code == 3) {
					for (int i = 0; i < p_wave_data->frames * p_wave_data->format_channels; i++) {
						//32 bit IEEE Float

						p_wave_data->data.write[i] = p_file->get_float();
					}
				} else if (p_wave_data->format_bits == 16) {
					for (int i = 0; i < p_wave_data->frames * p_wave_data->format_channels; i++) {
						//16 bit SIGNED

						p_wave_data->data.write[i] = int16_t(p_file->get_16()) / 32768.f;
					}
				} else {
					for (int i = 0; i < p_wave_data->frames * p_wave_data->format_channels; i++) {
						//16+ bits samples are SIGNED
						// if sample is > 16 bits, just read extra bytes

						uint32_t s = 0;
						for (int b = 0; b < (p_wave_data->format_bits >> 3); b++) {

							s |= ((uint32_t)p_file->get_8()) << (b * 8);
						}
						s <<= (32 - p_wave_data->format_bits);

						p_wave_data->data.write[i] = (int32_t(s) >> 16) / 32768.f;
					}
				}

				if (p_file->eof_reached()) {
					p_file->close();
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

				for (int i = 0; i < 10; i++)
					p_file->get_32(); // i wish to know why should i do this... no doc!

				// only read 0x00 (loop forward), 0x01 (loop ping-pong) and 0x02 (loop backward)
				// Skip anything else because it's not supported, reserved for future uses or sampler specific
				// from https://sites.google.com/site/musicgapi/technical-documents/wav-file-format#smpl (loop type values table)
				int loop_type = p_file->get_32();
				if (loop_type == 0x00 || loop_type == 0x01 || loop_type == 0x02) {
					if (loop_type == 0x00) {
						p_wave_data->loop = AudioStreamSample::LOOP_FORWARD;
					} else if (loop_type == 0x01) {
						p_wave_data->loop = AudioStreamSample::LOOP_PING_PONG;
					} else if (loop_type == 0x02) {
						p_wave_data->loop = AudioStreamSample::LOOP_BACKWARD;
					}
					p_wave_data->loop_begin = p_file->get_32();
					p_wave_data->loop_end = p_file->get_32();
				} else {
					p_wave_data->loop_begin = 0;
					p_wave_data->loop_end = p_wave_data->frames;
				}
			}
			p_file->seek(file_pos + chunksize);
		}

		return OK;
	}

private:
	friend class AudioStreamPlaybackSample;

	enum {
		DATA_PAD = 16 //padding for interpolation
	};

	Format format;
	LoopMode loop_mode;
	bool stereo;
	int loop_begin;
	int loop_end;
	int mix_rate;
	void *data;
	uint32_t data_bytes;

protected:
	static void _bind_methods();

public:
	void set_format(Format p_format);
	Format get_format() const;

	void set_loop_mode(LoopMode p_loop_mode);
	LoopMode get_loop_mode() const;

	void set_loop_begin(int p_frame);
	int get_loop_begin() const;

	void set_loop_end(int p_frame);
	int get_loop_end() const;

	void set_mix_rate(int p_hz);
	int get_mix_rate() const;

	void set_stereo(bool p_enable);
	bool is_stereo() const;

	virtual float get_length() const; //if supported, otherwise return 0

	void set_data(const PoolVector<uint8_t> &p_data);
	PoolVector<uint8_t> get_data() const;

	Error load(const String &p_path);
	Error save_to_wav(const String &p_path);

	virtual Ref<AudioStreamPlayback> instance_playback();
	virtual String get_stream_name() const;

	AudioStreamSample();
	~AudioStreamSample();
};

VARIANT_ENUM_CAST(AudioStreamSample::Format)
VARIANT_ENUM_CAST(AudioStreamSample::LoopMode)

#endif // AUDIO_STREAM_SAMPLE_H
