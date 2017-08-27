/*************************************************************************/
/*  audio_stream_speex.cpp                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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
#include "audio_stream_speex.h"

#include "os/os.h"
#include "os_support.h"
#define READ_CHUNK 1024

static _FORCE_INLINE_ uint16_t le_short(uint16_t s) {
	uint16_t ret = s;
#if 0 //def BIG_ENDIAN_ENABLED
   ret =  s>>8;
   ret += s<<8;
#endif
	return ret;
}

int AudioStreamPlaybackSpeex::mix(int16_t *p_buffer, int p_frames) {

	//printf("update, loops %i, read ofs %i\n", (int)loops, read_ofs);
	//printf("playing %i, paused %i\n", (int)playing, (int)paused);

	if (!active || !playing || !data.size())
		return 0;

	/*
	if (read_ofs >= data.size()) {
		if (loops) {
			reload();
			++loop_count;
		} else {
			return;
		};
	};
	*/

	int todo = p_frames;
	if (todo < page_size) {
		return 0;
	};

	int eos = 0;

	while (todo > page_size) {

		int ret = 0;
		while ((todo > page_size && packets_available && !eos) || (ret = ogg_sync_pageout(&oy, &og)) == 1) {

			if (!packets_available) {
				/*Add page to the bitstream*/
				ogg_stream_pagein(&os, &og);
				page_granule = ogg_page_granulepos(&og);
				page_nb_packets = ogg_page_packets(&og);
				packet_no = 0;
				if (page_granule > 0 && frame_size) {
					skip_samples = page_nb_packets * frame_size * nframes - (page_granule - last_granule);
					if (ogg_page_eos(&og))
						skip_samples = -skip_samples;
					/*else if (!ogg_page_bos(&og))
					skip_samples = 0;*/
				} else {
					skip_samples = 0;
				}

				last_granule = page_granule;
				packets_available = true;
			}
			/*Extract all available packets*/
			while (todo > page_size && !eos) {

				if (ogg_stream_packetout(&os, &op) != 1) {
					packets_available = false;
					break;
				}

				packet_no++;

				/*End of stream condition*/
				if (op.e_o_s)
					eos = 1;

				/*Copy Ogg packet to Speex bitstream*/
				speex_bits_read_from(&bits, (char *)op.packet, op.bytes);

				for (int j = 0; j != nframes; j++) {

					int16_t *out = p_buffer;

					int ret;
					/*Decode frame*/
					ret = speex_decode_int(st, &bits, out);

					/*for (i=0;i<frame_size*channels;i++)
					  printf ("%d\n", (int)output[i]);*/

					if (ret == -1) {
						printf("decode returned -1\n");
						break;
					};
					if (ret == -2) {
						OS::get_singleton()->printerr("Decoding error: corrupted stream?\n");
						break;
					}
					if (speex_bits_remaining(&bits) < 0) {
						OS::get_singleton()->printerr("Decoding overflow: corrupted stream?\n");
						break;
					}
					//if (channels==2)
					//	speex_decode_stereo_int(output, frame_size, &stereo);

					/*Convert to short and save to output file*/
					for (int i = 0; i < frame_size * stream_channels; i++) {
						out[i] = le_short(out[i]);
					}

					{

						int new_frame_size = frame_size;

						/*printf ("packet %d %d\n", packet_no, skip_samples);*/
						if (packet_no == 1 && j == 0 && skip_samples > 0) {
							/*printf ("chopping first packet\n");*/
							new_frame_size -= skip_samples;
						}
						if (packet_no == page_nb_packets && skip_samples < 0) {
							int packet_length = nframes * frame_size + skip_samples;
							new_frame_size = packet_length - j * frame_size;
							if (new_frame_size < 0)
								new_frame_size = 0;
							if (new_frame_size > frame_size)
								new_frame_size = frame_size;
							/*printf ("chopping end: %d %d %d\n", new_frame_size, packet_length, packet_no);*/
						}

						p_buffer += new_frame_size * stream_channels;
						todo -= new_frame_size;
					}
				}
			};
		};
		//todo = get_todo();

		//todo is still greater than page size, can write more
		if (todo > page_size || eos) {
			if (read_ofs < data.size()) {

				//char *buf;
				int nb_read = MIN(data.size() - read_ofs, READ_CHUNK);

				/*Get the ogg buffer for writing*/
				char *ogg_dst = ogg_sync_buffer(&oy, nb_read);
				/*Read bitstream from input file*/
				copymem(ogg_dst, &data[read_ofs], nb_read);
				read_ofs += nb_read;
				ogg_sync_wrote(&oy, nb_read);
			} else {
				if (loops) {
					reload();
					++loop_count;
					//break;
				} else {
					playing = false;
					unload();
					break;
				};
			}
		};
	};

	return p_frames - todo;
};

void AudioStreamPlaybackSpeex::unload() {

	if (!active) return;

	speex_bits_destroy(&bits);
	if (st)
		speex_decoder_destroy(st);

	ogg_sync_clear(&oy);
	active = false;
	//data.resize(0);
	st = NULL;

	frame_size = 0;
	page_size = 0;
	loop_count = 0;
}

void *AudioStreamPlaybackSpeex::process_header(ogg_packet *op, int *frame_size, int *rate, int *nframes, int *channels, int *extra_headers) {

	SpeexHeader *header;
	int modeID;

	header = speex_packet_to_header((char *)op->packet, op->bytes);
	if (!header) {
		OS::get_singleton()->printerr("Cannot read header\n");
		return NULL;
	}
	if (header->mode >= SPEEX_NB_MODES) {
		OS::get_singleton()->printerr("Mode number %d does not (yet/any longer) exist in this version\n",
				header->mode);
		return NULL;
	}

	modeID = header->mode;

	const SpeexMode *mode = speex_lib_get_mode(modeID);

	if (header->speex_version_id > 1) {
		OS::get_singleton()->printerr("This file was encoded with Speex bit-stream version %d, which I don't know how to decode\n", header->speex_version_id);
		return NULL;
	}

	if (mode->bitstream_version < header->mode_bitstream_version) {
		OS::get_singleton()->printerr("The file was encoded with a newer version of Speex. You need to upgrade in order to play it.\n");
		return NULL;
	}
	if (mode->bitstream_version > header->mode_bitstream_version) {
		OS::get_singleton()->printerr("The file was encoded with an older version of Speex. You would need to downgrade the version in order to play it.\n");
		return NULL;
	}

	void *state = speex_decoder_init(mode);
	if (!state) {
		OS::get_singleton()->printerr("Decoder initialization failed.\n");
		return NULL;
	}
	//speex_decoder_ctl(state, SPEEX_SET_ENH, &enh_enabled);
	speex_decoder_ctl(state, SPEEX_GET_FRAME_SIZE, frame_size);

	if (!*rate)
		*rate = header->rate;

	speex_decoder_ctl(state, SPEEX_SET_SAMPLING_RATE, rate);

	*nframes = header->frames_per_packet;

	*channels = header->nb_channels;

	if (*channels != 1) {
		OS::get_singleton()->printerr("Only MONO speex streams supported\n");
		return NULL;
	}

	*extra_headers = header->extra_headers;

	speex_free(header);
	return state;
}

void AudioStreamPlaybackSpeex::reload() {

	if (active)
		unload();

	if (!data.size())
		return;

	ogg_sync_init(&oy);
	speex_bits_init(&bits);

	read_ofs = 0;
	//	char *buf;

	int packet_count = 0;
	int extra_headers = 0;
	int stream_init = 0;

	page_granule = 0;
	last_granule = 0;
	skip_samples = 0;
	page_nb_packets = 0;
	packets_available = false;
	packet_no = 0;

	int eos = 0;

	do {

		/*Get the ogg buffer for writing*/
		int nb_read = MIN(data.size() - read_ofs, READ_CHUNK);
		char *ogg_dst = ogg_sync_buffer(&oy, nb_read);
		/*Read bitstream from input file*/
		copymem(ogg_dst, &data[read_ofs], nb_read);
		read_ofs += nb_read;
		ogg_sync_wrote(&oy, nb_read);

		/*Loop for all complete pages we got (most likely only one)*/
		while (ogg_sync_pageout(&oy, &og) == 1) {

			if (stream_init == 0) {
				ogg_stream_init(&os, ogg_page_serialno(&og));
				stream_init = 1;
			}
			/*Add page to the bitstream*/
			ogg_stream_pagein(&os, &og);
			page_granule = ogg_page_granulepos(&og);
			page_nb_packets = ogg_page_packets(&og);
			if (page_granule > 0 && frame_size) {
				skip_samples = page_nb_packets * frame_size * nframes - (page_granule - last_granule);
				if (ogg_page_eos(&og))
					skip_samples = -skip_samples;
				/*else if (!ogg_page_bos(&og))
				  skip_samples = 0;*/
			} else {
				skip_samples = 0;
			}

			last_granule = page_granule;
			/*Extract all available packets*/
			while (!eos && ogg_stream_packetout(&os, &op) == 1) {
				/*If first packet, process as Speex header*/
				if (packet_count == 0) {
					int rate = 0;
					int channels;
					st = process_header(&op, &frame_size, &rate, &nframes, &channels, &extra_headers);
					if (!nframes)
						nframes = 1;
					if (!st) {
						unload();
						return;
					};

					page_size = nframes * frame_size;
					stream_srate = rate;
					stream_channels = channels;
					stream_minbuff_size = page_size;

				} else if (packet_count == 1) {
				} else if (packet_count <= 1 + extra_headers) {
					/* Ignore extra headers */
				};
			};
			++packet_count;
		};

	} while (packet_count <= extra_headers);

	active = true;
}

void AudioStreamPlaybackSpeex::_bind_methods() {

	//ObjectTypeDB::bind_method(_MD("set_file","file"),&AudioStreamPlaybackSpeex::set_file);
	//	ObjectTypeDB::bind_method(_MD("get_file"),&AudioStreamPlaybackSpeex::get_file);

	ObjectTypeDB::bind_method(_MD("_set_bundled"), &AudioStreamPlaybackSpeex::_set_bundled);
	ObjectTypeDB::bind_method(_MD("_get_bundled"), &AudioStreamPlaybackSpeex::_get_bundled);

	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "_bundled", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_BUNDLE), _SCS("_set_bundled"), _SCS("_get_bundled"));
	//ADD_PROPERTY( PropertyInfo(Variant::STRING,"file",PROPERTY_HINT_FILE,"*.spx"),_SCS("set_file"),_SCS("get_file"));
};

void AudioStreamPlaybackSpeex::_set_bundled(const Dictionary &dict) {

	ERR_FAIL_COND(!dict.has("filename"));
	ERR_FAIL_COND(!dict.has("data"));

	filename = dict["filename"];
	data = dict["data"];
};

Dictionary AudioStreamPlaybackSpeex::_get_bundled() const {

	Dictionary d;
	d["filename"] = filename;
	d["data"] = data;
	return d;
};

void AudioStreamPlaybackSpeex::set_data(const Vector<uint8_t> &p_data) {

	data = p_data;
	reload();
}

void AudioStreamPlaybackSpeex::play(float p_from_pos) {

	reload();
	if (!active)
		return;
	playing = true;
}
void AudioStreamPlaybackSpeex::stop() {

	unload();
	playing = false;
}
bool AudioStreamPlaybackSpeex::is_playing() const {

	return playing;
}

void AudioStreamPlaybackSpeex::set_loop(bool p_enable) {

	loops = p_enable;
}
bool AudioStreamPlaybackSpeex::has_loop() const {

	return loops;
}

float AudioStreamPlaybackSpeex::get_length() const {

	return 0;
}

String AudioStreamPlaybackSpeex::get_stream_name() const {

	return "";
}

int AudioStreamPlaybackSpeex::get_loop_count() const {

	return 0;
}

float AudioStreamPlaybackSpeex::get_pos() const {

	return 0;
}
void AudioStreamPlaybackSpeex::seek_pos(float p_time){

};

AudioStreamPlaybackSpeex::AudioStreamPlaybackSpeex() {

	active = false;
	st = NULL;
	stream_channels = 1;
	stream_srate = 1;
	stream_minbuff_size = 1;
	playing = false;
}

AudioStreamPlaybackSpeex::~AudioStreamPlaybackSpeex() {

	unload();
}

////////////////////////////////////////

void AudioStreamSpeex::set_file(const String &p_file) {

	if (this->file == p_file)
		return;

	this->file = p_file;

	if (p_file == "") {
		data.resize(0);
		return;
	};

	Error err;
	FileAccess *file = FileAccess::open(p_file, FileAccess::READ, &err);
	if (err != OK) {
		data.resize(0);
	};
	ERR_FAIL_COND(err != OK);

	this->file = p_file;
	data.resize(file->get_len());
	int read = file->get_buffer(&data[0], data.size());
	memdelete(file);
}

RES ResourceFormatLoaderAudioStreamSpeex::load(const String &p_path, const String &p_original_path, Error *r_error) {

	if (r_error)
		*r_error = OK;

	AudioStreamSpeex *stream = memnew(AudioStreamSpeex);
	stream->set_file(p_path);
	return Ref<AudioStreamSpeex>(stream);
}

void ResourceFormatLoaderAudioStreamSpeex::get_recognized_extensions(List<String> *p_extensions) const {

	p_extensions->push_back("spx");
}
bool ResourceFormatLoaderAudioStreamSpeex::handles_type(const String &p_type) const {

	return (p_type == "AudioStream" || p_type == "AudioStreamSpeex");
}

String ResourceFormatLoaderAudioStreamSpeex::get_resource_type(const String &p_path) const {

	if (p_path.extension().to_lower() == "spx")
		return "AudioStreamSpeex";
	return "";
}
