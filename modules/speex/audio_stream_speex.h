/*************************************************************************/
/*  audio_stream_speex.h                                                 */
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
#ifndef AUDIO_STREAM_SPEEX_H
#define AUDIO_STREAM_SPEEX_H

#include "io/resource_loader.h"
#include "os/file_access.h"
#include "os/thread_safe.h"
#include "scene/resources/audio_stream.h"

#include <speex/speex.h>
#include <speex/speex_callbacks.h>
#include <speex/speex_header.h>
#include <speex/speex_stereo.h>
// (akien) Prevents unbundling properly, but it's only for 2.1.x as speex
// is dropped in 3.0+, so don't want to lose too much time on this.
// Needed for speex_free (internal).
#include "thirdparty/speex/os_support.h"

#include <ogg/ogg.h>

class AudioStreamPlaybackSpeex : public AudioStreamPlayback {

	OBJ_TYPE(AudioStreamPlaybackSpeex, AudioStreamPlayback);

	void *st;
	SpeexBits bits;
	Vector<uint8_t> data;
	int read_ofs;
	bool active;
	String filename;
	int loop_count;
	bool loops;
	int page_size;
	bool playing;
	bool packets_available;

	void unload();
	void reload();

	ogg_sync_state oy;
	ogg_page og;
	ogg_packet op;
	ogg_stream_state os;
	int nframes;
	int frame_size;
	int packet_no;

	ogg_int64_t page_granule, last_granule;
	int skip_samples, page_nb_packets;
	int stream_channels;
	int stream_srate;
	int stream_minbuff_size;

	void *process_header(ogg_packet *op, int *frame_size, int *rate, int *nframes, int *channels, int *extra_headers);

	static void _bind_methods();

protected:
	//virtual bool _can_mix() const;

	Dictionary _get_bundled() const;
	void _set_bundled(const Dictionary &dict);

public:
	void set_data(const Vector<uint8_t> &p_data);

	virtual void play(float p_from_pos = 0);
	virtual void stop();
	virtual bool is_playing() const;

	virtual void set_loop(bool p_enable);
	virtual bool has_loop() const;

	virtual float get_length() const;

	virtual String get_stream_name() const;

	virtual int get_loop_count() const;

	virtual float get_pos() const;
	virtual void seek_pos(float p_time);

	virtual int get_channels() const { return stream_channels; }
	virtual int get_mix_rate() const { return stream_srate; }

	virtual int get_minimum_buffer_size() const { return stream_minbuff_size; }
	virtual int mix(int16_t *p_bufer, int p_frames);

	virtual void set_loop_restart_time(float p_time) {} //no loop restart, ignore

	AudioStreamPlaybackSpeex();
	~AudioStreamPlaybackSpeex();
};

class AudioStreamSpeex : public AudioStream {

	OBJ_TYPE(AudioStreamSpeex, AudioStream);

	Vector<uint8_t> data;
	String file;

public:
	Ref<AudioStreamPlayback> instance_playback() {
		Ref<AudioStreamPlaybackSpeex> pb = memnew(AudioStreamPlaybackSpeex);
		pb->set_data(data);
		return pb;
	}

	void set_file(const String &p_file);
};

class ResourceFormatLoaderAudioStreamSpeex : public ResourceFormatLoader {
public:
	virtual RES load(const String &p_path, const String &p_original_path = "", Error *r_error = NULL);
	virtual void get_recognized_extensions(List<String> *p_extensions) const;
	virtual bool handles_type(const String &p_type) const;
	virtual String get_resource_type(const String &p_path) const;
};

#endif // AUDIO_STREAM_SPEEX_H
