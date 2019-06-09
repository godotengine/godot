/*************************************************************************/
/*  audio_stream_sample.h                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#ifndef AUDIOSTREAMMUSIC_H
#define AUDIOSTREAMMUSIC_H
#include "core/io/resource_loader.h"
#include "core/os/file_access.h"
#include "core/os/thread_safe.h"

#include "servers/audio/audio_stream.h"
#include "jar_xm.h"

class AudioStreamMusic;

class AudioStreamPlaybackMusic : public AudioStreamPlayback {

	GDCLASS(AudioStreamPlaybackMusic, AudioStreamPlayback)
    enum {
                PCM_BUFFER_SIZE = 4096
    };

	bool active;
	friend class AudioStreamMusic;
	Ref<AudioStreamMusic> base;
	jar_xm_context_t* music_state_copy;
    void *pcm_buffer;

public:
	virtual void start(float p_from_pos = 0.0);
	virtual void stop();
	virtual bool is_playing() const;

	virtual int get_loop_count() const; //times it looped

	virtual float get_playback_position() const;
	virtual void seek(float p_time);

	virtual void mix(AudioFrame *p_buffer, float p_rate_scale, int p_frames);

	AudioStreamPlaybackMusic();
	~AudioStreamPlaybackMusic(); 
};

class AudioStreamMusic : public AudioStream {
	GDCLASS(AudioStreamMusic, AudioStream)
	RES_BASE_EXTENSION("xm")

private:
	friend class AudioStreamPlaybackMusic;

    jar_xm_context_t *musicptr;
	bool is_loaded;
	int mix_rate;

	PoolVector<uint8_t> mod_data;

protected:
	static void _bind_methods();

public:
	void set_mix_rate(int p_hz);
	int get_mix_rate() const;

	virtual float get_length() const; //if supported, otherwise return 0

	//void set_data(const PoolVector<uint8_t> &p_data);
	//PoolVector<uint8_t> get_data() const;

	virtual Ref<AudioStreamPlayback> instance_playback();
	virtual String get_stream_name() const;
	
	Error set_file(const String &p_file);

	AudioStreamMusic();
	~AudioStreamMusic();
};

class ResourceFormatLoaderAudioStreamMusic : public ResourceFormatLoader {
	GDCLASS(ResourceFormatLoaderAudioStreamMusic, ResourceFormatLoader)
public:
	virtual RES load(const String &p_path, const String &p_original_path = "", Error *r_error = NULL);
	virtual void get_recognized_extensions(List<String> *p_extensions) const;
	virtual bool handles_type(const String &p_type) const;
	virtual String get_resource_type(const String &p_path) const;
};



#endif // AUDIOSTREAMMusic_H
