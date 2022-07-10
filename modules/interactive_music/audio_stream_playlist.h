/*************************************************************************/
/*  audio_stream_playlist.h                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef AUDIO_STREAM_PLAYLIST_H
#define AUDIO_STREAM_PLAYLIST_H

#include "core/io/resource.h"
#include "servers/audio/audio_stream.h"

class AudioStreamPlaybackPlaylist;

class AudioStreamPlaylist : public AudioStream {
	GDCLASS(AudioStreamPlaylist, AudioStream)
	OBJ_SAVE_TYPE(AudioStream)

private:
	friend class AudioStreamPlaybackPlaylist;
	uint64_t pos = 0;
	int sample_rate = 44100;
	bool stereo = true;
	int stream_count = 1;
	double bpm = 120;
	float length = 0;

	double time = 0;

	enum {
		MAX_STREAMS = 64
	};

	int beat_count = 0;
	bool shuffle = false;
	bool loop = true;

	Ref<AudioStream> audio_streams[MAX_STREAMS];
	HashSet<AudioStreamPlaybackPlaylist *> playbacks;

public:
	void reset();
	void set_position(uint64_t pos);

	void set_stream_beats(int beats);
	int get_stream_beats();
	void set_bpm(double beats_per_minute);
	virtual double get_bpm() const override;
	void set_stream_count(int count);
	int get_stream_count();
	void set_shuffle(bool p_shuffle);
	bool get_shuffle();
	void set_loop(bool p_loop);
	bool get_loop();
	void set_list_stream(int stream_number, Ref<AudioStream> p_stream);
	Ref<AudioStream> get_list_stream(int stream_number);

	virtual Ref<AudioStreamPlayback> instantiate_playback() override;
	virtual String get_stream_name() const override;
	virtual float get_length() const override;
	AudioStreamPlaylist();

protected:
	static void _bind_methods();
	void _validate_property(PropertyInfo &property) const;
};

///////////////////////////////////////

class AudioStreamPlaybackPlaylist : public AudioStreamPlayback {
	GDCLASS(AudioStreamPlaybackPlaylist, AudioStreamPlayback)
	friend class AudioStreamPlaylist;

private:
	enum {
		MIX_BUFFER_SIZE = 128
	};
	enum {
		MIX_FRAC_BITS = 13,
		MIX_FRAC_LEN = (1 << MIX_FRAC_BITS),
		MIX_FRAC_MASK = MIX_FRAC_LEN - 1,
	};
	AudioFrame pcm_buffer[MIX_BUFFER_SIZE];
	AudioFrame aux_buffer[MIX_BUFFER_SIZE];

	Ref<AudioStreamPlaylist> playlist;
	Ref<AudioStreamPlayback> playback[AudioStreamPlaylist::MAX_STREAMS];
	int bpm_list[AudioStreamPlaylist::MAX_STREAMS];
	int beats_list[AudioStreamPlaylist::MAX_STREAMS];

	Ref<AudioStream> last_stream;

	int current = 0;
	bool fading = false;
	int fading_samples_total = 0;
	int fading_time = 1;
	int fading_samples = 0;
	float beat_size = 0;
	int beat_amount_remaining = 0;

	bool active = false;

	virtual void _update_playback_instances();
	virtual void _update_bpm_info();

public:
	virtual void start(float p_from_pos = 0.0) override;
	virtual void stop() override;
	virtual bool is_playing() const override;
	void clear_buffer(int samples);
	virtual int get_loop_count() const override; // times it looped
	virtual float get_playback_position() const override;
	virtual void seek(float p_time) override;
	void add_stream_to_buffer(Ref<AudioStreamPlayback> playback, int samples, float p_rate_scale, float initial_volume, float final_volume);
	virtual int mix(AudioFrame *p_buffer, float p_rate_scale, int p_frames) override;

	virtual void tag_used_streams();

	AudioStreamPlaybackPlaylist();
	~AudioStreamPlaybackPlaylist();
};

#endif // AUDIO_STREAM_PLAYLIST_H
