/*************************************************************************/
/*  gibberish_stream.h                                                   */
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
#ifndef GIBBERISH_STREAM_H
#define GIBBERISH_STREAM_H

//TODO: This class needs to be adapted to the new AudioStream API,
// or dropped if nobody cares about fixing it :) (GH-3307)

#if 0
#include "scene/resources/audio_stream.h"
#include "scene/resources/sample_library.h"
class AudioStreamGibberish : public AudioStream {

	GDCLASS( AudioStreamGibberish, AudioStream );

	enum {

		FP_BITS = 12,
		FP_LEN = (1<<12),
	};
	bool active;
	bool paused;

	float xfade_time;
	float pitch_scale;
	float pitch_random_scale;
	Vector<Ref<Sample> > _samples;
	Vector<int> _rand_pool;
	int rand_idx;
	_FORCE_INLINE_ int randomize();

	struct Playback {

		int idx;
		uint64_t fp_pos;
		float scale;
	};

	Playback playback[2];
	int active_voices;

	Ref<SampleLibrary> phonemes;
protected:

	virtual int get_channel_count() const;
	virtual bool mix(int32_t *p_buffer, int p_frames);

	static void _bind_methods();

public:

	void set_phonemes(const Ref<SampleLibrary>& p_phonemes);
	Ref<SampleLibrary> get_phonemes() const;

	virtual void play();
	virtual void stop();
	virtual bool is_playing() const;

	virtual void set_paused(bool p_paused);
	virtual bool is_paused(bool p_paused) const;

	virtual void set_loop(bool p_enable);
	virtual bool has_loop() const;

	virtual float get_length() const;

	virtual String get_stream_name() const;

	virtual int get_loop_count() const;

	virtual float get_pos() const;
	virtual void seek_pos(float p_time);

	virtual UpdateMode get_update_mode() const;
	virtual void update();

	void set_xfade_time(float p_xfade);
	float get_xfade_time() const;

	void set_pitch_scale(float p_scale);
	float get_pitch_scale() const;

	void set_pitch_random_scale(float p_random_scale);
	float get_pitch_random_scale() const;

	AudioStreamGibberish();
};

#endif

#endif // GIBBERISH_STREAM_H
