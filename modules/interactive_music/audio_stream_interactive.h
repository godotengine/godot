/*************************************************************************/
/*  audio_stream_interactive.h                                           */
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

#ifndef AUDIO_STREAM_INTERACTIVE_H
#define AUDIO_STREAM_INTERACTIVE_H

#include "core/io/resource.h"
#include "servers/audio/audio_stream.h"

class AudioStreamPlaybackInteractive;

class AudioStreamInteractive : public AudioStream {
	GDCLASS(AudioStreamInteractive, AudioStream)
	OBJ_SAVE_TYPE(AudioStream)
public:
	enum TransitionFromTime {
		TRANSITION_FROM_TIME_NOW,
		TRANSITION_FROM_TIME_NEXT_BEAT,
		TRANSITION_FROM_TIME_NEXT_BAR,
		TRANSITION_FROM_TIME_END,
	};

	enum TransitionToTime {
		TRANSITION_TO_TIME_SAME_POSITION,
		TRANSITION_TO_TIME_START,
		TRANSITION_TO_TIME_PREVIOUS_POSITION,
	};

	enum TransitionClip {
		TRANSITION_CLIP_ANY,
		TRANSITION_CLIP_MULTIPLE,
		TRANSITION_CLIP_SINGLE,
	};

	enum FadeMode {
		FADE_DISABLED,
		FADE_IN,
		FADE_OUT,
		FADE_CROSS
	};

	enum AutoAdvanceMode {
		AUTO_ADVANCE_DISABLED,
		AUTO_ADVANCE_ENABLED,
		AUTO_ADVANCE_RETURN_TO_HOLD,
	};

private:
	friend class AudioStreamPlaybackInteractive;
	int sample_rate = 44100;
	bool stereo = true;
	int initial_clip = 0;

	double time = 0;

	enum {
		MAX_CLIPS = 63, // because we use bitmasks for transition matching
		MAX_TRANSITIONS = 63
	};

	struct Clip {
		StringName name;
		Ref<AudioStream> stream;

		AutoAdvanceMode auto_advance = AUTO_ADVANCE_DISABLED;
		int auto_advance_next_clip = 0;
	};

	struct Transition {
		TransitionFromTime from_time = TRANSITION_FROM_TIME_NEXT_BEAT;
		TransitionToTime to_time = TRANSITION_TO_TIME_START;
		TransitionClip source = TRANSITION_CLIP_ANY;
		TransitionClip dest = TRANSITION_CLIP_ANY;
		int source_clip = 0;
		int dest_clip = 0;
		uint64_t source_mask = 0;
		uint64_t dest_mask = 0;
		FadeMode fade_mode = FADE_CROSS;
		int fade_beats = 1;
		bool use_filler_clip = false;
		int filler_clip = 0;
		bool hold_previous = false;
	};

	Clip clips[MAX_CLIPS];
	Transition transitions[MAX_TRANSITIONS];

	uint64_t version = 1; // used to stop playback instances for incompatibility
	int clip_count = 0;
	int transition_count = 0;

	HashSet<AudioStreamPlaybackInteractive *> playbacks;

#ifdef TOOLS_ENABLED

	int clip_preview_set = -1;
	bool clip_preview_changed = false;

	mutable String stream_name_cache;
	String _get_streams_hint() const;
	PackedStringArray _get_linked_undo_properties(const String &p_property, const Variant &p_new_value) const;
	void _inspector_array_swap_clip(uint32_t p_item_a, uint32_t p_item_b);

	void _set_transition_from_index(int p_transition, int p_index);
	int _get_transition_from_index(int p_transition) const;

	void _set_transition_to_index(int p_transition, int p_index);
	int _get_transition_to_index(int p_transition) const;

#endif
public:
	// CLIPS

#ifdef TOOLS_ENABLED
	void _set_clip_preview(int p_clip);
	int _get_clip_preview() const;
#endif
	void set_clip_count(int p_count);
	int get_clip_count() const;

	void set_initial_clip(int p_clip);
	int get_initial_clip() const;

	void set_clip_name(int p_clip, const StringName &p_name);
	StringName get_clip_name(int p_clip) const;

	void set_clip_stream(int p_clip, const Ref<AudioStream> &p_stream);
	Ref<AudioStream> get_clip_stream(int p_clip) const;

	void set_clip_auto_advance(int p_clip, AutoAdvanceMode p_mode);
	AutoAdvanceMode get_clip_auto_advance(int p_clip) const;

	void set_clip_auto_advance_next_clip(int p_clip, int p_index);
	int get_clip_auto_advance_next_clip(int p_clip) const;

	// TRANSITIONS

	void set_transition_count(int p_count);
	int get_transition_count();

	void set_transition_from_time(int p_transition, TransitionFromTime p_from);
	TransitionFromTime get_transition_from_time(int p_transition) const;

	void set_transition_to_time(int p_transition, TransitionToTime p_to);
	TransitionToTime get_transition_to_time(int p_transition) const;

	void set_transition_source(int p_transition, TransitionClip p_source);
	TransitionClip get_transition_source(int p_transition) const;

	void set_transition_source_clip(int p_transition, int p_index);
	int get_transition_source_clip(int p_transition) const;

	void set_transition_source_mask(int p_transition, uint64_t p_mask);
	uint64_t get_transition_source_mask(int p_transition) const;

	void set_transition_dest(int p_transition, TransitionClip p_dest);
	TransitionClip get_transition_dest(int p_transition) const;

	void set_transition_dest_clip(int p_transition, int p_index);
	int get_transition_dest_clip(int p_transition) const;

	void set_transition_dest_mask(int p_transition, uint64_t p_mask);
	uint64_t get_transition_dest_mask(int p_transition) const;

	void set_transition_fade_mode(int p_transition, FadeMode p_mode);
	FadeMode get_transition_fade_mode(int p_transition) const;

	void set_transition_fade_beats(int p_transition, float p_beats);
	float get_transition_fade_beats(int p_transition) const;

	void set_transition_use_filler_clip(int p_transition, bool p_enable);
	bool is_transition_using_filler_clip(int p_transition) const;

	void set_transition_filler_clip(int p_transition, int p_clip_index);
	int get_transition_filler_clip(int p_transition) const;

	void set_transition_holds_previous(int p_transition, bool p_hold);
	bool is_transition_holding_previous(int p_transition) const;

	virtual Ref<AudioStreamPlayback> instantiate_playback() override;
	virtual String get_stream_name() const override;
	virtual float get_length() const override { return 0; }
	AudioStreamInteractive();

protected:
	static void _bind_methods();
	void _validate_property(PropertyInfo &property) const;
};

VARIANT_ENUM_CAST(AudioStreamInteractive::TransitionFromTime)
VARIANT_ENUM_CAST(AudioStreamInteractive::TransitionToTime)
VARIANT_ENUM_CAST(AudioStreamInteractive::TransitionClip)
VARIANT_ENUM_CAST(AudioStreamInteractive::AutoAdvanceMode)
VARIANT_ENUM_CAST(AudioStreamInteractive::FadeMode)

class AudioStreamPlaybackInteractive : public AudioStreamPlayback {
	GDCLASS(AudioStreamPlaybackInteractive, AudioStreamPlayback)
	friend class AudioStreamInteractive;

private:
	Ref<AudioStreamInteractive> stream;
	uint64_t version = 0;

	enum {
		BUFFER_SIZE = 1024
	};

	AudioFrame mix_buffer[BUFFER_SIZE];
	AudioFrame temp_buffer[BUFFER_SIZE];

	struct State {
		Ref<AudioStream> stream;
		Ref<AudioStreamPlayback> playback;
		bool active = false;
		double fade_wait = 0; // Time to wait until fade kicks-in
		double fade_volume = 1.0;
		double fade_speed = 0; // Fade speed, negative or positive
		int auto_advance = -1;
		bool first_mix = true;
		double previous_position = 0;

		void reset_fade() {
			fade_wait = 0;
			fade_volume = 1.0;
			fade_speed = 0;
		}
	};

	State states[AudioStreamInteractive::MAX_CLIPS];
	int playback_current = -1;

	bool active = false;
	int return_memory = -1;

	void _mix_internal(int p_frames);
	void _mix_internal_state(int p_state_idx, int p_frames);

	void _queue(int p_to_clip_index, bool p_is_auto_advance);

	int switch_request = -1;

protected:
	static void _bind_methods();

public:
	virtual void start(float p_from_pos = 0.0) override;
	virtual void stop() override;
	virtual bool is_playing() const override;
	virtual int get_loop_count() const; // times it looped
	virtual float get_playback_position() const override;
	virtual void seek(float p_time) override;
	virtual int mix(AudioFrame *p_buffer, float p_rate_scale, int p_frames) override;

	virtual void tag_used_streams();

	void switch_to_clip(int p_index);

	AudioStreamPlaybackInteractive();
	~AudioStreamPlaybackInteractive();
};

#endif // AUDIO_STREAM_INTERACTIVE_H
