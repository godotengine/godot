/**************************************************************************/
/*  audio_stream_interactive.hpp                                          */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#pragma once

#include <godot_cpp/classes/audio_stream.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>
#include <godot_cpp/variant/string_name.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class AudioStreamInteractive : public AudioStream {
	GDEXTENSION_CLASS(AudioStreamInteractive, AudioStream)

public:
	enum TransitionFromTime {
		TRANSITION_FROM_TIME_IMMEDIATE = 0,
		TRANSITION_FROM_TIME_NEXT_BEAT = 1,
		TRANSITION_FROM_TIME_NEXT_BAR = 2,
		TRANSITION_FROM_TIME_END = 3,
	};

	enum TransitionToTime {
		TRANSITION_TO_TIME_SAME_POSITION = 0,
		TRANSITION_TO_TIME_START = 1,
	};

	enum FadeMode {
		FADE_DISABLED = 0,
		FADE_IN = 1,
		FADE_OUT = 2,
		FADE_CROSS = 3,
		FADE_AUTOMATIC = 4,
	};

	enum AutoAdvanceMode {
		AUTO_ADVANCE_DISABLED = 0,
		AUTO_ADVANCE_ENABLED = 1,
		AUTO_ADVANCE_RETURN_TO_HOLD = 2,
	};

	static const int CLIP_ANY = -1;

	void set_clip_count(int32_t p_clip_count);
	int32_t get_clip_count() const;
	void set_initial_clip(int32_t p_clip_index);
	int32_t get_initial_clip() const;
	void set_clip_name(int32_t p_clip_index, const StringName &p_name);
	StringName get_clip_name(int32_t p_clip_index) const;
	void set_clip_stream(int32_t p_clip_index, const Ref<AudioStream> &p_stream);
	Ref<AudioStream> get_clip_stream(int32_t p_clip_index) const;
	void set_clip_auto_advance(int32_t p_clip_index, AudioStreamInteractive::AutoAdvanceMode p_mode);
	AudioStreamInteractive::AutoAdvanceMode get_clip_auto_advance(int32_t p_clip_index) const;
	void set_clip_auto_advance_next_clip(int32_t p_clip_index, int32_t p_auto_advance_next_clip);
	int32_t get_clip_auto_advance_next_clip(int32_t p_clip_index) const;
	void add_transition(int32_t p_from_clip, int32_t p_to_clip, AudioStreamInteractive::TransitionFromTime p_from_time, AudioStreamInteractive::TransitionToTime p_to_time, AudioStreamInteractive::FadeMode p_fade_mode, float p_fade_beats, bool p_use_filler_clip = false, int32_t p_filler_clip = -1, bool p_hold_previous = false);
	bool has_transition(int32_t p_from_clip, int32_t p_to_clip) const;
	void erase_transition(int32_t p_from_clip, int32_t p_to_clip);
	PackedInt32Array get_transition_list() const;
	AudioStreamInteractive::TransitionFromTime get_transition_from_time(int32_t p_from_clip, int32_t p_to_clip) const;
	AudioStreamInteractive::TransitionToTime get_transition_to_time(int32_t p_from_clip, int32_t p_to_clip) const;
	AudioStreamInteractive::FadeMode get_transition_fade_mode(int32_t p_from_clip, int32_t p_to_clip) const;
	float get_transition_fade_beats(int32_t p_from_clip, int32_t p_to_clip) const;
	bool is_transition_using_filler_clip(int32_t p_from_clip, int32_t p_to_clip) const;
	int32_t get_transition_filler_clip(int32_t p_from_clip, int32_t p_to_clip) const;
	bool is_transition_holding_previous(int32_t p_from_clip, int32_t p_to_clip) const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		AudioStream::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(AudioStreamInteractive::TransitionFromTime);
VARIANT_ENUM_CAST(AudioStreamInteractive::TransitionToTime);
VARIANT_ENUM_CAST(AudioStreamInteractive::FadeMode);
VARIANT_ENUM_CAST(AudioStreamInteractive::AutoAdvanceMode);

