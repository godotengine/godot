/**************************************************************************/
/*  audio_effect_chorus.hpp                                               */
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

#include <godot_cpp/classes/audio_effect.hpp>
#include <godot_cpp/classes/ref.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class AudioEffectChorus : public AudioEffect {
	GDEXTENSION_CLASS(AudioEffectChorus, AudioEffect)

public:
	void set_voice_count(int32_t p_voices);
	int32_t get_voice_count() const;
	void set_voice_delay_ms(int32_t p_voice_idx, float p_delay_ms);
	float get_voice_delay_ms(int32_t p_voice_idx) const;
	void set_voice_rate_hz(int32_t p_voice_idx, float p_rate_hz);
	float get_voice_rate_hz(int32_t p_voice_idx) const;
	void set_voice_depth_ms(int32_t p_voice_idx, float p_depth_ms);
	float get_voice_depth_ms(int32_t p_voice_idx) const;
	void set_voice_level_db(int32_t p_voice_idx, float p_level_db);
	float get_voice_level_db(int32_t p_voice_idx) const;
	void set_voice_cutoff_hz(int32_t p_voice_idx, float p_cutoff_hz);
	float get_voice_cutoff_hz(int32_t p_voice_idx) const;
	void set_voice_pan(int32_t p_voice_idx, float p_pan);
	float get_voice_pan(int32_t p_voice_idx) const;
	void set_wet(float p_amount);
	float get_wet() const;
	void set_dry(float p_amount);
	float get_dry() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		AudioEffect::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

