/**************************************************************************/
/*  animated_sprite3d.hpp                                                 */
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

#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/sprite_base3d.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/string_name.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class SpriteFrames;

class AnimatedSprite3D : public SpriteBase3D {
	GDEXTENSION_CLASS(AnimatedSprite3D, SpriteBase3D)

public:
	void set_sprite_frames(const Ref<SpriteFrames> &p_sprite_frames);
	Ref<SpriteFrames> get_sprite_frames() const;
	void set_animation(const StringName &p_name);
	StringName get_animation() const;
	void set_autoplay(const String &p_name);
	String get_autoplay() const;
	bool is_playing() const;
	void play(const StringName &p_name = StringName(), float p_custom_speed = 1.0, bool p_from_end = false);
	void play_backwards(const StringName &p_name = StringName());
	void pause();
	void stop();
	void set_frame(int32_t p_frame);
	int32_t get_frame() const;
	void set_frame_progress(float p_progress);
	float get_frame_progress() const;
	void set_frame_and_progress(int32_t p_frame, float p_progress);
	void set_speed_scale(float p_speed_scale);
	float get_speed_scale() const;
	float get_playing_speed() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		SpriteBase3D::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

