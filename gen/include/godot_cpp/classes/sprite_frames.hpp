/**************************************************************************/
/*  sprite_frames.hpp                                                     */
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
#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class StringName;
class Texture2D;

class SpriteFrames : public Resource {
	GDEXTENSION_CLASS(SpriteFrames, Resource)

public:
	void add_animation(const StringName &p_anim);
	bool has_animation(const StringName &p_anim) const;
	void duplicate_animation(const StringName &p_anim_from, const StringName &p_anim_to);
	void remove_animation(const StringName &p_anim);
	void rename_animation(const StringName &p_anim, const StringName &p_newname);
	PackedStringArray get_animation_names() const;
	void set_animation_speed(const StringName &p_anim, double p_fps);
	double get_animation_speed(const StringName &p_anim) const;
	void set_animation_loop(const StringName &p_anim, bool p_loop);
	bool get_animation_loop(const StringName &p_anim) const;
	void add_frame(const StringName &p_anim, const Ref<Texture2D> &p_texture, float p_duration = 1.0, int32_t p_at_position = -1);
	void set_frame(const StringName &p_anim, int32_t p_idx, const Ref<Texture2D> &p_texture, float p_duration = 1.0);
	void remove_frame(const StringName &p_anim, int32_t p_idx);
	int32_t get_frame_count(const StringName &p_anim) const;
	Ref<Texture2D> get_frame_texture(const StringName &p_anim, int32_t p_idx) const;
	float get_frame_duration(const StringName &p_anim, int32_t p_idx) const;
	void clear(const StringName &p_anim);
	void clear_all();

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Resource::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

