/**************************************************************************/
/*  animation_node_animation.hpp                                          */
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

#include <godot_cpp/classes/animation.hpp>
#include <godot_cpp/classes/animation_root_node.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/variant/string_name.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class AnimationNodeAnimation : public AnimationRootNode {
	GDEXTENSION_CLASS(AnimationNodeAnimation, AnimationRootNode)

public:
	enum PlayMode {
		PLAY_MODE_FORWARD = 0,
		PLAY_MODE_BACKWARD = 1,
	};

	void set_animation(const StringName &p_name);
	StringName get_animation() const;
	void set_play_mode(AnimationNodeAnimation::PlayMode p_mode);
	AnimationNodeAnimation::PlayMode get_play_mode() const;
	void set_advance_on_start(bool p_advance_on_start);
	bool is_advance_on_start() const;
	void set_use_custom_timeline(bool p_use_custom_timeline);
	bool is_using_custom_timeline() const;
	void set_timeline_length(double p_timeline_length);
	double get_timeline_length() const;
	void set_stretch_time_scale(bool p_stretch_time_scale);
	bool is_stretching_time_scale() const;
	void set_start_offset(double p_start_offset);
	double get_start_offset() const;
	void set_loop_mode(Animation::LoopMode p_loop_mode);
	Animation::LoopMode get_loop_mode() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		AnimationRootNode::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(AnimationNodeAnimation::PlayMode);

