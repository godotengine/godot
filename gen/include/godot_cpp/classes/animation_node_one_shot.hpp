/**************************************************************************/
/*  animation_node_one_shot.hpp                                           */
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

#include <godot_cpp/classes/animation_node_sync.hpp>
#include <godot_cpp/classes/ref.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Curve;

class AnimationNodeOneShot : public AnimationNodeSync {
	GDEXTENSION_CLASS(AnimationNodeOneShot, AnimationNodeSync)

public:
	enum OneShotRequest {
		ONE_SHOT_REQUEST_NONE = 0,
		ONE_SHOT_REQUEST_FIRE = 1,
		ONE_SHOT_REQUEST_ABORT = 2,
		ONE_SHOT_REQUEST_FADE_OUT = 3,
	};

	enum MixMode {
		MIX_MODE_BLEND = 0,
		MIX_MODE_ADD = 1,
	};

	void set_fadein_time(double p_time);
	double get_fadein_time() const;
	void set_fadein_curve(const Ref<Curve> &p_curve);
	Ref<Curve> get_fadein_curve() const;
	void set_fadeout_time(double p_time);
	double get_fadeout_time() const;
	void set_fadeout_curve(const Ref<Curve> &p_curve);
	Ref<Curve> get_fadeout_curve() const;
	void set_break_loop_at_end(bool p_enable);
	bool is_loop_broken_at_end() const;
	void set_abort_on_reset(bool p_enable);
	bool is_aborted_on_reset() const;
	void set_autorestart(bool p_active);
	bool has_autorestart() const;
	void set_autorestart_delay(double p_time);
	double get_autorestart_delay() const;
	void set_autorestart_random_delay(double p_time);
	double get_autorestart_random_delay() const;
	void set_mix_mode(AnimationNodeOneShot::MixMode p_mode);
	AnimationNodeOneShot::MixMode get_mix_mode() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		AnimationNodeSync::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(AnimationNodeOneShot::OneShotRequest);
VARIANT_ENUM_CAST(AnimationNodeOneShot::MixMode);

