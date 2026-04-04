/**************************************************************************/
/*  animation_mixer.hpp                                                   */
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

#include <godot_cpp/classes/global_constants.hpp>
#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/tween.hpp>
#include <godot_cpp/variant/node_path.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/quaternion.hpp>
#include <godot_cpp/variant/string_name.hpp>
#include <godot_cpp/variant/typed_array.hpp>
#include <godot_cpp/variant/variant.hpp>
#include <godot_cpp/variant/vector3.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Animation;
class AnimationLibrary;

class AnimationMixer : public Node {
	GDEXTENSION_CLASS(AnimationMixer, Node)

public:
	enum AnimationCallbackModeProcess {
		ANIMATION_CALLBACK_MODE_PROCESS_PHYSICS = 0,
		ANIMATION_CALLBACK_MODE_PROCESS_IDLE = 1,
		ANIMATION_CALLBACK_MODE_PROCESS_MANUAL = 2,
	};

	enum AnimationCallbackModeMethod {
		ANIMATION_CALLBACK_MODE_METHOD_DEFERRED = 0,
		ANIMATION_CALLBACK_MODE_METHOD_IMMEDIATE = 1,
	};

	enum AnimationCallbackModeDiscrete {
		ANIMATION_CALLBACK_MODE_DISCRETE_DOMINANT = 0,
		ANIMATION_CALLBACK_MODE_DISCRETE_RECESSIVE = 1,
		ANIMATION_CALLBACK_MODE_DISCRETE_FORCE_CONTINUOUS = 2,
	};

	Error add_animation_library(const StringName &p_name, const Ref<AnimationLibrary> &p_library);
	void remove_animation_library(const StringName &p_name);
	void rename_animation_library(const StringName &p_name, const StringName &p_newname);
	bool has_animation_library(const StringName &p_name) const;
	Ref<AnimationLibrary> get_animation_library(const StringName &p_name) const;
	TypedArray<StringName> get_animation_library_list() const;
	bool has_animation(const StringName &p_name) const;
	Ref<Animation> get_animation(const StringName &p_name) const;
	PackedStringArray get_animation_list() const;
	void set_active(bool p_active);
	bool is_active() const;
	void set_deterministic(bool p_deterministic);
	bool is_deterministic() const;
	void set_root_node(const NodePath &p_path);
	NodePath get_root_node() const;
	void set_callback_mode_process(AnimationMixer::AnimationCallbackModeProcess p_mode);
	AnimationMixer::AnimationCallbackModeProcess get_callback_mode_process() const;
	void set_callback_mode_method(AnimationMixer::AnimationCallbackModeMethod p_mode);
	AnimationMixer::AnimationCallbackModeMethod get_callback_mode_method() const;
	void set_callback_mode_discrete(AnimationMixer::AnimationCallbackModeDiscrete p_mode);
	AnimationMixer::AnimationCallbackModeDiscrete get_callback_mode_discrete() const;
	void set_audio_max_polyphony(int32_t p_max_polyphony);
	int32_t get_audio_max_polyphony() const;
	void set_root_motion_track(const NodePath &p_path);
	NodePath get_root_motion_track() const;
	void set_root_motion_local(bool p_enabled);
	bool is_root_motion_local() const;
	Vector3 get_root_motion_position() const;
	Quaternion get_root_motion_rotation() const;
	Vector3 get_root_motion_scale() const;
	Vector3 get_root_motion_position_accumulator() const;
	Quaternion get_root_motion_rotation_accumulator() const;
	Vector3 get_root_motion_scale_accumulator() const;
	void clear_caches();
	void advance(double p_delta);
	void capture(const StringName &p_name, double p_duration, Tween::TransitionType p_trans_type = (Tween::TransitionType)0, Tween::EaseType p_ease_type = (Tween::EaseType)0);
	void set_reset_on_save_enabled(bool p_enabled);
	bool is_reset_on_save_enabled() const;
	StringName find_animation(const Ref<Animation> &p_animation) const;
	StringName find_animation_library(const Ref<Animation> &p_animation) const;
	virtual Variant _post_process_key_value(const Ref<Animation> &p_animation, int32_t p_track, const Variant &p_value, uint64_t p_object_id, int32_t p_object_sub_idx) const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Node::register_virtuals<T, B>();
		if constexpr (!std::is_same_v<decltype(&B::_post_process_key_value), decltype(&T::_post_process_key_value)>) {
			BIND_VIRTUAL_METHOD(T, _post_process_key_value, 2716908335);
		}
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(AnimationMixer::AnimationCallbackModeProcess);
VARIANT_ENUM_CAST(AnimationMixer::AnimationCallbackModeMethod);
VARIANT_ENUM_CAST(AnimationMixer::AnimationCallbackModeDiscrete);

