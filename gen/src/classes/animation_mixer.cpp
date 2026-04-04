/**************************************************************************/
/*  animation_mixer.cpp                                                   */
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

#include <godot_cpp/classes/animation_mixer.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/animation.hpp>
#include <godot_cpp/classes/animation_library.hpp>

namespace godot {

Error AnimationMixer::add_animation_library(const StringName &p_name, const Ref<AnimationLibrary> &p_library) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationMixer::get_class_static()._native_ptr(), StringName("add_animation_library")._native_ptr(), 618909818);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_name, (p_library != nullptr ? &p_library->_owner : nullptr));
}

void AnimationMixer::remove_animation_library(const StringName &p_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationMixer::get_class_static()._native_ptr(), StringName("remove_animation_library")._native_ptr(), 3304788590);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name);
}

void AnimationMixer::rename_animation_library(const StringName &p_name, const StringName &p_newname) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationMixer::get_class_static()._native_ptr(), StringName("rename_animation_library")._native_ptr(), 3740211285);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name, &p_newname);
}

bool AnimationMixer::has_animation_library(const StringName &p_name) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationMixer::get_class_static()._native_ptr(), StringName("has_animation_library")._native_ptr(), 2619796661);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_name);
}

Ref<AnimationLibrary> AnimationMixer::get_animation_library(const StringName &p_name) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationMixer::get_class_static()._native_ptr(), StringName("get_animation_library")._native_ptr(), 147342321);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<AnimationLibrary>()));
	return Ref<AnimationLibrary>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<AnimationLibrary>(_gde_method_bind, _owner, &p_name));
}

TypedArray<StringName> AnimationMixer::get_animation_library_list() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationMixer::get_class_static()._native_ptr(), StringName("get_animation_library_list")._native_ptr(), 3995934104);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<StringName>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<StringName>>(_gde_method_bind, _owner);
}

bool AnimationMixer::has_animation(const StringName &p_name) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationMixer::get_class_static()._native_ptr(), StringName("has_animation")._native_ptr(), 2619796661);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_name);
}

Ref<Animation> AnimationMixer::get_animation(const StringName &p_name) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationMixer::get_class_static()._native_ptr(), StringName("get_animation")._native_ptr(), 2933122410);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Animation>()));
	return Ref<Animation>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Animation>(_gde_method_bind, _owner, &p_name));
}

PackedStringArray AnimationMixer::get_animation_list() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationMixer::get_class_static()._native_ptr(), StringName("get_animation_list")._native_ptr(), 1139954409);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedStringArray()));
	return ::godot::internal::_call_native_mb_ret<PackedStringArray>(_gde_method_bind, _owner);
}

void AnimationMixer::set_active(bool p_active) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationMixer::get_class_static()._native_ptr(), StringName("set_active")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_active_encoded;
	PtrToArg<bool>::encode(p_active, &p_active_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_active_encoded);
}

bool AnimationMixer::is_active() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationMixer::get_class_static()._native_ptr(), StringName("is_active")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void AnimationMixer::set_deterministic(bool p_deterministic) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationMixer::get_class_static()._native_ptr(), StringName("set_deterministic")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_deterministic_encoded;
	PtrToArg<bool>::encode(p_deterministic, &p_deterministic_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_deterministic_encoded);
}

bool AnimationMixer::is_deterministic() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationMixer::get_class_static()._native_ptr(), StringName("is_deterministic")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void AnimationMixer::set_root_node(const NodePath &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationMixer::get_class_static()._native_ptr(), StringName("set_root_node")._native_ptr(), 1348162250);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_path);
}

NodePath AnimationMixer::get_root_node() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationMixer::get_class_static()._native_ptr(), StringName("get_root_node")._native_ptr(), 4075236667);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (NodePath()));
	return ::godot::internal::_call_native_mb_ret<NodePath>(_gde_method_bind, _owner);
}

void AnimationMixer::set_callback_mode_process(AnimationMixer::AnimationCallbackModeProcess p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationMixer::get_class_static()._native_ptr(), StringName("set_callback_mode_process")._native_ptr(), 2153733086);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

AnimationMixer::AnimationCallbackModeProcess AnimationMixer::get_callback_mode_process() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationMixer::get_class_static()._native_ptr(), StringName("get_callback_mode_process")._native_ptr(), 1394468472);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (AnimationMixer::AnimationCallbackModeProcess(0)));
	return (AnimationMixer::AnimationCallbackModeProcess)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void AnimationMixer::set_callback_mode_method(AnimationMixer::AnimationCallbackModeMethod p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationMixer::get_class_static()._native_ptr(), StringName("set_callback_mode_method")._native_ptr(), 742218271);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

AnimationMixer::AnimationCallbackModeMethod AnimationMixer::get_callback_mode_method() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationMixer::get_class_static()._native_ptr(), StringName("get_callback_mode_method")._native_ptr(), 489449656);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (AnimationMixer::AnimationCallbackModeMethod(0)));
	return (AnimationMixer::AnimationCallbackModeMethod)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void AnimationMixer::set_callback_mode_discrete(AnimationMixer::AnimationCallbackModeDiscrete p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationMixer::get_class_static()._native_ptr(), StringName("set_callback_mode_discrete")._native_ptr(), 1998944670);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

AnimationMixer::AnimationCallbackModeDiscrete AnimationMixer::get_callback_mode_discrete() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationMixer::get_class_static()._native_ptr(), StringName("get_callback_mode_discrete")._native_ptr(), 3493168860);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (AnimationMixer::AnimationCallbackModeDiscrete(0)));
	return (AnimationMixer::AnimationCallbackModeDiscrete)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void AnimationMixer::set_audio_max_polyphony(int32_t p_max_polyphony) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationMixer::get_class_static()._native_ptr(), StringName("set_audio_max_polyphony")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_max_polyphony_encoded;
	PtrToArg<int64_t>::encode(p_max_polyphony, &p_max_polyphony_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_max_polyphony_encoded);
}

int32_t AnimationMixer::get_audio_max_polyphony() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationMixer::get_class_static()._native_ptr(), StringName("get_audio_max_polyphony")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void AnimationMixer::set_root_motion_track(const NodePath &p_path) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationMixer::get_class_static()._native_ptr(), StringName("set_root_motion_track")._native_ptr(), 1348162250);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_path);
}

NodePath AnimationMixer::get_root_motion_track() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationMixer::get_class_static()._native_ptr(), StringName("get_root_motion_track")._native_ptr(), 4075236667);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (NodePath()));
	return ::godot::internal::_call_native_mb_ret<NodePath>(_gde_method_bind, _owner);
}

void AnimationMixer::set_root_motion_local(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationMixer::get_class_static()._native_ptr(), StringName("set_root_motion_local")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool AnimationMixer::is_root_motion_local() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationMixer::get_class_static()._native_ptr(), StringName("is_root_motion_local")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

Vector3 AnimationMixer::get_root_motion_position() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationMixer::get_class_static()._native_ptr(), StringName("get_root_motion_position")._native_ptr(), 3360562783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

Quaternion AnimationMixer::get_root_motion_rotation() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationMixer::get_class_static()._native_ptr(), StringName("get_root_motion_rotation")._native_ptr(), 1222331677);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Quaternion()));
	return ::godot::internal::_call_native_mb_ret<Quaternion>(_gde_method_bind, _owner);
}

Vector3 AnimationMixer::get_root_motion_scale() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationMixer::get_class_static()._native_ptr(), StringName("get_root_motion_scale")._native_ptr(), 3360562783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

Vector3 AnimationMixer::get_root_motion_position_accumulator() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationMixer::get_class_static()._native_ptr(), StringName("get_root_motion_position_accumulator")._native_ptr(), 3360562783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

Quaternion AnimationMixer::get_root_motion_rotation_accumulator() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationMixer::get_class_static()._native_ptr(), StringName("get_root_motion_rotation_accumulator")._native_ptr(), 1222331677);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Quaternion()));
	return ::godot::internal::_call_native_mb_ret<Quaternion>(_gde_method_bind, _owner);
}

Vector3 AnimationMixer::get_root_motion_scale_accumulator() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationMixer::get_class_static()._native_ptr(), StringName("get_root_motion_scale_accumulator")._native_ptr(), 3360562783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

void AnimationMixer::clear_caches() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationMixer::get_class_static()._native_ptr(), StringName("clear_caches")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void AnimationMixer::advance(double p_delta) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationMixer::get_class_static()._native_ptr(), StringName("advance")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_delta_encoded;
	PtrToArg<double>::encode(p_delta, &p_delta_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_delta_encoded);
}

void AnimationMixer::capture(const StringName &p_name, double p_duration, Tween::TransitionType p_trans_type, Tween::EaseType p_ease_type) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationMixer::get_class_static()._native_ptr(), StringName("capture")._native_ptr(), 1333632127);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_duration_encoded;
	PtrToArg<double>::encode(p_duration, &p_duration_encoded);
	int64_t p_trans_type_encoded;
	PtrToArg<int64_t>::encode(p_trans_type, &p_trans_type_encoded);
	int64_t p_ease_type_encoded;
	PtrToArg<int64_t>::encode(p_ease_type, &p_ease_type_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name, &p_duration_encoded, &p_trans_type_encoded, &p_ease_type_encoded);
}

void AnimationMixer::set_reset_on_save_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationMixer::get_class_static()._native_ptr(), StringName("set_reset_on_save_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool AnimationMixer::is_reset_on_save_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationMixer::get_class_static()._native_ptr(), StringName("is_reset_on_save_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

StringName AnimationMixer::find_animation(const Ref<Animation> &p_animation) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationMixer::get_class_static()._native_ptr(), StringName("find_animation")._native_ptr(), 1559484580);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (StringName()));
	return ::godot::internal::_call_native_mb_ret<StringName>(_gde_method_bind, _owner, (p_animation != nullptr ? &p_animation->_owner : nullptr));
}

StringName AnimationMixer::find_animation_library(const Ref<Animation> &p_animation) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationMixer::get_class_static()._native_ptr(), StringName("find_animation_library")._native_ptr(), 1559484580);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (StringName()));
	return ::godot::internal::_call_native_mb_ret<StringName>(_gde_method_bind, _owner, (p_animation != nullptr ? &p_animation->_owner : nullptr));
}

Variant AnimationMixer::_post_process_key_value(const Ref<Animation> &p_animation, int32_t p_track, const Variant &p_value, uint64_t p_object_id, int32_t p_object_sub_idx) const {
	return Variant();
}

} // namespace godot
