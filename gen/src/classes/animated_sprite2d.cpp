/**************************************************************************/
/*  animated_sprite2d.cpp                                                 */
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

#include <godot_cpp/classes/animated_sprite2d.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/sprite_frames.hpp>

namespace godot {

void AnimatedSprite2D::set_sprite_frames(const Ref<SpriteFrames> &p_sprite_frames) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimatedSprite2D::get_class_static()._native_ptr(), StringName("set_sprite_frames")._native_ptr(), 905781144);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_sprite_frames != nullptr ? &p_sprite_frames->_owner : nullptr));
}

Ref<SpriteFrames> AnimatedSprite2D::get_sprite_frames() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimatedSprite2D::get_class_static()._native_ptr(), StringName("get_sprite_frames")._native_ptr(), 3804851214);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<SpriteFrames>()));
	return Ref<SpriteFrames>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<SpriteFrames>(_gde_method_bind, _owner));
}

void AnimatedSprite2D::set_animation(const StringName &p_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimatedSprite2D::get_class_static()._native_ptr(), StringName("set_animation")._native_ptr(), 3304788590);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name);
}

StringName AnimatedSprite2D::get_animation() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimatedSprite2D::get_class_static()._native_ptr(), StringName("get_animation")._native_ptr(), 2002593661);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (StringName()));
	return ::godot::internal::_call_native_mb_ret<StringName>(_gde_method_bind, _owner);
}

void AnimatedSprite2D::set_autoplay(const String &p_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimatedSprite2D::get_class_static()._native_ptr(), StringName("set_autoplay")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name);
}

String AnimatedSprite2D::get_autoplay() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimatedSprite2D::get_class_static()._native_ptr(), StringName("get_autoplay")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

bool AnimatedSprite2D::is_playing() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimatedSprite2D::get_class_static()._native_ptr(), StringName("is_playing")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void AnimatedSprite2D::play(const StringName &p_name, float p_custom_speed, bool p_from_end) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimatedSprite2D::get_class_static()._native_ptr(), StringName("play")._native_ptr(), 3269405555);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_custom_speed_encoded;
	PtrToArg<double>::encode(p_custom_speed, &p_custom_speed_encoded);
	int8_t p_from_end_encoded;
	PtrToArg<bool>::encode(p_from_end, &p_from_end_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name, &p_custom_speed_encoded, &p_from_end_encoded);
}

void AnimatedSprite2D::play_backwards(const StringName &p_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimatedSprite2D::get_class_static()._native_ptr(), StringName("play_backwards")._native_ptr(), 3323268493);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name);
}

void AnimatedSprite2D::pause() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimatedSprite2D::get_class_static()._native_ptr(), StringName("pause")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void AnimatedSprite2D::stop() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimatedSprite2D::get_class_static()._native_ptr(), StringName("stop")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void AnimatedSprite2D::set_centered(bool p_centered) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimatedSprite2D::get_class_static()._native_ptr(), StringName("set_centered")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_centered_encoded;
	PtrToArg<bool>::encode(p_centered, &p_centered_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_centered_encoded);
}

bool AnimatedSprite2D::is_centered() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimatedSprite2D::get_class_static()._native_ptr(), StringName("is_centered")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void AnimatedSprite2D::set_offset(const Vector2 &p_offset) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimatedSprite2D::get_class_static()._native_ptr(), StringName("set_offset")._native_ptr(), 743155724);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_offset);
}

Vector2 AnimatedSprite2D::get_offset() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimatedSprite2D::get_class_static()._native_ptr(), StringName("get_offset")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

void AnimatedSprite2D::set_flip_h(bool p_flip_h) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimatedSprite2D::get_class_static()._native_ptr(), StringName("set_flip_h")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_flip_h_encoded;
	PtrToArg<bool>::encode(p_flip_h, &p_flip_h_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_flip_h_encoded);
}

bool AnimatedSprite2D::is_flipped_h() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimatedSprite2D::get_class_static()._native_ptr(), StringName("is_flipped_h")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void AnimatedSprite2D::set_flip_v(bool p_flip_v) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimatedSprite2D::get_class_static()._native_ptr(), StringName("set_flip_v")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_flip_v_encoded;
	PtrToArg<bool>::encode(p_flip_v, &p_flip_v_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_flip_v_encoded);
}

bool AnimatedSprite2D::is_flipped_v() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimatedSprite2D::get_class_static()._native_ptr(), StringName("is_flipped_v")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void AnimatedSprite2D::set_frame(int32_t p_frame) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimatedSprite2D::get_class_static()._native_ptr(), StringName("set_frame")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_frame_encoded;
	PtrToArg<int64_t>::encode(p_frame, &p_frame_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_frame_encoded);
}

int32_t AnimatedSprite2D::get_frame() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimatedSprite2D::get_class_static()._native_ptr(), StringName("get_frame")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void AnimatedSprite2D::set_frame_progress(float p_progress) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimatedSprite2D::get_class_static()._native_ptr(), StringName("set_frame_progress")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_progress_encoded;
	PtrToArg<double>::encode(p_progress, &p_progress_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_progress_encoded);
}

float AnimatedSprite2D::get_frame_progress() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimatedSprite2D::get_class_static()._native_ptr(), StringName("get_frame_progress")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void AnimatedSprite2D::set_frame_and_progress(int32_t p_frame, float p_progress) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimatedSprite2D::get_class_static()._native_ptr(), StringName("set_frame_and_progress")._native_ptr(), 1602489585);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_frame_encoded;
	PtrToArg<int64_t>::encode(p_frame, &p_frame_encoded);
	double p_progress_encoded;
	PtrToArg<double>::encode(p_progress, &p_progress_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_frame_encoded, &p_progress_encoded);
}

void AnimatedSprite2D::set_speed_scale(float p_speed_scale) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimatedSprite2D::get_class_static()._native_ptr(), StringName("set_speed_scale")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_speed_scale_encoded;
	PtrToArg<double>::encode(p_speed_scale, &p_speed_scale_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_speed_scale_encoded);
}

float AnimatedSprite2D::get_speed_scale() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimatedSprite2D::get_class_static()._native_ptr(), StringName("get_speed_scale")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

float AnimatedSprite2D::get_playing_speed() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimatedSprite2D::get_class_static()._native_ptr(), StringName("get_playing_speed")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

} // namespace godot
