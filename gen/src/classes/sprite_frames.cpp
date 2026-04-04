/**************************************************************************/
/*  sprite_frames.cpp                                                     */
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

#include <godot_cpp/classes/sprite_frames.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/texture2d.hpp>
#include <godot_cpp/variant/string_name.hpp>

namespace godot {

void SpriteFrames::add_animation(const StringName &p_anim) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpriteFrames::get_class_static()._native_ptr(), StringName("add_animation")._native_ptr(), 3304788590);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_anim);
}

bool SpriteFrames::has_animation(const StringName &p_anim) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpriteFrames::get_class_static()._native_ptr(), StringName("has_animation")._native_ptr(), 2619796661);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_anim);
}

void SpriteFrames::duplicate_animation(const StringName &p_anim_from, const StringName &p_anim_to) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpriteFrames::get_class_static()._native_ptr(), StringName("duplicate_animation")._native_ptr(), 3740211285);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_anim_from, &p_anim_to);
}

void SpriteFrames::remove_animation(const StringName &p_anim) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpriteFrames::get_class_static()._native_ptr(), StringName("remove_animation")._native_ptr(), 3304788590);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_anim);
}

void SpriteFrames::rename_animation(const StringName &p_anim, const StringName &p_newname) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpriteFrames::get_class_static()._native_ptr(), StringName("rename_animation")._native_ptr(), 3740211285);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_anim, &p_newname);
}

PackedStringArray SpriteFrames::get_animation_names() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpriteFrames::get_class_static()._native_ptr(), StringName("get_animation_names")._native_ptr(), 1139954409);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedStringArray()));
	return ::godot::internal::_call_native_mb_ret<PackedStringArray>(_gde_method_bind, _owner);
}

void SpriteFrames::set_animation_speed(const StringName &p_anim, double p_fps) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpriteFrames::get_class_static()._native_ptr(), StringName("set_animation_speed")._native_ptr(), 4135858297);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_fps_encoded;
	PtrToArg<double>::encode(p_fps, &p_fps_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_anim, &p_fps_encoded);
}

double SpriteFrames::get_animation_speed(const StringName &p_anim) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpriteFrames::get_class_static()._native_ptr(), StringName("get_animation_speed")._native_ptr(), 2349060816);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_anim);
}

void SpriteFrames::set_animation_loop(const StringName &p_anim, bool p_loop) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpriteFrames::get_class_static()._native_ptr(), StringName("set_animation_loop")._native_ptr(), 2524380260);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_loop_encoded;
	PtrToArg<bool>::encode(p_loop, &p_loop_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_anim, &p_loop_encoded);
}

bool SpriteFrames::get_animation_loop(const StringName &p_anim) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpriteFrames::get_class_static()._native_ptr(), StringName("get_animation_loop")._native_ptr(), 2619796661);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_anim);
}

void SpriteFrames::add_frame(const StringName &p_anim, const Ref<Texture2D> &p_texture, float p_duration, int32_t p_at_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpriteFrames::get_class_static()._native_ptr(), StringName("add_frame")._native_ptr(), 1351332740);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_duration_encoded;
	PtrToArg<double>::encode(p_duration, &p_duration_encoded);
	int64_t p_at_position_encoded;
	PtrToArg<int64_t>::encode(p_at_position, &p_at_position_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_anim, (p_texture != nullptr ? &p_texture->_owner : nullptr), &p_duration_encoded, &p_at_position_encoded);
}

void SpriteFrames::set_frame(const StringName &p_anim, int32_t p_idx, const Ref<Texture2D> &p_texture, float p_duration) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpriteFrames::get_class_static()._native_ptr(), StringName("set_frame")._native_ptr(), 56804795);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	double p_duration_encoded;
	PtrToArg<double>::encode(p_duration, &p_duration_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_anim, &p_idx_encoded, (p_texture != nullptr ? &p_texture->_owner : nullptr), &p_duration_encoded);
}

void SpriteFrames::remove_frame(const StringName &p_anim, int32_t p_idx) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpriteFrames::get_class_static()._native_ptr(), StringName("remove_frame")._native_ptr(), 2415702435);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_anim, &p_idx_encoded);
}

int32_t SpriteFrames::get_frame_count(const StringName &p_anim) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpriteFrames::get_class_static()._native_ptr(), StringName("get_frame_count")._native_ptr(), 2458036349);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_anim);
}

Ref<Texture2D> SpriteFrames::get_frame_texture(const StringName &p_anim, int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpriteFrames::get_class_static()._native_ptr(), StringName("get_frame_texture")._native_ptr(), 2900517879);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Texture2D>()));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return Ref<Texture2D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Texture2D>(_gde_method_bind, _owner, &p_anim, &p_idx_encoded));
}

float SpriteFrames::get_frame_duration(const StringName &p_anim, int32_t p_idx) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpriteFrames::get_class_static()._native_ptr(), StringName("get_frame_duration")._native_ptr(), 1129309260);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_anim, &p_idx_encoded);
}

void SpriteFrames::clear(const StringName &p_anim) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpriteFrames::get_class_static()._native_ptr(), StringName("clear")._native_ptr(), 3304788590);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_anim);
}

void SpriteFrames::clear_all() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(SpriteFrames::get_class_static()._native_ptr(), StringName("clear_all")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

} // namespace godot
