/**************************************************************************/
/*  canvas_item_material.cpp                                              */
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

#include <godot_cpp/classes/canvas_item_material.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

void CanvasItemMaterial::set_blend_mode(CanvasItemMaterial::BlendMode p_blend_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItemMaterial::get_class_static()._native_ptr(), StringName("set_blend_mode")._native_ptr(), 1786054936);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_blend_mode_encoded;
	PtrToArg<int64_t>::encode(p_blend_mode, &p_blend_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_blend_mode_encoded);
}

CanvasItemMaterial::BlendMode CanvasItemMaterial::get_blend_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItemMaterial::get_class_static()._native_ptr(), StringName("get_blend_mode")._native_ptr(), 3318684035);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (CanvasItemMaterial::BlendMode(0)));
	return (CanvasItemMaterial::BlendMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void CanvasItemMaterial::set_light_mode(CanvasItemMaterial::LightMode p_light_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItemMaterial::get_class_static()._native_ptr(), StringName("set_light_mode")._native_ptr(), 628074070);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_light_mode_encoded;
	PtrToArg<int64_t>::encode(p_light_mode, &p_light_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_light_mode_encoded);
}

CanvasItemMaterial::LightMode CanvasItemMaterial::get_light_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItemMaterial::get_class_static()._native_ptr(), StringName("get_light_mode")._native_ptr(), 3863292382);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (CanvasItemMaterial::LightMode(0)));
	return (CanvasItemMaterial::LightMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void CanvasItemMaterial::set_particles_animation(bool p_particles_anim) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItemMaterial::get_class_static()._native_ptr(), StringName("set_particles_animation")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_particles_anim_encoded;
	PtrToArg<bool>::encode(p_particles_anim, &p_particles_anim_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_particles_anim_encoded);
}

bool CanvasItemMaterial::get_particles_animation() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItemMaterial::get_class_static()._native_ptr(), StringName("get_particles_animation")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void CanvasItemMaterial::set_particles_anim_h_frames(int32_t p_frames) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItemMaterial::get_class_static()._native_ptr(), StringName("set_particles_anim_h_frames")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_frames_encoded;
	PtrToArg<int64_t>::encode(p_frames, &p_frames_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_frames_encoded);
}

int32_t CanvasItemMaterial::get_particles_anim_h_frames() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItemMaterial::get_class_static()._native_ptr(), StringName("get_particles_anim_h_frames")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void CanvasItemMaterial::set_particles_anim_v_frames(int32_t p_frames) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItemMaterial::get_class_static()._native_ptr(), StringName("set_particles_anim_v_frames")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_frames_encoded;
	PtrToArg<int64_t>::encode(p_frames, &p_frames_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_frames_encoded);
}

int32_t CanvasItemMaterial::get_particles_anim_v_frames() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItemMaterial::get_class_static()._native_ptr(), StringName("get_particles_anim_v_frames")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void CanvasItemMaterial::set_particles_anim_loop(bool p_loop) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItemMaterial::get_class_static()._native_ptr(), StringName("set_particles_anim_loop")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_loop_encoded;
	PtrToArg<bool>::encode(p_loop, &p_loop_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_loop_encoded);
}

bool CanvasItemMaterial::get_particles_anim_loop() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(CanvasItemMaterial::get_class_static()._native_ptr(), StringName("get_particles_anim_loop")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

} // namespace godot
