/**************************************************************************/
/*  viewport.cpp                                                          */
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

#include <godot_cpp/classes/viewport.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/audio_listener2d.hpp>
#include <godot_cpp/classes/audio_listener3d.hpp>
#include <godot_cpp/classes/camera2d.hpp>
#include <godot_cpp/classes/camera3d.hpp>
#include <godot_cpp/classes/control.hpp>
#include <godot_cpp/classes/input_event.hpp>
#include <godot_cpp/classes/texture2d.hpp>
#include <godot_cpp/classes/viewport_texture.hpp>
#include <godot_cpp/classes/window.hpp>
#include <godot_cpp/classes/world2d.hpp>
#include <godot_cpp/classes/world3d.hpp>

namespace godot {

void Viewport::set_world_2d(const Ref<World2D> &p_world_2d) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("set_world_2d")._native_ptr(), 2736080068);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_world_2d != nullptr ? &p_world_2d->_owner : nullptr));
}

Ref<World2D> Viewport::get_world_2d() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("get_world_2d")._native_ptr(), 2339128592);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<World2D>()));
	return Ref<World2D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<World2D>(_gde_method_bind, _owner));
}

Ref<World2D> Viewport::find_world_2d() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("find_world_2d")._native_ptr(), 2339128592);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<World2D>()));
	return Ref<World2D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<World2D>(_gde_method_bind, _owner));
}

void Viewport::set_canvas_transform(const Transform2D &p_xform) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("set_canvas_transform")._native_ptr(), 2761652528);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_xform);
}

Transform2D Viewport::get_canvas_transform() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("get_canvas_transform")._native_ptr(), 3814499831);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Transform2D()));
	return ::godot::internal::_call_native_mb_ret<Transform2D>(_gde_method_bind, _owner);
}

void Viewport::set_global_canvas_transform(const Transform2D &p_xform) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("set_global_canvas_transform")._native_ptr(), 2761652528);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_xform);
}

Transform2D Viewport::get_global_canvas_transform() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("get_global_canvas_transform")._native_ptr(), 3814499831);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Transform2D()));
	return ::godot::internal::_call_native_mb_ret<Transform2D>(_gde_method_bind, _owner);
}

Transform2D Viewport::get_stretch_transform() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("get_stretch_transform")._native_ptr(), 3814499831);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Transform2D()));
	return ::godot::internal::_call_native_mb_ret<Transform2D>(_gde_method_bind, _owner);
}

Transform2D Viewport::get_final_transform() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("get_final_transform")._native_ptr(), 3814499831);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Transform2D()));
	return ::godot::internal::_call_native_mb_ret<Transform2D>(_gde_method_bind, _owner);
}

Transform2D Viewport::get_screen_transform() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("get_screen_transform")._native_ptr(), 3814499831);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Transform2D()));
	return ::godot::internal::_call_native_mb_ret<Transform2D>(_gde_method_bind, _owner);
}

Rect2 Viewport::get_visible_rect() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("get_visible_rect")._native_ptr(), 1639390495);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Rect2()));
	return ::godot::internal::_call_native_mb_ret<Rect2>(_gde_method_bind, _owner);
}

void Viewport::set_transparent_background(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("set_transparent_background")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool Viewport::has_transparent_background() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("has_transparent_background")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Viewport::set_use_hdr_2d(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("set_use_hdr_2d")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool Viewport::is_using_hdr_2d() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("is_using_hdr_2d")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Viewport::set_msaa_2d(Viewport::MSAA p_msaa) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("set_msaa_2d")._native_ptr(), 3330258708);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_msaa_encoded;
	PtrToArg<int64_t>::encode(p_msaa, &p_msaa_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_msaa_encoded);
}

Viewport::MSAA Viewport::get_msaa_2d() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("get_msaa_2d")._native_ptr(), 2542055527);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Viewport::MSAA(0)));
	return (Viewport::MSAA)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Viewport::set_msaa_3d(Viewport::MSAA p_msaa) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("set_msaa_3d")._native_ptr(), 3330258708);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_msaa_encoded;
	PtrToArg<int64_t>::encode(p_msaa, &p_msaa_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_msaa_encoded);
}

Viewport::MSAA Viewport::get_msaa_3d() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("get_msaa_3d")._native_ptr(), 2542055527);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Viewport::MSAA(0)));
	return (Viewport::MSAA)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Viewport::set_screen_space_aa(Viewport::ScreenSpaceAA p_screen_space_aa) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("set_screen_space_aa")._native_ptr(), 3544169389);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_screen_space_aa_encoded;
	PtrToArg<int64_t>::encode(p_screen_space_aa, &p_screen_space_aa_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_screen_space_aa_encoded);
}

Viewport::ScreenSpaceAA Viewport::get_screen_space_aa() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("get_screen_space_aa")._native_ptr(), 1390814124);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Viewport::ScreenSpaceAA(0)));
	return (Viewport::ScreenSpaceAA)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Viewport::set_use_taa(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("set_use_taa")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool Viewport::is_using_taa() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("is_using_taa")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Viewport::set_use_debanding(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("set_use_debanding")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool Viewport::is_using_debanding() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("is_using_debanding")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Viewport::set_use_occlusion_culling(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("set_use_occlusion_culling")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool Viewport::is_using_occlusion_culling() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("is_using_occlusion_culling")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Viewport::set_debug_draw(Viewport::DebugDraw p_debug_draw) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("set_debug_draw")._native_ptr(), 1970246205);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_debug_draw_encoded;
	PtrToArg<int64_t>::encode(p_debug_draw, &p_debug_draw_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_debug_draw_encoded);
}

Viewport::DebugDraw Viewport::get_debug_draw() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("get_debug_draw")._native_ptr(), 579191299);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Viewport::DebugDraw(0)));
	return (Viewport::DebugDraw)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Viewport::set_use_oversampling(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("set_use_oversampling")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool Viewport::is_using_oversampling() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("is_using_oversampling")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Viewport::set_oversampling_override(float p_oversampling) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("set_oversampling_override")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_oversampling_encoded;
	PtrToArg<double>::encode(p_oversampling, &p_oversampling_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_oversampling_encoded);
}

float Viewport::get_oversampling_override() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("get_oversampling_override")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

float Viewport::get_oversampling() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("get_oversampling")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

int32_t Viewport::get_render_info(Viewport::RenderInfoType p_type, Viewport::RenderInfo p_info) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("get_render_info")._native_ptr(), 481977019);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_type_encoded;
	PtrToArg<int64_t>::encode(p_type, &p_type_encoded);
	int64_t p_info_encoded;
	PtrToArg<int64_t>::encode(p_info, &p_info_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_type_encoded, &p_info_encoded);
}

Ref<ViewportTexture> Viewport::get_texture() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("get_texture")._native_ptr(), 1746695840);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<ViewportTexture>()));
	return Ref<ViewportTexture>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<ViewportTexture>(_gde_method_bind, _owner));
}

void Viewport::set_physics_object_picking(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("set_physics_object_picking")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool Viewport::get_physics_object_picking() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("get_physics_object_picking")._native_ptr(), 2240911060);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Viewport::set_physics_object_picking_sort(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("set_physics_object_picking_sort")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool Viewport::get_physics_object_picking_sort() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("get_physics_object_picking_sort")._native_ptr(), 2240911060);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Viewport::set_physics_object_picking_first_only(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("set_physics_object_picking_first_only")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool Viewport::get_physics_object_picking_first_only() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("get_physics_object_picking_first_only")._native_ptr(), 2240911060);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

RID Viewport::get_viewport_rid() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("get_viewport_rid")._native_ptr(), 2944877500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

void Viewport::push_text_input(const String &p_text) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("push_text_input")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_text);
}

void Viewport::push_input(const Ref<InputEvent> &p_event, bool p_in_local_coords) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("push_input")._native_ptr(), 3644664830);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_in_local_coords_encoded;
	PtrToArg<bool>::encode(p_in_local_coords, &p_in_local_coords_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_event != nullptr ? &p_event->_owner : nullptr), &p_in_local_coords_encoded);
}

void Viewport::push_unhandled_input(const Ref<InputEvent> &p_event, bool p_in_local_coords) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("push_unhandled_input")._native_ptr(), 3644664830);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_in_local_coords_encoded;
	PtrToArg<bool>::encode(p_in_local_coords, &p_in_local_coords_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_event != nullptr ? &p_event->_owner : nullptr), &p_in_local_coords_encoded);
}

void Viewport::notify_mouse_entered() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("notify_mouse_entered")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void Viewport::notify_mouse_exited() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("notify_mouse_exited")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

Vector2 Viewport::get_mouse_position() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("get_mouse_position")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

void Viewport::warp_mouse(const Vector2 &p_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("warp_mouse")._native_ptr(), 743155724);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_position);
}

void Viewport::update_mouse_cursor_state() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("update_mouse_cursor_state")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void Viewport::gui_cancel_drag() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("gui_cancel_drag")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

Variant Viewport::gui_get_drag_data() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("gui_get_drag_data")._native_ptr(), 1214101251);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant()));
	return ::godot::internal::_call_native_mb_ret<Variant>(_gde_method_bind, _owner);
}

String Viewport::gui_get_drag_description() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("gui_get_drag_description")._native_ptr(), 201670096);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner);
}

void Viewport::gui_set_drag_description(const String &p_description) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("gui_set_drag_description")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_description);
}

bool Viewport::gui_is_dragging() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("gui_is_dragging")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool Viewport::gui_is_drag_successful() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("gui_is_drag_successful")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Viewport::gui_release_focus() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("gui_release_focus")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

Control *Viewport::gui_get_focus_owner() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("gui_get_focus_owner")._native_ptr(), 2783021301);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<Control>(_gde_method_bind, _owner);
}

Control *Viewport::gui_get_hovered_control() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("gui_get_hovered_control")._native_ptr(), 2783021301);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<Control>(_gde_method_bind, _owner);
}

void Viewport::set_disable_input(bool p_disable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("set_disable_input")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_disable_encoded;
	PtrToArg<bool>::encode(p_disable, &p_disable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_disable_encoded);
}

bool Viewport::is_input_disabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("is_input_disabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Viewport::set_positional_shadow_atlas_size(int32_t p_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("set_positional_shadow_atlas_size")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_size_encoded;
	PtrToArg<int64_t>::encode(p_size, &p_size_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_size_encoded);
}

int32_t Viewport::get_positional_shadow_atlas_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("get_positional_shadow_atlas_size")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Viewport::set_positional_shadow_atlas_16_bits(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("set_positional_shadow_atlas_16_bits")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool Viewport::get_positional_shadow_atlas_16_bits() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("get_positional_shadow_atlas_16_bits")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Viewport::set_snap_controls_to_pixels(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("set_snap_controls_to_pixels")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool Viewport::is_snap_controls_to_pixels_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("is_snap_controls_to_pixels_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Viewport::set_snap_2d_transforms_to_pixel(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("set_snap_2d_transforms_to_pixel")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool Viewport::is_snap_2d_transforms_to_pixel_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("is_snap_2d_transforms_to_pixel_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Viewport::set_snap_2d_vertices_to_pixel(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("set_snap_2d_vertices_to_pixel")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool Viewport::is_snap_2d_vertices_to_pixel_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("is_snap_2d_vertices_to_pixel_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Viewport::set_positional_shadow_atlas_quadrant_subdiv(int32_t p_quadrant, Viewport::PositionalShadowAtlasQuadrantSubdiv p_subdiv) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("set_positional_shadow_atlas_quadrant_subdiv")._native_ptr(), 2596956071);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_quadrant_encoded;
	PtrToArg<int64_t>::encode(p_quadrant, &p_quadrant_encoded);
	int64_t p_subdiv_encoded;
	PtrToArg<int64_t>::encode(p_subdiv, &p_subdiv_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_quadrant_encoded, &p_subdiv_encoded);
}

Viewport::PositionalShadowAtlasQuadrantSubdiv Viewport::get_positional_shadow_atlas_quadrant_subdiv(int32_t p_quadrant) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("get_positional_shadow_atlas_quadrant_subdiv")._native_ptr(), 2676778355);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Viewport::PositionalShadowAtlasQuadrantSubdiv(0)));
	int64_t p_quadrant_encoded;
	PtrToArg<int64_t>::encode(p_quadrant, &p_quadrant_encoded);
	return (Viewport::PositionalShadowAtlasQuadrantSubdiv)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_quadrant_encoded);
}

void Viewport::set_input_as_handled() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("set_input_as_handled")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

bool Viewport::is_input_handled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("is_input_handled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Viewport::set_handle_input_locally(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("set_handle_input_locally")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool Viewport::is_handling_input_locally() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("is_handling_input_locally")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Viewport::set_default_canvas_item_texture_filter(Viewport::DefaultCanvasItemTextureFilter p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("set_default_canvas_item_texture_filter")._native_ptr(), 2815160100);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

Viewport::DefaultCanvasItemTextureFilter Viewport::get_default_canvas_item_texture_filter() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("get_default_canvas_item_texture_filter")._native_ptr(), 896601198);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Viewport::DefaultCanvasItemTextureFilter(0)));
	return (Viewport::DefaultCanvasItemTextureFilter)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Viewport::set_embedding_subwindows(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("set_embedding_subwindows")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool Viewport::is_embedding_subwindows() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("is_embedding_subwindows")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

TypedArray<Window> Viewport::get_embedded_subwindows() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("get_embedded_subwindows")._native_ptr(), 3995934104);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<Window>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<Window>>(_gde_method_bind, _owner);
}

void Viewport::set_drag_threshold(int32_t p_threshold) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("set_drag_threshold")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_threshold_encoded;
	PtrToArg<int64_t>::encode(p_threshold, &p_threshold_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_threshold_encoded);
}

int32_t Viewport::get_drag_threshold() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("get_drag_threshold")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Viewport::set_canvas_cull_mask(uint32_t p_mask) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("set_canvas_cull_mask")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mask_encoded;
	PtrToArg<int64_t>::encode(p_mask, &p_mask_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mask_encoded);
}

uint32_t Viewport::get_canvas_cull_mask() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("get_canvas_cull_mask")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Viewport::set_canvas_cull_mask_bit(uint32_t p_layer, bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("set_canvas_cull_mask_bit")._native_ptr(), 300928843);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_layer_encoded, &p_enable_encoded);
}

bool Viewport::get_canvas_cull_mask_bit(uint32_t p_layer) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("get_canvas_cull_mask_bit")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_layer_encoded;
	PtrToArg<int64_t>::encode(p_layer, &p_layer_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_layer_encoded);
}

void Viewport::set_default_canvas_item_texture_repeat(Viewport::DefaultCanvasItemTextureRepeat p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("set_default_canvas_item_texture_repeat")._native_ptr(), 1658513413);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

Viewport::DefaultCanvasItemTextureRepeat Viewport::get_default_canvas_item_texture_repeat() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("get_default_canvas_item_texture_repeat")._native_ptr(), 4049774160);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Viewport::DefaultCanvasItemTextureRepeat(0)));
	return (Viewport::DefaultCanvasItemTextureRepeat)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Viewport::set_sdf_oversize(Viewport::SDFOversize p_oversize) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("set_sdf_oversize")._native_ptr(), 2574159017);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_oversize_encoded;
	PtrToArg<int64_t>::encode(p_oversize, &p_oversize_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_oversize_encoded);
}

Viewport::SDFOversize Viewport::get_sdf_oversize() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("get_sdf_oversize")._native_ptr(), 2631427510);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Viewport::SDFOversize(0)));
	return (Viewport::SDFOversize)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Viewport::set_sdf_scale(Viewport::SDFScale p_scale) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("set_sdf_scale")._native_ptr(), 1402773951);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_scale_encoded;
	PtrToArg<int64_t>::encode(p_scale, &p_scale_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_scale_encoded);
}

Viewport::SDFScale Viewport::get_sdf_scale() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("get_sdf_scale")._native_ptr(), 3162688184);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Viewport::SDFScale(0)));
	return (Viewport::SDFScale)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Viewport::set_mesh_lod_threshold(float p_pixels) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("set_mesh_lod_threshold")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_pixels_encoded;
	PtrToArg<double>::encode(p_pixels, &p_pixels_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_pixels_encoded);
}

float Viewport::get_mesh_lod_threshold() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("get_mesh_lod_threshold")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Viewport::set_as_audio_listener_2d(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("set_as_audio_listener_2d")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool Viewport::is_audio_listener_2d() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("is_audio_listener_2d")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

AudioListener2D *Viewport::get_audio_listener_2d() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("get_audio_listener_2d")._native_ptr(), 1840977180);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<AudioListener2D>(_gde_method_bind, _owner);
}

Camera2D *Viewport::get_camera_2d() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("get_camera_2d")._native_ptr(), 3551466917);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<Camera2D>(_gde_method_bind, _owner);
}

void Viewport::set_world_3d(const Ref<World3D> &p_world_3d) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("set_world_3d")._native_ptr(), 1400875337);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_world_3d != nullptr ? &p_world_3d->_owner : nullptr));
}

Ref<World3D> Viewport::get_world_3d() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("get_world_3d")._native_ptr(), 317588385);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<World3D>()));
	return Ref<World3D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<World3D>(_gde_method_bind, _owner));
}

Ref<World3D> Viewport::find_world_3d() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("find_world_3d")._native_ptr(), 317588385);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<World3D>()));
	return Ref<World3D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<World3D>(_gde_method_bind, _owner));
}

void Viewport::set_use_own_world_3d(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("set_use_own_world_3d")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool Viewport::is_using_own_world_3d() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("is_using_own_world_3d")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

AudioListener3D *Viewport::get_audio_listener_3d() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("get_audio_listener_3d")._native_ptr(), 3472246991);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<AudioListener3D>(_gde_method_bind, _owner);
}

Camera3D *Viewport::get_camera_3d() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("get_camera_3d")._native_ptr(), 2285090890);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<Camera3D>(_gde_method_bind, _owner);
}

void Viewport::set_as_audio_listener_3d(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("set_as_audio_listener_3d")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool Viewport::is_audio_listener_3d() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("is_audio_listener_3d")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Viewport::set_disable_3d(bool p_disable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("set_disable_3d")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_disable_encoded;
	PtrToArg<bool>::encode(p_disable, &p_disable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_disable_encoded);
}

bool Viewport::is_3d_disabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("is_3d_disabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Viewport::set_use_xr(bool p_use) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("set_use_xr")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_use_encoded;
	PtrToArg<bool>::encode(p_use, &p_use_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_use_encoded);
}

bool Viewport::is_using_xr() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("is_using_xr")._native_ptr(), 2240911060);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Viewport::set_scaling_3d_mode(Viewport::Scaling3DMode p_scaling_3d_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("set_scaling_3d_mode")._native_ptr(), 1531597597);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_scaling_3d_mode_encoded;
	PtrToArg<int64_t>::encode(p_scaling_3d_mode, &p_scaling_3d_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_scaling_3d_mode_encoded);
}

Viewport::Scaling3DMode Viewport::get_scaling_3d_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("get_scaling_3d_mode")._native_ptr(), 2597660574);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Viewport::Scaling3DMode(0)));
	return (Viewport::Scaling3DMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Viewport::set_scaling_3d_scale(float p_scale) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("set_scaling_3d_scale")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_scale_encoded;
	PtrToArg<double>::encode(p_scale, &p_scale_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_scale_encoded);
}

float Viewport::get_scaling_3d_scale() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("get_scaling_3d_scale")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Viewport::set_fsr_sharpness(float p_fsr_sharpness) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("set_fsr_sharpness")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_fsr_sharpness_encoded;
	PtrToArg<double>::encode(p_fsr_sharpness, &p_fsr_sharpness_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_fsr_sharpness_encoded);
}

float Viewport::get_fsr_sharpness() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("get_fsr_sharpness")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Viewport::set_texture_mipmap_bias(float p_texture_mipmap_bias) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("set_texture_mipmap_bias")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_texture_mipmap_bias_encoded;
	PtrToArg<double>::encode(p_texture_mipmap_bias, &p_texture_mipmap_bias_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_texture_mipmap_bias_encoded);
}

float Viewport::get_texture_mipmap_bias() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("get_texture_mipmap_bias")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void Viewport::set_anisotropic_filtering_level(Viewport::AnisotropicFiltering p_anisotropic_filtering_level) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("set_anisotropic_filtering_level")._native_ptr(), 3445583046);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_anisotropic_filtering_level_encoded;
	PtrToArg<int64_t>::encode(p_anisotropic_filtering_level, &p_anisotropic_filtering_level_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_anisotropic_filtering_level_encoded);
}

Viewport::AnisotropicFiltering Viewport::get_anisotropic_filtering_level() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("get_anisotropic_filtering_level")._native_ptr(), 3991528932);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Viewport::AnisotropicFiltering(0)));
	return (Viewport::AnisotropicFiltering)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Viewport::set_vrs_mode(Viewport::VRSMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("set_vrs_mode")._native_ptr(), 2749867817);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

Viewport::VRSMode Viewport::get_vrs_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("get_vrs_mode")._native_ptr(), 349660525);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Viewport::VRSMode(0)));
	return (Viewport::VRSMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Viewport::set_vrs_update_mode(Viewport::VRSUpdateMode p_mode) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("set_vrs_update_mode")._native_ptr(), 3182412319);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mode_encoded;
	PtrToArg<int64_t>::encode(p_mode, &p_mode_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mode_encoded);
}

Viewport::VRSUpdateMode Viewport::get_vrs_update_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("get_vrs_update_mode")._native_ptr(), 2255951583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Viewport::VRSUpdateMode(0)));
	return (Viewport::VRSUpdateMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void Viewport::set_vrs_texture(const Ref<Texture2D> &p_texture) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("set_vrs_texture")._native_ptr(), 4051416890);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_texture != nullptr ? &p_texture->_owner : nullptr));
}

Ref<Texture2D> Viewport::get_vrs_texture() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Viewport::get_class_static()._native_ptr(), StringName("get_vrs_texture")._native_ptr(), 3635182373);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Texture2D>()));
	return Ref<Texture2D>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Texture2D>(_gde_method_bind, _owner));
}

} // namespace godot
