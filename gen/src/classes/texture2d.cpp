/**************************************************************************/
/*  texture2d.cpp                                                         */
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

#include <godot_cpp/classes/texture2d.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/image.hpp>
#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/variant/rect2.hpp>
#include <godot_cpp/variant/rid.hpp>

namespace godot {

int32_t Texture2D::get_width() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Texture2D::get_class_static()._native_ptr(), StringName("get_width")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

int32_t Texture2D::get_height() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Texture2D::get_class_static()._native_ptr(), StringName("get_height")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

Vector2 Texture2D::get_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Texture2D::get_class_static()._native_ptr(), StringName("get_size")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

bool Texture2D::has_alpha() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Texture2D::get_class_static()._native_ptr(), StringName("has_alpha")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void Texture2D::draw(const RID &p_canvas_item, const Vector2 &p_position, const Color &p_modulate, bool p_transpose) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Texture2D::get_class_static()._native_ptr(), StringName("draw")._native_ptr(), 2729649137);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_transpose_encoded;
	PtrToArg<bool>::encode(p_transpose, &p_transpose_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_canvas_item, &p_position, &p_modulate, &p_transpose_encoded);
}

void Texture2D::draw_rect(const RID &p_canvas_item, const Rect2 &p_rect, bool p_tile, const Color &p_modulate, bool p_transpose) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Texture2D::get_class_static()._native_ptr(), StringName("draw_rect")._native_ptr(), 3499451691);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_tile_encoded;
	PtrToArg<bool>::encode(p_tile, &p_tile_encoded);
	int8_t p_transpose_encoded;
	PtrToArg<bool>::encode(p_transpose, &p_transpose_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_canvas_item, &p_rect, &p_tile_encoded, &p_modulate, &p_transpose_encoded);
}

void Texture2D::draw_rect_region(const RID &p_canvas_item, const Rect2 &p_rect, const Rect2 &p_src_rect, const Color &p_modulate, bool p_transpose, bool p_clip_uv) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Texture2D::get_class_static()._native_ptr(), StringName("draw_rect_region")._native_ptr(), 2963678660);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_transpose_encoded;
	PtrToArg<bool>::encode(p_transpose, &p_transpose_encoded);
	int8_t p_clip_uv_encoded;
	PtrToArg<bool>::encode(p_clip_uv, &p_clip_uv_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_canvas_item, &p_rect, &p_src_rect, &p_modulate, &p_transpose_encoded, &p_clip_uv_encoded);
}

Ref<Image> Texture2D::get_image() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Texture2D::get_class_static()._native_ptr(), StringName("get_image")._native_ptr(), 4190603485);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Image>()));
	return Ref<Image>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Image>(_gde_method_bind, _owner));
}

Ref<Resource> Texture2D::create_placeholder() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(Texture2D::get_class_static()._native_ptr(), StringName("create_placeholder")._native_ptr(), 121922552);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<Resource>()));
	return Ref<Resource>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<Resource>(_gde_method_bind, _owner));
}

int32_t Texture2D::_get_width() const {
	return 0;
}

int32_t Texture2D::_get_height() const {
	return 0;
}

bool Texture2D::_is_pixel_opaque(int32_t p_x, int32_t p_y) const {
	return false;
}

bool Texture2D::_has_alpha() const {
	return false;
}

void Texture2D::_draw(const RID &p_to_canvas_item, const Vector2 &p_pos, const Color &p_modulate, bool p_transpose) const {}

void Texture2D::_draw_rect(const RID &p_to_canvas_item, const Rect2 &p_rect, bool p_tile, const Color &p_modulate, bool p_transpose) const {}

void Texture2D::_draw_rect_region(const RID &p_to_canvas_item, const Rect2 &p_rect, const Rect2 &p_src_rect, const Color &p_modulate, bool p_transpose, bool p_clip_uv) const {}

} // namespace godot
