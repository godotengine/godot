/**************************************************************************/
/*  render_scene_data.cpp                                                 */
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

#include <godot_cpp/classes/render_scene_data.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

Transform3D RenderSceneData::get_cam_transform() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderSceneData::get_class_static()._native_ptr(), StringName("get_cam_transform")._native_ptr(), 3229777777);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Transform3D()));
	return ::godot::internal::_call_native_mb_ret<Transform3D>(_gde_method_bind, _owner);
}

Projection RenderSceneData::get_cam_projection() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderSceneData::get_class_static()._native_ptr(), StringName("get_cam_projection")._native_ptr(), 2910717950);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Projection()));
	return ::godot::internal::_call_native_mb_ret<Projection>(_gde_method_bind, _owner);
}

uint32_t RenderSceneData::get_view_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderSceneData::get_class_static()._native_ptr(), StringName("get_view_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

Vector3 RenderSceneData::get_view_eye_offset(uint32_t p_view) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderSceneData::get_class_static()._native_ptr(), StringName("get_view_eye_offset")._native_ptr(), 711720468);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	int64_t p_view_encoded;
	PtrToArg<int64_t>::encode(p_view, &p_view_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner, &p_view_encoded);
}

Projection RenderSceneData::get_view_projection(uint32_t p_view) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderSceneData::get_class_static()._native_ptr(), StringName("get_view_projection")._native_ptr(), 3179846605);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Projection()));
	int64_t p_view_encoded;
	PtrToArg<int64_t>::encode(p_view, &p_view_encoded);
	return ::godot::internal::_call_native_mb_ret<Projection>(_gde_method_bind, _owner, &p_view_encoded);
}

RID RenderSceneData::get_uniform_buffer() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(RenderSceneData::get_class_static()._native_ptr(), StringName("get_uniform_buffer")._native_ptr(), 2944877500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

} // namespace godot
