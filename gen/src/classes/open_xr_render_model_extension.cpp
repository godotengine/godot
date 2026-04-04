/**************************************************************************/
/*  open_xr_render_model_extension.cpp                                    */
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

#include <godot_cpp/classes/open_xr_render_model_extension.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/node3d.hpp>

namespace godot {

bool OpenXRRenderModelExtension::is_active() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRRenderModelExtension::get_class_static()._native_ptr(), StringName("is_active")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

RID OpenXRRenderModelExtension::render_model_create(uint64_t p_render_model_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRRenderModelExtension::get_class_static()._native_ptr(), StringName("render_model_create")._native_ptr(), 937000113);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	int64_t p_render_model_id_encoded;
	PtrToArg<int64_t>::encode(p_render_model_id, &p_render_model_id_encoded);
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_render_model_id_encoded);
}

void OpenXRRenderModelExtension::render_model_destroy(const RID &p_render_model) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRRenderModelExtension::get_class_static()._native_ptr(), StringName("render_model_destroy")._native_ptr(), 2722037293);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_render_model);
}

TypedArray<RID> OpenXRRenderModelExtension::render_model_get_all() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRRenderModelExtension::get_class_static()._native_ptr(), StringName("render_model_get_all")._native_ptr(), 2915620761);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<RID>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<RID>>(_gde_method_bind, _owner);
}

Node3D *OpenXRRenderModelExtension::render_model_new_scene_instance(const RID &p_render_model) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRRenderModelExtension::get_class_static()._native_ptr(), StringName("render_model_new_scene_instance")._native_ptr(), 788010739);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (nullptr));
	return ::godot::internal::_call_native_mb_ret_obj<Node3D>(_gde_method_bind, _owner, &p_render_model);
}

PackedStringArray OpenXRRenderModelExtension::render_model_get_subaction_paths(const RID &p_render_model) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRRenderModelExtension::get_class_static()._native_ptr(), StringName("render_model_get_subaction_paths")._native_ptr(), 2801473409);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedStringArray()));
	return ::godot::internal::_call_native_mb_ret<PackedStringArray>(_gde_method_bind, _owner, &p_render_model);
}

String OpenXRRenderModelExtension::render_model_get_top_level_path(const RID &p_render_model) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRRenderModelExtension::get_class_static()._native_ptr(), StringName("render_model_get_top_level_path")._native_ptr(), 642473191);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_render_model);
}

XRPose::TrackingConfidence OpenXRRenderModelExtension::render_model_get_confidence(const RID &p_render_model) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRRenderModelExtension::get_class_static()._native_ptr(), StringName("render_model_get_confidence")._native_ptr(), 2350330949);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (XRPose::TrackingConfidence(0)));
	return (XRPose::TrackingConfidence)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_render_model);
}

Transform3D OpenXRRenderModelExtension::render_model_get_root_transform(const RID &p_render_model) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRRenderModelExtension::get_class_static()._native_ptr(), StringName("render_model_get_root_transform")._native_ptr(), 1128465797);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Transform3D()));
	return ::godot::internal::_call_native_mb_ret<Transform3D>(_gde_method_bind, _owner, &p_render_model);
}

uint32_t OpenXRRenderModelExtension::render_model_get_animatable_node_count(const RID &p_render_model) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRRenderModelExtension::get_class_static()._native_ptr(), StringName("render_model_get_animatable_node_count")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_render_model);
}

String OpenXRRenderModelExtension::render_model_get_animatable_node_name(const RID &p_render_model, uint32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRRenderModelExtension::get_class_static()._native_ptr(), StringName("render_model_get_animatable_node_name")._native_ptr(), 1464764419);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_render_model, &p_index_encoded);
}

bool OpenXRRenderModelExtension::render_model_is_animatable_node_visible(const RID &p_render_model, uint32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRRenderModelExtension::get_class_static()._native_ptr(), StringName("render_model_is_animatable_node_visible")._native_ptr(), 3120086654);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_render_model, &p_index_encoded);
}

Transform3D OpenXRRenderModelExtension::render_model_get_animatable_node_transform(const RID &p_render_model, uint32_t p_index) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRRenderModelExtension::get_class_static()._native_ptr(), StringName("render_model_get_animatable_node_transform")._native_ptr(), 1050775521);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Transform3D()));
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	return ::godot::internal::_call_native_mb_ret<Transform3D>(_gde_method_bind, _owner, &p_render_model, &p_index_encoded);
}

} // namespace godot
