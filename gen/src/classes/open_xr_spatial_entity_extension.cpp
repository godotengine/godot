/**************************************************************************/
/*  open_xr_spatial_entity_extension.cpp                                  */
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

#include <godot_cpp/classes/open_xr_spatial_entity_extension.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/open_xr_future_result.hpp>
#include <godot_cpp/classes/open_xr_spatial_capability_configuration_base_header.hpp>
#include <godot_cpp/classes/open_xr_spatial_component_data.hpp>
#include <godot_cpp/variant/packed_int64_array.hpp>

namespace godot {

bool OpenXRSpatialEntityExtension::supports_capability(OpenXRSpatialEntityExtension::Capability p_capability) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRSpatialEntityExtension::get_class_static()._native_ptr(), StringName("supports_capability")._native_ptr(), 1940837202);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_capability_encoded;
	PtrToArg<int64_t>::encode(p_capability, &p_capability_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_capability_encoded);
}

bool OpenXRSpatialEntityExtension::supports_component_type(OpenXRSpatialEntityExtension::Capability p_capability, OpenXRSpatialEntityExtension::ComponentType p_component_type) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRSpatialEntityExtension::get_class_static()._native_ptr(), StringName("supports_component_type")._native_ptr(), 26842779);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_capability_encoded;
	PtrToArg<int64_t>::encode(p_capability, &p_capability_encoded);
	int64_t p_component_type_encoded;
	PtrToArg<int64_t>::encode(p_component_type, &p_component_type_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_capability_encoded, &p_component_type_encoded);
}

Ref<OpenXRFutureResult> OpenXRSpatialEntityExtension::create_spatial_context(const TypedArray<Ref<OpenXRSpatialCapabilityConfigurationBaseHeader>> &p_capability_configurations, const Ref<OpenXRStructureBase> &p_next, const Callable &p_user_callback) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRSpatialEntityExtension::get_class_static()._native_ptr(), StringName("create_spatial_context")._native_ptr(), 1874506473);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<OpenXRFutureResult>()));
	return Ref<OpenXRFutureResult>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<OpenXRFutureResult>(_gde_method_bind, _owner, &p_capability_configurations, (p_next != nullptr ? &p_next->_owner : nullptr), &p_user_callback));
}

bool OpenXRSpatialEntityExtension::get_spatial_context_ready(const RID &p_spatial_context) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRSpatialEntityExtension::get_class_static()._native_ptr(), StringName("get_spatial_context_ready")._native_ptr(), 4155700596);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_spatial_context);
}

void OpenXRSpatialEntityExtension::free_spatial_context(const RID &p_spatial_context) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRSpatialEntityExtension::get_class_static()._native_ptr(), StringName("free_spatial_context")._native_ptr(), 2722037293);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_spatial_context);
}

uint64_t OpenXRSpatialEntityExtension::get_spatial_context_handle(const RID &p_spatial_context) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRSpatialEntityExtension::get_class_static()._native_ptr(), StringName("get_spatial_context_handle")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<uint64_t>(_gde_method_bind, _owner, &p_spatial_context);
}

Ref<OpenXRFutureResult> OpenXRSpatialEntityExtension::discover_spatial_entities(const RID &p_spatial_context, const PackedInt64Array &p_component_types, const Ref<OpenXRStructureBase> &p_next, const Callable &p_user_callback) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRSpatialEntityExtension::get_class_static()._native_ptr(), StringName("discover_spatial_entities")._native_ptr(), 2252833536);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Ref<OpenXRFutureResult>()));
	return Ref<OpenXRFutureResult>::_gde_internal_constructor(::godot::internal::_call_native_mb_ret_obj<OpenXRFutureResult>(_gde_method_bind, _owner, &p_spatial_context, &p_component_types, (p_next != nullptr ? &p_next->_owner : nullptr), &p_user_callback));
}

RID OpenXRSpatialEntityExtension::update_spatial_entities(const RID &p_spatial_context, const TypedArray<RID> &p_entities, const PackedInt64Array &p_component_types, const Ref<OpenXRStructureBase> &p_next) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRSpatialEntityExtension::get_class_static()._native_ptr(), StringName("update_spatial_entities")._native_ptr(), 3446086438);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_spatial_context, &p_entities, &p_component_types, (p_next != nullptr ? &p_next->_owner : nullptr));
}

void OpenXRSpatialEntityExtension::free_spatial_snapshot(const RID &p_spatial_snapshot) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRSpatialEntityExtension::get_class_static()._native_ptr(), StringName("free_spatial_snapshot")._native_ptr(), 2722037293);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_spatial_snapshot);
}

uint64_t OpenXRSpatialEntityExtension::get_spatial_snapshot_handle(const RID &p_spatial_snapshot) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRSpatialEntityExtension::get_class_static()._native_ptr(), StringName("get_spatial_snapshot_handle")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<uint64_t>(_gde_method_bind, _owner, &p_spatial_snapshot);
}

RID OpenXRSpatialEntityExtension::get_spatial_snapshot_context(const RID &p_spatial_snapshot) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRSpatialEntityExtension::get_class_static()._native_ptr(), StringName("get_spatial_snapshot_context")._native_ptr(), 3814569979);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_spatial_snapshot);
}

bool OpenXRSpatialEntityExtension::query_snapshot(const RID &p_spatial_snapshot, const TypedArray<Ref<OpenXRSpatialComponentData>> &p_component_data, const Ref<OpenXRStructureBase> &p_next) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRSpatialEntityExtension::get_class_static()._native_ptr(), StringName("query_snapshot")._native_ptr(), 641015484);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_spatial_snapshot, &p_component_data, (p_next != nullptr ? &p_next->_owner : nullptr));
}

String OpenXRSpatialEntityExtension::get_string(const RID &p_spatial_snapshot, uint64_t p_buffer_id) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRSpatialEntityExtension::get_class_static()._native_ptr(), StringName("get_string")._native_ptr(), 1464764419);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_buffer_id_encoded;
	PtrToArg<int64_t>::encode(p_buffer_id, &p_buffer_id_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_spatial_snapshot, &p_buffer_id_encoded);
}

PackedByteArray OpenXRSpatialEntityExtension::get_uint8_buffer(const RID &p_spatial_snapshot, uint64_t p_buffer_id) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRSpatialEntityExtension::get_class_static()._native_ptr(), StringName("get_uint8_buffer")._native_ptr(), 3570600051);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedByteArray()));
	int64_t p_buffer_id_encoded;
	PtrToArg<int64_t>::encode(p_buffer_id, &p_buffer_id_encoded);
	return ::godot::internal::_call_native_mb_ret<PackedByteArray>(_gde_method_bind, _owner, &p_spatial_snapshot, &p_buffer_id_encoded);
}

PackedInt32Array OpenXRSpatialEntityExtension::get_uint16_buffer(const RID &p_spatial_snapshot, uint64_t p_buffer_id) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRSpatialEntityExtension::get_class_static()._native_ptr(), StringName("get_uint16_buffer")._native_ptr(), 3393655756);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedInt32Array()));
	int64_t p_buffer_id_encoded;
	PtrToArg<int64_t>::encode(p_buffer_id, &p_buffer_id_encoded);
	return ::godot::internal::_call_native_mb_ret<PackedInt32Array>(_gde_method_bind, _owner, &p_spatial_snapshot, &p_buffer_id_encoded);
}

PackedInt32Array OpenXRSpatialEntityExtension::get_uint32_buffer(const RID &p_spatial_snapshot, uint64_t p_buffer_id) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRSpatialEntityExtension::get_class_static()._native_ptr(), StringName("get_uint32_buffer")._native_ptr(), 3393655756);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedInt32Array()));
	int64_t p_buffer_id_encoded;
	PtrToArg<int64_t>::encode(p_buffer_id, &p_buffer_id_encoded);
	return ::godot::internal::_call_native_mb_ret<PackedInt32Array>(_gde_method_bind, _owner, &p_spatial_snapshot, &p_buffer_id_encoded);
}

PackedFloat32Array OpenXRSpatialEntityExtension::get_float_buffer(const RID &p_spatial_snapshot, uint64_t p_buffer_id) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRSpatialEntityExtension::get_class_static()._native_ptr(), StringName("get_float_buffer")._native_ptr(), 2313216651);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedFloat32Array()));
	int64_t p_buffer_id_encoded;
	PtrToArg<int64_t>::encode(p_buffer_id, &p_buffer_id_encoded);
	return ::godot::internal::_call_native_mb_ret<PackedFloat32Array>(_gde_method_bind, _owner, &p_spatial_snapshot, &p_buffer_id_encoded);
}

PackedVector2Array OpenXRSpatialEntityExtension::get_vector2_buffer(const RID &p_spatial_snapshot, uint64_t p_buffer_id) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRSpatialEntityExtension::get_class_static()._native_ptr(), StringName("get_vector2_buffer")._native_ptr(), 110850971);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedVector2Array()));
	int64_t p_buffer_id_encoded;
	PtrToArg<int64_t>::encode(p_buffer_id, &p_buffer_id_encoded);
	return ::godot::internal::_call_native_mb_ret<PackedVector2Array>(_gde_method_bind, _owner, &p_spatial_snapshot, &p_buffer_id_encoded);
}

PackedVector3Array OpenXRSpatialEntityExtension::get_vector3_buffer(const RID &p_spatial_snapshot, uint64_t p_buffer_id) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRSpatialEntityExtension::get_class_static()._native_ptr(), StringName("get_vector3_buffer")._native_ptr(), 1166453791);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedVector3Array()));
	int64_t p_buffer_id_encoded;
	PtrToArg<int64_t>::encode(p_buffer_id, &p_buffer_id_encoded);
	return ::godot::internal::_call_native_mb_ret<PackedVector3Array>(_gde_method_bind, _owner, &p_spatial_snapshot, &p_buffer_id_encoded);
}

RID OpenXRSpatialEntityExtension::find_spatial_entity(uint64_t p_entity_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRSpatialEntityExtension::get_class_static()._native_ptr(), StringName("find_spatial_entity")._native_ptr(), 937000113);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	int64_t p_entity_id_encoded;
	PtrToArg<int64_t>::encode(p_entity_id, &p_entity_id_encoded);
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_entity_id_encoded);
}

RID OpenXRSpatialEntityExtension::add_spatial_entity(const RID &p_spatial_context, uint64_t p_entity_id, uint64_t p_entity) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRSpatialEntityExtension::get_class_static()._native_ptr(), StringName("add_spatial_entity")._native_ptr(), 2256026069);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	int64_t p_entity_id_encoded;
	PtrToArg<int64_t>::encode(p_entity_id, &p_entity_id_encoded);
	int64_t p_entity_encoded;
	PtrToArg<int64_t>::encode(p_entity, &p_entity_encoded);
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_spatial_context, &p_entity_id_encoded, &p_entity_encoded);
}

RID OpenXRSpatialEntityExtension::make_spatial_entity(const RID &p_spatial_context, uint64_t p_entity_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRSpatialEntityExtension::get_class_static()._native_ptr(), StringName("make_spatial_entity")._native_ptr(), 2233757277);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	int64_t p_entity_id_encoded;
	PtrToArg<int64_t>::encode(p_entity_id, &p_entity_id_encoded);
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_spatial_context, &p_entity_id_encoded);
}

uint64_t OpenXRSpatialEntityExtension::get_spatial_entity_id(const RID &p_entity) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRSpatialEntityExtension::get_class_static()._native_ptr(), StringName("get_spatial_entity_id")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<uint64_t>(_gde_method_bind, _owner, &p_entity);
}

RID OpenXRSpatialEntityExtension::get_spatial_entity_context(const RID &p_entity) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRSpatialEntityExtension::get_class_static()._native_ptr(), StringName("get_spatial_entity_context")._native_ptr(), 3814569979);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_entity);
}

void OpenXRSpatialEntityExtension::free_spatial_entity(const RID &p_entity) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRSpatialEntityExtension::get_class_static()._native_ptr(), StringName("free_spatial_entity")._native_ptr(), 2722037293);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_entity);
}

} // namespace godot
