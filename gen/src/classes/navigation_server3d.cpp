/**************************************************************************/
/*  navigation_server3d.cpp                                               */
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

#include <godot_cpp/classes/navigation_server3d.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/navigation_mesh.hpp>
#include <godot_cpp/classes/navigation_mesh_source_geometry_data3d.hpp>
#include <godot_cpp/classes/navigation_path_query_parameters3d.hpp>
#include <godot_cpp/classes/navigation_path_query_result3d.hpp>
#include <godot_cpp/classes/node.hpp>

namespace godot {

NavigationServer3D *NavigationServer3D::singleton = nullptr;

NavigationServer3D *NavigationServer3D::get_singleton() {
	if (unlikely(singleton == nullptr)) {
		GDExtensionObjectPtr singleton_obj = ::godot::gdextension_interface::global_get_singleton(NavigationServer3D::get_class_static()._native_ptr());
#ifdef DEBUG_ENABLED
		ERR_FAIL_NULL_V(singleton_obj, nullptr);
#endif // DEBUG_ENABLED
		singleton = reinterpret_cast<NavigationServer3D *>(::godot::gdextension_interface::object_get_instance_binding(singleton_obj, ::godot::gdextension_interface::token, &NavigationServer3D::_gde_binding_callbacks));
#ifdef DEBUG_ENABLED
		ERR_FAIL_NULL_V(singleton, nullptr);
#endif // DEBUG_ENABLED
		if (likely(singleton)) {
			ClassDB::_register_engine_singleton(NavigationServer3D::get_class_static(), singleton);
		}
	}
	return singleton;
}

NavigationServer3D::~NavigationServer3D() {
	if (singleton == this) {
		ClassDB::_unregister_engine_singleton(NavigationServer3D::get_class_static());
		singleton = nullptr;
	}
}

TypedArray<RID> NavigationServer3D::get_maps() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("get_maps")._native_ptr(), 3995934104);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<RID>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<RID>>(_gde_method_bind, _owner);
}

RID NavigationServer3D::map_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("map_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

void NavigationServer3D::map_set_active(const RID &p_map, bool p_active) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("map_set_active")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_active_encoded;
	PtrToArg<bool>::encode(p_active, &p_active_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_map, &p_active_encoded);
}

bool NavigationServer3D::map_is_active(const RID &p_map) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("map_is_active")._native_ptr(), 4155700596);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_map);
}

void NavigationServer3D::map_set_up(const RID &p_map, const Vector3 &p_up) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("map_set_up")._native_ptr(), 3227306858);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_map, &p_up);
}

Vector3 NavigationServer3D::map_get_up(const RID &p_map) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("map_get_up")._native_ptr(), 531438156);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner, &p_map);
}

void NavigationServer3D::map_set_cell_size(const RID &p_map, float p_cell_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("map_set_cell_size")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_cell_size_encoded;
	PtrToArg<double>::encode(p_cell_size, &p_cell_size_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_map, &p_cell_size_encoded);
}

float NavigationServer3D::map_get_cell_size(const RID &p_map) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("map_get_cell_size")._native_ptr(), 866169185);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_map);
}

void NavigationServer3D::map_set_cell_height(const RID &p_map, float p_cell_height) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("map_set_cell_height")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_cell_height_encoded;
	PtrToArg<double>::encode(p_cell_height, &p_cell_height_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_map, &p_cell_height_encoded);
}

float NavigationServer3D::map_get_cell_height(const RID &p_map) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("map_get_cell_height")._native_ptr(), 866169185);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_map);
}

void NavigationServer3D::map_set_merge_rasterizer_cell_scale(const RID &p_map, float p_scale) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("map_set_merge_rasterizer_cell_scale")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_scale_encoded;
	PtrToArg<double>::encode(p_scale, &p_scale_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_map, &p_scale_encoded);
}

float NavigationServer3D::map_get_merge_rasterizer_cell_scale(const RID &p_map) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("map_get_merge_rasterizer_cell_scale")._native_ptr(), 866169185);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_map);
}

void NavigationServer3D::map_set_use_edge_connections(const RID &p_map, bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("map_set_use_edge_connections")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_map, &p_enabled_encoded);
}

bool NavigationServer3D::map_get_use_edge_connections(const RID &p_map) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("map_get_use_edge_connections")._native_ptr(), 4155700596);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_map);
}

void NavigationServer3D::map_set_edge_connection_margin(const RID &p_map, float p_margin) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("map_set_edge_connection_margin")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_margin_encoded;
	PtrToArg<double>::encode(p_margin, &p_margin_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_map, &p_margin_encoded);
}

float NavigationServer3D::map_get_edge_connection_margin(const RID &p_map) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("map_get_edge_connection_margin")._native_ptr(), 866169185);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_map);
}

void NavigationServer3D::map_set_link_connection_radius(const RID &p_map, float p_radius) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("map_set_link_connection_radius")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_radius_encoded;
	PtrToArg<double>::encode(p_radius, &p_radius_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_map, &p_radius_encoded);
}

float NavigationServer3D::map_get_link_connection_radius(const RID &p_map) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("map_get_link_connection_radius")._native_ptr(), 866169185);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_map);
}

PackedVector3Array NavigationServer3D::map_get_path(const RID &p_map, const Vector3 &p_origin, const Vector3 &p_destination, bool p_optimize, uint32_t p_navigation_layers) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("map_get_path")._native_ptr(), 276783190);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedVector3Array()));
	int8_t p_optimize_encoded;
	PtrToArg<bool>::encode(p_optimize, &p_optimize_encoded);
	int64_t p_navigation_layers_encoded;
	PtrToArg<int64_t>::encode(p_navigation_layers, &p_navigation_layers_encoded);
	return ::godot::internal::_call_native_mb_ret<PackedVector3Array>(_gde_method_bind, _owner, &p_map, &p_origin, &p_destination, &p_optimize_encoded, &p_navigation_layers_encoded);
}

Vector3 NavigationServer3D::map_get_closest_point_to_segment(const RID &p_map, const Vector3 &p_start, const Vector3 &p_end, bool p_use_collision) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("map_get_closest_point_to_segment")._native_ptr(), 3830095642);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	int8_t p_use_collision_encoded;
	PtrToArg<bool>::encode(p_use_collision, &p_use_collision_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner, &p_map, &p_start, &p_end, &p_use_collision_encoded);
}

Vector3 NavigationServer3D::map_get_closest_point(const RID &p_map, const Vector3 &p_to_point) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("map_get_closest_point")._native_ptr(), 2056183332);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner, &p_map, &p_to_point);
}

Vector3 NavigationServer3D::map_get_closest_point_normal(const RID &p_map, const Vector3 &p_to_point) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("map_get_closest_point_normal")._native_ptr(), 2056183332);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner, &p_map, &p_to_point);
}

RID NavigationServer3D::map_get_closest_point_owner(const RID &p_map, const Vector3 &p_to_point) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("map_get_closest_point_owner")._native_ptr(), 553364610);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_map, &p_to_point);
}

TypedArray<RID> NavigationServer3D::map_get_links(const RID &p_map) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("map_get_links")._native_ptr(), 2684255073);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<RID>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<RID>>(_gde_method_bind, _owner, &p_map);
}

TypedArray<RID> NavigationServer3D::map_get_regions(const RID &p_map) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("map_get_regions")._native_ptr(), 2684255073);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<RID>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<RID>>(_gde_method_bind, _owner, &p_map);
}

TypedArray<RID> NavigationServer3D::map_get_agents(const RID &p_map) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("map_get_agents")._native_ptr(), 2684255073);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<RID>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<RID>>(_gde_method_bind, _owner, &p_map);
}

TypedArray<RID> NavigationServer3D::map_get_obstacles(const RID &p_map) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("map_get_obstacles")._native_ptr(), 2684255073);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<RID>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<RID>>(_gde_method_bind, _owner, &p_map);
}

void NavigationServer3D::map_force_update(const RID &p_map) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("map_force_update")._native_ptr(), 2722037293);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_map);
}

uint32_t NavigationServer3D::map_get_iteration_id(const RID &p_map) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("map_get_iteration_id")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_map);
}

void NavigationServer3D::map_set_use_async_iterations(const RID &p_map, bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("map_set_use_async_iterations")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_map, &p_enabled_encoded);
}

bool NavigationServer3D::map_get_use_async_iterations(const RID &p_map) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("map_get_use_async_iterations")._native_ptr(), 4155700596);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_map);
}

Vector3 NavigationServer3D::map_get_random_point(const RID &p_map, uint32_t p_navigation_layers, bool p_uniformly) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("map_get_random_point")._native_ptr(), 722801526);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	int64_t p_navigation_layers_encoded;
	PtrToArg<int64_t>::encode(p_navigation_layers, &p_navigation_layers_encoded);
	int8_t p_uniformly_encoded;
	PtrToArg<bool>::encode(p_uniformly, &p_uniformly_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner, &p_map, &p_navigation_layers_encoded, &p_uniformly_encoded);
}

void NavigationServer3D::query_path(const Ref<NavigationPathQueryParameters3D> &p_parameters, const Ref<NavigationPathQueryResult3D> &p_result, const Callable &p_callback) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("query_path")._native_ptr(), 2146930868);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_parameters != nullptr ? &p_parameters->_owner : nullptr), (p_result != nullptr ? &p_result->_owner : nullptr), &p_callback);
}

RID NavigationServer3D::region_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("region_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

uint32_t NavigationServer3D::region_get_iteration_id(const RID &p_region) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("region_get_iteration_id")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_region);
}

void NavigationServer3D::region_set_use_async_iterations(const RID &p_region, bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("region_set_use_async_iterations")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_region, &p_enabled_encoded);
}

bool NavigationServer3D::region_get_use_async_iterations(const RID &p_region) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("region_get_use_async_iterations")._native_ptr(), 4155700596);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_region);
}

void NavigationServer3D::region_set_enabled(const RID &p_region, bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("region_set_enabled")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_region, &p_enabled_encoded);
}

bool NavigationServer3D::region_get_enabled(const RID &p_region) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("region_get_enabled")._native_ptr(), 4155700596);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_region);
}

void NavigationServer3D::region_set_use_edge_connections(const RID &p_region, bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("region_set_use_edge_connections")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_region, &p_enabled_encoded);
}

bool NavigationServer3D::region_get_use_edge_connections(const RID &p_region) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("region_get_use_edge_connections")._native_ptr(), 4155700596);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_region);
}

void NavigationServer3D::region_set_enter_cost(const RID &p_region, float p_enter_cost) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("region_set_enter_cost")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_enter_cost_encoded;
	PtrToArg<double>::encode(p_enter_cost, &p_enter_cost_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_region, &p_enter_cost_encoded);
}

float NavigationServer3D::region_get_enter_cost(const RID &p_region) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("region_get_enter_cost")._native_ptr(), 866169185);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_region);
}

void NavigationServer3D::region_set_travel_cost(const RID &p_region, float p_travel_cost) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("region_set_travel_cost")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_travel_cost_encoded;
	PtrToArg<double>::encode(p_travel_cost, &p_travel_cost_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_region, &p_travel_cost_encoded);
}

float NavigationServer3D::region_get_travel_cost(const RID &p_region) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("region_get_travel_cost")._native_ptr(), 866169185);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_region);
}

void NavigationServer3D::region_set_owner_id(const RID &p_region, uint64_t p_owner_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("region_set_owner_id")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_owner_id_encoded;
	PtrToArg<int64_t>::encode(p_owner_id, &p_owner_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_region, &p_owner_id_encoded);
}

uint64_t NavigationServer3D::region_get_owner_id(const RID &p_region) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("region_get_owner_id")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<uint64_t>(_gde_method_bind, _owner, &p_region);
}

bool NavigationServer3D::region_owns_point(const RID &p_region, const Vector3 &p_point) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("region_owns_point")._native_ptr(), 2360011153);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_region, &p_point);
}

void NavigationServer3D::region_set_map(const RID &p_region, const RID &p_map) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("region_set_map")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_region, &p_map);
}

RID NavigationServer3D::region_get_map(const RID &p_region) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("region_get_map")._native_ptr(), 3814569979);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_region);
}

void NavigationServer3D::region_set_navigation_layers(const RID &p_region, uint32_t p_navigation_layers) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("region_set_navigation_layers")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_navigation_layers_encoded;
	PtrToArg<int64_t>::encode(p_navigation_layers, &p_navigation_layers_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_region, &p_navigation_layers_encoded);
}

uint32_t NavigationServer3D::region_get_navigation_layers(const RID &p_region) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("region_get_navigation_layers")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_region);
}

void NavigationServer3D::region_set_transform(const RID &p_region, const Transform3D &p_transform) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("region_set_transform")._native_ptr(), 3935195649);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_region, &p_transform);
}

Transform3D NavigationServer3D::region_get_transform(const RID &p_region) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("region_get_transform")._native_ptr(), 1128465797);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Transform3D()));
	return ::godot::internal::_call_native_mb_ret<Transform3D>(_gde_method_bind, _owner, &p_region);
}

void NavigationServer3D::region_set_navigation_mesh(const RID &p_region, const Ref<NavigationMesh> &p_navigation_mesh) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("region_set_navigation_mesh")._native_ptr(), 2764952978);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_region, (p_navigation_mesh != nullptr ? &p_navigation_mesh->_owner : nullptr));
}

void NavigationServer3D::region_bake_navigation_mesh(const Ref<NavigationMesh> &p_navigation_mesh, Node *p_root_node) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("region_bake_navigation_mesh")._native_ptr(), 1401173477);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_navigation_mesh != nullptr ? &p_navigation_mesh->_owner : nullptr), (p_root_node != nullptr ? &p_root_node->_owner : nullptr));
}

int32_t NavigationServer3D::region_get_connections_count(const RID &p_region) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("region_get_connections_count")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_region);
}

Vector3 NavigationServer3D::region_get_connection_pathway_start(const RID &p_region, int32_t p_connection) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("region_get_connection_pathway_start")._native_ptr(), 3440143363);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	int64_t p_connection_encoded;
	PtrToArg<int64_t>::encode(p_connection, &p_connection_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner, &p_region, &p_connection_encoded);
}

Vector3 NavigationServer3D::region_get_connection_pathway_end(const RID &p_region, int32_t p_connection) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("region_get_connection_pathway_end")._native_ptr(), 3440143363);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	int64_t p_connection_encoded;
	PtrToArg<int64_t>::encode(p_connection, &p_connection_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner, &p_region, &p_connection_encoded);
}

Vector3 NavigationServer3D::region_get_closest_point_to_segment(const RID &p_region, const Vector3 &p_start, const Vector3 &p_end, bool p_use_collision) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("region_get_closest_point_to_segment")._native_ptr(), 3830095642);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	int8_t p_use_collision_encoded;
	PtrToArg<bool>::encode(p_use_collision, &p_use_collision_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner, &p_region, &p_start, &p_end, &p_use_collision_encoded);
}

Vector3 NavigationServer3D::region_get_closest_point(const RID &p_region, const Vector3 &p_to_point) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("region_get_closest_point")._native_ptr(), 2056183332);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner, &p_region, &p_to_point);
}

Vector3 NavigationServer3D::region_get_closest_point_normal(const RID &p_region, const Vector3 &p_to_point) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("region_get_closest_point_normal")._native_ptr(), 2056183332);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner, &p_region, &p_to_point);
}

Vector3 NavigationServer3D::region_get_random_point(const RID &p_region, uint32_t p_navigation_layers, bool p_uniformly) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("region_get_random_point")._native_ptr(), 722801526);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	int64_t p_navigation_layers_encoded;
	PtrToArg<int64_t>::encode(p_navigation_layers, &p_navigation_layers_encoded);
	int8_t p_uniformly_encoded;
	PtrToArg<bool>::encode(p_uniformly, &p_uniformly_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner, &p_region, &p_navigation_layers_encoded, &p_uniformly_encoded);
}

AABB NavigationServer3D::region_get_bounds(const RID &p_region) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("region_get_bounds")._native_ptr(), 974181306);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (AABB()));
	return ::godot::internal::_call_native_mb_ret<AABB>(_gde_method_bind, _owner, &p_region);
}

RID NavigationServer3D::link_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("link_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

uint32_t NavigationServer3D::link_get_iteration_id(const RID &p_link) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("link_get_iteration_id")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_link);
}

void NavigationServer3D::link_set_map(const RID &p_link, const RID &p_map) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("link_set_map")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_link, &p_map);
}

RID NavigationServer3D::link_get_map(const RID &p_link) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("link_get_map")._native_ptr(), 3814569979);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_link);
}

void NavigationServer3D::link_set_enabled(const RID &p_link, bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("link_set_enabled")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_link, &p_enabled_encoded);
}

bool NavigationServer3D::link_get_enabled(const RID &p_link) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("link_get_enabled")._native_ptr(), 4155700596);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_link);
}

void NavigationServer3D::link_set_bidirectional(const RID &p_link, bool p_bidirectional) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("link_set_bidirectional")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_bidirectional_encoded;
	PtrToArg<bool>::encode(p_bidirectional, &p_bidirectional_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_link, &p_bidirectional_encoded);
}

bool NavigationServer3D::link_is_bidirectional(const RID &p_link) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("link_is_bidirectional")._native_ptr(), 4155700596);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_link);
}

void NavigationServer3D::link_set_navigation_layers(const RID &p_link, uint32_t p_navigation_layers) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("link_set_navigation_layers")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_navigation_layers_encoded;
	PtrToArg<int64_t>::encode(p_navigation_layers, &p_navigation_layers_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_link, &p_navigation_layers_encoded);
}

uint32_t NavigationServer3D::link_get_navigation_layers(const RID &p_link) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("link_get_navigation_layers")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_link);
}

void NavigationServer3D::link_set_start_position(const RID &p_link, const Vector3 &p_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("link_set_start_position")._native_ptr(), 3227306858);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_link, &p_position);
}

Vector3 NavigationServer3D::link_get_start_position(const RID &p_link) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("link_get_start_position")._native_ptr(), 531438156);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner, &p_link);
}

void NavigationServer3D::link_set_end_position(const RID &p_link, const Vector3 &p_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("link_set_end_position")._native_ptr(), 3227306858);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_link, &p_position);
}

Vector3 NavigationServer3D::link_get_end_position(const RID &p_link) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("link_get_end_position")._native_ptr(), 531438156);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner, &p_link);
}

void NavigationServer3D::link_set_enter_cost(const RID &p_link, float p_enter_cost) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("link_set_enter_cost")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_enter_cost_encoded;
	PtrToArg<double>::encode(p_enter_cost, &p_enter_cost_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_link, &p_enter_cost_encoded);
}

float NavigationServer3D::link_get_enter_cost(const RID &p_link) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("link_get_enter_cost")._native_ptr(), 866169185);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_link);
}

void NavigationServer3D::link_set_travel_cost(const RID &p_link, float p_travel_cost) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("link_set_travel_cost")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_travel_cost_encoded;
	PtrToArg<double>::encode(p_travel_cost, &p_travel_cost_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_link, &p_travel_cost_encoded);
}

float NavigationServer3D::link_get_travel_cost(const RID &p_link) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("link_get_travel_cost")._native_ptr(), 866169185);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_link);
}

void NavigationServer3D::link_set_owner_id(const RID &p_link, uint64_t p_owner_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("link_set_owner_id")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_owner_id_encoded;
	PtrToArg<int64_t>::encode(p_owner_id, &p_owner_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_link, &p_owner_id_encoded);
}

uint64_t NavigationServer3D::link_get_owner_id(const RID &p_link) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("link_get_owner_id")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<uint64_t>(_gde_method_bind, _owner, &p_link);
}

RID NavigationServer3D::agent_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("agent_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

void NavigationServer3D::agent_set_avoidance_enabled(const RID &p_agent, bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("agent_set_avoidance_enabled")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_agent, &p_enabled_encoded);
}

bool NavigationServer3D::agent_get_avoidance_enabled(const RID &p_agent) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("agent_get_avoidance_enabled")._native_ptr(), 4155700596);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_agent);
}

void NavigationServer3D::agent_set_use_3d_avoidance(const RID &p_agent, bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("agent_set_use_3d_avoidance")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_agent, &p_enabled_encoded);
}

bool NavigationServer3D::agent_get_use_3d_avoidance(const RID &p_agent) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("agent_get_use_3d_avoidance")._native_ptr(), 4155700596);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_agent);
}

void NavigationServer3D::agent_set_map(const RID &p_agent, const RID &p_map) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("agent_set_map")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_agent, &p_map);
}

RID NavigationServer3D::agent_get_map(const RID &p_agent) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("agent_get_map")._native_ptr(), 3814569979);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_agent);
}

void NavigationServer3D::agent_set_paused(const RID &p_agent, bool p_paused) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("agent_set_paused")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_paused_encoded;
	PtrToArg<bool>::encode(p_paused, &p_paused_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_agent, &p_paused_encoded);
}

bool NavigationServer3D::agent_get_paused(const RID &p_agent) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("agent_get_paused")._native_ptr(), 4155700596);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_agent);
}

void NavigationServer3D::agent_set_neighbor_distance(const RID &p_agent, float p_distance) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("agent_set_neighbor_distance")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_distance_encoded;
	PtrToArg<double>::encode(p_distance, &p_distance_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_agent, &p_distance_encoded);
}

float NavigationServer3D::agent_get_neighbor_distance(const RID &p_agent) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("agent_get_neighbor_distance")._native_ptr(), 866169185);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_agent);
}

void NavigationServer3D::agent_set_max_neighbors(const RID &p_agent, int32_t p_count) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("agent_set_max_neighbors")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_count_encoded;
	PtrToArg<int64_t>::encode(p_count, &p_count_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_agent, &p_count_encoded);
}

int32_t NavigationServer3D::agent_get_max_neighbors(const RID &p_agent) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("agent_get_max_neighbors")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_agent);
}

void NavigationServer3D::agent_set_time_horizon_agents(const RID &p_agent, float p_time_horizon) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("agent_set_time_horizon_agents")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_time_horizon_encoded;
	PtrToArg<double>::encode(p_time_horizon, &p_time_horizon_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_agent, &p_time_horizon_encoded);
}

float NavigationServer3D::agent_get_time_horizon_agents(const RID &p_agent) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("agent_get_time_horizon_agents")._native_ptr(), 866169185);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_agent);
}

void NavigationServer3D::agent_set_time_horizon_obstacles(const RID &p_agent, float p_time_horizon) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("agent_set_time_horizon_obstacles")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_time_horizon_encoded;
	PtrToArg<double>::encode(p_time_horizon, &p_time_horizon_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_agent, &p_time_horizon_encoded);
}

float NavigationServer3D::agent_get_time_horizon_obstacles(const RID &p_agent) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("agent_get_time_horizon_obstacles")._native_ptr(), 866169185);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_agent);
}

void NavigationServer3D::agent_set_radius(const RID &p_agent, float p_radius) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("agent_set_radius")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_radius_encoded;
	PtrToArg<double>::encode(p_radius, &p_radius_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_agent, &p_radius_encoded);
}

float NavigationServer3D::agent_get_radius(const RID &p_agent) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("agent_get_radius")._native_ptr(), 866169185);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_agent);
}

void NavigationServer3D::agent_set_height(const RID &p_agent, float p_height) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("agent_set_height")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_height_encoded;
	PtrToArg<double>::encode(p_height, &p_height_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_agent, &p_height_encoded);
}

float NavigationServer3D::agent_get_height(const RID &p_agent) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("agent_get_height")._native_ptr(), 866169185);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_agent);
}

void NavigationServer3D::agent_set_max_speed(const RID &p_agent, float p_max_speed) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("agent_set_max_speed")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_max_speed_encoded;
	PtrToArg<double>::encode(p_max_speed, &p_max_speed_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_agent, &p_max_speed_encoded);
}

float NavigationServer3D::agent_get_max_speed(const RID &p_agent) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("agent_get_max_speed")._native_ptr(), 866169185);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_agent);
}

void NavigationServer3D::agent_set_velocity_forced(const RID &p_agent, const Vector3 &p_velocity) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("agent_set_velocity_forced")._native_ptr(), 3227306858);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_agent, &p_velocity);
}

void NavigationServer3D::agent_set_velocity(const RID &p_agent, const Vector3 &p_velocity) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("agent_set_velocity")._native_ptr(), 3227306858);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_agent, &p_velocity);
}

Vector3 NavigationServer3D::agent_get_velocity(const RID &p_agent) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("agent_get_velocity")._native_ptr(), 531438156);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner, &p_agent);
}

void NavigationServer3D::agent_set_position(const RID &p_agent, const Vector3 &p_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("agent_set_position")._native_ptr(), 3227306858);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_agent, &p_position);
}

Vector3 NavigationServer3D::agent_get_position(const RID &p_agent) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("agent_get_position")._native_ptr(), 531438156);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner, &p_agent);
}

bool NavigationServer3D::agent_is_map_changed(const RID &p_agent) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("agent_is_map_changed")._native_ptr(), 4155700596);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_agent);
}

void NavigationServer3D::agent_set_avoidance_callback(const RID &p_agent, const Callable &p_callback) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("agent_set_avoidance_callback")._native_ptr(), 3379118538);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_agent, &p_callback);
}

bool NavigationServer3D::agent_has_avoidance_callback(const RID &p_agent) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("agent_has_avoidance_callback")._native_ptr(), 4155700596);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_agent);
}

void NavigationServer3D::agent_set_avoidance_layers(const RID &p_agent, uint32_t p_layers) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("agent_set_avoidance_layers")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layers_encoded;
	PtrToArg<int64_t>::encode(p_layers, &p_layers_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_agent, &p_layers_encoded);
}

uint32_t NavigationServer3D::agent_get_avoidance_layers(const RID &p_agent) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("agent_get_avoidance_layers")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_agent);
}

void NavigationServer3D::agent_set_avoidance_mask(const RID &p_agent, uint32_t p_mask) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("agent_set_avoidance_mask")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mask_encoded;
	PtrToArg<int64_t>::encode(p_mask, &p_mask_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_agent, &p_mask_encoded);
}

uint32_t NavigationServer3D::agent_get_avoidance_mask(const RID &p_agent) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("agent_get_avoidance_mask")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_agent);
}

void NavigationServer3D::agent_set_avoidance_priority(const RID &p_agent, float p_priority) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("agent_set_avoidance_priority")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_priority_encoded;
	PtrToArg<double>::encode(p_priority, &p_priority_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_agent, &p_priority_encoded);
}

float NavigationServer3D::agent_get_avoidance_priority(const RID &p_agent) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("agent_get_avoidance_priority")._native_ptr(), 866169185);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_agent);
}

RID NavigationServer3D::obstacle_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("obstacle_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

void NavigationServer3D::obstacle_set_avoidance_enabled(const RID &p_obstacle, bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("obstacle_set_avoidance_enabled")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_obstacle, &p_enabled_encoded);
}

bool NavigationServer3D::obstacle_get_avoidance_enabled(const RID &p_obstacle) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("obstacle_get_avoidance_enabled")._native_ptr(), 4155700596);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_obstacle);
}

void NavigationServer3D::obstacle_set_use_3d_avoidance(const RID &p_obstacle, bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("obstacle_set_use_3d_avoidance")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_obstacle, &p_enabled_encoded);
}

bool NavigationServer3D::obstacle_get_use_3d_avoidance(const RID &p_obstacle) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("obstacle_get_use_3d_avoidance")._native_ptr(), 4155700596);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_obstacle);
}

void NavigationServer3D::obstacle_set_map(const RID &p_obstacle, const RID &p_map) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("obstacle_set_map")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_obstacle, &p_map);
}

RID NavigationServer3D::obstacle_get_map(const RID &p_obstacle) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("obstacle_get_map")._native_ptr(), 3814569979);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_obstacle);
}

void NavigationServer3D::obstacle_set_paused(const RID &p_obstacle, bool p_paused) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("obstacle_set_paused")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_paused_encoded;
	PtrToArg<bool>::encode(p_paused, &p_paused_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_obstacle, &p_paused_encoded);
}

bool NavigationServer3D::obstacle_get_paused(const RID &p_obstacle) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("obstacle_get_paused")._native_ptr(), 4155700596);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_obstacle);
}

void NavigationServer3D::obstacle_set_radius(const RID &p_obstacle, float p_radius) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("obstacle_set_radius")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_radius_encoded;
	PtrToArg<double>::encode(p_radius, &p_radius_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_obstacle, &p_radius_encoded);
}

float NavigationServer3D::obstacle_get_radius(const RID &p_obstacle) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("obstacle_get_radius")._native_ptr(), 866169185);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_obstacle);
}

void NavigationServer3D::obstacle_set_height(const RID &p_obstacle, float p_height) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("obstacle_set_height")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_height_encoded;
	PtrToArg<double>::encode(p_height, &p_height_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_obstacle, &p_height_encoded);
}

float NavigationServer3D::obstacle_get_height(const RID &p_obstacle) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("obstacle_get_height")._native_ptr(), 866169185);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_obstacle);
}

void NavigationServer3D::obstacle_set_velocity(const RID &p_obstacle, const Vector3 &p_velocity) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("obstacle_set_velocity")._native_ptr(), 3227306858);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_obstacle, &p_velocity);
}

Vector3 NavigationServer3D::obstacle_get_velocity(const RID &p_obstacle) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("obstacle_get_velocity")._native_ptr(), 531438156);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner, &p_obstacle);
}

void NavigationServer3D::obstacle_set_position(const RID &p_obstacle, const Vector3 &p_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("obstacle_set_position")._native_ptr(), 3227306858);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_obstacle, &p_position);
}

Vector3 NavigationServer3D::obstacle_get_position(const RID &p_obstacle) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("obstacle_get_position")._native_ptr(), 531438156);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner, &p_obstacle);
}

void NavigationServer3D::obstacle_set_vertices(const RID &p_obstacle, const PackedVector3Array &p_vertices) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("obstacle_set_vertices")._native_ptr(), 4030257846);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_obstacle, &p_vertices);
}

PackedVector3Array NavigationServer3D::obstacle_get_vertices(const RID &p_obstacle) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("obstacle_get_vertices")._native_ptr(), 808965560);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedVector3Array()));
	return ::godot::internal::_call_native_mb_ret<PackedVector3Array>(_gde_method_bind, _owner, &p_obstacle);
}

void NavigationServer3D::obstacle_set_avoidance_layers(const RID &p_obstacle, uint32_t p_layers) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("obstacle_set_avoidance_layers")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layers_encoded;
	PtrToArg<int64_t>::encode(p_layers, &p_layers_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_obstacle, &p_layers_encoded);
}

uint32_t NavigationServer3D::obstacle_get_avoidance_layers(const RID &p_obstacle) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("obstacle_get_avoidance_layers")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_obstacle);
}

void NavigationServer3D::parse_source_geometry_data(const Ref<NavigationMesh> &p_navigation_mesh, const Ref<NavigationMeshSourceGeometryData3D> &p_source_geometry_data, Node *p_root_node, const Callable &p_callback) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("parse_source_geometry_data")._native_ptr(), 3172802542);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_navigation_mesh != nullptr ? &p_navigation_mesh->_owner : nullptr), (p_source_geometry_data != nullptr ? &p_source_geometry_data->_owner : nullptr), (p_root_node != nullptr ? &p_root_node->_owner : nullptr), &p_callback);
}

void NavigationServer3D::bake_from_source_geometry_data(const Ref<NavigationMesh> &p_navigation_mesh, const Ref<NavigationMeshSourceGeometryData3D> &p_source_geometry_data, const Callable &p_callback) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("bake_from_source_geometry_data")._native_ptr(), 1286748856);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_navigation_mesh != nullptr ? &p_navigation_mesh->_owner : nullptr), (p_source_geometry_data != nullptr ? &p_source_geometry_data->_owner : nullptr), &p_callback);
}

void NavigationServer3D::bake_from_source_geometry_data_async(const Ref<NavigationMesh> &p_navigation_mesh, const Ref<NavigationMeshSourceGeometryData3D> &p_source_geometry_data, const Callable &p_callback) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("bake_from_source_geometry_data_async")._native_ptr(), 1286748856);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_navigation_mesh != nullptr ? &p_navigation_mesh->_owner : nullptr), (p_source_geometry_data != nullptr ? &p_source_geometry_data->_owner : nullptr), &p_callback);
}

bool NavigationServer3D::is_baking_navigation_mesh(const Ref<NavigationMesh> &p_navigation_mesh) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("is_baking_navigation_mesh")._native_ptr(), 3142026141);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, (p_navigation_mesh != nullptr ? &p_navigation_mesh->_owner : nullptr));
}

RID NavigationServer3D::source_geometry_parser_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("source_geometry_parser_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

void NavigationServer3D::source_geometry_parser_set_callback(const RID &p_parser, const Callable &p_callback) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("source_geometry_parser_set_callback")._native_ptr(), 3379118538);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_parser, &p_callback);
}

PackedVector3Array NavigationServer3D::simplify_path(const PackedVector3Array &p_path, float p_epsilon) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("simplify_path")._native_ptr(), 2344122170);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedVector3Array()));
	double p_epsilon_encoded;
	PtrToArg<double>::encode(p_epsilon, &p_epsilon_encoded);
	return ::godot::internal::_call_native_mb_ret<PackedVector3Array>(_gde_method_bind, _owner, &p_path, &p_epsilon_encoded);
}

void NavigationServer3D::free_rid(const RID &p_rid) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("free_rid")._native_ptr(), 2722037293);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_rid);
}

void NavigationServer3D::set_active(bool p_active) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("set_active")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_active_encoded;
	PtrToArg<bool>::encode(p_active, &p_active_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_active_encoded);
}

void NavigationServer3D::set_debug_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("set_debug_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool NavigationServer3D::get_debug_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("get_debug_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

int32_t NavigationServer3D::get_process_info(NavigationServer3D::ProcessInfo p_process_info) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer3D::get_class_static()._native_ptr(), StringName("get_process_info")._native_ptr(), 1938440894);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_process_info_encoded;
	PtrToArg<int64_t>::encode(p_process_info, &p_process_info_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_process_info_encoded);
}

} // namespace godot
