/**************************************************************************/
/*  navigation_server2d.cpp                                               */
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

#include <godot_cpp/classes/navigation_server2d.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/navigation_mesh_source_geometry_data2d.hpp>
#include <godot_cpp/classes/navigation_path_query_parameters2d.hpp>
#include <godot_cpp/classes/navigation_path_query_result2d.hpp>
#include <godot_cpp/classes/navigation_polygon.hpp>
#include <godot_cpp/classes/node.hpp>

namespace godot {

NavigationServer2D *NavigationServer2D::singleton = nullptr;

NavigationServer2D *NavigationServer2D::get_singleton() {
	if (unlikely(singleton == nullptr)) {
		GDExtensionObjectPtr singleton_obj = ::godot::gdextension_interface::global_get_singleton(NavigationServer2D::get_class_static()._native_ptr());
#ifdef DEBUG_ENABLED
		ERR_FAIL_NULL_V(singleton_obj, nullptr);
#endif // DEBUG_ENABLED
		singleton = reinterpret_cast<NavigationServer2D *>(::godot::gdextension_interface::object_get_instance_binding(singleton_obj, ::godot::gdextension_interface::token, &NavigationServer2D::_gde_binding_callbacks));
#ifdef DEBUG_ENABLED
		ERR_FAIL_NULL_V(singleton, nullptr);
#endif // DEBUG_ENABLED
		if (likely(singleton)) {
			ClassDB::_register_engine_singleton(NavigationServer2D::get_class_static(), singleton);
		}
	}
	return singleton;
}

NavigationServer2D::~NavigationServer2D() {
	if (singleton == this) {
		ClassDB::_unregister_engine_singleton(NavigationServer2D::get_class_static());
		singleton = nullptr;
	}
}

TypedArray<RID> NavigationServer2D::get_maps() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("get_maps")._native_ptr(), 3995934104);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<RID>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<RID>>(_gde_method_bind, _owner);
}

RID NavigationServer2D::map_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("map_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

void NavigationServer2D::map_set_active(const RID &p_map, bool p_active) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("map_set_active")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_active_encoded;
	PtrToArg<bool>::encode(p_active, &p_active_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_map, &p_active_encoded);
}

bool NavigationServer2D::map_is_active(const RID &p_map) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("map_is_active")._native_ptr(), 4155700596);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_map);
}

void NavigationServer2D::map_set_cell_size(const RID &p_map, float p_cell_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("map_set_cell_size")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_cell_size_encoded;
	PtrToArg<double>::encode(p_cell_size, &p_cell_size_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_map, &p_cell_size_encoded);
}

float NavigationServer2D::map_get_cell_size(const RID &p_map) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("map_get_cell_size")._native_ptr(), 866169185);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_map);
}

void NavigationServer2D::map_set_merge_rasterizer_cell_scale(const RID &p_map, float p_scale) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("map_set_merge_rasterizer_cell_scale")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_scale_encoded;
	PtrToArg<double>::encode(p_scale, &p_scale_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_map, &p_scale_encoded);
}

float NavigationServer2D::map_get_merge_rasterizer_cell_scale(const RID &p_map) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("map_get_merge_rasterizer_cell_scale")._native_ptr(), 866169185);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_map);
}

void NavigationServer2D::map_set_use_edge_connections(const RID &p_map, bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("map_set_use_edge_connections")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_map, &p_enabled_encoded);
}

bool NavigationServer2D::map_get_use_edge_connections(const RID &p_map) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("map_get_use_edge_connections")._native_ptr(), 4155700596);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_map);
}

void NavigationServer2D::map_set_edge_connection_margin(const RID &p_map, float p_margin) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("map_set_edge_connection_margin")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_margin_encoded;
	PtrToArg<double>::encode(p_margin, &p_margin_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_map, &p_margin_encoded);
}

float NavigationServer2D::map_get_edge_connection_margin(const RID &p_map) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("map_get_edge_connection_margin")._native_ptr(), 866169185);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_map);
}

void NavigationServer2D::map_set_link_connection_radius(const RID &p_map, float p_radius) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("map_set_link_connection_radius")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_radius_encoded;
	PtrToArg<double>::encode(p_radius, &p_radius_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_map, &p_radius_encoded);
}

float NavigationServer2D::map_get_link_connection_radius(const RID &p_map) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("map_get_link_connection_radius")._native_ptr(), 866169185);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_map);
}

PackedVector2Array NavigationServer2D::map_get_path(const RID &p_map, const Vector2 &p_origin, const Vector2 &p_destination, bool p_optimize, uint32_t p_navigation_layers) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("map_get_path")._native_ptr(), 1279824844);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedVector2Array()));
	int8_t p_optimize_encoded;
	PtrToArg<bool>::encode(p_optimize, &p_optimize_encoded);
	int64_t p_navigation_layers_encoded;
	PtrToArg<int64_t>::encode(p_navigation_layers, &p_navigation_layers_encoded);
	return ::godot::internal::_call_native_mb_ret<PackedVector2Array>(_gde_method_bind, _owner, &p_map, &p_origin, &p_destination, &p_optimize_encoded, &p_navigation_layers_encoded);
}

Vector2 NavigationServer2D::map_get_closest_point(const RID &p_map, const Vector2 &p_to_point) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("map_get_closest_point")._native_ptr(), 1358334418);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner, &p_map, &p_to_point);
}

RID NavigationServer2D::map_get_closest_point_owner(const RID &p_map, const Vector2 &p_to_point) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("map_get_closest_point_owner")._native_ptr(), 1353467510);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_map, &p_to_point);
}

TypedArray<RID> NavigationServer2D::map_get_links(const RID &p_map) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("map_get_links")._native_ptr(), 2684255073);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<RID>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<RID>>(_gde_method_bind, _owner, &p_map);
}

TypedArray<RID> NavigationServer2D::map_get_regions(const RID &p_map) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("map_get_regions")._native_ptr(), 2684255073);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<RID>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<RID>>(_gde_method_bind, _owner, &p_map);
}

TypedArray<RID> NavigationServer2D::map_get_agents(const RID &p_map) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("map_get_agents")._native_ptr(), 2684255073);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<RID>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<RID>>(_gde_method_bind, _owner, &p_map);
}

TypedArray<RID> NavigationServer2D::map_get_obstacles(const RID &p_map) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("map_get_obstacles")._native_ptr(), 2684255073);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<RID>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<RID>>(_gde_method_bind, _owner, &p_map);
}

void NavigationServer2D::map_force_update(const RID &p_map) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("map_force_update")._native_ptr(), 2722037293);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_map);
}

uint32_t NavigationServer2D::map_get_iteration_id(const RID &p_map) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("map_get_iteration_id")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_map);
}

void NavigationServer2D::map_set_use_async_iterations(const RID &p_map, bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("map_set_use_async_iterations")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_map, &p_enabled_encoded);
}

bool NavigationServer2D::map_get_use_async_iterations(const RID &p_map) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("map_get_use_async_iterations")._native_ptr(), 4155700596);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_map);
}

Vector2 NavigationServer2D::map_get_random_point(const RID &p_map, uint32_t p_navigation_layers, bool p_uniformly) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("map_get_random_point")._native_ptr(), 3271000763);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	int64_t p_navigation_layers_encoded;
	PtrToArg<int64_t>::encode(p_navigation_layers, &p_navigation_layers_encoded);
	int8_t p_uniformly_encoded;
	PtrToArg<bool>::encode(p_uniformly, &p_uniformly_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner, &p_map, &p_navigation_layers_encoded, &p_uniformly_encoded);
}

void NavigationServer2D::query_path(const Ref<NavigationPathQueryParameters2D> &p_parameters, const Ref<NavigationPathQueryResult2D> &p_result, const Callable &p_callback) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("query_path")._native_ptr(), 1254915886);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_parameters != nullptr ? &p_parameters->_owner : nullptr), (p_result != nullptr ? &p_result->_owner : nullptr), &p_callback);
}

RID NavigationServer2D::region_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("region_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

uint32_t NavigationServer2D::region_get_iteration_id(const RID &p_region) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("region_get_iteration_id")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_region);
}

void NavigationServer2D::region_set_use_async_iterations(const RID &p_region, bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("region_set_use_async_iterations")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_region, &p_enabled_encoded);
}

bool NavigationServer2D::region_get_use_async_iterations(const RID &p_region) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("region_get_use_async_iterations")._native_ptr(), 4155700596);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_region);
}

void NavigationServer2D::region_set_enabled(const RID &p_region, bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("region_set_enabled")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_region, &p_enabled_encoded);
}

bool NavigationServer2D::region_get_enabled(const RID &p_region) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("region_get_enabled")._native_ptr(), 4155700596);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_region);
}

void NavigationServer2D::region_set_use_edge_connections(const RID &p_region, bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("region_set_use_edge_connections")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_region, &p_enabled_encoded);
}

bool NavigationServer2D::region_get_use_edge_connections(const RID &p_region) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("region_get_use_edge_connections")._native_ptr(), 4155700596);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_region);
}

void NavigationServer2D::region_set_enter_cost(const RID &p_region, float p_enter_cost) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("region_set_enter_cost")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_enter_cost_encoded;
	PtrToArg<double>::encode(p_enter_cost, &p_enter_cost_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_region, &p_enter_cost_encoded);
}

float NavigationServer2D::region_get_enter_cost(const RID &p_region) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("region_get_enter_cost")._native_ptr(), 866169185);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_region);
}

void NavigationServer2D::region_set_travel_cost(const RID &p_region, float p_travel_cost) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("region_set_travel_cost")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_travel_cost_encoded;
	PtrToArg<double>::encode(p_travel_cost, &p_travel_cost_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_region, &p_travel_cost_encoded);
}

float NavigationServer2D::region_get_travel_cost(const RID &p_region) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("region_get_travel_cost")._native_ptr(), 866169185);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_region);
}

void NavigationServer2D::region_set_owner_id(const RID &p_region, uint64_t p_owner_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("region_set_owner_id")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_owner_id_encoded;
	PtrToArg<int64_t>::encode(p_owner_id, &p_owner_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_region, &p_owner_id_encoded);
}

uint64_t NavigationServer2D::region_get_owner_id(const RID &p_region) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("region_get_owner_id")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<uint64_t>(_gde_method_bind, _owner, &p_region);
}

bool NavigationServer2D::region_owns_point(const RID &p_region, const Vector2 &p_point) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("region_owns_point")._native_ptr(), 219849798);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_region, &p_point);
}

void NavigationServer2D::region_set_map(const RID &p_region, const RID &p_map) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("region_set_map")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_region, &p_map);
}

RID NavigationServer2D::region_get_map(const RID &p_region) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("region_get_map")._native_ptr(), 3814569979);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_region);
}

void NavigationServer2D::region_set_navigation_layers(const RID &p_region, uint32_t p_navigation_layers) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("region_set_navigation_layers")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_navigation_layers_encoded;
	PtrToArg<int64_t>::encode(p_navigation_layers, &p_navigation_layers_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_region, &p_navigation_layers_encoded);
}

uint32_t NavigationServer2D::region_get_navigation_layers(const RID &p_region) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("region_get_navigation_layers")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_region);
}

void NavigationServer2D::region_set_transform(const RID &p_region, const Transform2D &p_transform) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("region_set_transform")._native_ptr(), 1246044741);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_region, &p_transform);
}

Transform2D NavigationServer2D::region_get_transform(const RID &p_region) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("region_get_transform")._native_ptr(), 213527486);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Transform2D()));
	return ::godot::internal::_call_native_mb_ret<Transform2D>(_gde_method_bind, _owner, &p_region);
}

void NavigationServer2D::region_set_navigation_polygon(const RID &p_region, const Ref<NavigationPolygon> &p_navigation_polygon) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("region_set_navigation_polygon")._native_ptr(), 3633623451);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_region, (p_navigation_polygon != nullptr ? &p_navigation_polygon->_owner : nullptr));
}

int32_t NavigationServer2D::region_get_connections_count(const RID &p_region) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("region_get_connections_count")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_region);
}

Vector2 NavigationServer2D::region_get_connection_pathway_start(const RID &p_region, int32_t p_connection) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("region_get_connection_pathway_start")._native_ptr(), 2546185844);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	int64_t p_connection_encoded;
	PtrToArg<int64_t>::encode(p_connection, &p_connection_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner, &p_region, &p_connection_encoded);
}

Vector2 NavigationServer2D::region_get_connection_pathway_end(const RID &p_region, int32_t p_connection) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("region_get_connection_pathway_end")._native_ptr(), 2546185844);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	int64_t p_connection_encoded;
	PtrToArg<int64_t>::encode(p_connection, &p_connection_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner, &p_region, &p_connection_encoded);
}

Vector2 NavigationServer2D::region_get_closest_point(const RID &p_region, const Vector2 &p_to_point) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("region_get_closest_point")._native_ptr(), 1358334418);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner, &p_region, &p_to_point);
}

Vector2 NavigationServer2D::region_get_random_point(const RID &p_region, uint32_t p_navigation_layers, bool p_uniformly) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("region_get_random_point")._native_ptr(), 3271000763);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	int64_t p_navigation_layers_encoded;
	PtrToArg<int64_t>::encode(p_navigation_layers, &p_navigation_layers_encoded);
	int8_t p_uniformly_encoded;
	PtrToArg<bool>::encode(p_uniformly, &p_uniformly_encoded);
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner, &p_region, &p_navigation_layers_encoded, &p_uniformly_encoded);
}

Rect2 NavigationServer2D::region_get_bounds(const RID &p_region) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("region_get_bounds")._native_ptr(), 1097232729);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Rect2()));
	return ::godot::internal::_call_native_mb_ret<Rect2>(_gde_method_bind, _owner, &p_region);
}

RID NavigationServer2D::link_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("link_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

uint32_t NavigationServer2D::link_get_iteration_id(const RID &p_link) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("link_get_iteration_id")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_link);
}

void NavigationServer2D::link_set_map(const RID &p_link, const RID &p_map) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("link_set_map")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_link, &p_map);
}

RID NavigationServer2D::link_get_map(const RID &p_link) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("link_get_map")._native_ptr(), 3814569979);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_link);
}

void NavigationServer2D::link_set_enabled(const RID &p_link, bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("link_set_enabled")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_link, &p_enabled_encoded);
}

bool NavigationServer2D::link_get_enabled(const RID &p_link) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("link_get_enabled")._native_ptr(), 4155700596);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_link);
}

void NavigationServer2D::link_set_bidirectional(const RID &p_link, bool p_bidirectional) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("link_set_bidirectional")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_bidirectional_encoded;
	PtrToArg<bool>::encode(p_bidirectional, &p_bidirectional_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_link, &p_bidirectional_encoded);
}

bool NavigationServer2D::link_is_bidirectional(const RID &p_link) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("link_is_bidirectional")._native_ptr(), 4155700596);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_link);
}

void NavigationServer2D::link_set_navigation_layers(const RID &p_link, uint32_t p_navigation_layers) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("link_set_navigation_layers")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_navigation_layers_encoded;
	PtrToArg<int64_t>::encode(p_navigation_layers, &p_navigation_layers_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_link, &p_navigation_layers_encoded);
}

uint32_t NavigationServer2D::link_get_navigation_layers(const RID &p_link) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("link_get_navigation_layers")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_link);
}

void NavigationServer2D::link_set_start_position(const RID &p_link, const Vector2 &p_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("link_set_start_position")._native_ptr(), 3201125042);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_link, &p_position);
}

Vector2 NavigationServer2D::link_get_start_position(const RID &p_link) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("link_get_start_position")._native_ptr(), 2440833711);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner, &p_link);
}

void NavigationServer2D::link_set_end_position(const RID &p_link, const Vector2 &p_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("link_set_end_position")._native_ptr(), 3201125042);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_link, &p_position);
}

Vector2 NavigationServer2D::link_get_end_position(const RID &p_link) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("link_get_end_position")._native_ptr(), 2440833711);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner, &p_link);
}

void NavigationServer2D::link_set_enter_cost(const RID &p_link, float p_enter_cost) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("link_set_enter_cost")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_enter_cost_encoded;
	PtrToArg<double>::encode(p_enter_cost, &p_enter_cost_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_link, &p_enter_cost_encoded);
}

float NavigationServer2D::link_get_enter_cost(const RID &p_link) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("link_get_enter_cost")._native_ptr(), 866169185);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_link);
}

void NavigationServer2D::link_set_travel_cost(const RID &p_link, float p_travel_cost) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("link_set_travel_cost")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_travel_cost_encoded;
	PtrToArg<double>::encode(p_travel_cost, &p_travel_cost_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_link, &p_travel_cost_encoded);
}

float NavigationServer2D::link_get_travel_cost(const RID &p_link) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("link_get_travel_cost")._native_ptr(), 866169185);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_link);
}

void NavigationServer2D::link_set_owner_id(const RID &p_link, uint64_t p_owner_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("link_set_owner_id")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_owner_id_encoded;
	PtrToArg<int64_t>::encode(p_owner_id, &p_owner_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_link, &p_owner_id_encoded);
}

uint64_t NavigationServer2D::link_get_owner_id(const RID &p_link) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("link_get_owner_id")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<uint64_t>(_gde_method_bind, _owner, &p_link);
}

RID NavigationServer2D::agent_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("agent_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

void NavigationServer2D::agent_set_avoidance_enabled(const RID &p_agent, bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("agent_set_avoidance_enabled")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_agent, &p_enabled_encoded);
}

bool NavigationServer2D::agent_get_avoidance_enabled(const RID &p_agent) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("agent_get_avoidance_enabled")._native_ptr(), 4155700596);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_agent);
}

void NavigationServer2D::agent_set_map(const RID &p_agent, const RID &p_map) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("agent_set_map")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_agent, &p_map);
}

RID NavigationServer2D::agent_get_map(const RID &p_agent) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("agent_get_map")._native_ptr(), 3814569979);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_agent);
}

void NavigationServer2D::agent_set_paused(const RID &p_agent, bool p_paused) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("agent_set_paused")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_paused_encoded;
	PtrToArg<bool>::encode(p_paused, &p_paused_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_agent, &p_paused_encoded);
}

bool NavigationServer2D::agent_get_paused(const RID &p_agent) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("agent_get_paused")._native_ptr(), 4155700596);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_agent);
}

void NavigationServer2D::agent_set_neighbor_distance(const RID &p_agent, float p_distance) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("agent_set_neighbor_distance")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_distance_encoded;
	PtrToArg<double>::encode(p_distance, &p_distance_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_agent, &p_distance_encoded);
}

float NavigationServer2D::agent_get_neighbor_distance(const RID &p_agent) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("agent_get_neighbor_distance")._native_ptr(), 866169185);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_agent);
}

void NavigationServer2D::agent_set_max_neighbors(const RID &p_agent, int32_t p_count) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("agent_set_max_neighbors")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_count_encoded;
	PtrToArg<int64_t>::encode(p_count, &p_count_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_agent, &p_count_encoded);
}

int32_t NavigationServer2D::agent_get_max_neighbors(const RID &p_agent) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("agent_get_max_neighbors")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_agent);
}

void NavigationServer2D::agent_set_time_horizon_agents(const RID &p_agent, float p_time_horizon) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("agent_set_time_horizon_agents")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_time_horizon_encoded;
	PtrToArg<double>::encode(p_time_horizon, &p_time_horizon_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_agent, &p_time_horizon_encoded);
}

float NavigationServer2D::agent_get_time_horizon_agents(const RID &p_agent) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("agent_get_time_horizon_agents")._native_ptr(), 866169185);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_agent);
}

void NavigationServer2D::agent_set_time_horizon_obstacles(const RID &p_agent, float p_time_horizon) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("agent_set_time_horizon_obstacles")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_time_horizon_encoded;
	PtrToArg<double>::encode(p_time_horizon, &p_time_horizon_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_agent, &p_time_horizon_encoded);
}

float NavigationServer2D::agent_get_time_horizon_obstacles(const RID &p_agent) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("agent_get_time_horizon_obstacles")._native_ptr(), 866169185);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_agent);
}

void NavigationServer2D::agent_set_radius(const RID &p_agent, float p_radius) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("agent_set_radius")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_radius_encoded;
	PtrToArg<double>::encode(p_radius, &p_radius_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_agent, &p_radius_encoded);
}

float NavigationServer2D::agent_get_radius(const RID &p_agent) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("agent_get_radius")._native_ptr(), 866169185);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_agent);
}

void NavigationServer2D::agent_set_max_speed(const RID &p_agent, float p_max_speed) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("agent_set_max_speed")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_max_speed_encoded;
	PtrToArg<double>::encode(p_max_speed, &p_max_speed_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_agent, &p_max_speed_encoded);
}

float NavigationServer2D::agent_get_max_speed(const RID &p_agent) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("agent_get_max_speed")._native_ptr(), 866169185);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_agent);
}

void NavigationServer2D::agent_set_velocity_forced(const RID &p_agent, const Vector2 &p_velocity) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("agent_set_velocity_forced")._native_ptr(), 3201125042);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_agent, &p_velocity);
}

void NavigationServer2D::agent_set_velocity(const RID &p_agent, const Vector2 &p_velocity) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("agent_set_velocity")._native_ptr(), 3201125042);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_agent, &p_velocity);
}

Vector2 NavigationServer2D::agent_get_velocity(const RID &p_agent) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("agent_get_velocity")._native_ptr(), 2440833711);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner, &p_agent);
}

void NavigationServer2D::agent_set_position(const RID &p_agent, const Vector2 &p_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("agent_set_position")._native_ptr(), 3201125042);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_agent, &p_position);
}

Vector2 NavigationServer2D::agent_get_position(const RID &p_agent) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("agent_get_position")._native_ptr(), 2440833711);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner, &p_agent);
}

bool NavigationServer2D::agent_is_map_changed(const RID &p_agent) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("agent_is_map_changed")._native_ptr(), 4155700596);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_agent);
}

void NavigationServer2D::agent_set_avoidance_callback(const RID &p_agent, const Callable &p_callback) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("agent_set_avoidance_callback")._native_ptr(), 3379118538);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_agent, &p_callback);
}

bool NavigationServer2D::agent_has_avoidance_callback(const RID &p_agent) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("agent_has_avoidance_callback")._native_ptr(), 4155700596);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_agent);
}

void NavigationServer2D::agent_set_avoidance_layers(const RID &p_agent, uint32_t p_layers) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("agent_set_avoidance_layers")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layers_encoded;
	PtrToArg<int64_t>::encode(p_layers, &p_layers_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_agent, &p_layers_encoded);
}

uint32_t NavigationServer2D::agent_get_avoidance_layers(const RID &p_agent) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("agent_get_avoidance_layers")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_agent);
}

void NavigationServer2D::agent_set_avoidance_mask(const RID &p_agent, uint32_t p_mask) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("agent_set_avoidance_mask")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mask_encoded;
	PtrToArg<int64_t>::encode(p_mask, &p_mask_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_agent, &p_mask_encoded);
}

uint32_t NavigationServer2D::agent_get_avoidance_mask(const RID &p_agent) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("agent_get_avoidance_mask")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_agent);
}

void NavigationServer2D::agent_set_avoidance_priority(const RID &p_agent, float p_priority) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("agent_set_avoidance_priority")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_priority_encoded;
	PtrToArg<double>::encode(p_priority, &p_priority_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_agent, &p_priority_encoded);
}

float NavigationServer2D::agent_get_avoidance_priority(const RID &p_agent) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("agent_get_avoidance_priority")._native_ptr(), 866169185);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_agent);
}

RID NavigationServer2D::obstacle_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("obstacle_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

void NavigationServer2D::obstacle_set_avoidance_enabled(const RID &p_obstacle, bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("obstacle_set_avoidance_enabled")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_obstacle, &p_enabled_encoded);
}

bool NavigationServer2D::obstacle_get_avoidance_enabled(const RID &p_obstacle) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("obstacle_get_avoidance_enabled")._native_ptr(), 4155700596);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_obstacle);
}

void NavigationServer2D::obstacle_set_map(const RID &p_obstacle, const RID &p_map) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("obstacle_set_map")._native_ptr(), 395945892);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_obstacle, &p_map);
}

RID NavigationServer2D::obstacle_get_map(const RID &p_obstacle) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("obstacle_get_map")._native_ptr(), 3814569979);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_obstacle);
}

void NavigationServer2D::obstacle_set_paused(const RID &p_obstacle, bool p_paused) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("obstacle_set_paused")._native_ptr(), 1265174801);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_paused_encoded;
	PtrToArg<bool>::encode(p_paused, &p_paused_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_obstacle, &p_paused_encoded);
}

bool NavigationServer2D::obstacle_get_paused(const RID &p_obstacle) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("obstacle_get_paused")._native_ptr(), 4155700596);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_obstacle);
}

void NavigationServer2D::obstacle_set_radius(const RID &p_obstacle, float p_radius) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("obstacle_set_radius")._native_ptr(), 1794382983);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_radius_encoded;
	PtrToArg<double>::encode(p_radius, &p_radius_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_obstacle, &p_radius_encoded);
}

float NavigationServer2D::obstacle_get_radius(const RID &p_obstacle) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("obstacle_get_radius")._native_ptr(), 866169185);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_obstacle);
}

void NavigationServer2D::obstacle_set_velocity(const RID &p_obstacle, const Vector2 &p_velocity) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("obstacle_set_velocity")._native_ptr(), 3201125042);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_obstacle, &p_velocity);
}

Vector2 NavigationServer2D::obstacle_get_velocity(const RID &p_obstacle) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("obstacle_get_velocity")._native_ptr(), 2440833711);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner, &p_obstacle);
}

void NavigationServer2D::obstacle_set_position(const RID &p_obstacle, const Vector2 &p_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("obstacle_set_position")._native_ptr(), 3201125042);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_obstacle, &p_position);
}

Vector2 NavigationServer2D::obstacle_get_position(const RID &p_obstacle) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("obstacle_get_position")._native_ptr(), 2440833711);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner, &p_obstacle);
}

void NavigationServer2D::obstacle_set_vertices(const RID &p_obstacle, const PackedVector2Array &p_vertices) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("obstacle_set_vertices")._native_ptr(), 29476483);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_obstacle, &p_vertices);
}

PackedVector2Array NavigationServer2D::obstacle_get_vertices(const RID &p_obstacle) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("obstacle_get_vertices")._native_ptr(), 2222557395);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedVector2Array()));
	return ::godot::internal::_call_native_mb_ret<PackedVector2Array>(_gde_method_bind, _owner, &p_obstacle);
}

void NavigationServer2D::obstacle_set_avoidance_layers(const RID &p_obstacle, uint32_t p_layers) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("obstacle_set_avoidance_layers")._native_ptr(), 3411492887);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layers_encoded;
	PtrToArg<int64_t>::encode(p_layers, &p_layers_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_obstacle, &p_layers_encoded);
}

uint32_t NavigationServer2D::obstacle_get_avoidance_layers(const RID &p_obstacle) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("obstacle_get_avoidance_layers")._native_ptr(), 2198884583);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_obstacle);
}

void NavigationServer2D::parse_source_geometry_data(const Ref<NavigationPolygon> &p_navigation_polygon, const Ref<NavigationMeshSourceGeometryData2D> &p_source_geometry_data, Node *p_root_node, const Callable &p_callback) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("parse_source_geometry_data")._native_ptr(), 1766905497);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_navigation_polygon != nullptr ? &p_navigation_polygon->_owner : nullptr), (p_source_geometry_data != nullptr ? &p_source_geometry_data->_owner : nullptr), (p_root_node != nullptr ? &p_root_node->_owner : nullptr), &p_callback);
}

void NavigationServer2D::bake_from_source_geometry_data(const Ref<NavigationPolygon> &p_navigation_polygon, const Ref<NavigationMeshSourceGeometryData2D> &p_source_geometry_data, const Callable &p_callback) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("bake_from_source_geometry_data")._native_ptr(), 2179660022);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_navigation_polygon != nullptr ? &p_navigation_polygon->_owner : nullptr), (p_source_geometry_data != nullptr ? &p_source_geometry_data->_owner : nullptr), &p_callback);
}

void NavigationServer2D::bake_from_source_geometry_data_async(const Ref<NavigationPolygon> &p_navigation_polygon, const Ref<NavigationMeshSourceGeometryData2D> &p_source_geometry_data, const Callable &p_callback) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("bake_from_source_geometry_data_async")._native_ptr(), 2179660022);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_navigation_polygon != nullptr ? &p_navigation_polygon->_owner : nullptr), (p_source_geometry_data != nullptr ? &p_source_geometry_data->_owner : nullptr), &p_callback);
}

bool NavigationServer2D::is_baking_navigation_polygon(const Ref<NavigationPolygon> &p_navigation_polygon) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("is_baking_navigation_polygon")._native_ptr(), 3729405808);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, (p_navigation_polygon != nullptr ? &p_navigation_polygon->_owner : nullptr));
}

RID NavigationServer2D::source_geometry_parser_create() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("source_geometry_parser_create")._native_ptr(), 529393457);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

void NavigationServer2D::source_geometry_parser_set_callback(const RID &p_parser, const Callable &p_callback) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("source_geometry_parser_set_callback")._native_ptr(), 3379118538);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_parser, &p_callback);
}

PackedVector2Array NavigationServer2D::simplify_path(const PackedVector2Array &p_path, float p_epsilon) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("simplify_path")._native_ptr(), 2457191505);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedVector2Array()));
	double p_epsilon_encoded;
	PtrToArg<double>::encode(p_epsilon, &p_epsilon_encoded);
	return ::godot::internal::_call_native_mb_ret<PackedVector2Array>(_gde_method_bind, _owner, &p_path, &p_epsilon_encoded);
}

void NavigationServer2D::free_rid(const RID &p_rid) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("free_rid")._native_ptr(), 2722037293);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_rid);
}

void NavigationServer2D::set_active(bool p_active) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("set_active")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_active_encoded;
	PtrToArg<bool>::encode(p_active, &p_active_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_active_encoded);
}

void NavigationServer2D::set_debug_enabled(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("set_debug_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool NavigationServer2D::get_debug_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("get_debug_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

int32_t NavigationServer2D::get_process_info(NavigationServer2D::ProcessInfo p_process_info) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationServer2D::get_class_static()._native_ptr(), StringName("get_process_info")._native_ptr(), 1640219858);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_process_info_encoded;
	PtrToArg<int64_t>::encode(p_process_info, &p_process_info_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_process_info_encoded);
}

} // namespace godot
