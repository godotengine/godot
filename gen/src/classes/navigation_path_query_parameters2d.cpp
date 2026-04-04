/**************************************************************************/
/*  navigation_path_query_parameters2d.cpp                                */
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

#include <godot_cpp/classes/navigation_path_query_parameters2d.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

namespace godot {

void NavigationPathQueryParameters2D::set_pathfinding_algorithm(NavigationPathQueryParameters2D::PathfindingAlgorithm p_pathfinding_algorithm) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationPathQueryParameters2D::get_class_static()._native_ptr(), StringName("set_pathfinding_algorithm")._native_ptr(), 2783519915);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_pathfinding_algorithm_encoded;
	PtrToArg<int64_t>::encode(p_pathfinding_algorithm, &p_pathfinding_algorithm_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_pathfinding_algorithm_encoded);
}

NavigationPathQueryParameters2D::PathfindingAlgorithm NavigationPathQueryParameters2D::get_pathfinding_algorithm() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationPathQueryParameters2D::get_class_static()._native_ptr(), StringName("get_pathfinding_algorithm")._native_ptr(), 3000421146);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (NavigationPathQueryParameters2D::PathfindingAlgorithm(0)));
	return (NavigationPathQueryParameters2D::PathfindingAlgorithm)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void NavigationPathQueryParameters2D::set_path_postprocessing(NavigationPathQueryParameters2D::PathPostProcessing p_path_postprocessing) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationPathQueryParameters2D::get_class_static()._native_ptr(), StringName("set_path_postprocessing")._native_ptr(), 2864409082);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_path_postprocessing_encoded;
	PtrToArg<int64_t>::encode(p_path_postprocessing, &p_path_postprocessing_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_path_postprocessing_encoded);
}

NavigationPathQueryParameters2D::PathPostProcessing NavigationPathQueryParameters2D::get_path_postprocessing() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationPathQueryParameters2D::get_class_static()._native_ptr(), StringName("get_path_postprocessing")._native_ptr(), 3798118993);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (NavigationPathQueryParameters2D::PathPostProcessing(0)));
	return (NavigationPathQueryParameters2D::PathPostProcessing)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void NavigationPathQueryParameters2D::set_map(const RID &p_map) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationPathQueryParameters2D::get_class_static()._native_ptr(), StringName("set_map")._native_ptr(), 2722037293);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_map);
}

RID NavigationPathQueryParameters2D::get_map() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationPathQueryParameters2D::get_class_static()._native_ptr(), StringName("get_map")._native_ptr(), 2944877500);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner);
}

void NavigationPathQueryParameters2D::set_start_position(const Vector2 &p_start_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationPathQueryParameters2D::get_class_static()._native_ptr(), StringName("set_start_position")._native_ptr(), 743155724);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_start_position);
}

Vector2 NavigationPathQueryParameters2D::get_start_position() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationPathQueryParameters2D::get_class_static()._native_ptr(), StringName("get_start_position")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

void NavigationPathQueryParameters2D::set_target_position(const Vector2 &p_target_position) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationPathQueryParameters2D::get_class_static()._native_ptr(), StringName("set_target_position")._native_ptr(), 743155724);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_target_position);
}

Vector2 NavigationPathQueryParameters2D::get_target_position() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationPathQueryParameters2D::get_class_static()._native_ptr(), StringName("get_target_position")._native_ptr(), 3341600327);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector2()));
	return ::godot::internal::_call_native_mb_ret<Vector2>(_gde_method_bind, _owner);
}

void NavigationPathQueryParameters2D::set_navigation_layers(uint32_t p_navigation_layers) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationPathQueryParameters2D::get_class_static()._native_ptr(), StringName("set_navigation_layers")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_navigation_layers_encoded;
	PtrToArg<int64_t>::encode(p_navigation_layers, &p_navigation_layers_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_navigation_layers_encoded);
}

uint32_t NavigationPathQueryParameters2D::get_navigation_layers() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationPathQueryParameters2D::get_class_static()._native_ptr(), StringName("get_navigation_layers")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void NavigationPathQueryParameters2D::set_metadata_flags(BitField<NavigationPathQueryParameters2D::PathMetadataFlags> p_flags) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationPathQueryParameters2D::get_class_static()._native_ptr(), StringName("set_metadata_flags")._native_ptr(), 24274129);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_flags);
}

BitField<NavigationPathQueryParameters2D::PathMetadataFlags> NavigationPathQueryParameters2D::get_metadata_flags() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationPathQueryParameters2D::get_class_static()._native_ptr(), StringName("get_metadata_flags")._native_ptr(), 488152976);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (BitField<NavigationPathQueryParameters2D::PathMetadataFlags>(0)));
	return (int64_t)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void NavigationPathQueryParameters2D::set_simplify_path(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationPathQueryParameters2D::get_class_static()._native_ptr(), StringName("set_simplify_path")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

bool NavigationPathQueryParameters2D::get_simplify_path() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationPathQueryParameters2D::get_class_static()._native_ptr(), StringName("get_simplify_path")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void NavigationPathQueryParameters2D::set_simplify_epsilon(float p_epsilon) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationPathQueryParameters2D::get_class_static()._native_ptr(), StringName("set_simplify_epsilon")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_epsilon_encoded;
	PtrToArg<double>::encode(p_epsilon, &p_epsilon_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_epsilon_encoded);
}

float NavigationPathQueryParameters2D::get_simplify_epsilon() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationPathQueryParameters2D::get_class_static()._native_ptr(), StringName("get_simplify_epsilon")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void NavigationPathQueryParameters2D::set_included_regions(const TypedArray<RID> &p_regions) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationPathQueryParameters2D::get_class_static()._native_ptr(), StringName("set_included_regions")._native_ptr(), 381264803);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_regions);
}

TypedArray<RID> NavigationPathQueryParameters2D::get_included_regions() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationPathQueryParameters2D::get_class_static()._native_ptr(), StringName("get_included_regions")._native_ptr(), 3995934104);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<RID>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<RID>>(_gde_method_bind, _owner);
}

void NavigationPathQueryParameters2D::set_excluded_regions(const TypedArray<RID> &p_regions) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationPathQueryParameters2D::get_class_static()._native_ptr(), StringName("set_excluded_regions")._native_ptr(), 381264803);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_regions);
}

TypedArray<RID> NavigationPathQueryParameters2D::get_excluded_regions() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationPathQueryParameters2D::get_class_static()._native_ptr(), StringName("get_excluded_regions")._native_ptr(), 3995934104);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (TypedArray<RID>()));
	return ::godot::internal::_call_native_mb_ret<TypedArray<RID>>(_gde_method_bind, _owner);
}

void NavigationPathQueryParameters2D::set_path_return_max_length(float p_length) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationPathQueryParameters2D::get_class_static()._native_ptr(), StringName("set_path_return_max_length")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_length_encoded;
	PtrToArg<double>::encode(p_length, &p_length_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_length_encoded);
}

float NavigationPathQueryParameters2D::get_path_return_max_length() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationPathQueryParameters2D::get_class_static()._native_ptr(), StringName("get_path_return_max_length")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void NavigationPathQueryParameters2D::set_path_return_max_radius(float p_radius) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationPathQueryParameters2D::get_class_static()._native_ptr(), StringName("set_path_return_max_radius")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_radius_encoded;
	PtrToArg<double>::encode(p_radius, &p_radius_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_radius_encoded);
}

float NavigationPathQueryParameters2D::get_path_return_max_radius() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationPathQueryParameters2D::get_class_static()._native_ptr(), StringName("get_path_return_max_radius")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void NavigationPathQueryParameters2D::set_path_search_max_polygons(int32_t p_max_polygons) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationPathQueryParameters2D::get_class_static()._native_ptr(), StringName("set_path_search_max_polygons")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_max_polygons_encoded;
	PtrToArg<int64_t>::encode(p_max_polygons, &p_max_polygons_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_max_polygons_encoded);
}

int32_t NavigationPathQueryParameters2D::get_path_search_max_polygons() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationPathQueryParameters2D::get_class_static()._native_ptr(), StringName("get_path_search_max_polygons")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void NavigationPathQueryParameters2D::set_path_search_max_distance(float p_distance) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationPathQueryParameters2D::get_class_static()._native_ptr(), StringName("set_path_search_max_distance")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_distance_encoded;
	PtrToArg<double>::encode(p_distance, &p_distance_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_distance_encoded);
}

float NavigationPathQueryParameters2D::get_path_search_max_distance() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationPathQueryParameters2D::get_class_static()._native_ptr(), StringName("get_path_search_max_distance")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

} // namespace godot
