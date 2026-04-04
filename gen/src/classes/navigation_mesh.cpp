/**************************************************************************/
/*  navigation_mesh.cpp                                                   */
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

#include <godot_cpp/classes/navigation_mesh.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/mesh.hpp>

namespace godot {

void NavigationMesh::set_sample_partition_type(NavigationMesh::SamplePartitionType p_sample_partition_type) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMesh::get_class_static()._native_ptr(), StringName("set_sample_partition_type")._native_ptr(), 2472437533);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_sample_partition_type_encoded;
	PtrToArg<int64_t>::encode(p_sample_partition_type, &p_sample_partition_type_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_sample_partition_type_encoded);
}

NavigationMesh::SamplePartitionType NavigationMesh::get_sample_partition_type() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMesh::get_class_static()._native_ptr(), StringName("get_sample_partition_type")._native_ptr(), 833513918);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (NavigationMesh::SamplePartitionType(0)));
	return (NavigationMesh::SamplePartitionType)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void NavigationMesh::set_parsed_geometry_type(NavigationMesh::ParsedGeometryType p_geometry_type) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMesh::get_class_static()._native_ptr(), StringName("set_parsed_geometry_type")._native_ptr(), 3064713163);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_geometry_type_encoded;
	PtrToArg<int64_t>::encode(p_geometry_type, &p_geometry_type_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_geometry_type_encoded);
}

NavigationMesh::ParsedGeometryType NavigationMesh::get_parsed_geometry_type() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMesh::get_class_static()._native_ptr(), StringName("get_parsed_geometry_type")._native_ptr(), 3928011953);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (NavigationMesh::ParsedGeometryType(0)));
	return (NavigationMesh::ParsedGeometryType)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void NavigationMesh::set_collision_mask(uint32_t p_mask) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMesh::get_class_static()._native_ptr(), StringName("set_collision_mask")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mask_encoded;
	PtrToArg<int64_t>::encode(p_mask, &p_mask_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mask_encoded);
}

uint32_t NavigationMesh::get_collision_mask() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMesh::get_class_static()._native_ptr(), StringName("get_collision_mask")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void NavigationMesh::set_collision_mask_value(int32_t p_layer_number, bool p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMesh::get_class_static()._native_ptr(), StringName("set_collision_mask_value")._native_ptr(), 300928843);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_layer_number_encoded;
	PtrToArg<int64_t>::encode(p_layer_number, &p_layer_number_encoded);
	int8_t p_value_encoded;
	PtrToArg<bool>::encode(p_value, &p_value_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_layer_number_encoded, &p_value_encoded);
}

bool NavigationMesh::get_collision_mask_value(int32_t p_layer_number) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMesh::get_class_static()._native_ptr(), StringName("get_collision_mask_value")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_layer_number_encoded;
	PtrToArg<int64_t>::encode(p_layer_number, &p_layer_number_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_layer_number_encoded);
}

void NavigationMesh::set_source_geometry_mode(NavigationMesh::SourceGeometryMode p_mask) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMesh::get_class_static()._native_ptr(), StringName("set_source_geometry_mode")._native_ptr(), 2700825194);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_mask_encoded;
	PtrToArg<int64_t>::encode(p_mask, &p_mask_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mask_encoded);
}

NavigationMesh::SourceGeometryMode NavigationMesh::get_source_geometry_mode() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMesh::get_class_static()._native_ptr(), StringName("get_source_geometry_mode")._native_ptr(), 2770484141);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (NavigationMesh::SourceGeometryMode(0)));
	return (NavigationMesh::SourceGeometryMode)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void NavigationMesh::set_source_group_name(const StringName &p_mask) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMesh::get_class_static()._native_ptr(), StringName("set_source_group_name")._native_ptr(), 3304788590);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_mask);
}

StringName NavigationMesh::get_source_group_name() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMesh::get_class_static()._native_ptr(), StringName("get_source_group_name")._native_ptr(), 2002593661);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (StringName()));
	return ::godot::internal::_call_native_mb_ret<StringName>(_gde_method_bind, _owner);
}

void NavigationMesh::set_cell_size(float p_cell_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMesh::get_class_static()._native_ptr(), StringName("set_cell_size")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_cell_size_encoded;
	PtrToArg<double>::encode(p_cell_size, &p_cell_size_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_cell_size_encoded);
}

float NavigationMesh::get_cell_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMesh::get_class_static()._native_ptr(), StringName("get_cell_size")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void NavigationMesh::set_cell_height(float p_cell_height) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMesh::get_class_static()._native_ptr(), StringName("set_cell_height")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_cell_height_encoded;
	PtrToArg<double>::encode(p_cell_height, &p_cell_height_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_cell_height_encoded);
}

float NavigationMesh::get_cell_height() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMesh::get_class_static()._native_ptr(), StringName("get_cell_height")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void NavigationMesh::set_border_size(float p_border_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMesh::get_class_static()._native_ptr(), StringName("set_border_size")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_border_size_encoded;
	PtrToArg<double>::encode(p_border_size, &p_border_size_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_border_size_encoded);
}

float NavigationMesh::get_border_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMesh::get_class_static()._native_ptr(), StringName("get_border_size")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void NavigationMesh::set_agent_height(float p_agent_height) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMesh::get_class_static()._native_ptr(), StringName("set_agent_height")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_agent_height_encoded;
	PtrToArg<double>::encode(p_agent_height, &p_agent_height_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_agent_height_encoded);
}

float NavigationMesh::get_agent_height() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMesh::get_class_static()._native_ptr(), StringName("get_agent_height")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void NavigationMesh::set_agent_radius(float p_agent_radius) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMesh::get_class_static()._native_ptr(), StringName("set_agent_radius")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_agent_radius_encoded;
	PtrToArg<double>::encode(p_agent_radius, &p_agent_radius_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_agent_radius_encoded);
}

float NavigationMesh::get_agent_radius() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMesh::get_class_static()._native_ptr(), StringName("get_agent_radius")._native_ptr(), 191475506);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void NavigationMesh::set_agent_max_climb(float p_agent_max_climb) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMesh::get_class_static()._native_ptr(), StringName("set_agent_max_climb")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_agent_max_climb_encoded;
	PtrToArg<double>::encode(p_agent_max_climb, &p_agent_max_climb_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_agent_max_climb_encoded);
}

float NavigationMesh::get_agent_max_climb() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMesh::get_class_static()._native_ptr(), StringName("get_agent_max_climb")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void NavigationMesh::set_agent_max_slope(float p_agent_max_slope) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMesh::get_class_static()._native_ptr(), StringName("set_agent_max_slope")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_agent_max_slope_encoded;
	PtrToArg<double>::encode(p_agent_max_slope, &p_agent_max_slope_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_agent_max_slope_encoded);
}

float NavigationMesh::get_agent_max_slope() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMesh::get_class_static()._native_ptr(), StringName("get_agent_max_slope")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void NavigationMesh::set_region_min_size(float p_region_min_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMesh::get_class_static()._native_ptr(), StringName("set_region_min_size")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_region_min_size_encoded;
	PtrToArg<double>::encode(p_region_min_size, &p_region_min_size_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_region_min_size_encoded);
}

float NavigationMesh::get_region_min_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMesh::get_class_static()._native_ptr(), StringName("get_region_min_size")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void NavigationMesh::set_region_merge_size(float p_region_merge_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMesh::get_class_static()._native_ptr(), StringName("set_region_merge_size")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_region_merge_size_encoded;
	PtrToArg<double>::encode(p_region_merge_size, &p_region_merge_size_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_region_merge_size_encoded);
}

float NavigationMesh::get_region_merge_size() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMesh::get_class_static()._native_ptr(), StringName("get_region_merge_size")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void NavigationMesh::set_edge_max_length(float p_edge_max_length) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMesh::get_class_static()._native_ptr(), StringName("set_edge_max_length")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_edge_max_length_encoded;
	PtrToArg<double>::encode(p_edge_max_length, &p_edge_max_length_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_edge_max_length_encoded);
}

float NavigationMesh::get_edge_max_length() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMesh::get_class_static()._native_ptr(), StringName("get_edge_max_length")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void NavigationMesh::set_edge_max_error(float p_edge_max_error) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMesh::get_class_static()._native_ptr(), StringName("set_edge_max_error")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_edge_max_error_encoded;
	PtrToArg<double>::encode(p_edge_max_error, &p_edge_max_error_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_edge_max_error_encoded);
}

float NavigationMesh::get_edge_max_error() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMesh::get_class_static()._native_ptr(), StringName("get_edge_max_error")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void NavigationMesh::set_vertices_per_polygon(float p_vertices_per_polygon) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMesh::get_class_static()._native_ptr(), StringName("set_vertices_per_polygon")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_vertices_per_polygon_encoded;
	PtrToArg<double>::encode(p_vertices_per_polygon, &p_vertices_per_polygon_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_vertices_per_polygon_encoded);
}

float NavigationMesh::get_vertices_per_polygon() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMesh::get_class_static()._native_ptr(), StringName("get_vertices_per_polygon")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void NavigationMesh::set_detail_sample_distance(float p_detail_sample_dist) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMesh::get_class_static()._native_ptr(), StringName("set_detail_sample_distance")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_detail_sample_dist_encoded;
	PtrToArg<double>::encode(p_detail_sample_dist, &p_detail_sample_dist_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_detail_sample_dist_encoded);
}

float NavigationMesh::get_detail_sample_distance() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMesh::get_class_static()._native_ptr(), StringName("get_detail_sample_distance")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void NavigationMesh::set_detail_sample_max_error(float p_detail_sample_max_error) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMesh::get_class_static()._native_ptr(), StringName("set_detail_sample_max_error")._native_ptr(), 373806689);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_detail_sample_max_error_encoded;
	PtrToArg<double>::encode(p_detail_sample_max_error, &p_detail_sample_max_error_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_detail_sample_max_error_encoded);
}

float NavigationMesh::get_detail_sample_max_error() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMesh::get_class_static()._native_ptr(), StringName("get_detail_sample_max_error")._native_ptr(), 1740695150);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void NavigationMesh::set_filter_low_hanging_obstacles(bool p_filter_low_hanging_obstacles) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMesh::get_class_static()._native_ptr(), StringName("set_filter_low_hanging_obstacles")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_filter_low_hanging_obstacles_encoded;
	PtrToArg<bool>::encode(p_filter_low_hanging_obstacles, &p_filter_low_hanging_obstacles_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_filter_low_hanging_obstacles_encoded);
}

bool NavigationMesh::get_filter_low_hanging_obstacles() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMesh::get_class_static()._native_ptr(), StringName("get_filter_low_hanging_obstacles")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void NavigationMesh::set_filter_ledge_spans(bool p_filter_ledge_spans) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMesh::get_class_static()._native_ptr(), StringName("set_filter_ledge_spans")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_filter_ledge_spans_encoded;
	PtrToArg<bool>::encode(p_filter_ledge_spans, &p_filter_ledge_spans_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_filter_ledge_spans_encoded);
}

bool NavigationMesh::get_filter_ledge_spans() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMesh::get_class_static()._native_ptr(), StringName("get_filter_ledge_spans")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void NavigationMesh::set_filter_walkable_low_height_spans(bool p_filter_walkable_low_height_spans) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMesh::get_class_static()._native_ptr(), StringName("set_filter_walkable_low_height_spans")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_filter_walkable_low_height_spans_encoded;
	PtrToArg<bool>::encode(p_filter_walkable_low_height_spans, &p_filter_walkable_low_height_spans_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_filter_walkable_low_height_spans_encoded);
}

bool NavigationMesh::get_filter_walkable_low_height_spans() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMesh::get_class_static()._native_ptr(), StringName("get_filter_walkable_low_height_spans")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void NavigationMesh::set_filter_baking_aabb(const AABB &p_baking_aabb) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMesh::get_class_static()._native_ptr(), StringName("set_filter_baking_aabb")._native_ptr(), 259215842);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_baking_aabb);
}

AABB NavigationMesh::get_filter_baking_aabb() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMesh::get_class_static()._native_ptr(), StringName("get_filter_baking_aabb")._native_ptr(), 1068685055);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (AABB()));
	return ::godot::internal::_call_native_mb_ret<AABB>(_gde_method_bind, _owner);
}

void NavigationMesh::set_filter_baking_aabb_offset(const Vector3 &p_baking_aabb_offset) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMesh::get_class_static()._native_ptr(), StringName("set_filter_baking_aabb_offset")._native_ptr(), 3460891852);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_baking_aabb_offset);
}

Vector3 NavigationMesh::get_filter_baking_aabb_offset() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMesh::get_class_static()._native_ptr(), StringName("get_filter_baking_aabb_offset")._native_ptr(), 3360562783);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Vector3()));
	return ::godot::internal::_call_native_mb_ret<Vector3>(_gde_method_bind, _owner);
}

void NavigationMesh::set_vertices(const PackedVector3Array &p_vertices) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMesh::get_class_static()._native_ptr(), StringName("set_vertices")._native_ptr(), 334873810);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_vertices);
}

PackedVector3Array NavigationMesh::get_vertices() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMesh::get_class_static()._native_ptr(), StringName("get_vertices")._native_ptr(), 497664490);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedVector3Array()));
	return ::godot::internal::_call_native_mb_ret<PackedVector3Array>(_gde_method_bind, _owner);
}

void NavigationMesh::add_polygon(const PackedInt32Array &p_polygon) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMesh::get_class_static()._native_ptr(), StringName("add_polygon")._native_ptr(), 3614634198);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_polygon);
}

int32_t NavigationMesh::get_polygon_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMesh::get_class_static()._native_ptr(), StringName("get_polygon_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

PackedInt32Array NavigationMesh::get_polygon(int32_t p_idx) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMesh::get_class_static()._native_ptr(), StringName("get_polygon")._native_ptr(), 3668444399);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedInt32Array()));
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_native_mb_ret<PackedInt32Array>(_gde_method_bind, _owner, &p_idx_encoded);
}

void NavigationMesh::clear_polygons() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMesh::get_class_static()._native_ptr(), StringName("clear_polygons")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void NavigationMesh::create_from_mesh(const Ref<Mesh> &p_mesh) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMesh::get_class_static()._native_ptr(), StringName("create_from_mesh")._native_ptr(), 194775623);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_mesh != nullptr ? &p_mesh->_owner : nullptr));
}

void NavigationMesh::clear() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(NavigationMesh::get_class_static()._native_ptr(), StringName("clear")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

} // namespace godot
