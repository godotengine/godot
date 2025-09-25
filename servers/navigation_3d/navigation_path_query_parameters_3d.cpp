/**************************************************************************/
/*  navigation_path_query_parameters_3d.cpp                               */
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

#include "navigation_path_query_parameters_3d.h"

#include "core/variant/typed_array.h"

void NavigationPathQueryParameters3D::set_pathfinding_algorithm(const NavigationPathQueryParameters3D::PathfindingAlgorithm p_pathfinding_algorithm) {
	pathfinding_algorithm = p_pathfinding_algorithm;
}

NavigationPathQueryParameters3D::PathfindingAlgorithm NavigationPathQueryParameters3D::get_pathfinding_algorithm() const {
	return pathfinding_algorithm;
}

void NavigationPathQueryParameters3D::set_path_postprocessing(const NavigationPathQueryParameters3D::PathPostProcessing p_path_postprocessing) {
	path_postprocessing = p_path_postprocessing;
}

NavigationPathQueryParameters3D::PathPostProcessing NavigationPathQueryParameters3D::get_path_postprocessing() const {
	return path_postprocessing;
}

void NavigationPathQueryParameters3D::set_map(RID p_map) {
	map = p_map;
}

RID NavigationPathQueryParameters3D::get_map() const {
	return map;
}

void NavigationPathQueryParameters3D::set_start_position(Vector3 p_start_position) {
	start_position = p_start_position;
}

Vector3 NavigationPathQueryParameters3D::get_start_position() const {
	return start_position;
}

void NavigationPathQueryParameters3D::set_target_position(Vector3 p_target_position) {
	target_position = p_target_position;
}

Vector3 NavigationPathQueryParameters3D::get_target_position() const {
	return target_position;
}

void NavigationPathQueryParameters3D::set_navigation_layers(uint32_t p_navigation_layers) {
	navigation_layers = p_navigation_layers;
}

uint32_t NavigationPathQueryParameters3D::get_navigation_layers() const {
	return navigation_layers;
}

void NavigationPathQueryParameters3D::set_metadata_flags(BitField<NavigationPathQueryParameters3D::PathMetadataFlags> p_flags) {
	metadata_flags = (int64_t)p_flags;
}

BitField<NavigationPathQueryParameters3D::PathMetadataFlags> NavigationPathQueryParameters3D::get_metadata_flags() const {
	return (int64_t)metadata_flags;
}

void NavigationPathQueryParameters3D::set_simplify_path(bool p_enabled) {
	simplify_path = p_enabled;
}

bool NavigationPathQueryParameters3D::get_simplify_path() const {
	return simplify_path;
}

void NavigationPathQueryParameters3D::set_simplify_epsilon(real_t p_epsilon) {
	simplify_epsilon = MAX(0.0, p_epsilon);
}

real_t NavigationPathQueryParameters3D::get_simplify_epsilon() const {
	return simplify_epsilon;
}

void NavigationPathQueryParameters3D::set_included_regions(const TypedArray<RID> &p_regions) {
	_included_regions.resize(p_regions.size());
	for (uint32_t i = 0; i < _included_regions.size(); i++) {
		_included_regions[i] = p_regions[i];
	}
}

TypedArray<RID> NavigationPathQueryParameters3D::get_included_regions() const {
	TypedArray<RID> r_regions;
	r_regions.resize(_included_regions.size());
	for (uint32_t i = 0; i < _included_regions.size(); i++) {
		r_regions[i] = _included_regions[i];
	}
	return r_regions;
}

void NavigationPathQueryParameters3D::set_excluded_regions(const TypedArray<RID> &p_regions) {
	_excluded_regions.resize(p_regions.size());
	for (uint32_t i = 0; i < _excluded_regions.size(); i++) {
		_excluded_regions[i] = p_regions[i];
	}
}

TypedArray<RID> NavigationPathQueryParameters3D::get_excluded_regions() const {
	TypedArray<RID> r_regions;
	r_regions.resize(_excluded_regions.size());
	for (uint32_t i = 0; i < _excluded_regions.size(); i++) {
		r_regions[i] = _excluded_regions[i];
	}
	return r_regions;
}

void NavigationPathQueryParameters3D::set_path_return_max_length(float p_length) {
	path_return_max_length = MAX(0.0, p_length);
}

float NavigationPathQueryParameters3D::get_path_return_max_length() const {
	return path_return_max_length;
}

void NavigationPathQueryParameters3D::set_path_return_max_radius(float p_radius) {
	path_return_max_radius = MAX(0.0, p_radius);
}

float NavigationPathQueryParameters3D::get_path_return_max_radius() const {
	return path_return_max_radius;
}

void NavigationPathQueryParameters3D::set_path_search_max_polygons(int p_max_polygons) {
	path_search_max_polygons = p_max_polygons;
}

int NavigationPathQueryParameters3D::get_path_search_max_polygons() const {
	return path_search_max_polygons;
}

void NavigationPathQueryParameters3D::set_path_search_max_distance(float p_distance) {
	path_search_max_distance = MAX(0.0, p_distance);
}

float NavigationPathQueryParameters3D::get_path_search_max_distance() const {
	return path_search_max_distance;
}

void NavigationPathQueryParameters3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_pathfinding_algorithm", "pathfinding_algorithm"), &NavigationPathQueryParameters3D::set_pathfinding_algorithm);
	ClassDB::bind_method(D_METHOD("get_pathfinding_algorithm"), &NavigationPathQueryParameters3D::get_pathfinding_algorithm);

	ClassDB::bind_method(D_METHOD("set_path_postprocessing", "path_postprocessing"), &NavigationPathQueryParameters3D::set_path_postprocessing);
	ClassDB::bind_method(D_METHOD("get_path_postprocessing"), &NavigationPathQueryParameters3D::get_path_postprocessing);

	ClassDB::bind_method(D_METHOD("set_map", "map"), &NavigationPathQueryParameters3D::set_map);
	ClassDB::bind_method(D_METHOD("get_map"), &NavigationPathQueryParameters3D::get_map);

	ClassDB::bind_method(D_METHOD("set_start_position", "start_position"), &NavigationPathQueryParameters3D::set_start_position);
	ClassDB::bind_method(D_METHOD("get_start_position"), &NavigationPathQueryParameters3D::get_start_position);

	ClassDB::bind_method(D_METHOD("set_target_position", "target_position"), &NavigationPathQueryParameters3D::set_target_position);
	ClassDB::bind_method(D_METHOD("get_target_position"), &NavigationPathQueryParameters3D::get_target_position);

	ClassDB::bind_method(D_METHOD("set_navigation_layers", "navigation_layers"), &NavigationPathQueryParameters3D::set_navigation_layers);
	ClassDB::bind_method(D_METHOD("get_navigation_layers"), &NavigationPathQueryParameters3D::get_navigation_layers);

	ClassDB::bind_method(D_METHOD("set_metadata_flags", "flags"), &NavigationPathQueryParameters3D::set_metadata_flags);
	ClassDB::bind_method(D_METHOD("get_metadata_flags"), &NavigationPathQueryParameters3D::get_metadata_flags);

	ClassDB::bind_method(D_METHOD("set_simplify_path", "enabled"), &NavigationPathQueryParameters3D::set_simplify_path);
	ClassDB::bind_method(D_METHOD("get_simplify_path"), &NavigationPathQueryParameters3D::get_simplify_path);

	ClassDB::bind_method(D_METHOD("set_simplify_epsilon", "epsilon"), &NavigationPathQueryParameters3D::set_simplify_epsilon);
	ClassDB::bind_method(D_METHOD("get_simplify_epsilon"), &NavigationPathQueryParameters3D::get_simplify_epsilon);

	ClassDB::bind_method(D_METHOD("set_included_regions", "regions"), &NavigationPathQueryParameters3D::set_included_regions);
	ClassDB::bind_method(D_METHOD("get_included_regions"), &NavigationPathQueryParameters3D::get_included_regions);

	ClassDB::bind_method(D_METHOD("set_excluded_regions", "regions"), &NavigationPathQueryParameters3D::set_excluded_regions);
	ClassDB::bind_method(D_METHOD("get_excluded_regions"), &NavigationPathQueryParameters3D::get_excluded_regions);

	ClassDB::bind_method(D_METHOD("set_path_return_max_length", "length"), &NavigationPathQueryParameters3D::set_path_return_max_length);
	ClassDB::bind_method(D_METHOD("get_path_return_max_length"), &NavigationPathQueryParameters3D::get_path_return_max_length);

	ClassDB::bind_method(D_METHOD("set_path_return_max_radius", "radius"), &NavigationPathQueryParameters3D::set_path_return_max_radius);
	ClassDB::bind_method(D_METHOD("get_path_return_max_radius"), &NavigationPathQueryParameters3D::get_path_return_max_radius);

	ClassDB::bind_method(D_METHOD("set_path_search_max_polygons", "max_polygons"), &NavigationPathQueryParameters3D::set_path_search_max_polygons);
	ClassDB::bind_method(D_METHOD("get_path_search_max_polygons"), &NavigationPathQueryParameters3D::get_path_search_max_polygons);

	ClassDB::bind_method(D_METHOD("set_path_search_max_distance", "distance"), &NavigationPathQueryParameters3D::set_path_search_max_distance);
	ClassDB::bind_method(D_METHOD("get_path_search_max_distance"), &NavigationPathQueryParameters3D::get_path_search_max_distance);

	ADD_PROPERTY(PropertyInfo(Variant::RID, "map"), "set_map", "get_map");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "start_position"), "set_start_position", "get_start_position");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "target_position"), "set_target_position", "get_target_position");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "navigation_layers", PROPERTY_HINT_LAYERS_3D_NAVIGATION), "set_navigation_layers", "get_navigation_layers");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "pathfinding_algorithm", PROPERTY_HINT_ENUM, "AStar"), "set_pathfinding_algorithm", "get_pathfinding_algorithm");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "path_postprocessing", PROPERTY_HINT_ENUM, "Corridorfunnel,Edgecentered,None"), "set_path_postprocessing", "get_path_postprocessing");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "metadata_flags", PROPERTY_HINT_FLAGS, "Include Types,Include RIDs,Include Owners"), "set_metadata_flags", "get_metadata_flags");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "simplify_path"), "set_simplify_path", "get_simplify_path");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "simplify_epsilon"), "set_simplify_epsilon", "get_simplify_epsilon");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "excluded_regions", PROPERTY_HINT_ARRAY_TYPE, "RID"), "set_excluded_regions", "get_excluded_regions");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "included_regions", PROPERTY_HINT_ARRAY_TYPE, "RID"), "set_included_regions", "get_included_regions");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "path_return_max_length"), "set_path_return_max_length", "get_path_return_max_length");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "path_return_max_radius"), "set_path_return_max_radius", "get_path_return_max_radius");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "path_search_max_polygons"), "set_path_search_max_polygons", "get_path_search_max_polygons");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "path_search_max_distance"), "set_path_search_max_distance", "get_path_search_max_distance");

	BIND_ENUM_CONSTANT(PATHFINDING_ALGORITHM_ASTAR);

	BIND_ENUM_CONSTANT(PATH_POSTPROCESSING_CORRIDORFUNNEL);
	BIND_ENUM_CONSTANT(PATH_POSTPROCESSING_EDGECENTERED);
	BIND_ENUM_CONSTANT(PATH_POSTPROCESSING_NONE);

	BIND_BITFIELD_FLAG(PATH_METADATA_INCLUDE_NONE);
	BIND_BITFIELD_FLAG(PATH_METADATA_INCLUDE_TYPES);
	BIND_BITFIELD_FLAG(PATH_METADATA_INCLUDE_RIDS);
	BIND_BITFIELD_FLAG(PATH_METADATA_INCLUDE_OWNERS);
	BIND_BITFIELD_FLAG(PATH_METADATA_INCLUDE_ALL);
}
