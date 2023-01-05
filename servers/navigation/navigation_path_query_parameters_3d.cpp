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

void NavigationPathQueryParameters3D::set_pathfinding_algorithm(const NavigationPathQueryParameters3D::PathfindingAlgorithm p_pathfinding_algorithm) {
	switch (p_pathfinding_algorithm) {
		case PATHFINDING_ALGORITHM_ASTAR: {
			parameters.pathfinding_algorithm = NavigationUtilities::PathfindingAlgorithm::PATHFINDING_ALGORITHM_ASTAR;
		} break;
		default: {
			WARN_PRINT_ONCE("No match for used PathfindingAlgorithm - fallback to default");
			parameters.pathfinding_algorithm = NavigationUtilities::PathfindingAlgorithm::PATHFINDING_ALGORITHM_ASTAR;
		} break;
	}
}

NavigationPathQueryParameters3D::PathfindingAlgorithm NavigationPathQueryParameters3D::get_pathfinding_algorithm() const {
	switch (parameters.pathfinding_algorithm) {
		case NavigationUtilities::PathfindingAlgorithm::PATHFINDING_ALGORITHM_ASTAR:
			return PATHFINDING_ALGORITHM_ASTAR;
		default:
			WARN_PRINT_ONCE("No match for used PathfindingAlgorithm - fallback to default");
			return PATHFINDING_ALGORITHM_ASTAR;
	}
}

void NavigationPathQueryParameters3D::set_path_postprocessing(const NavigationPathQueryParameters3D::PathPostProcessing p_path_postprocessing) {
	switch (p_path_postprocessing) {
		case PATH_POSTPROCESSING_CORRIDORFUNNEL: {
			parameters.path_postprocessing = NavigationUtilities::PathPostProcessing::PATH_POSTPROCESSING_CORRIDORFUNNEL;
		} break;
		case PATH_POSTPROCESSING_EDGECENTERED: {
			parameters.path_postprocessing = NavigationUtilities::PathPostProcessing::PATH_POSTPROCESSING_EDGECENTERED;
		} break;
		default: {
			WARN_PRINT_ONCE("No match for used PathPostProcessing - fallback to default");
			parameters.path_postprocessing = NavigationUtilities::PathPostProcessing::PATH_POSTPROCESSING_CORRIDORFUNNEL;
		} break;
	}
}

NavigationPathQueryParameters3D::PathPostProcessing NavigationPathQueryParameters3D::get_path_postprocessing() const {
	switch (parameters.path_postprocessing) {
		case NavigationUtilities::PathPostProcessing::PATH_POSTPROCESSING_CORRIDORFUNNEL:
			return PATH_POSTPROCESSING_CORRIDORFUNNEL;
		case NavigationUtilities::PathPostProcessing::PATH_POSTPROCESSING_EDGECENTERED:
			return PATH_POSTPROCESSING_EDGECENTERED;
		default:
			WARN_PRINT_ONCE("No match for used PathPostProcessing - fallback to default");
			return PATH_POSTPROCESSING_CORRIDORFUNNEL;
	}
}

void NavigationPathQueryParameters3D::set_map(const RID &p_map) {
	parameters.map = p_map;
}

const RID &NavigationPathQueryParameters3D::get_map() const {
	return parameters.map;
}

void NavigationPathQueryParameters3D::set_start_position(const Vector3 &p_start_position) {
	parameters.start_position = p_start_position;
}

const Vector3 &NavigationPathQueryParameters3D::get_start_position() const {
	return parameters.start_position;
}

void NavigationPathQueryParameters3D::set_target_position(const Vector3 &p_target_position) {
	parameters.target_position = p_target_position;
}

const Vector3 &NavigationPathQueryParameters3D::get_target_position() const {
	return parameters.target_position;
}

void NavigationPathQueryParameters3D::set_navigation_layers(uint32_t p_navigation_layers) {
	parameters.navigation_layers = p_navigation_layers;
}

uint32_t NavigationPathQueryParameters3D::get_navigation_layers() const {
	return parameters.navigation_layers;
}

void NavigationPathQueryParameters3D::set_metadata_flags(BitField<NavigationPathQueryParameters3D::PathMetadataFlags> p_flags) {
	parameters.metadata_flags = (int64_t)p_flags;
}

BitField<NavigationPathQueryParameters3D::PathMetadataFlags> NavigationPathQueryParameters3D::get_metadata_flags() const {
	return (int64_t)parameters.metadata_flags;
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

	ADD_PROPERTY(PropertyInfo(Variant::RID, "map"), "set_map", "get_map");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "start_position"), "set_start_position", "get_start_position");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "target_position"), "set_target_position", "get_target_position");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "navigation_layers", PROPERTY_HINT_LAYERS_3D_NAVIGATION), "set_navigation_layers", "get_navigation_layers");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "pathfinding_algorithm", PROPERTY_HINT_ENUM, "AStar"), "set_pathfinding_algorithm", "get_pathfinding_algorithm");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "path_postprocessing", PROPERTY_HINT_ENUM, "Corridorfunnel,Edgecentered"), "set_path_postprocessing", "get_path_postprocessing");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "metadata_flags", PROPERTY_HINT_FLAGS, "Include Types,Include RIDs,Include Owners"), "set_metadata_flags", "get_metadata_flags");

	BIND_ENUM_CONSTANT(PATHFINDING_ALGORITHM_ASTAR);

	BIND_ENUM_CONSTANT(PATH_POSTPROCESSING_CORRIDORFUNNEL);
	BIND_ENUM_CONSTANT(PATH_POSTPROCESSING_EDGECENTERED);

	BIND_BITFIELD_FLAG(PATH_METADATA_INCLUDE_NONE);
	BIND_BITFIELD_FLAG(PATH_METADATA_INCLUDE_TYPES);
	BIND_BITFIELD_FLAG(PATH_METADATA_INCLUDE_RIDS);
	BIND_BITFIELD_FLAG(PATH_METADATA_INCLUDE_OWNERS);
	BIND_BITFIELD_FLAG(PATH_METADATA_INCLUDE_ALL);
}
