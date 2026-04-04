/**************************************************************************/
/*  navigation_path_query_parameters3d.hpp                                */
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

#pragma once

#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/rid.hpp>
#include <godot_cpp/variant/typed_array.hpp>
#include <godot_cpp/variant/vector3.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class NavigationPathQueryParameters3D : public RefCounted {
	GDEXTENSION_CLASS(NavigationPathQueryParameters3D, RefCounted)

public:
	enum PathfindingAlgorithm {
		PATHFINDING_ALGORITHM_ASTAR = 0,
	};

	enum PathPostProcessing {
		PATH_POSTPROCESSING_CORRIDORFUNNEL = 0,
		PATH_POSTPROCESSING_EDGECENTERED = 1,
		PATH_POSTPROCESSING_NONE = 2,
	};

	enum PathMetadataFlags : uint64_t {
		PATH_METADATA_INCLUDE_NONE = 0,
		PATH_METADATA_INCLUDE_TYPES = 1,
		PATH_METADATA_INCLUDE_RIDS = 2,
		PATH_METADATA_INCLUDE_OWNERS = 4,
		PATH_METADATA_INCLUDE_ALL = 7,
	};

	void set_pathfinding_algorithm(NavigationPathQueryParameters3D::PathfindingAlgorithm p_pathfinding_algorithm);
	NavigationPathQueryParameters3D::PathfindingAlgorithm get_pathfinding_algorithm() const;
	void set_path_postprocessing(NavigationPathQueryParameters3D::PathPostProcessing p_path_postprocessing);
	NavigationPathQueryParameters3D::PathPostProcessing get_path_postprocessing() const;
	void set_map(const RID &p_map);
	RID get_map() const;
	void set_start_position(const Vector3 &p_start_position);
	Vector3 get_start_position() const;
	void set_target_position(const Vector3 &p_target_position);
	Vector3 get_target_position() const;
	void set_navigation_layers(uint32_t p_navigation_layers);
	uint32_t get_navigation_layers() const;
	void set_metadata_flags(BitField<NavigationPathQueryParameters3D::PathMetadataFlags> p_flags);
	BitField<NavigationPathQueryParameters3D::PathMetadataFlags> get_metadata_flags() const;
	void set_simplify_path(bool p_enabled);
	bool get_simplify_path() const;
	void set_simplify_epsilon(float p_epsilon);
	float get_simplify_epsilon() const;
	void set_included_regions(const TypedArray<RID> &p_regions);
	TypedArray<RID> get_included_regions() const;
	void set_excluded_regions(const TypedArray<RID> &p_regions);
	TypedArray<RID> get_excluded_regions() const;
	void set_path_return_max_length(float p_length);
	float get_path_return_max_length() const;
	void set_path_return_max_radius(float p_radius);
	float get_path_return_max_radius() const;
	void set_path_search_max_polygons(int32_t p_max_polygons);
	int32_t get_path_search_max_polygons() const;
	void set_path_search_max_distance(float p_distance);
	float get_path_search_max_distance() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		RefCounted::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(NavigationPathQueryParameters3D::PathfindingAlgorithm);
VARIANT_ENUM_CAST(NavigationPathQueryParameters3D::PathPostProcessing);
VARIANT_BITFIELD_CAST(NavigationPathQueryParameters3D::PathMetadataFlags);

