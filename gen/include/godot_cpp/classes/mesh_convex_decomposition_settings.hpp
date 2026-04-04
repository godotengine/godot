/**************************************************************************/
/*  mesh_convex_decomposition_settings.hpp                                */
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

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class MeshConvexDecompositionSettings : public RefCounted {
	GDEXTENSION_CLASS(MeshConvexDecompositionSettings, RefCounted)

public:
	enum Mode {
		CONVEX_DECOMPOSITION_MODE_VOXEL = 0,
		CONVEX_DECOMPOSITION_MODE_TETRAHEDRON = 1,
	};

	void set_max_concavity(float p_max_concavity);
	float get_max_concavity() const;
	void set_symmetry_planes_clipping_bias(float p_symmetry_planes_clipping_bias);
	float get_symmetry_planes_clipping_bias() const;
	void set_revolution_axes_clipping_bias(float p_revolution_axes_clipping_bias);
	float get_revolution_axes_clipping_bias() const;
	void set_min_volume_per_convex_hull(float p_min_volume_per_convex_hull);
	float get_min_volume_per_convex_hull() const;
	void set_resolution(uint32_t p_min_volume_per_convex_hull);
	uint32_t get_resolution() const;
	void set_max_num_vertices_per_convex_hull(uint32_t p_max_num_vertices_per_convex_hull);
	uint32_t get_max_num_vertices_per_convex_hull() const;
	void set_plane_downsampling(uint32_t p_plane_downsampling);
	uint32_t get_plane_downsampling() const;
	void set_convex_hull_downsampling(uint32_t p_convex_hull_downsampling);
	uint32_t get_convex_hull_downsampling() const;
	void set_normalize_mesh(bool p_normalize_mesh);
	bool get_normalize_mesh() const;
	void set_mode(MeshConvexDecompositionSettings::Mode p_mode);
	MeshConvexDecompositionSettings::Mode get_mode() const;
	void set_convex_hull_approximation(bool p_convex_hull_approximation);
	bool get_convex_hull_approximation() const;
	void set_max_convex_hulls(uint32_t p_max_convex_hulls);
	uint32_t get_max_convex_hulls() const;
	void set_project_hull_vertices(bool p_project_hull_vertices);
	bool get_project_hull_vertices() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		RefCounted::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(MeshConvexDecompositionSettings::Mode);

