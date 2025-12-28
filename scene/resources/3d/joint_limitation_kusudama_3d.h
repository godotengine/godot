/**************************************************************************/
/*  joint_limitation_kusudama_3d.h                                        */
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

#pragma once

#include "scene/resources/3d/joint_limitation_3d.h"

class JointLimitationKusudama3D : public JointLimitation3D {
	GDCLASS(JointLimitationKusudama3D, JointLimitation3D);

	// Cones data: Storage format uses 3 consecutive Vector4s per cone pair [cone1, tan2, tan1]
	// Group i is stored at indices [i*3+0, i*3+1, i*3+2]
	// Note: tangents are swapped in storage (+1 stores tan2, +2 stores tan1) to match shader expectations
	// cone2 of group i is the same as cone1 of group i+1 (or stored separately for the last group)
	// Each Vector4 is (x, y, z, radius)
	LocalVector<Vector4> cones;

#ifdef TOOLS_ENABLED
	typedef Pair<Vector3, Vector3> Segment;
#endif // TOOLS_ENABLED

protected:
	static void _bind_methods();
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

	virtual Vector3 _solve(const Vector3 &p_direction) const override;

	void _update_quad_tangents(int p_quad_index);

public:
	void set_cone_count(int p_count);
	int get_cone_count() const;

	void set_cone_center(int p_index, const Vector3 &p_center);
	Vector3 get_cone_center(int p_index) const;

	void set_cone_radius(int p_index, real_t p_radius);
	real_t get_cone_radius(int p_index) const;

#ifdef TOOLS_ENABLED
	LocalVector<Segment> get_icosahedron_sphere(int p_subdiv) const;
	LocalVector<Segment> cull_lines_by_boundary(const LocalVector<Segment> &p_segments, LocalVector<Vector3> &r_crossed_points) const;
	bool is_in_boundary(const Vector3 &p_point, Vector3 &r_solved) const;
	LocalVector<Vector3> sort_by_nearest_point(const LocalVector<Vector3> &p_points) const;
	virtual void draw_shape(Ref<SurfaceTool> &p_surface_tool, const Transform3D &p_transform, float p_bone_length, const Color &p_color, int p_bone_index = -1) const override;
#endif // TOOLS_ENABLED
};
