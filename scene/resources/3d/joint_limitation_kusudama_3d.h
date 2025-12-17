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

#include "core/math/math_defs.h"
#include "scene/resources/3d/joint_limitation_3d.h"

class JointLimitationKusudama3D : public JointLimitation3D {
	GDCLASS(JointLimitationKusudama3D, JointLimitation3D);

	// Cones data: each cone is stored as Vector4(center_x, center_y, center_z, radius)
	Vector<Vector4> cones;

#ifdef TOOLS_ENABLED
	typedef Pair<Vector3, Vector3> Segment;
#endif // TOOLS_ENABLED

protected:
	static void _bind_methods();
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

	virtual Vector3 _solve(const Vector3 &p_direction) const override;

private:
	bool is_point_in_cone(const Vector3 &p_point, const Vector3 &p_cone_center, real_t p_cone_radius) const;
	bool is_point_in_tangent_path(const Vector3 &p_point, const Vector3 &p_center1, real_t p_radius1, const Vector3 &p_center2, real_t p_radius2) const;
	Vector3 get_on_great_tangent_triangle(const Vector3 &p_point, const Vector3 &p_center1, real_t p_radius1, const Vector3 &p_center2, real_t p_radius2) const;
	void extend_ray(Vector3 &r_start, Vector3 &r_end, real_t p_amount) const;
	int ray_sphere_intersection_full(const Vector3 &p_ray_start, const Vector3 &p_ray_end, const Vector3 &p_sphere_center, real_t p_radius, Vector3 *r_intersection1, Vector3 *r_intersection2) const;
	void compute_tangent_circles(const Vector3 &p_center1, real_t p_radius1, const Vector3 &p_center2, real_t p_radius2, Vector3 &r_tangent1, Vector3 &r_tangent2, real_t &r_tangent_radius) const;

public:
	void set_cones(const Vector<Vector4> &p_cones);
	Vector<Vector4> get_cones() const;

	void set_cone_count(int p_count);
	int get_cone_count() const;

	void set_cone_center(int p_index, const Vector3 &p_center);
	Vector3 get_cone_center(int p_index) const;

	void set_cone_radius(int p_index, real_t p_radius);
	real_t get_cone_radius(int p_index) const;

#ifdef TOOLS_ENABLED
	LocalVector<Segment> get_icosahedron_sphere(int p_subdiv) const;
	void get_icosahedron_triangles(int p_subdiv, LocalVector<Vector3> &r_triangles) const;
	LocalVector<Segment> cull_lines_by_boundary(const LocalVector<Segment> &p_segments, LocalVector<Vector3> &r_crossed_points) const;
	bool is_in_boundary(const Vector3 &p_point, Vector3 &r_solved) const;
	LocalVector<Vector3> sort_by_nearest_point(const LocalVector<Vector3> &p_points) const;
	virtual void draw_shape(Ref<SurfaceTool> p_surface_tool, const Transform3D &p_transform, float p_bone_length, const Color &p_color, int p_bone_index = -1, Ref<SurfaceTool> p_fill_surface_tool = Ref<SurfaceTool>()) const override;
#endif // TOOLS_ENABLED
};
