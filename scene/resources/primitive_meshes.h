/*************************************************************************/
/*  primitive_meshes.h                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#ifndef PRIMITIVE_MESHES_H
#define PRIMITIVE_MESHES_H

#include "scene/resources/mesh.h"

///@TODO probably should change a few integers to unsigned integers...

/**
	@author Bastiaan Olij <mux213@gmail.com>

	Base class for all the classes in this file, handles a number of code functions that are shared among all meshes.
	This class is set apart that it assumes a single surface is always generated for our mesh.
*/
class PrimitiveMesh : public Mesh {
	GDCLASS(PrimitiveMesh, Mesh);

private:
	RID mesh;
	mutable AABB aabb;
	AABB custom_aabb;

	mutable int array_len = 0;
	mutable int index_array_len = 0;

	Ref<Material> material;
	bool flip_faces = false;

	// make sure we do an update after we've finished constructing our object
	mutable bool pending_request = true;
	void _update() const;

protected:
	// assume primitive triangles as the type, correct for all but one and it will change this :)
	Mesh::PrimitiveType primitive_type = Mesh::PRIMITIVE_TRIANGLES;

	static void _bind_methods();

	virtual void _create_mesh_array(Array &p_arr) const = 0;
	void _request_update();

public:
	virtual int get_surface_count() const override;
	virtual int surface_get_array_len(int p_idx) const override;
	virtual int surface_get_array_index_len(int p_idx) const override;
	virtual Array surface_get_arrays(int p_surface) const override;
	virtual Array surface_get_blend_shape_arrays(int p_surface) const override;
	virtual Dictionary surface_get_lods(int p_surface) const override;
	virtual uint32_t surface_get_format(int p_idx) const override;
	virtual Mesh::PrimitiveType surface_get_primitive_type(int p_idx) const override;
	virtual void surface_set_material(int p_idx, const Ref<Material> &p_material) override;
	virtual Ref<Material> surface_get_material(int p_idx) const override;
	virtual int get_blend_shape_count() const override;
	virtual StringName get_blend_shape_name(int p_index) const override;
	virtual void set_blend_shape_name(int p_index, const StringName &p_name) override;
	virtual AABB get_aabb() const override;
	virtual RID get_rid() const override;

	void set_material(const Ref<Material> &p_material);
	Ref<Material> get_material() const;

	Array get_mesh_arrays() const;

	void set_custom_aabb(const AABB &p_custom);
	AABB get_custom_aabb() const;

	void set_flip_faces(bool p_enable);
	bool get_flip_faces() const;

	PrimitiveMesh();
	~PrimitiveMesh();
};

/**
	Mesh for a simple capsule
*/
class CapsuleMesh : public PrimitiveMesh {
	GDCLASS(CapsuleMesh, PrimitiveMesh);

private:
	float radius = 1.0;
	float height = 3.0;
	int radial_segments = 64;
	int rings = 8;

protected:
	static void _bind_methods();
	virtual void _create_mesh_array(Array &p_arr) const override;

public:
	void set_radius(const float p_radius);
	float get_radius() const;

	void set_height(const float p_height);
	float get_height() const;

	void set_radial_segments(const int p_segments);
	int get_radial_segments() const;

	void set_rings(const int p_rings);
	int get_rings() const;

	CapsuleMesh();
};

/**
	A box
*/
class BoxMesh : public PrimitiveMesh {
	GDCLASS(BoxMesh, PrimitiveMesh);

private:
	Vector3 size = Vector3(2.0, 2.0, 2.0);
	int subdivide_w = 0;
	int subdivide_h = 0;
	int subdivide_d = 0;

protected:
	static void _bind_methods();
	virtual void _create_mesh_array(Array &p_arr) const override;

public:
	void set_size(const Vector3 &p_size);
	Vector3 get_size() const;

	void set_subdivide_width(const int p_divisions);
	int get_subdivide_width() const;

	void set_subdivide_height(const int p_divisions);
	int get_subdivide_height() const;

	void set_subdivide_depth(const int p_divisions);
	int get_subdivide_depth() const;

	BoxMesh();
};

/**
	A cylinder
*/

class CylinderMesh : public PrimitiveMesh {
	GDCLASS(CylinderMesh, PrimitiveMesh);

private:
	float top_radius = 1.0;
	float bottom_radius = 1.0;
	float height = 2.0;
	int radial_segments = 64;
	int rings = 4;

protected:
	static void _bind_methods();
	virtual void _create_mesh_array(Array &p_arr) const override;

public:
	void set_top_radius(const float p_radius);
	float get_top_radius() const;

	void set_bottom_radius(const float p_radius);
	float get_bottom_radius() const;

	void set_height(const float p_height);
	float get_height() const;

	void set_radial_segments(const int p_segments);
	int get_radial_segments() const;

	void set_rings(const int p_rings);
	int get_rings() const;

	CylinderMesh();
};

/**
	Similar to quadmesh but with tessellation support
*/
class PlaneMesh : public PrimitiveMesh {
	GDCLASS(PlaneMesh, PrimitiveMesh);

private:
	Size2 size = Size2(2.0, 2.0);
	int subdivide_w = 0;
	int subdivide_d = 0;
	Vector3 center_offset;

protected:
	static void _bind_methods();
	virtual void _create_mesh_array(Array &p_arr) const override;

public:
	void set_size(const Size2 &p_size);
	Size2 get_size() const;

	void set_subdivide_width(const int p_divisions);
	int get_subdivide_width() const;

	void set_subdivide_depth(const int p_divisions);
	int get_subdivide_depth() const;

	void set_center_offset(const Vector3 p_offset);
	Vector3 get_center_offset() const;

	PlaneMesh();
};

/**
	A prism shapen, handy for ramps, triangles, etc.
*/
class PrismMesh : public PrimitiveMesh {
	GDCLASS(PrismMesh, PrimitiveMesh);

private:
	float left_to_right = 0.5;
	Vector3 size = Vector3(2.0, 2.0, 2.0);
	int subdivide_w = 0;
	int subdivide_h = 0;
	int subdivide_d = 0;

protected:
	static void _bind_methods();
	virtual void _create_mesh_array(Array &p_arr) const override;

public:
	void set_left_to_right(const float p_left_to_right);
	float get_left_to_right() const;

	void set_size(const Vector3 &p_size);
	Vector3 get_size() const;

	void set_subdivide_width(const int p_divisions);
	int get_subdivide_width() const;

	void set_subdivide_height(const int p_divisions);
	int get_subdivide_height() const;

	void set_subdivide_depth(const int p_divisions);
	int get_subdivide_depth() const;

	PrismMesh();
};

/**
	Our original quadmesh...
*/

class QuadMesh : public PrimitiveMesh {
	GDCLASS(QuadMesh, PrimitiveMesh);

private:
	Size2 size = Size2(1.0, 1.0);
	Vector3 center_offset;

protected:
	static void _bind_methods();
	virtual void _create_mesh_array(Array &p_arr) const override;

public:
	virtual uint32_t surface_get_format(int p_idx) const override;

	QuadMesh();

	void set_size(const Size2 &p_size);
	Size2 get_size() const;

	void set_center_offset(const Vector3 p_offset);
	Vector3 get_center_offset() const;
};

/**
	A sphere..
*/
class SphereMesh : public PrimitiveMesh {
	GDCLASS(SphereMesh, PrimitiveMesh);

private:
	float radius = 1.0;
	float height = 2.0;
	int radial_segments = 64;
	int rings = 32;
	bool is_hemisphere = false;

protected:
	static void _bind_methods();
	virtual void _create_mesh_array(Array &p_arr) const override;

public:
	void set_radius(const float p_radius);
	float get_radius() const;

	void set_height(const float p_height);
	float get_height() const;

	void set_radial_segments(const int p_radial_segments);
	int get_radial_segments() const;

	void set_rings(const int p_rings);
	int get_rings() const;

	void set_is_hemisphere(const bool p_is_hemisphere);
	bool get_is_hemisphere() const;

	SphereMesh();
};

/**
	A single point for use in particle systems
*/

class PointMesh : public PrimitiveMesh {
	GDCLASS(PointMesh, PrimitiveMesh)

protected:
	virtual void _create_mesh_array(Array &p_arr) const override;

public:
	PointMesh();
};

class TubeTrailMesh : public PrimitiveMesh {
	GDCLASS(TubeTrailMesh, PrimitiveMesh);

private:
	float radius = 1.0;
	int radial_steps = 8;
	int sections = 5;
	float section_length = 0.2;
	int section_rings = 3;

	Ref<Curve> curve;

	void _curve_changed();

protected:
	static void _bind_methods();
	virtual void _create_mesh_array(Array &p_arr) const override;

public:
	void set_radius(const float p_radius);
	float get_radius() const;

	void set_radial_steps(const int p_radial_steps);
	int get_radial_steps() const;

	void set_sections(const int p_sections);
	int get_sections() const;

	void set_section_length(float p_sectionlength);
	float get_section_length() const;

	void set_section_rings(const int p_section_rings);
	int get_section_rings() const;

	void set_curve(const Ref<Curve> &p_curve);
	Ref<Curve> get_curve() const;

	virtual int get_builtin_bind_pose_count() const override;
	virtual Transform3D get_builtin_bind_pose(int p_index) const override;

	TubeTrailMesh();
};

class RibbonTrailMesh : public PrimitiveMesh {
	GDCLASS(RibbonTrailMesh, PrimitiveMesh);

public:
	enum Shape {
		SHAPE_FLAT,
		SHAPE_CROSS
	};

private:
	float size = 1.0;
	int sections = 5;
	float section_length = 0.2;
	int section_segments = 3;

	Shape shape = SHAPE_CROSS;

	Ref<Curve> curve;

	void _curve_changed();

protected:
	static void _bind_methods();
	virtual void _create_mesh_array(Array &p_arr) const override;

public:
	void set_shape(Shape p_shape);
	Shape get_shape() const;

	void set_size(const float p_size);
	float get_size() const;

	void set_sections(const int p_sections);
	int get_sections() const;

	void set_section_length(float p_sectionlength);
	float get_section_length() const;

	void set_section_segments(const int p_section_segments);
	int get_section_segments() const;

	void set_curve(const Ref<Curve> &p_curve);
	Ref<Curve> get_curve() const;

	virtual int get_builtin_bind_pose_count() const override;
	virtual Transform3D get_builtin_bind_pose(int p_index) const override;

	RibbonTrailMesh();
};

VARIANT_ENUM_CAST(RibbonTrailMesh::Shape)
#endif
