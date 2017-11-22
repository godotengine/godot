/*************************************************************************/
/*  primitive_meshes.h                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
	This class is set appart that it assumes a single surface is always generated for our mesh.
*/
class PrimitiveMesh : public Mesh {

	GDCLASS(PrimitiveMesh, Mesh);

private:
	RID mesh;
	mutable AABB aabb;

	Ref<Material> material;

	mutable bool pending_request;
	void _update() const;

protected:
	Mesh::PrimitiveType primitive_type;

	static void _bind_methods();

	virtual void _create_mesh_array(Array &p_arr) const = 0;
	void _request_update();

public:
	virtual int get_surface_count() const;
	virtual int surface_get_array_len(int p_idx) const;
	virtual int surface_get_array_index_len(int p_idx) const;
	virtual Array surface_get_arrays(int p_surface) const;
	virtual Array surface_get_blend_shape_arrays(int p_surface) const;
	virtual uint32_t surface_get_format(int p_idx) const;
	virtual Mesh::PrimitiveType surface_get_primitive_type(int p_idx) const;
	virtual Ref<Material> surface_get_material(int p_idx) const;
	virtual int get_blend_shape_count() const;
	virtual StringName get_blend_shape_name(int p_index) const;
	virtual AABB get_aabb() const;
	virtual RID get_rid() const;

	void set_material(const Ref<Material> &p_material);
	Ref<Material> get_material() const;

	Array get_mesh_arrays() const;

	PrimitiveMesh();
	~PrimitiveMesh();
};

/**
	Mesh for a simple capsule
*/
class CapsuleMesh : public PrimitiveMesh {
	GDCLASS(CapsuleMesh, PrimitiveMesh);

private:
	float radius;
	float mid_height;
	int radial_segments;
	int rings;

protected:
	static void _bind_methods();
	virtual void _create_mesh_array(Array &p_arr) const;

public:
	void set_radius(const float p_radius);
	float get_radius() const;

	void set_mid_height(const float p_mid_height);
	float get_mid_height() const;

	void set_radial_segments(const int p_segments);
	int get_radial_segments() const;

	void set_rings(const int p_rings);
	int get_rings() const;

	CapsuleMesh();
};

/**
	Similar to test cube but with subdivision support and different texture coordinates
*/
class CubeMesh : public PrimitiveMesh {

	GDCLASS(CubeMesh, PrimitiveMesh);

private:
	Vector3 size;
	int subdivide_w;
	int subdivide_h;
	int subdivide_d;

protected:
	static void _bind_methods();
	virtual void _create_mesh_array(Array &p_arr) const;

public:
	void set_size(const Vector3 &p_size);
	Vector3 get_size() const;

	void set_subdivide_width(const int p_divisions);
	int get_subdivide_width() const;

	void set_subdivide_height(const int p_divisions);
	int get_subdivide_height() const;

	void set_subdivide_depth(const int p_divisions);
	int get_subdivide_depth() const;

	CubeMesh();
};

/**
	A cylinder
*/

class CylinderMesh : public PrimitiveMesh {

	GDCLASS(CylinderMesh, PrimitiveMesh);

private:
	float top_radius;
	float bottom_radius;
	float height;
	int radial_segments;
	int rings;

protected:
	static void _bind_methods();
	virtual void _create_mesh_array(Array &p_arr) const;

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
	Similar to quadmesh but with tesselation support
*/
class PlaneMesh : public PrimitiveMesh {

	GDCLASS(PlaneMesh, PrimitiveMesh);

private:
	Size2 size;
	int subdivide_w;
	int subdivide_d;

protected:
	static void _bind_methods();
	virtual void _create_mesh_array(Array &p_arr) const;

public:
	void set_size(const Size2 &p_size);
	Size2 get_size() const;

	void set_subdivide_width(const int p_divisions);
	int get_subdivide_width() const;

	void set_subdivide_depth(const int p_divisions);
	int get_subdivide_depth() const;

	PlaneMesh();
};

/**
	A prism shapen, handy for ramps, triangles, etc.
*/
class PrismMesh : public PrimitiveMesh {

	GDCLASS(PrismMesh, PrimitiveMesh);

private:
	float left_to_right;
	Vector3 size;
	int subdivide_w;
	int subdivide_h;
	int subdivide_d;

protected:
	static void _bind_methods();
	virtual void _create_mesh_array(Array &p_arr) const;

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

	GDCLASS(QuadMesh, PrimitiveMesh)

private:
	Size2 size;

protected:
	static void _bind_methods();
	virtual void _create_mesh_array(Array &p_arr) const;

public:
	QuadMesh();

	void set_size(const Size2 &p_size);
	Size2 get_size() const;
};

/**
	A sphere..
*/
class SphereMesh : public PrimitiveMesh {

	GDCLASS(SphereMesh, PrimitiveMesh);

private:
	float radius;
	float height;
	int radial_segments;
	int rings;
	bool is_hemisphere;

protected:
	static void _bind_methods();
	virtual void _create_mesh_array(Array &p_arr) const;

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

#endif
