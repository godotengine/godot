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

#include "scene/resources/font.h"
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

	Ref<Material> material;
	bool flip_faces;

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
	virtual void surface_set_material(int p_idx, const Ref<Material> &p_material);
	virtual Ref<Material> surface_get_material(int p_idx) const;
	virtual int get_blend_shape_count() const;
	virtual StringName get_blend_shape_name(int p_index) const;
	virtual void set_blend_shape_name(int p_index, const StringName &p_name);
	virtual AABB get_aabb() const;
	virtual RID get_rid() const;

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
	static constexpr int default_radial_segments = 64;
	static constexpr int default_rings = 8;

private:
	float radius;
	float mid_height;
	int radial_segments;
	int rings;

protected:
	static void _bind_methods();
	virtual void _create_mesh_array(Array &p_arr) const;

public:
	static void create_mesh_array(Array &p_arr, float radius, float mid_height, int radial_segments = default_radial_segments, int rings = default_rings);

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
	static constexpr int default_subdivide_w = 0;
	static constexpr int default_subdivide_h = 0;
	static constexpr int default_subdivide_d = 0;

private:
	Vector3 size;
	int subdivide_w;
	int subdivide_h;
	int subdivide_d;

protected:
	static void _bind_methods();
	virtual void _create_mesh_array(Array &p_arr) const;

public:
	static void create_mesh_array(Array &p_arr, Vector3 size, int subdivide_w = default_subdivide_w, int subdivide_h = default_subdivide_h, int subdivide_d = default_subdivide_d);

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
	static constexpr int default_radial_segments = 64;
	static constexpr int default_rings = 4;

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
	static void create_mesh_array(Array &p_arr, float top_radius, float bottom_radius, float height, int radial_segments = default_radial_segments, int rings = default_rings);

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
	Size2 size;
	int subdivide_w;
	int subdivide_d;
	Vector3 center_offset;

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
	GDCLASS(QuadMesh, PrimitiveMesh);

private:
	Size2 size;
	Vector3 center_offset;

protected:
	static void _bind_methods();
	virtual void _create_mesh_array(Array &p_arr) const;

public:
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
	static constexpr int default_radial_segments = 64;
	static constexpr int default_rings = 32;
	static constexpr bool default_is_hemisphere = false;

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
	static void create_mesh_array(Array &p_arr, float radius, float height, int radial_segments = default_radial_segments, int rings = default_rings, bool is_hemisphere = default_is_hemisphere);

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
	Big donut
*/
class TorusMesh : public PrimitiveMesh {
	GDCLASS(TorusMesh, PrimitiveMesh);

private:
	float inner_radius = 0.5;
	float outer_radius = 1.0;
	int rings = 64;
	int ring_segments = 32;

protected:
	static void _bind_methods();
	virtual void _create_mesh_array(Array &p_arr) const;

public:
	void set_inner_radius(const float p_inner_radius);
	float get_inner_radius() const;

	void set_outer_radius(const float p_outer_radius);
	float get_outer_radius() const;

	void set_rings(const int p_rings);
	int get_rings() const;

	void set_ring_segments(const int p_ring_segments);
	int get_ring_segments() const;

	TorusMesh();
};

/**
	A single point for use in particle systems
*/

class PointMesh : public PrimitiveMesh {
	GDCLASS(PointMesh, PrimitiveMesh)

protected:
	virtual void _create_mesh_array(Array &p_arr) const;

public:
	PointMesh();
};

/**
	Text...
*/

class TextMesh : public PrimitiveMesh {
	GDCLASS(TextMesh, PrimitiveMesh);

public:
	enum Align {

		ALIGN_LEFT,
		ALIGN_CENTER,
		ALIGN_RIGHT
	};

private:
	struct ContourPoint {
		Vector2 point;
		bool sharp = false;

		ContourPoint(){};
		ContourPoint(const Vector2 &p_pt, bool p_sharp) {
			point = p_pt;
			sharp = p_sharp;
		};
	};
	struct ContourInfo {
		real_t length = 0.0;
		bool ccw = true;
		ContourInfo(){};
		ContourInfo(real_t p_len, bool p_ccw) {
			length = p_len;
			ccw = p_ccw;
		}
	};
	struct GlyphMeshData {
		Vector<Vector2> triangles;
		Vector<Vector<ContourPoint>> contours;
		Vector<ContourInfo> contours_info;
		Vector2 min_p = Vector2(INFINITY, INFINITY);
		Vector2 max_p = Vector2(-INFINITY, -INFINITY);
	};
	mutable HashMap<uint32_t, GlyphMeshData> cache;

	String text;
	String xl_text;

	Ref<Font> font_override;

	Align horizontal_alignment = ALIGN_CENTER;
	bool uppercase = false;

	real_t depth = 0.05;
	real_t pixel_size = 0.01;
	real_t curve_step = 0.5;

	mutable bool dirty_cache = true;

	void _generate_glyph_mesh_data(uint32_t p_utf32_char, const Ref<Font> &p_font, CharType p_char, CharType p_next) const;
	void _font_changed();

protected:
	static void _bind_methods();
	void _notification(int p_what);

	virtual void _create_mesh_array(Array &p_arr) const;

public:
	TextMesh();
	~TextMesh();

	void set_horizontal_alignment(Align p_alignment);
	Align get_horizontal_alignment() const;

	void set_text(const String &p_string);
	String get_text() const;

	void set_font(const Ref<Font> &p_font);
	Ref<Font> get_font() const;
	Ref<Font> _get_font_or_default() const;

	void set_uppercase(bool p_uppercase);
	bool is_uppercase() const;

	void set_depth(real_t p_depth);
	real_t get_depth() const;

	void set_curve_step(real_t p_step);
	real_t get_curve_step() const;

	void set_pixel_size(real_t p_amount);
	real_t get_pixel_size() const;
};

VARIANT_ENUM_CAST(TextMesh::Align);

#endif // PRIMITIVE_MESHES_H
