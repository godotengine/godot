/*************************************************************************/
/*  mesh.h                                                               */
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
#ifndef MESH_H
#define MESH_H

#include "resource.h"
#include "scene/resources/material.h"
#include "scene/resources/shape.h"
#include "servers/visual_server.h"
#include "triangle_mesh.h"
/**
	@author Juan Linietsky <reduzio@gmail.com>
*/

class Mesh : public Resource {
	GDCLASS(Mesh, Resource);

	mutable Ref<TriangleMesh> triangle_mesh; //cached
protected:
	void _clear_triangle_mesh() const;

	static void _bind_methods();

public:
	enum {

		NO_INDEX_ARRAY = VisualServer::NO_INDEX_ARRAY,
		ARRAY_WEIGHTS_SIZE = VisualServer::ARRAY_WEIGHTS_SIZE
	};

	enum ArrayType {

		ARRAY_VERTEX = VisualServer::ARRAY_VERTEX,
		ARRAY_NORMAL = VisualServer::ARRAY_NORMAL,
		ARRAY_TANGENT = VisualServer::ARRAY_TANGENT,
		ARRAY_COLOR = VisualServer::ARRAY_COLOR,
		ARRAY_TEX_UV = VisualServer::ARRAY_TEX_UV,
		ARRAY_TEX_UV2 = VisualServer::ARRAY_TEX_UV2,
		ARRAY_BONES = VisualServer::ARRAY_BONES,
		ARRAY_WEIGHTS = VisualServer::ARRAY_WEIGHTS,
		ARRAY_INDEX = VisualServer::ARRAY_INDEX,
		ARRAY_MAX = VisualServer::ARRAY_MAX

	};

	enum ArrayFormat {
		/* ARRAY FORMAT FLAGS */
		ARRAY_FORMAT_VERTEX = 1 << ARRAY_VERTEX, // mandatory
		ARRAY_FORMAT_NORMAL = 1 << ARRAY_NORMAL,
		ARRAY_FORMAT_TANGENT = 1 << ARRAY_TANGENT,
		ARRAY_FORMAT_COLOR = 1 << ARRAY_COLOR,
		ARRAY_FORMAT_TEX_UV = 1 << ARRAY_TEX_UV,
		ARRAY_FORMAT_TEX_UV2 = 1 << ARRAY_TEX_UV2,
		ARRAY_FORMAT_BONES = 1 << ARRAY_BONES,
		ARRAY_FORMAT_WEIGHTS = 1 << ARRAY_WEIGHTS,
		ARRAY_FORMAT_INDEX = 1 << ARRAY_INDEX,

		ARRAY_COMPRESS_BASE = (ARRAY_INDEX + 1),
		ARRAY_COMPRESS_VERTEX = 1 << (ARRAY_VERTEX + ARRAY_COMPRESS_BASE), // mandatory
		ARRAY_COMPRESS_NORMAL = 1 << (ARRAY_NORMAL + ARRAY_COMPRESS_BASE),
		ARRAY_COMPRESS_TANGENT = 1 << (ARRAY_TANGENT + ARRAY_COMPRESS_BASE),
		ARRAY_COMPRESS_COLOR = 1 << (ARRAY_COLOR + ARRAY_COMPRESS_BASE),
		ARRAY_COMPRESS_TEX_UV = 1 << (ARRAY_TEX_UV + ARRAY_COMPRESS_BASE),
		ARRAY_COMPRESS_TEX_UV2 = 1 << (ARRAY_TEX_UV2 + ARRAY_COMPRESS_BASE),
		ARRAY_COMPRESS_BONES = 1 << (ARRAY_BONES + ARRAY_COMPRESS_BASE),
		ARRAY_COMPRESS_WEIGHTS = 1 << (ARRAY_WEIGHTS + ARRAY_COMPRESS_BASE),
		ARRAY_COMPRESS_INDEX = 1 << (ARRAY_INDEX + ARRAY_COMPRESS_BASE),

		ARRAY_FLAG_USE_2D_VERTICES = ARRAY_COMPRESS_INDEX << 1,
		ARRAY_FLAG_USE_16_BIT_BONES = ARRAY_COMPRESS_INDEX << 2,

		ARRAY_COMPRESS_DEFAULT = ARRAY_COMPRESS_VERTEX | ARRAY_COMPRESS_NORMAL | ARRAY_COMPRESS_TANGENT | ARRAY_COMPRESS_COLOR | ARRAY_COMPRESS_TEX_UV | ARRAY_COMPRESS_TEX_UV2 | ARRAY_COMPRESS_WEIGHTS

	};

	enum PrimitiveType {
		PRIMITIVE_POINTS = VisualServer::PRIMITIVE_POINTS,
		PRIMITIVE_LINES = VisualServer::PRIMITIVE_LINES,
		PRIMITIVE_LINE_STRIP = VisualServer::PRIMITIVE_LINE_STRIP,
		PRIMITIVE_LINE_LOOP = VisualServer::PRIMITIVE_LINE_LOOP,
		PRIMITIVE_TRIANGLES = VisualServer::PRIMITIVE_TRIANGLES,
		PRIMITIVE_TRIANGLE_STRIP = VisualServer::PRIMITIVE_TRIANGLE_STRIP,
		PRIMITIVE_TRIANGLE_FAN = VisualServer::PRIMITIVE_TRIANGLE_FAN,
	};

	enum BlendShapeMode {

		BLEND_SHAPE_MODE_NORMALIZED = VS::BLEND_SHAPE_MODE_NORMALIZED,
		BLEND_SHAPE_MODE_RELATIVE = VS::BLEND_SHAPE_MODE_RELATIVE,
	};

	virtual int get_surface_count() const = 0;
	virtual int surface_get_array_len(int p_idx) const = 0;
	virtual int surface_get_array_index_len(int p_idx) const = 0;
	virtual Array surface_get_arrays(int p_surface) const = 0;
	virtual Array surface_get_blend_shape_arrays(int p_surface) const = 0;
	virtual uint32_t surface_get_format(int p_idx) const = 0;
	virtual PrimitiveType surface_get_primitive_type(int p_idx) const = 0;
	virtual Ref<Material> surface_get_material(int p_idx) const = 0;
	virtual int get_blend_shape_count() const = 0;
	virtual StringName get_blend_shape_name(int p_index) const = 0;

	PoolVector<Face3> get_faces() const;
	Ref<TriangleMesh> generate_triangle_mesh() const;

	Ref<Shape> create_trimesh_shape() const;
	Ref<Shape> create_convex_shape() const;

	Ref<Mesh> create_outline(float p_margin) const;

	virtual Rect3 get_aabb() const = 0;

	Mesh();
};

class ArrayMesh : public Mesh {

	GDCLASS(ArrayMesh, Mesh);
	RES_BASE_EXTENSION("mesh");

private:
	struct Surface {
		String name;
		Rect3 aabb;
		Ref<Material> material;
		bool is_2d;
	};
	Vector<Surface> surfaces;
	RID mesh;
	Rect3 aabb;
	BlendShapeMode blend_shape_mode;
	Vector<StringName> blend_shapes;
	Rect3 custom_aabb;

	void _recompute_aabb();

protected:
	virtual bool _is_generated() const { return false; }

	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

	static void _bind_methods();

public:
	void add_surface_from_arrays(PrimitiveType p_primitive, const Array &p_arrays, const Array &p_blend_shapes = Array(), uint32_t p_flags = ARRAY_COMPRESS_DEFAULT);
	void add_surface(uint32_t p_format, PrimitiveType p_primitive, const PoolVector<uint8_t> &p_array, int p_vertex_count, const PoolVector<uint8_t> &p_index_array, int p_index_count, const Rect3 &p_aabb, const Vector<PoolVector<uint8_t> > &p_blend_shapes = Vector<PoolVector<uint8_t> >(), const Vector<Rect3> &p_bone_aabbs = Vector<Rect3>());

	Array surface_get_arrays(int p_surface) const;
	Array surface_get_blend_shape_arrays(int p_surface) const;

	void add_blend_shape(const StringName &p_name);
	int get_blend_shape_count() const;
	StringName get_blend_shape_name(int p_index) const;
	void clear_blend_shapes();

	void set_blend_shape_mode(BlendShapeMode p_mode);
	BlendShapeMode get_blend_shape_mode() const;

	int get_surface_count() const;
	void surface_remove(int p_idx);

	void surface_set_custom_aabb(int p_idx, const Rect3 &p_aabb); //only recognized by driver

	int surface_get_array_len(int p_idx) const;
	int surface_get_array_index_len(int p_idx) const;
	uint32_t surface_get_format(int p_idx) const;
	PrimitiveType surface_get_primitive_type(int p_idx) const;
	bool surface_is_alpha_sorting_enabled(int p_idx) const;

	void surface_set_material(int p_idx, const Ref<Material> &p_material);
	Ref<Material> surface_get_material(int p_idx) const;

	void surface_set_name(int p_idx, const String &p_name);
	String surface_get_name(int p_idx) const;

	void add_surface_from_mesh_data(const Geometry::MeshData &p_mesh_data);

	void set_custom_aabb(const Rect3 &p_custom);
	Rect3 get_custom_aabb() const;

	Rect3 get_aabb() const;
	virtual RID get_rid() const;

	void center_geometry();
	void regen_normalmaps();

	ArrayMesh();

	~ArrayMesh();
};

VARIANT_ENUM_CAST(Mesh::ArrayType);
VARIANT_ENUM_CAST(Mesh::ArrayFormat);
VARIANT_ENUM_CAST(Mesh::PrimitiveType);
VARIANT_ENUM_CAST(Mesh::BlendShapeMode);

#endif
