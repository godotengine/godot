/*************************************************************************/
/*  csg_tool.h                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef CSG_TOOL_H
#define CSG_TOOL_H

#include "csg.h"

#include "core/math/geometry_2d.h"
// #include "scene/resources/convex_polygon_shape_3d.h"
#include "scene/resources/concave_polygon_shape_3d.h"
#include "scene/resources/material.h"
#include "scene/resources/mesh.h"
#include "thirdparty/misc/mikktspace.h"

class CSGPrimitiveShape3D : public Resource {
	GDCLASS(CSGPrimitiveShape3D, Resource);

	bool invert_faces = false;
	bool smooth_faces = true;
	Ref<Material> material;

protected:
	CSGBrush brush;
	bool dirty = true;

	virtual void _update_brush() = 0;

	void _create_brush_from_arrays(const Vector<Vector3> &p_vertices, const Vector<Vector2> &p_uv, const Vector<bool> &p_smooth, const Vector<Ref<Material>> &p_materials);

	_ALWAYS_INLINE_ void _recreate_brush() {
		brush = CSGBrush();
	}

	static void _bind_methods();

public:
	CSGBrush get_brush();

	bool is_inverting_faces() const;
	void set_invert_faces(bool p_invert);

	bool get_smooth_faces() const;
	void set_smooth_faces(bool p_smooth_faces);

	Ref<Material> get_material() const;
	void set_material(const Ref<Material> &p_material);
};

class CSGMeshShape3D : public CSGPrimitiveShape3D {
	GDCLASS(CSGMeshShape3D, CSGPrimitiveShape3D);

	Ref<Mesh> mesh;

	virtual void _update_brush() override;

protected:
	static void _bind_methods();

public:
	Ref<Mesh> get_mesh() const;
	void set_mesh(const Ref<Mesh> &p_mesh);
};

class CSGBoxShape3D : public CSGPrimitiveShape3D {
	GDCLASS(CSGBoxShape3D, CSGPrimitiveShape3D);

	Vector3 size = Vector3(2, 2, 2);

	virtual void _update_brush() override;

protected:
	static void _bind_methods();

public:
	Vector3 get_size() const;
	void set_size(const Vector3 &p_size);
};

class CSGCylinderShape3D : public CSGPrimitiveShape3D {
	GDCLASS(CSGCylinderShape3D, CSGPrimitiveShape3D);

	float radius = 1.0;
	float height = 1.0;
	int sides = 8;
	bool cone = false;

	virtual void _update_brush() override;

protected:
	static void _bind_methods();

public:
	float get_radius() const;
	void set_radius(float p_radius);

	float get_height() const;
	void set_height(float p_height);

	int get_sides() const;
	void set_sides(int p_sides);

	bool is_cone() const;
	void set_cone(bool p_cone);
};

class CSGSphereShape3D : public CSGPrimitiveShape3D {
	GDCLASS(CSGSphereShape3D, CSGPrimitiveShape3D);

	float radius = 1.0;
	// Should they match SphereMesh3D?
	int radial_segments = 12;
	int rings = 6;

	virtual void _update_brush() override;

protected:
	static void _bind_methods();

public:
	float get_radius() const;
	void set_radius(float p_radius);

	int get_radial_segments() const;
	void set_radial_segments(int p_radial_segments);

	int get_rings() const;
	void set_rings(int p_rings);
};

class CSGTorusShape3D : public CSGPrimitiveShape3D {
	GDCLASS(CSGTorusShape3D, CSGPrimitiveShape3D);

	float inner_radius = 2.0;
	float outer_radius = 3.0;
	int sides = 8;
	int ring_sides = 6;

	virtual void _update_brush() override;

protected:
	static void _bind_methods();

public:
	float get_inner_radius() const;
	void set_inner_radius(float p_inner_radius);
	float get_outer_radius() const;
	void set_outer_radius(float p_outer_radius);
	int get_sides() const;
	void set_sides(int p_sides);
	int get_ring_sides() const;
	void set_ring_sides(int p_ring_sides);
};

// TODO: migrate this

class CSGPolygonShape3D : public CSGPrimitiveShape3D {
	GDCLASS(CSGPolygonShape3D, CSGPrimitiveShape3D);

public:
	enum Mode {
		MODE_DEPTH,
		MODE_SPIN,
		MODE_PATH
	};

	enum PathRotation {
		PATH_ROTATION_POLYGON,
		PATH_ROTATION_PATH,
		PATH_ROTATION_PATH_FOLLOW,
	};

private:
	Vector<Vector2> polygon;

	Mode mode = MODE_DEPTH;

	float depth = 1.0;

	float spin_degrees = 360.0;
	int spin_sides = 8;

	Ref<Curve3D> path_curve;
	float path_interval = 1.0;
	PathRotation path_rotation = PATH_ROTATION_PATH;
	Transform path_transform;

	bool path_continuous_u = false;
	bool path_joined = false;

	void _path_changed();
	void _path_exited();

	virtual void _update_brush() override;

protected:
	static void _bind_methods();
	virtual void _validate_property(PropertyInfo &property) const override;

public:
	void set_polygon(const Vector<Vector2> &p_polygon);
	Vector<Vector2> get_polygon() const;

	void set_mode(Mode p_mode);
	Mode get_mode() const;

	void set_depth(float p_depth);
	float get_depth() const;

	void set_spin_degrees(float p_spin_degrees);
	float get_spin_degrees() const;

	void set_spin_sides(int p_spin_sides);
	int get_spin_sides() const;

	void set_path_curve(Ref<Curve3D> &p_curve);
	Ref<Curve3D> get_path_curve() const;

	void set_path_interval(float p_interval);
	float get_path_interval() const;

	void set_path_rotation(PathRotation p_rotation);
	PathRotation get_path_rotation() const;

	void set_path_transform(Transform p_xform);
	Transform get_path_transform() const;

	void set_path_continuous_u(bool p_enable);
	bool is_path_continuous_u() const;

	void set_path_joined(bool p_enable);
	bool is_path_joined() const;
};

VARIANT_ENUM_CAST(CSGPolygonShape3D::Mode)
VARIANT_ENUM_CAST(CSGPolygonShape3D::PathRotation)

class CSGTool : public Reference {
	GDCLASS(CSGTool, Reference);

public:
	enum Operation {
		OPERATION_UNION = CSGBrushOperation::OPERATION_UNION,
		OPERATION_INTERSECTION = CSGBrushOperation::OPERATION_INTERSECTION,
		OPERATION_SUBTRACTION = CSGBrushOperation::OPERATION_SUBSTRACTION,
	};

private:
	CSGBrush brush;

	struct ShapeUpdateSurface {
		Vector<Vector3> vertices;
		Vector<Vector3> normals;
		Vector<Vector2> uvs;
		Vector<float> tans;
		Ref<Material> material;
		int last_added = 0;

		Vector3 *verticesw = nullptr;
		Vector3 *normalsw = nullptr;
		Vector2 *uvsw = nullptr;
		float *tansw = nullptr;
	};

	//mikktspace callbacks
	static int mikktGetNumFaces(const SMikkTSpaceContext *pContext);
	static int mikktGetNumVerticesOfFace(const SMikkTSpaceContext *pContext, const int iFace);
	static void mikktGetPosition(const SMikkTSpaceContext *pContext, float fvPosOut[], const int iFace, const int iVert);
	static void mikktGetNormal(const SMikkTSpaceContext *pContext, float fvNormOut[], const int iFace, const int iVert);
	static void mikktGetTexCoord(const SMikkTSpaceContext *pContext, float fvTexcOut[], const int iFace, const int iVert);
	static void mikktSetTSpaceDefault(const SMikkTSpaceContext *pContext, const float fvTangent[], const float fvBiTangent[], const float fMagS, const float fMagT,
			const tbool bIsOrientationPreserving, const int iFace, const int iVert);

protected:
	static void _bind_methods();

public:
	void add_brush(CSGBrush &p_brush, Operation p_operation = OPERATION_UNION, float p_vertex_snap = 0.001);
	Vector<Vector3> get_brush_faces() const;

	_ALWAYS_INLINE_ CSGBrush get_brush() {
		return brush;
	}

	void add_primitive(Ref<CSGPrimitiveShape3D> p_primitive, Operation p_operation = OPERATION_UNION, Transform p_xform = Transform(), float p_vertex_snap = 0.001);

	Ref<ConcavePolygonShape3D> create_trimesh_shape() const;
	// Ref<ConvexPolygonShape3D> create_convex_shape();

	Ref<ArrayMesh> commit(const Ref<ArrayMesh> &p_existing, bool p_generate_tangents = false);
	AABB get_aabb() const;
};

VARIANT_ENUM_CAST(CSGTool::Operation)

#endif // CSG_TOOL_H
