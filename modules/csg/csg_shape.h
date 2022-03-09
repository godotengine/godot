/*************************************************************************/
/*  csg_shape.h                                                          */
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

#ifndef CSG_SHAPE_H
#define CSG_SHAPE_H

#define CSGJS_HEADER_ONLY

#include "csg.h"
#include "scene/3d/path.h"
#include "scene/3d/visual_instance.h"
#include "scene/resources/concave_polygon_shape.h"
#include "thirdparty/misc/mikktspace.h"

class CSGShape : public GeometryInstance {
	GDCLASS(CSGShape, GeometryInstance);

public:
	enum Operation {
		OPERATION_UNION,
		OPERATION_INTERSECTION,
		OPERATION_SUBTRACTION,

	};

private:
	Operation operation;
	CSGShape *parent_shape;

	CSGBrush *brush;

	AABB node_aabb;

	bool dirty;
	bool last_visible = false;
	float snap;

	bool use_collision;
	uint32_t collision_layer;
	uint32_t collision_mask;
	Ref<ConcavePolygonShape> root_collision_shape;
	RID root_collision_instance;

	bool calculate_tangents;

	Ref<ArrayMesh> root_mesh;

	struct Vector3Hasher {
		_ALWAYS_INLINE_ uint32_t hash(const Vector3 &p_vec3) const {
			uint32_t h = hash_djb2_one_float(p_vec3.x);
			h = hash_djb2_one_float(p_vec3.y, h);
			h = hash_djb2_one_float(p_vec3.z, h);
			return h;
		}
	};

	struct ShapeUpdateSurface {
		PoolVector<Vector3> vertices;
		PoolVector<Vector3> normals;
		PoolVector<Vector2> uvs;
		PoolVector<float> tans;
		Ref<Material> material;
		int last_added;

		PoolVector<Vector3>::Write verticesw;
		PoolVector<Vector3>::Write normalsw;
		PoolVector<Vector2>::Write uvsw;
		PoolVector<float>::Write tansw;
	};

	//mikktspace callbacks
	static int mikktGetNumFaces(const SMikkTSpaceContext *pContext);
	static int mikktGetNumVerticesOfFace(const SMikkTSpaceContext *pContext, const int iFace);
	static void mikktGetPosition(const SMikkTSpaceContext *pContext, float fvPosOut[], const int iFace, const int iVert);
	static void mikktGetNormal(const SMikkTSpaceContext *pContext, float fvNormOut[], const int iFace, const int iVert);
	static void mikktGetTexCoord(const SMikkTSpaceContext *pContext, float fvTexcOut[], const int iFace, const int iVert);
	static void mikktSetTSpaceDefault(const SMikkTSpaceContext *pContext, const float fvTangent[], const float fvBiTangent[], const float fMagS, const float fMagT,
			const tbool bIsOrientationPreserving, const int iFace, const int iVert);

	void _update_shape();
	void _update_collision_faces();

protected:
	void _notification(int p_what);
	virtual CSGBrush *_build_brush() = 0;
	void _make_dirty(bool p_parent_removing = false);

	static void _bind_methods();

	friend class CSGCombiner;
	CSGBrush *_get_brush();

	virtual void _validate_property(PropertyInfo &property) const;

public:
	Array get_meshes() const;

	void force_update_shape();

	void set_operation(Operation p_operation);
	Operation get_operation() const;

	virtual PoolVector<Vector3> get_brush_faces();

	virtual AABB get_aabb() const;
	virtual PoolVector<Face3> get_faces(uint32_t p_usage_flags) const;

	void set_use_collision(bool p_enable);
	bool is_using_collision() const;

	void set_collision_layer(uint32_t p_layer);
	uint32_t get_collision_layer() const;

	void set_collision_mask(uint32_t p_mask);
	uint32_t get_collision_mask() const;

	void set_collision_layer_bit(int p_bit, bool p_value);
	bool get_collision_layer_bit(int p_bit) const;

	void set_collision_mask_bit(int p_bit, bool p_value);
	bool get_collision_mask_bit(int p_bit) const;

	void set_snap(float p_snap);
	float get_snap() const;

	void set_calculate_tangents(bool p_calculate_tangents);
	bool is_calculating_tangents() const;

	bool is_root_shape() const;
	CSGShape();
	~CSGShape();
};

VARIANT_ENUM_CAST(CSGShape::Operation)

class CSGCombiner : public CSGShape {
	GDCLASS(CSGCombiner, CSGShape);

private:
	virtual CSGBrush *_build_brush();

public:
	CSGCombiner();
};

class CSGPrimitive : public CSGShape {
	GDCLASS(CSGPrimitive, CSGShape);

protected:
	bool invert_faces;
	CSGBrush *_create_brush_from_arrays(const PoolVector<Vector3> &p_vertices, const PoolVector<Vector2> &p_uv, const PoolVector<bool> &p_smooth, const PoolVector<Ref<Material>> &p_materials);
	static void _bind_methods();

public:
	void set_invert_faces(bool p_invert);
	bool is_inverting_faces();

	CSGPrimitive();
};

class CSGMesh : public CSGPrimitive {
	GDCLASS(CSGMesh, CSGPrimitive);

	virtual CSGBrush *_build_brush();

	Ref<Mesh> mesh;
	Ref<Material> material;

	void _mesh_changed();

protected:
	static void _bind_methods();

public:
	void set_mesh(const Ref<Mesh> &p_mesh);
	Ref<Mesh> get_mesh();

	void set_material(const Ref<Material> &p_material);
	Ref<Material> get_material() const;
};

class CSGSphere : public CSGPrimitive {
	GDCLASS(CSGSphere, CSGPrimitive);
	virtual CSGBrush *_build_brush();

	Ref<Material> material;
	bool smooth_faces;
	float radius;
	int radial_segments;
	int rings;

protected:
	static void _bind_methods();

public:
	void set_radius(const float p_radius);
	float get_radius() const;

	void set_radial_segments(const int p_radial_segments);
	int get_radial_segments() const;

	void set_rings(const int p_rings);
	int get_rings() const;

	void set_material(const Ref<Material> &p_material);
	Ref<Material> get_material() const;

	void set_smooth_faces(bool p_smooth_faces);
	bool get_smooth_faces() const;

	CSGSphere();
};

class CSGBox : public CSGPrimitive {
	GDCLASS(CSGBox, CSGPrimitive);
	virtual CSGBrush *_build_brush();

	Ref<Material> material;
	float width;
	float height;
	float depth;

protected:
	static void _bind_methods();

public:
	void set_width(const float p_width);
	float get_width() const;

	void set_height(const float p_height);
	float get_height() const;

	void set_depth(const float p_depth);
	float get_depth() const;

	void set_material(const Ref<Material> &p_material);
	Ref<Material> get_material() const;

	CSGBox();
};

class CSGCylinder : public CSGPrimitive {
	GDCLASS(CSGCylinder, CSGPrimitive);
	virtual CSGBrush *_build_brush();

	Ref<Material> material;
	float radius;
	float height;
	int sides;
	bool cone;
	bool smooth_faces;

protected:
	static void _bind_methods();

public:
	void set_radius(const float p_radius);
	float get_radius() const;

	void set_height(const float p_height);
	float get_height() const;

	void set_sides(const int p_sides);
	int get_sides() const;

	void set_cone(const bool p_cone);
	bool is_cone() const;

	void set_smooth_faces(bool p_smooth_faces);
	bool get_smooth_faces() const;

	void set_material(const Ref<Material> &p_material);
	Ref<Material> get_material() const;

	CSGCylinder();
};

class CSGTorus : public CSGPrimitive {
	GDCLASS(CSGTorus, CSGPrimitive);
	virtual CSGBrush *_build_brush();

	Ref<Material> material;
	float inner_radius;
	float outer_radius;
	int sides;
	int ring_sides;
	bool smooth_faces;

protected:
	static void _bind_methods();

public:
	void set_inner_radius(const float p_inner_radius);
	float get_inner_radius() const;

	void set_outer_radius(const float p_outer_radius);
	float get_outer_radius() const;

	void set_sides(const int p_sides);
	int get_sides() const;

	void set_ring_sides(const int p_ring_sides);
	int get_ring_sides() const;

	void set_smooth_faces(bool p_smooth_faces);
	bool get_smooth_faces() const;

	void set_material(const Ref<Material> &p_material);
	Ref<Material> get_material() const;

	CSGTorus();
};

class CSGPolygon : public CSGPrimitive {
	GDCLASS(CSGPolygon, CSGPrimitive);

public:
	enum Mode {
		MODE_DEPTH,
		MODE_SPIN,
		MODE_PATH
	};

	enum PathIntervalType {
		PATH_INTERVAL_DISTANCE,
		PATH_INTERVAL_SUBDIVIDE
	};

	enum PathRotation {
		PATH_ROTATION_POLYGON,
		PATH_ROTATION_PATH,
		PATH_ROTATION_PATH_FOLLOW,
	};

private:
	virtual CSGBrush *_build_brush();

	Vector<Vector2> polygon;
	Ref<Material> material;

	Mode mode;

	float depth;

	float spin_degrees;
	int spin_sides;

	NodePath path_node;
	PathIntervalType path_interval_type;
	float path_interval;
	float path_simplify_angle;
	PathRotation path_rotation;
	bool path_local;

	Path *path;

	bool smooth_faces;
	bool path_continuous_u;
	real_t path_u_distance;
	bool path_joined;

	bool _is_editable_3d_polygon() const;
	bool _has_editable_3d_polygon_no_depth() const;

	void _path_changed();
	void _path_exited();

protected:
	static void _bind_methods();
	virtual void _validate_property(PropertyInfo &property) const;
	void _notification(int p_what);

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

	void set_path_node(const NodePath &p_path);
	NodePath get_path_node() const;

	void set_path_interval_type(PathIntervalType p_interval_type);
	PathIntervalType get_path_interval_type() const;

	void set_path_interval(float p_interval);
	float get_path_interval() const;

	void set_path_simplify_angle(float p_angle);
	float get_path_simplify_angle() const;

	void set_path_rotation(PathRotation p_rotation);
	PathRotation get_path_rotation() const;

	void set_path_local(bool p_enable);
	bool is_path_local() const;

	void set_path_continuous_u(bool p_enable);
	bool is_path_continuous_u() const;

	void set_path_u_distance(real_t p_path_u_distance);
	real_t get_path_u_distance() const;

	void set_path_joined(bool p_enable);
	bool is_path_joined() const;

	void set_smooth_faces(bool p_smooth_faces);
	bool get_smooth_faces() const;

	void set_material(const Ref<Material> &p_material);
	Ref<Material> get_material() const;

	CSGPolygon();
};

VARIANT_ENUM_CAST(CSGPolygon::Mode)
VARIANT_ENUM_CAST(CSGPolygon::PathRotation)
VARIANT_ENUM_CAST(CSGPolygon::PathIntervalType)

#endif // CSG_SHAPE_H
