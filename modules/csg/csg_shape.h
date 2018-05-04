#ifndef CSG_SHAPE_H
#define CSG_SHAPE_H

#define CSGJS_HEADER_ONLY

#include "csg.h"
#include "scene/3d/visual_instance.h"
#include "scene/resources/concave_polygon_shape.h"

class CSGShape : public VisualInstance {
	GDCLASS(CSGShape, VisualInstance);

public:
	enum Operation {
		OPERATION_UNION,
		OPERATION_INTERSECTION,
		OPERATION_SUBTRACTION,

	};

private:
	Operation operation;
	CSGShape *parent;

	CSGBrush *brush;

	AABB node_aabb;

	bool dirty;
	float snap;

	bool use_collision;
	Ref<ConcavePolygonShape> root_collision_shape;
	RID root_collision_instance;

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
		Ref<Material> material;
		int last_added;

		PoolVector<Vector3>::Write verticesw;
		PoolVector<Vector3>::Write normalsw;
		PoolVector<Vector2>::Write uvsw;
	};

	void _update_shape();

protected:
	void _notification(int p_what);
	virtual CSGBrush *_build_brush() = 0;
	void _make_dirty();

	static void _bind_methods();

	friend class CSGCombiner;
	CSGBrush *_get_brush();

	virtual void _validate_property(PropertyInfo &property) const;

public:
	void set_operation(Operation p_operation);
	Operation get_operation() const;

	virtual PoolVector<Vector3> get_brush_faces();

	virtual AABB get_aabb() const;
	virtual PoolVector<Face3> get_faces(uint32_t p_usage_flags) const;

	void set_use_collision(bool p_enable);
	bool is_using_collision() const;

	void set_snap(float p_snap);
	float get_snap() const;

	bool is_root_shape() const;
	CSGShape();
	~CSGShape();
};

VARIANT_ENUM_CAST(CSGShape::Operation)

class CSGCombiner : public CSGShape {
	GDCLASS(CSGCombiner, CSGShape)
private:
	virtual CSGBrush *_build_brush();

public:
	CSGCombiner();
};

class CSGPrimitive : public CSGShape {
	GDCLASS(CSGPrimitive, CSGShape)

private:
	bool invert_faces;

protected:
	CSGBrush *_create_brush_from_arrays(const PoolVector<Vector3> &p_vertices, const PoolVector<Vector2> &p_uv, const PoolVector<bool> &p_smooth, const PoolVector<Ref<Material> > &p_materials);
	static void _bind_methods();

public:
	void set_invert_faces(bool p_invert);
	bool is_inverting_faces();

	CSGPrimitive();
};

class CSGMesh : public CSGPrimitive {
	GDCLASS(CSGMesh, CSGPrimitive)

	virtual CSGBrush *_build_brush();

	Ref<Mesh> mesh;

	void _mesh_changed();

protected:
	static void _bind_methods();

public:
	void set_mesh(const Ref<Mesh> &p_mesh);
	Ref<Mesh> get_mesh();
};

class CSGSphere : public CSGPrimitive {

	GDCLASS(CSGSphere, CSGPrimitive)
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

	GDCLASS(CSGBox, CSGPrimitive)
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

	GDCLASS(CSGCylinder, CSGPrimitive)
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

	GDCLASS(CSGTorus, CSGPrimitive)
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

	GDCLASS(CSGPolygon, CSGPrimitive)

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
	virtual CSGBrush *_build_brush();

	Vector<Vector2> polygon;
	Ref<Material> material;

	Mode mode;

	float depth;

	float spin_degrees;
	int spin_sides;

	NodePath path_node;
	float path_interval;
	PathRotation path_rotation;

	Node *path_cache;

	bool smooth_faces;

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

	void set_spin_sides(int p_sides);
	int get_spin_sides() const;

	void set_path_node(const NodePath &p_path);
	NodePath get_path_node() const;

	void set_path_interval(float p_interval);
	float get_path_interval() const;

	void set_path_rotation(PathRotation p_rotation);
	PathRotation get_path_rotation() const;

	void set_smooth_faces(bool p_smooth_faces);
	bool get_smooth_faces() const;

	void set_material(const Ref<Material> &p_material);
	Ref<Material> get_material() const;

	CSGPolygon();
};

VARIANT_ENUM_CAST(CSGPolygon::Mode)
VARIANT_ENUM_CAST(CSGPolygon::PathRotation)

#endif // CSG_SHAPE_H
