/*************************************************************************/
/*  shape_3d_sw.h                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef SHAPE_SW_H
#define SHAPE_SW_H

#include "core/math/geometry_3d.h"
#include "servers/physics_server_3d.h"
/*

SHAPE_LINE, ///< plane:"plane"
SHAPE_SEGMENT, ///< real_t:"length"
SHAPE_CIRCLE, ///< real_t:"radius"
SHAPE_RECTANGLE, ///< vec3:"extents"
SHAPE_CONVEX_POLYGON, ///< array of planes:"planes"
SHAPE_CONCAVE_POLYGON, ///< Vector3 array:"triangles" , or Dictionary with "indices" (int array) and "triangles" (Vector3 array)
SHAPE_CUSTOM, ///< Server-Implementation based custom shape, calling shape_create() with this value will result in an error

*/

class Shape3DSW;

class ShapeOwner3DSW {
public:
	virtual void _shape_changed() = 0;
	virtual void remove_shape(Shape3DSW *p_shape) = 0;

	virtual ~ShapeOwner3DSW() {}
};

class Shape3DSW {
	RID self;
	AABB aabb;
	bool configured;
	real_t custom_bias;

	Map<ShapeOwner3DSW *, int> owners;

protected:
	void configure(const AABB &p_aabb);

public:
	enum {
		MAX_SUPPORTS = 8
	};

	virtual real_t get_area() const { return aabb.get_area(); }

	_FORCE_INLINE_ void set_self(const RID &p_self) { self = p_self; }
	_FORCE_INLINE_ RID get_self() const { return self; }

	virtual PhysicsServer3D::ShapeType get_type() const = 0;

	_FORCE_INLINE_ AABB get_aabb() const { return aabb; }
	_FORCE_INLINE_ bool is_configured() const { return configured; }

	virtual bool is_concave() const { return false; }

	virtual void project_range(const Vector3 &p_normal, const Transform &p_transform, real_t &r_min, real_t &r_max) const = 0;
	virtual Vector3 get_support(const Vector3 &p_normal) const;
	virtual void get_supports(const Vector3 &p_normal, int p_max, Vector3 *r_supports, int &r_amount) const = 0;
	virtual Vector3 get_closest_point_to(const Vector3 &p_point) const = 0;
	virtual bool intersect_segment(const Vector3 &p_begin, const Vector3 &p_end, Vector3 &r_point, Vector3 &r_normal) const = 0;
	virtual bool intersect_point(const Vector3 &p_point) const = 0;
	virtual Vector3 get_moment_of_inertia(real_t p_mass) const = 0;

	virtual void set_data(const Variant &p_data) = 0;
	virtual Variant get_data() const = 0;

	_FORCE_INLINE_ void set_custom_bias(real_t p_bias) { custom_bias = p_bias; }
	_FORCE_INLINE_ real_t get_custom_bias() const { return custom_bias; }

	void add_owner(ShapeOwner3DSW *p_owner);
	void remove_owner(ShapeOwner3DSW *p_owner);
	bool is_owner(ShapeOwner3DSW *p_owner) const;
	const Map<ShapeOwner3DSW *, int> &get_owners() const;

	Shape3DSW();
	virtual ~Shape3DSW();
};

class ConcaveShape3DSW : public Shape3DSW {
public:
	virtual bool is_concave() const { return true; }
	typedef void (*Callback)(void *p_userdata, Shape3DSW *p_convex);
	virtual void get_supports(const Vector3 &p_normal, int p_max, Vector3 *r_supports, int &r_amount) const { r_amount = 0; }

	virtual void cull(const AABB &p_local_aabb, Callback p_callback, void *p_userdata) const = 0;

	ConcaveShape3DSW() {}
};

class PlaneShape3DSW : public Shape3DSW {
	Plane plane;

	void _setup(const Plane &p_plane);

public:
	Plane get_plane() const;

	virtual real_t get_area() const { return Math_INF; }
	virtual PhysicsServer3D::ShapeType get_type() const { return PhysicsServer3D::SHAPE_PLANE; }
	virtual void project_range(const Vector3 &p_normal, const Transform &p_transform, real_t &r_min, real_t &r_max) const;
	virtual Vector3 get_support(const Vector3 &p_normal) const;
	virtual void get_supports(const Vector3 &p_normal, int p_max, Vector3 *r_supports, int &r_amount) const { r_amount = 0; }

	virtual bool intersect_segment(const Vector3 &p_begin, const Vector3 &p_end, Vector3 &r_result, Vector3 &r_normal) const;
	virtual bool intersect_point(const Vector3 &p_point) const;
	virtual Vector3 get_closest_point_to(const Vector3 &p_point) const;
	virtual Vector3 get_moment_of_inertia(real_t p_mass) const;

	virtual void set_data(const Variant &p_data);
	virtual Variant get_data() const;

	PlaneShape3DSW();
};

class RayShape3DSW : public Shape3DSW {
	real_t length;
	bool slips_on_slope;

	void _setup(real_t p_length, bool p_slips_on_slope);

public:
	real_t get_length() const;
	bool get_slips_on_slope() const;

	virtual real_t get_area() const { return 0.0; }
	virtual PhysicsServer3D::ShapeType get_type() const { return PhysicsServer3D::SHAPE_RAY; }
	virtual void project_range(const Vector3 &p_normal, const Transform &p_transform, real_t &r_min, real_t &r_max) const;
	virtual Vector3 get_support(const Vector3 &p_normal) const;
	virtual void get_supports(const Vector3 &p_normal, int p_max, Vector3 *r_supports, int &r_amount) const;

	virtual bool intersect_segment(const Vector3 &p_begin, const Vector3 &p_end, Vector3 &r_result, Vector3 &r_normal) const;
	virtual bool intersect_point(const Vector3 &p_point) const;
	virtual Vector3 get_closest_point_to(const Vector3 &p_point) const;

	virtual Vector3 get_moment_of_inertia(real_t p_mass) const;

	virtual void set_data(const Variant &p_data);
	virtual Variant get_data() const;

	RayShape3DSW();
};

class SphereShape3DSW : public Shape3DSW {
	real_t radius;

	void _setup(real_t p_radius);

public:
	real_t get_radius() const;

	virtual real_t get_area() const { return 4.0 / 3.0 * Math_PI * radius * radius * radius; }

	virtual PhysicsServer3D::ShapeType get_type() const { return PhysicsServer3D::SHAPE_SPHERE; }

	virtual void project_range(const Vector3 &p_normal, const Transform &p_transform, real_t &r_min, real_t &r_max) const;
	virtual Vector3 get_support(const Vector3 &p_normal) const;
	virtual void get_supports(const Vector3 &p_normal, int p_max, Vector3 *r_supports, int &r_amount) const;
	virtual bool intersect_segment(const Vector3 &p_begin, const Vector3 &p_end, Vector3 &r_result, Vector3 &r_normal) const;
	virtual bool intersect_point(const Vector3 &p_point) const;
	virtual Vector3 get_closest_point_to(const Vector3 &p_point) const;

	virtual Vector3 get_moment_of_inertia(real_t p_mass) const;

	virtual void set_data(const Variant &p_data);
	virtual Variant get_data() const;

	SphereShape3DSW();
};

class BoxShape3DSW : public Shape3DSW {
	Vector3 half_extents;
	void _setup(const Vector3 &p_half_extents);

public:
	_FORCE_INLINE_ Vector3 get_half_extents() const { return half_extents; }
	virtual real_t get_area() const { return 8 * half_extents.x * half_extents.y * half_extents.z; }

	virtual PhysicsServer3D::ShapeType get_type() const { return PhysicsServer3D::SHAPE_BOX; }

	virtual void project_range(const Vector3 &p_normal, const Transform &p_transform, real_t &r_min, real_t &r_max) const;
	virtual Vector3 get_support(const Vector3 &p_normal) const;
	virtual void get_supports(const Vector3 &p_normal, int p_max, Vector3 *r_supports, int &r_amount) const;
	virtual bool intersect_segment(const Vector3 &p_begin, const Vector3 &p_end, Vector3 &r_result, Vector3 &r_normal) const;
	virtual bool intersect_point(const Vector3 &p_point) const;
	virtual Vector3 get_closest_point_to(const Vector3 &p_point) const;

	virtual Vector3 get_moment_of_inertia(real_t p_mass) const;

	virtual void set_data(const Variant &p_data);
	virtual Variant get_data() const;

	BoxShape3DSW();
};

class CapsuleShape3DSW : public Shape3DSW {
	real_t height;
	real_t radius;

	void _setup(real_t p_height, real_t p_radius);

public:
	_FORCE_INLINE_ real_t get_height() const { return height; }
	_FORCE_INLINE_ real_t get_radius() const { return radius; }

	virtual real_t get_area() const { return 4.0 / 3.0 * Math_PI * radius * radius * radius + height * Math_PI * radius * radius; }

	virtual PhysicsServer3D::ShapeType get_type() const { return PhysicsServer3D::SHAPE_CAPSULE; }

	virtual void project_range(const Vector3 &p_normal, const Transform &p_transform, real_t &r_min, real_t &r_max) const;
	virtual Vector3 get_support(const Vector3 &p_normal) const;
	virtual void get_supports(const Vector3 &p_normal, int p_max, Vector3 *r_supports, int &r_amount) const;
	virtual bool intersect_segment(const Vector3 &p_begin, const Vector3 &p_end, Vector3 &r_result, Vector3 &r_normal) const;
	virtual bool intersect_point(const Vector3 &p_point) const;
	virtual Vector3 get_closest_point_to(const Vector3 &p_point) const;

	virtual Vector3 get_moment_of_inertia(real_t p_mass) const;

	virtual void set_data(const Variant &p_data);
	virtual Variant get_data() const;

	CapsuleShape3DSW();
};

struct ConvexPolygonShape3DSW : public Shape3DSW {
	Geometry3D::MeshData mesh;

	void _setup(const Vector<Vector3> &p_vertices);

public:
	const Geometry3D::MeshData &get_mesh() const { return mesh; }

	virtual PhysicsServer3D::ShapeType get_type() const { return PhysicsServer3D::SHAPE_CONVEX_POLYGON; }

	virtual void project_range(const Vector3 &p_normal, const Transform &p_transform, real_t &r_min, real_t &r_max) const;
	virtual Vector3 get_support(const Vector3 &p_normal) const;
	virtual void get_supports(const Vector3 &p_normal, int p_max, Vector3 *r_supports, int &r_amount) const;
	virtual bool intersect_segment(const Vector3 &p_begin, const Vector3 &p_end, Vector3 &r_result, Vector3 &r_normal) const;
	virtual bool intersect_point(const Vector3 &p_point) const;
	virtual Vector3 get_closest_point_to(const Vector3 &p_point) const;

	virtual Vector3 get_moment_of_inertia(real_t p_mass) const;

	virtual void set_data(const Variant &p_data);
	virtual Variant get_data() const;

	ConvexPolygonShape3DSW();
};

struct _VolumeSW_BVH;
struct FaceShape3DSW;

struct ConcavePolygonShape3DSW : public ConcaveShape3DSW {
	// always a trimesh

	struct Face {
		Vector3 normal;
		int indices[3];
	};

	Vector<Face> faces;
	Vector<Vector3> vertices;

	struct BVH {
		AABB aabb;
		int left;
		int right;

		int face_index;
	};

	Vector<BVH> bvh;

	struct _CullParams {
		AABB aabb;
		Callback callback;
		void *userdata;
		const Face *faces;
		const Vector3 *vertices;
		const BVH *bvh;
		FaceShape3DSW *face;
	};

	struct _SegmentCullParams {
		Vector3 from;
		Vector3 to;
		const Face *faces;
		const Vector3 *vertices;
		const BVH *bvh;
		Vector3 dir;

		Vector3 result;
		Vector3 normal;
		real_t min_d;
		int collisions;
	};

	void _cull_segment(int p_idx, _SegmentCullParams *p_params) const;
	void _cull(int p_idx, _CullParams *p_params) const;

	void _fill_bvh(_VolumeSW_BVH *p_bvh_tree, BVH *p_bvh_array, int &p_idx);

	void _setup(Vector<Vector3> p_faces);

public:
	Vector<Vector3> get_faces() const;

	virtual PhysicsServer3D::ShapeType get_type() const { return PhysicsServer3D::SHAPE_CONCAVE_POLYGON; }

	virtual void project_range(const Vector3 &p_normal, const Transform &p_transform, real_t &r_min, real_t &r_max) const;
	virtual Vector3 get_support(const Vector3 &p_normal) const;

	virtual bool intersect_segment(const Vector3 &p_begin, const Vector3 &p_end, Vector3 &r_result, Vector3 &r_normal) const;
	virtual bool intersect_point(const Vector3 &p_point) const;
	virtual Vector3 get_closest_point_to(const Vector3 &p_point) const;

	virtual void cull(const AABB &p_local_aabb, Callback p_callback, void *p_userdata) const;

	virtual Vector3 get_moment_of_inertia(real_t p_mass) const;

	virtual void set_data(const Variant &p_data);
	virtual Variant get_data() const;

	ConcavePolygonShape3DSW();
};

struct HeightMapShape3DSW : public ConcaveShape3DSW {
	Vector<real_t> heights;
	int width;
	int depth;
	real_t cell_size;

	//void _cull_segment(int p_idx,_SegmentCullParams *p_params) const;
	//void _cull(int p_idx,_CullParams *p_params) const;

	void _setup(Vector<real_t> p_heights, int p_width, int p_depth, real_t p_cell_size);

public:
	Vector<real_t> get_heights() const;
	int get_width() const;
	int get_depth() const;
	real_t get_cell_size() const;

	virtual PhysicsServer3D::ShapeType get_type() const { return PhysicsServer3D::SHAPE_HEIGHTMAP; }

	virtual void project_range(const Vector3 &p_normal, const Transform &p_transform, real_t &r_min, real_t &r_max) const;
	virtual Vector3 get_support(const Vector3 &p_normal) const;
	virtual bool intersect_segment(const Vector3 &p_begin, const Vector3 &p_end, Vector3 &r_point, Vector3 &r_normal) const;
	virtual bool intersect_point(const Vector3 &p_point) const;

	virtual Vector3 get_closest_point_to(const Vector3 &p_point) const;
	virtual void cull(const AABB &p_local_aabb, Callback p_callback, void *p_userdata) const;

	virtual Vector3 get_moment_of_inertia(real_t p_mass) const;

	virtual void set_data(const Variant &p_data);
	virtual Variant get_data() const;

	HeightMapShape3DSW();
};

//used internally
struct FaceShape3DSW : public Shape3DSW {
	Vector3 normal; //cache
	Vector3 vertex[3];

	virtual PhysicsServer3D::ShapeType get_type() const { return PhysicsServer3D::SHAPE_CONCAVE_POLYGON; }

	const Vector3 &get_vertex(int p_idx) const { return vertex[p_idx]; }

	void project_range(const Vector3 &p_normal, const Transform &p_transform, real_t &r_min, real_t &r_max) const;
	Vector3 get_support(const Vector3 &p_normal) const;
	virtual void get_supports(const Vector3 &p_normal, int p_max, Vector3 *r_supports, int &r_amount) const;
	bool intersect_segment(const Vector3 &p_begin, const Vector3 &p_end, Vector3 &r_result, Vector3 &r_normal) const;
	virtual bool intersect_point(const Vector3 &p_point) const;
	virtual Vector3 get_closest_point_to(const Vector3 &p_point) const;

	Vector3 get_moment_of_inertia(real_t p_mass) const;

	virtual void set_data(const Variant &p_data) {}
	virtual Variant get_data() const { return Variant(); }

	FaceShape3DSW();
};

struct MotionShape3DSW : public Shape3DSW {
	Shape3DSW *shape;
	Vector3 motion;

	virtual PhysicsServer3D::ShapeType get_type() const { return PhysicsServer3D::SHAPE_CONVEX_POLYGON; }

	void project_range(const Vector3 &p_normal, const Transform &p_transform, real_t &r_min, real_t &r_max) const {
		Vector3 cast = p_transform.basis.xform(motion);
		real_t mina, maxa;
		real_t minb, maxb;
		Transform ofsb = p_transform;
		ofsb.origin += cast;
		shape->project_range(p_normal, p_transform, mina, maxa);
		shape->project_range(p_normal, ofsb, minb, maxb);
		r_min = MIN(mina, minb);
		r_max = MAX(maxa, maxb);
	}

	Vector3 get_support(const Vector3 &p_normal) const {
		Vector3 support = shape->get_support(p_normal);
		if (p_normal.dot(motion) > 0) {
			support += motion;
		}
		return support;
	}
	virtual void get_supports(const Vector3 &p_normal, int p_max, Vector3 *r_supports, int &r_amount) const { r_amount = 0; }
	bool intersect_segment(const Vector3 &p_begin, const Vector3 &p_end, Vector3 &r_result, Vector3 &r_normal) const { return false; }
	virtual bool intersect_point(const Vector3 &p_point) const { return false; }
	virtual Vector3 get_closest_point_to(const Vector3 &p_point) const { return p_point; }

	Vector3 get_moment_of_inertia(real_t p_mass) const { return Vector3(); }

	virtual void set_data(const Variant &p_data) {}
	virtual Variant get_data() const { return Variant(); }

	MotionShape3DSW() { configure(AABB()); }
};

#endif // SHAPE_SW_H
