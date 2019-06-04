/*************************************************************************/
/*  spatial_editor_gizmos.h                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifndef SPATIAL_EDITOR_GIZMOS_H
#define SPATIAL_EDITOR_GIZMOS_H

#include "editor/plugins/spatial_editor_plugin.h"
#include "scene/3d/body_shape.h"
#include "scene/3d/camera.h"
#include "scene/3d/light.h"
#include "scene/3d/listener.h"
#include "scene/3d/mesh_instance.h"
#include "scene/3d/navigation_mesh.h"
#include "scene/3d/portal.h"
#include "scene/3d/position_3d.h"
#include "scene/3d/ray_cast.h"
#include "scene/3d/room_instance.h"
#include "scene/3d/spatial_sample_player.h"
#include "scene/3d/spatial_stream_player.h"
#include "scene/3d/test_cube.h"
#include "scene/3d/visibility_notifier.h"

#include "scene/3d/collision_polygon.h"
#include "scene/3d/physics_joint.h"
#include "scene/3d/vehicle_body.h"

class Camera;

class EditorSpatialGizmo : public SpatialEditorGizmo {

	OBJ_TYPE(EditorSpatialGizmo, SpatialGizmo);

	struct Instance {

		RID instance;
		Ref<Mesh> mesh;
		RID skeleton;
		bool billboard;
		bool unscaled;
		bool can_intersect;
		bool extra_margin;
		Instance() {

			billboard = false;
			unscaled = false;
			can_intersect = false;
			extra_margin = false;
		}

		void create_instance(Spatial *p_base);
	};

	Vector<Vector3> collision_segments;
	Ref<TriangleMesh> collision_mesh;

	struct Handle {
		Vector3 pos;
		bool billboard;
	};

	Vector<Vector3> handles;
	Vector<Vector3> secondary_handles;
	bool billboard_handle;

	bool valid;
	Spatial *base;
	Vector<Instance> instances;
	Spatial *spatial_node;

	void _set_spatial_node(Node *p_node) { set_spatial_node(p_node->cast_to<Spatial>()); }

protected:
	void add_lines(const Vector<Vector3> &p_lines, const Ref<Material> &p_material, bool p_billboard = false);
	void add_mesh(const Ref<Mesh> &p_mesh, bool p_billboard = false, const RID &p_skeleton = RID());
	void add_collision_segments(const Vector<Vector3> &p_lines);
	void add_collision_triangles(const Ref<TriangleMesh> &p_tmesh);
	void add_unscaled_billboard(const Ref<Material> &p_material, float p_scale = 1);
	void add_handles(const Vector<Vector3> &p_handles, bool p_billboard = false, bool p_secondary = false);

	void set_spatial_node(Spatial *p_node);

	static void _bind_methods();

public:
	virtual Vector3 get_handle_pos(int p_idx) const;
	virtual bool intersect_frustum(const Camera *p_camera, const Vector<Plane> &p_frustum);
	virtual bool intersect_ray(const Camera *p_camera, const Point2 &p_point, Vector3 &r_pos, Vector3 &r_normal, int *r_gizmo_handle = NULL, bool p_sec_first = false);

	void clear();
	void create();
	void transform();
	virtual void redraw();
	void free();

	EditorSpatialGizmo();
	~EditorSpatialGizmo();
};

class LightSpatialGizmo : public EditorSpatialGizmo {

	OBJ_TYPE(LightSpatialGizmo, EditorSpatialGizmo);

	Light *light;

public:
	virtual String get_handle_name(int p_idx) const;
	virtual Variant get_handle_value(int p_idx) const;
	virtual void set_handle(int p_idx, Camera *p_camera, const Point2 &p_point);
	virtual void commit_handle(int p_idx, const Variant &p_restore, bool p_cancel = false);

	void redraw();
	LightSpatialGizmo(Light *p_light = NULL);
};

class ListenerSpatialGizmo : public EditorSpatialGizmo {

	OBJ_TYPE(ListenerSpatialGizmo, EditorSpatialGizmo);

	Listener *listener;

public:
	void redraw();
	ListenerSpatialGizmo(Listener *p_listener = NULL);
};

class CameraSpatialGizmo : public EditorSpatialGizmo {

	OBJ_TYPE(CameraSpatialGizmo, EditorSpatialGizmo);

	Camera *camera;

public:
	virtual String get_handle_name(int p_idx) const;
	virtual Variant get_handle_value(int p_idx) const;
	virtual void set_handle(int p_idx, Camera *p_camera, const Point2 &p_point);
	virtual void commit_handle(int p_idx, const Variant &p_restore, bool p_cancel = false);

	void redraw();
	CameraSpatialGizmo(Camera *p_camera = NULL);
};

class MeshInstanceSpatialGizmo : public EditorSpatialGizmo {

	OBJ_TYPE(MeshInstanceSpatialGizmo, EditorSpatialGizmo);

	MeshInstance *mesh;

public:
	void redraw();
	MeshInstanceSpatialGizmo(MeshInstance *p_mesh = NULL);
};

class Position3DSpatialGizmo : public EditorSpatialGizmo {

	OBJ_TYPE(Position3DSpatialGizmo, EditorSpatialGizmo);

	Position3D *p3d;

public:
	void redraw();
	Position3DSpatialGizmo(Position3D *p_p3d = NULL);
};

class SkeletonSpatialGizmo : public EditorSpatialGizmo {

	OBJ_TYPE(SkeletonSpatialGizmo, EditorSpatialGizmo);

	Skeleton *skel;

public:
	void redraw();
	SkeletonSpatialGizmo(Skeleton *p_skel = NULL);
};

class SpatialPlayerSpatialGizmo : public EditorSpatialGizmo {

	OBJ_TYPE(SpatialPlayerSpatialGizmo, EditorSpatialGizmo);

	SpatialPlayer *splayer;

public:
	void redraw();
	SpatialPlayerSpatialGizmo(SpatialPlayer *p_splayer = NULL);
};

class TestCubeSpatialGizmo : public EditorSpatialGizmo {

	OBJ_TYPE(TestCubeSpatialGizmo, EditorSpatialGizmo);

	TestCube *tc;

public:
	void redraw();
	TestCubeSpatialGizmo(TestCube *p_tc = NULL);
};

class RoomSpatialGizmo : public EditorSpatialGizmo {

	OBJ_TYPE(RoomSpatialGizmo, EditorSpatialGizmo);

	struct _EdgeKey {

		Vector3 from;
		Vector3 to;

		bool operator<(const _EdgeKey &p_with) const { return from == p_with.from ? to < p_with.to : from < p_with.from; }
	};

	Room *room;

public:
	void redraw();
	RoomSpatialGizmo(Room *p_room = NULL);
};

class PortalSpatialGizmo : public EditorSpatialGizmo {

	OBJ_TYPE(PortalSpatialGizmo, EditorSpatialGizmo);

	Portal *portal;

public:
	void redraw();
	PortalSpatialGizmo(Portal *p_portal = NULL);
};

class VisibilityNotifierGizmo : public EditorSpatialGizmo {

	OBJ_TYPE(VisibilityNotifierGizmo, EditorSpatialGizmo);

	VisibilityNotifier *notifier;

public:
	virtual String get_handle_name(int p_idx) const;
	virtual Variant get_handle_value(int p_idx) const;
	virtual void set_handle(int p_idx, Camera *p_camera, const Point2 &p_point);
	virtual void commit_handle(int p_idx, const Variant &p_restore, bool p_cancel = false);

	void redraw();
	VisibilityNotifierGizmo(VisibilityNotifier *p_notifier = NULL);
};

class CollisionShapeSpatialGizmo : public EditorSpatialGizmo {

	OBJ_TYPE(CollisionShapeSpatialGizmo, EditorSpatialGizmo);

	CollisionShape *cs;

public:
	virtual String get_handle_name(int p_idx) const;
	virtual Variant get_handle_value(int p_idx) const;
	virtual void set_handle(int p_idx, Camera *p_camera, const Point2 &p_point);
	virtual void commit_handle(int p_idx, const Variant &p_restore, bool p_cancel = false);
	void redraw();
	CollisionShapeSpatialGizmo(CollisionShape *p_cs = NULL);
};

class CollisionPolygonSpatialGizmo : public EditorSpatialGizmo {

	OBJ_TYPE(CollisionPolygonSpatialGizmo, EditorSpatialGizmo);

	CollisionPolygon *polygon;

public:
	void redraw();
	CollisionPolygonSpatialGizmo(CollisionPolygon *p_polygon = NULL);
};

class RayCastSpatialGizmo : public EditorSpatialGizmo {

	OBJ_TYPE(RayCastSpatialGizmo, EditorSpatialGizmo);

	RayCast *raycast;

public:
	void redraw();
	RayCastSpatialGizmo(RayCast *p_raycast = NULL);
};

class VehicleWheelSpatialGizmo : public EditorSpatialGizmo {

	OBJ_TYPE(VehicleWheelSpatialGizmo, EditorSpatialGizmo);

	VehicleWheel *car_wheel;

public:
	void redraw();
	VehicleWheelSpatialGizmo(VehicleWheel *p_car_wheel = NULL);
};

class NavigationMeshSpatialGizmo : public EditorSpatialGizmo {

	OBJ_TYPE(NavigationMeshSpatialGizmo, EditorSpatialGizmo);

	struct _EdgeKey {

		Vector3 from;
		Vector3 to;

		bool operator<(const _EdgeKey &p_with) const { return from == p_with.from ? to < p_with.to : from < p_with.from; }
	};

	NavigationMeshInstance *navmesh;

public:
	void redraw();
	NavigationMeshSpatialGizmo(NavigationMeshInstance *p_navmesh = NULL);
};

class PinJointSpatialGizmo : public EditorSpatialGizmo {

	OBJ_TYPE(PinJointSpatialGizmo, EditorSpatialGizmo);

	PinJoint *p3d;

public:
	void redraw();
	PinJointSpatialGizmo(PinJoint *p_p3d = NULL);
};

class HingeJointSpatialGizmo : public EditorSpatialGizmo {

	OBJ_TYPE(HingeJointSpatialGizmo, EditorSpatialGizmo);

	HingeJoint *p3d;

public:
	void redraw();
	HingeJointSpatialGizmo(HingeJoint *p_p3d = NULL);
};

class SliderJointSpatialGizmo : public EditorSpatialGizmo {

	OBJ_TYPE(SliderJointSpatialGizmo, EditorSpatialGizmo);

	SliderJoint *p3d;

public:
	void redraw();
	SliderJointSpatialGizmo(SliderJoint *p_p3d = NULL);
};

class ConeTwistJointSpatialGizmo : public EditorSpatialGizmo {

	OBJ_TYPE(ConeTwistJointSpatialGizmo, EditorSpatialGizmo);

	ConeTwistJoint *p3d;

public:
	void redraw();
	ConeTwistJointSpatialGizmo(ConeTwistJoint *p_p3d = NULL);
};

class Generic6DOFJointSpatialGizmo : public EditorSpatialGizmo {

	OBJ_TYPE(Generic6DOFJointSpatialGizmo, EditorSpatialGizmo);

	Generic6DOFJoint *p3d;

public:
	void redraw();
	Generic6DOFJointSpatialGizmo(Generic6DOFJoint *p_p3d = NULL);
};

class SpatialEditorGizmos {
public:
	Ref<FixedMaterial> create_line_material(const Color &p_base_color);
	Ref<FixedMaterial> create_solid_material(const Color &p_base_color);
	Ref<FixedMaterial> handle2_material;
	Ref<FixedMaterial> handle_material;
	Ref<FixedMaterial> light_material;
	Ref<FixedMaterial> light_material_omni_icon;
	Ref<FixedMaterial> light_material_directional_icon;
	Ref<FixedMaterial> camera_material;
	Ref<FixedMaterial> skeleton_material;
	Ref<FixedMaterial> room_material;
	Ref<FixedMaterial> portal_material;
	Ref<FixedMaterial> raycast_material;
	Ref<FixedMaterial> visibility_notifier_material;
	Ref<FixedMaterial> car_wheel_material;
	Ref<FixedMaterial> joint_material;

	Ref<FixedMaterial> navmesh_edge_material;
	Ref<FixedMaterial> navmesh_solid_material;
	Ref<FixedMaterial> navmesh_edge_material_disabled;
	Ref<FixedMaterial> navmesh_solid_material_disabled;

	Ref<FixedMaterial> listener_icon;

	Ref<FixedMaterial> sample_player_icon;
	Ref<FixedMaterial> stream_player_icon;
	Ref<FixedMaterial> visibility_notifier_icon;

	Ref<FixedMaterial> shape_material;
	Ref<Texture> handle_t;

	Ref<Mesh> pos3d_mesh;
	Ref<Mesh> listener_line_mesh;
	static SpatialEditorGizmos *singleton;

	Ref<TriangleMesh> test_cube_tm;

	Ref<SpatialEditorGizmo> get_gizmo(Spatial *p_spatial);

	SpatialEditorGizmos();
};

#endif // SPATIAL_EDITOR_GIZMOS_H
