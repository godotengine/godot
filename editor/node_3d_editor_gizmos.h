/*************************************************************************/
/*  node_3d_editor_gizmos.h                                              */
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

#ifndef SPATIAL_EDITOR_GIZMOS_H
#define SPATIAL_EDITOR_GIZMOS_H

#include "editor/plugins/node_3d_editor_plugin.h"
#include "scene/3d/camera_3d.h"

class Camera3D;

class LightNode3DGizmoPlugin : public EditorNode3DGizmoPlugin {

	GDCLASS(LightNode3DGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial);
	String get_name() const;
	int get_priority() const;

	String get_handle_name(const EditorNode3DGizmo *p_gizmo, int p_idx) const;
	Variant get_handle_value(EditorNode3DGizmo *p_gizmo, int p_idx) const;
	void set_handle(EditorNode3DGizmo *p_gizmo, int p_idx, Camera3D *p_camera, const Point2 &p_point);
	void commit_handle(EditorNode3DGizmo *p_gizmo, int p_idx, const Variant &p_restore, bool p_cancel = false);
	void redraw(EditorNode3DGizmo *p_gizmo);

	LightNode3DGizmoPlugin();
};

class AudioStreamPlayer3DNode3DGizmoPlugin : public EditorNode3DGizmoPlugin {

	GDCLASS(AudioStreamPlayer3DNode3DGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial);
	String get_name() const;
	int get_priority() const;

	String get_handle_name(const EditorNode3DGizmo *p_gizmo, int p_idx) const;
	Variant get_handle_value(EditorNode3DGizmo *p_gizmo, int p_idx) const;
	void set_handle(EditorNode3DGizmo *p_gizmo, int p_idx, Camera3D *p_camera, const Point2 &p_point);
	void commit_handle(EditorNode3DGizmo *p_gizmo, int p_idx, const Variant &p_restore, bool p_cancel = false);
	void redraw(EditorNode3DGizmo *p_gizmo);

	AudioStreamPlayer3DNode3DGizmoPlugin();
};

class CameraNode3DGizmoPlugin : public EditorNode3DGizmoPlugin {

	GDCLASS(CameraNode3DGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial);
	String get_name() const;
	int get_priority() const;

	String get_handle_name(const EditorNode3DGizmo *p_gizmo, int p_idx) const;
	Variant get_handle_value(EditorNode3DGizmo *p_gizmo, int p_idx) const;
	void set_handle(EditorNode3DGizmo *p_gizmo, int p_idx, Camera3D *p_camera, const Point2 &p_point);
	void commit_handle(EditorNode3DGizmo *p_gizmo, int p_idx, const Variant &p_restore, bool p_cancel = false);
	void redraw(EditorNode3DGizmo *p_gizmo);

	CameraNode3DGizmoPlugin();
};

class MeshInstanceNode3DGizmoPlugin : public EditorNode3DGizmoPlugin {

	GDCLASS(MeshInstanceNode3DGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial);
	String get_name() const;
	int get_priority() const;
	bool can_be_hidden() const;
	void redraw(EditorNode3DGizmo *p_gizmo);

	MeshInstanceNode3DGizmoPlugin();
};

class Sprite3DNode3DGizmoPlugin : public EditorNode3DGizmoPlugin {

	GDCLASS(Sprite3DNode3DGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial);
	String get_name() const;
	int get_priority() const;
	bool can_be_hidden() const;
	void redraw(EditorNode3DGizmo *p_gizmo);

	Sprite3DNode3DGizmoPlugin();
};

class Position3DNode3DGizmoPlugin : public EditorNode3DGizmoPlugin {

	GDCLASS(Position3DNode3DGizmoPlugin, EditorNode3DGizmoPlugin);

	Ref<ArrayMesh> pos3d_mesh;
	Vector<Vector3> cursor_points;

public:
	bool has_gizmo(Node3D *p_spatial);
	String get_name() const;
	int get_priority() const;
	void redraw(EditorNode3DGizmo *p_gizmo);

	Position3DNode3DGizmoPlugin();
};

class SkeletonNode3DGizmoPlugin : public EditorNode3DGizmoPlugin {

	GDCLASS(SkeletonNode3DGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial);
	String get_name() const;
	int get_priority() const;
	void redraw(EditorNode3DGizmo *p_gizmo);

	SkeletonNode3DGizmoPlugin();
};

class PhysicalBoneNode3DGizmoPlugin : public EditorNode3DGizmoPlugin {

	GDCLASS(PhysicalBoneNode3DGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial);
	String get_name() const;
	int get_priority() const;
	void redraw(EditorNode3DGizmo *p_gizmo);

	PhysicalBoneNode3DGizmoPlugin();
};

class RayCastNode3DGizmoPlugin : public EditorNode3DGizmoPlugin {

	GDCLASS(RayCastNode3DGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial);
	String get_name() const;
	int get_priority() const;
	void redraw(EditorNode3DGizmo *p_gizmo);

	RayCastNode3DGizmoPlugin();
};

class SpringArmNode3DGizmoPlugin : public EditorNode3DGizmoPlugin {

	GDCLASS(SpringArmNode3DGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial);
	String get_name() const;
	int get_priority() const;
	void redraw(EditorNode3DGizmo *p_gizmo);

	SpringArmNode3DGizmoPlugin();
};

class VehicleWheelNode3DGizmoPlugin : public EditorNode3DGizmoPlugin {

	GDCLASS(VehicleWheelNode3DGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial);
	String get_name() const;
	int get_priority() const;
	void redraw(EditorNode3DGizmo *p_gizmo);

	VehicleWheelNode3DGizmoPlugin();
};

class SoftBodyNode3DGizmoPlugin : public EditorNode3DGizmoPlugin {

	GDCLASS(SoftBodyNode3DGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial);
	String get_name() const;
	int get_priority() const;
	bool is_selectable_when_hidden() const;
	void redraw(EditorNode3DGizmo *p_gizmo);

	String get_handle_name(const EditorNode3DGizmo *p_gizmo, int p_idx) const;
	Variant get_handle_value(EditorNode3DGizmo *p_gizmo, int p_idx) const;
	void commit_handle(EditorNode3DGizmo *p_gizmo, int p_idx, const Variant &p_restore, bool p_cancel);
	bool is_handle_highlighted(const EditorNode3DGizmo *p_gizmo, int idx) const;

	SoftBodyNode3DGizmoPlugin();
};

class VisibilityNotifierGizmoPlugin : public EditorNode3DGizmoPlugin {

	GDCLASS(VisibilityNotifierGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial);
	String get_name() const;
	int get_priority() const;
	void redraw(EditorNode3DGizmo *p_gizmo);

	String get_handle_name(const EditorNode3DGizmo *p_gizmo, int p_idx) const;
	Variant get_handle_value(EditorNode3DGizmo *p_gizmo, int p_idx) const;
	void set_handle(EditorNode3DGizmo *p_gizmo, int p_idx, Camera3D *p_camera, const Point2 &p_point);
	void commit_handle(EditorNode3DGizmo *p_gizmo, int p_idx, const Variant &p_restore, bool p_cancel = false);

	VisibilityNotifierGizmoPlugin();
};

class CPUParticlesGizmoPlugin : public EditorNode3DGizmoPlugin {
	GDCLASS(CPUParticlesGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial);
	String get_name() const;
	int get_priority() const;
	bool is_selectable_when_hidden() const;
	void redraw(EditorNode3DGizmo *p_gizmo);
	CPUParticlesGizmoPlugin();
};

class ParticlesGizmoPlugin : public EditorNode3DGizmoPlugin {

	GDCLASS(ParticlesGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial);
	String get_name() const;
	int get_priority() const;
	bool is_selectable_when_hidden() const;
	void redraw(EditorNode3DGizmo *p_gizmo);

	String get_handle_name(const EditorNode3DGizmo *p_gizmo, int p_idx) const;
	Variant get_handle_value(EditorNode3DGizmo *p_gizmo, int p_idx) const;
	void set_handle(EditorNode3DGizmo *p_gizmo, int p_idx, Camera3D *p_camera, const Point2 &p_point);
	void commit_handle(EditorNode3DGizmo *p_gizmo, int p_idx, const Variant &p_restore, bool p_cancel = false);

	ParticlesGizmoPlugin();
};

class ReflectionProbeGizmoPlugin : public EditorNode3DGizmoPlugin {

	GDCLASS(ReflectionProbeGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial);
	String get_name() const;
	int get_priority() const;
	void redraw(EditorNode3DGizmo *p_gizmo);

	String get_handle_name(const EditorNode3DGizmo *p_gizmo, int p_idx) const;
	Variant get_handle_value(EditorNode3DGizmo *p_gizmo, int p_idx) const;
	void set_handle(EditorNode3DGizmo *p_gizmo, int p_idx, Camera3D *p_camera, const Point2 &p_point);
	void commit_handle(EditorNode3DGizmo *p_gizmo, int p_idx, const Variant &p_restore, bool p_cancel = false);

	ReflectionProbeGizmoPlugin();
};

class GIProbeGizmoPlugin : public EditorNode3DGizmoPlugin {

	GDCLASS(GIProbeGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial);
	String get_name() const;
	int get_priority() const;
	void redraw(EditorNode3DGizmo *p_gizmo);

	String get_handle_name(const EditorNode3DGizmo *p_gizmo, int p_idx) const;
	Variant get_handle_value(EditorNode3DGizmo *p_gizmo, int p_idx) const;
	void set_handle(EditorNode3DGizmo *p_gizmo, int p_idx, Camera3D *p_camera, const Point2 &p_point);
	void commit_handle(EditorNode3DGizmo *p_gizmo, int p_idx, const Variant &p_restore, bool p_cancel = false);

	GIProbeGizmoPlugin();
};

#if 0
class BakedIndirectLightGizmoPlugin : public EditorNode3DGizmoPlugin {

	GDCLASS(BakedIndirectLightGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Spatial *p_spatial);
	String get_name() const;
	int get_priority() const;
	void redraw(EditorNode3DGizmo *p_gizmo);

	String get_handle_name(const EditorNode3DGizmo *p_gizmo, int p_idx) const;
	Variant get_handle_value(EditorNode3DGizmo *p_gizmo, int p_idx) const;
	void set_handle(EditorNode3DGizmo *p_gizmo, int p_idx, Camera *p_camera, const Point2 &p_point);
	void commit_handle(EditorNode3DGizmo *p_gizmo, int p_idx, const Variant &p_restore, bool p_cancel = false);

	BakedIndirectLightGizmoPlugin();
};
#endif
class CollisionShapeNode3DGizmoPlugin : public EditorNode3DGizmoPlugin {

	GDCLASS(CollisionShapeNode3DGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial);
	String get_name() const;
	int get_priority() const;
	void redraw(EditorNode3DGizmo *p_gizmo);

	String get_handle_name(const EditorNode3DGizmo *p_gizmo, int p_idx) const;
	Variant get_handle_value(EditorNode3DGizmo *p_gizmo, int p_idx) const;
	void set_handle(EditorNode3DGizmo *p_gizmo, int p_idx, Camera3D *p_camera, const Point2 &p_point);
	void commit_handle(EditorNode3DGizmo *p_gizmo, int p_idx, const Variant &p_restore, bool p_cancel = false);

	CollisionShapeNode3DGizmoPlugin();
};

class CollisionPolygonNode3DGizmoPlugin : public EditorNode3DGizmoPlugin {
	GDCLASS(CollisionPolygonNode3DGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial);
	String get_name() const;
	int get_priority() const;
	void redraw(EditorNode3DGizmo *p_gizmo);
	CollisionPolygonNode3DGizmoPlugin();
};

class NavigationMeshNode3DGizmoPlugin : public EditorNode3DGizmoPlugin {

	GDCLASS(NavigationMeshNode3DGizmoPlugin, EditorNode3DGizmoPlugin);

	struct _EdgeKey {

		Vector3 from;
		Vector3 to;

		bool operator<(const _EdgeKey &p_with) const { return from == p_with.from ? to < p_with.to : from < p_with.from; }
	};

public:
	bool has_gizmo(Node3D *p_spatial);
	String get_name() const;
	int get_priority() const;
	void redraw(EditorNode3DGizmo *p_gizmo);

	NavigationMeshNode3DGizmoPlugin();
};

class JointGizmosDrawer {
public:
	static Basis look_body(const Transform &p_joint_transform, const Transform &p_body_transform);
	static Basis look_body_toward(Vector3::Axis p_axis, const Transform &joint_transform, const Transform &body_transform);
	static Basis look_body_toward_x(const Transform &p_joint_transform, const Transform &p_body_transform);
	static Basis look_body_toward_y(const Transform &p_joint_transform, const Transform &p_body_transform);
	/// Special function just used for physics joints, it returns a basis constrained toward Joint Z axis
	/// with axis X and Y that are looking toward the body and oriented toward up
	static Basis look_body_toward_z(const Transform &p_joint_transform, const Transform &p_body_transform);

	// Draw circle around p_axis
	static void draw_circle(Vector3::Axis p_axis, real_t p_radius, const Transform &p_offset, const Basis &p_base, real_t p_limit_lower, real_t p_limit_upper, Vector<Vector3> &r_points, bool p_inverse = false);
	static void draw_cone(const Transform &p_offset, const Basis &p_base, real_t p_swing, real_t p_twist, Vector<Vector3> &r_points);
};

class JointNode3DGizmoPlugin : public EditorNode3DGizmoPlugin {

	GDCLASS(JointNode3DGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial);
	String get_name() const;
	int get_priority() const;
	void redraw(EditorNode3DGizmo *p_gizmo);

	static void CreatePinJointGizmo(const Transform &p_offset, Vector<Vector3> &r_cursor_points);
	static void CreateHingeJointGizmo(const Transform &p_offset, const Transform &p_trs_joint, const Transform &p_trs_body_a, const Transform &p_trs_body_b, real_t p_limit_lower, real_t p_limit_upper, bool p_use_limit, Vector<Vector3> &r_common_points, Vector<Vector3> *r_body_a_points, Vector<Vector3> *r_body_b_points);
	static void CreateSliderJointGizmo(const Transform &p_offset, const Transform &p_trs_joint, const Transform &p_trs_body_a, const Transform &p_trs_body_b, real_t p_angular_limit_lower, real_t p_angular_limit_upper, real_t p_linear_limit_lower, real_t p_linear_limit_upper, Vector<Vector3> &r_points, Vector<Vector3> *r_body_a_points, Vector<Vector3> *r_body_b_points);
	static void CreateConeTwistJointGizmo(const Transform &p_offset, const Transform &p_trs_joint, const Transform &p_trs_body_a, const Transform &p_trs_body_b, real_t p_swing, real_t p_twist, Vector<Vector3> *r_body_a_points, Vector<Vector3> *r_body_b_points);
	static void CreateGeneric6DOFJointGizmo(
			const Transform &p_offset,
			const Transform &p_trs_joint,
			const Transform &p_trs_body_a,
			const Transform &p_trs_body_b,
			real_t p_angular_limit_lower_x,
			real_t p_angular_limit_upper_x,
			real_t p_linear_limit_lower_x,
			real_t p_linear_limit_upper_x,
			bool p_enable_angular_limit_x,
			bool p_enable_linear_limit_x,
			real_t p_angular_limit_lower_y,
			real_t p_angular_limit_upper_y,
			real_t p_linear_limit_lower_y,
			real_t p_linear_limit_upper_y,
			bool p_enable_angular_limit_y,
			bool p_enable_linear_limit_y,
			real_t p_angular_limit_lower_z,
			real_t p_angular_limit_upper_z,
			real_t p_linear_limit_lower_z,
			real_t p_linear_limit_upper_z,
			bool p_enable_angular_limit_z,
			bool p_enable_linear_limit_z,
			Vector<Vector3> &r_points,
			Vector<Vector3> *r_body_a_points,
			Vector<Vector3> *r_body_b_points);

	JointNode3DGizmoPlugin();
};

#endif // SPATIAL_EDITOR_GIZMOS_H
