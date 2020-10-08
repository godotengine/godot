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

class Light3DGizmoPlugin : public EditorNode3DGizmoPlugin {
	GDCLASS(Light3DGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial) override;
	String get_name() const override;
	int get_priority() const override;

	String get_handle_name(const EditorNode3DGizmo *p_gizmo, int p_idx) const override;
	Variant get_handle_value(EditorNode3DGizmo *p_gizmo, int p_idx) const override;
	void set_handle(EditorNode3DGizmo *p_gizmo, int p_idx, Camera3D *p_camera, const Point2 &p_point) override;
	void commit_handle(EditorNode3DGizmo *p_gizmo, int p_idx, const Variant &p_restore, bool p_cancel = false) override;
	void redraw(EditorNode3DGizmo *p_gizmo) override;

	Light3DGizmoPlugin();
};

class AudioStreamPlayer3DGizmoPlugin : public EditorNode3DGizmoPlugin {
	GDCLASS(AudioStreamPlayer3DGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial) override;
	String get_name() const override;
	int get_priority() const override;

	String get_handle_name(const EditorNode3DGizmo *p_gizmo, int p_idx) const override;
	Variant get_handle_value(EditorNode3DGizmo *p_gizmo, int p_idx) const override;
	void set_handle(EditorNode3DGizmo *p_gizmo, int p_idx, Camera3D *p_camera, const Point2 &p_point) override;
	void commit_handle(EditorNode3DGizmo *p_gizmo, int p_idx, const Variant &p_restore, bool p_cancel = false) override;
	void redraw(EditorNode3DGizmo *p_gizmo) override;

	AudioStreamPlayer3DGizmoPlugin();
};

class Camera3DGizmoPlugin : public EditorNode3DGizmoPlugin {
	GDCLASS(Camera3DGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial) override;
	String get_name() const override;
	int get_priority() const override;

	String get_handle_name(const EditorNode3DGizmo *p_gizmo, int p_idx) const override;
	Variant get_handle_value(EditorNode3DGizmo *p_gizmo, int p_idx) const override;
	void set_handle(EditorNode3DGizmo *p_gizmo, int p_idx, Camera3D *p_camera, const Point2 &p_point) override;
	void commit_handle(EditorNode3DGizmo *p_gizmo, int p_idx, const Variant &p_restore, bool p_cancel = false) override;
	void redraw(EditorNode3DGizmo *p_gizmo) override;

	Camera3DGizmoPlugin();
};

class MeshInstance3DGizmoPlugin : public EditorNode3DGizmoPlugin {
	GDCLASS(MeshInstance3DGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial) override;
	String get_name() const override;
	int get_priority() const override;
	bool can_be_hidden() const override;
	void redraw(EditorNode3DGizmo *p_gizmo) override;

	MeshInstance3DGizmoPlugin();
};

class Sprite3DGizmoPlugin : public EditorNode3DGizmoPlugin {
	GDCLASS(Sprite3DGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial) override;
	String get_name() const override;
	int get_priority() const override;
	bool can_be_hidden() const override;
	void redraw(EditorNode3DGizmo *p_gizmo) override;

	Sprite3DGizmoPlugin();
};

class Position3DGizmoPlugin : public EditorNode3DGizmoPlugin {
	GDCLASS(Position3DGizmoPlugin, EditorNode3DGizmoPlugin);

	Ref<ArrayMesh> pos3d_mesh;
	Vector<Vector3> cursor_points;

public:
	bool has_gizmo(Node3D *p_spatial) override;
	String get_name() const override;
	int get_priority() const override;
	void redraw(EditorNode3DGizmo *p_gizmo) override;

	Position3DGizmoPlugin();
};

class Skeleton3DGizmoPlugin : public EditorNode3DGizmoPlugin {
	GDCLASS(Skeleton3DGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial) override;
	String get_name() const override;
	int get_priority() const override;
	void redraw(EditorNode3DGizmo *p_gizmo) override;

	Skeleton3DGizmoPlugin();
};

class PhysicalBone3DGizmoPlugin : public EditorNode3DGizmoPlugin {
	GDCLASS(PhysicalBone3DGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial) override;
	String get_name() const override;
	int get_priority() const override;
	void redraw(EditorNode3DGizmo *p_gizmo) override;

	PhysicalBone3DGizmoPlugin();
};

class RayCast3DGizmoPlugin : public EditorNode3DGizmoPlugin {
	GDCLASS(RayCast3DGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial) override;
	String get_name() const override;
	int get_priority() const override;
	void redraw(EditorNode3DGizmo *p_gizmo) override;

	RayCast3DGizmoPlugin();
};

class SpringArm3DGizmoPlugin : public EditorNode3DGizmoPlugin {
	GDCLASS(SpringArm3DGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial) override;
	String get_name() const override;
	int get_priority() const override;
	void redraw(EditorNode3DGizmo *p_gizmo) override;

	SpringArm3DGizmoPlugin();
};

class VehicleWheel3DGizmoPlugin : public EditorNode3DGizmoPlugin {
	GDCLASS(VehicleWheel3DGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial) override;
	String get_name() const override;
	int get_priority() const override;
	void redraw(EditorNode3DGizmo *p_gizmo) override;

	VehicleWheel3DGizmoPlugin();
};

class SoftBody3DGizmoPlugin : public EditorNode3DGizmoPlugin {
	GDCLASS(SoftBody3DGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial) override;
	String get_name() const override;
	int get_priority() const override;
	bool is_selectable_when_hidden() const override;
	void redraw(EditorNode3DGizmo *p_gizmo) override;

	String get_handle_name(const EditorNode3DGizmo *p_gizmo, int p_idx) const override;
	Variant get_handle_value(EditorNode3DGizmo *p_gizmo, int p_idx) const override;
	void commit_handle(EditorNode3DGizmo *p_gizmo, int p_idx, const Variant &p_restore, bool p_cancel) override;
	bool is_handle_highlighted(const EditorNode3DGizmo *p_gizmo, int idx) const override;

	SoftBody3DGizmoPlugin();
};

class VisibilityNotifier3DGizmoPlugin : public EditorNode3DGizmoPlugin {
	GDCLASS(VisibilityNotifier3DGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial) override;
	String get_name() const override;
	int get_priority() const override;
	void redraw(EditorNode3DGizmo *p_gizmo) override;

	String get_handle_name(const EditorNode3DGizmo *p_gizmo, int p_idx) const override;
	Variant get_handle_value(EditorNode3DGizmo *p_gizmo, int p_idx) const override;
	void set_handle(EditorNode3DGizmo *p_gizmo, int p_idx, Camera3D *p_camera, const Point2 &p_point) override;
	void commit_handle(EditorNode3DGizmo *p_gizmo, int p_idx, const Variant &p_restore, bool p_cancel = false) override;

	VisibilityNotifier3DGizmoPlugin();
};

class CPUParticles3DGizmoPlugin : public EditorNode3DGizmoPlugin {
	GDCLASS(CPUParticles3DGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial) override;
	String get_name() const override;
	int get_priority() const override;
	bool is_selectable_when_hidden() const override;
	void redraw(EditorNode3DGizmo *p_gizmo) override;
	CPUParticles3DGizmoPlugin();
};

class GPUParticles3DGizmoPlugin : public EditorNode3DGizmoPlugin {
	GDCLASS(GPUParticles3DGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial) override;
	String get_name() const override;
	int get_priority() const override;
	bool is_selectable_when_hidden() const override;
	void redraw(EditorNode3DGizmo *p_gizmo) override;

	String get_handle_name(const EditorNode3DGizmo *p_gizmo, int p_idx) const override;
	Variant get_handle_value(EditorNode3DGizmo *p_gizmo, int p_idx) const override;
	void set_handle(EditorNode3DGizmo *p_gizmo, int p_idx, Camera3D *p_camera, const Point2 &p_point) override;
	void commit_handle(EditorNode3DGizmo *p_gizmo, int p_idx, const Variant &p_restore, bool p_cancel = false) override;

	GPUParticles3DGizmoPlugin();
};

class GPUParticlesCollision3DGizmoPlugin : public EditorNode3DGizmoPlugin {
	GDCLASS(GPUParticlesCollision3DGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial) override;
	String get_name() const override;
	int get_priority() const override;
	void redraw(EditorNode3DGizmo *p_gizmo) override;

	String get_handle_name(const EditorNode3DGizmo *p_gizmo, int p_idx) const override;
	Variant get_handle_value(EditorNode3DGizmo *p_gizmo, int p_idx) const override;
	void set_handle(EditorNode3DGizmo *p_gizmo, int p_idx, Camera3D *p_camera, const Point2 &p_point) override;
	void commit_handle(EditorNode3DGizmo *p_gizmo, int p_idx, const Variant &p_restore, bool p_cancel = false) override;

	GPUParticlesCollision3DGizmoPlugin();
};

class ReflectionProbeGizmoPlugin : public EditorNode3DGizmoPlugin {
	GDCLASS(ReflectionProbeGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial) override;
	String get_name() const override;
	int get_priority() const override;
	void redraw(EditorNode3DGizmo *p_gizmo) override;

	String get_handle_name(const EditorNode3DGizmo *p_gizmo, int p_idx) const override;
	Variant get_handle_value(EditorNode3DGizmo *p_gizmo, int p_idx) const override;
	void set_handle(EditorNode3DGizmo *p_gizmo, int p_idx, Camera3D *p_camera, const Point2 &p_point) override;
	void commit_handle(EditorNode3DGizmo *p_gizmo, int p_idx, const Variant &p_restore, bool p_cancel = false) override;

	ReflectionProbeGizmoPlugin();
};

class DecalGizmoPlugin : public EditorNode3DGizmoPlugin {
	GDCLASS(DecalGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial) override;
	String get_name() const override;
	int get_priority() const override;
	void redraw(EditorNode3DGizmo *p_gizmo) override;

	String get_handle_name(const EditorNode3DGizmo *p_gizmo, int p_idx) const override;
	Variant get_handle_value(EditorNode3DGizmo *p_gizmo, int p_idx) const override;
	void set_handle(EditorNode3DGizmo *p_gizmo, int p_idx, Camera3D *p_camera, const Point2 &p_point) override;
	void commit_handle(EditorNode3DGizmo *p_gizmo, int p_idx, const Variant &p_restore, bool p_cancel = false) override;

	DecalGizmoPlugin();
};

class GIProbeGizmoPlugin : public EditorNode3DGizmoPlugin {
	GDCLASS(GIProbeGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial) override;
	String get_name() const override;
	int get_priority() const override;
	void redraw(EditorNode3DGizmo *p_gizmo) override;

	String get_handle_name(const EditorNode3DGizmo *p_gizmo, int p_idx) const override;
	Variant get_handle_value(EditorNode3DGizmo *p_gizmo, int p_idx) const override;
	void set_handle(EditorNode3DGizmo *p_gizmo, int p_idx, Camera3D *p_camera, const Point2 &p_point) override;
	void commit_handle(EditorNode3DGizmo *p_gizmo, int p_idx, const Variant &p_restore, bool p_cancel = false) override;

	GIProbeGizmoPlugin();
};

class BakedLightmapGizmoPlugin : public EditorNode3DGizmoPlugin {
	GDCLASS(BakedLightmapGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial) override;
	String get_name() const override;
	int get_priority() const override;
	void redraw(EditorNode3DGizmo *p_gizmo) override;

	String get_handle_name(const EditorNode3DGizmo *p_gizmo, int p_idx) const override;
	Variant get_handle_value(EditorNode3DGizmo *p_gizmo, int p_idx) const override;
	void set_handle(EditorNode3DGizmo *p_gizmo, int p_idx, Camera3D *p_camera, const Point2 &p_point) override;
	void commit_handle(EditorNode3DGizmo *p_gizmo, int p_idx, const Variant &p_restore, bool p_cancel = false) override;

	BakedLightmapGizmoPlugin();
};

class LightmapProbeGizmoPlugin : public EditorNode3DGizmoPlugin {
	GDCLASS(LightmapProbeGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial) override;
	String get_name() const override;
	int get_priority() const override;
	void redraw(EditorNode3DGizmo *p_gizmo) override;

	String get_handle_name(const EditorNode3DGizmo *p_gizmo, int p_idx) const override;
	Variant get_handle_value(EditorNode3DGizmo *p_gizmo, int p_idx) const override;
	void set_handle(EditorNode3DGizmo *p_gizmo, int p_idx, Camera3D *p_camera, const Point2 &p_point) override;
	void commit_handle(EditorNode3DGizmo *p_gizmo, int p_idx, const Variant &p_restore, bool p_cancel = false) override;

	LightmapProbeGizmoPlugin();
};

class CollisionShape3DGizmoPlugin : public EditorNode3DGizmoPlugin {
	GDCLASS(CollisionShape3DGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial) override;
	String get_name() const override;
	int get_priority() const override;
	void redraw(EditorNode3DGizmo *p_gizmo) override;

	String get_handle_name(const EditorNode3DGizmo *p_gizmo, int p_idx) const override;
	Variant get_handle_value(EditorNode3DGizmo *p_gizmo, int p_idx) const override;
	void set_handle(EditorNode3DGizmo *p_gizmo, int p_idx, Camera3D *p_camera, const Point2 &p_point) override;
	void commit_handle(EditorNode3DGizmo *p_gizmo, int p_idx, const Variant &p_restore, bool p_cancel = false) override;

	CollisionShape3DGizmoPlugin();
};

class CollisionPolygon3DGizmoPlugin : public EditorNode3DGizmoPlugin {
	GDCLASS(CollisionPolygon3DGizmoPlugin, EditorNode3DGizmoPlugin);

public:
	bool has_gizmo(Node3D *p_spatial) override;
	String get_name() const override;
	int get_priority() const override;
	void redraw(EditorNode3DGizmo *p_gizmo) override;
	CollisionPolygon3DGizmoPlugin();
};

class NavigationRegion3DGizmoPlugin : public EditorNode3DGizmoPlugin {
	GDCLASS(NavigationRegion3DGizmoPlugin, EditorNode3DGizmoPlugin);

	struct _EdgeKey {
		Vector3 from;
		Vector3 to;

		bool operator<(const _EdgeKey &p_with) const { return from == p_with.from ? to < p_with.to : from < p_with.from; }
	};

public:
	bool has_gizmo(Node3D *p_spatial) override;
	String get_name() const override;
	int get_priority() const override;
	void redraw(EditorNode3DGizmo *p_gizmo) override;

	NavigationRegion3DGizmoPlugin();
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

class Joint3DGizmoPlugin : public EditorNode3DGizmoPlugin {
	GDCLASS(Joint3DGizmoPlugin, EditorNode3DGizmoPlugin);

	Timer *update_timer;
	uint64_t update_idx = 0;

	void incremental_update_gizmos();

public:
	bool has_gizmo(Node3D *p_spatial) override;
	String get_name() const override;
	int get_priority() const override;
	void redraw(EditorNode3DGizmo *p_gizmo) override;

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

	Joint3DGizmoPlugin();
};

#endif // SPATIAL_EDITOR_GIZMOS_H
