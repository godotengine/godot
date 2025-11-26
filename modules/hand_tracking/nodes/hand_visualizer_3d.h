/**************************************************************************/
/*  hand_visualizer_3d.h                                                  */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md).*/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#ifndef HAND_VISUALIZER_3D_H
#define HAND_VISUALIZER_3D_H

#include "scene/3d/mesh_instance_3d.h"
#include "scene/3d/node_3d.h"
#include "scene/resources/3d/primitive_meshes.h"
#include "scene/resources/material.h"
#include "servers/xr/xr_hand_tracker.h"

/// HandVisualizer3D renders a hand skeleton for debugging
/// Shows joint positions as spheres and bones as cylinders
class HandVisualizer3D : public Node3D {
	GDCLASS(HandVisualizer3D, Node3D);

public:
	enum Hand {
		HAND_LEFT,
		HAND_RIGHT,
		HAND_MAX
	};

private:
	// Configuration
	Hand hand = HAND_LEFT;
	bool show_joints = true;
	bool show_bones = true;
	bool show_palm = true;

	float joint_radius = 0.01f;      // 1cm spheres
	float bone_radius = 0.005f;      // 0.5cm cylinders
	Color joint_color = Color(1, 0, 0); // Red joints
	Color bone_color = Color(0.8, 0.8, 0.8); // Gray bones

	// Internal state
	Ref<XRHandTracker> tracker;
	bool initialized = false;

	// Visual elements (created dynamically)
	struct JointVisual {
		MeshInstance3D *mesh_instance = nullptr;
		Ref<SphereMesh> sphere_mesh;
		Ref<StandardMaterial3D> material;
	};

	struct BoneVisual {
		MeshInstance3D *mesh_instance = nullptr;
		Ref<CylinderMesh> cylinder_mesh;
		Ref<StandardMaterial3D> material;
	};

	Vector<JointVisual> joint_visuals;
	Vector<BoneVisual> bone_visuals;

	// Bone connections (pairs of joint indices)
	struct BoneConnection {
		XRHandTracker::HandJoint from;
		XRHandTracker::HandJoint to;
	};
	Vector<BoneConnection> bone_connections;

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	HandVisualizer3D();
	~HandVisualizer3D();

	// Configuration
	void set_hand(Hand p_hand);
	Hand get_hand() const { return hand; }

	void set_show_joints(bool p_show);
	bool get_show_joints() const { return show_joints; }

	void set_show_bones(bool p_show);
	bool get_show_bones() const { return show_bones; }

	void set_show_palm(bool p_show);
	bool get_show_palm() const { return show_palm; }

	void set_joint_radius(float p_radius);
	float get_joint_radius() const { return joint_radius; }

	void set_bone_radius(float p_radius);
	float get_bone_radius() const { return bone_radius; }

	void set_joint_color(const Color &p_color);
	Color get_joint_color() const { return joint_color; }

	void set_bone_color(const Color &p_color);
	Color get_bone_color() const { return bone_color; }

private:
	void _initialize_visuals();
	void _cleanup_visuals();
	void _update_tracker_reference();
	void _update_visuals();
	void _setup_bone_connections();
	void _update_joint_visual(int index, const Transform3D &transform, bool visible);
	void _update_bone_visual(int index, const Vector3 &from, const Vector3 &to, bool visible);
};

VARIANT_ENUM_CAST(HandVisualizer3D::Hand);

#endif // HAND_VISUALIZER_3D_H
