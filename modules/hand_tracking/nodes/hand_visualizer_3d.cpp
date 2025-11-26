/**************************************************************************/
/*  hand_visualizer_3d.cpp                                                */
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

#include "hand_visualizer_3d.h"

#include "../hand_tracking_server.h"
#include "scene/resources/3d/primitive_meshes.h"
#include "scene/resources/material.h"

void HandVisualizer3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_hand", "hand"), &HandVisualizer3D::set_hand);
	ClassDB::bind_method(D_METHOD("get_hand"), &HandVisualizer3D::get_hand);

	ClassDB::bind_method(D_METHOD("set_show_joints", "show"), &HandVisualizer3D::set_show_joints);
	ClassDB::bind_method(D_METHOD("get_show_joints"), &HandVisualizer3D::get_show_joints);

	ClassDB::bind_method(D_METHOD("set_show_bones", "show"), &HandVisualizer3D::set_show_bones);
	ClassDB::bind_method(D_METHOD("get_show_bones"), &HandVisualizer3D::get_show_bones);

	ClassDB::bind_method(D_METHOD("set_show_palm", "show"), &HandVisualizer3D::set_show_palm);
	ClassDB::bind_method(D_METHOD("get_show_palm"), &HandVisualizer3D::get_show_palm);

	ClassDB::bind_method(D_METHOD("set_joint_radius", "radius"), &HandVisualizer3D::set_joint_radius);
	ClassDB::bind_method(D_METHOD("get_joint_radius"), &HandVisualizer3D::get_joint_radius);

	ClassDB::bind_method(D_METHOD("set_bone_radius", "radius"), &HandVisualizer3D::set_bone_radius);
	ClassDB::bind_method(D_METHOD("get_bone_radius"), &HandVisualizer3D::get_bone_radius);

	ClassDB::bind_method(D_METHOD("set_joint_color", "color"), &HandVisualizer3D::set_joint_color);
	ClassDB::bind_method(D_METHOD("get_joint_color"), &HandVisualizer3D::get_joint_color);

	ClassDB::bind_method(D_METHOD("set_bone_color", "color"), &HandVisualizer3D::set_bone_color);
	ClassDB::bind_method(D_METHOD("get_bone_color"), &HandVisualizer3D::get_bone_color);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "hand", PROPERTY_HINT_ENUM, "Left,Right"), "set_hand", "get_hand");

	ADD_GROUP("Visibility", "show_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "show_joints"), "set_show_joints", "get_show_joints");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "show_bones"), "set_show_bones", "get_show_bones");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "show_palm"), "set_show_palm", "get_show_palm");

	ADD_GROUP("Appearance", "");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "joint_radius", PROPERTY_HINT_RANGE, "0.001,0.1,0.001"), "set_joint_radius", "get_joint_radius");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "bone_radius", PROPERTY_HINT_RANGE, "0.001,0.05,0.001"), "set_bone_radius", "get_bone_radius");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "joint_color"), "set_joint_color", "get_joint_color");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "bone_color"), "set_bone_color", "get_bone_color");

	BIND_ENUM_CONSTANT(HAND_LEFT);
	BIND_ENUM_CONSTANT(HAND_RIGHT);
	BIND_ENUM_CONSTANT(HAND_MAX);
}

void HandVisualizer3D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			_update_tracker_reference();
			_initialize_visuals();
			set_process_internal(true);
		} break;

		case NOTIFICATION_EXIT_TREE: {
			_cleanup_visuals();
			tracker.unref();
			set_process_internal(false);
		} break;

		case NOTIFICATION_INTERNAL_PROCESS: {
			_update_visuals();
		} break;
	}
}

HandVisualizer3D::HandVisualizer3D() {
	_setup_bone_connections();
}

HandVisualizer3D::~HandVisualizer3D() {
	_cleanup_visuals();
}

// ========================================
// Configuration
// ========================================

void HandVisualizer3D::set_hand(Hand p_hand) {
	if (hand != p_hand) {
		hand = p_hand;
		_update_tracker_reference();
	}
}

void HandVisualizer3D::set_show_joints(bool p_show) {
	show_joints = p_show;
}

void HandVisualizer3D::set_show_bones(bool p_show) {
	show_bones = p_show;
}

void HandVisualizer3D::set_show_palm(bool p_show) {
	show_palm = p_show;
}

void HandVisualizer3D::set_joint_radius(float p_radius) {
	joint_radius = p_radius;
	// Update existing visuals
	for (int i = 0; i < joint_visuals.size(); i++) {
		if (joint_visuals[i].sphere_mesh.is_valid()) {
			joint_visuals[i].sphere_mesh->set_radius(joint_radius);
			joint_visuals[i].sphere_mesh->set_height(joint_radius * 2.0f);
		}
	}
}

void HandVisualizer3D::set_bone_radius(float p_radius) {
	bone_radius = p_radius;
	// Bones will be updated in _update_visuals
}

void HandVisualizer3D::set_joint_color(const Color &p_color) {
	joint_color = p_color;
	// Update existing materials
	for (int i = 0; i < joint_visuals.size(); i++) {
		if (joint_visuals[i].material.is_valid()) {
			joint_visuals[i].material->set_albedo(joint_color);
		}
	}
}

void HandVisualizer3D::set_bone_color(const Color &p_color) {
	bone_color = p_color;
	// Update existing materials
	for (int i = 0; i < bone_visuals.size(); i++) {
		if (bone_visuals[i].material.is_valid()) {
			bone_visuals[i].material->set_albedo(bone_color);
		}
	}
}

// ========================================
// Initialization
// ========================================

void HandVisualizer3D::_setup_bone_connections() {
	bone_connections.clear();

	// Thumb chain
	bone_connections.push_back({ XRHandTracker::HAND_JOINT_WRIST, XRHandTracker::HAND_JOINT_THUMB_METACARPAL });
	bone_connections.push_back({ XRHandTracker::HAND_JOINT_THUMB_METACARPAL, XRHandTracker::HAND_JOINT_THUMB_PHALANX_PROXIMAL });
	bone_connections.push_back({ XRHandTracker::HAND_JOINT_THUMB_PHALANX_PROXIMAL, XRHandTracker::HAND_JOINT_THUMB_PHALANX_DISTAL });
	bone_connections.push_back({ XRHandTracker::HAND_JOINT_THUMB_PHALANX_DISTAL, XRHandTracker::HAND_JOINT_THUMB_TIP });

	// Index finger chain
	bone_connections.push_back({ XRHandTracker::HAND_JOINT_WRIST, XRHandTracker::HAND_JOINT_INDEX_FINGER_METACARPAL });
	bone_connections.push_back({ XRHandTracker::HAND_JOINT_INDEX_FINGER_METACARPAL, XRHandTracker::HAND_JOINT_INDEX_FINGER_PHALANX_PROXIMAL });
	bone_connections.push_back({ XRHandTracker::HAND_JOINT_INDEX_FINGER_PHALANX_PROXIMAL, XRHandTracker::HAND_JOINT_INDEX_FINGER_PHALANX_INTERMEDIATE });
	bone_connections.push_back({ XRHandTracker::HAND_JOINT_INDEX_FINGER_PHALANX_INTERMEDIATE, XRHandTracker::HAND_JOINT_INDEX_FINGER_PHALANX_DISTAL });
	bone_connections.push_back({ XRHandTracker::HAND_JOINT_INDEX_FINGER_PHALANX_DISTAL, XRHandTracker::HAND_JOINT_INDEX_FINGER_TIP });

	// Middle finger chain
	bone_connections.push_back({ XRHandTracker::HAND_JOINT_WRIST, XRHandTracker::HAND_JOINT_MIDDLE_FINGER_METACARPAL });
	bone_connections.push_back({ XRHandTracker::HAND_JOINT_MIDDLE_FINGER_METACARPAL, XRHandTracker::HAND_JOINT_MIDDLE_FINGER_PHALANX_PROXIMAL });
	bone_connections.push_back({ XRHandTracker::HAND_JOINT_MIDDLE_FINGER_PHALANX_PROXIMAL, XRHandTracker::HAND_JOINT_MIDDLE_FINGER_PHALANX_INTERMEDIATE });
	bone_connections.push_back({ XRHandTracker::HAND_JOINT_MIDDLE_FINGER_PHALANX_INTERMEDIATE, XRHandTracker::HAND_JOINT_MIDDLE_FINGER_PHALANX_DISTAL });
	bone_connections.push_back({ XRHandTracker::HAND_JOINT_MIDDLE_FINGER_PHALANX_DISTAL, XRHandTracker::HAND_JOINT_MIDDLE_FINGER_TIP });

	// Ring finger chain
	bone_connections.push_back({ XRHandTracker::HAND_JOINT_WRIST, XRHandTracker::HAND_JOINT_RING_FINGER_METACARPAL });
	bone_connections.push_back({ XRHandTracker::HAND_JOINT_RING_FINGER_METACARPAL, XRHandTracker::HAND_JOINT_RING_FINGER_PHALANX_PROXIMAL });
	bone_connections.push_back({ XRHandTracker::HAND_JOINT_RING_FINGER_PHALANX_PROXIMAL, XRHandTracker::HAND_JOINT_RING_FINGER_PHALANX_INTERMEDIATE });
	bone_connections.push_back({ XRHandTracker::HAND_JOINT_RING_FINGER_PHALANX_INTERMEDIATE, XRHandTracker::HAND_JOINT_RING_FINGER_PHALANX_DISTAL });
	bone_connections.push_back({ XRHandTracker::HAND_JOINT_RING_FINGER_PHALANX_DISTAL, XRHandTracker::HAND_JOINT_RING_FINGER_TIP });

	// Pinky finger chain
	bone_connections.push_back({ XRHandTracker::HAND_JOINT_WRIST, XRHandTracker::HAND_JOINT_PINKY_FINGER_METACARPAL });
	bone_connections.push_back({ XRHandTracker::HAND_JOINT_PINKY_FINGER_METACARPAL, XRHandTracker::HAND_JOINT_PINKY_FINGER_PHALANX_PROXIMAL });
	bone_connections.push_back({ XRHandTracker::HAND_JOINT_PINKY_FINGER_PHALANX_PROXIMAL, XRHandTracker::HAND_JOINT_PINKY_FINGER_PHALANX_INTERMEDIATE });
	bone_connections.push_back({ XRHandTracker::HAND_JOINT_PINKY_FINGER_PHALANX_INTERMEDIATE, XRHandTracker::HAND_JOINT_PINKY_FINGER_PHALANX_DISTAL });
	bone_connections.push_back({ XRHandTracker::HAND_JOINT_PINKY_FINGER_PHALANX_DISTAL, XRHandTracker::HAND_JOINT_PINKY_FINGER_TIP });

	// Palm connections (optional, controlled by show_palm)
	bone_connections.push_back({ XRHandTracker::HAND_JOINT_PALM, XRHandTracker::HAND_JOINT_WRIST });
	bone_connections.push_back({ XRHandTracker::HAND_JOINT_PALM, XRHandTracker::HAND_JOINT_THUMB_METACARPAL });
	bone_connections.push_back({ XRHandTracker::HAND_JOINT_PALM, XRHandTracker::HAND_JOINT_INDEX_FINGER_METACARPAL });
	bone_connections.push_back({ XRHandTracker::HAND_JOINT_PALM, XRHandTracker::HAND_JOINT_MIDDLE_FINGER_METACARPAL });
	bone_connections.push_back({ XRHandTracker::HAND_JOINT_PALM, XRHandTracker::HAND_JOINT_RING_FINGER_METACARPAL });
	bone_connections.push_back({ XRHandTracker::HAND_JOINT_PALM, XRHandTracker::HAND_JOINT_PINKY_FINGER_METACARPAL });
}

void HandVisualizer3D::_initialize_visuals() {
	if (initialized) {
		return;
	}

	// Create joint visuals (one for each possible joint)
	joint_visuals.resize(XRHandTracker::HAND_JOINT_MAX);
	for (int i = 0; i < XRHandTracker::HAND_JOINT_MAX; i++) {
		JointVisual &visual = joint_visuals.write[i];

		// Create mesh instance
		visual.mesh_instance = memnew(MeshInstance3D);
		add_child(visual.mesh_instance);

		// Create sphere mesh
		visual.sphere_mesh.instantiate();
		visual.sphere_mesh->set_radius(joint_radius);
		visual.sphere_mesh->set_height(joint_radius * 2.0f);
		visual.sphere_mesh->set_radial_segments(8);
		visual.sphere_mesh->set_rings(4);

		// Create material
		visual.material.instantiate();
		visual.material->set_albedo(joint_color);
		visual.material->set_shading_mode(BaseMaterial3D::SHADING_MODE_UNSHADED);

		// Apply to mesh instance
		visual.mesh_instance->set_mesh(visual.sphere_mesh);
		visual.mesh_instance->set_surface_override_material(0, visual.material);
		visual.mesh_instance->set_visible(false); // Hidden by default
	}

	// Create bone visuals
	bone_visuals.resize(bone_connections.size());
	for (int i = 0; i < bone_connections.size(); i++) {
		BoneVisual &visual = bone_visuals.write[i];

		// Create mesh instance
		visual.mesh_instance = memnew(MeshInstance3D);
		add_child(visual.mesh_instance);

		// Create cylinder mesh
		visual.cylinder_mesh.instantiate();
		visual.cylinder_mesh->set_top_radius(bone_radius);
		visual.cylinder_mesh->set_bottom_radius(bone_radius);
		visual.cylinder_mesh->set_height(1.0f); // Will be scaled dynamically
		visual.cylinder_mesh->set_radial_segments(6);
		visual.cylinder_mesh->set_rings(1);

		// Create material
		visual.material.instantiate();
		visual.material->set_albedo(bone_color);
		visual.material->set_shading_mode(BaseMaterial3D::SHADING_MODE_UNSHADED);

		// Apply to mesh instance
		visual.mesh_instance->set_mesh(visual.cylinder_mesh);
		visual.mesh_instance->set_surface_override_material(0, visual.material);
		visual.mesh_instance->set_visible(false); // Hidden by default
	}

	initialized = true;
}

void HandVisualizer3D::_cleanup_visuals() {
	// Clean up joint visuals
	for (int i = 0; i < joint_visuals.size(); i++) {
		if (joint_visuals[i].mesh_instance) {
			joint_visuals[i].mesh_instance->queue_free();
			joint_visuals.write[i].mesh_instance = nullptr;
		}
	}
	joint_visuals.clear();

	// Clean up bone visuals
	for (int i = 0; i < bone_visuals.size(); i++) {
		if (bone_visuals[i].mesh_instance) {
			bone_visuals[i].mesh_instance->queue_free();
			bone_visuals.write[i].mesh_instance = nullptr;
		}
	}
	bone_visuals.clear();

	initialized = false;
}

void HandVisualizer3D::_update_tracker_reference() {
	HandTrackingServer *server = HandTrackingServer::get_singleton();
	if (!server) {
		return;
	}

	// Get the appropriate hand tracker
	if (hand == HAND_LEFT) {
		tracker = server->get_left_hand_tracker();
	} else {
		tracker = server->get_right_hand_tracker();
	}
}

// ========================================
// Update Visuals
// ========================================

void HandVisualizer3D::_update_visuals() {
	if (!initialized || tracker.is_null()) {
		return;
	}

	// Check if we have tracking data
	bool has_data = tracker->get_has_tracking_data();

	// Update joint visuals
	if (show_joints) {
		for (int i = 0; i < XRHandTracker::HAND_JOINT_MAX; i++) {
			XRHandTracker::HandJoint joint = static_cast<XRHandTracker::HandJoint>(i);
			BitField<XRHandTracker::HandJointFlags> flags = tracker->get_hand_joint_flags(joint);
			bool valid = has_data && flags.has_flag(XRHandTracker::HAND_JOINT_FLAG_POSITION_VALID);

			if (valid) {
				Transform3D transform = tracker->get_hand_joint_transform(joint);
				_update_joint_visual(i, transform, true);
			} else {
				_update_joint_visual(i, Transform3D(), false);
			}
		}
	} else {
		// Hide all joints
		for (int i = 0; i < joint_visuals.size(); i++) {
			if (joint_visuals[i].mesh_instance) {
				joint_visuals[i].mesh_instance->set_visible(false);
			}
		}
	}

	// Update bone visuals
	if (show_bones) {
		for (int i = 0; i < bone_connections.size(); i++) {
			const BoneConnection &conn = bone_connections[i];

			// Check if this is a palm connection and if we should show it
			bool is_palm_connection = (conn.from == XRHandTracker::HAND_JOINT_PALM || conn.to == XRHandTracker::HAND_JOINT_PALM);
			if (is_palm_connection && !show_palm) {
				_update_bone_visual(i, Vector3(), Vector3(), false);
				continue;
			}

			// Get flags for both joints
			BitField<XRHandTracker::HandJointFlags> flags_from = tracker->get_hand_joint_flags(conn.from);
			BitField<XRHandTracker::HandJointFlags> flags_to = tracker->get_hand_joint_flags(conn.to);

			bool valid_from = has_data && flags_from.has_flag(XRHandTracker::HAND_JOINT_FLAG_POSITION_VALID);
			bool valid_to = has_data && flags_to.has_flag(XRHandTracker::HAND_JOINT_FLAG_POSITION_VALID);

			if (valid_from && valid_to) {
				Transform3D transform_from = tracker->get_hand_joint_transform(conn.from);
				Transform3D transform_to = tracker->get_hand_joint_transform(conn.to);
				_update_bone_visual(i, transform_from.origin, transform_to.origin, true);
			} else {
				_update_bone_visual(i, Vector3(), Vector3(), false);
			}
		}
	} else {
		// Hide all bones
		for (int i = 0; i < bone_visuals.size(); i++) {
			if (bone_visuals[i].mesh_instance) {
				bone_visuals[i].mesh_instance->set_visible(false);
			}
		}
	}
}

void HandVisualizer3D::_update_joint_visual(int index, const Transform3D &transform, bool visible) {
	if (index < 0 || index >= joint_visuals.size()) {
		return;
	}

	JointVisual &visual = joint_visuals.write[index];
	if (!visual.mesh_instance) {
		return;
	}

	visual.mesh_instance->set_visible(visible);
	if (visible) {
		visual.mesh_instance->set_global_transform(transform);
	}
}

void HandVisualizer3D::_update_bone_visual(int index, const Vector3 &from, const Vector3 &to, bool visible) {
	if (index < 0 || index >= bone_visuals.size()) {
		return;
	}

	BoneVisual &visual = bone_visuals.write[index];
	if (!visual.mesh_instance) {
		return;
	}

	visual.mesh_instance->set_visible(visible);
	if (visible) {
		// Calculate midpoint
		Vector3 midpoint = (from + to) * 0.5f;

		// Calculate direction and length
		Vector3 direction = to - from;
		float length = direction.length();

		if (length < 0.0001f) {
			visual.mesh_instance->set_visible(false);
			return;
		}

		direction = direction.normalized();

		// Create transform for cylinder
		Transform3D bone_transform;
		bone_transform.origin = midpoint;

		// Rotate cylinder to align with bone direction
		// Cylinder default is along Y axis, so we need to rotate to match direction
		Vector3 up = Vector3(0, 1, 0);
		if (Math::abs(direction.dot(up)) > 0.99f) {
			// Nearly parallel, use different up vector
			up = Vector3(1, 0, 0);
		}

		Vector3 right = direction.cross(up).normalized();
		up = right.cross(direction).normalized();

		bone_transform.basis.set_columns(right, direction, up);

		// Scale to match bone length
		bone_transform.basis.scale(Vector3(1, length, 1));

		visual.mesh_instance->set_global_transform(bone_transform);
	}
}
