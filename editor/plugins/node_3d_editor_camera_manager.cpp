/**************************************************************************/
/*  node_3d_editor_camera_manager.cpp                                     */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
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

#include "node_3d_editor_camera_manager.h"

#include "editor/editor_data.h"
#include "editor/editor_settings.h"
#include "editor/editor_undo_redo_manager.h"
#include "editor/plugins/node_3d_editor_plugin.h"
#include "editor/scene_tree_dock.h"

void Node3DEditorCameraManager::set_camera_settings(float p_fov, float p_z_near, float p_z_far) {
	fov = p_fov;
	z_near = p_z_near;
	z_far = p_z_far;
	update_camera(0.0);
}

void Node3DEditorCameraManager::reset() {
	stop_piloting();
	stop_previewing_camera();
	set_cinematic_preview_mode(false);
	set_orthogonal(false);
	cursor = Node3DEditorCameraCursor();
}

void Node3DEditorCameraManager::setup(Camera3D* p_editor_camera, Viewport* p_viewport) {
	editor_camera = p_editor_camera;
	viewport = p_viewport;
}

Node3DEditorCameraCursor Node3DEditorCameraManager::get_cursor() const {
	return cursor;
}

void Node3DEditorCameraManager::set_cursor_state(const Vector3& position, real_t x_rot, real_t y_rot, real_t distance) {
	cursor.orbit_to(x_rot, y_rot);
	cursor.move_to(position);
	cursor.move_distance_to(distance);
}

Camera3D* Node3DEditorCameraManager::get_current_camera() const {
	Camera3D* cam = get_previewing_or_cinematic_camera();
	if (cam) {
		return cam;
	}
	else {
		return editor_camera;
	}
}

Camera3D* Node3DEditorCameraManager::get_previewing_or_cinematic_camera() const {
	if (cinematic_preview_mode && cinematic_camera) {
		return cinematic_camera;
	}
	else if (previewing_camera) {
		return previewing_camera;
	}
	else {
		return nullptr;
	}
}

void Node3DEditorCameraManager::pilot_selection() {
	Node3D* selected_node = Node3DEditor::get_singleton()->get_single_selected_node();
	if (selected_node) {
		pilot(selected_node);
	}
}

void Node3DEditorCameraManager::pilot(Node3D* p_node) {
	if (p_node == nullptr || p_node == node_being_piloted) {
		return;
	}
	if (cinematic_preview_mode) {
		set_cinematic_preview_mode(false);
	}
	if (p_node != previewing_camera) {
		stop_previewing_camera();
	}
	stop_piloting();
	Camera3D* node_as_camera = Object::cast_to<Camera3D>(p_node);
	if (node_as_camera) {
		set_orthogonal(node_as_camera->get_projection() == Camera3D::PROJECTION_ORTHOGONAL);
	}
	else {
		set_orthogonal(false);
	}
	node_being_piloted = p_node;
	node_being_piloted->connect("tree_exited", callable_mp(this, &Node3DEditorCameraManager::stop_piloting));
	Transform3D transform = Transform3D(Basis(), node_being_piloted->get_global_position());
	transform.basis.set_euler(node_being_piloted->get_global_rotation());
	pilot_previous_transform = transform;
	editor_camera->set_global_transform(transform);
	cursor.set_camera_transform(transform);
	cursor.stop_interpolation(true);
	emit_signal(SNAME("camera_mode_changed"));
	update_camera();
}

void Node3DEditorCameraManager::stop_piloting() {
	if (!node_being_piloted) {
		return;
	}
	cursor.stop_interpolation(true);
	commit_pilot_transform();
	node_being_piloted->disconnect("tree_exited", callable_mp(this, &Node3DEditorCameraManager::stop_piloting));
	node_being_piloted = nullptr;
	emit_signal(SNAME("camera_mode_changed"));
	update_camera();
}

Node3D* Node3DEditorCameraManager::get_node_being_piloted() const {
	return node_being_piloted;
}

void Node3DEditorCameraManager::set_allow_pilot_previewing_camera(bool p_allow_pilot_camera) {
	allow_pilot_previewing_camera = p_allow_pilot_camera;
	if (previewing_camera) {
		bool is_piloting = node_being_piloted == previewing_camera;
		if (is_piloting && !p_allow_pilot_camera) {
			stop_piloting();
		} else if (!is_piloting && p_allow_pilot_camera) {
			pilot(previewing_camera);
		}
	}
}

void Node3DEditorCameraManager::preview_camera(Camera3D* p_camera) {
	if (p_camera == nullptr || cinematic_preview_mode || p_camera == previewing_camera) {
		return;
	}
	bool is_piloting_camera_now = node_being_piloted == p_camera;
	stop_piloting();
	previewing_camera = p_camera;
	previewing_camera->connect("tree_exiting", callable_mp(this, &Node3DEditorCameraManager::stop_previewing_camera));
	RS::get_singleton()->viewport_attach_camera(viewport->get_viewport_rid(), previewing_camera->get_camera()); //replace
	emit_signal(SNAME("camera_mode_changed"));
	if (is_piloting_camera_now || allow_pilot_previewing_camera) {
		allow_pilot_previewing_camera = true;
		pilot(previewing_camera);
	}
	update_camera();
}

Camera3D* Node3DEditorCameraManager::get_previewing_camera() const {
	return previewing_camera;
}

void Node3DEditorCameraManager::stop_previewing_camera() {
	if (!previewing_camera) {
		return;
	}
	previewing_camera->disconnect("tree_exiting", callable_mp(this, &Node3DEditorCameraManager::stop_previewing_camera));
	previewing_camera = nullptr;
	stop_piloting();
	RS::get_singleton()->viewport_attach_camera(viewport->get_viewport_rid(), editor_camera->get_camera()); //restore
	emit_signal(SNAME("camera_mode_changed"));
	update_camera();
}

void Node3DEditorCameraManager::set_cinematic_preview_mode(bool p_cinematic_mode) {
	if (cinematic_preview_mode == p_cinematic_mode) {
		return;
	}
	cinematic_preview_mode = p_cinematic_mode;
	if (p_cinematic_mode) {
		stop_previewing_camera();
		stop_piloting();
		update_cinematic_preview();
	}
	else {
		if (cinematic_camera) {
			cinematic_camera->disconnect("tree_exiting", callable_mp(this, &Node3DEditorCameraManager::stop_previews_and_pilots));
			cinematic_camera = nullptr;
		}
		RS::get_singleton()->viewport_attach_camera(viewport->get_viewport_rid(), editor_camera->get_camera()); //restore
	}
	emit_signal(SNAME("camera_mode_changed"));
	update_camera();
}

bool Node3DEditorCameraManager::is_in_cinematic_preview_mode() const {
	return cinematic_preview_mode;
}

void Node3DEditorCameraManager::set_orthogonal(bool p_orthogonal) {
	orthogonal = p_orthogonal;
	update_camera(0.0);
	if (node_being_piloted) {
		Camera3D* camera_being_piloted = Object::cast_to<Camera3D>(node_being_piloted);
		if (camera_being_piloted) {
			if (p_orthogonal && camera_being_piloted->get_projection() != Camera3D::PROJECTION_ORTHOGONAL) {
				stop_piloting();
			}
			else if (!p_orthogonal && camera_being_piloted->get_projection() == Camera3D::PROJECTION_ORTHOGONAL) {
				stop_piloting();
			}
		}
		else if (p_orthogonal) {
			stop_piloting();
		}
	}
}

bool Node3DEditorCameraManager::is_orthogonal() const {
	return orthogonal;
}

void Node3DEditorCameraManager::set_fov_scale(real_t p_scale) {
	cursor.set_fov_scale(p_scale);
}

void Node3DEditorCameraManager::set_freelook_active(bool p_active_now) {
	cursor.set_freelook_mode(p_active_now);
	if (!p_active_now) {
		commit_pilot_transform();
	}
}

void Node3DEditorCameraManager::navigation_move(float p_right, float p_forward, float p_speed) {
	const Node3DEditorViewport::FreelookNavigationScheme navigation_scheme = (Node3DEditorViewport::FreelookNavigationScheme)EditorSettings::get_singleton()->get("editors/3d/freelook/freelook_navigation_scheme").operator int();
	Vector3 forward;
	if (navigation_scheme == Node3DEditorViewport::FreelookNavigationScheme::FREELOOK_FULLY_AXIS_LOCKED) {
		// Forward/backward keys will always go straight forward/backward, never moving on the Y axis.
		forward = Vector3(0, 0, p_forward).rotated(Vector3(0, 1, 0), editor_camera->get_rotation().y);
	}
	else {
		// Forward/backward keys will be relative to the camera pitch.
		forward = editor_camera->get_transform().basis.xform(Vector3(0, 0, p_forward));
	}
	const Vector3 right = editor_camera->get_transform().basis.xform(Vector3(p_right, 0, 0));
	const Vector3 direction = forward + right;
	const Vector3 motion = direction * p_speed;
	cursor.move(motion);
}

void Node3DEditorCameraManager::navigation_freelook_move(const Vector3& p_direction, real_t p_speed, real_t p_delta) {
	cursor.move_freelook(p_direction, p_speed, p_delta);
}

void Node3DEditorCameraManager::navigation_look(const Vector2& p_axis_movement, float p_speed) {
	real_t x_rot = cursor.get_current_values().x_rot;
	real_t y_rot = cursor.get_current_values().y_rot;
	Vector3 eye_position = cursor.get_current_values().eye_position;

	x_rot += p_axis_movement.y * p_speed;
	// Clamp the Y rotation to roughly -90..90 degrees so the user can't look upside-down and end up disoriented.
	x_rot = CLAMP(x_rot, -1.57, 1.57);

	y_rot += p_axis_movement.x * p_speed;

	cursor.look_to(x_rot, y_rot);
}

void Node3DEditorCameraManager::navigation_pan(const Vector2& p_direction, float p_speed) {
	Transform3D camera_transform;
	Node3DEditorCameraCursor::Values cursor_values = cursor.get_current_values();
	camera_transform.translate_local(cursor_values.position);
	camera_transform.basis.rotate(Vector3(1, 0, 0), -cursor_values.x_rot);
	camera_transform.basis.rotate(Vector3(0, 1, 0), -cursor_values.y_rot);
	const bool invert_x_axis = EDITOR_GET("editors/3d/navigation/invert_x_axis");
	const bool invert_y_axis = EDITOR_GET("editors/3d/navigation/invert_y_axis");
	Vector3 translation(
		(invert_x_axis ? -1 : 1) * -p_direction.x * p_speed,
		(invert_y_axis ? -1 : 1) * p_direction.y * p_speed,
		0);
	const static real_t distance_default = 4.0;
	translation *= cursor_values.distance / distance_default;
	camera_transform.translate_local(translation);
	cursor.move_to(camera_transform.origin);
}

void Node3DEditorCameraManager::navigation_zoom_to_distance(float p_zoom) {
	cursor.move_distance_to(p_zoom);
}

void Node3DEditorCameraManager::navigation_orbit(const Vector2& p_rotation) {
	cursor.orbit(p_rotation.x, p_rotation.y);
}

void Node3DEditorCameraManager::orbit_view_down() {
	// Clamp rotation to roughly -90..90 degrees so the user can't look upside-down and end up disoriented.
	cursor.orbit_to(CLAMP(cursor.get_target_values().x_rot - Math_PI / 12.0, -1.57, 1.57), cursor.get_target_values().y_rot);
	stop_previews_and_pilots();
}

void Node3DEditorCameraManager::orbit_view_up() {
	// Clamp rotation to roughly -90..90 degrees so the user can't look upside-down and end up disoriented.
	cursor.orbit_to(CLAMP(cursor.get_target_values().x_rot + Math_PI / 12.0, -1.57, 1.57), cursor.get_target_values().y_rot);
	stop_previews_and_pilots();
}

void Node3DEditorCameraManager::orbit_view_right() {
	cursor.orbit_to(cursor.get_target_values().x_rot, cursor.get_target_values().y_rot - Math_PI / 12.0);
	stop_previews_and_pilots();
}

void Node3DEditorCameraManager::orbit_view_left() {
	cursor.orbit_to(cursor.get_target_values().x_rot, cursor.get_target_values().y_rot + Math_PI / 12.0);
	stop_previews_and_pilots();
}

void Node3DEditorCameraManager::orbit_view_180() {
	cursor.orbit_to(cursor.get_target_values().x_rot, cursor.get_target_values().y_rot + Math_PI);
	stop_previews_and_pilots();
}

void Node3DEditorCameraManager::view_top() {
	cursor.orbit_to(Math_PI / 2.0, 0.0);
	stop_previews_and_pilots();
}

void Node3DEditorCameraManager::view_bottom() {
	cursor.orbit_to(-Math_PI / 2.0, 0.0);
	stop_previews_and_pilots();
}

void Node3DEditorCameraManager::view_left() {
	cursor.orbit_to(0.0, Math_PI / 2.0);
	stop_previews_and_pilots();
}

void Node3DEditorCameraManager::view_right() {
	cursor.orbit_to(0.0, -Math_PI / 2.0);
	stop_previews_and_pilots();
}

void Node3DEditorCameraManager::view_front() {
	cursor.orbit_to(0.0, 0.0);
	stop_previews_and_pilots();
}

void Node3DEditorCameraManager::view_rear() {
	cursor.orbit_to(0.0, Math_PI);
	stop_previews_and_pilots();
}

void Node3DEditorCameraManager::center_to_origin() {
	cursor.move_to(Vector3(0.0, 0.0, 0.0));
	stop_previews_and_pilots();
}

void Node3DEditorCameraManager::focus_selection(const Vector3& p_center_point) {
	cursor.move_to(p_center_point);
	stop_previews_and_pilots();
}

void Node3DEditorCameraManager::update(float p_delta_time) {
	update_cinematic_preview();
	update_camera(p_delta_time);
}

void Node3DEditorCameraManager::update_camera() {
	update_camera(0.0);
}

void Node3DEditorCameraManager::update_camera(float p_interp_delta) {
	bool is_camera_orthogonal = editor_camera->get_projection() == Camera3D::PROJECTION_ORTHOGONAL;
	bool cursor_changed = false;
	if (p_interp_delta != 0.0) {
		cursor_changed = cursor.update_interpolation(p_interp_delta);
	}
	Node3DEditorCameraCursor::Values cursor_values = cursor.get_current_values();

	if (cursor_changed || p_interp_delta == 0 || is_camera_orthogonal != orthogonal) {
		editor_camera->set_global_transform(cursor.get_current_camera_transform());

		if (orthogonal) {
			float half_fov = Math::deg_to_rad(fov * cursor_values.fov_scale) / 2.0;
			float height = 2.0 * cursor_values.distance * Math::tan(half_fov);
			editor_camera->set_orthogonal(height, z_near, z_far);
		}
		else {
			editor_camera->set_perspective(fov * cursor_values.fov_scale, z_near, z_far);
		}
		update_pilot_transform();
		emit_signal(SNAME("camera_updated"));
	}
	// If not in free look mode, will commit the pilot transform each time the movement interpolation stops;
	// also, it will do nothing if the pilot didn't move at all:
	if (!cursor.get_freelook_mode() && !cursor_changed) {
		commit_pilot_transform();
	}
}

void Node3DEditorCameraManager::update_cinematic_preview() {
	Node* scene_root = SceneTreeDock::get_singleton()->get_editor_data()->get_edited_scene_root();
	if (cinematic_preview_mode && scene_root != nullptr) {
		Camera3D* cam = scene_root->get_viewport()->get_camera_3d();
		if (cam != nullptr && cam != cinematic_camera) {
			//then switch the viewport's camera to the scene's viewport camera
			if (cinematic_camera != nullptr) {
				cinematic_camera->disconnect("tree_exiting", callable_mp(this, &Node3DEditorCameraManager::stop_previews_and_pilots));
			}
			cinematic_camera = cam;
			cinematic_camera->connect("tree_exiting", callable_mp(this, &Node3DEditorCameraManager::stop_previews_and_pilots));
			RS::get_singleton()->viewport_attach_camera(viewport->get_viewport_rid(), cam->get_camera());
		}
	}
}

void Node3DEditorCameraManager::stop_previews_and_pilots() {
	stop_previewing_camera();
	stop_piloting();
	set_cinematic_preview_mode(false);
}

void Node3DEditorCameraManager::update_pilot_transform() {
	if (!node_being_piloted) {
		return;
	}
	Transform3D transform = editor_camera->get_global_transform();
	node_being_piloted->set_global_position(transform.origin);
	node_being_piloted->set_global_rotation(transform.basis.get_euler());
}

void Node3DEditorCameraManager::commit_pilot_transform() {
	if (!node_being_piloted) {
		return;
	}
	// Always commit using the cursor's transform to avoid commiting a transform that is being interpolated to smooth the movement:
	Transform3D transform_to_commit = cursor.get_target_camera_transform();
	if (transform_to_commit != pilot_previous_transform) {
		EditorUndoRedoManager* undo_redo = EditorUndoRedoManager::get_singleton();
		undo_redo->create_action(TTR("Piloting Transform"));
		undo_redo->add_do_method(this, "_undo_redo_pilot_transform", node_being_piloted, transform_to_commit);
		undo_redo->add_undo_method(this, "_undo_redo_pilot_transform", node_being_piloted, pilot_previous_transform);
		undo_redo->commit_action(false);
		pilot_previous_transform = transform_to_commit;
	}
}

void Node3DEditorCameraManager::_undo_redo_pilot_transform(Node3D* p_node, const Transform3D& p_transform) {
	p_node->set_global_transform(p_transform);
	// If the node is still in pilot mode, we need to restart it to avoid it being out of sync with the editor's camera:
	if (p_node == node_being_piloted) {
		stop_piloting();
		pilot(p_node);
	}
}

void Node3DEditorCameraManager::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_undo_redo_pilot_transform"), &Node3DEditorCameraManager::_undo_redo_pilot_transform);
	ADD_SIGNAL(MethodInfo("camera_updated"));
	ADD_SIGNAL(MethodInfo("camera_mode_changed"));
}

Node3DEditorCameraManager::Node3DEditorCameraManager() {
	editor_camera = nullptr;
	previewing_camera = nullptr;
	cinematic_camera = nullptr;
	node_being_piloted = nullptr;
	cinematic_preview_mode = false;
	allow_pilot_previewing_camera = false;
	orthogonal = false;
	z_near = 0.0;
	z_far = 0.0;
	fov = 0.0;
}

Node3DEditorCameraManager::~Node3DEditorCameraManager() {
}

//////
