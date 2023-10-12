/**************************************************************************/
/*  node_3d_editor_camera_manager.h                                       */
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

#ifndef NODE_3D_EDITOR_CAMERA_MANAGER_H
#define NODE_3D_EDITOR_CAMERA_MANAGER_H

#include "scene/main/node.h"
#include "node_3d_editor_camera_cursor.h"

class Camera3D;
class Node3D;

/**
* Manages the camera of the 3D editor, including features like navigating, preview, cinematic preview and pilot.
*/
class Node3DEditorCameraManager : public Node {
	GDCLASS(Node3DEditorCameraManager, Node);

public:
	struct CameraSettings {
		float z_near;
		float z_far;
		float fov;
		CameraSettings();
	};

private:
	Viewport* viewport = nullptr;
	Camera3D* editor_camera = nullptr;
	Camera3D* previewing_camera = nullptr;
	Camera3D* cinematic_camera = nullptr;
	Node3D* node_being_piloted = nullptr;
	bool cinematic_mode = false;
	Transform3D pilot_previous_transform;
	bool allow_pilot_previewing_camera = false;
	bool orthogonal = false;
	CameraSettings camera_settings;
	Node3DEditorCameraCursor cursor;

public:
	void reset();

	/** Sets the camera node being used by the editor's viewport in order to be managed by this. */
	void setup(Camera3D* p_editor_camera, Viewport* p_viewport);

	/** Returns the current camera in use (editor's camera, previewing camera or cinematic camera). */
	Camera3D* get_current_camera() const;

	/** Returns the current previewing or cinematic previewing camera, or nullptr if none. */
	Camera3D* get_previewing_or_cinematic_camera() const;

	/** Starts the pilot mode for the current single selected node, if any. */
	void pilot_selection();

	/** Starts the pilot mode with the given node. */
	void pilot(Node3D* p_node);

	/** Ends the pilot mode for whatever node is being piloted now. */
	void stop_piloting();

	/** Returns the node being piloted, or nullptr if none. */
	Node3D* get_node_being_piloted() const;

	/** Sets a flag indicating the previewing camera should enter pilot mode automatically. */
	void set_allow_pilot_previewing_camera(bool p_allow_pilot_camera);

	/** Starts the previewing mode with the given camera. */
	void preview_camera(Camera3D* p_camera);

	/** Returns the camera being previewed, or nullptr if none. */
	Camera3D* get_previewing_camera() const;

	/** Ends the previewing mode with whatever camera is being previewed now. */
	void stop_previewing_camera();

	/** Starts or ends the cinematic mode, which will automatically preview the current cinematic camera. */
	void set_cinematic_mode(bool p_cinematic_mode);

	/** Returns true if it is in cinematic preview mode. */
	bool is_in_cinematic_mode() const;

	/** Changes the camera to orthogonal or, if false, to perspective. */
	void set_orthogonal(bool p_orthogonal);

	bool is_orthogonal() const;

	/** Changes the camera to orthogonal or, if false, to perspective. */
	void set_camera_settings(const CameraSettings& p_camera_settings);

	/** Starts or ends the freelook navigation mode. */
	void set_freelook_active(bool p_active_now);

	void navigation_move(float p_right, float p_forward, float p_speed);
	void navigation_freelook_move(const Vector3& p_direction, real_t p_speed, real_t p_delta);
	void navigation_look(const Vector2& p_axis_movement, float p_speed);
	void navigation_pan(const Vector2& p_direction, float p_speed);
	void navigation_zoom_to_distance(float p_zoom);
	void navigation_orbit(const Vector2& p_rotation);

	void orbit_view_down();
	void orbit_view_up();
	void orbit_view_right();
	void orbit_view_left();
	void orbit_view_180();

	void view_top();
	void view_bottom();
	void view_left();
	void view_right();
	void view_front();
	void view_rear();

	void center_to_origin();
	void focus_selection(const Vector3& p_center_point);

	void set_fov_scale(real_t p_scale);

	/** Returns the current 3D invisible camera cursor. */
	Node3DEditorCameraCursor get_cursor() const;

	void set_cursor_state(const Vector3& position, real_t x_rot, real_t y_rot, real_t distance);

	/** Updates the camera, cursor and cinematic preview. To be called every frame. */
	void update(float p_delta_time);

	/** Updates the camera properties. */
	void update_camera();

private:
	void update_camera(float p_interp_delta);
	void update_cinematic_preview();
	void stop_previews_and_pilots();
	void update_pilot_transform();
	void commit_pilot_transform();
	void _undo_redo_pilot_transform(Node3D* p_node, const Transform3D& p_transform);

protected:
	static void _bind_methods();

public:
	Node3DEditorCameraManager();
	~Node3DEditorCameraManager();
};

#endif // NODE_3D_EDITOR_CAMERA_MANAGER_H
