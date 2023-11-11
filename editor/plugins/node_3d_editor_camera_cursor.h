/**************************************************************************/
/*  node_3d_editor_camera_cursor.h                                        */
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

#ifndef NODE_3D_EDITOR_CAMERA_CURSOR_H
#define NODE_3D_EDITOR_CAMERA_CURSOR_H

#include "scene/main/node.h"

/**
 * The invisible cursor 3D that the camera follows and rotates around in the viewport. Contains interpolation
 * for a smooth movement.
 */
class Node3DEditorCameraCursor {
public:
	struct Values {
		/** The position the cursor points to. */
		Vector3 position;

		/** The position of the camera. */
		Vector3 eye_position;

		/** X angle in radians. */
		real_t x_rot;

		/** Y angle in radians. */
		real_t y_rot;

		/** Distance between position and the eye position. */
		real_t distance;

		/** FOV scale of the camera. */
		real_t fov_scale;

		bool operator==(const Values &other) const;
		bool operator!=(const Values &other) const;

		Values();

		friend class EditorNode3DCameraCursor;
	};

	enum FreelookNavigationScheme {
		FREELOOK_DEFAULT,
		FREELOOK_PARTIALLY_AXIS_LOCKED,
		FREELOOK_FULLY_AXIS_LOCKED,
	};

private:
	Values current_values;
	Values target_values;
	bool freelook_mode = false;
	bool orthogonal = false;
	float z_near = 0.0;
	float z_far = 0.0;

public:
	/** Returns the current values being interpolated. */
	Values get_current_values() const;

	/** Returns the target values of the interpolation. */
	Values get_target_values() const;

	/** Moves the position and eye position given the motion vector. */
	void move(const Vector3 &p_delta);

	/** Moves the position to the given the point. */
	void move_to(const Vector3 &p_position);

	/** Rotates the given delta angles in radians around the cursor's position. */
	void orbit(real_t p_x, real_t p_y);

	/** Rotates to the given angles in radians around the cursor's position. */
	void orbit_to(real_t p_x, real_t p_y);

	/** Rotates the given delta angles in radians around the cursor's eye position. */
	void look(real_t p_x, real_t p_y);

	/** Rotates to the given angles in radians around the cursor's eye position. */
	void look_to(real_t p_x, real_t p_y);

	void set_fov_scale(real_t p_fov_scale);

	/** Enables the free look mode, which may affect the way the interpolation is calculated. */
	void set_freelook_mode(bool p_enabled);

	bool get_freelook_mode() const;

	/** Moves in free look mode. Free look mode must be enabled. */
	void move_freelook(const Vector3 &p_direction, real_t p_speed, real_t p_delta);

	/** Increases or decreases the distance of the eye's position to the cursor's position. */
	void move_distance(real_t p_delta);

	/** Sets the distance of the eye's position to the cursor's position. */
	void move_distance_to(real_t p_distance);

	/** Stops the interpolation at the current values or at the target values, if p_go_to_target is true. */
	void stop_interpolation(bool p_go_to_target);

	/** Calculates and updates the current values of the interpolation. Returns true if any value changed. */
	bool update_interpolation(float p_interp_delta);

	/** Sets the cursor to orthogonal view mode. Z near and far values are needed for camera transform calculations in this case. */
	void set_orthogonal(float p_z_near, float p_z_far);

	/** Sets the cursor to perspective view mode. */
	void set_perspective();

	/** Get the camera's transform given the current values of the cursor. */
	Transform3D get_current_camera_transform() const;

	/** Get the camera's transform given the target values of the cursor. */
	Transform3D get_target_camera_transform() const;

	/** Sets the values to the cursor to match the given camera's transform. */
	void set_camera_transform(const Transform3D &p_transform);

private:
	Transform3D values_to_camera_transform(const Values &p_values) const;
	void recalculate_eye_position(Values &p_values);
	void recalculate_position(Values &p_values);

public:
	Node3DEditorCameraCursor();
};

#endif // NODE_3D_EDITOR_CAMERA_CURSOR_H
