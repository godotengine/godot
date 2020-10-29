/*************************************************************************/
/*  xr_nodes.h                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef XR_NODES_H
#define XR_NODES_H

#include "scene/3d/camera_3d.h"
#include "scene/3d/node_3d.h"
#include "scene/resources/mesh.h"
#include "servers/xr/xr_positional_tracker.h"

/**
	@author Bastiaan Olij <mux213@gmail.com>
**/

/*
	XRCamera is a subclass of camera which will register itself with its parent XROrigin and as a result is automatically positioned
*/
class XRCamera3D : public Camera3D {
	GDCLASS(XRCamera3D, Camera3D);

protected:
	void _notification(int p_what);

public:
	TypedArray<String> get_configuration_warnings() const override;

	virtual Vector3 project_local_ray_normal(const Point2 &p_pos) const override;
	virtual Point2 unproject_position(const Vector3 &p_pos) const override;
	virtual Vector3 project_position(const Point2 &p_point, float p_z_depth) const override;
	virtual Vector<Plane> get_frustum() const override;

	XRCamera3D() {}
	~XRCamera3D() {}
};

/*
	XRController3D is a helper node that automatically updates its position based on tracker data.

	It must be a child node of our XROrigin node
*/

class XRController3D : public Node3D {
	GDCLASS(XRController3D, Node3D);

private:
	int controller_id = 1;
	bool is_active = true;
	int button_states = 0;
	Ref<Mesh> mesh;

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_controller_id(int p_controller_id);
	int get_controller_id() const;
	String get_controller_name() const;

	int get_joystick_id() const;
	bool is_button_pressed(int p_button) const;
	float get_joystick_axis(int p_axis) const;

	real_t get_rumble() const;
	void set_rumble(real_t p_rumble);

	bool get_is_active() const;
	XRPositionalTracker::TrackerHand get_tracker_hand() const;

	Ref<Mesh> get_mesh() const;

	TypedArray<String> get_configuration_warnings() const override;

	XRController3D() {}
	~XRController3D() {}
};

/*
	XRAnchor3D is a helper node that automatically updates its position based on anchor data, it represents a real world location.
	It must be a child node of our XROrigin3D node
*/

class XRAnchor3D : public Node3D {
	GDCLASS(XRAnchor3D, Node3D);

private:
	int anchor_id = 1;
	bool is_active = true;
	Vector3 size;
	Ref<Mesh> mesh;

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_anchor_id(int p_anchor_id);
	int get_anchor_id() const;
	String get_anchor_name() const;

	bool get_is_active() const;
	Vector3 get_size() const;

	Plane get_plane() const;

	Ref<Mesh> get_mesh() const;

	TypedArray<String> get_configuration_warnings() const override;

	XRAnchor3D() {}
	~XRAnchor3D() {}
};

/*
	XROrigin3D is special spatial node that acts as our origin point mapping our real world center of our tracking volume into our virtual world.

	It is this point that you will move around the world as the player 'moves while standing still', i.e. the player moves through teleporting or controller inputs as opposed to physically moving.

	Our camera and controllers will always be child nodes and thus place relative to this origin point.
	This node will automatically locate any camera child nodes and update its position while our XRController3D node will handle tracked controllers.
*/
class XROrigin3D : public Node3D {
	GDCLASS(XROrigin3D, Node3D);

private:
	XRCamera3D *tracked_camera = nullptr;

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	TypedArray<String> get_configuration_warnings() const override;

	void set_tracked_camera(XRCamera3D *p_tracked_camera);
	void clear_tracked_camera_if(XRCamera3D *p_tracked_camera);

	float get_world_scale() const;
	void set_world_scale(float p_world_scale);

	XROrigin3D() {}
	~XROrigin3D() {}
};

#endif /* XR_NODES_H */
