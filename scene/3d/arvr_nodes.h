/*************************************************************************/
/*  arvr_nodes.h                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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

#ifndef ARVR_NODES_H
#define ARVR_NODES_H

#include "scene/3d/camera.h"
#include "scene/3d/spatial.h"
#include "servers/arvr/arvr_positional_tracker.h"

/**
	@author Bastiaan Olij <mux213@gmail.com>
**/

/*
	ARVRCamera is a subclass of camera which will register itself with its parent ARVROrigin and as a result is automatically positioned
*/
class ARVRCamera : public Camera {

	GDCLASS(ARVRCamera, Camera);

protected:
	void _notification(int p_what);

public:
	String get_configuration_warning() const;

	virtual Vector3 project_local_ray_normal(const Point2 &p_pos) const;
	virtual Point2 unproject_position(const Vector3 &p_pos) const;
	virtual Vector3 project_position(const Point2 &p_point) const;
	virtual Vector<Plane> get_frustum() const;

	ARVRCamera();
	~ARVRCamera();
};

/*
	ARVRController is a helper node that automatically updates it's position based on tracker data.

	It must be a child node of our ARVROrigin node
*/

class ARVRController : public Spatial {

	GDCLASS(ARVRController, Spatial);

private:
	int controller_id;
	bool is_active;
	int button_states;

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_controller_id(int p_controller_id);
	int get_controller_id(void) const;
	String get_controller_name(void) const;

	int get_joystick_id() const;
	int is_button_pressed(int p_button) const;
	float get_joystick_axis(int p_axis) const;

	bool get_is_active() const;
	ARVRPositionalTracker::TrackerHand get_hand() const;

	String get_configuration_warning() const;

	ARVRController();
	~ARVRController();
};

/*
	ARVRAnchor is a helper node that automatically updates it's position based on anchor data, it represents a real world location.
	It must be a child node of our ARVROrigin node
*/

class ARVRAnchor : public Spatial {
	GDCLASS(ARVRAnchor, Spatial);

private:
	int anchor_id;
	bool is_active;
	Vector3 size;

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_anchor_id(int p_anchor_id);
	int get_anchor_id(void) const;
	String get_anchor_name(void) const;

	bool get_is_active() const;
	Vector3 get_size() const;

	Plane get_plane() const;

	String get_configuration_warning() const;

	ARVRAnchor();
	~ARVRAnchor();
};

/*
	ARVROrigin is special spatial node that acts as our origin point mapping our real world center of our tracking volume into our virtual world.

	It is this point that you will move around the world as the player 'moves while standing still', i.e. the player moves through teleporting or controller inputs as opposed to physically moving.

	Our camera and controllers will always be child nodes and thus place relative to this origin point.
	This node will automatically locate any camera child nodes and update its position while our ARVRController node will handle tracked controllers.
*/
class ARVROrigin : public Spatial {

	GDCLASS(ARVROrigin, Spatial);

private:
	ARVRCamera *tracked_camera;

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	String get_configuration_warning() const;

	void set_tracked_camera(ARVRCamera *p_tracked_camera);
	void clear_tracked_camera_if(ARVRCamera *p_tracked_camera);

	float get_world_scale() const;
	void set_world_scale(float p_world_scale);

	ARVROrigin();
	~ARVROrigin();
};

#endif /* ARVR_NODES_H */
