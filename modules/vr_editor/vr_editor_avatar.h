/**************************************************************************/
/*  vr_editor_avatar.h                                                    */
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

#ifndef VR_EDITOR_AVATAR_H
#define VR_EDITOR_AVATAR_H

#include "vr_collision.h"
#include "vr_keyboard.h"

#include "scene/3d/label_3d.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/3d/xr_nodes.h"
#include "scene/resources/material.h"
#include "scene/resources/primitive_meshes.h"

// VRPoke is our main UI device, it supports a raycast to interact with UI elements at distance, and supports close range touch detection
class VRPoke : public Node3D {
	GDCLASS(VRPoke, Node3D);

	// TODO add class for doing hits, now that we do raycast we can't do what we currently do.

private:
	float radius = 0.01;
	bool touch_enabled = true;
	bool ray_enabled = true;

	Color normal_color = Color(0.4, 0.4, 1.0, 0.5);
	Color touch_color = Color(0.8, 0.8, 1.0, 0.75);

	Ref<StandardMaterial3D> material;
	MeshInstance3D *sphere = nullptr;
	Ref<BoxMesh> ray_mesh;
	MeshInstance3D *cast = nullptr;

	VRCollision *last_collision = nullptr;
	Vector3 last_position;
	bool last_was_pressed[2] = { false, false };

	void _set_ray_visible_length(float p_length);

protected:
	void _notification(int p_notification);

public:
	void set_touch_enabled(bool p_enabled);
	bool get_touch_enabled() const { return touch_enabled; }

	void set_ray_enabled(bool p_enabled);
	bool get_ray_enabled() const { return ray_enabled; }

	Vector2 get_scroll();
	bool is_select();
	bool is_alt_select();

	VRPoke();
	~VRPoke();
};

// Our VRGrabDetect node detects when the user grabs an object
class VRGrabDetect : public Node3D {
	GDCLASS(VRGrabDetect, Node3D);

private:
	float radius = 0.3;

protected:
public:
	void set_radius(float radius);
	float get_radius() const { return radius; }

	void _on_grab_pressed();
	void _on_grab_released();

	VRGrabDetect();
	~VRGrabDetect();
};

class VRHand : public XRController3D {
	GDCLASS(VRHand, XRController3D);

private:
	VRPoke *poke = nullptr;
	VRGrabDetect *grab_detect = nullptr;

	bool grab_pressed = false;

protected:
	void _notification(int p_notification);

public:
	enum Hands {
		HAND_LEFT,
		HAND_RIGHT
	};

	VRPoke *get_poke() const { return poke; }
	void set_ray_enabled(bool p_enabled) { poke->set_ray_enabled(p_enabled); }

	VRHand(Hands p_hand);
	~VRHand();
};

class VREditorAvatar : public XROrigin3D {
	GDCLASS(VREditorAvatar, XROrigin3D);

private:
	XRCamera3D *camera = nullptr;
	bool camera_is_tracking = false;

	Node3D *hud_pivot = nullptr;
	Node3D *hud_root = nullptr;
	real_t hud_offset = 0.0; // offset from eye height of our hud
	real_t hud_distance = 0.5; // desired distance of our hud to our player
	bool hud_moving = true; // we are adjusting the position of the hud

	VRHand::Hands ray_active_on_hand = VRHand::HAND_RIGHT;
	VRHand *left_hand = nullptr;
	VRHand *right_hand = nullptr;
	void set_ray_active_on_hand(VRHand::Hands p_hand);
	void _on_button_pressed_on_hand(const String p_action, int p_hand);

	VRKeyboard *keyboard = nullptr;

protected:
	void _notification(int p_notification);

public:
	Node3D *get_hud_root() const { return hud_root; }

	real_t get_hud_offset() const { return hud_offset; }
	void set_hud_offset(real_t p_offset);

	real_t get_hud_distance() const { return hud_distance; }
	void set_hud_distance(real_t p_distance);

	void set_camera_cull_layers(uint32_t p_layers);
	Transform3D get_camera_transform() const { return camera->get_global_transform(); }

	VREditorAvatar();
	~VREditorAvatar();
};

#endif // VR_EDITOR_AVATAR_H
