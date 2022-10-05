/**************************************************************************/
/*  vr_collision.h                                                        */
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

#ifndef VR_COLLISION_H
#define VR_COLLISION_H

#include "scene/3d/node_3d.h"

// VRCollision is a helper class for our poke and grab detection nodes to test if we're interacting with a shape.
// We can't use the physics engine in the editor so this is a bare minimum implementation suitable specifically
// for our VR interface. Note that it is inherited from Node3D so it can be added to our scene tree.

class VRCollision : public Node3D {
	GDCLASS(VRCollision, Node3D);

private:
	bool enabled = true;
	bool can_interact = true;
	bool can_grab = false;

	static Vector<VRCollision *> collisions;

protected:
	static void _bind_methods();

public:
	bool is_enabled() const { return enabled; }
	void set_enabled(bool p_enabled) { enabled = p_enabled; }

	bool get_can_interact() const { return can_interact; }
	void set_can_interact(bool p_enable) { can_interact = p_enable; }

	bool get_can_grab() const { return can_grab; }
	void set_can_grab(bool p_enable) { can_grab = p_enable; }

	virtual bool raycast(const Vector3 &p_global_origin, const Vector3 &p_global_dir, Vector3 &r_position) = 0;
	virtual bool within_sphere(const Vector3 &p_global_origin, float p_radius, Vector3 &r_position) = 0;

	static Vector<VRCollision *> get_hit_tests(bool p_inc_can_interact, bool p_inc_can_grab);

	enum InteractType {
		INTERACT_ENTER,
		INTERACT_MOVED,
		INTERACT_LEAVE,
		INTERACT_PRESSED,
		INTERACT_RELEASED
	};

	void _on_interact_enter(const Vector3 &p_position);
	void _on_interact_moved(const Vector3 &p_position, float p_pressure);
	void _on_interact_leave(const Vector3 &p_position);
	void _on_interact_pressed(const Vector3 &p_position, MouseButton p_button);
	void _on_interact_released(const Vector3 &p_position, MouseButton p_button);
	void _on_interact_scrolled(const Vector3 &p_position, const Vector2 &p_scroll_delta);

	VRCollision();
	virtual ~VRCollision();
};

#endif // VR_COLLISION_H
