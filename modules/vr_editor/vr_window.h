/**************************************************************************/
/*  vr_window.h                                                           */
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

#ifndef VR_WINDOW_H
#define VR_WINDOW_H

#include "scene/3d/mesh_instance_3d.h"
#include "scene/3d/node_3d.h"
#include "scene/main/viewport.h"

#include "vr_collision.h"

class VRCollisionWindow : public VRCollision {
	GDCLASS(VRCollisionWindow, VRCollision);

private:
	Size2 size;
	float curve_depth = 0.0;

protected:
public:
	Size2 get_size() const { return size; }
	void set_size(Size2 p_size) { size = p_size; }

	float get_curve_depth() const { return curve_depth; }
	void set_curve_depth(float p_curve_depth) { curve_depth = p_curve_depth; }

	virtual bool raycast(const Vector3 &p_global_origin, const Vector3 &p_global_dir, Vector3 &r_position) override;
	virtual bool within_sphere(const Vector3 &p_global_origin, float p_radius, Vector3 &r_position) override;
};

class VRWindow : public Node3D {
	GDCLASS(VRWindow, Node3D);

private:
	bool window_is_visible = true;
	bool transparent_background = false;

	Size2 viewport_size; // original viewport size (but as float so we can do calculations)
	real_t viewport_scale = 0.00075; // viewport 2D size to 3D size ratio
	Size2 mesh_size; // mesh size (viewport_size * viewport_scale)
	float curve_depth = 0.0; // Depth of our curve

	float press_distance = 0.01; // distance from Window to finger where we assume the finger is pressing on the screen (i.e. click)

	Vector2 last_position;
	Vector2 _calc_mouse_position(const Vector3 &p_position);

	VRCollisionWindow *collision = nullptr;
	BitField<MouseButtonMask> buttons_state;

	void _on_scroll_input(const Vector2 &p_position, MouseButton p_wheel_button, float p_delta);

protected:
	SubViewport *subviewport = nullptr; // viewport to which we render our content
	Ref<Shader> opaque_shader; // opaque version of our shader
	Ref<Shader> transparent_shader; // transparent version of our shader
	Ref<ShaderMaterial> material; // material used to draw our window
	MeshInstance3D *mesh_instance = nullptr; // mesh instance used to show our window

	virtual void input(const Ref<InputEvent> &p_event) override;

	void _on_interact_enter(const Vector3 &p_position);
	void _on_interact_moved(const Vector3 &p_position, float p_pressure);
	void _on_interact_leave(const Vector3 &p_position);
	void _on_interact_pressed(const Vector3 &p_position, MouseButton p_button);
	void _on_interact_scrolled(const Vector3 &p_position, const Vector2 p_scroll_delta);
	void _on_interact_released(const Vector3 &p_position, MouseButton p_button);

public:
	SubViewport *get_scene_root() { return subviewport; }

	bool get_window_is_visible() const { return window_is_visible; }
	void set_window_is_visible(bool p_visible);

	bool get_transparent_background() const { return transparent_background; }
	void set_transparent_background(bool p_is_transparent);

	float get_curve_depth() const { return curve_depth; }
	void set_curve_depth(float p_curve_depth);

	VRWindow(Size2i p_viewport_size, real_t p_viewport_scale = 0.00075);
	~VRWindow();
};

#endif // VR_WINDOW_H
