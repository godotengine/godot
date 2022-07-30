/*************************************************************************/
/*  camera_2d.h                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef CAMERA_2D_H
#define CAMERA_2D_H

#include "scene/2d/node_2d.h"

class Camera2D : public Node2D {
	GDCLASS(Camera2D, Node2D);

public:
	enum AnchorMode {
		ANCHOR_MODE_FIXED_TOP_LEFT,
		ANCHOR_MODE_DRAG_CENTER
	};

	enum Camera2DProcessCallback {
		CAMERA2D_PROCESS_PHYSICS,
		CAMERA2D_PROCESS_IDLE
	};

private:
	bool initialized = false;
	Point2 current_position;
	Point2 target_position;
	Vector2 screen_size;

	ObjectID custom_viewport_id; // to check validity
	Viewport *custom_viewport = nullptr;
	Viewport *viewport = nullptr;

	StringName group_name;
	StringName canvas_group_name;
	RID canvas;
	Vector2 offset;
	Vector2 zoom = Vector2(1, 1);
	Vector2 zoom_scale = Vector2(1, 1);
	AnchorMode anchor_mode = ANCHOR_MODE_DRAG_CENTER;
	bool rotating = false;
	bool current = false;
	real_t smoothing_speed = 5.0;
	bool smoothing_enabled = false;
	// Smoothing can be enabled but not active in the editor.
	bool smoothing_active = false;
	int limit[4];
	bool limit_smoothing_enabled = false;
	real_t drag_margin[4];
	bool drag_horizontal_enabled = false;
	bool drag_vertical_enabled = false;
	real_t drag_horizontal_offset = 0.0;
	real_t drag_vertical_offset = 0.0;
	bool drag_horizontal_offset_changed = false;
	bool drag_vertical_offset_changed = false;
	bool screen_drawing_enabled = true;
	bool limit_drawing_enabled = false;
	bool margin_drawing_enabled = false;

	Camera2DProcessCallback process_callback = CAMERA2D_PROCESS_IDLE;

	void _update_process_callback();
	void _setup_viewport();
	Transform2D _get_camera_transform();
	void _update_viewport();
	void _update_size();
	void _update_position();
	void _update_scroll();
	void _clear_current();

protected:
	void _notification(int p_what);
	static void _bind_methods();
	void _validate_property(PropertyInfo &p_property) const;

public:
	void set_offset(const Vector2 &p_offset);
	Vector2 get_offset() const;

	void set_anchor_mode(AnchorMode p_anchor_mode);
	AnchorMode get_anchor_mode() const;

	void set_rotating(bool p_rotating);
	bool is_rotating() const;

	void set_limit(Side p_side, int p_limit);
	int get_limit(Side p_side) const;

	void set_limit_smoothing_enabled(bool enable);
	bool is_limit_smoothing_enabled() const;

	void set_drag_horizontal_enabled(bool p_enabled);
	bool is_drag_horizontal_enabled() const;

	void set_drag_vertical_enabled(bool p_enabled);
	bool is_drag_vertical_enabled() const;

	void set_drag_margin(Side p_side, real_t p_drag_margin);
	real_t get_drag_margin(Side p_side) const;

	void set_drag_horizontal_offset(real_t p_offset);
	real_t get_drag_horizontal_offset() const;

	void set_drag_vertical_offset(real_t p_offset);
	real_t get_drag_vertical_offset() const;

	void set_enable_follow_smoothing(bool p_enabled);
	bool is_follow_smoothing_enabled() const;

	void set_smoothing_speed(real_t p_smoothing_speed);
	real_t get_smoothing_speed() const;

	void set_process_callback(Camera2DProcessCallback p_mode);
	Camera2DProcessCallback get_process_callback() const;

	void set_current(bool p_current);
	bool is_current() const;

	void set_zoom(const Vector2 &p_zoom);
	Vector2 get_zoom() const;

	void set_custom_viewport(Node *p_viewport);
	Node *get_custom_viewport() const;

	Point2 get_camera_screen_center() const;
	Point2 get_camera_position() const;
	void reset_smoothing();
	void align();

	void set_screen_drawing_enabled(bool enable);
	bool is_screen_drawing_enabled() const;

	void set_limit_drawing_enabled(bool enable);
	bool is_limit_drawing_enabled() const;

	void set_margin_drawing_enabled(bool enable);
	bool is_margin_drawing_enabled() const;

	Camera2D();
};

VARIANT_ENUM_CAST(Camera2D::AnchorMode);
VARIANT_ENUM_CAST(Camera2D::Camera2DProcessCallback);

#endif // CAMERA_2D_H
