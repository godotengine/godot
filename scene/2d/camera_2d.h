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
#include "scene/main/viewport.h"

class Camera2D : public Node2D {
	GDCLASS(Camera2D, Node2D);

public:
	enum AnchorMode {
		ANCHOR_MODE_FIXED_TOP_LEFT,
		ANCHOR_MODE_DRAG_CENTER
	};

	enum Camera2DProcessMode {
		CAMERA2D_PROCESS_PHYSICS,
		CAMERA2D_PROCESS_IDLE
	};

private:
	bool initialized;
	Point2 current_position;
	Point2 target_position;
	Vector2 screen_size;

	ObjectID custom_viewport_id; // to check validity
	Viewport *custom_viewport;
	Viewport *viewport;

	StringName group_name;
	StringName canvas_group_name;
	RID canvas;
	Vector2 offset;
	Vector2 zoom;
	AnchorMode anchor_mode;
	bool rotating;
	bool current;
	float smoothing_speed;
	bool smoothing_enabled;
	// Smoothing can be enabled but not active in the editor.
	bool smoothing_active;
	int limit[4];
	bool limit_smoothing_enabled;
	float drag_margin[4];
	bool h_drag_enabled;
	bool v_drag_enabled;
	float h_ofs;
	float v_ofs;
	bool h_offset_changed;
	bool v_offset_changed;
	bool screen_drawing_enabled;
	bool limit_drawing_enabled;
	bool margin_drawing_enabled;

	Camera2DProcessMode process_mode;

	void _update_process_mode();
	void _setup_viewport();
	Transform2D _get_camera_transform();
	void _update_viewport();
	void _update_size();
	void _update_position();
	void _update_scroll();
	void _clear_current();
	void _set_current(bool p_current);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_offset(const Vector2 &p_offset);
	Vector2 get_offset() const;

	void set_anchor_mode(AnchorMode p_anchor_mode);
	AnchorMode get_anchor_mode() const;

	void set_rotating(bool p_rotating);
	bool is_rotating() const;

	void set_limit(Margin p_margin, int p_limit);
	int get_limit(Margin p_margin) const;

	void set_limit_smoothing_enabled(bool enable);
	bool is_limit_smoothing_enabled() const;

	void set_h_drag_enabled(bool p_enabled);
	bool is_h_drag_enabled() const;

	void set_v_drag_enabled(bool p_enabled);
	bool is_v_drag_enabled() const;

	void set_drag_margin(Margin p_margin, float p_drag_margin);
	float get_drag_margin(Margin p_margin) const;

	void set_v_offset(float p_offset);
	float get_v_offset() const;

	void set_h_offset(float p_offset);
	float get_h_offset() const;

	void set_enable_follow_smoothing(bool p_enabled);
	bool is_follow_smoothing_enabled() const;

	void set_smoothing_speed(float p_smoothing_speed);
	float get_smoothing_speed() const;

	void set_process_mode(Camera2DProcessMode p_mode);
	Camera2DProcessMode get_process_mode() const;

	void make_current();
	void clear_current();
	bool is_current() const;

	void set_zoom(const Vector2 &p_zoom);
	Vector2 get_zoom() const;

	void set_custom_viewport(Node *p_viewport);
	Node *get_custom_viewport() const;

	Point2 get_camera_screen_center() const;
	Point2 get_camera_position() const;
	void force_update_scroll();
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
VARIANT_ENUM_CAST(Camera2D::Camera2DProcessMode);

#endif // CAMERA_2D_H
