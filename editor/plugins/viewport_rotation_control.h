/*************************************************************************/
/*  viewport_rotation_control.h                                          */
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

#ifndef VIEWPORT_ROTATION_CONTROL_H
#define VIEWPORT_ROTATION_CONTROL_H

#include "editor/editor_scale.h"
#include "node_3d_editor_viewport.h"

class ViewportRotationControl : public Control {
	GDCLASS(ViewportRotationControl, Control);

	struct Axis2D {
		Vector2i screen_point;
		float z_axis = -99.0;
		int axis = -1;
	};

	struct Axis2DCompare {
		_FORCE_INLINE_ bool operator()(const Axis2D &l, const Axis2D &r) const {
			return l.z_axis < r.z_axis;
		}
	};

	Node3DEditorViewport *viewport = nullptr;
	Vector<Color> axis_colors;
	Vector<int> axis_menu_options;
	Vector2i orbiting_mouse_start;
	int orbiting_index = -1;
	int focused_axis = -2;

	const float AXIS_CIRCLE_RADIUS = 8.0f * EDSCALE;

protected:
	void _notification(int p_what);
	virtual void gui_input(const Ref<InputEvent> &p_event) override;
	void _draw();
	void _draw_axis(const Axis2D &p_axis);
	void _get_sorted_axis(Vector<Axis2D> &r_axis);
	void _update_focus();
	void _on_mouse_exited();
	void _process_click(int p_index, Vector2 p_position, bool p_pressed);
	void _process_drag(Ref<InputEventWithModifiers> p_event, int p_index, Vector2 p_position, Vector2 p_relative_position);

public:
	void set_viewport(Node3DEditorViewport *p_viewport);
};

#endif // VIEWPORT_ROTATION_CONTROL_H
