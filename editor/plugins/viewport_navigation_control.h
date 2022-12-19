/*************************************************************************/
/*  viewport_navigation_control.h                                        */
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

#ifndef VIEWPORT_NAVIGATION_CONTROL_H
#define VIEWPORT_NAVIGATION_CONTROL_H

#include "editor/editor_scale.h"
#include "node_3d_editor_viewport.h"

class ViewportNavigationControl : public Control {
	GDCLASS(ViewportNavigationControl, Control);

	Node3DEditorViewport *viewport = nullptr;
	Vector2i focused_mouse_start;
	Vector2 focused_pos;
	bool hovered = false;
	int focused_index = -1;
	Node3DEditorViewport::NavigationMode nav_mode = Node3DEditorViewport::NavigationMode::NAVIGATION_NONE;

	const float AXIS_CIRCLE_RADIUS = 30.0f * EDSCALE;

protected:
	void _notification(int p_what);
	virtual void gui_input(const Ref<InputEvent> &p_event) override;
	void _draw();
	void _on_mouse_entered();
	void _on_mouse_exited();
	void _process_click(int p_index, Vector2 p_position, bool p_pressed);
	void _process_drag(int p_index, Vector2 p_position, Vector2 p_relative_position);
	void _update_navigation();

public:
	void set_navigation_mode(Node3DEditorViewport::NavigationMode p_nav_mode);
	void set_viewport(Node3DEditorViewport *p_viewport);
};

#endif // VIEWPORT_NAVIGATION_CONTROL_H
