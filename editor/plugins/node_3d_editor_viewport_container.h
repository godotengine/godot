/*************************************************************************/
/*  node_3d_editor_viewport_container.h                                  */
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

#ifndef NODE_3D_EDITOR_VIEWPORT_CONTAINER_H
#define NODE_3D_EDITOR_VIEWPORT_CONTAINER_H

#include "scene/gui/container.h"

class Node3DEditorViewportContainer : public Container {
	GDCLASS(Node3DEditorViewportContainer, Container);

public:
	enum View {
		VIEW_USE_1_VIEWPORT,
		VIEW_USE_2_VIEWPORTS,
		VIEW_USE_2_VIEWPORTS_ALT,
		VIEW_USE_3_VIEWPORTS,
		VIEW_USE_3_VIEWPORTS_ALT,
		VIEW_USE_4_VIEWPORTS,
	};

private:
	View view;
	bool mouseover;
	real_t ratio_h;
	real_t ratio_v;

	bool hovering_v;
	bool hovering_h;

	bool dragging_v;
	bool dragging_h;
	Vector2 drag_begin_pos;
	Vector2 drag_begin_ratio;

	virtual void gui_input(const Ref<InputEvent> &p_event) override;

protected:
	void _notification(int p_what);

public:
	void set_view(View p_view);
	View get_view();

	Node3DEditorViewportContainer();
};

#endif // NODE_3D_EDITOR_VIEWPORT_CONTAINER_H
