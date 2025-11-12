/**************************************************************************/
/*  margin_container_editor_plugin.cpp                                    */
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

#include "margin_container_editor_plugin.h"

#include "editor/scene/canvas_item_editor_plugin.h"
#include "editor/themes/editor_scale.h"

void MarginContainerEditorPlugin::edit(Object *p_object) {
	if (margin_container) {
		margin_container->disconnect(SNAME("draw"), callable_mp(CanvasItemEditor::get_singleton(), &CanvasItemEditor::update_viewport));
	}

	margin_container = Object::cast_to<MarginContainer>(p_object);

	if (margin_container) {
		margin_container->connect(SNAME("draw"), callable_mp(CanvasItemEditor::get_singleton(), &CanvasItemEditor::update_viewport));
	}
	CanvasItemEditor::get_singleton()->update_viewport();
}

bool MarginContainerEditorPlugin::handles(Object *p_object) const {
	return Object::cast_to<MarginContainer>(p_object) != nullptr;
}

void MarginContainerEditorPlugin::forward_canvas_draw_over_viewport(Control *p_viewport_control) {
	if (!margin_container || !margin_container->is_visible_in_tree()) {
		return;
	}

	Transform2D xform = CanvasItemEditor::get_singleton()->get_canvas_transform() * margin_container->get_screen_transform();

	// NOTE: This color is copied from Camera2DEditor::forward_canvas_draw_over_viewport.
	// We may want to unify them somehow in the future.
	Color border_color = Color(1, 1, 0.25, 0.63);

	int border_width = Math::round(1 * EDSCALE);

	Rect2 rect = margin_container->_edit_get_rect();

	int margin_left = margin_container->get_margin_size(SIDE_LEFT);
	int margin_top = margin_container->get_margin_size(SIDE_TOP);
	int margin_right = margin_container->get_margin_size(SIDE_RIGHT);
	int margin_bottom = margin_container->get_margin_size(SIDE_BOTTOM);

	Vector2 p1, p2;

	// Calculate left margin line.
	p1 = xform.xform(rect.position + Vector2(margin_left, margin_top));
	p2 = xform.xform(rect.position + Vector2(margin_left, rect.size.y - margin_bottom));
	p_viewport_control->draw_line(p1, p2, border_color, border_width);

	// Calculate top margin line.
	p1 = xform.xform(rect.position + Vector2(margin_left, margin_top));
	p2 = xform.xform(rect.position + Vector2(rect.size.x - margin_right, margin_top));
	p_viewport_control->draw_line(p1, p2, border_color, border_width);

	// Calculate right margin line.
	p1 = xform.xform(rect.position + Vector2(rect.size.x - margin_right, margin_top));
	p2 = xform.xform(rect.position + Vector2(rect.size.x - margin_right, rect.size.y - margin_bottom));
	p_viewport_control->draw_line(p1, p2, border_color, border_width);

	// Calculate bottom margin line.
	p1 = xform.xform(rect.position + Vector2(margin_left, rect.size.y - margin_bottom));
	p2 = xform.xform(rect.position + Vector2(rect.size.x - margin_right, rect.size.y - margin_bottom));
	p_viewport_control->draw_line(p1, p2, border_color, border_width);
}
