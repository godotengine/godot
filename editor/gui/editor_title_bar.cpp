/**************************************************************************/
/*  editor_title_bar.cpp                                                  */
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

#include "editor_title_bar.h"

void EditorTitleBar::_update_rects() {
	if (!is_inside_tree()) {
		return;
	}
	if (!DisplayServer::get_singleton()) {
		return;
	}
	if (!DisplayServer::get_singleton()->has_feature(DisplayServer::FEATURE_CLIENT_SIDE_DECORATIONS)) {
		return;
	}

	DisplayServer::WindowID wid = get_viewport()->get_window_id();
	for (int &id : ids) {
		DisplayServer::get_singleton()->window_remove_decoration(id, wid);
	}
	ids.clear();

	if (can_move) {
		Vector<Rect2i> rects;
		int prev_pos = 0;
		int count = get_child_count();
		for (int i = 0; i < count; i++) {
			Control *n = Object::cast_to<Control>(get_child(i));
			if (n && n->get_mouse_filter() != Control::MOUSE_FILTER_PASS) {
				int start = n->get_position().x;
				rects.push_back(Rect2(prev_pos, 0, start - prev_pos, get_size().y));
				prev_pos = start + n->get_size().x;
			}
		}
		if (prev_pos != 0) {
			rects.push_back(Rect2(prev_pos, 0, get_size().x - prev_pos, get_size().y));
		}

		for (Rect2i &rect : rects) {
			Vector<Point2> polygon_global;
			polygon_global.push_back(rect.position);
			polygon_global.push_back(rect.position + Vector2(rect.size.x, 0));
			polygon_global.push_back(rect.position + rect.size);
			polygon_global.push_back(rect.position + Vector2(0, rect.size.y));

			Transform2D t = get_global_transform();
			for (Vector2 &E : polygon_global) {
				E = t.xform(E);
			}
			int id = DisplayServer::get_singleton()->window_add_decoration(polygon_global, DisplayServer::WINDOW_DECORATION_MOVE, wid);
			ids.push_back(id);
		}
	}
}

void EditorTitleBar::_global_transform_changed() {
	_update_rects();
}

void EditorTitleBar::add_child_notify(Node *p_child) {
	Control::add_child_notify(p_child);

	Control *control = Object::cast_to<Control>(p_child);
	if (!control) {
		return;
	}

	control->connect(SceneStringName(item_rect_changed), callable_mp(this, &EditorTitleBar::_update_rects));

	_update_rects();
}

void EditorTitleBar::move_child_notify(Node *p_child) {
	Control::move_child_notify(p_child);

	if (!Object::cast_to<Control>(p_child)) {
		return;
	}

	_update_rects();
}

void EditorTitleBar::remove_child_notify(Node *p_child) {
	Control::remove_child_notify(p_child);

	Control *control = Object::cast_to<Control>(p_child);
	if (!control) {
		return;
	}

	control->disconnect(SceneStringName(item_rect_changed), callable_mp(this, &EditorTitleBar::_update_rects));

	_update_rects();
}

void EditorTitleBar::_notification(int p_notification) {
	switch (p_notification) {
		case NOTIFICATION_RESIZED: {
			_update_rects();
		} break;

		case NOTIFICATION_ENTER_TREE: {
			get_viewport()->connect("size_changed", callable_mp(this, &EditorTitleBar::_update_rects));
			_update_rects();
		} break;

		case NOTIFICATION_EXIT_TREE: {
			get_viewport()->disconnect("size_changed", callable_mp(this, &EditorTitleBar::_update_rects));
			if (!DisplayServer::get_singleton()) {
				return;
			}
			if (!DisplayServer::get_singleton()->has_feature(DisplayServer::FEATURE_CLIENT_SIDE_DECORATIONS)) {
				return;
			}
			DisplayServer::WindowID wid = get_viewport()->get_window_id();
			for (int &id : ids) {
				DisplayServer::get_singleton()->window_remove_decoration(id, wid);
			}
			ids.clear();
		} break;
	}
}

void EditorTitleBar::set_can_move_window(bool p_enabled) {
	if (can_move != p_enabled) {
		can_move = p_enabled;
		_update_rects();
	}
}

bool EditorTitleBar::get_can_move_window() const {
	return can_move;
}
