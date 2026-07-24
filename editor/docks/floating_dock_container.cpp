/**************************************************************************/
/*  floating_dock_container.cpp                                           */
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

#include "floating_dock_container.h"

#include "core/object/callable_mp.h"
#include "editor/docks/editor_dock_manager.h"
#include "editor/editor_node.h"
#include "editor/themes/editor_scale.h"
#include "servers/display/display_server.h"

void FloatingDockContainer::_update_window_title() {
	EditorDock *current_dock = get_dock(get_current_tab());
	if (current_dock) {
		window->set_title(vformat(TTR("%s - Godot Engine"), TTR(current_dock->get_display_title())));
	}
}

void FloatingDockContainer::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_POSTINITIALIZE: {
			set_anchors_and_offsets_preset(PRESET_FULL_RECT);
			window->add_child(this);
			drag_hint->reparent(window);
			connect("tab_changed", callable_mp(this, &FloatingDockContainer::_update_window_title).unbind(1));
		} break;

		case NOTIFICATION_TRANSLATION_CHANGED: {
			_update_window_title();
		} break;
	}
}

void FloatingDockContainer::dock_added(EditorDock *p_dock) {
	owned_docks.insert(p_dock);
	if (owned_docks.size() == 1) {
		_update_window_title();
	}
}

void FloatingDockContainer::dock_removed(EditorDock *p_dock) {
	owned_docks.erase(p_dock);
	if (owned_docks.is_empty()) {
		EditorDockManager::get_singleton()->destroy_floating_slot(this);
	}
}

void FloatingDockContainer::dock_focused(EditorDock *p_dock, bool p_was_visible) {
	get_window()->grab_focus();
	get_tab_bar()->grab_focus();
}

void FloatingDockContainer::update_visibility() {
	window->set_visible(get_tab_count() > 0);
}

Rect2 FloatingDockContainer::get_drag_hint_rect() const {
	return DockTabContainer::get_drag_hint_rect().grow(-2 * EDSCALE);
}

bool FloatingDockContainer::can_dock_float(EditorDock *p_dock, String &r_float_info) {
	if (owned_docks.size() == 1) {
		r_float_info = TTRC("Can't detach a single floating dock.");
		return false;
	}
	return DockTabContainer::can_dock_float(p_dock, r_float_info);
}

void FloatingDockContainer::add_child_notify(Node *p_child) {
	DockTabContainer::add_child_notify(p_child);
	callable_mp(this, &FloatingDockContainer::_update_window_title).call_deferred();
}

Dictionary FloatingDockContainer::get_window_layout() const {
	Dictionary window_layout;
	window_layout["window_rect"] = Rect2i(window->get_position(), window->get_size());
	window_layout["screen_idx"] = window->get_current_screen();
	window_layout["screen_rect"] = DisplayServer::get_singleton()->screen_get_usable_rect(window->get_current_screen());
	return window_layout;
}

Rect2i FloatingDockContainer::get_window_rect_from_layout(const Dictionary &p_layout) {
	Rect2i window_rect = p_layout.get("window_rect", Rect2i());
	int screen = p_layout.get("screen_idx", -1);
	Rect2i restored_screen_rect = p_layout.get("screen_rect", Rect2i());

	if (DisplayServer::get_singleton()->has_feature(DisplayServerEnums::FEATURE_SELF_FITTING_WINDOWS)) {
		window_rect = Rect2i();
		restored_screen_rect = Rect2i();
	}

	if (screen < 0 || screen >= DisplayServer::get_singleton()->get_screen_count()) {
		// Fallback to the main window screen if the saved screen is not available.
		screen = EditorNode::get_singleton()->get_window()->get_current_screen();
	}

	Rect2i real_screen_rect = DisplayServer::get_singleton()->screen_get_usable_rect(screen);
	if (restored_screen_rect == Rect2i()) {
		// Fallback to the target screen rect.
		restored_screen_rect = real_screen_rect;
	}

	if (window_rect == Rect2i()) {
		// Fallback to a standard rect.
		window_rect = Rect2i(restored_screen_rect.position + restored_screen_rect.size / 4, restored_screen_rect.size / 2);
	}

	// Adjust the window rect size in case the resolution changes.
	Vector2 screen_ratio = Vector2(real_screen_rect.size) / Vector2(restored_screen_rect.size);

	// The screen positioning may change, so remove the original screen position.
	window_rect.position -= restored_screen_rect.position;
	window_rect = Rect2i(window_rect.position * screen_ratio, window_rect.size * screen_ratio);
	window_rect.position += real_screen_rect.position;

	return Rect2i(window_rect.position, window_rect.size);
}

FloatingDockContainer::FloatingDockContainer(int p_slot) :
		DockTabContainer(p_slot) {
	layout = EditorDock::DOCK_LAYOUT_FLOATING;

	window = memnew(Window);
	window->set_wrap_controls(true);
	window->set_propagate_shortcuts_to_parent(true);
	window->hide();
}
