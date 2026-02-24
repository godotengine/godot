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

#include "editor/docks/editor_dock_manager.h"
#include "editor/editor_node.h"

void FloatingDockContainer::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_POSTINITIALIZE: {
			set_anchors_and_offsets_preset(PRESET_FULL_RECT);
			window->add_child(this);
			drag_hint->reparent(window);
		} break;

		case NOTIFICATION_TRANSLATION_CHANGED: {
			window->set_title(vformat(TTR("%s - Godot Engine"), TTR("Floating Docks")));
		} break;
	}
}

void FloatingDockContainer::dock_added(EditorDock *p_dock) {
	owned_docks.insert(p_dock);
}

void FloatingDockContainer::dock_removed(EditorDock *p_dock) {
	owned_docks.erase(p_dock);
	if (owned_docks.is_empty()) {
		EditorDockManager::get_singleton()->destroy_floating_slot(this);
	}
}

void FloatingDockContainer::update_visibility() {
	window->set_visible(get_tab_count() > 0);
}

FloatingDockContainer::FloatingDockContainer(int p_slot) :
		DockTabContainer(p_slot) {
	layout = EditorDock::DOCK_LAYOUT_FLOATING;

	window = memnew(Window);
	window->set_wrap_controls(true);
	window->hide();
}
