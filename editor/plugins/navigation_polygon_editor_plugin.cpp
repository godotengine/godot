/**************************************************************************/
/*  navigation_polygon_editor_plugin.cpp                                  */
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

#include "navigation_polygon_editor_plugin.h"

#include "editor/editor_node.h"
#include "editor/editor_settings.h"
#include "editor/editor_undo_redo_manager.h"
#include "scene/2d/navigation_region_2d.h"
#include "scene/gui/dialogs.h"

Ref<NavigationPolygon> NavigationPolygonEditor::_ensure_navpoly() const {
	Ref<NavigationPolygon> navpoly = target_region->get_navigation_polygon();
	if (!navpoly.is_valid()) {
		navpoly = Ref<NavigationPolygon>(memnew(NavigationPolygon));
		target_region->set_navigation_polygon(navpoly);
	}
	return navpoly;
}

Node2D *NavigationPolygonEditor::_get_target_node() const {
	return target_region;
}

void NavigationPolygonEditor::_set_target_node(Node2D *p_node) {
	target_region = Object::cast_to<NavigationRegion2D>(p_node);
	if (target_region) {
		Ref<NavigationPolygon> navpoly = target_region->get_navigation_polygon();
		if (navpoly.is_valid() && navpoly->get_outline_count() > 0 && navpoly->get_polygon_count() == 0) {
			// We have outlines drawn / added by the user but no polygons were created for this navmesh yet so let's bake once immediately.
			_rebake_timer_timeout();
		}
	}
}

int NavigationPolygonEditor::_get_polygon_count() const {
	Ref<NavigationPolygon> navpoly = target_region->get_navigation_polygon();
	if (navpoly.is_valid()) {
		return navpoly->get_outline_count();
	} else {
		return 0;
	}
}

Variant NavigationPolygonEditor::_get_polygon(int p_idx) const {
	Ref<NavigationPolygon> navpoly = target_region->get_navigation_polygon();
	if (navpoly.is_valid()) {
		return navpoly->get_outline(p_idx);
	} else {
		return Variant(Vector<Vector2>());
	}
}

void NavigationPolygonEditor::_set_polygon(int p_idx, const Variant &p_polygon) const {
	Ref<NavigationPolygon> navpoly = _ensure_navpoly();
	navpoly->set_outline(p_idx, p_polygon);

	if (rebake_timer && _rebake_timer_delay >= 0.0) {
		rebake_timer->start();
	}
}

void NavigationPolygonEditor::_action_add_polygon(const Variant &p_polygon) {
	Ref<NavigationPolygon> navpoly = _ensure_navpoly();
	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->add_do_method(navpoly.ptr(), "add_outline", p_polygon);
	undo_redo->add_undo_method(navpoly.ptr(), "remove_outline", navpoly->get_outline_count());

	if (rebake_timer && _rebake_timer_delay >= 0.0) {
		rebake_timer->start();
	}
}

void NavigationPolygonEditor::_action_remove_polygon(int p_idx) {
	Ref<NavigationPolygon> navpoly = _ensure_navpoly();
	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->add_do_method(navpoly.ptr(), "remove_outline", p_idx);
	undo_redo->add_undo_method(navpoly.ptr(), "add_outline_at_index", navpoly->get_outline(p_idx), p_idx);

	if (rebake_timer && _rebake_timer_delay >= 0.0) {
		rebake_timer->start();
	}
}

void NavigationPolygonEditor::_action_set_polygon(int p_idx, const Variant &p_previous, const Variant &p_polygon) {
	Ref<NavigationPolygon> navpoly = _ensure_navpoly();
	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->add_do_method(navpoly.ptr(), "set_outline", p_idx, p_polygon);
	undo_redo->add_undo_method(navpoly.ptr(), "set_outline", p_idx, p_previous);

	if (rebake_timer && _rebake_timer_delay >= 0.0) {
		rebake_timer->start();
	}
}

bool NavigationPolygonEditor::_has_resource() const {
	return target_region && target_region->get_navigation_polygon().is_valid();
}

void NavigationPolygonEditor::_create_resource() {
	if (!target_region) {
		return;
	}

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Create Navigation Polygon"));
	undo_redo->add_do_method(target_region, "set_navigation_polygon", Ref<NavigationPolygon>(memnew(NavigationPolygon)));
	undo_redo->add_undo_method(target_region, "set_navigation_polygon", Variant(Ref<RefCounted>()));
	undo_redo->commit_action();

	_menu_option(MODE_CREATE);
}

NavigationPolygonEditor::NavigationPolygonEditor() {
	bake_hbox = memnew(HBoxContainer);
	add_child(bake_hbox);

	button_bake = memnew(Button);
	button_bake->set_flat(true);
	bake_hbox->add_child(button_bake);
	button_bake->set_toggle_mode(true);
	button_bake->set_text(TTR("Bake NavigationPolygon"));
	button_bake->set_tooltip_text(TTR("Bakes the NavigationPolygon by first parsing the scene for source geometry and then creating the navigation polygon vertices and polygons."));
	button_bake->connect(SceneStringName(pressed), callable_mp(this, &NavigationPolygonEditor::_bake_pressed));

	button_reset = memnew(Button);
	button_reset->set_flat(true);
	bake_hbox->add_child(button_reset);
	button_reset->set_text(TTR("Clear NavigationPolygon"));
	button_reset->set_tooltip_text(TTR("Clears the internal NavigationPolygon outlines, vertices and polygons."));
	button_reset->connect(SceneStringName(pressed), callable_mp(this, &NavigationPolygonEditor::_clear_pressed));

	bake_info = memnew(Label);
	bake_hbox->add_child(bake_info);

	rebake_timer = memnew(Timer);
	add_child(rebake_timer);
	rebake_timer->set_one_shot(true);
	_rebake_timer_delay = EDITOR_GET("editors/polygon_editor/auto_bake_delay");
	if (_rebake_timer_delay >= 0.0) {
		rebake_timer->set_wait_time(_rebake_timer_delay);
	}
	rebake_timer->connect("timeout", callable_mp(this, &NavigationPolygonEditor::_rebake_timer_timeout));

	err_dialog = memnew(AcceptDialog);
	add_child(err_dialog);
	target_region = nullptr;
}

void NavigationPolygonEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			button_bake->set_icon(get_editor_theme_icon(SNAME("Bake")));
			button_reset->set_icon(get_editor_theme_icon(SNAME("Reload")));
		} break;
		case EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED: {
			if (rebake_timer) {
				_rebake_timer_delay = EDITOR_GET("editors/polygon_editor/auto_bake_delay");
				if (_rebake_timer_delay >= 0.0) {
					rebake_timer->set_wait_time(_rebake_timer_delay);
				}
			}
		} break;
	}
}

void NavigationPolygonEditor::_bake_pressed() {
	if (rebake_timer) {
		rebake_timer->stop();
	}
	button_bake->set_pressed(false);

	ERR_FAIL_NULL(target_region);
	Ref<NavigationPolygon> navigation_polygon = target_region->get_navigation_polygon();
	if (navigation_polygon.is_null()) {
		err_dialog->set_text(TTR("A NavigationPolygon resource must be set or created for this node to work."));
		err_dialog->popup_centered();
		return;
	}

	target_region->bake_navigation_polygon(true);

	target_region->queue_redraw();
}

void NavigationPolygonEditor::_clear_pressed() {
	if (rebake_timer) {
		rebake_timer->stop();
	}
	if (target_region) {
		if (target_region->get_navigation_polygon().is_valid()) {
			target_region->get_navigation_polygon()->clear();
			// Needed to update all the region internals.
			target_region->set_navigation_polygon(target_region->get_navigation_polygon());
		}
	}

	button_bake->set_pressed(false);
	bake_info->set_text("");

	if (target_region) {
		target_region->queue_redraw();
	}
}

void NavigationPolygonEditor::_update_polygon_editing_state() {
	if (!target_region) {
		return;
	}

	if (target_region != nullptr && target_region->get_navigation_polygon().is_valid()) {
		bake_hbox->show();
	} else {
		bake_hbox->hide();
	}
}

void NavigationPolygonEditor::_rebake_timer_timeout() {
	if (!target_region) {
		return;
	}
	Ref<NavigationPolygon> navigation_polygon = target_region->get_navigation_polygon();
	if (!navigation_polygon.is_valid()) {
		return;
	}

	target_region->bake_navigation_polygon(true);
	target_region->queue_redraw();
}

NavigationPolygonEditorPlugin::NavigationPolygonEditorPlugin() :
		AbstractPolygon2DEditorPlugin(memnew(NavigationPolygonEditor), "NavigationRegion2D") {
}
