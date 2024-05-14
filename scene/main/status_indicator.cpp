/**************************************************************************/
/*  status_indicator.cpp                                                  */
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

#include "status_indicator.h"

#include "scene/gui/popup_menu.h"

void StatusIndicator::_notification(int p_what) {
	ERR_MAIN_THREAD_GUARD;
#ifdef TOOLS_ENABLED
	if (is_part_of_edited_scene()) {
		return;
	}
#endif

	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			if (DisplayServer::get_singleton()->has_feature(DisplayServer::FEATURE_STATUS_INDICATOR)) {
				if (visible && iid == DisplayServer::INVALID_INDICATOR_ID) {
					iid = DisplayServer::get_singleton()->create_status_indicator(icon, tooltip, callable_mp(this, &StatusIndicator::_callback));
					PopupMenu *pm = Object::cast_to<PopupMenu>(get_node_or_null(menu));
					if (pm) {
						RID menu_rid = pm->bind_global_menu();
						DisplayServer::get_singleton()->status_indicator_set_menu(iid, menu_rid);
					}
				}
			}
		} break;
		case NOTIFICATION_EXIT_TREE: {
			if (DisplayServer::get_singleton()->has_feature(DisplayServer::FEATURE_STATUS_INDICATOR)) {
				if (iid != DisplayServer::INVALID_INDICATOR_ID) {
					PopupMenu *pm = Object::cast_to<PopupMenu>(get_node_or_null(menu));
					if (pm) {
						pm->unbind_global_menu();
						DisplayServer::get_singleton()->status_indicator_set_menu(iid, RID());
					}
					DisplayServer::get_singleton()->delete_status_indicator(iid);
					iid = DisplayServer::INVALID_INDICATOR_ID;
				}
			}
		} break;
		default:
			break;
	}
}

void StatusIndicator::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_tooltip", "tooltip"), &StatusIndicator::set_tooltip);
	ClassDB::bind_method(D_METHOD("get_tooltip"), &StatusIndicator::get_tooltip);
	ClassDB::bind_method(D_METHOD("set_icon", "texture"), &StatusIndicator::set_icon);
	ClassDB::bind_method(D_METHOD("get_icon"), &StatusIndicator::get_icon);
	ClassDB::bind_method(D_METHOD("set_visible", "visible"), &StatusIndicator::set_visible);
	ClassDB::bind_method(D_METHOD("is_visible"), &StatusIndicator::is_visible);
	ClassDB::bind_method(D_METHOD("set_menu", "menu"), &StatusIndicator::set_menu);
	ClassDB::bind_method(D_METHOD("get_menu"), &StatusIndicator::get_menu);
	ClassDB::bind_method(D_METHOD("get_rect"), &StatusIndicator::get_rect);

	ADD_SIGNAL(MethodInfo("pressed", PropertyInfo(Variant::INT, "mouse_button"), PropertyInfo(Variant::VECTOR2I, "mouse_position")));

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "tooltip", PROPERTY_HINT_MULTILINE_TEXT), "set_tooltip", "get_tooltip");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "icon", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D"), "set_icon", "get_icon");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "menu", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "PopupMenu"), "set_menu", "get_menu");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "visible"), "set_visible", "is_visible");
}

void StatusIndicator::_callback(MouseButton p_index, const Point2i &p_pos) {
	emit_signal(SceneStringName(pressed), p_index, p_pos);
}

void StatusIndicator::set_icon(const Ref<Texture2D> &p_icon) {
	ERR_MAIN_THREAD_GUARD;
	icon = p_icon;
	if (iid != DisplayServer::INVALID_INDICATOR_ID) {
		DisplayServer::get_singleton()->status_indicator_set_icon(iid, icon);
	}
}

Ref<Texture2D> StatusIndicator::get_icon() const {
	return icon;
}

void StatusIndicator::set_tooltip(const String &p_tooltip) {
	ERR_MAIN_THREAD_GUARD;
	tooltip = p_tooltip;
	if (iid != DisplayServer::INVALID_INDICATOR_ID) {
		DisplayServer::get_singleton()->status_indicator_set_tooltip(iid, tooltip);
	}
}

String StatusIndicator::get_tooltip() const {
	return tooltip;
}

void StatusIndicator::set_menu(const NodePath &p_menu) {
	PopupMenu *pm = Object::cast_to<PopupMenu>(get_node_or_null(menu));
	if (pm) {
		pm->unbind_global_menu();
		if (iid != DisplayServer::INVALID_INDICATOR_ID) {
			DisplayServer::get_singleton()->status_indicator_set_menu(iid, RID());
		}
	}

	menu = p_menu;

	pm = Object::cast_to<PopupMenu>(get_node_or_null(menu));
	if (pm) {
		if (iid != DisplayServer::INVALID_INDICATOR_ID) {
			RID menu_rid = pm->bind_global_menu();
			DisplayServer::get_singleton()->status_indicator_set_menu(iid, menu_rid);
		}
	}
}

NodePath StatusIndicator::get_menu() const {
	return menu;
}

void StatusIndicator::set_visible(bool p_visible) {
	ERR_MAIN_THREAD_GUARD;
	if (visible == p_visible) {
		return;
	}
	visible = p_visible;

	if (!is_inside_tree()) {
		return;
	}

#ifdef TOOLS_ENABLED
	if (is_part_of_edited_scene()) {
		return;
	}
#endif

	if (DisplayServer::get_singleton()->has_feature(DisplayServer::FEATURE_STATUS_INDICATOR)) {
		if (visible && iid == DisplayServer::INVALID_INDICATOR_ID) {
			iid = DisplayServer::get_singleton()->create_status_indicator(icon, tooltip, callable_mp(this, &StatusIndicator::_callback));
			PopupMenu *pm = Object::cast_to<PopupMenu>(get_node_or_null(menu));
			if (pm) {
				RID menu_rid = pm->bind_global_menu();
				DisplayServer::get_singleton()->status_indicator_set_menu(iid, menu_rid);
			}
		}
		if (!visible && iid != DisplayServer::INVALID_INDICATOR_ID) {
			PopupMenu *pm = Object::cast_to<PopupMenu>(get_node_or_null(menu));
			if (pm) {
				pm->unbind_global_menu();
				DisplayServer::get_singleton()->status_indicator_set_menu(iid, RID());
			}
			DisplayServer::get_singleton()->delete_status_indicator(iid);
			iid = DisplayServer::INVALID_INDICATOR_ID;
		}
	}
}

bool StatusIndicator::is_visible() const {
	return visible;
}

Rect2 StatusIndicator::get_rect() const {
	if (iid == DisplayServer::INVALID_INDICATOR_ID) {
		return Rect2();
	}
	return DisplayServer::get_singleton()->status_indicator_get_rect(iid);
}
