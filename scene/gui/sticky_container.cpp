/**************************************************************************/
/*  sticky_container.cpp                                                  */
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

#include "sticky_container.h"

#include "scene/gui/scroll_container.h"

void StickyContainer::_reparent_children() {
	if (is_part_of_edited_scene()) {
		return;
	}

	if (!is_inside_tree()) {
		return;
	}

	if (!scroll_container) {
		return;
	}

	for (int c = 0; c < get_child_count(); c++) {
		Control *child = Object::cast_to<Control>(get_child(c));
		if (child) {
			managed_children.push_back(child->get_instance_id());
			scroll_container->_register_sticky_control(child->get_instance_id(), this);
		}
	}
}

Size2 StickyContainer::get_minimum_size() const {
	Size2 mins;
	if (is_part_of_edited_scene()) {
		for (int i = 0; i < get_child_count(); i++) {
			Control *child = as_sortable_control(get_child(i));
			if (child) {
				mins = mins.max(child->get_combined_minimum_size());
			}
		}
		return mins;
	}

	for (int i = 0; i < managed_children.size(); i++) {
		Control *child = Object::cast_to<Control>(ObjectDB::get_instance(managed_children[i]));
		if (child) {
			mins = mins.max(child->get_combined_minimum_size());
		}
	}
	return mins;
}

void StickyContainer::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			Control *node = this;
			while (node->get_parent_control()) {
				node = node->get_parent_control();
				if (Object::cast_to<ScrollContainer>(node)) {
					scroll_container = Object::cast_to<ScrollContainer>(node);
					_reparent_children();
					break;
				}
			}
			set_bounding_container_path(bounding_path);
		} break;

		case NOTIFICATION_EXIT_TREE: {
			if (scroll_container) {
				for (ObjectID oid : managed_children) {
					Control *child = Object::cast_to<Control>(ObjectDB::get_instance(oid));
					if (child && child->get_parent() != this) {
						child->reparent(this);
					}
				}
			}
		} break;

		case NOTIFICATION_CHILD_ORDER_CHANGED: {
			_reparent_children();
		} break;
	}
}

void StickyContainer::set_side_sticky(Side p_side, bool p_sticky) {
	if (p_sticky) {
		sticky_sides.set_flag(1 << p_side);
	} else {
		sticky_sides.clear_flag(1 << p_side);
	}
}

bool StickyContainer::is_side_sticky(Side p_side) {
	return sticky_sides.has_flag(1 << p_side);
}

void StickyContainer::set_stacking(bool p_enabled) {
	stacking = p_enabled;
}

bool StickyContainer::is_stacking() {
	return stacking;
}

void StickyContainer::set_sticky_status(StickyStatus p_status) {
	if (sticky_status != p_status) {
		sticky_status = p_status;
		emit_signal("sticky_status_changed");
	}
}

StickyContainer::StickyStatus StickyContainer::get_sticky_status() {
	return sticky_status;
}

void StickyContainer::set_bounding_container_path(const NodePath &p_path) {
	bounding_path = p_path;
	if (is_inside_tree() && !p_path.is_empty()) {
		bounding_container = Object::cast_to<Control>(get_node_or_null(p_path));
		if (scroll_container) {
			scroll_container->call("queue_sort");
		}
	}
}

NodePath StickyContainer::get_bounding_container_path() {
	return bounding_path;
}

Control *StickyContainer::get_bounding_container() {
	return bounding_container;
}

int StickyContainer::get_sticky_child_count() {
	return managed_children.size();
}

Control *StickyContainer::get_sticky_child(int p_i) {
	ERR_FAIL_INDEX_V(p_i, managed_children.size(), nullptr);
	Control *child = Object::cast_to<Control>(ObjectDB::get_instance(managed_children[p_i]));
	return child;
}

void StickyContainer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_side_sticky", "side", "enabled"), &StickyContainer::set_side_sticky);
	ClassDB::bind_method(D_METHOD("is_side_sticky", "side"), &StickyContainer::is_side_sticky);
	ClassDB::bind_method(D_METHOD("set_stacking", "enabled"), &StickyContainer::set_stacking);
	ClassDB::bind_method(D_METHOD("is_stacking"), &StickyContainer::is_stacking);
	ClassDB::bind_method(D_METHOD("set_bounding_container_path", "container_path"), &StickyContainer::set_bounding_container_path);
	ClassDB::bind_method(D_METHOD("get_bounding_container_path"), &StickyContainer::get_bounding_container_path);
	ClassDB::bind_method(D_METHOD("get_sticky_status"), &StickyContainer::get_sticky_status);
	ClassDB::bind_method(D_METHOD("get_sticky_child_count"), &StickyContainer::get_sticky_child_count);
	ClassDB::bind_method(D_METHOD("get_sticky_child", "index"), &StickyContainer::get_sticky_child);

	ADD_SIGNAL(MethodInfo("sticky_status_changed"));

	BIND_ENUM_CONSTANT(STATUS_NORMAL);
	BIND_ENUM_CONSTANT(STATUS_STICKING);
	BIND_ENUM_CONSTANT(STATUS_HIDDEN);

	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "stick_to_left"), "set_side_sticky", "is_side_sticky", SIDE_LEFT);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "stick_to_top"), "set_side_sticky", "is_side_sticky", SIDE_TOP);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "stick_to_right"), "set_side_sticky", "is_side_sticky", SIDE_RIGHT);
	ADD_PROPERTYI(PropertyInfo(Variant::BOOL, "stick_to_bottom"), "set_side_sticky", "is_side_sticky", SIDE_BOTTOM);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "stack"), "set_stacking", "is_stacking");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "bounding_container"), "set_bounding_container_path", "get_bounding_container_path");
}

StickyContainer::StickyContainer() {}
