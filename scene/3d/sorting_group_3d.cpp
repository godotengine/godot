/*************************************************************************/
/*  sorting_group_3d.cpp                                                 */
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

#include "sorting_group_3d.h"

bool SortingGroup3D::dirty = false;
SortingGroup3D *SortingGroup3D::manager_group = nullptr;
List<SortingGroup3D *> SortingGroup3D::root_groups;

bool SortingGroup3D::has_any() {
	// This can be used as a quick way to determine if it's worth
	// walking up the tree, looking for a parent sorting group.
	return manager_group != nullptr;
}

void SortingGroup3D::_add_root_group(SortingGroup3D *p_group) {
	root_groups.push_back(p_group);
	_set_dirty(p_group);
}

void SortingGroup3D::_remove_root_group(SortingGroup3D *p_group) {
	root_groups.erase(p_group);
}

void SortingGroup3D::_set_dirty(SortingGroup3D *p_group_changed) {
	if (dirty) {
		return;
	}

	dirty = true;

	if (manager_group == nullptr) {
		manager_group = p_group_changed;
	}
	// Update, but first give some margin for other groups to be added
	manager_group->call_deferred(SNAME("update_all_groups"));
}

void SortingGroup3D::_update_group(SortingGroup3D *p_group, int8_t p_inherited_sort_order, uint32_t &r_index) {
	struct SortByGroupOrder {
		_FORCE_INLINE_ bool operator()(const SortingGroup3D *p_a, const SortingGroup3D *p_b) const {
			return p_a->sort_order < p_b->sort_order;
		}
	};

	p_group->_set_group_index_and_inherited_order(r_index++, p_inherited_sort_order);

	// Sort children by sort group order
	p_group->child_groups.sort_custom<SortByGroupOrder>();

	for (SortingGroup3D *child_group : p_group->child_groups) {
		_update_group(child_group, p_inherited_sort_order, r_index);
	}
}

void SortingGroup3D::update_all_groups() {
	if (!dirty) {
		return;
	}

	// Start with 1, as 0 is considered outside groups
	uint32_t index = 1;

	for (SortingGroup3D *group : root_groups) {
		_update_group(group, group->sort_order, index);
	}

	dirty = false;
}

void SortingGroup3D::_set_group_index_and_inherited_order(uint32_t p_group_index, int8_t p_inherited_sort_order) {
	if (group_index == p_group_index && inherited_sort_order == p_inherited_sort_order) {
		return;
	}

	group_index = p_group_index;
	inherited_sort_order = p_inherited_sort_order;

	for (VisualInstance3D *instance : visual_instances) {
		RS::get_singleton()->instance_set_sort_group(instance->get_instance(), group_index, inherited_sort_order);
	}
}

List<SortingGroup3D *>::Element *SortingGroup3D::_add_child_group(SortingGroup3D *p_group) {
	List<SortingGroup3D *>::Element *element = child_groups.push_back(p_group);
	_set_dirty(p_group);
	return element;
}

void SortingGroup3D::_remove_child_group(List<SortingGroup3D *>::Element *p_group) {
	child_groups.erase(p_group);
}

void SortingGroup3D::_add_as_root_or_child() {
	SortingGroup3D *found_parent = nullptr;

	if (!sort_as_root) {
		Node *parent = get_parent();
		while (parent) {
			SortingGroup3D *group = Object::cast_to<SortingGroup3D>(parent);
			if (group) {
				found_parent = group;
				break;
			}
			parent = parent->get_parent();
		}
	}

	if (found_parent) {
		parent_entry = found_parent->_add_child_group(this);
		parent_group = found_parent;
	} else {
		_add_root_group(this);
	}
}

void SortingGroup3D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			if (manager_group == nullptr) {
				manager_group = this;
				// There's no other group, so this can't be a nested group.
				_add_root_group(this);
			} else {
				_add_as_root_or_child();
			}
		} break;

		case NOTIFICATION_EXIT_TREE: {
			if (manager_group == this) {
				SortingGroup3D *new_manager = nullptr;
				for (SortingGroup3D *group : root_groups) {
					if (group != this) {
						new_manager = group;
						break;
					}
				}
				manager_group = new_manager;
				if (manager_group && dirty) {
					manager_group->call_deferred(SNAME("update_all_groups"));
				}
			}

			if (parent_group) {
				parent_group->_remove_child_group(parent_entry);
				parent_entry = nullptr;
				parent_group = nullptr;
			} else {
				_remove_root_group(this);
			}
		} break;
	}
}

void SortingGroup3D::add_visual_instance(VisualInstance3D *p_instance) {
	visual_instances.push_back(p_instance);
	RS::get_singleton()->instance_set_sort_group(p_instance->get_instance(), group_index, inherited_sort_order);
}

void SortingGroup3D::remove_visual_instance(VisualInstance3D *p_instance) {
	visual_instances.erase(p_instance);
	RS::get_singleton()->instance_set_sort_group(p_instance->get_instance(), 0, 0);
}

void SortingGroup3D::set_sort_as_root(bool p_sort_as_root) {
	if (sort_as_root == p_sort_as_root) {
		return;
	}

	sort_as_root = p_sort_as_root;

	if (sort_as_root) {
		if (parent_group) {
			parent_group->_remove_child_group(parent_entry);
			parent_entry = nullptr;
			parent_group = nullptr;
		}
	} else {
		_remove_root_group(this);
	}

	_add_as_root_or_child();
}

bool SortingGroup3D::get_sort_as_root() const {
	return sort_as_root;
}

void SortingGroup3D::set_sort_order(int8_t p_sort_order) {
	if (sort_order == p_sort_order) {
		return;
	}

	sort_order = p_sort_order;
	_set_dirty(this);
}

int8_t SortingGroup3D::get_sort_order() const {
	return sort_order;
}

void SortingGroup3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_visual_instance", "instance"), &SortingGroup3D::add_visual_instance);
	ClassDB::bind_method(D_METHOD("remove_visual_instance", "instance"), &SortingGroup3D::remove_visual_instance);

	ClassDB::bind_method(D_METHOD("update_all_groups"), &SortingGroup3D::update_all_groups);

	ClassDB::bind_method(D_METHOD("set_sort_as_root", "enabled"), &SortingGroup3D::set_sort_as_root);
	ClassDB::bind_method(D_METHOD("get_sort_as_root"), &SortingGroup3D::get_sort_as_root);

	ClassDB::bind_method(D_METHOD("set_sort_order", "sort_order"), &SortingGroup3D::set_sort_order);
	ClassDB::bind_method(D_METHOD("get_sort_order"), &SortingGroup3D::get_sort_order);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "sort_as_root"), "set_sort_as_root", "get_sort_as_root");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "sort_order", PROPERTY_HINT_RANGE, "-128,127,1"), "set_sort_order", "get_sort_order");
}
