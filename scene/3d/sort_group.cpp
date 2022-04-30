/*************************************************************************/
/*  sort_group.cpp                                                       */
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

#include "sort_group.h"

#include "scene/3d/label_3d.h"
#include "scene/3d/sprite_3d.h"

void SortGroup::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_PARENTED: {
			int32_t sort_group_parent = 0;
			if (is_inside_tree()) {
				Node *p = get_parent();
				while (p) {
					SortGroup *sg = Object::cast_to<SortGroup>(p);
					if (sg) {
						sort_group_parent = sg->get_sort_group();
						break;
					}
					p = p->get_parent();
				}
			}
			RS::get_singleton()->sort_group_set_parent(sort_group, sort_group_parent);
		} break;
		case NOTIFICATION_UNPARENTED: {
			RS::get_singleton()->sort_group_set_parent(sort_group, 0);
		} break;
	}
}

void SortGroup::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_render_priority", "priority"), &SortGroup::set_render_priority);
	ClassDB::bind_method(D_METHOD("get_render_priority"), &SortGroup::get_render_priority);

	ClassDB::bind_method(D_METHOD("get_sort_group"), &SortGroup::get_sort_group);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "render_priority", PROPERTY_HINT_RANGE, itos(RS::MATERIAL_RENDER_PRIORITY_MIN) + "," + itos(RS::MATERIAL_RENDER_PRIORITY_MAX) + ",1"), "set_render_priority", "get_render_priority");
}

void SortGroup::_update_childs(Node *p_node) {
	for (int i = 0; i < p_node->get_child_count(); i++) {
		Node *n = p_node->get_child(i);
		if (n) {
			Label3D *lbl = Object::cast_to<Label3D>(n);
			if (lbl) {
				lbl->_queue_update();
			}
			Sprite3D *spr = Object::cast_to<Sprite3D>(n);
			if (spr) {
				spr->_queue_update();
			}

			_update_childs(n);
		}
	}
}

void SortGroup::set_render_priority(int p_priority) {
	if (render_priority != p_priority) {
		render_priority = p_priority;
		RS::get_singleton()->sort_group_set_render_priority(sort_group, p_priority);
		if (is_inside_tree()) {
			_update_childs(this);
		}
	}
}

int SortGroup::get_render_priority() const {
	return render_priority;
}

int SortGroup::get_sort_group() const {
	return sort_group;
}

SortGroup::SortGroup() {
	sort_group = RS::get_singleton()->sort_group_allocate();
}

SortGroup::~SortGroup() {
	RS::get_singleton()->sort_group_free(sort_group);
}
