/**************************************************************************/
/*  proximity_group.cpp                                                   */
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

#include "proximity_group.h"

#include "core/math/math_funcs.h"

void ProximityGroup::_clear_groups() {
	Map<StringName, uint32_t>::Element *E;
	const int size = 16;

	do {
		StringName remove_list[size];
		E = groups.front();
		int num = 0;
		while (E && num < size) {
			if (E->get() != group_version) {
				remove_list[num++] = E->key();
			}

			E = E->next();
		}
		for (int i = 0; i < num; i++) {
			groups.erase(remove_list[i]);
		}
	} while (E);
}

void ProximityGroup::_update_groups() {
	if (grid_radius == Vector3(0, 0, 0)) {
		return;
	}

	++group_version;

	Vector3 pos = get_global_transform().get_origin();
	Vector3 vcell = pos / cell_size;
	int cell[3] = { Math::fast_ftoi(vcell.x), Math::fast_ftoi(vcell.y), Math::fast_ftoi(vcell.z) };

	_add_groups(cell, group_name, 0);

	_clear_groups();
}

void ProximityGroup::_add_groups(int *p_cell, String p_base, int p_depth) {
	p_base = p_base + "|";
	if (grid_radius[p_depth] == 0) {
		if (p_depth == 2) {
			_new_group(p_base);
		} else {
			_add_groups(p_cell, p_base, p_depth + 1);
		}
	}

	int start = p_cell[p_depth] - grid_radius[p_depth];
	int end = p_cell[p_depth] + grid_radius[p_depth];

	for (int i = start; i <= end; i++) {
		String gname = p_base + itos(i);
		if (p_depth == 2) {
			_new_group(gname);
		} else {
			_add_groups(p_cell, gname, p_depth + 1);
		}
	}
}

void ProximityGroup::_new_group(StringName p_name) {
	const Map<StringName, uint32_t>::Element *E = groups.find(p_name);
	if (!E) {
		add_to_group(p_name);
	}

	groups[p_name] = group_version;
}

void ProximityGroup::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_EXIT_TREE:
			++group_version;
			_clear_groups();
			break;
		case NOTIFICATION_TRANSFORM_CHANGED:
			_update_groups();
			break;
	}
}

void ProximityGroup::broadcast(String p_method, Variant p_parameters) {
	Map<StringName, uint32_t>::Element *E;
	E = groups.front();
	while (E) {
		get_tree()->call_group_flags(SceneTree::GROUP_CALL_DEFAULT, E->key(), "_proximity_group_broadcast", p_method, p_parameters);
		E = E->next();
	}
}

void ProximityGroup::_proximity_group_broadcast(String p_method, Variant p_parameters) {
	if (dispatch_mode == MODE_PROXY) {
		ERR_FAIL_COND(!is_inside_tree());
		get_parent()->call(p_method, p_parameters);
	} else {
		emit_signal("broadcast", p_method, p_parameters);
	}
}

void ProximityGroup::set_group_name(const String &p_group_name) {
	group_name = p_group_name;
}

String ProximityGroup::get_group_name() const {
	return group_name;
}

void ProximityGroup::set_dispatch_mode(DispatchMode p_mode) {
	dispatch_mode = p_mode;
}

ProximityGroup::DispatchMode ProximityGroup::get_dispatch_mode() const {
	return dispatch_mode;
}

void ProximityGroup::set_grid_radius(const Vector3 &p_radius) {
	grid_radius = p_radius;
}

Vector3 ProximityGroup::get_grid_radius() const {
	return grid_radius;
}

void ProximityGroup::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_group_name", "name"), &ProximityGroup::set_group_name);
	ClassDB::bind_method(D_METHOD("get_group_name"), &ProximityGroup::get_group_name);
	ClassDB::bind_method(D_METHOD("set_dispatch_mode", "mode"), &ProximityGroup::set_dispatch_mode);
	ClassDB::bind_method(D_METHOD("get_dispatch_mode"), &ProximityGroup::get_dispatch_mode);
	ClassDB::bind_method(D_METHOD("set_grid_radius", "radius"), &ProximityGroup::set_grid_radius);
	ClassDB::bind_method(D_METHOD("get_grid_radius"), &ProximityGroup::get_grid_radius);

	ClassDB::bind_method(D_METHOD("broadcast", "method", "parameters"), &ProximityGroup::broadcast);

	ClassDB::bind_method(D_METHOD("_proximity_group_broadcast", "method", "parameters"), &ProximityGroup::_proximity_group_broadcast);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "group_name"), "set_group_name", "get_group_name");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "dispatch_mode", PROPERTY_HINT_ENUM, "Proxy,Signal"), "set_dispatch_mode", "get_dispatch_mode");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "grid_radius"), "set_grid_radius", "get_grid_radius");

	ADD_SIGNAL(MethodInfo("broadcast", PropertyInfo(Variant::STRING, "method"), PropertyInfo(Variant::ARRAY, "parameters")));

	BIND_ENUM_CONSTANT(MODE_PROXY);
	BIND_ENUM_CONSTANT(MODE_SIGNAL);
}

ProximityGroup::ProximityGroup() {
	group_version = 0;
	dispatch_mode = MODE_PROXY;

	cell_size = 1.0;
	grid_radius = Vector3(1, 1, 1);
	set_notify_transform(true);
}
