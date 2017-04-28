/*************************************************************************/
/*  proximity_group.cpp                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "proximity_group.h"

#include "math_funcs.h"

void ProximityGroup::clear_groups() {

	Map<StringName, uint32_t>::Element *E;

	{
		const int size = 16;
		StringName remove_list[size];
		E = groups.front();
		int num = 0;
		while (E && num < size) {

			if (E->get() != group_version) {
				remove_list[num++] = E->key();
			};

			E = E->next();
		};
		for (int i = 0; i < num; i++) {

			groups.erase(remove_list[i]);
		};
	};

	if (E) {
		clear_groups(); // call until we go through the whole list
	};
};

void ProximityGroup::update_groups() {

	if (grid_radius == Vector3(0, 0, 0))
		return;

	++group_version;

	Vector3 pos = get_global_transform().get_origin();
	Vector3 vcell = pos / cell_size;
	int cell[3] = { Math::fast_ftoi(vcell.x), Math::fast_ftoi(vcell.y), Math::fast_ftoi(vcell.z) };

	add_groups(cell, group_name, 0);

	clear_groups();
};

void ProximityGroup::add_groups(int *p_cell, String p_base, int p_depth) {

	p_base = p_base + "|";
	if (grid_radius[p_depth] == 0) {

		if (p_depth == 2) {
			_new_group(p_base);
		} else {
			add_groups(p_cell, p_base, p_depth + 1);
		};
	};

	int start = p_cell[p_depth] - grid_radius[p_depth];
	int end = p_cell[p_depth] + grid_radius[p_depth];

	for (int i = start; i <= end; i++) {

		String gname = p_base + itos(i);
		if (p_depth == 2) {
			_new_group(gname);
		} else {
			add_groups(p_cell, gname, p_depth + 1);
		};
	};
};

void ProximityGroup::_new_group(StringName p_name) {

	const Map<StringName, uint32_t>::Element *E = groups.find(p_name);
	if (!E) {
		add_to_group(p_name);
	};

	groups[p_name] = group_version;
};

void ProximityGroup::set_group_name(String p_group_name) {

	group_name = p_group_name;
};

void ProximityGroup::_notification(int what) {

	switch (what) {

		case NOTIFICATION_EXIT_TREE:
			++group_version;
			clear_groups();
			break;
		case NOTIFICATION_TRANSFORM_CHANGED:
			update_groups();
			break;
	};
};

void ProximityGroup::broadcast(String p_name, Variant p_params) {

	Map<StringName, uint32_t>::Element *E;
	E = groups.front();
	while (E) {

		get_tree()->call_group_flags(SceneTree::GROUP_CALL_DEFAULT, E->key(), "_proximity_group_broadcast", p_name, p_params);
		E = E->next();
	};
};

void ProximityGroup::_proximity_group_broadcast(String p_name, Variant p_params) {

	if (dispatch_mode == MODE_PROXY) {

		get_parent()->call(p_name, p_params);
	} else {

		emit_signal("broadcast", p_name, p_params);
	};
};

void ProximityGroup::set_dispatch_mode(int p_mode) {

	dispatch_mode = (DispatchMode)p_mode;
};

void ProximityGroup::set_grid_radius(const Vector3 &p_radius) {

	grid_radius = p_radius;
};

Vector3 ProximityGroup::get_grid_radius() const {

	return grid_radius;
};

void ProximityGroup::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_group_name", "name"), &ProximityGroup::set_group_name);
	ClassDB::bind_method(D_METHOD("broadcast", "name", "parameters"), &ProximityGroup::broadcast);
	ClassDB::bind_method(D_METHOD("set_dispatch_mode", "mode"), &ProximityGroup::set_dispatch_mode);
	ClassDB::bind_method(D_METHOD("_proximity_group_broadcast", "name", "params"), &ProximityGroup::_proximity_group_broadcast);
	ClassDB::bind_method(D_METHOD("set_grid_radius", "radius"), &ProximityGroup::set_grid_radius);
	ClassDB::bind_method(D_METHOD("get_grid_radius"), &ProximityGroup::get_grid_radius);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "grid_radius"), "set_grid_radius", "get_grid_radius");

	ADD_SIGNAL(MethodInfo("broadcast", PropertyInfo(Variant::STRING, "name"), PropertyInfo(Variant::ARRAY, "parameters")));
};

ProximityGroup::ProximityGroup() {

	group_version = 0;
	dispatch_mode = MODE_PROXY;

	grid_radius = Vector3(1, 1, 1);
	set_notify_transform(true);
};

ProximityGroup::~ProximityGroup(){

};
