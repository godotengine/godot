/*************************************************************************/
/*  navigation_3d.cpp                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "navigation_3d.h"

#include "servers/navigation_server_3d.h"

Vector<Vector3> Navigation3D::get_simple_path(const Vector3 &p_start, const Vector3 &p_end, bool p_optimize) const {
	return NavigationServer3D::get_singleton()->map_get_path(map, p_start, p_end, p_optimize);
}

Vector3 Navigation3D::get_closest_point_to_segment(const Vector3 &p_from, const Vector3 &p_to, bool p_use_collision) const {
	return NavigationServer3D::get_singleton()->map_get_closest_point_to_segment(map, p_from, p_to, p_use_collision);
}

Vector3 Navigation3D::get_closest_point(const Vector3 &p_point) const {
	return NavigationServer3D::get_singleton()->map_get_closest_point(map, p_point);
}

Vector3 Navigation3D::get_closest_point_normal(const Vector3 &p_point) const {
	return NavigationServer3D::get_singleton()->map_get_closest_point_normal(map, p_point);
}

RID Navigation3D::get_closest_point_owner(const Vector3 &p_point) const {
	return NavigationServer3D::get_singleton()->map_get_closest_point_owner(map, p_point);
}

void Navigation3D::set_up_vector(const Vector3 &p_up) {
	up = p_up;
	NavigationServer3D::get_singleton()->map_set_up(map, up);
}

Vector3 Navigation3D::get_up_vector() const {
	return up;
}

void Navigation3D::set_cell_size(float p_cell_size) {
	cell_size = p_cell_size;
	NavigationServer3D::get_singleton()->map_set_cell_size(map, cell_size);
}

void Navigation3D::set_edge_connection_margin(float p_edge_connection_margin) {
	edge_connection_margin = p_edge_connection_margin;
	NavigationServer3D::get_singleton()->map_set_edge_connection_margin(map, edge_connection_margin);
}

void Navigation3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_rid"), &Navigation3D::get_rid);

	ClassDB::bind_method(D_METHOD("get_simple_path", "start", "end", "optimize"), &Navigation3D::get_simple_path, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("get_closest_point_to_segment", "start", "end", "use_collision"), &Navigation3D::get_closest_point_to_segment, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("get_closest_point", "to_point"), &Navigation3D::get_closest_point);
	ClassDB::bind_method(D_METHOD("get_closest_point_normal", "to_point"), &Navigation3D::get_closest_point_normal);
	ClassDB::bind_method(D_METHOD("get_closest_point_owner", "to_point"), &Navigation3D::get_closest_point_owner);

	ClassDB::bind_method(D_METHOD("set_up_vector", "up"), &Navigation3D::set_up_vector);
	ClassDB::bind_method(D_METHOD("get_up_vector"), &Navigation3D::get_up_vector);

	ClassDB::bind_method(D_METHOD("set_cell_size", "cell_size"), &Navigation3D::set_cell_size);
	ClassDB::bind_method(D_METHOD("get_cell_size"), &Navigation3D::get_cell_size);

	ClassDB::bind_method(D_METHOD("set_edge_connection_margin", "margin"), &Navigation3D::set_edge_connection_margin);
	ClassDB::bind_method(D_METHOD("get_edge_connection_margin"), &Navigation3D::get_edge_connection_margin);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "up_vector"), "set_up_vector", "get_up_vector");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "cell_size"), "set_cell_size", "get_cell_size");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "edge_connection_margin"), "set_edge_connection_margin", "get_edge_connection_margin");
}

void Navigation3D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			NavigationServer3D::get_singleton()->map_set_active(map, true);
		} break;
		case NOTIFICATION_EXIT_TREE: {
			NavigationServer3D::get_singleton()->map_set_active(map, false);
		} break;
	}
}

Navigation3D::Navigation3D() {
	map = NavigationServer3D::get_singleton()->map_create();

	set_cell_size(0.3);
	set_edge_connection_margin(5.0); // Five meters, depends a lot on the agent's radius

	up = Vector3(0, 1, 0);
}

Navigation3D::~Navigation3D() {
	NavigationServer3D::get_singleton()->free(map);
}
