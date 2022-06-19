/*************************************************************************/
/*  navigation.cpp                                                       */
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

#include "navigation.h"

#include "servers/navigation_server.h"

Vector<Vector3> Navigation::get_simple_path(const Vector3 &p_start, const Vector3 &p_end, bool p_optimize) const {
	WARN_DEPRECATED_MSG("'Navigation' node and 'Navigation.get_simple_path()' are deprecated and will be removed in a future version. Use 'NavigationServer.map_get_path()' instead.");
	return NavigationServer::get_singleton()->map_get_path(map, p_start, p_end, p_optimize, navigation_layers);
}

String Navigation::get_configuration_warning() const {
	return TTR("'Navigation' node and 'Navigation.get_simple_path()' are deprecated and will be removed in a future version. Use 'NavigationServer.map_get_path()' instead.");
}

Vector3 Navigation::get_closest_point_to_segment(const Vector3 &p_from, const Vector3 &p_to, bool p_use_collision) const {
	return NavigationServer::get_singleton()->map_get_closest_point_to_segment(map, p_from, p_to, p_use_collision);
}

Vector3 Navigation::get_closest_point(const Vector3 &p_point) const {
	return NavigationServer::get_singleton()->map_get_closest_point(map, p_point);
}

Vector3 Navigation::get_closest_point_normal(const Vector3 &p_point) const {
	return NavigationServer::get_singleton()->map_get_closest_point_normal(map, p_point);
}

RID Navigation::get_closest_point_owner(const Vector3 &p_point) const {
	return NavigationServer::get_singleton()->map_get_closest_point_owner(map, p_point);
}

void Navigation::set_up_vector(const Vector3 &p_up) {
	up = p_up;
	NavigationServer::get_singleton()->map_set_up(map, up);
}

Vector3 Navigation::get_up_vector() const {
	return up;
}

void Navigation::set_cell_size(float p_cell_size) {
	cell_size = p_cell_size;
	NavigationServer::get_singleton()->map_set_cell_size(map, cell_size);
}

void Navigation::set_cell_height(float p_cell_height) {
	cell_height = p_cell_height;
	NavigationServer::get_singleton()->map_set_cell_height(map, cell_height);
}

void Navigation::set_edge_connection_margin(float p_edge_connection_margin) {
	edge_connection_margin = p_edge_connection_margin;
	NavigationServer::get_singleton()->map_set_edge_connection_margin(map, edge_connection_margin);
}

void Navigation::set_navigation_layers(uint32_t p_navigation_layers) {
	navigation_layers = p_navigation_layers;
}

uint32_t Navigation::get_navigation_layers() const {
	return navigation_layers;
}

void Navigation::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_rid"), &Navigation::get_rid);

	ClassDB::bind_method(D_METHOD("get_simple_path", "start", "end", "optimize"), &Navigation::get_simple_path, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("get_closest_point_to_segment", "start", "end", "use_collision"), &Navigation::get_closest_point_to_segment, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("get_closest_point", "to_point"), &Navigation::get_closest_point);
	ClassDB::bind_method(D_METHOD("get_closest_point_normal", "to_point"), &Navigation::get_closest_point_normal);
	ClassDB::bind_method(D_METHOD("get_closest_point_owner", "to_point"), &Navigation::get_closest_point_owner);

	ClassDB::bind_method(D_METHOD("set_up_vector", "up"), &Navigation::set_up_vector);
	ClassDB::bind_method(D_METHOD("get_up_vector"), &Navigation::get_up_vector);

	ClassDB::bind_method(D_METHOD("set_cell_size", "cell_size"), &Navigation::set_cell_size);
	ClassDB::bind_method(D_METHOD("get_cell_size"), &Navigation::get_cell_size);

	ClassDB::bind_method(D_METHOD("set_cell_height", "cell_height"), &Navigation::set_cell_height);
	ClassDB::bind_method(D_METHOD("get_cell_height"), &Navigation::get_cell_height);

	ClassDB::bind_method(D_METHOD("set_edge_connection_margin", "margin"), &Navigation::set_edge_connection_margin);
	ClassDB::bind_method(D_METHOD("get_edge_connection_margin"), &Navigation::get_edge_connection_margin);

	ClassDB::bind_method(D_METHOD("set_navigation_layers", "navigation_layers"), &Navigation::set_navigation_layers);
	ClassDB::bind_method(D_METHOD("get_navigation_layers"), &Navigation::get_navigation_layers);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "up_vector"), "set_up_vector", "get_up_vector");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "cell_size"), "set_cell_size", "get_cell_size");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "cell_height"), "set_cell_height", "get_cell_height");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "edge_connection_margin"), "set_edge_connection_margin", "get_edge_connection_margin");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "navigation_layers", PROPERTY_HINT_LAYERS_3D_NAVIGATION), "set_navigation_layers", "get_navigation_layers");

	ADD_SIGNAL(MethodInfo("map_changed", PropertyInfo(Variant::_RID, "map")));
}

void Navigation::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			NavigationServer::get_singleton()->map_set_active(map, true);
		} break;
		case NOTIFICATION_EXIT_TREE: {
			// FIXME 3.5 with this old navigation node only
			// if the node gets deleted this exit causes annoying error prints in debug
			// It tries to deactivate a map that itself has sent a free command to the server.
			//NavigationServer::get_singleton()->map_set_active(map, false);
		} break;
	}
}

Navigation::Navigation() {
	map = NavigationServer::get_singleton()->map_create();
	NavigationServer::get_singleton()->map_set_active(map, true);
	NavigationServer::get_singleton()->map_set_up(map, get_up_vector());
	NavigationServer::get_singleton()->map_set_cell_size(map, get_cell_size());
	NavigationServer::get_singleton()->map_set_cell_height(map, get_cell_height());
	NavigationServer::get_singleton()->map_set_edge_connection_margin(map, get_edge_connection_margin());
}

Navigation::~Navigation() {
	NavigationServer::get_singleton()->free(map);
}
