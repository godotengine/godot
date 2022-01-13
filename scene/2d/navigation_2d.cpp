/*************************************************************************/
/*  navigation_2d.cpp                                                    */
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

#include "navigation_2d.h"

#include "servers/navigation_2d_server.h"

void Navigation2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_rid"), &Navigation2D::get_rid);

	ClassDB::bind_method(D_METHOD("get_simple_path", "start", "end", "optimize"), &Navigation2D::get_simple_path, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("get_closest_point", "to_point"), &Navigation2D::get_closest_point);
	ClassDB::bind_method(D_METHOD("get_closest_point_owner", "to_point"), &Navigation2D::get_closest_point_owner);

	ClassDB::bind_method(D_METHOD("set_cell_size", "cell_size"), &Navigation2D::set_cell_size);
	ClassDB::bind_method(D_METHOD("get_cell_size"), &Navigation2D::get_cell_size);

	ClassDB::bind_method(D_METHOD("set_edge_connection_margin", "margin"), &Navigation2D::set_edge_connection_margin);
	ClassDB::bind_method(D_METHOD("get_edge_connection_margin"), &Navigation2D::get_edge_connection_margin);

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "cell_size"), "set_cell_size", "get_cell_size");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "edge_connection_margin"), "set_edge_connection_margin", "get_edge_connection_margin");
}

void Navigation2D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			Navigation2DServer::get_singleton()->map_set_active(map, true);
		} break;
		case NOTIFICATION_EXIT_TREE: {
			Navigation2DServer::get_singleton()->map_set_active(map, false);
		} break;
	}
}

void Navigation2D::set_cell_size(float p_cell_size) {
	cell_size = p_cell_size;
	Navigation2DServer::get_singleton()->map_set_cell_size(map, cell_size);
}

void Navigation2D::set_edge_connection_margin(float p_edge_connection_margin) {
	edge_connection_margin = p_edge_connection_margin;
	Navigation2DServer::get_singleton()->map_set_edge_connection_margin(map, edge_connection_margin);
}

Vector<Vector2> Navigation2D::get_simple_path(const Vector2 &p_start, const Vector2 &p_end, bool p_optimize) const {
	return Navigation2DServer::get_singleton()->map_get_path(map, p_start, p_end, p_optimize);
}

Vector2 Navigation2D::get_closest_point(const Vector2 &p_point) const {
	return Navigation2DServer::get_singleton()->map_get_closest_point(map, p_point);
}

RID Navigation2D::get_closest_point_owner(const Vector2 &p_point) const {
	return Navigation2DServer::get_singleton()->map_get_closest_point_owner(map, p_point);
}

Navigation2D::Navigation2D() {
	map = Navigation2DServer::get_singleton()->map_create();
	set_cell_size(10); // Ten pixels
	set_edge_connection_margin(100);
}

Navigation2D::~Navigation2D() {
	Navigation2DServer::get_singleton()->free(map);
}
