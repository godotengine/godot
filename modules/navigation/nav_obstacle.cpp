/**************************************************************************/
/*  nav_obstacle.cpp                                                      */
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

#include "nav_obstacle.h"

#include "nav_map.h"

NavObstacle::NavObstacle() {}

NavObstacle::~NavObstacle() {}

void NavObstacle::set_map(NavMap *p_map) {
	if (map != p_map) {
		map = p_map;
		obstacle_dirty = true;
	}
}

void NavObstacle::set_position(const Vector3 p_position) {
	if (position != p_position) {
		position = p_position;
		obstacle_dirty = true;
	}
}

void NavObstacle::set_height(const real_t p_height) {
	if (height != p_height) {
		height = p_height;
		obstacle_dirty = true;
	}
}

void NavObstacle::set_vertices(const Vector<Vector3> &p_vertices) {
	if (vertices != p_vertices) {
		vertices = p_vertices;
		obstacle_dirty = true;
	}
}

bool NavObstacle::is_map_changed() {
	if (map) {
		bool is_changed = map->get_map_update_id() != map_update_id;
		map_update_id = map->get_map_update_id();
		return is_changed;
	} else {
		return false;
	}
}

void NavObstacle::set_avoidance_layers(uint32_t p_layers) {
	avoidance_layers = p_layers;
	obstacle_dirty = true;
}

bool NavObstacle::check_dirty() {
	const bool was_dirty = obstacle_dirty;
	obstacle_dirty = false;
	return was_dirty;
}
