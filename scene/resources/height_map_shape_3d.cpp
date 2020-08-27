/*************************************************************************/
/*  height_map_shape_3d.cpp                                              */
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

#include "height_map_shape_3d.h"
#include "servers/physics_server_3d.h"

Vector<Vector3> HeightMapShape3D::get_debug_mesh_lines() const {
	Vector<Vector3> points;

	if ((map_width != 0) && (map_depth != 0)) {
		// This will be slow for large maps...
		// also we'll have to figure out how well bullet centers this shape...

		Vector2 size(map_width - 1, map_depth - 1);
		Vector2 start = size * -0.5;

		const real_t *r = map_data.ptr();

		// reserve some memory for our points..
		points.resize(((map_width - 1) * map_depth * 2) + (map_width * (map_depth - 1) * 2));

		// now set our points
		int r_offset = 0;
		int w_offset = 0;
		for (int d = 0; d < map_depth; d++) {
			Vector3 height(start.x, 0.0, start.y);

			for (int w = 0; w < map_width; w++) {
				height.y = r[r_offset++];

				if (w != map_width - 1) {
					points.write[w_offset++] = height;
					points.write[w_offset++] = Vector3(height.x + 1.0, r[r_offset], height.z);
				}

				if (d != map_depth - 1) {
					points.write[w_offset++] = height;
					points.write[w_offset++] = Vector3(height.x, r[r_offset + map_width - 1], height.z + 1.0);
				}

				height.x += 1.0;
			}

			start.y += 1.0;
		}
	}

	return points;
}

real_t HeightMapShape3D::get_enclosing_radius() const {
	return Vector3(real_t(map_width), max_height - min_height, real_t(map_depth)).length();
}

void HeightMapShape3D::_update_shape() {
	Dictionary d;
	d["width"] = map_width;
	d["depth"] = map_depth;
	d["heights"] = map_data;
	d["min_height"] = min_height;
	d["max_height"] = max_height;
	PhysicsServer3D::get_singleton()->shape_set_data(get_shape(), d);
	Shape3D::_update_shape();
}

void HeightMapShape3D::set_map_width(int p_new) {
	if (p_new < 1) {
		// ignore
	} else if (map_width != p_new) {
		int was_size = map_width * map_depth;
		map_width = p_new;

		int new_size = map_width * map_depth;
		map_data.resize(map_width * map_depth);

		real_t *w = map_data.ptrw();
		while (was_size < new_size) {
			w[was_size++] = 0.0;
		}

		_update_shape();
		notify_change_to_owners();
		_change_notify("map_width");
		_change_notify("map_data");
	}
}

int HeightMapShape3D::get_map_width() const {
	return map_width;
}

void HeightMapShape3D::set_map_depth(int p_new) {
	if (p_new < 1) {
		// ignore
	} else if (map_depth != p_new) {
		int was_size = map_width * map_depth;
		map_depth = p_new;

		int new_size = map_width * map_depth;
		map_data.resize(new_size);

		real_t *w = map_data.ptrw();
		while (was_size < new_size) {
			w[was_size++] = 0.0;
		}

		_update_shape();
		notify_change_to_owners();
		_change_notify("map_depth");
		_change_notify("map_data");
	}
}

int HeightMapShape3D::get_map_depth() const {
	return map_depth;
}

void HeightMapShape3D::set_map_data(PackedFloat32Array p_new) {
	int size = (map_width * map_depth);
	if (p_new.size() != size) {
		// fail
		return;
	}

	// copy
	real_t *w = map_data.ptrw();
	const real_t *r = p_new.ptr();
	for (int i = 0; i < size; i++) {
		float val = r[i];
		w[i] = val;
		if (i == 0) {
			min_height = val;
			max_height = val;
		} else {
			if (min_height > val) {
				min_height = val;
			}

			if (max_height < val) {
				max_height = val;
			}
		}
	}

	_update_shape();
	notify_change_to_owners();
	_change_notify("map_data");
}

PackedFloat32Array HeightMapShape3D::get_map_data() const {
	return map_data;
}

void HeightMapShape3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_map_width", "width"), &HeightMapShape3D::set_map_width);
	ClassDB::bind_method(D_METHOD("get_map_width"), &HeightMapShape3D::get_map_width);
	ClassDB::bind_method(D_METHOD("set_map_depth", "height"), &HeightMapShape3D::set_map_depth);
	ClassDB::bind_method(D_METHOD("get_map_depth"), &HeightMapShape3D::get_map_depth);
	ClassDB::bind_method(D_METHOD("set_map_data", "data"), &HeightMapShape3D::set_map_data);
	ClassDB::bind_method(D_METHOD("get_map_data"), &HeightMapShape3D::get_map_data);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "map_width", PROPERTY_HINT_RANGE, "1,4096,1"), "set_map_width", "get_map_width");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "map_depth", PROPERTY_HINT_RANGE, "1,4096,1"), "set_map_depth", "get_map_depth");
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_FLOAT32_ARRAY, "map_data"), "set_map_data", "get_map_data");
}

HeightMapShape3D::HeightMapShape3D() :
		Shape3D(PhysicsServer3D::get_singleton()->shape_create(PhysicsServer3D::SHAPE_HEIGHTMAP)) {
	map_width = 2;
	map_depth = 2;
	map_data.resize(map_width * map_depth);
	real_t *w = map_data.ptrw();
	w[0] = 0.0;
	w[1] = 0.0;
	w[2] = 0.0;
	w[3] = 0.0;
	min_height = 0.0;
	max_height = 0.0;

	_update_shape();
}
