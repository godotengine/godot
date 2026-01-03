/**************************************************************************/
/*  tapered_capsule_mesh.cpp                                              */
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

#include "tapered_capsule_mesh.h"

#include "core/config/project_settings.h"
#include "core/math/math_funcs.h"
#include "scene/resources/theme.h"
#include "scene/theme/theme_db.h"
#include "servers/rendering/rendering_server.h"
#include "thirdparty/misc/polypartition.h"

void TaperedCapsuleMesh::_update_lightmap_size() {
	if (get_add_uv2()) {
		// size must have changed, update lightmap size hint
		Size2i _lightmap_size_hint;
		real_t padding = get_uv2_padding();

		real_t max_radius = MAX(radius_top, radius_bottom);

		real_t radial_length = max_radius * Math::PI * 0.5; // circumference of 90 degree bend
		real_t vertical_length = radial_length * 2 + mid_height + padding; // total vertical length

		_lightmap_size_hint.x = MAX(1.0, 4.0 * radial_length / texel_size) + padding;
		_lightmap_size_hint.y = MAX(1.0, vertical_length / texel_size) + padding;

		set_lightmap_size_hint(_lightmap_size_hint);
	}
}

void TaperedCapsuleMesh::_create_mesh_array(Array &p_arr) const {
	bool _add_uv2 = get_add_uv2();
	real_t _uv2_padding = get_uv2_padding() * texel_size;

	create_mesh_array(p_arr, radius_top, radius_bottom, mid_height, radial_segments, rings, _add_uv2, _uv2_padding);
}

void TaperedCapsuleMesh::create_mesh_array(Array &p_arr, real_t p_radius_top, real_t p_radius_bottom, real_t p_mid_height, int p_radial_segments, int p_rings, bool p_add_uv2, const real_t p_uv2_padding) {
	int i, j, prevrow, thisrow, point;
	real_t x, y, z, u, v, w;

	// Use LocalVector for operations and copy to Vector at the end to save the cost of CoW semantics which aren't
	// needed here and are very expensive in such a hot loop. Use reserve to avoid repeated memory allocations.
	int num_points = (p_rings + 2) * (p_radial_segments + 1) * 2 + (p_rings + 2) * (p_radial_segments + 1);
	LocalVector<Vector3> points;
	points.reserve(num_points);
	LocalVector<Vector3> normals;
	normals.reserve(num_points);
	LocalVector<float> tangents;
	tangents.reserve(num_points * 4);
	LocalVector<Vector2> uvs;
	uvs.reserve(num_points);
	LocalVector<Vector2> uv2s;
	if (p_add_uv2) {
		uv2s.reserve(num_points);
	}
	LocalVector<int> indices;
	indices.reserve((p_rings + 1) * (p_radial_segments) * 6 * 2 + (p_rings + 1) * (p_radial_segments) * 6);
	point = 0;

#define ADD_TANGENT(m_x, m_y, m_z, m_d) \
	tangents.push_back(m_x);            \
	tangents.push_back(m_y);            \
	tangents.push_back(m_z);            \
	tangents.push_back(m_d);

	// Calculate UV2 parameters
	real_t total_height = p_mid_height + p_radius_top + p_radius_bottom;
	real_t total_vertical_length = total_height + p_uv2_padding;

	real_t uv2_v_top = p_radius_top / total_vertical_length;
	real_t uv2_v_cylinder = p_mid_height / total_vertical_length;
	real_t uv2_v_bottom = p_radius_bottom / total_vertical_length;

	real_t max_circumference = MAX(p_radius_top, p_radius_bottom) * Math::TAU;
	real_t uv2_h_scale = max_circumference / (max_circumference + p_uv2_padding);

	/* top hemisphere */
	thisrow = 0;
	prevrow = 0;
	for (j = 0; j <= (p_rings + 1); j++) {
		v = j;
		v /= (p_rings + 1);
		if (j == (p_rings + 1)) {
			w = 1.0;
			y = 0.0;
		} else {
			w = Math::sin(0.5 * Math::PI * v);
			y = Math::cos(0.5 * Math::PI * v);
		}

		for (i = 0; i <= p_radial_segments; i++) {
			u = i;
			u /= p_radial_segments;

			if (i == p_radial_segments) {
				x = 0.0;
				z = 1.0;
			} else {
				x = -Math::sin(u * Math::TAU);
				z = Math::cos(u * Math::TAU);
			}

			Vector3 p = Vector3(x * w, y, -z * w);
			points.push_back(p * p_radius_top + Vector3(0.0, total_height * 0.5 - p_radius_top, 0.0));
			normals.push_back(p);
			ADD_TANGENT(-z, 0.0, -x, 1.0)
			uvs.push_back(Vector2(u, v * 0.5)); // UV for top hemisphere
			if (p_add_uv2) {
				uv2s.push_back(Vector2(u * uv2_h_scale, v * uv2_v_top));
			}
			point++;

			if (i > 0 && j > 0) {
				indices.push_back(prevrow + i - 1);
				indices.push_back(prevrow + i);
				indices.push_back(thisrow + i - 1);

				indices.push_back(prevrow + i);
				indices.push_back(thisrow + i);
				indices.push_back(thisrow + i - 1);
			}
		}

		prevrow = thisrow;
		thisrow = point;
	}

	/* cylinder */
	thisrow = point;
	prevrow = point - (p_radial_segments + 1); // Start from the last row of the top hemisphere
	for (j = 0; j <= (p_rings + 1); j++) { // Use rings for cylinder segments too for consistency
		v = j;
		v /= (p_rings + 1);

		real_t current_radius = p_radius_top + (p_radius_bottom - p_radius_top) * v;
		y = (total_height * 0.5 - p_radius_top) - (p_mid_height * v);

		for (i = 0; i <= p_radial_segments; i++) {
			u = i;
			u /= p_radial_segments;

			if (i == p_radial_segments) {
				x = 0.0;
				z = 1.0;
			} else {
				x = -Math::sin(u * Math::TAU);
				z = Math::cos(u * Math::TAU);
			}

			Vector3 p = Vector3(x * current_radius, y, -z * current_radius);
			points.push_back(p);
			normals.push_back(Vector3(x, 0.0, -z).normalized()); // Normal points outwards
			ADD_TANGENT(-z, 0.0, -x, 1.0)
			uvs.push_back(Vector2(u, 0.5 + v * 0.5)); // UV for cylinder
			if (p_add_uv2) {
				uv2s.push_back(Vector2(u * uv2_h_scale, uv2_v_top + v * uv2_v_cylinder));
			}
			point++;

			if (i > 0 && j > 0) {
				indices.push_back(prevrow + i - 1);
				indices.push_back(prevrow + i);
				indices.push_back(thisrow + i - 1);

				indices.push_back(prevrow + i);
				indices.push_back(thisrow + i);
				indices.push_back(thisrow + i - 1);
			}
		}

		prevrow = thisrow;
		thisrow = point;
	}

	/* bottom hemisphere */
	thisrow = point;
	prevrow = point - (p_radial_segments + 1); // Start from the last row of the cylinder
	for (j = 0; j <= (p_rings + 1); j++) {
		v = j;
		v /= (p_rings + 1);
		if (j == (p_rings + 1)) {
			w = 0.0;
			y = -1.0;
		} else {
			w = Math::cos(0.5 * Math::PI * v);
			y = -Math::sin(0.5 * Math::PI * v);
		}

		for (i = 0; i <= p_radial_segments; i++) {
			u = i;
			u /= p_radial_segments;

			if (i == p_radial_segments) {
				x = 0.0;
				z = 1.0;
			} else {
				x = -Math::sin(u * Math::TAU);
				z = Math::cos(u * Math::TAU);
			}

			Vector3 p = Vector3(x * w, y, -z * w);
			points.push_back(p * p_radius_bottom + Vector3(0.0, -total_height * 0.5 + p_radius_bottom, 0.0));
			normals.push_back(p);
			ADD_TANGENT(-z, 0.0, -x, 1.0)
			uvs.push_back(Vector2(u, 0.5 + v * 0.5)); // UV for bottom hemisphere (can reuse cylinder UV space)
			if (p_add_uv2) {
				uv2s.push_back(Vector2(u * uv2_h_scale, uv2_v_top + uv2_v_cylinder + v * uv2_v_bottom));
			}
			point++;

			if (i > 0 && j > 0) {
				indices.push_back(prevrow + i - 1);
				indices.push_back(prevrow + i);
				indices.push_back(thisrow + i - 1);

				indices.push_back(prevrow + i);
				indices.push_back(thisrow + i);
				indices.push_back(thisrow + i - 1);
			}
		}

		prevrow = thisrow;
		thisrow = point;
	}

	p_arr[RS::ARRAY_VERTEX] = Vector<Vector3>(points);
	p_arr[RS::ARRAY_NORMAL] = Vector<Vector3>(normals);
	p_arr[RS::ARRAY_TANGENT] = Vector<float>(tangents);
	p_arr[RS::ARRAY_TEX_UV] = Vector<Vector2>(uvs);
	if (p_add_uv2) {
		p_arr[RS::ARRAY_TEX_UV2] = Vector<Vector2>(uv2s);
	}
	p_arr[RS::ARRAY_INDEX] = Vector<int>(indices);
}

void TaperedCapsuleMesh::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_radius_top", "radius_top"), &TaperedCapsuleMesh::set_radius_top);
	ClassDB::bind_method(D_METHOD("get_radius_top"), &TaperedCapsuleMesh::get_radius_top);
	ClassDB::bind_method(D_METHOD("set_radius_bottom", "radius_bottom"), &TaperedCapsuleMesh::set_radius_bottom);
	ClassDB::bind_method(D_METHOD("get_radius_bottom"), &TaperedCapsuleMesh::get_radius_bottom);
	ClassDB::bind_method(D_METHOD("set_mid_height", "mid_height"), &TaperedCapsuleMesh::set_mid_height);
	ClassDB::bind_method(D_METHOD("get_mid_height"), &TaperedCapsuleMesh::get_mid_height);
	ClassDB::bind_method(D_METHOD("set_height", "height"), &TaperedCapsuleMesh::set_height);
	ClassDB::bind_method(D_METHOD("get_height"), &TaperedCapsuleMesh::get_height);

	ClassDB::bind_method(D_METHOD("set_radial_segments", "segments"), &TaperedCapsuleMesh::set_radial_segments);
	ClassDB::bind_method(D_METHOD("get_radial_segments"), &TaperedCapsuleMesh::get_radial_segments);
	ClassDB::bind_method(D_METHOD("set_rings", "rings"), &TaperedCapsuleMesh::set_rings);
	ClassDB::bind_method(D_METHOD("get_rings"), &TaperedCapsuleMesh::get_rings);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "radius_top", PROPERTY_HINT_RANGE, "0.001,100.0,0.001,or_greater,suffix:m"), "set_radius_top", "get_radius_top");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "radius_bottom", PROPERTY_HINT_RANGE, "0.001,100.0,0.001,or_greater,suffix:m"), "set_radius_bottom", "get_radius_bottom");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "mid_height", PROPERTY_HINT_RANGE, "0.001,100.0,0.001,or_greater,suffix:m"), "set_mid_height", "get_mid_height");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "height", PROPERTY_HINT_RANGE, "0.001,100.0,0.001,or_greater,suffix:m"), "set_height", "get_height");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "radial_segments", PROPERTY_HINT_RANGE, "1,100,1,or_greater"), "set_radial_segments", "get_radial_segments");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "rings", PROPERTY_HINT_RANGE, "0,100,1,or_greater"), "set_rings", "get_rings");

	ADD_LINKED_PROPERTY("radius_top", "height");
	ADD_LINKED_PROPERTY("radius_bottom", "height");
	ADD_LINKED_PROPERTY("mid_height", "height");
	ADD_LINKED_PROPERTY("height", "radius_top");
	ADD_LINKED_PROPERTY("height", "radius_bottom");
	ADD_LINKED_PROPERTY("height", "mid_height");
}

void TaperedCapsuleMesh::set_radius_top(const real_t p_radius_top) {
	if (Math::is_equal_approx(radius_top, p_radius_top)) {
		return;
	}

	radius_top = p_radius_top;
	_update_lightmap_size();
	request_update();
}

real_t TaperedCapsuleMesh::get_radius_top() const {
	return radius_top;
}

void TaperedCapsuleMesh::set_radius_bottom(const real_t p_radius_bottom) {
	if (Math::is_equal_approx(radius_bottom, p_radius_bottom)) {
		return;
	}

	radius_bottom = p_radius_bottom;
	_update_lightmap_size();
	request_update();
}

real_t TaperedCapsuleMesh::get_radius_bottom() const {
	return radius_bottom;
}

void TaperedCapsuleMesh::set_mid_height(const real_t p_mid_height) {
	ERR_FAIL_COND(p_mid_height <= 0);
	if (Math::is_equal_approx(mid_height, p_mid_height)) {
		return;
	}

	mid_height = p_mid_height;
	_update_lightmap_size();
	request_update();
}

real_t TaperedCapsuleMesh::get_mid_height() const {
	return mid_height;
}

void TaperedCapsuleMesh::set_height(const real_t p_height) {
	real_t new_mid_height = p_height - radius_top - radius_bottom;
	if (new_mid_height <= 0) {
		new_mid_height = 0.001f; // Minimum to avoid invalid mesh
	}
	set_mid_height(new_mid_height);
}

real_t TaperedCapsuleMesh::get_height() const {
	return mid_height + radius_top + radius_bottom;
}

void TaperedCapsuleMesh::set_radial_segments(const int p_segments) {
	if (radial_segments == p_segments) {
		return;
	}

	radial_segments = p_segments > 4 ? p_segments : 4;
	request_update();
}

int TaperedCapsuleMesh::get_radial_segments() const {
	return radial_segments;
}

void TaperedCapsuleMesh::set_rings(const int p_rings) {
	if (rings == p_rings) {
		return;
	}

	ERR_FAIL_COND(p_rings < 0);
	rings = p_rings;
	request_update();
}

int TaperedCapsuleMesh::get_rings() const {
	return rings;
}
