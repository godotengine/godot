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

#include "core/object/class_db.h"

#include <thirdparty/misc/polypartition.h>

void TaperedCapsuleMesh::_update_lightmap_size() {
	if (get_add_uv2()) {
		// size must have changed, update lightmap size hint
		Size2i _lightmap_size_hint;
		real_t padding = get_uv2_padding();

		real_t max_radius = MAX(top_radius, bottom_radius);

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

	create_mesh_array(p_arr, top_radius, bottom_radius, mid_height, radial_segments, rings, _add_uv2, _uv2_padding);
}

bool TaperedCapsuleMesh::is_sphere(real_t p_top_radius, real_t p_bottom_radius, real_t p_mid_height) {
	return Math::abs(p_top_radius - p_bottom_radius) > p_mid_height;
}

real_t TaperedCapsuleMesh::get_tangent_angle(real_t p_top_radius, real_t p_bottom_radius, real_t p_mid_height) {
	return Math::asin((p_bottom_radius - p_top_radius) / p_mid_height);
}

void TaperedCapsuleMesh::create_mesh_array(Array &p_arr, real_t p_top_radius, real_t p_bottom_radius, real_t p_mid_height, int p_radial_segments, int p_rings, bool p_add_uv2, const real_t p_uv2_padding) {
#define ADD_TANGENT(m_x, m_y, m_z, m_d) \
	tangents.push_back(m_x); \
	tangents.push_back(m_y); \
	tangents.push_back(m_z); \
	tangents.push_back(m_d);

	const bool has_tangent = !is_sphere(p_top_radius, p_bottom_radius, p_mid_height);

	// angle of the tangent of the two circles
	const real_t angle = get_tangent_angle(p_top_radius, p_bottom_radius, p_mid_height);
	const real_t angle_per_ring = Math::PI / p_rings;
	const real_t top_arc_length = Math::PI / 2.0 - angle; //arc length of the top hemisphere, for UV computation
	const real_t bottom_arc_length = Math::PI / 2.0 + angle; // arc length of the bottom hemisphere, for UV computation

	int top_rings = Math::ceil(top_arc_length / angle_per_ring); // Number of rings to be allocated to the top end of the capsule
	int bottom_rings = Math::ceil(bottom_arc_length / angle_per_ring); //Number of ring to be allocated to the bottom end of the capsule
	if (p_top_radius <= 0) { //If the radius is null, show no rings
		top_rings = 0;
	}
	if (p_bottom_radius <= 0) {
		bottom_rings = 0;
	}
	// Make it so that both ends don't have more than the requested amount of rings by removing rings from the bigger end first
	// Only one or two iterations should be needed
	while (top_rings + bottom_rings > p_rings) {
		if (top_rings > bottom_rings) {
			top_rings--;
		} else {
			bottom_rings--;
		}
	}
	//Number of points left to ring
	const int num_columns = p_radial_segments + 1;
	//Number of points up to down, including the hemispheres and the tangent
	const int num_rows = top_rings + bottom_rings + 2;
	const int num_vertices = (num_columns) * (num_rows);

	LocalVector<Vector3> positions, normals;
	positions.reserve(num_vertices);
	normals.reserve(num_vertices);
	LocalVector<float> tangents;
	tangents.reserve(num_vertices * 4);
	LocalVector<Vector2> uvs, uv2s;
	uvs.reserve(num_vertices);
	if (p_add_uv2) {
		uv2s.reserve(num_vertices);
	}
	const real_t top_perimeter = p_top_radius * (Math::PI / 2 - angle);
	const real_t bottom_perimeter = p_bottom_radius * (angle + Math::PI / 2);
	// Use pythagoras to compute the length of the tangent when it exists, for the UV
	const real_t tangent_perimeter = has_tangent ? Math::sqrt(Math::pow(p_mid_height, 2) - Math::pow(p_bottom_radius - p_top_radius, 2)) : 0;
	const real_t total_perimeter_vertical_slice = top_perimeter + bottom_perimeter + tangent_perimeter;

	for (int r = 0; r < num_columns; r++) {
		const real_t u = (real_t)r / p_radial_segments;
		const real_t phi = u * Math::TAU;
		const real_t xsin = -Math::sin(phi);
		const real_t zcos = -Math::cos(phi);
		for (int t = 0; t < num_rows; t++) {
			real_t theta;
			real_t radius;
			bool istop = t <= top_rings;
			if (istop) {
				if (top_rings == 0) {
					//Case where the top isn't shown
					if (has_tangent) {
						theta = angle; // 0 radius, this points to the side with the angle of the tangent
					} else {
						theta = Math::PI / 2; // when it's a sphere, this points up
					}
				} else {
					theta = Math::lerp((real_t)Math::PI / 2, angle, (real_t)t / top_rings);
				}
				radius = p_top_radius;
			} else {
				if (bottom_rings == 0) {
					// Case where the bottom isn't shown
					if (has_tangent) {
						theta = angle; // 0 radius, this points to the side with the angle of the tangent
					} else {
						theta = -Math::PI / 2; // when it's a sphere, this end points down
					}
				} else {
					theta = Math::lerp(angle, (real_t)-Math::PI / 2, (real_t)(t - top_rings - 1) / bottom_rings);
				}
				radius = p_bottom_radius;
			}

			const real_t perimeter_from_top = istop ? (Math::PI / 2 - theta) * radius : total_perimeter_vertical_slice - (Math::PI / 2 + theta) * radius;
			const real_t v = perimeter_from_top / total_perimeter_vertical_slice;

			const real_t y = Math::sin(theta);
			const real_t rxz = Math::cos(theta);
			const real_t x = xsin * rxz;
			const real_t z = zcos * rxz;

			const Vector3 p = Vector3(x, y, z);
			positions.push_back(p * radius + Vector3(0, istop ? p_mid_height / 2 : -p_mid_height / 2, 0));
			normals.push_back(p);
			ADD_TANGENT(zcos, 0.0, -xsin, 1.0)
			uvs.push_back(Vector2(u, v));
			if (p_add_uv2) {
				uv2s.push_back(Vector2(u * (1 - p_uv2_padding * 2) + p_uv2_padding, v * (1 - p_uv2_padding * 2) + p_uv2_padding));
			}
		}
	}
#undef ADD_TANGENT

	LocalVector<int> indices;
	indices.reserve(p_radial_segments * (num_rows - 1) * 6);
	// Vertices go from top to bottom, then towards the right
	// There are num_rows vertices vertically, and num_columns vertices horizontally
	// clockwise winding order
	for (int i = 0; i < p_radial_segments; i++) // left to right
	{
		for (int j = 0; j < num_rows - 1; j++) // up to down
		{
			int tlvidx = i * num_rows + j; // top left vertex
			int blvidx = tlvidx + 1; // bottom left vertex
			int trvidx = tlvidx + num_rows; // top right vertex
			int brvidx = trvidx + 1; // bottom left vertex
			// top and left tris
			indices.push_back(blvidx);
			indices.push_back(tlvidx);
			indices.push_back(trvidx);
			// bottom and right tris
			indices.push_back(blvidx);
			indices.push_back(trvidx);
			indices.push_back(brvidx);
		}
	}

	p_arr[RSE::ARRAY_VERTEX] = Vector<Vector3>(positions);
	p_arr[RSE::ARRAY_NORMAL] = Vector<Vector3>(normals);
	p_arr[RSE::ARRAY_TANGENT] = Vector<float>(tangents);
	p_arr[RSE::ARRAY_TEX_UV] = Vector<Vector2>(uvs);
	if (p_add_uv2) {
		p_arr[RSE::ARRAY_TEX_UV2] = Vector<Vector2>(uv2s);
	}
	p_arr[RSE::ARRAY_INDEX] = Vector<int>(indices);
}

void TaperedCapsuleMesh::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_top_radius", "top_radius"), &TaperedCapsuleMesh::set_top_radius);
	ClassDB::bind_method(D_METHOD("get_top_radius"), &TaperedCapsuleMesh::get_top_radius);
	ClassDB::bind_method(D_METHOD("set_bottom_radius", "bottom_radius"), &TaperedCapsuleMesh::set_bottom_radius);
	ClassDB::bind_method(D_METHOD("get_bottom_radius"), &TaperedCapsuleMesh::get_bottom_radius);
	ClassDB::bind_method(D_METHOD("set_radius", "radius"), &TaperedCapsuleMesh::set_radius);
	ClassDB::bind_method(D_METHOD("get_radius"), &TaperedCapsuleMesh::get_radius);
	ClassDB::bind_method(D_METHOD("set_mid_height", "mid_height"), &TaperedCapsuleMesh::set_mid_height);
	ClassDB::bind_method(D_METHOD("get_mid_height"), &TaperedCapsuleMesh::get_mid_height);
	ClassDB::bind_method(D_METHOD("set_height", "height"), &TaperedCapsuleMesh::set_height);
	ClassDB::bind_method(D_METHOD("get_height"), &TaperedCapsuleMesh::get_height);

	ClassDB::bind_method(D_METHOD("set_radial_segments", "segments"), &TaperedCapsuleMesh::set_radial_segments);
	ClassDB::bind_method(D_METHOD("get_radial_segments"), &TaperedCapsuleMesh::get_radial_segments);
	ClassDB::bind_method(D_METHOD("set_rings", "rings"), &TaperedCapsuleMesh::set_rings);
	ClassDB::bind_method(D_METHOD("get_rings"), &TaperedCapsuleMesh::get_rings);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "top_radius", PROPERTY_HINT_RANGE, "0,100.0,0.001,or_greater,suffix:m"), "set_top_radius", "get_top_radius");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "bottom_radius", PROPERTY_HINT_RANGE, "0,100.0,0.001,or_greater,suffix:m"), "set_bottom_radius", "get_bottom_radius");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "radius", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE), "set_radius", "get_radius");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "mid_height", PROPERTY_HINT_RANGE, "0,100.0,0.001,or_greater,suffix:m"), "set_mid_height", "get_mid_height");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "height", PROPERTY_HINT_RANGE, "0,100.0,0.001,or_greater,suffix:m", PROPERTY_USAGE_EDITOR), "set_height", "get_height");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "radial_segments", PROPERTY_HINT_RANGE, "4,128,1,or_greater"), "set_radial_segments", "get_radial_segments");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "rings", PROPERTY_HINT_RANGE, "4,64,1,or_greater"), "set_rings", "get_rings");

	ADD_LINKED_PROPERTY("radius", "top_radius");
	ADD_LINKED_PROPERTY("radius", "bottom_radius");

	ADD_LINKED_PROPERTY("top_radius", "height");
	ADD_LINKED_PROPERTY("bottom_radius", "height");
	ADD_LINKED_PROPERTY("mid_height", "height");

	ADD_LINKED_PROPERTY("height", "top_radius");
	ADD_LINKED_PROPERTY("height", "bottom_radius");
	ADD_LINKED_PROPERTY("height", "mid_height");
}

void TaperedCapsuleMesh::set_top_radius(const real_t p_top_radius) {
	if (Math::is_equal_approx(top_radius, p_top_radius)) {
		return;
	}

	top_radius = p_top_radius < 0 ? 0 : p_top_radius;
	_update_lightmap_size();
	request_update();
}

real_t TaperedCapsuleMesh::get_top_radius() const {
	return top_radius;
}

void TaperedCapsuleMesh::set_bottom_radius(const real_t p_bottom_radius) {
	if (Math::is_equal_approx(bottom_radius, p_bottom_radius)) {
		return;
	}

	bottom_radius = p_bottom_radius < 0 ? 0 : p_bottom_radius;
	_update_lightmap_size();
	request_update();
}

real_t TaperedCapsuleMesh::get_bottom_radius() const {
	return bottom_radius;
}

void TaperedCapsuleMesh::set_radius(const real_t p_radius) {
	set_top_radius(p_radius);
	set_bottom_radius(p_radius);
}

real_t TaperedCapsuleMesh::get_radius() const {
	return (get_top_radius() + get_bottom_radius()) / 2;
}

void TaperedCapsuleMesh::set_mid_height(const real_t p_mid_height) {
	if (Math::is_equal_approx(mid_height, p_mid_height)) {
		return;
	}

	mid_height = p_mid_height < 0 ? 0 : p_mid_height;
	_update_lightmap_size();
	request_update();
}

real_t TaperedCapsuleMesh::get_mid_height() const {
	return mid_height;
}

void TaperedCapsuleMesh::set_height(const real_t p_height) {
	real_t new_mid_height = p_height - top_radius - bottom_radius;
	if (new_mid_height <= 0) {
		new_mid_height = 0.f; // Minimum to avoid invalid mesh
	}
	set_mid_height(new_mid_height);
}

real_t TaperedCapsuleMesh::get_height() const {
	return mid_height + top_radius + bottom_radius;
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

	rings = p_rings > 4 ? p_rings : 4;
	request_update();
}

int TaperedCapsuleMesh::get_rings() const {
	return rings;
}
