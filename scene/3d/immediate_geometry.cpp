/**************************************************************************/
/*  immediate_geometry.cpp                                                */
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

#include "immediate_geometry.h"

void ImmediateGeometry::begin(Mesh::PrimitiveType p_primitive, const Ref<Texture> &p_texture) {
	VS::get_singleton()->immediate_begin(im, (VS::PrimitiveType)p_primitive, p_texture.is_valid() ? p_texture->get_rid() : RID());
	if (p_texture.is_valid()) {
		cached_textures.push_back(p_texture);
	}
}

void ImmediateGeometry::set_normal(const Vector3 &p_normal) {
	VS::get_singleton()->immediate_normal(im, p_normal);
}

void ImmediateGeometry::set_tangent(const Plane &p_tangent) {
	VS::get_singleton()->immediate_tangent(im, p_tangent);
}

void ImmediateGeometry::set_color(const Color &p_color) {
	VS::get_singleton()->immediate_color(im, p_color);
}

void ImmediateGeometry::set_uv(const Vector2 &p_uv) {
	VS::get_singleton()->immediate_uv(im, p_uv);
}

void ImmediateGeometry::set_uv2(const Vector2 &p_uv2) {
	VS::get_singleton()->immediate_uv2(im, p_uv2);
}

void ImmediateGeometry::add_vertex(const Vector3 &p_vertex) {
	VS::get_singleton()->immediate_vertex(im, p_vertex);
	if (empty) {
		aabb.position = p_vertex;
		aabb.size = Vector3();
		empty = false;
	} else {
		aabb.expand_to(p_vertex);
	}
}

void ImmediateGeometry::end() {
	VS::get_singleton()->immediate_end(im);
}

void ImmediateGeometry::clear() {
	VS::get_singleton()->immediate_clear(im);
	empty = true;
	cached_textures.clear();
}

AABB ImmediateGeometry::get_aabb() const {
	return aabb;
}
PoolVector<Face3> ImmediateGeometry::get_faces(uint32_t p_usage_flags) const {
	return PoolVector<Face3>();
}

void ImmediateGeometry::add_sphere(int p_lats, int p_lons, float p_radius, bool p_add_uv) {
	for (int i = 1; i <= p_lats; i++) {
		double lat0 = Math_PI * (-0.5 + (double)(i - 1) / p_lats);
		double z0 = Math::sin(lat0);
		double zr0 = Math::cos(lat0);

		double lat1 = Math_PI * (-0.5 + (double)i / p_lats);
		double z1 = Math::sin(lat1);
		double zr1 = Math::cos(lat1);

		for (int j = p_lons; j >= 1; j--) {
			double lng0 = 2 * Math_PI * (double)(j - 1) / p_lons;
			double x0 = Math::cos(lng0);
			double y0 = Math::sin(lng0);

			double lng1 = 2 * Math_PI * (double)(j) / p_lons;
			double x1 = Math::cos(lng1);
			double y1 = Math::sin(lng1);

			Vector3 v[4] = {
				Vector3(x1 * zr0, z0, y1 * zr0),
				Vector3(x1 * zr1, z1, y1 * zr1),
				Vector3(x0 * zr1, z1, y0 * zr1),
				Vector3(x0 * zr0, z0, y0 * zr0)
			};

#define ADD_POINT(m_idx)                                                                                    \
	if (p_add_uv) {                                                                                         \
		set_uv(Vector2(Math::atan2(v[m_idx].x, v[m_idx].z) / Math_PI * 0.5 + 0.5, v[m_idx].y * 0.5 + 0.5)); \
		set_tangent(Plane(Vector3(-v[m_idx].z, v[m_idx].y, v[m_idx].x), 1));                                \
	}                                                                                                       \
	set_normal(v[m_idx]);                                                                                   \
	add_vertex(v[m_idx] * p_radius);

			ADD_POINT(0);
			ADD_POINT(1);
			ADD_POINT(2);

			ADD_POINT(2);
			ADD_POINT(3);
			ADD_POINT(0);
		}
	}
}

void ImmediateGeometry::_bind_methods() {
	ClassDB::bind_method(D_METHOD("begin", "primitive", "texture"), &ImmediateGeometry::begin, DEFVAL(Ref<Texture>()));
	ClassDB::bind_method(D_METHOD("set_normal", "normal"), &ImmediateGeometry::set_normal);
	ClassDB::bind_method(D_METHOD("set_tangent", "tangent"), &ImmediateGeometry::set_tangent);
	ClassDB::bind_method(D_METHOD("set_color", "color"), &ImmediateGeometry::set_color);
	ClassDB::bind_method(D_METHOD("set_uv", "uv"), &ImmediateGeometry::set_uv);
	ClassDB::bind_method(D_METHOD("set_uv2", "uv"), &ImmediateGeometry::set_uv2);
	ClassDB::bind_method(D_METHOD("add_vertex", "position"), &ImmediateGeometry::add_vertex);
	ClassDB::bind_method(D_METHOD("add_sphere", "lats", "lons", "radius", "add_uv"), &ImmediateGeometry::add_sphere, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("end"), &ImmediateGeometry::end);
	ClassDB::bind_method(D_METHOD("clear"), &ImmediateGeometry::clear);
}

ImmediateGeometry::ImmediateGeometry() {
	im = RID_PRIME(VisualServer::get_singleton()->immediate_create());
	set_base(im);
	empty = true;
}

ImmediateGeometry::~ImmediateGeometry() {
	VisualServer::get_singleton()->free(im);
}
