/**************************************************************************/
/*  shape_3d.cpp                                                          */
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

#include "shape_3d.h"

#include "core/os/os.h"
#include "scene/main/scene_tree.h"
#include "scene/resources/mesh.h"
#include "servers/physics_server_3d.h"

void Shape3D::add_vertices_to_array(Vector<Vector3> &array, const Transform3D &p_xform) {
	Vector<Vector3> toadd = get_debug_mesh_lines();

	if (toadd.size()) {
		int base = array.size();
		array.resize(base + toadd.size());
		Vector3 *w = array.ptrw();
		for (int i = 0; i < toadd.size(); i++) {
			w[i + base] = p_xform.xform(toadd[i]);
		}
	}
}

void Shape3D::set_custom_solver_bias(real_t p_bias) {
	custom_bias = p_bias;
	PhysicsServer3D::get_singleton()->shape_set_custom_solver_bias(shape, custom_bias);
}

real_t Shape3D::get_custom_solver_bias() const {
	return custom_bias;
}

real_t Shape3D::get_margin() const {
	return margin;
}

void Shape3D::set_margin(real_t p_margin) {
	margin = p_margin;
	PhysicsServer3D::get_singleton()->shape_set_margin(shape, margin);
}

Ref<ArrayMesh> Shape3D::get_debug_mesh(const Color &p_debug_color, bool p_with_shape_faces) {
#ifdef DEBUG_ENABLED
	if (p_with_shape_faces) {
		if (debug_mesh_face_cache.has(p_debug_color)) {
			return debug_mesh_face_cache.get(p_debug_color);
		}
	} else {
		if (debug_mesh_lines_cache.has(p_debug_color)) {
			return debug_mesh_lines_cache.get(p_debug_color);
		}
	}

	Vector<Vector3> lines = get_debug_mesh_lines();

	Ref<ArrayMesh> mesh = Ref<ArrayMesh>(memnew(ArrayMesh));

	if (!lines.is_empty()) {
		//make mesh
		Vector<Vector3> array;
		Vector<Color> colors;
		colors.resize(lines.size());
		array.resize(lines.size());
		{
			Vector3 *w = array.ptrw();
			Color *c = colors.ptrw();
			for (int i = 0; i < lines.size(); i++) {
				w[i] = lines[i];
				c[i] = p_debug_color;
			}
		}

		Array arr;
		arr.resize(Mesh::ARRAY_MAX);
		arr[Mesh::ARRAY_VERTEX] = array;
		arr[Mesh::ARRAY_COLOR] = colors;

		SceneTree *st = cast_to<SceneTree>(OS::get_singleton()->get_main_loop());

		mesh->add_surface_from_arrays(Mesh::PRIMITIVE_LINES, arr);
		if (p_with_shape_faces) {
			const Ref<ArrayMesh> face_mesh = get_debug_face_mesh(p_debug_color);
			if (face_mesh.is_valid()) {
				mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES,
						face_mesh->surface_get_arrays(0));
			}
		}

		if (st) {
			mesh->surface_set_material(0, st->get_debug_collision_material());
			if (p_with_shape_faces && mesh->get_surface_count() > 1) {
				mesh->surface_set_material(1, st->get_debug_collision_face_material());
			}
		}
	}

	if (p_with_shape_faces) {
		debug_mesh_face_cache.insert(p_debug_color, mesh);
	} else {
		debug_mesh_lines_cache.insert(p_debug_color, mesh);
	}

	return mesh;
#else
	return nullptr;
#endif // DEBUG_ENABLED
}

Ref<ArrayMesh> Shape3D::get_debug_face_mesh(const Color &p_vertex_color) const {
	Array arr;
	arr.resize(Mesh::ARRAY_MAX);

	get_debug_face_mesh_arrays(arr);

	const Vector<Vector3> &vertices = arr[Mesh::ARRAY_VERTEX];
	if (vertices.is_empty()) {
		return nullptr;
	}

	Vector<Color> colors;
	colors.resize(vertices.size());

	Color *w = colors.ptrw();
	for (int i = 0; i < colors.size(); i++) {
		w[i] = p_vertex_color;
	}
	arr[Mesh::ARRAY_COLOR] = colors;

	Ref<ArrayMesh> array_mesh = memnew(ArrayMesh);
	array_mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, arr);
	return array_mesh;
}

void Shape3D::_update_shape() {
	emit_changed();
#ifdef DEBUG_ENABLED
	debug_mesh_lines_cache.clear();
	debug_mesh_face_cache.clear();
#endif // DEBUG_ENABLED
}

void Shape3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_custom_solver_bias", "bias"), &Shape3D::set_custom_solver_bias);
	ClassDB::bind_method(D_METHOD("get_custom_solver_bias"), &Shape3D::get_custom_solver_bias);

	ClassDB::bind_method(D_METHOD("set_margin", "margin"), &Shape3D::set_margin);
	ClassDB::bind_method(D_METHOD("get_margin"), &Shape3D::get_margin);

	ClassDB::bind_method(D_METHOD("get_debug_mesh", "vertex_color", "with_shape_faces"), &Shape3D::get_debug_mesh);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "custom_solver_bias", PROPERTY_HINT_RANGE, "0,1,0.001"), "set_custom_solver_bias", "get_custom_solver_bias");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "margin", PROPERTY_HINT_RANGE, "0,10,0.001,or_greater,suffix:m"), "set_margin", "get_margin");
}

Shape3D::Shape3D() {
	ERR_PRINT("Default constructor must not be called!");
}

Shape3D::Shape3D(RID p_shape) :
		shape(p_shape) {}

Shape3D::~Shape3D() {
	ERR_FAIL_NULL(PhysicsServer3D::get_singleton());
	PhysicsServer3D::get_singleton()->free(shape);
}
