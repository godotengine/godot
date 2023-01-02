/*************************************************************************/
/*  concave_mesh_shape_3d.cpp                                            */
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

#include "concave_mesh_shape_3d.h"

#include "core/core_string_names.h"
#include "core/math/convex_hull.h"
#include "servers/physics_server_3d.h"

Vector<Vector3> ConcaveMeshShape3D::get_debug_mesh_lines() const {
	HashSet<DrawEdge, DrawEdge> edges;

	int index_count = faces.size();
	ERR_FAIL_COND_V((index_count % 3) != 0, Vector<Vector3>());

	const Vector3 *r = faces.ptr();

	for (int i = 0; i < index_count; i += 3) {
		for (int j = 0; j < 3; j++) {
			DrawEdge de(r[i + j], r[i + ((j + 1) % 3)]);
			edges.insert(de);
		}
	}

	Vector<Vector3> points;
	points.resize(edges.size() * 2);
	int idx = 0;
	for (const DrawEdge &E : edges) {
		points.write[idx + 0] = E.a;
		points.write[idx + 1] = E.b;
		idx += 2;
	}

	return points;
}

real_t ConcaveMeshShape3D::get_enclosing_radius() const {
	Vector<Vector3> data = get_faces();
	const Vector3 *read = data.ptr();
	real_t r = 0.0;
	for (int i(0); i < data.size(); i++) {
		r = MAX(read[i].length_squared(), r);
	}
	return Math::sqrt(r);
}

void ConcaveMeshShape3D::_update_shape() {
	if (!mesh.is_null()) {
		set_faces(mesh->create_trimesh_faces());
	} else {
		set_faces(Vector<Vector3>());
	}

	Dictionary d;
	d["faces"] = faces;
	d["backface_collision"] = backface_collision;
	PhysicsServer3D::get_singleton()->shape_set_data(get_shape(), d);
	Shape3D::_update_shape();
}

void ConcaveMeshShape3D::_update_mesh() {
	_update_shape();
	notify_change_to_owners();
}

void ConcaveMeshShape3D::set_mesh(const Ref<Mesh> &p_mesh) {
	if (mesh == p_mesh) {
		return;
	}

	if (mesh.is_valid()) {
		mesh->disconnect(CoreStringNames::get_singleton()->changed, callable_mp(this, &ConcaveMeshShape3D::_update_mesh));
	}

	mesh = p_mesh;

	if (!mesh.is_null() && mesh.is_valid()) {
		mesh->connect(CoreStringNames::get_singleton()->changed, callable_mp(this, &ConcaveMeshShape3D::_update_mesh));
	}

	_update_mesh();
}

Ref<Mesh> ConcaveMeshShape3D::get_mesh() const {
	return mesh;
}

void ConcaveMeshShape3D::set_faces(const Vector<Vector3> &p_faces) {
	faces = p_faces;
}

Vector<Vector3> ConcaveMeshShape3D::get_faces() const {
	return faces;
}

void ConcaveMeshShape3D::set_backface_collision_enabled(bool p_enabled) {
	backface_collision = p_enabled;

	if (!mesh.is_null()) {
		_update_shape();
		notify_change_to_owners();
	}
}

bool ConcaveMeshShape3D::is_backface_collision_enabled() const {
	return backface_collision;
}

void ConcaveMeshShape3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_mesh", "mesh"), &ConcaveMeshShape3D::set_mesh);
	ClassDB::bind_method(D_METHOD("get_mesh"), &ConcaveMeshShape3D::get_mesh);

	ClassDB::bind_method(D_METHOD("set_faces", "faces"), &ConcaveMeshShape3D::set_faces);
	ClassDB::bind_method(D_METHOD("get_faces"), &ConcaveMeshShape3D::get_faces);

	ClassDB::bind_method(D_METHOD("set_backface_collision_enabled", "enabled"), &ConcaveMeshShape3D::set_backface_collision_enabled);
	ClassDB::bind_method(D_METHOD("is_backface_collision_enabled"), &ConcaveMeshShape3D::is_backface_collision_enabled);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "mesh", PROPERTY_HINT_RESOURCE_TYPE, "Mesh"), "set_mesh", "get_mesh");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "backface_collision"), "set_backface_collision_enabled", "is_backface_collision_enabled");
}

ConcaveMeshShape3D::ConcaveMeshShape3D() :
		Shape3D(PhysicsServer3D::get_singleton()->shape_create(PhysicsServer3D::SHAPE_CONCAVE_POLYGON)) {
}

ConcaveMeshShape3D::ConcaveMeshShape3D(const Ref<Mesh> &p_mesh) :
		ConcaveMeshShape3D() {
	set_mesh(p_mesh);
}
