/**************************************************************************/
/*  gltf_collider.cpp                                                     */
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

#include "gltf_collider.h"

#include "../../gltf_state.h"
#include "core/math/convex_hull.h"
#include "scene/3d/area.h"
#include "scene/resources/box_shape.h"
#include "scene/resources/capsule_shape.h"
#include "scene/resources/concave_polygon_shape.h"
#include "scene/resources/convex_polygon_shape.h"
#include "scene/resources/cylinder_shape.h"
#include "scene/resources/sphere_shape.h"

void GLTFCollider::_bind_methods() {
	ClassDB::bind_method(D_METHOD("to_node", "cache_shapes"), &GLTFCollider::to_node, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("to_dictionary"), &GLTFCollider::to_dictionary);

	ClassDB::bind_method(D_METHOD("get_shape_type"), &GLTFCollider::get_shape_type);
	ClassDB::bind_method(D_METHOD("set_shape_type", "shape_type"), &GLTFCollider::set_shape_type);
	ClassDB::bind_method(D_METHOD("get_size"), &GLTFCollider::get_size);
	ClassDB::bind_method(D_METHOD("set_size", "size"), &GLTFCollider::set_size);
	ClassDB::bind_method(D_METHOD("get_radius"), &GLTFCollider::get_radius);
	ClassDB::bind_method(D_METHOD("set_radius", "radius"), &GLTFCollider::set_radius);
	ClassDB::bind_method(D_METHOD("get_height"), &GLTFCollider::get_height);
	ClassDB::bind_method(D_METHOD("set_height", "height"), &GLTFCollider::set_height);
	ClassDB::bind_method(D_METHOD("get_is_trigger"), &GLTFCollider::get_is_trigger);
	ClassDB::bind_method(D_METHOD("set_is_trigger", "is_trigger"), &GLTFCollider::set_is_trigger);
	ClassDB::bind_method(D_METHOD("get_mesh_index"), &GLTFCollider::get_mesh_index);
	ClassDB::bind_method(D_METHOD("set_mesh_index", "mesh_index"), &GLTFCollider::set_mesh_index);
	ClassDB::bind_method(D_METHOD("get_array_mesh"), &GLTFCollider::get_array_mesh);
	ClassDB::bind_method(D_METHOD("set_array_mesh", "array_mesh"), &GLTFCollider::set_array_mesh);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "shape_type"), "set_shape_type", "get_shape_type");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "size"), "set_size", "get_size");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "radius"), "set_radius", "get_radius");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "height"), "set_height", "get_height");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "is_trigger"), "set_is_trigger", "get_is_trigger");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "mesh_index"), "set_mesh_index", "get_mesh_index");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "array_mesh", PROPERTY_HINT_RESOURCE_TYPE, "ArrayMesh"), "set_array_mesh", "get_array_mesh");
}

String GLTFCollider::get_shape_type() const {
	return shape_type;
}

void GLTFCollider::set_shape_type(String p_shape_type) {
	shape_type = p_shape_type;
}

Vector3 GLTFCollider::get_size() const {
	return size;
}

void GLTFCollider::set_size(Vector3 p_size) {
	size = p_size;
}

real_t GLTFCollider::get_radius() const {
	return radius;
}

void GLTFCollider::set_radius(real_t p_radius) {
	radius = p_radius;
}

real_t GLTFCollider::get_height() const {
	return height;
}

void GLTFCollider::set_height(real_t p_height) {
	height = p_height;
}

bool GLTFCollider::get_is_trigger() const {
	return is_trigger;
}

void GLTFCollider::set_is_trigger(bool p_is_trigger) {
	is_trigger = p_is_trigger;
}

GLTFMeshIndex GLTFCollider::get_mesh_index() const {
	return mesh_index;
}

void GLTFCollider::set_mesh_index(GLTFMeshIndex p_mesh_index) {
	mesh_index = p_mesh_index;
}

Ref<ArrayMesh> GLTFCollider::get_array_mesh() const {
	return array_mesh;
}

void GLTFCollider::set_array_mesh(Ref<ArrayMesh> p_array_mesh) {
	array_mesh = p_array_mesh;
}

Ref<GLTFCollider> GLTFCollider::from_node(const CollisionShape *p_collider_node) {
	Ref<GLTFCollider> collider;
	collider.instance();
	ERR_FAIL_NULL_V_MSG(p_collider_node, collider, "Tried to create a GLTFCollider from a CollisionShape node, but the given node was null.");
	Node *parent = p_collider_node->get_parent();
	if (cast_to<const Area>(parent)) {
		collider->set_is_trigger(true);
	}
	// All the code for working with the shape is below this comment.
	Ref<Shape> shape = p_collider_node->get_shape();
	ERR_FAIL_COND_V_MSG(shape.is_null(), collider, "Tried to create a GLTFCollider from a CollisionShape node, but the given node had a null shape.");
	collider->_shape_cache = shape;
	if (cast_to<BoxShape>(shape.ptr())) {
		collider->shape_type = "box";
		Ref<BoxShape> box = shape;
		collider->set_size(box->get_extents() * 2.0f);
	} else if (cast_to<const CapsuleShape>(shape.ptr())) {
		collider->shape_type = "capsule";
		Ref<CapsuleShape> capsule = shape;
		collider->set_radius(capsule->get_radius());
		collider->set_height(capsule->get_height());
	} else if (cast_to<const CylinderShape>(shape.ptr())) {
		collider->shape_type = "cylinder";
		Ref<CylinderShape> cylinder = shape;
		collider->set_radius(cylinder->get_radius());
		collider->set_height(cylinder->get_height());
	} else if (cast_to<const SphereShape>(shape.ptr())) {
		collider->shape_type = "sphere";
		Ref<SphereShape> sphere = shape;
		collider->set_radius(sphere->get_radius());
	} else if (cast_to<const ConvexPolygonShape>(shape.ptr())) {
		collider->shape_type = "hull";
		Ref<ConvexPolygonShape> convex = shape;
		PoolVector<Vector3> hull_points = convex->get_points();
		ERR_FAIL_COND_V_MSG(hull_points.size() < 3, collider, "GLTFCollider: Convex hull has fewer points (" + itos(hull_points.size()) + ") than the minimum of 3. At least 3 points are required in order to save to GLTF, since it uses a mesh to represent convex hulls.");
		if (hull_points.size() > 255) {
			WARN_PRINT("GLTFCollider: Convex hull has more points (" + itos(hull_points.size()) + ") than the recommended maximum of 255. This may not load correctly in other engines.");
		}
		// Convert the convex hull points into an array of faces.
		Geometry::MeshData md;
		Error err = ConvexHullComputer::convex_hull(hull_points, md);
		ERR_FAIL_COND_V_MSG(err != OK, collider, "GLTFCollider: Failed to compute convex hull.");
		Vector<Vector3> face_vertices;
		for (uint32_t i = 0; i < (uint32_t)md.faces.size(); i++) {
			uint32_t index_count = md.faces[i].indices.size();
			for (uint32_t j = 1; j < index_count - 1; j++) {
				face_vertices.push_back(hull_points[md.faces[i].indices[0]]);
				face_vertices.push_back(hull_points[md.faces[i].indices[j]]);
				face_vertices.push_back(hull_points[md.faces[i].indices[j + 1]]);
			}
		}
		// Create an ArrayMesh from the faces.
		Ref<ArrayMesh> array_mesh;
		array_mesh.instance();
		Array surface_array;
		surface_array.resize(Mesh::ArrayType::ARRAY_MAX);
		surface_array[Mesh::ArrayType::ARRAY_VERTEX] = face_vertices;
		array_mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, surface_array);
		collider->set_array_mesh(array_mesh);
	} else if (cast_to<const ConcavePolygonShape>(shape.ptr())) {
		collider->shape_type = "trimesh";
		Ref<ConcavePolygonShape> concave = shape;
		Ref<ArrayMesh> array_mesh;
		array_mesh.instance();
		Array surface_array;
		surface_array.resize(Mesh::ArrayType::ARRAY_MAX);
		surface_array[Mesh::ArrayType::ARRAY_VERTEX] = concave->get_faces();
		array_mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, surface_array);
		collider->set_array_mesh(array_mesh);
	} else {
		ERR_PRINT("Tried to create a GLTFCollider from a CollisionShape node, but the given node's shape '" + String(Variant(shape)) +
				"' had an unsupported shape type. Only BoxShape, CapsuleShape, CylinderShape, SphereShape, ConcavePolygonShape, and ConvexPolygonShape are supported.");
	}
	return collider;
}

CollisionShape *GLTFCollider::to_node(bool p_cache_shapes) {
	CollisionShape *collider = memnew(CollisionShape);
	if (!p_cache_shapes || _shape_cache == nullptr) {
		if (shape_type == "box") {
			Ref<BoxShape> box;
			box.instance();
			box->set_extents(size * 0.5f);
			_shape_cache = box;
		} else if (shape_type == "capsule") {
			Ref<CapsuleShape> capsule;
			capsule.instance();
			capsule->set_radius(radius);
			capsule->set_height(height);
			_shape_cache = capsule;
		} else if (shape_type == "cylinder") {
			Ref<CylinderShape> cylinder;
			cylinder.instance();
			cylinder->set_radius(radius);
			cylinder->set_height(height);
			_shape_cache = cylinder;
		} else if (shape_type == "sphere") {
			Ref<SphereShape> sphere;
			sphere.instance();
			sphere->set_radius(radius);
			_shape_cache = sphere;
		} else if (shape_type == "hull") {
			ERR_FAIL_COND_V_MSG(array_mesh.is_null(), collider, "GLTFCollider: Error converting convex hull collider to a node: The mesh resource is null.");
			Ref<ConvexPolygonShape> convex = array_mesh->create_convex_shape();
			_shape_cache = convex;
		} else if (shape_type == "trimesh") {
			ERR_FAIL_COND_V_MSG(array_mesh.is_null(), collider, "GLTFCollider: Error converting concave mesh collider to a node: The mesh resource is null.");
			Ref<ConcavePolygonShape> concave = array_mesh->create_trimesh_shape();
			_shape_cache = concave;
		} else {
			ERR_PRINT("GLTFCollider: Error converting to a node: Shape type '" + shape_type + "' is unknown.");
		}
	}
	collider->set_shape(_shape_cache);
	return collider;
}

Ref<GLTFCollider> GLTFCollider::from_dictionary(const Dictionary p_dictionary) {
	ERR_FAIL_COND_V_MSG(!p_dictionary.has("type"), Ref<GLTFCollider>(), "Failed to parse GLTF collider, missing required field 'type'.");
	Ref<GLTFCollider> collider;
	collider.instance();
	const String &shape_type = p_dictionary["type"];
	collider->shape_type = shape_type;
	if (shape_type != "box" && shape_type != "capsule" && shape_type != "cylinder" && shape_type != "sphere" && shape_type != "hull" && shape_type != "trimesh") {
		ERR_PRINT("Error parsing GLTF collider: Shape type '" + shape_type + "' is unknown. Only box, capsule, cylinder, sphere, hull, and trimesh are supported.");
	}
	if (p_dictionary.has("radius")) {
		collider->set_radius(p_dictionary["radius"]);
	}
	if (p_dictionary.has("height")) {
		collider->set_height(p_dictionary["height"]);
	}
	if (p_dictionary.has("size")) {
		const Array &arr = p_dictionary["size"];
		if (arr.size() == 3) {
			collider->set_size(Vector3(arr[0], arr[1], arr[2]));
		} else {
			ERR_PRINT("Error parsing GLTF collider: The size must have exactly 3 numbers.");
		}
	}
	if (p_dictionary.has("isTrigger")) {
		collider->set_is_trigger(p_dictionary["isTrigger"]);
	}
	if (p_dictionary.has("mesh")) {
		collider->set_mesh_index(p_dictionary["mesh"]);
	}
	if (unlikely(collider->get_mesh_index() < 0 && (shape_type == "hull" || shape_type == "trimesh"))) {
		ERR_PRINT("Error parsing GLTF collider: The mesh-based shape type '" + shape_type + "' does not have a valid mesh index.");
	}
	return collider;
}

Dictionary GLTFCollider::to_dictionary() const {
	Dictionary d;
	d["type"] = shape_type;
	if (shape_type == "box") {
		Array size_array;
		size_array.resize(3);
		size_array[0] = size.x;
		size_array[1] = size.y;
		size_array[2] = size.z;
		d["size"] = size_array;
	} else if (shape_type == "capsule") {
		d["radius"] = get_radius();
		d["height"] = get_height();
	} else if (shape_type == "cylinder") {
		d["radius"] = get_radius();
		d["height"] = get_height();
	} else if (shape_type == "sphere") {
		d["radius"] = get_radius();
	} else if (shape_type == "trimesh" || shape_type == "hull") {
		d["mesh"] = get_mesh_index();
	}
	if (is_trigger) {
		d["isTrigger"] = is_trigger;
	}
	return d;
}
