/**************************************************************************/
/*  static_body_3d.cpp                                                    */
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

#include "static_body_3d.h"

#ifndef NAVIGATION_3D_DISABLED
#include "core/math/convex_hull.h"
#include "scene/resources/3d/box_shape_3d.h"
#include "scene/resources/3d/capsule_shape_3d.h"
#include "scene/resources/3d/concave_polygon_shape_3d.h"
#include "scene/resources/3d/convex_polygon_shape_3d.h"
#include "scene/resources/3d/cylinder_shape_3d.h"
#include "scene/resources/3d/height_map_shape_3d.h"
#include "scene/resources/3d/navigation_mesh_source_geometry_data_3d.h"
#include "scene/resources/3d/primitive_meshes.h"
#include "scene/resources/3d/shape_3d.h"
#include "scene/resources/3d/sphere_shape_3d.h"
#include "scene/resources/3d/world_boundary_shape_3d.h"
#include "scene/resources/navigation_mesh.h"
#include "servers/navigation_3d/navigation_server_3d.h"

Callable StaticBody3D::_navmesh_source_geometry_parsing_callback;
RID StaticBody3D::_navmesh_source_geometry_parser;
#endif // NAVIGATION_3D_DISABLED

void StaticBody3D::set_physics_material_override(const Ref<PhysicsMaterial> &p_physics_material_override) {
	if (physics_material_override.is_valid()) {
		physics_material_override->disconnect_changed(callable_mp(this, &StaticBody3D::_reload_physics_characteristics));
	}

	physics_material_override = p_physics_material_override;

	if (physics_material_override.is_valid()) {
		physics_material_override->connect_changed(callable_mp(this, &StaticBody3D::_reload_physics_characteristics));
	}
	_reload_physics_characteristics();
}

Ref<PhysicsMaterial> StaticBody3D::get_physics_material_override() const {
	return physics_material_override;
}

void StaticBody3D::set_constant_linear_velocity(const Vector3 &p_vel) {
	constant_linear_velocity = p_vel;

	PhysicsServer3D::get_singleton()->body_set_state(get_rid(), PhysicsServer3D::BODY_STATE_LINEAR_VELOCITY, constant_linear_velocity);
}

void StaticBody3D::set_constant_angular_velocity(const Vector3 &p_vel) {
	constant_angular_velocity = p_vel;

	PhysicsServer3D::get_singleton()->body_set_state(get_rid(), PhysicsServer3D::BODY_STATE_ANGULAR_VELOCITY, constant_angular_velocity);
}

Vector3 StaticBody3D::get_constant_linear_velocity() const {
	return constant_linear_velocity;
}

Vector3 StaticBody3D::get_constant_angular_velocity() const {
	return constant_angular_velocity;
}

void StaticBody3D::_reload_physics_characteristics() {
	if (physics_material_override.is_null()) {
		PhysicsServer3D::get_singleton()->body_set_param(get_rid(), PhysicsServer3D::BODY_PARAM_BOUNCE, 0);
		PhysicsServer3D::get_singleton()->body_set_param(get_rid(), PhysicsServer3D::BODY_PARAM_FRICTION, 1);
	} else {
		PhysicsServer3D::get_singleton()->body_set_param(get_rid(), PhysicsServer3D::BODY_PARAM_BOUNCE, physics_material_override->computed_bounce());
		PhysicsServer3D::get_singleton()->body_set_param(get_rid(), PhysicsServer3D::BODY_PARAM_FRICTION, physics_material_override->computed_friction());
	}
}

#ifndef NAVIGATION_3D_DISABLED
void StaticBody3D::navmesh_parse_init() {
	ERR_FAIL_NULL(NavigationServer3D::get_singleton());
	if (!_navmesh_source_geometry_parser.is_valid()) {
		_navmesh_source_geometry_parsing_callback = callable_mp_static(&StaticBody3D::navmesh_parse_source_geometry);
		_navmesh_source_geometry_parser = NavigationServer3D::get_singleton()->source_geometry_parser_create();
		NavigationServer3D::get_singleton()->source_geometry_parser_set_callback(_navmesh_source_geometry_parser, _navmesh_source_geometry_parsing_callback);
	}
}

void StaticBody3D::navmesh_parse_source_geometry(const Ref<NavigationMesh> &p_navigation_mesh, Ref<NavigationMeshSourceGeometryData3D> p_source_geometry_data, Node *p_node) {
	StaticBody3D *static_body = Object::cast_to<StaticBody3D>(p_node);

	if (static_body == nullptr) {
		return;
	}

	NavigationMesh::ParsedGeometryType parsed_geometry_type = p_navigation_mesh->get_parsed_geometry_type();
	uint32_t parsed_collision_mask = p_navigation_mesh->get_collision_mask();

	if ((parsed_geometry_type == NavigationMesh::PARSED_GEOMETRY_STATIC_COLLIDERS || parsed_geometry_type == NavigationMesh::PARSED_GEOMETRY_BOTH) && (static_body->get_collision_layer() & parsed_collision_mask)) {
		List<uint32_t> shape_owners;
		static_body->get_shape_owners(&shape_owners);
		for (uint32_t shape_owner : shape_owners) {
			if (static_body->is_shape_owner_disabled(shape_owner)) {
				continue;
			}
			const int shape_count = static_body->shape_owner_get_shape_count(shape_owner);
			for (int shape_index = 0; shape_index < shape_count; shape_index++) {
				Ref<Shape3D> s = static_body->shape_owner_get_shape(shape_owner, shape_index);
				if (s.is_null()) {
					continue;
				}

				const Transform3D transform = static_body->get_global_transform() * static_body->shape_owner_get_transform(shape_owner);

				BoxShape3D *box = Object::cast_to<BoxShape3D>(*s);
				if (box) {
					Vector<Vector3> triangles = box->get_triangles();
					if (!triangles.is_empty()) {
						p_source_geometry_data->add_faces(triangles, transform);
					}
				}

				CapsuleShape3D *capsule = Object::cast_to<CapsuleShape3D>(*s);
				if (capsule) {
					Vector<Vector3> triangles = capsule->get_triangles();
					if (!triangles.is_empty()) {
						p_source_geometry_data->add_faces(triangles, transform);
					}
				}

				CylinderShape3D *cylinder = Object::cast_to<CylinderShape3D>(*s);
				if (cylinder) {
					Vector<Vector3> triangles = cylinder->get_triangles();
					if (!triangles.is_empty()) {
						p_source_geometry_data->add_faces(triangles, transform);
					}
				}

				SphereShape3D *sphere = Object::cast_to<SphereShape3D>(*s);
				if (sphere) {
					Vector<Vector3> triangles = sphere->get_triangles();
					if (!triangles.is_empty()) {
						p_source_geometry_data->add_faces(triangles, transform);
					}
				}

				ConcavePolygonShape3D *concave_polygon = Object::cast_to<ConcavePolygonShape3D>(*s);
				if (concave_polygon) {
					Vector<Vector3> triangles = concave_polygon->get_triangles();
					if (!triangles.is_empty()) {
						p_source_geometry_data->add_faces(triangles, transform);
					}
				}

				ConvexPolygonShape3D *convex_polygon = Object::cast_to<ConvexPolygonShape3D>(*s);
				if (convex_polygon) {
					Vector<Vector3> triangles = convex_polygon->get_triangles();
					if (!triangles.is_empty()) {
						p_source_geometry_data->add_faces(triangles, transform);
					}
				}

				HeightMapShape3D *heightmap_shape = Object::cast_to<HeightMapShape3D>(*s);
				if (heightmap_shape) {
					Vector<Vector3> triangles = heightmap_shape->get_triangles();
					if (!triangles.is_empty()) {
						p_source_geometry_data->add_faces(triangles, transform);
					}
				}
			}
		}
	}
}
#endif // NAVIGATION_3D_DISABLED

void StaticBody3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_constant_linear_velocity", "vel"), &StaticBody3D::set_constant_linear_velocity);
	ClassDB::bind_method(D_METHOD("set_constant_angular_velocity", "vel"), &StaticBody3D::set_constant_angular_velocity);
	ClassDB::bind_method(D_METHOD("get_constant_linear_velocity"), &StaticBody3D::get_constant_linear_velocity);
	ClassDB::bind_method(D_METHOD("get_constant_angular_velocity"), &StaticBody3D::get_constant_angular_velocity);

	ClassDB::bind_method(D_METHOD("set_physics_material_override", "physics_material_override"), &StaticBody3D::set_physics_material_override);
	ClassDB::bind_method(D_METHOD("get_physics_material_override"), &StaticBody3D::get_physics_material_override);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "physics_material_override", PROPERTY_HINT_RESOURCE_TYPE, PhysicsMaterial::get_class_static()), "set_physics_material_override", "get_physics_material_override");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "constant_linear_velocity", PROPERTY_HINT_NONE, "suffix:m/s"), "set_constant_linear_velocity", "get_constant_linear_velocity");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "constant_angular_velocity", PROPERTY_HINT_NONE, U"radians_as_degrees,suffix:\u00B0/s"), "set_constant_angular_velocity", "get_constant_angular_velocity");
}

StaticBody3D::StaticBody3D(PhysicsServer3D::BodyMode p_mode) :
		PhysicsBody3D(p_mode) {
}
