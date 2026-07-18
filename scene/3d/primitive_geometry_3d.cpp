/**************************************************************************/
/*  primitive_geometry_3d.cpp                                             */
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

#include "primitive_geometry_3d.h"
#include "core/math/aabb.h"
#include "core/math/triangle_mesh.h"
#include "core/object/class_db.h"
#include "core/object/property_info.h"
#include "core/os/memory.h"
#include "scene/main/node.h"
#include "scene/resources/3d/box_shape_3d.h"
#include "scene/resources/3d/capsule_shape_3d.h"
#include "scene/resources/3d/cylinder_shape_3d.h"
#include "scene/resources/3d/primitive_meshes.h"
#include "scene/resources/3d/sphere_shape_3d.h"
#include "servers/physics_3d/physics_server_3d.h"

void PrimitiveGeometry3D::set_material(const Ref<Material> &p_material) {
	material = p_material;
	if (mesh.is_valid()) {
		mesh->set_material(material);
	}
};
Ref<Material> PrimitiveGeometry3D::get_material() const {
	return material;
};

Ref<TriangleMesh> PrimitiveGeometry3D::generate_triangle_mesh() const {
	return mesh->generate_triangle_mesh();
}

#ifndef PHYSICS_3D_DISABLED
void PrimitiveGeometry3D::initiate_collision() {
	PhysicsServer3D *singleton = PhysicsServer3D::get_singleton();
	instantiate_collision_shape();
	collision_body = singleton->body_create();
	singleton->body_set_mode(collision_body, PS3DE::BODY_MODE_STATIC);
	singleton->body_set_state(collision_body, PS3DE::BODY_STATE_TRANSFORM, get_global_transform());
	singleton->body_add_shape(collision_body, shape->get_rid());
	singleton->body_set_space(collision_body, get_world_3d()->get_space());
	singleton->body_attach_object_instance_id(collision_body, get_instance_id());
	set_collision_layer(collision_layer);
	set_collision_mask(collision_mask);
	set_collision_priority(collision_priority);
}

void PrimitiveGeometry3D::destroy_collision() {
	PhysicsServer3D::get_singleton()->free_rid(collision_body);
	collision_body = RID();
	shape = nullptr;
}


void PrimitiveGeometry3D::set_use_collision(bool p_enable) {
	if (use_collision == p_enable) {
		return;
	}

	use_collision = p_enable;

	if (!is_inside_tree()) {
		return;
	}

	if (use_collision) {
		initiate_collision();
	} else {
		destroy_collision();
	}

	notify_property_list_changed();
	update_gizmos();
}

bool PrimitiveGeometry3D::is_using_collision() const {
	return use_collision;
}

void PrimitiveGeometry3D::set_collision_layer(uint32_t p_layer) {
	collision_layer = p_layer;
	if (collision_body.is_valid()) {
		PhysicsServer3D::get_singleton()->body_set_collision_layer(collision_body, p_layer);
	}
}

uint32_t PrimitiveGeometry3D::get_collision_layer() const {
	return collision_layer;
}

void PrimitiveGeometry3D::set_collision_mask(uint32_t p_mask) {
	collision_mask = p_mask;
	if (collision_body.is_valid()) {
		PhysicsServer3D::get_singleton()->body_set_collision_mask(collision_body, p_mask);
	}
}

uint32_t PrimitiveGeometry3D::get_collision_mask() const {
	return collision_mask;
}

void PrimitiveGeometry3D::set_collision_layer_value(int p_layer_number, bool p_value) {
	ERR_FAIL_COND_MSG(p_layer_number < 1, "Collision layer number must be between 1 and 32 inclusive.");
	ERR_FAIL_COND_MSG(p_layer_number > 32, "Collision layer number must be between 1 and 32 inclusive.");
	uint32_t layer = get_collision_layer();
	if (p_value) {
		layer |= 1 << (p_layer_number - 1);
	} else {
		layer &= ~(1 << (p_layer_number - 1));
	}
	set_collision_layer(layer);
}

bool PrimitiveGeometry3D::get_collision_layer_value(int p_layer_number) const {
	ERR_FAIL_COND_V_MSG(p_layer_number < 1, false, "Collision layer number must be between 1 and 32 inclusive.");
	ERR_FAIL_COND_V_MSG(p_layer_number > 32, false, "Collision layer number must be between 1 and 32 inclusive.");
	return get_collision_layer() & (1 << (p_layer_number - 1));
}

void PrimitiveGeometry3D::set_collision_mask_value(int p_layer_number, bool p_value) {
	ERR_FAIL_COND_MSG(p_layer_number < 1, "Collision layer number must be between 1 and 32 inclusive.");
	ERR_FAIL_COND_MSG(p_layer_number > 32, "Collision layer number must be between 1 and 32 inclusive.");
	uint32_t mask = get_collision_mask();
	if (p_value) {
		mask |= 1 << (p_layer_number - 1);
	} else {
		mask &= ~(1 << (p_layer_number - 1));
	}
	set_collision_mask(mask);
}

bool PrimitiveGeometry3D::get_collision_mask_value(int p_layer_number) const {
	ERR_FAIL_COND_V_MSG(p_layer_number < 1, false, "Collision layer number must be between 1 and 32 inclusive.");
	ERR_FAIL_COND_V_MSG(p_layer_number > 32, false, "Collision layer number must be between 1 and 32 inclusive.");
	return get_collision_mask() & (1 << (p_layer_number - 1));
}

void PrimitiveGeometry3D::set_collision_priority(real_t p_priority) {
	collision_priority = p_priority;
	if (collision_body.is_valid()) {
		PhysicsServer3D::get_singleton()->body_set_collision_priority(collision_body, p_priority);
	}
}

real_t PrimitiveGeometry3D::get_collision_priority() const {
	return collision_priority;
}
#endif // PHYSICS_3D_DISABLED

AABB PrimitiveGeometry3D::get_aabb() const {
	return mesh->get_aabb();
}

void PrimitiveGeometry3D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
#ifndef PHYSICS_3D_DISABLED
			if (use_collision) {
				initiate_collision();
			}
#endif
			instantiate_mesh();
		} break;
		case NOTIFICATION_EXIT_TREE: {
#ifndef PHYSICS_3D_DISABLED
			if (collision_body.is_valid()) {
				destroy_collision();
			}
#endif
			mesh = nullptr;
		} break;

#ifndef PHYSICS_3D_DISABLED
		case NOTIFICATION_TRANSFORM_CHANGED: {
			if (collision_body.is_valid()) {
				PhysicsServer3D::get_singleton()->body_set_state(collision_body, PS3DE::BODY_STATE_TRANSFORM, get_global_transform());
			}
		} break;
#endif
	}
};

void PrimitiveGeometry3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_material", "material"), &Box3D::set_material);
	ClassDB::bind_method(D_METHOD("get_material"), &Box3D::get_material);

#ifndef PHYSICS_3D_DISABLED
	ClassDB::bind_method(D_METHOD("set_use_collision", "operation"), &PrimitiveGeometry3D::set_use_collision);
	ClassDB::bind_method(D_METHOD("is_using_collision"), &PrimitiveGeometry3D::is_using_collision);

	ClassDB::bind_method(D_METHOD("set_collision_layer", "layer"), &PrimitiveGeometry3D::set_collision_layer);
	ClassDB::bind_method(D_METHOD("get_collision_layer"), &PrimitiveGeometry3D::get_collision_layer);

	ClassDB::bind_method(D_METHOD("set_collision_mask", "mask"), &PrimitiveGeometry3D::set_collision_mask);
	ClassDB::bind_method(D_METHOD("get_collision_mask"), &PrimitiveGeometry3D::get_collision_mask);

	ClassDB::bind_method(D_METHOD("set_collision_mask_value", "layer_number", "value"), &PrimitiveGeometry3D::set_collision_mask_value);
	ClassDB::bind_method(D_METHOD("get_collision_mask_value", "layer_number"), &PrimitiveGeometry3D::get_collision_mask_value);

	ClassDB::bind_method(D_METHOD("set_collision_layer_value", "layer_number", "value"), &PrimitiveGeometry3D::set_collision_layer_value);
	ClassDB::bind_method(D_METHOD("get_collision_layer_value", "layer_number"), &PrimitiveGeometry3D::get_collision_layer_value);

	ClassDB::bind_method(D_METHOD("set_collision_priority", "priority"), &PrimitiveGeometry3D::set_collision_priority);
	ClassDB::bind_method(D_METHOD("get_collision_priority"), &PrimitiveGeometry3D::get_collision_priority);
#endif // PHYSICS_3D_DISABLED

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "material", PROPERTY_HINT_RESOURCE_TYPE, "BaseMaterial3D,ShaderMaterial"), "set_material", "get_material");

#ifndef PHYSICS_3D_DISABLED
	ADD_GROUP("Collision", "collision_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "use_collision"), "set_use_collision", "is_using_collision");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "collision_layer", PROPERTY_HINT_LAYERS_3D_PHYSICS), "set_collision_layer", "get_collision_layer");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "collision_mask", PROPERTY_HINT_LAYERS_3D_PHYSICS), "set_collision_mask", "get_collision_mask");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "collision_priority"), "set_collision_priority", "get_collision_priority");
#endif // PHYSICS_3D_DISABLED
}

void PrimitiveGeometry3D::_validate_property(PropertyInfo &p_property) const {
	if (!use_collision && p_property.name.begins_with("collision_")) {
		p_property.usage = PROPERTY_USAGE_NO_EDITOR;
	}
}

// Primitive classes

void Box3D::set_size(const Vector3 &p_size) {
	ERR_FAIL_COND_MSG(p_size.x < 0.f || p_size.y < 0.f || p_size.z < 0.f, "Box3D size cannot be negative.");
	size = p_size;
	if (mesh.is_valid()) {
		((Ref<BoxMesh>)mesh)->set_size(size);
	}
#ifndef PHYSICS_3D_DISABLED
	if (shape.is_valid()) {
		((Ref<BoxShape3D>)shape)->set_size(size);
	}
#endif
	update_gizmos();
}

Vector3 Box3D::get_size() const {
	return size;
}

void Box3D::instantiate_mesh() {
	Ref<BoxMesh> box_mesh = memnew(BoxMesh);
	box_mesh->set_size(size);
	box_mesh->set_material(material);
	set_base(box_mesh->get_rid());
	mesh = box_mesh;
}

#ifndef PHYSICS_3D_DISABLED
void Box3D::instantiate_collision_shape() {
	Ref<BoxShape3D> box_shape = memnew(BoxShape3D);
	box_shape->set_size(size);
	shape = box_shape;
};
#endif

void Box3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_size", "size"), &Box3D::set_size);
	ClassDB::bind_method(D_METHOD("get_size"), &Box3D::get_size);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "size", PROPERTY_HINT_NONE, "suffix:m"), "set_size", "get_size");
}


void Sphere3D::set_radius(const float p_radius) {
	ERR_FAIL_COND_MSG(p_radius < 0.f, "Sphere3D radius cannot be negative.");
	radius = p_radius;
	if (mesh.is_valid()) {
		((Ref<SphereMesh>)mesh)->set_radius(p_radius);
		((Ref<SphereMesh>)mesh)->set_height(p_radius*2.f);
	}
#ifndef PHYSICS_3D_DISABLED
	if (shape.is_valid()) {
		((Ref<SphereShape3D>)shape)->set_radius(radius);
	}
#endif
	update_gizmos();
}

float Sphere3D::get_radius() const {
	return radius;
}

void Sphere3D::instantiate_mesh() {
	Ref<SphereMesh> sphere_mesh = memnew(SphereMesh);
	sphere_mesh->set_radius(radius);
	sphere_mesh->set_height(radius*2.f);
	sphere_mesh->set_material(material);
	set_base(sphere_mesh->get_rid());
	mesh = sphere_mesh;
}

#ifndef PHYSICS_3D_DISABLED
void Sphere3D::instantiate_collision_shape() {
	Ref<SphereShape3D> sphere_shape = memnew(SphereShape3D);
	sphere_shape->set_radius(radius);
	shape = sphere_shape;
};
#endif

void Sphere3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_radius", "radius"), &Sphere3D::set_radius);
	ClassDB::bind_method(D_METHOD("get_radius"), &Sphere3D::get_radius);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "radius", PROPERTY_HINT_NONE, "suffix:m"), "set_radius", "get_radius");
}


void Cylinder3D::set_radius(const float p_radius) {
	ERR_FAIL_COND_MSG(p_radius < 0.f, "Cylinder3D radius cannot be negative.");
	radius = p_radius;
	if (mesh.is_valid()) {
		((Ref<CylinderMesh>)mesh)->set_bottom_radius(p_radius);
		((Ref<CylinderMesh>)mesh)->set_top_radius(p_radius);
	}
#ifndef PHYSICS_3D_DISABLED
	if (shape.is_valid()) {
		((Ref<CylinderShape3D>)shape)->set_radius(radius);
	}
#endif
	update_gizmos();
}

float Cylinder3D::get_radius() const {
	return radius;
}

void Cylinder3D::set_height(const float p_height) {
	ERR_FAIL_COND_MSG(p_height < 0.f, "Cylinder3D height cannot be negative.");
	height = p_height;
	if (mesh.is_valid()) {
		((Ref<CylinderMesh>)mesh)->set_height(p_height);
	}
#ifndef PHYSICS_3D_DISABLED
	if (shape.is_valid()) {
		((Ref<CylinderShape3D>)shape)->set_height(radius);
	}
#endif
	update_gizmos();
}

float Cylinder3D::get_height() const {
	return height;
}

void Cylinder3D::instantiate_mesh() {
	Ref<CylinderMesh> cylinder_mesh = memnew(CylinderMesh);
	cylinder_mesh->set_bottom_radius(radius);
	cylinder_mesh->set_top_radius(radius);
	cylinder_mesh->set_height(height);
	cylinder_mesh->set_material(material);
	set_base(cylinder_mesh->get_rid());
	mesh = cylinder_mesh;
}

#ifndef PHYSICS_3D_DISABLED
void Cylinder3D::instantiate_collision_shape() {
	Ref<CylinderShape3D> cylinder_shape = memnew(CylinderShape3D);
	cylinder_shape->set_radius(radius);
	cylinder_shape->set_height(height);
	shape = cylinder_shape;
};
#endif

void Cylinder3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_radius", "radius"), &Cylinder3D::set_radius);
	ClassDB::bind_method(D_METHOD("get_radius"), &Cylinder3D::get_radius);
	ClassDB::bind_method(D_METHOD("set_height", "height"), &Cylinder3D::set_height);
	ClassDB::bind_method(D_METHOD("get_height"), &Cylinder3D::get_height);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "radius", PROPERTY_HINT_NONE, "suffix:m"), "set_radius", "get_radius");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "height", PROPERTY_HINT_NONE, "suffix:m"), "set_height", "get_height");
}


void Capsule3D::set_radius(const float p_radius) {
	ERR_FAIL_COND_MSG(p_radius < 0.f, "Capsule3D radius cannot be negative.");
	radius = p_radius;
	if (mesh.is_valid()) {
		((Ref<CapsuleMesh>)mesh)->set_radius(p_radius);
	}
#ifndef PHYSICS_3D_DISABLED
	if (shape.is_valid()) {
		((Ref<CapsuleShape3D>)shape)->set_radius(radius);
	}
#endif
	update_gizmos();
}

float Capsule3D::get_radius() const {
	return radius;
}

void Capsule3D::set_height(const float p_height) {
	ERR_FAIL_COND_MSG(p_height < 0.f, "Capsule3D height cannot be negative.");
	height = p_height;
	if (mesh.is_valid()) {
		((Ref<CapsuleMesh>)mesh)->set_height(p_height);
	}
#ifndef PHYSICS_3D_DISABLED
	if (shape.is_valid()) {
		((Ref<CapsuleShape3D>)shape)->set_height(radius);
	}
#endif
	update_gizmos();
}

float Capsule3D::get_height() const {
	return height;
}

void Capsule3D::instantiate_mesh() {
	Ref<CapsuleMesh> capsule_mesh = memnew(CapsuleMesh);
	capsule_mesh->set_radius(radius);
	capsule_mesh->set_height(height);
	capsule_mesh->set_material(material);
	set_base(capsule_mesh->get_rid());
	mesh = capsule_mesh;
}

#ifndef PHYSICS_3D_DISABLED
void Capsule3D::instantiate_collision_shape() {
	Ref<CapsuleShape3D> capsule_shape = memnew(CapsuleShape3D);
	capsule_shape->set_radius(radius);
	capsule_shape->set_height(height);
	shape = capsule_shape;
};
#endif

void Capsule3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_radius", "radius"), &Capsule3D::set_radius);
	ClassDB::bind_method(D_METHOD("get_radius"), &Capsule3D::get_radius);
	ClassDB::bind_method(D_METHOD("set_height", "height"), &Capsule3D::set_height);
	ClassDB::bind_method(D_METHOD("get_height"), &Capsule3D::get_height);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "radius", PROPERTY_HINT_NONE, "suffix:m"), "set_radius", "get_radius");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "height", PROPERTY_HINT_NONE, "suffix:m"), "set_height", "get_height");
}
