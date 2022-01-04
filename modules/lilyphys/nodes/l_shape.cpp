//
// Created by amara on 25/11/2021.
//

#include "l_shape.h"

LShape::LShape(LShapeType p_type) {
    self = LilyphysServer::get_singleton()->create_shape(p_type);
}

RID LShape::get_self() const {
    return self;
}

void LShape::update_shape() {
    emit_changed();
}


void LBoxShape::set_extents(const Vector3 &p_extends) {
    extents = p_extends;
    notify_change_to_owners();
    _change_notify("extents");
    update_shape();
}

Vector3 LBoxShape::get_extents() const {
    return extents;
}

void LBoxShape::update_shape() {
    LShape::update_shape();
    LilyphysServer::get_singleton()->shape_set_data(get_self(), extents);
}

void LBoxShape::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_extents", "extents"), &LBoxShape::set_extents);
    ClassDB::bind_method(D_METHOD("get_extents"), &LBoxShape::get_extents);

    ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "extents"), "set_extents", "get_extents");
}

LBoxShape::LBoxShape() : LShape(LShapeType::SHAPE_BOX){
}

void LSphereShape::update_shape() {
    LShape::update_shape();
    LilyphysServer::get_singleton()->shape_set_data(get_self(), radius);
}

void LSphereShape::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_radius", "p_radius"), &LSphereShape::set_radius);
    ClassDB::bind_method(D_METHOD("get_radius"), &LSphereShape::get_radius);

    ADD_PROPERTY(PropertyInfo(Variant::REAL, "radius"), "set_radius", "get_radius");
}

void LSphereShape::set_radius(const real_t &p_radius) {
    radius = p_radius;
    notify_change_to_owners();
    _change_notify("radius");
    update_shape();
}

real_t LSphereShape::get_radius() const {
    return radius;
}

LSphereShape::LSphereShape() : LShape(LShapeType::SHAPE_SPHERE){
}