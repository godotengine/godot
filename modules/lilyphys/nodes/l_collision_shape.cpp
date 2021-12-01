//
// Created by amara on 23/11/2021.
//

#include "l_collision_shape.h"

#include "l_collision_object.h"

void LCollisionShape::_bind_methods() {
    ClassDB::bind_method(D_METHOD("resource_changed", "resource"), &LCollisionShape::resource_changed);
    ClassDB::bind_method(D_METHOD("set_shape", "shape"), &LCollisionShape::set_shape);
    ClassDB::bind_method(D_METHOD("get_shape"), &LCollisionShape::get_shape);
    ClassDB::bind_method(D_METHOD("set_disabled", "enable"), &LCollisionShape::set_disabled);
    ClassDB::bind_method(D_METHOD("is_disabled"), &LCollisionShape::is_disabled);

    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "shape", PROPERTY_HINT_RESOURCE_TYPE, "LShape"), "set_shape", "get_shape");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "disabled"), "set_disabled", "is_disabled");
}

void LCollisionShape::set_shape(const Ref<LShape> &p_shape) {
    if (p_shape == shape) {
        return;
    }
    if (shape.is_valid()) {
        shape->unregister_owner(this);
    }
    shape = p_shape;
    if (shape.is_valid()) {
        shape->register_owner(this);
        if (parent) {
            parent->shape_owner_set_shape(owner_id, shape);
        }
    }
    update_gizmo();
}

Ref<LShape> LCollisionShape::get_shape() const {
    return shape;
}

void LCollisionShape::set_disabled(bool p_disabled) {
    disabled = p_disabled;
    update_gizmo();
    if (parent) {
        update_parent();
    }
}

bool LCollisionShape::is_disabled() const {
    return disabled;
}

void LCollisionShape::_notification(int p_what) {
    switch (p_what) {
        case NOTIFICATION_PARENTED:
            parent = Object::cast_to<LCollisionObject>(get_parent());
            if (parent) {
                owner_id = parent->create_shape_owner(this);
                if (shape.is_valid()) {
                    parent->shape_owner_set_shape(owner_id, shape);
                }
                update_parent();
            }
            break;
        case NOTIFICATION_ENTER_TREE:
            if (parent) {
                update_parent();
            }
            break;
        case NOTIFICATION_LOCAL_TRANSFORM_CHANGED:
            if (parent) {
                update_parent(true);
            }
            break;
        case NOTIFICATION_UNPARENTED:
            if (parent) {
                parent->shape_owner_remove_shape(owner_id);
            }
            owner_id = 0;
            parent = nullptr;
            break;
    }
}

void LCollisionShape::update_parent(bool p_transform_only) {
    parent->shape_owner_set_transform(owner_id, get_transform());
    if (p_transform_only) {
        return;
    }
    parent->shape_owner_set_disabled(owner_id, disabled);
}

LCollisionShape::~LCollisionShape() {
    if (shape.is_valid()) {
        shape->unregister_owner(this);
    }
}

void LCollisionShape::resource_changed(RES res) {
    update_gizmo();
}

LCollisionShape::LCollisionShape() {
    set_notify_local_transform(true);
}


