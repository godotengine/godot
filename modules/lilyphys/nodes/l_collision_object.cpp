//
// Created by amara on 19/10/2021.
//

#include "l_collision_object.h"

#include "../lilyphys_server.h"

void LCollisionObject::_bind_methods() {
    ClassDB::bind_method(D_METHOD("get_rid"), &LCollisionObject::get_rid);
}

RID LCollisionObject::get_rid() {
    return rid;
}

size_t LCollisionObject::create_shape_owner(LCollisionShape *p_owner) {
    ShapeData sd;
    size_t id = last_idx;
    last_idx++;
    sd.owner = p_owner;
    shapes[id] = sd;

    return id;
}

void LCollisionObject::remove_shape_owner(size_t p_id) {
    shape_owner_remove_shape(p_id);
    shapes.erase(p_id);
}

void LCollisionObject::shape_owner_remove_shape(size_t p_id) {
    switch (type) {
        case TYPE_BODY:
            LilyphysServer::get_singleton()->physics_body_remove_shape(get_rid(), shapes[p_id].id);
            break;
    }
}

void LCollisionObject::shape_owner_set_shape(size_t p_id, const Ref<LShape> &p_shape) {
    ERR_FAIL_COND(!shapes.has(p_id));
    shapes[p_id].shape = p_shape;
    switch (type) {
        case TYPE_BODY:
            if (LilyphysServer::get_singleton()->physics_body_get_shape_exists(get_rid(), shapes[p_id].id)) {
                LilyphysServer::get_singleton()->physics_body_remove_shape(get_rid(), shapes[p_id].id);
            }
            shapes[p_id].id = LilyphysServer::get_singleton()->physics_body_add_shape(get_rid(), p_shape->get_self());
            break;
    }
}

void LCollisionObject::shape_owner_set_disabled(size_t p_id, bool p_disabled) {
    ERR_FAIL_COND(!shapes.has(p_id));
    shapes[p_id].disabled = p_disabled;
    switch (type) {
        case TYPE_BODY:
            LilyphysServer::get_singleton()->physics_body_shape_set_disabled(get_rid(), shapes[p_id].id, p_disabled);
            break;
    }
}

void LCollisionObject::shape_owner_set_transform(size_t p_id, const Transform &p_transform) {
    ERR_FAIL_COND(!shapes.has(p_id));
    shapes[p_id].xform = p_transform;
    switch (type) {
        case Type::TYPE_BODY:
            LilyphysServer::get_singleton()->physics_body_shape_set_transform(get_rid(), shapes[p_id].id, p_transform);
            break;
    }
}

LCollisionObject::~LCollisionObject() {
    LilyphysServer::get_singleton()->queue_free_rid(rid);
}

void LCollisionObject::_notification(int p_what) {
    switch (p_what) {
        case NOTIFICATION_ENTER_TREE:
        case NOTIFICATION_EXIT_TREE:
            LilyphysServer::get_singleton()->set_physics_body_parameter(get_rid(), NODE_PATH, get_path());
            break;
    }
}
