//
// Created by amara on 19/10/2021.
//

#include "li_collision_object.h"

#include "../lilyphys_server.h"

void LICollisionObject::set_self(RID p_rid) {
    self = p_rid;
}

RID LICollisionObject::get_self() {
    return self;
}

LICollisionObject::LICollisionObject(LICollisionObject::Type p_type) {
    type = p_type;
}

size_t LICollisionObject::add_shape(RID p_rid) {
    ShapeData sd;
    sd.shape = p_rid;
    LilyphysServer::get_singleton()->shape_add_owner(p_rid, this);
    size_t idx = last_idx;
    last_idx++;
    shapes[idx] = sd;
    return idx;
}

void LICollisionObject::remove_shape(size_t p_id) {
    ERR_FAIL_COND(!shapes.has(p_id));
    LilyphysServer::get_singleton()->shape_remove_owner(shapes[p_id].shape, this);
    shapes.erase(p_id);
}

void LICollisionObject::set_shape_disabled(size_t p_id, bool p_disabled) {
    ERR_FAIL_COND(!shapes.has(p_id));
    shapes[p_id].disabled = p_disabled;
}

void LICollisionObject::set_shape_transform(size_t p_id, const Transform &p_transform) {
    ERR_FAIL_COND(!shapes.has(p_id));
    shapes[p_id].transform = p_transform;
}

bool LICollisionObject::get_shape_exists(size_t p_id) {
    return shapes.has(p_id);
}

void LICollisionObject::clear_shapes() {
    for (Map<size_t, ShapeData>::Element *E = shapes.front(); E; E = E->next()) {
        remove_shape(E->key());
    }
}

void LICollisionObject::remove_shape(RID p_rid) {
    for (Map<size_t, ShapeData>::Element *E = shapes.front(); E; E = E->next()) {
        if (E->get().shape == p_rid) {
            remove_shape(E->key());
        }
    }
}

LICollisionObject::~LICollisionObject() {
    clear_shapes();
}
