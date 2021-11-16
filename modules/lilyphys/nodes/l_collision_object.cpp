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
