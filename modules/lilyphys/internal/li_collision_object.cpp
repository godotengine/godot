//
// Created by amara on 19/10/2021.
//

#include "li_collision_object.h"

void LICollisionObject::set_self(RID p_rid) {
    self = p_rid;
}

RID LICollisionObject::get_self() {
    return self;
}

LICollisionObject::LICollisionObject(LICollisionObject::Type p_type) {
    type = p_type;
}
