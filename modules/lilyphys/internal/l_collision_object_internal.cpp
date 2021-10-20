//
// Created by amara on 19/10/2021.
//

#include "l_collision_object_internal.h"

void LCollisionObjectInternal::set_self(RID p_rid) {
    self = p_rid;
}

RID LCollisionObjectInternal::get_self() {
    return self;
}
