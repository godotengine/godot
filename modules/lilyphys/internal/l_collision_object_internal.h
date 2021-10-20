//
// Created by amara on 19/10/2021.
//

#ifndef LILYPHYS_L_COLLISION_OBJECT_INTERNAL_H
#define LILYPHYS_L_COLLISION_OBJECT_INTERNAL_H

#include "core/rid.h"
#include "core/math/transform.h"

class LCollisionObjectInternal : public RID_Data {
    Transform transform;
    RID self;

public:
    Transform get_transform() { return transform; }
    void set_transform(const Transform& p_transform) { transform = p_transform; }
    void set_self(RID p_rid);
    RID get_self();
};


#endif //LILYPHYS_L_COLLISION_OBJECT_INTERNAL_H
