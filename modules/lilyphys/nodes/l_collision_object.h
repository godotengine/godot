//
// Created by amara on 19/10/2021.
//

#ifndef LILYPHYS_L_COLLISION_OBJECT_H
#define LILYPHYS_L_COLLISION_OBJECT_H

#include "scene/3d/spatial.h"

class LCollisionObject : public Spatial {
    GDCLASS(LCollisionObject, Spatial);

    RID rid;

protected:
    static void _bind_methods();

public:
    LCollisionObject();
    RID get_rid();
};


#endif //LILYPHYS_L_COLLISION_OBJECT_H
