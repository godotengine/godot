//
// Created by amara on 26/11/2021.
//

#ifndef LILYPHYS_L_COLLISION_SOLVER_H
#define LILYPHYS_L_COLLISION_SOLVER_H

#include "core/math/vector3.h"
#include "thirdparty/libccd/ccd/ccd.h"

class LICollisionObject;

struct CollisionResult {
    bool intersect;
    real_t depth;
    Vector3 dir;
    Vector3 pos;
};

class LCollisionSolver {
private:
    ccd_t ccd;

public:
    CollisionResult check_collision(LICollisionObject* object1, LICollisionObject* object2);
    LCollisionSolver();
};


#endif //LILYPHYS_L_COLLISION_SOLVER_H
