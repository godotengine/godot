//
// Created by amara on 17/11/2021.
//

#ifndef LILYPHYS_LI_FORCE_GENERATOR_H
#define LILYPHYS_LI_FORCE_GENERATOR_H

#include "core/rid.h"
#include "li_physics_body.h"

class LIForceGenerator : public RID_Data {
public:
    virtual void update_force(LIPhysicsBody* p_body, real_t p_delta) = 0;
};


#endif //LILYPHYS_LI_FORCE_GENERATOR_H
