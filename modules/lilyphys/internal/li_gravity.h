//
// Created by amara on 17/11/2021.
//

#ifndef LILYPHYS_LI_GRAVITY_H
#define LILYPHYS_LI_GRAVITY_H

#include "li_force_generator.h"

class LIGravity : public LIForceGenerator {
    void update_force(LIPhysicsBody* p_body, real_t p_delta) override;
};


#endif //LILYPHYS_LI_GRAVITY_H
