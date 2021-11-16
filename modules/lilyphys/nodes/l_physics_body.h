//
// Created by amara on 16/11/2021.
//

#ifndef LILYPHYS_L_PHYSICS_BODY_H
#define LILYPHYS_L_PHYSICS_BODY_H

#include "l_collision_object.h"

class LPhysicsBody : public LCollisionObject {
GDCLASS(LPhysicsBody, LCollisionObject);
private:
    Vector3 linear_velocity;
    Vector3 angular_velocity;
protected:
    static void _bind_methods();
public:
    LPhysicsBody();
    void _state_changed(Object *p_state);
};


#endif //LILYPHYS_L_PHYSICS_BODY_H
