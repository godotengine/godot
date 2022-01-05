//
// Created by amara on 05/01/2022.
//

#ifndef LILYPHYS_L_STATIC_BODY_H
#define LILYPHYS_L_STATIC_BODY_H

#include "l_collision_object.h"

class LStaticBody : public LCollisionObject {
GDCLASS(LStaticBody, LCollisionObject)
public:
    LStaticBody();
    void _state_changed(Object *p_state);
protected:
    static void _bind_methods();
    void _notification(int p_what);
};


#endif //LILYPHYS_L_STATIC_BODY_H
