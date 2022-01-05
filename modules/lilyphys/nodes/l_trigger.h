//
// Created by amara on 05/01/2022.
//

#ifndef LILYPHYS_L_TRIGGER_H
#define LILYPHYS_L_TRIGGER_H

#include "l_collision_object.h"

class LTrigger : public LCollisionObject {
    GDCLASS(LTrigger, LCollisionObject)
public:
    LTrigger();
    void _state_changed(Object *p_state);
protected:
    static void _bind_methods();
    void _notification(int p_what);
private:
    Array collisions;
    Map<RID, NodePath> body_register;
};


#endif //LILYPHYS_L_TRIGGER_H
