//
// Created by amara on 19/10/2021.
//

#ifndef LILYPHYS_LI_COLLISION_OBJECT_H
#define LILYPHYS_LI_COLLISION_OBJECT_H

#include "core/rid.h"

class LICollisionObject : public RID_Data {
public:
    enum Type {
        TYPE_BODY,
        TYPE_AREA // Not implemented yet.
    };

private:
    RID self;
    Type type;

public:
    LICollisionObject(Type p_type);
    void set_self(RID p_rid);
    RID get_self();
};


#endif //LILYPHYS_LI_COLLISION_OBJECT_H
