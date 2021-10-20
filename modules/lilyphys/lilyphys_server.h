//
// Created by amara on 19/10/2021.
//

#ifndef LILYPHYS_LILYPHYS_SERVER_H
#define LILYPHYS_LILYPHYS_SERVER_H

#include "core/object.h"
#include "core/rid.h"
#include "_lilyphys_server.h"
#include "internal/l_collision_object_internal.h"


class LilyphysServer : public Object {
    GDCLASS(LilyphysServer, Object);

    static LilyphysServer *singleton;

    RID_Owner<LCollisionObjectInternal> collision_object_owner;
    Set<RID> collision_objects;

protected:
    static void _bind_methods();

private:

public:
    static LilyphysServer *get_singleton();
    LilyphysServer();
    void init();
    void finish();

    void step(float p_step);

    RID create_collision_object();
    Transform get_collision_object_transform(RID p_rid);
};


#endif //LILYPHYS_LILYPHYS_SERVER_H
