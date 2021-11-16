//
// Created by amara on 19/10/2021.
//

#ifndef LILYPHYS_LILYPHYS_SERVER_H
#define LILYPHYS_LILYPHYS_SERVER_H

#include "core/object.h"
#include "core/rid.h"
#include "_lilyphys_server.h"
#include "internal/li_physics_body.h"


class LilyphysServer : public Object {
    GDCLASS(LilyphysServer, Object);
    static LilyphysServer *singleton;
    RID_Owner<LIPhysicsBody> body_owner;
    Set<RID> bodies;
    bool active = true;
protected:
    static void _bind_methods();
public:
    static LilyphysServer *get_singleton();
    LilyphysServer();
    void init();
    void finish();
    void set_active(bool p_active) { active = p_active; }
    void step(float p_step);
    RID create_physics_body();
    void set_integration_callback(RID p_body, Object *p_receiver, const StringName& p_method, const Variant& p_user_data = Variant());
};


#endif //LILYPHYS_LILYPHYS_SERVER_H
