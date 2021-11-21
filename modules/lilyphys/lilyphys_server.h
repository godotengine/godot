//
// Created by amara on 19/10/2021.
//

#ifndef LILYPHYS_LILYPHYS_SERVER_H
#define LILYPHYS_LILYPHYS_SERVER_H

#include "core/object.h"
#include "core/rid.h"
#include "_lilyphys_server.h"
#include "internal/li_physics_body.h"
#include "internal/li_force_generator.h"
#include "internal/li_gravity.h"


class LilyphysServer : public Object {
    GDCLASS(LilyphysServer, Object);
    static LilyphysServer *singleton;
    RID_Owner<LIPhysicsBody> body_owner;
    Set<RID> bodies;
    RID_Owner<LIForceGenerator> generator_owner;
    Set<RID> generators;
    bool active = true;
    struct Registration {
        RID body;
        RID generator;
    };
    List<Registration> registry;
    RID gravity;
    RID spring;
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
    void register_generator(RID p_body, RID p_generator);
    void unregister_generator(RID p_body, RID p_generator);
    void clear_registry();
    void set_physics_body_parameter(RID rid, LPhysicsBodyPropertyType type, const Variant& value);
    LIPhysicsBody* get_physics_body(RID p_rid);
};


#endif //LILYPHYS_LILYPHYS_SERVER_H
