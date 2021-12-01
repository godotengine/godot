//
// Created by amara on 19/10/2021.
//

#ifndef LILYPHYS_LILYPHYS_SERVER_H
#define LILYPHYS_LILYPHYS_SERVER_H

#include "_lilyphys_server.h"
#include "l_collision_solver.h"
#include "internal/li_physics_body.h"
#include "internal/li_force_generator.h"
#include "internal/li_gravity.h"
#include "internal/li_shape.h"

#include "core/object.h"
#include "core/rid.h"

enum LShapeType {
    SHAPE_BOX
};

class LilyphysServer : public Object {
    GDCLASS(LilyphysServer, Object);
    static LilyphysServer *singleton;
    RID_Owner<LIPhysicsBody> body_owner;
    Set<RID> bodies;
    RID_Owner<LIShape> shape_owner;
    Set<RID> shapes;
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
    LCollisionSolver solver{};
protected:
    static void _bind_methods();
public:
    static LilyphysServer *get_singleton();
    LilyphysServer();
    void init();
    void finish();
    void set_active(bool p_active) { active = p_active; }
    void step(float p_step);
    void free(RID p_rid);
    RID create_physics_body();
    void set_integration_callback(RID p_body, Object *p_receiver, const StringName& p_method, const Variant& p_user_data = Variant());
    void register_generator(RID p_body, RID p_generator);
    void unregister_generator(RID p_body, RID p_generator);
    void clear_registry();
    void set_physics_body_parameter(RID rid, LPhysicsBodyPropertyType type, const Variant& value);
    LIPhysicsBody* get_physics_body(RID p_rid);
    RID create_shape(LShapeType p_type);
    void shape_set_data(RID p_id, const Variant &p_data);
    Variant shape_get_data(RID p_id);
    Vector3 shape_get_support(RID p_id, Vector3 p_direction);
    void shape_add_owner(RID p_shape, LICollisionObject* owner);
    void shape_remove_owner(RID p_shape, LICollisionObject* owner);
    size_t physics_body_add_shape(RID p_body, RID p_shape);
    void physics_body_remove_shape(RID p_body, size_t p_id);
    void physics_body_shape_set_disabled(RID p_body, size_t p_id, bool p_disabled);
    void physics_body_shape_set_transform(RID p_body, size_t p_id, const Transform& p_transform);
    bool physics_body_get_shape_exists(RID p_body, size_t p_id);
};


#endif //LILYPHYS_LILYPHYS_SERVER_H
