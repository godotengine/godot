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

#define FLOAT_TINY ((float) 1.0e-6)

enum LShapeType {
    SHAPE_BOX,
    SHAPE_SPHERE
};

class LilyphysServer : public Object {
    GDCLASS(LilyphysServer, Object);
    static LilyphysServer *singleton;
    Set<RID> deletion_queue;
    RID_Owner<LIPhysicsBody> body_owner;
    List<RID> bodies;
    RID_Owner<LIShape> shape_owner;
    Set<RID> shapes;
    RID_Owner<LIForceGenerator> generator_owner;
    Set<RID> generators;
    Set<RID> collision_results;
    RID_Owner<CollisionResult> collision_result_owner;
    bool active = true;
    struct Registration {
        RID body;
        RID generator;
    };
    List<Registration> registry;
    RID gravity;
    RID spring;
    LCollisionSolver solver{};
    Set<RID> active_bodies;
    List<RID> collisions;
    bool step_through = false;
    bool do_step = false;
    int collision_iterations = 4;
    // Number of timesteps to resolve penetration over.
    int num_penetration_relaxation_timesteps = 10;
    float allowed_penetration = 0.01f;
    // Vars to limit velocity during collision resolution.
    float max_velocity_magnitude = 100.0f;
    float min_velocity_for_processing = 0.001f;
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
    void queue_free_rid(RID p_rid);
    void set_integration_callback(RID p_body, Object *p_receiver, const StringName& p_method, const Variant& p_user_data = Variant());
    void register_generator(RID p_body, RID p_generator);
    void unregister_generator(RID p_body, RID p_generator);
    void clear_registry_for_body(RID p_body);
    void clear_registry();
    void set_physics_body_parameter(RID rid, LPhysicsBodyPropertyType type, const Variant& value);
    LIPhysicsBody* get_physics_body(RID p_rid);
    RID create_shape(LShapeType p_type);
    void shape_set_data(RID p_id, const Variant &p_data);
    Variant shape_get_data(RID p_id);
    Vector3 shape_get_support(RID p_id, Vector3 p_direction);
    Basis shape_get_inertia_tensor(RID p_id, real_t p_mass);
    void shape_add_owner(RID p_shape, LICollisionObject* owner);
    void shape_remove_owner(RID p_shape, LICollisionObject* owner);
    size_t physics_body_add_shape(RID p_body, RID p_shape);
    void physics_body_remove_shape(RID p_body, size_t p_id);
    void physics_body_shape_set_disabled(RID p_body, size_t p_id, bool p_disabled);
    void physics_body_shape_set_transform(RID p_body, size_t p_id, const Transform& p_transform);
    bool physics_body_get_shape_exists(RID p_body, size_t p_id);
    void set_collision_satisfied(RID p_collision, bool p_satisfied);
    Array get_body_collisions(RID p_body);
    void set_step_through(bool p_step_through) { step_through = p_step_through; }
    bool get_step_through() { return step_through; }
    void do_step_through() { do_step = true; }
private:
    void free_queue();
    void find_all_active_bodies();
    void copy_all_current_state_to_old();
    void restore_all_state();
    void clear_all_accumulators();
    void detect_all_collisions(float p_step);
    void handle_all_constraints(float p_step, int p_iterations, bool p_force_inelastic);
    void integrate_all_bodies(float p_step);
    void try_freeze_all_bodies(real_t p_step);
    void try_activate_all_frozen_bodies();
    void damp_all_for_deactivation();
    void activate_body(RID p_body);
    void update_all_positions(float p_step);
    void perform_all_callbacks();
    void preprocess_collision(float p_step, CollisionResult* p_result);
    bool process_collision(float p_step, CollisionResult* p_result, bool p_first_contact);
    void detect_collision(RID p_body, RID p_other_body);
    void detect_collisions_for_body(RID p_body);
    void process_shock_step(CollisionResult *p_result, real_t p_step);
    void do_shock_step(real_t p_step);
};


#endif //LILYPHYS_LILYPHYS_SERVER_H
