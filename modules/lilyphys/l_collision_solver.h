//
// Created by amara on 26/11/2021.
//

#ifndef LILYPHYS_L_COLLISION_SOLVER_H
#define LILYPHYS_L_COLLISION_SOLVER_H

#include "core/math/vector3.h"
#include "core/set.h"
#include "core/list.h"
#include "core/rid.h"
#include "core/math/transform.h"
#include "core/object.h"
#include "core/class_db.h"

#include "thirdparty/libccd/ccd/ccd.h"

class LICollisionObject;

class CollisionResult : public RID_Data {
public:
    RID rid;
    RID body0;
    RID body1;
    Transform shape_transform;
    bool intersect = false;
    real_t depth = 0.0f;
    Vector3 dir = {0.0f, 0.0f, 0.0f};
    Vector3 pos = {0.0f, 0.0f, 0.0f};
    // Have we been processed, is one of our bodies NOT affected by another constraint or collision?
    bool satisfied = false;
    // TODO: Material pair properties
    real_t denominator = 0.0f; // Used to cache some info
    Vector3 R0; // Shape position relative to body0
    Vector3 R1; // Shape position relative to body1
    real_t min_separation_velocity = 0.0f;
    // TODO: Make restitution configurable.
    real_t restitution = 0.09f;
    CollisionResult(RID p_body0, RID p_body1, Transform p_shape_transform, bool p_intersect, real_t p_depth,
                    Vector3 p_dir, Vector3 p_pos) : body0(p_body0), body1(p_body1), shape_transform(p_shape_transform),
                    intersect(p_intersect), depth(p_depth), dir(p_dir), pos(p_pos) {}
};

class LCollision : public Object {
    GDCLASS(LCollision, Object);
private:
    Vector3 direction;
    Vector3 position;
    real_t depth;
    RID body0;
    RID body1;
    Transform shape_transform;
protected:
    static void _bind_methods();
public:
    const Vector3 &get_direction() const {
        return direction;
    }

    const Vector3 &get_position() const {
        return position;
    }

    real_t get_depth() const {
        return depth;
    }

    const RID &get_body_0() const {
        return body0;
    }

    const RID &get_body_1() const {
        return body1;
    }

    const Transform &get_shape_transform() const {
        return shape_transform;
    }

    void init(const Vector3 &p_direction, const Vector3 &p_position, real_t p_depth, const RID &p_body_0,
                           const RID &p_body_1, const Transform &p_shape_transform) {
        direction = p_direction;
        position = p_position;
        depth = p_depth;
        body0 = p_body_0;
        body1 = p_body_1;
        shape_transform = p_shape_transform;
    }
};

class LCollisionSolver {
private:
    ccd_t ccd;

public:
    List<RID> check_collision(LICollisionObject* object1, LICollisionObject* object2, RID_Owner<CollisionResult>& p_owner);
    LCollisionSolver();
};


#endif //LILYPHYS_L_COLLISION_SOLVER_H
