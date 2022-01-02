//
// Created by amara on 26/11/2021.
//

#include "l_collision_solver.h"

#include "internal/li_collision_object.h"
#include "lilyphys_server.h"

#include "thirdparty/gjk_epa.h"
#include "thirdparty/libccd/ccd/ccd.h"
#include "thirdparty/libccd/ccd/vec3.h"

void LCollision::_bind_methods() {
    ClassDB::bind_method(D_METHOD("get_direction"), &LCollision::get_direction);
    ClassDB::bind_method(D_METHOD("get_position"), &LCollision::get_position);
    ClassDB::bind_method(D_METHOD("get_depth"), &LCollision::get_depth);
    ClassDB::bind_method(D_METHOD("get_body0"), &LCollision::get_body_0);
    ClassDB::bind_method(D_METHOD("get_body1"), &LCollision::get_body_1);
    ClassDB::bind_method(D_METHOD("get_shape_transform"), &LCollision::get_shape_transform);
}

struct CheckData {
    LICollisionObject *object;
    LICollisionObject::ShapeData* shape;
};

// Support function for libccd to use.
void support(const void *obj, const ccd_vec3_t *dir, ccd_vec3_t *vec) {
    auto object = (CheckData*)obj;

    //print_line("------");
    Vector3 point = LilyphysServer::get_singleton()->shape_get_support(object->shape->shape, Vector3{dir->v[0], dir->v[1], dir->v[2]});
    point = object->shape->transform.xform(point);
    point = object->object->get_transform().xform(point);
    ccdVec3Set(vec, point.x, point.y, point.z);
}

// For each shape object 1 has...
List<RID> LCollisionSolver::check_collision(LICollisionObject *object1, LICollisionObject *object2, RID_Owner<CollisionResult>& p_owner) {
    List<RID> results{};
    for (Map<size_t, LICollisionObject::ShapeData>::Element *E = object1->get_shapes().front(); E; E = E->next()) {
        // Check if it isn't disabled...
        if (E->get().disabled) {
            continue;
        }
        // And now for each shape object 2 has.... (aaaaa bad................)
        for (Map<size_t, LICollisionObject::ShapeData>::Element *F = object2->get_shapes().front(); F; F = F->next()) {
            // Check if it isn't disabled...
            if (F->get().disabled) {
                continue;
            }
            // Let's do the check.
            CheckData data1{object1, &E->get()};
            CheckData data2{object2, &F->get()};
            ccd_real_t depth;
            ccd_vec3_t dir, pos;
            int intersect = ccdGJKPenetration((const void*)&data1, (const void*)&data2, &ccd, &depth, &dir, &pos);
            if (intersect != -1) {
                CollisionResult* result = memnew(CollisionResult(object1->get_self(), object2->get_self(), E->get().transform, true, depth, Vector3{dir.v[0], dir.v[1], dir.v[2]}, Vector3{pos.v[0], pos.v[1], pos.v[2]}));
                RID id = p_owner.make_rid(result);
                result->rid = id;
                results.push_back(id);
            }
        }
    }
    return results;
}

LCollisionSolver::LCollisionSolver() {
    CCD_INIT(&ccd);
    ccd.support1 = support;
    ccd.support2 = support;
    ccd.max_iterations = 100;
    ccd.epa_tolerance = 0.0001;
}