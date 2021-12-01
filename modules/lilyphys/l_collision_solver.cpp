//
// Created by amara on 26/11/2021.
//

#include "l_collision_solver.h"

#include "internal/li_collision_object.h"
#include "lilyphys_server.h"

#include "thirdparty/gjk_epa.h"
#include "thirdparty/libccd/ccd/ccd.h"
#include "thirdparty/libccd/ccd/vec3.h"

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
    //print_line(object->shape->transform.origin);
    point = object->object->get_transform().xform(point);
    //print_line(object->object->get_transform().origin);
    ccdVec3Set(vec, point.x, point.y, point.z);
}

    // For each shape object 1 has...
CollisionResult LCollisionSolver::check_collision(LICollisionObject *object1, LICollisionObject *object2) {
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
            if (intersect == -1) {
                return CollisionResult{false};
            }
            return CollisionResult{true, depth, Vector3{dir.v[0], dir.v[1], dir.v[2]}, Vector3{pos.v[0], pos.v[1], pos.v[2]}};
        }
    }
    return CollisionResult{false};
}

LCollisionSolver::LCollisionSolver() {
    CCD_INIT(&ccd);
    ccd.support1 = support;
    ccd.support2 = support;
    ccd.max_iterations = 100;
    ccd.epa_tolerance = 0.0001;
}
