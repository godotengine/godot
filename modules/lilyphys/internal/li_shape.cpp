//
// Created by amara on 25/11/2021.
//

#include "li_shape.h"

#include "core/variant.h"

void LIBoxShape::set_data(const Variant &p_data) {
    set_half_extends(p_data);
}

Variant LIBoxShape::get_data() const {
    return half_extents;
}

Vector3 LIBoxShape::get_support(Vector3 p_direction) const {
    Vector3 point(
            (p_direction.x < 0) ? -half_extents.x : half_extents.x,
            (p_direction.y < 0) ? -half_extents.y : half_extents.y,
            (p_direction.z < 0) ? -half_extents.z : half_extents.z);

    return point;
}

void LIShape::add_owner(LICollisionObject *p_object) {
    Map<LICollisionObject*, int>::Element *E = owners.find(p_object);
    if (E) {
        E->get()++;
    } else {
        owners[p_object] = 1;
    }
}

void LIShape::remove_owner(LICollisionObject *p_object) {
    Map<LICollisionObject*, int>::Element *E = owners.find(p_object);
    ERR_FAIL_COND(!E);
    E->get()--;
    if (E->get() == 0) {
        owners.erase(E);
    }
}

bool LIShape::is_owner(LICollisionObject *p_object) const {
    return owners.has(p_object);
}

const Map<LICollisionObject *, int> &LIShape::get_owners() const {
    return owners;
}
