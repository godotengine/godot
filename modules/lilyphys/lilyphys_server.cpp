//
// Created by amara on 19/10/2021.
//

#include "lilyphys_server.h"

LilyphysServer *LilyphysServer::singleton = nullptr;

void LilyphysServer::_bind_methods() {

}

LilyphysServer *LilyphysServer::get_singleton() {
    return singleton;
}

LilyphysServer::LilyphysServer() {
    singleton = this;
}

void LilyphysServer::init() {
    print_line("LilyphysServer has been created!");
}

void LilyphysServer::finish() {

}

void LilyphysServer::step(float p_step) {
    for (Set<RID>::Element *e = collision_objects.front(); e; e = e->next()) {
        LCollisionObjectInternal *object = collision_object_owner.get(e->get());
        Transform trans = object->get_transform();
        //trans.origin.y += p_step;
        trans.rotate(Vector3(0, 1, 0), 1);
        object->set_transform(trans);
    }
}

RID LilyphysServer::create_collision_object() {
    LCollisionObjectInternal* object = memnew(LCollisionObjectInternal);
    RID rid = collision_object_owner.make_rid(object);
    collision_objects.insert(rid);
    object->set_self(rid);
    return rid;
}

Transform LilyphysServer::get_collision_object_transform(RID p_rid) {
    LCollisionObjectInternal* object = collision_object_owner.get(p_rid);
    ERR_FAIL_COND_V_MSG(!object, Transform(), "Given RID does not correspond with a collision object.");
    return object->get_transform();
}

