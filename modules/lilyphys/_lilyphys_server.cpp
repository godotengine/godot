//
// Created by amara on 19/10/2021.
//

#include "_lilyphys_server.h"



_LilyphysServer *_LilyphysServer::singleton = nullptr;
_LilyphysServer *_LilyphysServer::get_singleton() {
    return singleton;
}

void _LilyphysServer::_bind_methods() {
    ClassDB::bind_method(D_METHOD("get_pee_storage"), &_LilyphysServer::get_pee_storage);
    ClassDB::bind_method(D_METHOD("create_physics_body"), &_LilyphysServer::create_physics_body);
    ClassDB::bind_method(D_METHOD("get_body_collisions", "rid"), &_LilyphysServer::get_body_collisions);
    ClassDB::bind_method(D_METHOD("get_step_through"), &_LilyphysServer::get_step_through);
    ClassDB::bind_method(D_METHOD("set_step_through", "p_step_through"), &_LilyphysServer::set_step_through);
    ClassDB::bind_method(D_METHOD("do_step_through"), &_LilyphysServer::do_step_through);
}

_LilyphysServer::_LilyphysServer() {
    singleton = this;
}

_LilyphysServer::~_LilyphysServer() {

}

String _LilyphysServer::get_pee_storage() {
    return "the balls.";
}

RID _LilyphysServer::create_physics_body() {
    return LilyphysServer::get_singleton()->create_physics_body(false);
}

Array _LilyphysServer::get_body_collisions(RID rid) {
    return LilyphysServer::get_singleton()->get_body_collisions(rid);
}

bool _LilyphysServer::get_step_through() {
    return LilyphysServer::get_singleton()->get_step_through();
}

void _LilyphysServer::set_step_through(bool p_step_through) {
    LilyphysServer::get_singleton()->set_step_through(p_step_through);
}

void _LilyphysServer::do_step_through() {
    LilyphysServer::get_singleton()->do_step_through();
}
