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
    return LilyphysServer::get_singleton()->create_physics_body();
}
