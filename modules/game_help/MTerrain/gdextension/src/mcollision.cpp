#include "mcollision.h"





void MCollision::_bind_methods(){
    ClassDB::bind_method(D_METHOD("is_collided"), &MCollision::is_collided);
    ClassDB::bind_method(D_METHOD("get_collision_position"), &MCollision::get_collision_position);
}

bool MCollision::is_collided(){
    return collided;
}
Vector3 MCollision::get_collision_position(){
    return collision_position;
}