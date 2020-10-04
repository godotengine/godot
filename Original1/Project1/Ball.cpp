#include "Ball.h"
using namespace godot;



void Ball::_register_methods() {
	register_method("_init", &Ball::_init);
	register_method("_ready", &Ball::_ready);
	register_method("_process", &Ball::_process);
	register_method("_hit", &Ball::_hit);
	register_property("speed", &Ball::speed, 5);
	register_property("direction", &Ball::direction, Vector3(1, 0, 0));

}
void Ball::_init() {   
}

void Ball::_ready() {   
    int x = rand()%100 - 50;
    int y = rand()%100 - 50;
    int z = rand()%100 - 50;
    direction = Vector3(x, y, z).normalized();
    speed = rand()%10 + 3;
}

void Ball::_process(float delta) {   
    Ref<KinematicCollision> results = move_and_collide(direction*speed*delta);
    if(results.is_valid()){
        _hit(results->get_normal());
    }
}

void Ball::_hit(Vector3 normal) {   
    //R = I - 2N(N·I)
    direction = direction - 2 * normal * (normal.dot(direction));
}
