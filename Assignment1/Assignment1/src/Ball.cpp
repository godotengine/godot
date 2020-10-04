#include "Ball.h"
using namespace godot;

void Ball::_register_methods() {
	register_method("_init", &Ball::_init);
	register_method("_ready", &Ball::_ready);
	register_method("_process", &Ball::_process);
	register_property("direction", &Ball::direction, Vector3(1, 0, 0));
	register_property("speed", &Ball::speed, 5);
}
void Ball::_init() {
}

void Ball::_ready() {
	int x = rand() % 140 - 70;
	int y = rand() % 140 - 70;
	int z = rand() % 140 - 70;
	direction = Vector3(x, y, z).normalized();
	speed = rand() % 8 + 7;
}

void Ball::_process(float delta) {
	Ref<KinematicCollision> results = move_and_collide(direction * speed * delta);
	if (results.is_valid()) {

		Vector3 normal = results->get_normal();
		direction = direction - 2 * normal * (normal.dot(direction));

	}
}
