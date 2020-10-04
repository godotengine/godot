#ifndef BALL_H
#define BALL_H

#include <Godot.hpp>
#include <KinematicBody.hpp>
#include <KinematicCollision.hpp>
#include <cstdlib>

namespace godot {

class Ball : public KinematicBody {
	GODOT_CLASS(Ball, KinematicBody);
	Vector3 direction;
	int speed;

public:
	static void _register_methods();
	void _init();
	void _ready();
	void _process(float delta);
};

} 

#endif
