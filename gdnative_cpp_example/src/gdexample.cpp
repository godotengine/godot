#include "gdexample.h"

using namespace godot;

void GDExample::_register_methods() {
	register_method("_process", &GDExample::_process);
	register_property<GDExample, float>("amplitude", &GDExample::amplitude, 10.0);
}

GDExample::GDExample() {
}

GDExample::~GDExample() {
    // add your cleanup here
}

void GDExample::_init() {
	// initialize any variables here
	time_passed = 0.0;
	amplitude = 10.0;
}

void GDExample::_process(float delta) {
	time_passed += delta;

	Vector2 new_position = Vector2(
			amplitude + (amplitude * sin(time_passed * 2.0)),
			amplitude + (amplitude * cos(time_passed * 1.5)));

	set_position(new_position);
}
