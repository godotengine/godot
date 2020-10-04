#ifndef BALL_H
#define BALL_H


#include <Godot.hpp>
#include <KinematicBody.hpp>
#include <KinematicCollision.hpp>
#include <cstdlib>

namespace godot {

class Ball : public KinematicBody
{
	GODOT_CLASS(Ball, KinematicBody);
    int speed;
    Vector3 direction;

public:
    static void _register_methods();
    void _init();
    //place balls / walls(?)
    void _ready();
    //this is supposed to update meshes based on movement defined later
    void _process(float delta); //usure on float delta

    void _hit(Vector3 normal); //usure on float delta
};

}


#endif
