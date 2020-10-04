#ifndef GDEXAMPLE_H
#define GDEXAMPLE_H

#include <Godot.hpp>
#include <Sprite.hpp>

namespace godot {

    class GDExample : public Sprite {
        GODOT_CLASS(GDExample, Sprite)

    private:
        float time_passed;

	private:
		float time_passed;
		float amplitude;

    public:
        static void _register_methods();

        GDExample();
        ~GDExample();

        void _init(); // our initializer called by Godot

        void _process(float delta);
    };

}

#endif
