// flecs_World.h
#ifndef FLECS_WORLD_H
#define FLECS_WORLD_H

#include "scene/main/node.h"
#include "thirdparty/flecs.h"

class FlecsWorld : public Node {
    GDCLASS(FlecsWorld, Node);

private:
    static FlecsWorld* singleton;
    flecs::world world;
	Vector<class FlecsSingleton *> singletons;

protected:
    static void _bind_methods();
	void _process(double delta); // Override directly

	void _notification(int p_what);

public:
    FlecsWorld();
    ~FlecsWorld();

    static FlecsWorld* get_singleton() { return singleton; }

    void start_world();
	void register_singletons();
    void stop_world();
	void progress_world(double delta) const;
    flecs::world get_world() { return world; }
};

#endif // FLECS_WORLD_H
