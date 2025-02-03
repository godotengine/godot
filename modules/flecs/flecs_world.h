// flecs_World.h
#ifndef FLECS_WORLD_H
#define FLECS_WORLD_H

#include "core/object/object.h"
#include "core/object/ref_counted.h"
#include "scene/3d/node_3d.h"
#include "scene/main/node.h" // Change to inherit from Node
#include "thirdparty/flecs.h"

class FlecsWorld : public Node {  // Inherit from Node instead of Object
    GDCLASS(FlecsWorld, Node);

private:
    static FlecsWorld* singleton;
    flecs::world world;
    bool is_world_active{false};
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
    bool is_active() const { return is_world_active; }
};

#endif // FLECS_WORLD_H
