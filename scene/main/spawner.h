#ifndef SPAWNER_H
#define SPAWNER_H

#include "scene/main/timer.h"

class Spawner : public Timer {
	GDCLASS(Spawner, Timer);

	Ref<PackedScene> spawn_scene;
	Node * last_spawned;

protected:
	void _notification(int p_what);
	static void _bind_methods();
	
public:
	Spawner();
	void set_spawn_scene(const Ref<PackedScene> &scene);
	Ref<PackedScene> get_spawn_scene() const;
	Node *get_last_spawned() const;
	Node *spawn();
};

#endif
