#ifndef SPAWNER_H
#define SPAWNER_H

#include "scene/main/timer.h"

class Spawner : public Timer {
GDCLASS(Spawner, Timer);

	Ref<PackedScene> spawn_scene;
	Node *last_spawned;

	bool endless_spawn;
	bool stop_timer_on_cant_spawn;
	int spawn_quantity;

	Node *_spawn();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	Spawner();
	//setget
	void set_spawn_scene(const Ref<PackedScene> &scene);
	Ref<PackedScene> get_spawn_scene() const;
	void set_quantity_spawn(int quantity);
	int get_quantity_spawn() const;
	bool is_endless_spawn() const;
	void set_endless_spawn(bool enable);
	bool is_stop_timer_on_cant_spawn() const;
	void set_stop_timer_on_cant_spawn(bool enable);

	//other
	Node *get_last_spawned() const;
	Node *spawn();
	Node *spawn_bypass();
};

#endif
