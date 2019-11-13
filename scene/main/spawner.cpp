#include "spawner.h"
#include "core/engine.h"
#include "scene/main/viewport.h"
#include "scene/resources/packed_scene.h"

Spawner::Spawner() {
	last_spawned = nullptr;
	spawn_quantity = 5;
	endless_spawn = true;
	stop_timer_on_cant_spawn = true;
}

void Spawner::set_spawn_scene(const Ref<PackedScene> &scene) {
	spawn_scene = scene;
}
Ref<PackedScene> Spawner::get_spawn_scene() const {
	ERR_FAIL_COND_V_MSG(!spawn_scene.is_valid(), Ref<PackedScene>(), "No spawn Scene set!");
	return spawn_scene;
}

Node *Spawner::get_last_spawned() const {
	ERR_FAIL_COND_V_MSG(last_spawned == nullptr, nullptr, "No last spawned Node!");
	return last_spawned;
}

Node *Spawner::_spawn() {
	Node *new_node = spawn_scene->instance();
	add_child(new_node);
	emit_signal("spawned", new_node);
	last_spawned = new_node;
	return new_node;
}

Node *Spawner::spawn() {
	if (spawn_scene.is_valid()) {
		if (endless_spawn) {
			return _spawn();
		}
		if (spawn_quantity > 0) {
			spawn_quantity--;
			if (spawn_quantity == 0) {
				if (stop_timer_on_cant_spawn)
					stop();
				emit_signal("spawned_all");
			}
			return _spawn();
		}
	}
	if (stop_timer_on_cant_spawn)
		stop();
	return nullptr;
}

Node *Spawner::spawn_bypass() {
	ERR_FAIL_COND_V_MSG(!spawn_scene.is_valid(), nullptr, "No spawn Scene set!");
	return _spawn();
}

void Spawner::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			if (Engine::get_singleton()->is_editor_hint() == false)
				connect("timeout", this, "spawn");
		} break;
		default:;
	}
}

void Spawner::set_quantity_spawn(const int quantity) {
	spawn_quantity = quantity;
}

int Spawner::get_quantity_spawn() const {
	return spawn_quantity;
}

bool Spawner::is_endless_spawn() const {
	return endless_spawn;
}

void Spawner::set_endless_spawn(const bool enable) {
	endless_spawn = enable;
}

bool Spawner::is_stop_timer_on_cant_spawn() const {
	return stop_timer_on_cant_spawn;
}

void Spawner::set_stop_timer_on_cant_spawn(const bool enable) {
	stop_timer_on_cant_spawn = enable;
}

void Spawner::_bind_methods() {
	//scene
	ClassDB::bind_method(D_METHOD("set_spawn_scene", "scene_to_load"), &Spawner::set_spawn_scene);
	ClassDB::bind_method(D_METHOD("get_spawn_scene"), &Spawner::get_spawn_scene);
	//quantity
	ClassDB::bind_method(D_METHOD("set_quantity_spawn", "quantity"), &Spawner::set_quantity_spawn);
	ClassDB::bind_method(D_METHOD("get_quantity_spawn"), &Spawner::get_quantity_spawn);
	//endless
	ClassDB::bind_method(D_METHOD("set_endless_spawn", "enable"), &Spawner::set_endless_spawn);
	ClassDB::bind_method(D_METHOD("is_endless_spawn"), &Spawner::is_endless_spawn);
	//endless
	ClassDB::bind_method(D_METHOD("set_stop_timer_on_cant_spawn", "enable"), &Spawner::set_stop_timer_on_cant_spawn);
	ClassDB::bind_method(D_METHOD("is_stop_timer_on_cant_spawn"), &Spawner::is_stop_timer_on_cant_spawn);
	//other
	ClassDB::bind_method(D_METHOD("spawn"), &Spawner::spawn);
	ClassDB::bind_method(D_METHOD("spawn_bypass"), &Spawner::spawn_bypass);
	ClassDB::bind_method(D_METHOD("get_last_spawned"), &Spawner::get_last_spawned);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "spawn_scene", PROPERTY_HINT_RESOURCE_TYPE, "PackedScene"), "set_spawn_scene", "get_spawn_scene");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "quantity_spawn"), "set_quantity_spawn", "get_quantity_spawn");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "endless_spawn"), "set_endless_spawn", "is_endless_spawn");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "stop_timer_on_cant_spawn"), "set_stop_timer_on_cant_spawn", "is_stop_timer_on_cant_spawn");
	ADD_SIGNAL(MethodInfo("spawned", PropertyInfo(Variant::OBJECT, "node", PROPERTY_HINT_RESOURCE_TYPE, "Node")));
	ADD_SIGNAL(MethodInfo("spawned_all"));
}
