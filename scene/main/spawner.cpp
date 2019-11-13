#include "spawner.h"
#include "core/engine.h"
#include "scene/main/viewport.h"
#include "scene/resources/packed_scene.h"

Spawner::Spawner()
{
	last_spawned = nullptr;
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

Node *Spawner::spawn() {
	ERR_FAIL_COND_V_MSG(!spawn_scene.is_valid(), nullptr, "No spawn Scene set!");
	Node *new_node = spawn_scene->instance();
	add_child(new_node);
	emit_signal("spawned", new_node);
	last_spawned = new_node;
	return new_node;
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

void Spawner::_bind_methods() {
	//getset
	ClassDB::bind_method(D_METHOD("set_spawn_scene", "scene_to_load"), &Spawner::set_spawn_scene);
	ClassDB::bind_method(D_METHOD("get_spawn_scene"), &Spawner::get_spawn_scene);
	//
	ClassDB::bind_method(D_METHOD("spawn"), &Spawner::spawn);
	ClassDB::bind_method(D_METHOD("get_last_spawned"), &Spawner::get_last_spawned);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "spawn_scene", PROPERTY_HINT_RESOURCE_TYPE, "PackedScene"), "set_spawn_scene", "get_spawn_scene");
	ADD_SIGNAL(MethodInfo("spawned", PropertyInfo(Variant::OBJECT, "node", PROPERTY_HINT_RESOURCE_TYPE, "Node")));
}
