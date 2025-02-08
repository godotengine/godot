// flecs_World.cpp
#include "flecs_world.h"
#include "core/object/class_db.h"
#include "flecs_singleton.h"
FlecsWorld *FlecsWorld::singleton = nullptr;
void FlecsWorld::_bind_methods() {
	ClassDB::bind_method(D_METHOD("start_world"), &FlecsWorld::start_world);
	ClassDB::bind_method(D_METHOD("stop_world"), &FlecsWorld::stop_world);
}

FlecsWorld::FlecsWorld() {
	singleton = this;
}

FlecsWorld::~FlecsWorld() {
	stop_world();
	singleton = nullptr;
}


void FlecsWorld::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			// Start world when entering tree
			start_world();
			set_physics_process(true);
		} break;

		case NOTIFICATION_EXIT_TREE: {
			// Stop world when exiting tree
			stop_world();
			set_physics_process(false);
		} break;

		case NOTIFICATION_PHYSICS_PROCESS: {
			progress_world(get_physics_process_delta_time());
		} break;
	}
}

void FlecsWorld::start_world() {
	world = flecs::world();
	world.import <flecs::stats>();
	world.set<flecs::Rest>({});

	const int ThreadCount = OS::get_singleton()->get_processor_count();
	world.set_task_threads(ThreadCount);
	world.set_threads(ThreadCount);

	register_singletons();
}

void FlecsWorld::register_singletons() {
	List<StringName> classes;
	ClassDB::get_inheriters_from_class("FlecsSingleton", &classes);

	for (const StringName &class_name : classes) {
		Object *obj = ClassDB::instantiate(class_name);
		FlecsSingleton *new_singleton = Object::cast_to<FlecsSingleton>(obj);
		if (new_singleton) {
			new_singleton->_register_singleton(this);
			singletons.push_back(new_singleton);
		}
	}
}

void FlecsWorld::stop_world() {
	for (FlecsSingleton *delete_singleton : singletons) {
		memdelete(delete_singleton);
	}

	singletons.clear();
}

void FlecsWorld::progress_world(double delta) const {
	world.progress(delta);
}
