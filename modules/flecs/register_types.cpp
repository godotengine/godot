// register_types.cpp
#include "register_types.h"

#include "flecs_singleton.h"
#include "core/object/class_db.h"
#include "flecs_world.h"

static FlecsWorld* _flecs_world = nullptr;

void initialize_flecs_module(ModuleInitializationLevel p_level) {

	ClassDB::register_class<FlecsSingleton>();
    ClassDB::register_class<FlecsWorld>();
}

void uninitialize_flecs_module(ModuleInitializationLevel p_level) {

}
