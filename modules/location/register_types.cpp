/* register_types.cpp */

#include "register_types.h"

#include "core/class_db.h"
#include "location_manager.h"
#include "location_result.h"
#include "location_param.h"
#include "core/engine.h"

void register_location_types() {
    ClassDB::register_class<LocationManager>();
    ClassDB::register_class<LocationResult>();
    ClassDB::register_class<LocationParam>();
    Engine::get_singleton()->add_singleton(Engine::Singleton("LocationManager", LocationManager::get_singleton()));
}

void unregister_location_types() {
   // Nothing to do here in this example.
}