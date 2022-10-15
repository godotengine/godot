#include "register_types.h"
#include "core/class_db.h"
#include "core/engine.h"
#include "hub.h"

static Hub* HubPtr = NULL;

void register_hub_types(){
	ClassDB::register_class<Hub>();
	HubPtr = memnew(Hub);
	Engine::get_singleton()->add_singleton(Engine::Singleton("Hub", Hub::get_singleton()));
}

void unregister_hub_types(){
	memdelete(HubPtr);
}
