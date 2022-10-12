#include "register_types.h"
#include "core/class_db.h"
#include "core/engine.h"
#include "godotsteam.h"

static Steam* SteamPtr = NULL;

void register_godotsteam_types(){
	ClassDB::register_class<Steam>();
	SteamPtr = memnew(Steam);
	Engine::get_singleton()->add_singleton(Engine::Singleton("Steam",Steam::get_singleton()));
}

void unregister_godotsteam_types(){
	memdelete(SteamPtr);
}
