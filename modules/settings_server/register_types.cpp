#include "register_types.h"
#include "core/class_db.h"
#include "core/engine.h"
#include "settings_server.h"

static SettingsServer* SettingsServerPtr = nullptr;

void register_settings_server_types(){
	ClassDB::register_class<SettingsServer>();
	// if (Engine::get_singleton()->is_editor_hint()) return;
	SettingsServerPtr = memnew(SettingsServer);
	Engine::get_singleton()->add_singleton(Engine::Singleton("SettingsServer", SettingsServer::get_singleton()));
}
void unregister_settings_server_types(){
	// if (SettingsServerPtr)
	memdelete(SettingsServerPtr);
}