#include "register_types.h"
#include "core/class_db.h"
// #include "core/engine.h"
#include "serializablecpp.h"

void register_serializablecpp_types(){
	ClassDB::register_class<SerializableCPP>();
}

void unregister_serializablecpp_types(){
	return;
}
