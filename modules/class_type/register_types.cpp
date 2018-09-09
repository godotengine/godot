/* register_types.cpp */

#include "register_types.h"

#include "class_type.h"
#include "core/class_db.h"

void register_class_type_types() {

	ClassDB::register_class<ClassType>();
}

void unregister_class_type_types() {
}
