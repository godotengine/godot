/* register_types.cpp */

#include "register_types.h"
#include "class_db.h"
#include "sumator.h"

void register_sumator_types() {

	ClassDB::register_class<Sumator>();
}

void unregister_sumator_types() {
	//nothing to do here
}