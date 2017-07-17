/* register_types.cpp */

#include "register_types.h"
#include "class_db.h"
#include "priority_queue.h"

void register_priority_queue_types() {
	ClassDB::register_class<PriorityQueue>();
}

void unregister_priority_queue_types() {
	//nothing to do here
}