/* register_types.cpp */

#include "register_types.h"
#include "class_db.h"
#include "test_node.h"

void register_test_node_types() {
	ClassDB::register_class<TestNode>();
	ClassDB::register_class<BunnyPosition>();

	ClassDB::register_class<Bunny>();
	ClassDB::register_class<BunnyController>();
}

void unregister_test_node_types() {
	//nothing to do here
}