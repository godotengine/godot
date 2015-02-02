#include "register_types.h"
#include "object_type_db.h"

#include "core/bt_string_names.h"
#include "core/bt_root_node.h"
#include "core/bt_action_node.h"
#include "core/bt_composite_node.h"
#include "core/bt_custom_parallel_node.h"
#include "core/bt_decorator_node.h"

void register_behavior_tree_types() {

	BtStringNames::create();

	ObjectTypeDB::register_type<BtRootNode>();
	ObjectTypeDB::register_type<BtActionNode>();
	ObjectTypeDB::register_type<BtDecoratorNode>();
	ObjectTypeDB::register_type<BtSequenceNode>();
	ObjectTypeDB::register_type<BtSelectorNode>();
	ObjectTypeDB::register_type<BtParallelNode>();
	ObjectTypeDB::register_type<BtCustomParallelNode>();
}

void unregister_behavior_tree_types() {

	BtStringNames::free();
}

