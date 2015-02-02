#include "bt_utils.h"
#include "bt_node.h"

using namespace BehaviorTree;

static void create_structure_impl(BtStructure& structure_data, NodeList& node_list, BtNode& node, int& index) {
	BT_ASSERT(index < INDEX_TYPE_MAX);
	node_list.push_back(node.get_behavior_node());
	NodeData node_data;
	node_data.begin = index++;
	structure_data.push_back(node_data);
	NodeData& current_node_data = structure_data.back();
	int children_count = node.get_child_count();
	for (int i = 0; i < children_count; ++i) {
		BtNode* p_bt_node = node.get_child(i)->cast_to<BtNode>();
		if (p_bt_node)
			create_structure_impl(structure_data, node_list, *p_bt_node, index);
	}
	current_node_data.end = index;
}

void create_bt_structure(BtStructure& structure_data, NodeList& node_list, BtNode& node, int begin) {
	structure_data.clear();
	node_list.clear();
	create_structure_impl(structure_data, node_list, node, begin);
}
