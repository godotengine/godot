#include "bt_node.h"

void BtNode::move_child_notify(Node *p_child, int ) {
	BtNode* p_btnode = p_child->cast_to<BtNode>();
	ERR_EXPLAIN("Child node is not a BtNode.");
	ERR_FAIL_NULL(p_btnode);

	if (p_btnode) {
		Vector<BehaviorTree::Node*> node_hierarchy;
		move_child_node(*p_btnode, node_hierarchy);
	}
}

void BtNode::add_child_notify(Node *p_child) {
	BtNode* p_btnode = p_child->cast_to<BtNode>();
	ERR_EXPLAIN("Child node is not a BtNode.");
	ERR_FAIL_NULL(p_btnode);
	if (p_btnode) {
		Vector<BehaviorTree::Node*> node_hierarchy;
		add_child_node(*p_btnode, node_hierarchy);
	}
}

void BtNode::remove_child_notify(Node *p_child) {
	BtNode* p_btnode = p_child->cast_to<BtNode>();
	ERR_EXPLAIN("Child node is not a BtNode.");
	ERR_FAIL_NULL(p_btnode);
	if (p_btnode) {
		Vector<BehaviorTree::Node*> node_hierarchy;
		node_hierarchy.push_back(p_btnode->get_behavior_node());
		remove_child_node(*p_btnode, node_hierarchy);
	}
}

void BtNode::_bind_methods() {
	ObjectTypeDB::bind_integer_constant( get_type_static() , "BH_SUCCESS", BehaviorTree::BH_SUCCESS);
	ObjectTypeDB::bind_integer_constant( get_type_static() , "BH_FAILURE", BehaviorTree::BH_FAILURE);
	ObjectTypeDB::bind_integer_constant( get_type_static() , "BH_RUNNING", BehaviorTree::BH_RUNNING);
	ObjectTypeDB::bind_integer_constant( get_type_static() , "BH_ERROR", BehaviorTree::BH_ERROR);
}
