#include "bt_composite_node.h"

void BtCompositeNode::add_child_node(BtNode& child, Vector<BehaviorTree::Node*>& node_hierarchy) {
    BtNode* p_parent = get_parent() ? get_parent()->cast_to<BtNode>() : NULL;
	ERR_EXPLAIN("Parent node is not a BtNode.");
	ERR_FAIL_NULL(p_parent);
    if (p_parent) {
        node_hierarchy.push_back(get_behavior_node());
        p_parent->add_child_node(child, node_hierarchy);
    }
}

void BtCompositeNode::remove_child_node(BtNode& child, Vector<BehaviorTree::Node*>& node_hierarchy) {
    BtNode* p_parent = get_parent() ? get_parent()->cast_to<BtNode>() : NULL;
	//ERR_EXPLAIN("Parent node is not a BtNode.");
	//ERR_FAIL_NULL(p_parent);
    if (p_parent) {
        node_hierarchy.push_back(get_behavior_node());
        p_parent->remove_child_node(child, node_hierarchy);
    }
}

void BtCompositeNode::move_child_node(BtNode& child, Vector<BehaviorTree::Node*>& node_hierarchy) {
    BtNode* p_parent = get_parent() ? get_parent()->cast_to<BtNode>() : NULL;
	//ERR_EXPLAIN("Parent node is not a BtNode.");
	//ERR_FAIL_NULL(p_parent);
    if (p_parent) {
        node_hierarchy.push_back(get_behavior_node());
        p_parent->move_child_node(child, node_hierarchy);
    }
}

