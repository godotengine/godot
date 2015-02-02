#ifndef BT_COMPOSITE_NODE_H
#define BT_COMPOSITE_NODE_H

#include "bt_node.h"
#include "composite.h"

class BtCompositeNode : public BtNode
{
    OBJ_TYPE(BtCompositeNode, BtNode);

public:
    virtual void add_child_node(BtNode& child, Vector<BehaviorTree::Node*>& node_hierarchy) override;
    virtual void remove_child_node(BtNode& child, Vector<BehaviorTree::Node*>& node_hierarchy) override;
    virtual void move_child_node(BtNode& child, Vector<BehaviorTree::Node*>& node_hierarchy) override;
};

class BtSequenceNode : public BtCompositeNode
{
    OBJ_TYPE(BtSequenceNode, BtCompositeNode);
    BehaviorTree::Sequence behavior_node;
    virtual BehaviorTree::Node* get_behavior_node() override { return &behavior_node; }
};

class BtSelectorNode : public BtCompositeNode
{
    OBJ_TYPE(BtSelectorNode, BtCompositeNode);
    BehaviorTree::Selector behavior_node;
    virtual BehaviorTree::Node* get_behavior_node() override { return &behavior_node; }
};

class BtParallelNode : public BtCompositeNode
{
    OBJ_TYPE(BtParallelNode, BtCompositeNode);
    BehaviorTree::Parallel<BehaviorTree::BH_SUCCESS> behavior_node;
    virtual BehaviorTree::Node* get_behavior_node() override { return &behavior_node; }
};

#endif
