#ifndef BT_ACTION_NODE_H
#define BT_ACTION_NODE_H

#include "bt_node.h"
#include "bt_behavior_delegate.h"

class BtActionNode : public BtNode
{
    OBJ_TYPE(BtActionNode, BtNode);

    virtual void add_child_node(BtNode& child, Vector<BehaviorTree::Node*>& node_hierarchy) override;
    virtual void remove_child_node(BtNode& child, Vector<BehaviorTree::Node*>& node_hierarchy) override;
    virtual void move_child_node(BtNode& child, Vector<BehaviorTree::Node*>& node_hierarchy) override;

    struct Delegate : public BehaviorDelegate<BehaviorTree::Action>
    {
        typedef BehaviorDelegate<BehaviorTree::Action> super;

        Delegate(BtActionNode& node_):super(node_) {}

        virtual void restore_running(BehaviorTree::VirtualMachine& , BehaviorTree::IndexType index, void* context) override;
        virtual void prepare(BehaviorTree::VirtualMachine& , BehaviorTree::IndexType index, void* context) override;
        virtual BehaviorTree::E_State update(BehaviorTree::IndexType, void*) override;
        virtual void abort(BehaviorTree::VirtualMachine& , BehaviorTree::IndexType, void* ) override;
    };
    Delegate delegate;

    BehaviorTree::Node* get_behavior_node() { return &delegate; }

public:
    BtActionNode();

protected:
    static void _bind_methods();
};


#endif
