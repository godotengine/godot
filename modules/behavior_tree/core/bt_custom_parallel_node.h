#ifndef BT_CUSTOM_PARALLEL_NODE_H
#define BT_CUSTOM_PARALLEL_NODE_H

#include "bt_composite_node.h"
#include "bt_behavior_delegate.h"

class BtCustomParallelNode : public BtCompositeNode
{
	OBJ_TYPE(BtCustomParallelNode, BtCompositeNode);

	struct Delegate : public BehaviorDelegate<BehaviorTree::Composite>
	{
		typedef BehaviorDelegate<BehaviorTree::Composite> super;

		Delegate(BtCustomParallelNode& node_):super(node_) {}

		virtual void restore_running(BehaviorTree::VirtualMachine& , BehaviorTree::IndexType index, void* context, BehaviorTree::VMRunningData& running_data) override;
		virtual void prepare(BehaviorTree::VirtualMachine& , BehaviorTree::IndexType index, void* context, BehaviorTree::VMRunningData& running_data) override;
		virtual BehaviorTree::E_State child_update(BehaviorTree::VirtualMachine& , BehaviorTree::IndexType, void*, BehaviorTree::E_State child_state, BehaviorTree::VMRunningData& running_data) override;
		virtual void abort(BehaviorTree::VirtualMachine& , BehaviorTree::IndexType, void* , BehaviorTree::VMRunningData& running_data) override;
	};
	Delegate delegate;

	virtual BehaviorTree::Node* get_behavior_node() override { return &delegate; }

public:
	BtCustomParallelNode();

protected:
	static void _bind_methods();
};

#endif
