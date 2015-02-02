#ifndef BEHAVIOR_TREE_VIRTUAL_MACHINE_H
#define BEHAVIOR_TREE_VIRTUAL_MACHINE_H

#include "node.h"

namespace BehaviorTree
{

struct VMRunningData
{
	struct RunningNode
	{
		Node* node;
		NodeData data;
	};

	BTVector<RunningNode> running_nodes;
	BTVector<IndexType> this_tick_running;
	BTVector<IndexType> last_tick_running;
	IndexType index_marker;

	void tick_begin();
	void tick_end();

	void sort_last_running_nodes();
	bool is_current_node_running_on_last_tick() const;
	void pop_last_running_behavior();
	void add_running_node(Node* node, NodeData node_data);

	IndexType move_index_to_running_child();

	inline void increase_index() { ++index_marker; }
};

class VirtualMachine
{
private:
	NodeList& node_list;
	const BtStructure& structure_data;

public:
	VirtualMachine(NodeList& node_list_, const BtStructure& structure_data_)
		: node_list(node_list_), structure_data(structure_data_)
	{}

	// execute the whole behavior tree.
	void tick(void* context, VMRunningData& running_data);
	// running behavior tree step by step.
	void step(void* context, VMRunningData& running_data);

	void move_index_to_node_end(IndexType index, VMRunningData& running_data);

	inline NodeData get_node_data(IndexType index) const { return structure_data[index]; }

	bool is_child(IndexType parent_index, IndexType child_index) const;

private:
	void cancel_skipped_behaviors(void* context, VMRunningData& running_data);
	void cancel_behavior(void* context, VMRunningData& running_data);
	void run_composites(E_State state, void* context, VMRunningData& running_data);
	E_State run_action(Node& node, void* context, VMRunningData& running_data);
};

} /* BehaviorTree */ 

#endif
