#include "virtual_machine.h"

namespace BehaviorTree
{

void VirtualMachine::tick(void* context, VMRunningData& running_data) {
	BT_ASSERT(structure_data.size() == node_list.size());
	running_data.tick_begin();
	size_t num_nodes = structure_data.size();
	while (running_data.index_marker < num_nodes) {
		step(context, running_data);
	}
	running_data.tick_end();
}

void VirtualMachine::step(void* context, VMRunningData& running_data) {
	NodeData node_data = structure_data[running_data.index_marker];
	Node* node = node_list[running_data.index_marker];

	BT_ASSERT(node);
	E_State state = BH_ERROR;
	if (node) state = run_action(*node, context, running_data);
	// skip this node and its children if this node is null
	else running_data.index_marker = node_data.end;

	BT_ASSERT(running_data.index_marker <= node_data.end);
	if (running_data.index_marker < node_data.end) {
		// this node should be a composite or a decorator
		BT_ASSERT(state == BH_RUNNING);
		running_data.add_running_node(node, node_data);
	} else {
		if (state == BH_RUNNING) {
			running_data.this_tick_running.push_back(node_data.index);
		}
		run_composites(state, context, running_data);
	}
	cancel_skipped_behaviors(context, running_data);
}

void VirtualMachine::run_composites(E_State state, void* context, VMRunningData& running_data) {
	while (!running_data.running_nodes.empty()) {
		VMRunningData::RunningNode running_node = running_data.running_nodes.back();
		state = running_node.node->child_update(*this, running_node.data.index, context, state, running_data);
		BT_ASSERT(running_data.index_marker <= running_node.data.end);
		if (running_data.index_marker == running_node.data.end) {
			running_data.running_nodes.pop_back();
			if (state == BH_RUNNING) {
				running_data.this_tick_running.push_back(running_node.data.index);
			}
			continue;
		} else {
			break;
		}
	}
}

E_State VirtualMachine::run_action(Node& node, void* context, VMRunningData& running_data) {
	IndexType running_node_index = running_data.index_marker;
	if (running_data.is_current_node_running_on_last_tick()) {
		running_data.pop_last_running_behavior();
		node.restore_running(*this, running_node_index, context, running_data);
	} else {
		node.prepare(*this, running_node_index, context, running_data);
	}
	return node.self_update(*this, running_node_index, context, running_data);
}

void VirtualMachine::cancel_skipped_behaviors(void* context, VMRunningData& running_data) {
	while (!running_data.last_tick_running.empty() &&
			running_data.last_tick_running.back() < running_data.index_marker) {
		cancel_behavior(context, running_data);
		running_data.pop_last_running_behavior();
	}
}

void VirtualMachine::cancel_behavior(void* context, VMRunningData& running_data) {
	IndexType index = running_data.last_tick_running.back();
	Node* node = node_list[index];
	if (node) node->abort(*this, index, context, running_data);
}

void VirtualMachine::move_index_to_node_end(IndexType index, VMRunningData& running_data) {
	running_data.index_marker = structure_data[index].end;
}

bool VirtualMachine::is_child(IndexType parent_index, IndexType child_index) const {
	NodeData node_data = get_node_data(parent_index);
	return (node_data.begin < child_index && child_index < node_data.end);
}

// function related to VMRunningData
void VMRunningData::tick_begin() {
	this_tick_running.resize(0);
	running_nodes.resize(0);
	index_marker = 0;
	sort_last_running_nodes();
}

void VMRunningData::tick_end() {
	this_tick_running.swap(last_tick_running);
	this_tick_running.clear();
}

struct IndexGreatThanComp
{
	bool operator()(IndexType lhs, IndexType rhs) const { return lhs > rhs; }
};

void VMRunningData::sort_last_running_nodes() {
	sort<IndexGreatThanComp>(last_tick_running);
}

bool VMRunningData::is_current_node_running_on_last_tick() const { 
	return !last_tick_running.empty() && index_marker == last_tick_running.back();
}

void VMRunningData::pop_last_running_behavior() {
	last_tick_running.pop_back();
}

void VMRunningData::add_running_node(Node* node, NodeData node_data) {
	BT_ASSERT(node != NULL);
	RunningNode running = { node, node_data };
	running_nodes.push_back(running);
}

IndexType VMRunningData::move_index_to_running_child() {
	BT_ASSERT(!last_tick_running.empty());
	index_marker = last_tick_running.back();
	return index_marker;
}

} /* BehaviorTree */ 
