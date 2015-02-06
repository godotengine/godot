#include "composite.h"

namespace BehaviorTree
{

// Composite
E_State Composite::self_update(VirtualMachine& , IndexType , void*, VMRunningData& ) {
	return BH_RUNNING;
}


E_State Selector::child_update(VirtualMachine& vm, IndexType index, void*, E_State child_state, VMRunningData& running_data) {
	if (child_state != BH_FAILURE) {
		vm.move_index_to_node_end(index, running_data);
	}
	return child_state;
}

void Sequence::restore_running(VirtualMachine& vm, IndexType index, void* , VMRunningData& running_data) {
	IndexType child_index = running_data.move_index_to_running_child();
	bool is_child = vm.is_child(index, child_index);
	BT_ASSERT(is_child);
	if (!is_child) {
		vm.move_index_to_node_end(index, running_data);
	}
}

E_State Sequence::child_update(VirtualMachine& vm, IndexType index, void*, E_State child_state, VMRunningData& running_data) {
	if (child_state != BH_SUCCESS) {
		vm.move_index_to_node_end(index, running_data);
	}
	return child_state;
}

void ParallelSequence::prepare(VirtualMachine& vm, IndexType index, void* context, VMRunningData& running_data) {
	Composite::prepare(vm, index, context, running_data);
	running_count = 0;
	success_count = 0;
	failure_count = 0;
	error_count = 0;
}

void ParallelSequence::restore_running(VirtualMachine&, IndexType, void*, VMRunningData&) {

	running_count = 0;
	success_count = 0;
	failure_count = 0;
	error_count = 0;
}

E_State ParallelSequence::child_update(VirtualMachine& vm, IndexType index, void*, E_State child_state, VMRunningData& running_data) {

	switch (child_state) {
	case BH_ERROR: error_count ++; break;
	case BH_SUCCESS: success_count ++; break;
	case BH_FAILURE: failure_count ++; break;
	case BH_RUNNING: running_count ++; break;
	}
	VMRunningData::RunningNode running_node = running_data.running_nodes.back();
	if (running_data.index_marker == running_node.data.end) {

		if (error_count > 0)
			return BH_ERROR;
		if (running_count > 0)
			return BH_RUNNING;
		if (failure_count > 0)
			return BH_FAILURE;
	}
	return BH_SUCCESS;
};

void ParallelSelector::prepare(VirtualMachine& vm, IndexType index, void* context, VMRunningData& running_data) {
	Composite::prepare(vm, index, context, running_data);
	running_count = 0;
	success_count = 0;
	failure_count = 0;
	error_count = 0;
}

void ParallelSelector::restore_running(VirtualMachine&, IndexType, void*, VMRunningData&) {

	running_count = 0;
	success_count = 0;
	failure_count = 0;
	error_count = 0;
}

E_State ParallelSelector::child_update(VirtualMachine& vm, IndexType index, void*, E_State child_state, VMRunningData& running_data) {

	switch (child_state) {
	case BH_ERROR: error_count ++; break;
	case BH_SUCCESS: success_count ++; break;
	case BH_FAILURE: failure_count ++; break;
	case BH_RUNNING: running_count ++; break;
	}
	VMRunningData::RunningNode running_node = running_data.running_nodes.back();
	if (running_data.index_marker == running_node.data.end) {

		if (error_count > 0)
			return BH_ERROR;
		if (running_count > 0)
			return BH_RUNNING;
		if (success_count > 0)
			return BH_SUCCESS;
		return BH_FAILURE;
	}
	return BH_SUCCESS;
};

} /* BehaviorTree */ 
