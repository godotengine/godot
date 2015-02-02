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

} /* BehaviorTree */ 
