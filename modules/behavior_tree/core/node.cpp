#include "node.h"
#include "virtual_machine.h"

namespace BehaviorTree
{

// NodeImpl
void NodeImpl::restore_running(VirtualMachine& , IndexType, void* , VMRunningData& running_data) { running_data.increase_index(); }
void NodeImpl::prepare(VirtualMachine& , IndexType, void* , VMRunningData& running_data) { running_data.increase_index(); }
E_State NodeImpl::self_update(VirtualMachine& , IndexType, void*, VMRunningData& ) { return BH_ERROR; }
E_State NodeImpl::child_update(VirtualMachine& , IndexType, void* , E_State , VMRunningData& ) { return BH_ERROR; }
void NodeImpl::abort(VirtualMachine& , IndexType, void* , VMRunningData& ) {}

// Action
E_State Action::self_update(VirtualMachine&, IndexType index, void* context, VMRunningData& running_data) {
	return update(index, context, running_data);
}

E_State Action::update(IndexType, void*, VMRunningData& ) { return BH_SUCCESS; }

// Decorator
E_State Decorator::self_update(VirtualMachine& vm, IndexType index, void* context, VMRunningData& running_data) {
	E_State result = pre_update(index, context, running_data);
	if (result == BH_SUCCESS) {
		result = BH_RUNNING;
	} else {
		vm.move_index_to_node_end(index, running_data);
	}
	return result;
}

E_State Decorator::child_update(VirtualMachine&, IndexType index, void* context, E_State child_state, VMRunningData& running_data) {
	return post_update(index, context, child_state, running_data);
}

E_State Decorator::pre_update(IndexType, void*, VMRunningData& ) { return BH_SUCCESS; }
E_State Decorator::post_update(IndexType, void*, E_State child_state, VMRunningData& ) { return child_state; }

} /* BehaviorTree */ 
