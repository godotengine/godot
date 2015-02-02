#include "node.h"
#include "virtual_machine.h"

namespace BehaviorTree
{

// NodeImpl
void NodeImpl::restore_running(VirtualMachine& vm, IndexType, void* ) { vm.increase_index(); }
void NodeImpl::init(VirtualMachine& vm, IndexType, void* ) {}
void NodeImpl::prepare(VirtualMachine& vm, IndexType, void* ) { vm.increase_index(); }
E_State NodeImpl::self_update(VirtualMachine& , IndexType, void*) { return BH_ERROR; }
E_State NodeImpl::child_update(VirtualMachine& , IndexType, void* , E_State ) { return BH_ERROR; }
void NodeImpl::abort(VirtualMachine& , IndexType, void* ) {}

// Action
E_State Action::self_update(VirtualMachine&, IndexType index, void* context) { return update(index, context); }
E_State Action::update(IndexType, void*) { return BH_SUCCESS; }

// Decorator
E_State Decorator::self_update(VirtualMachine& vm, IndexType index, void* context) {
    E_State result = pre_update(index, context);
    if (result == BH_SUCCESS) {
        result = BH_RUNNING;
    } else {
        vm.move_index_to_node_end(index);
    }
    return result;
}

E_State Decorator::child_update(VirtualMachine&, IndexType index, void* context, E_State child_state) {
    return post_update(index, context, child_state);
}

E_State Decorator::pre_update(IndexType, void*) { return BH_SUCCESS; }
E_State Decorator::post_update(IndexType, void*, E_State child_state) { return child_state; }

} /* BehaviorTree */ 
