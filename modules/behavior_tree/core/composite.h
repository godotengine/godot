#ifndef BEHAVIOR_TREE_COMPOSITE_H
#define BEHAVIOR_TREE_COMPOSITE_H

#include "virtual_machine.h"

namespace BehaviorTree
{

class Composite : public NodeImpl
{
protected:
	virtual E_State self_update(VirtualMachine& , IndexType , void*, VMRunningData& ) override;
};

class Selector : public Composite
{
protected:
	virtual E_State child_update(VirtualMachine& vm, IndexType index, void*, E_State child_state, VMRunningData& ) override;
};

class Sequence : public Composite
{
protected:
	virtual void restore_running(VirtualMachine& vm, IndexType index, void* context, VMRunningData& ) override;
	virtual E_State child_update(VirtualMachine& vm, IndexType index, void*, E_State child_state, VMRunningData& ) override;
};

// instead of running children nodes in seperated thread,
// we run chilren nodes step by step without interruption.
template<E_State RESULT_STATE = BH_SUCCESS>
class Parallel : public Composite
{
public:
	Parallel() {
		BT_STATIC_ASSERT(RESULT_STATE != BH_ERROR, "RESULT_STATE cannot be BH_ERROR");
	}

protected:
	virtual E_State child_update(VirtualMachine& vm, IndexType index, void* context, E_State child_state, VMRunningData& ) override;
};


template<E_State RESULT_STATE>
E_State Parallel<RESULT_STATE>::child_update(VirtualMachine& , IndexType , void* , E_State child_state, VMRunningData& ) {
	return child_state == BH_ERROR ? BH_ERROR : RESULT_STATE;
}

} /* BehaviorTree */ 

#endif
