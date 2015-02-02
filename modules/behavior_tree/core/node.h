#ifndef BEHAVIOR_TREE_NODE_H
#define BEHAVIOR_TREE_NODE_H

#include "typedef.h"

namespace BehaviorTree { struct VMRunningData; }
namespace BehaviorTree { class VirtualMachine; }

namespace BehaviorTree
{

struct Node
{
	virtual ~Node() {}

	virtual void restore_running(VirtualMachine& , IndexType, void* , VMRunningData& ) = 0;
	virtual void prepare(VirtualMachine& , IndexType, void* , VMRunningData& ) = 0;

	virtual E_State self_update(VirtualMachine& , IndexType, void*, VMRunningData& ) = 0;
	virtual E_State child_update(VirtualMachine& , IndexType, void* , E_State child_state, VMRunningData& ) = 0;

	virtual void abort(VirtualMachine& , IndexType, void* , VMRunningData& ) = 0;
};

class NodeImpl : public Node
{
protected:
	virtual void restore_running(VirtualMachine& , IndexType, void* , VMRunningData& ) override;
	virtual void prepare(VirtualMachine& , IndexType, void* , VMRunningData& ) override;
	virtual void abort(VirtualMachine& , IndexType, void* , VMRunningData& ) override;
	virtual E_State self_update(VirtualMachine& , IndexType, void*, VMRunningData& ) override;
	virtual E_State child_update(VirtualMachine& , IndexType, void* , E_State , VMRunningData& ) override;
};

class Action : public NodeImpl
{
protected:
	virtual E_State self_update(VirtualMachine&, IndexType index, void* context, VMRunningData& ) override;
	virtual E_State update(IndexType, void*, VMRunningData& );
};

class Decorator : public NodeImpl
{
protected:
	virtual E_State self_update(VirtualMachine&, IndexType, void* context, VMRunningData& ) override;
	virtual E_State child_update(VirtualMachine&, IndexType , void* context, E_State child_state, VMRunningData& ) override;
	virtual E_State pre_update(IndexType, void*, VMRunningData& );
	virtual E_State post_update(IndexType, void*, E_State child_state, VMRunningData& );
};

} /* BehaviorTree */ 

#endif
