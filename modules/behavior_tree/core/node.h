#ifndef BEHAVIOR_TREE_NODE_H
#define BEHAVIOR_TREE_NODE_H

#include "typedef.h"

namespace BehaviorTree { class VirtualMachine; }

namespace BehaviorTree
{

struct Node
{
	bool initialized;
	Node() : initialized(false) {}
    virtual ~Node() {}

    virtual void restore_running(VirtualMachine& , IndexType, void* ) = 0;
    virtual void init(VirtualMachine& , IndexType, void* ) = 0;
    virtual void prepare(VirtualMachine& , IndexType, void* ) = 0;

    virtual E_State self_update(VirtualMachine& , IndexType, void*) = 0;
    virtual E_State child_update(VirtualMachine& , IndexType, void* , E_State child_state) = 0;

    virtual void abort(VirtualMachine& , IndexType, void* ) = 0;
};

class NodeImpl : public Node
{
protected:
    virtual void restore_running(VirtualMachine& , IndexType, void* ) override;
    virtual void init(VirtualMachine& , IndexType, void* ) override;
    virtual void prepare(VirtualMachine& , IndexType, void* ) override;
    virtual void abort(VirtualMachine& , IndexType, void* ) override;
    virtual E_State self_update(VirtualMachine& , IndexType, void*) override;
    virtual E_State child_update(VirtualMachine& , IndexType, void* , E_State ) override;
};

class Action : public NodeImpl
{
protected:
    virtual E_State self_update(VirtualMachine&, IndexType index, void* context) override;
    virtual E_State update(IndexType, void*);
};

class Decorator : public NodeImpl
{
protected:
    virtual E_State self_update(VirtualMachine&, IndexType, void* context) override;
    virtual E_State child_update(VirtualMachine&, IndexType , void* context, E_State child_state) override;
    virtual E_State pre_update(IndexType, void*);
    virtual E_State post_update(IndexType, void*, E_State child_state);
};

} /* BehaviorTree */ 

#endif
