#pragma once
#include "core/object/ref_counted.h"

/// 序列节点
class BeehaveNode : public RefCounted
{
    GDCLASS(BeehaveNode, RefCounted);
public:
    enum SequenceRunState
    {
        SUCCESS, FAILURE, RUNNING
    };
    int get_child_count() const { return children.size(); }

    Ref<BeehaveNode> get_child(int p_index) const
    {
        if(p_index < children.size())
        {
            return children[p_index];
        }
        return Ref<BeehaveNode>(); 
    }


    virtual int tick(Node * actor, Blackboard* blackboard) 
    {
        return SUCCESS;

    }
    // Called when this node needs to be interrupted before it can return FAILURE or SUCCESS.
    virtual void interrupt(Node * actor, Blackboard* blackboard) 
    {
        child_state.resize(children.size());
        child_state.fill(0);
        for(int i = 0; i < children.size(); ++i)
        {
            children[i]->interrupt(actor,blackboard);
        }
    }
    // Called before the first time it ticks by the parent.
    virtual void  before_run(Node * actor, Blackboard* blackboard)  
    {

    }
    // Called after the last time it ticks and returns
    // [code]SUCCESS[/code] or [code]FAILURE[/code].
    virtual void   after_run(Node * actor, Blackboard* blackboard)  
    {
        
    }
    virtual TypedArray<StringName> get_class_name()
    {
        return {"BeehaveNode"};
    }
    bool can_send_message(Blackboard* blackboard)
	{
        return blackboard->get_var("can_send_message", false);
    }
    virtual String get_tooltip()
    {
        return String(L"行为树节点");
    }
    virtual String get_lable_name()
    {
        return String(L"");
    }
    // 獲取支持放几个子节点,-1 是任意多子节点
    virtual int get_supper_child_count()
    {
        return 0;
    }


    LocalVector<Ref<BeehaveNode>> children;
    LocalVector<uint8_t> child_state;
};



class BeehaveComposite : public BeehaveNode
{
    GDCLASS(BeehaveComposite, BeehaveNode);
public:
    virtual String get_tooltip()override
    {
        return String(L"复合节点以特定方式控制其子节点的执行流。");
    }
    virtual String get_lable_name()
    {
        return String(L"组合节点");
    }
    virtual void interrupt(Node * actor, Blackboard* blackboard)override
    {
        base_class_type::interrupt(actor,blackboard);
    }
    virtual TypedArray<StringName> get_class_name()override
    {
        TypedArray<StringName> rs = base_class_type::get_class_name();
        rs.push_back("BeehaveComposite");
        return rs;
    }
    // 獲取支持放几个子节点,-1 是任意多子节点
    virtual int get_supper_child_count()
    {
        return -1;
    }
};

class BeehaveDecorator : public BeehaveNode
{
    GDCLASS(BeehaveDecorator, BeehaveNode);
public:
    virtual String get_tooltip()override
    {
        return String(L"装饰器节点用于转换其子节点接收到的结果。\n只能有一个子节点。");
    }
    virtual String get_lable_name()
    {
        return String(L"装饰器节点");
    }
    virtual void interrupt(Node * actor, Blackboard* blackboard)override
    {
        base_class_type::interrupt(actor,blackboard);
    }
    virtual TypedArray<StringName> get_class_name()override
    {
        TypedArray<StringName> rs = base_class_type::get_class_name();
        rs.push_back("BeehaveDecorator");
        return rs;
    }
    // 獲取支持放几个子节点,-1 是任意多子节点
    int get_supper_child_count()override
    {
        return 1;
    }
};

class BeehaveLeaf : public BeehaveNode
{
    GDCLASS(BeehaveLeaf, BeehaveNode);
public:
    virtual String get_tooltip()override
    {
        return String(L"树的所有叶节点的基类。");
    }
    virtual String get_lable_name()
    {
        return String(L"叶节点");
    }
    virtual void interrupt(Node * actor, Blackboard* blackboard)override
    {
        base_class_type::interrupt(actor,blackboard);
    }
    virtual TypedArray<StringName> get_class_name()override
    {
        TypedArray<StringName> rs = base_class_type::get_class_name();
        rs.push_back("BeehaveLeaf");
        return rs;
    }
    // 獲取支持放几个子节点,-1 是任意多子节点
    int get_supper_child_count()override
    {
        return 0;
    }
};

class BeehaveAction : public BeehaveLeaf
{
    GDCLASS(BeehaveAction, BeehaveLeaf);
    static void _bind_methods()
    {
	    GDVIRTUAL_BIND(_interrupt, "owenr_node", "blackboard");
	    GDVIRTUAL_BIND(_before_run, "owenr_node", "blackboard");
	    GDVIRTUAL_BIND(_after_run, "owenr_node", "blackboard");
	    GDVIRTUAL_BIND(_tick, "owenr_node", "blackboard");
    }
public:
    virtual void interrupt(Node * actor, Blackboard* blackboard)  override
    {
        base_class_type::interrupt(actor,blackboard);	
        GDVIRTUAL_CALL(_interrupt, actor, blackboard);
    }
    virtual void before_run(Node * actor, Blackboard* blackboard)  override
    {
        base_class_type::before_run(actor,blackboard);
        GDVIRTUAL_CALL(_before_run, actor, blackboard);
    }
    virtual void after_run(Node * actor, Blackboard* blackboard)  override
    {
        base_class_type::after_run(actor,blackboard);
        GDVIRTUAL_CALL(_after_run, actor, blackboard);
    }
    virtual int tick(Node * actor, Blackboard* blackboard)  override
    {
        if (GDVIRTUAL_IS_OVERRIDDEN(_tick))
        {            
            int rs;
            GDVIRTUAL_CALL(_tick, actor, blackboard,rs);
            return rs;
        }
        return base_class_type::tick(actor,blackboard);
    }

    
	GDVIRTUAL2(_interrupt,Node*,Blackboard*);
	GDVIRTUAL2(_before_run,Node*,Blackboard*);
	GDVIRTUAL2(_after_run,Node*,Blackboard*);
	GDVIRTUAL2R(int,_tick,Node*,Blackboard*);
};
