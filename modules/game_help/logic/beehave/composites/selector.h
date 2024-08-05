#pragma once
#include "../beehave_node.h"
class BeehaveCompositeSelector : public BeehaveComposite
{
    GDCLASS(BeehaveCompositeSelector, BeehaveComposite);
public:
    virtual TypedArray<StringName> get_class_name()override
    {
        TypedArray<StringName> rs = base_class_type::get_class_name();
        rs.push_back("BeehaveCompositeSelector");
        return rs;
    }
    // 獲取支持放几个子节点,-1 是任意多子节点
    virtual int get_supper_child_count()
    {
        return -1;
    }
    virtual String get_lable_name()
    {
        return String(L"选择组合节点");
    }
    virtual String get_tooltip()
    {
        return String(L"## 选择器节点将尝试执行其每个子节点,直到其中一个返回“SUCCESS”。\n如果所有子节点都返回“FAILURE”,则此节点也将 返回“FAILURE”。\n如果子节点返回“RUNNING”,它将再次运行。");
    }

    virtual int tick(Node * actor, Blackboard* blackboard)override
    {
        for(int i = last_execution_index; i < get_child_count(); i++)
        {
            if(child_state[i] == 0)
            {
                child_state[i] = 1;
                children[i]->before_run(actor,blackboard);
            }
            int rs = children[i]->tick(actor,blackboard);
            if(rs == SUCCESS )
            {
                last_execution_index = i + 1;
                child_state[i] = 2;
                children[i]->after_run(actor,blackboard);
                return SUCCESS;
            }
            else if(rs == FAILURE)
            {
                last_execution_index = i + 1;
                child_state[i] = 2;
                children[i]->after_run(actor,blackboard);
                return rs;
            }
            else if(rs == RUNNING)
            {
                return rs;
            }
        }
        return FAILURE;
    }
    virtual void interrupt(Node * actor, Blackboard* blackboard)override
    {
        base_class_type::interrupt(actor,blackboard);
        last_execution_index = 0;
    }
    virtual void after_run(Node * actor, Blackboard* blackboard)override
    {
        base_class_type::after_run(actor,blackboard);
        last_execution_index = 0;
    }


    int last_execution_index = 0;
};