#pragma once
#include "../beehave_node.h"
class BeehaveCompositeSequence : public BeehaveComposite
{
    GDCLASS(BeehaveCompositeSequence, BeehaveComposite);
public:
    virtual String get_tooltip()override
    {
        return String(L"## 序列节点将尝试执行其所有子节点，并报告, \n如果所有子节点都报告 `SUCCESS` 状态代码，则报告 `SUCCESS`。\n如果至少一个子节点报告 `FAILURE` 状态代码，则此节点也将返回 `FAILURE` 下次继续从这个位置开始。 \n如果子节点返回 `RUNNING`，则此节点将再次运行。");
    }
    virtual String get_tooltip()
    {
        return String(L"序列组合节点");
    }
    virtual void interrupt(Node * actor, Blackboard* blackboard)override
    {
        __supper::interrupt(actor,blackboard);
        successful_index = 0;
    }

    virtual TypedArray<StringName> get_class_name()override
    {
        TypedArray<StringName> rs = __supper::get_class_name();
        rs.push_back(StringName("BeehaveCompositeSequence"));
        return rs;  
    }

    virtual int tick(Node * actor, Blackboard* blackboard)override
    {
        for(int i = successful_index; i < get_child_count(); i++)
        {
            if(child_state[i] == 0)
            {
                child_state[i] = 1;
                children[i]->before_run(actor,blackboard);
            }
            int rs = children[i]->tick(actor,blackboard);
            if(rs == SUCCESS)
            {
                successful_index = i + 1;
                child_state[i] = 2;
                children[i]->after_run(actor,blackboard);
            }
            if(rs == FAILURE)
            {
                successful_index = i + 1;
                children[i]->after_run(actor,blackboard);
                child_state[i] = 0;
                return rs;
            }
            if(rs == RUNNING)
            {
                return rs;
            }
        }
        successful_index = 0;
        return SUCCESS;
    }
    int successful_index = 0;
};