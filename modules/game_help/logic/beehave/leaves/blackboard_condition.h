#pragma once
#include "../beehave_node.h"
#include "../../character_ai/animator_condition.h"

class BeehaveLeafBlackboardCondition : public BeehaveLeaf
{
    GDCLASS(BeehaveLeafBlackboardCondition, BeehaveLeaf);

public:
    virtual String get_tooltip()override
    {
        return String(L"判断黑板条件是否成功。");
    }
    virtual String get_lable_name()
    {
        return L"黑板条件叶节点";
    }
    virtual StringName get_icon()
    {
        return SNAME("condition");
    }
    virtual void after_run(Node * actor, Blackboard* blackboard) override
    {
        if(blackboard_condition.is_valid())
        {
            if(blackboard_condition->is_enable(blackboard))
            {
                return ;
            }
            else
            {
                return ;
            }
        }
        return ;
    }


public:
    Ref<CharacterAnimatorCondition> blackboard_condition;
};