#pragma once

#include "../beehave_node.h"
#include "../../character_ai/animator_blackboard_set.h"

class BeehaveLeafBlackboardSet : public BeehaveLeaf
{
    GDCLASS(BeehaveLeafBlackboardSet, BeehaveLeaf);

public:
    virtual String get_tooltip()override
    {
        return String(L"设置黑板的值。");
    }
    virtual String get_lable_name()
    {
        return String(L"设置黑板的叶节点");
    }
    virtual StringName get_icon()
    {
        return SNAME("BTSetVar");
    }
    virtual void after_run(Node * actor, Blackboard* blackboard) override
    {
        if(blackboard_condition.is_valid())
        {
            blackboard_condition->execute(blackboard);
        }
        return ;
    }


public:
    Ref<AnimatorBlackboardSet> blackboard_condition;
};