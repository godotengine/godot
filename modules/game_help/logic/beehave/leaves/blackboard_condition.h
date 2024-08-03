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
    virtual int tick(Node * actor, Blackboard* blackboard) override
    {
        if(blackboard_condition.is_valid())
        {
            if(blackboard_condition->is_enable(blackboard))
            {
                return SUCCESS;
            }
            else
            {
                return FAILURE;
            }
        }
        return FAILURE;
    }


public:
    Ref<CharacterAnimatorCondition> blackboard_condition;
};