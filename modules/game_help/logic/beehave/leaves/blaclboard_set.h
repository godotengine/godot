#pragma omce
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
    Ref<BeehaveLeafBlackboardSet> blackboard_condition;
};