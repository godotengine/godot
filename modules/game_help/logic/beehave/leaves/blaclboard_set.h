#pragma once

#include "../beehave_node.h"
#include "../../character_ai/animator_blackboard_set.h"

class BeehaveLeafBlackboardSet : public BeehaveLeaf
{
    GDCLASS(BeehaveLeafBlackboardSet, BeehaveLeaf);
    static void _bind_methods()
    {
        ClassDB::bind_method(D_METHOD("set_blackboard_set","blackboard_set"),&BeehaveLeafBlackboardSet::set_blackboard_set);
        ClassDB::bind_method(D_METHOD("get_blackboard_set"),&BeehaveLeafBlackboardSet::get_blackboard_set);

        ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "blackboard_set", PROPERTY_HINT_RESOURCE_TYPE, "CharacterAnimatorBlackboardSet"), "set_blackboard_set", "get_blackboard_set");
        
    }

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
        if(blackboard_set.is_valid())
        {
            blackboard_set->execute(blackboard);
        }
        return ;
    }
public:
    void set_blackboard_set(Ref<AnimatorBlackboardSet> p_blackboard_set)
    {
        blackboard_set = p_blackboard_set;
    }
    Ref<AnimatorBlackboardSet> get_blackboard_set()
    {
        return blackboard_set;
    }
protected:
    Ref<AnimatorBlackboardSet> blackboard_set;
};