#pragma once

#include "../animator_blackboard_set.h"

// 字符串表达式
class AnimatorAIStateBoolCondition : public AnimatorAIStateConditionBase
{
    GDCLASS(AnimatorAIStateBoolCondition,AnimatorAIStateConditionBase)

    static void _bind_methods()
    {
        ClassDB::bind_method(D_METHOD("set_value","value"),&AnimatorAIStateBoolCondition::set_value);
        ClassDB::bind_method(D_METHOD("get_value"),&AnimatorAIStateBoolCondition::get_value);

        ADD_PROPERTY(PropertyInfo(Variant::BOOL, "value"), "set_value", "get_value");
    }

public:
    void set_value(bool p_value)
    {
        value = p_value;
        update_name();
    }
    bool get_value()
    {
        return value;
    }

protected:
    virtual void update_name()
    {
    }
    virtual Array _get_compare_value() override
    {
        Array ret;
        ret.append(StringName("=="));
        ret.append(StringName("!="));
		return ret;
    }
    virtual Array _get_blackbord_propertys() override
    {
        Array rs;
        if(!blackboard_plan.is_null())
        {
            blackboard_plan->get_property_names_by_type(Variant::BOOL,rs);
        }
        return rs;
    }
    virtual bool is_enable(const Ref<BlackboardPlan>&  p_blackboard,bool p_is_include)override
    {
        if(!p_blackboard->has_var(propertyName))
        {
            if(p_is_include)
            {
                return true;
            }
            return false;
        }
        bool curr = p_blackboard->get_var(propertyName).get_value();
        switch (compareType)
        {
            case AnimatorAICompareType::Equal:
                return curr == value;
            case AnimatorAICompareType::NotEqual:
                return curr != value;
        }
        if(p_is_include)
        {
            return true;
        }
        return false;
    }
    virtual bool is_enable(Blackboard* p_blackboard,bool p_is_include)override
    {
        if(!p_blackboard->has_var(propertyName))
        {
            if(p_is_include)
            {
                return true;
            }
            return false;
        }
        bool curr = p_blackboard->get_var(propertyName, false);
        switch (compareType)
        {
            case AnimatorAICompareType::Equal:
                return curr == value;
            case AnimatorAICompareType::NotEqual:
                return curr != value;
        }
        if(p_is_include)
        {
            return true;
        }
        return false;
    }
    bool value;
};
