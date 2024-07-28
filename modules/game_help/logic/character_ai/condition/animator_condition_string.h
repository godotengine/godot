#pragma once

#include "../animator_blackboard_set.h"

// 字符串表达式
class AnimatorAIStateStringNameCondition : public AnimatorAIStateConditionBase
{
    GDCLASS(AnimatorAIStateStringNameCondition,AnimatorAIStateConditionBase)

    static void _bind_methods()
    {
        ClassDB::bind_method(D_METHOD("set_value","value"),&AnimatorAIStateStringNameCondition::set_value);
        ClassDB::bind_method(D_METHOD("get_value"),&AnimatorAIStateStringNameCondition::get_value);

        ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "value"), "set_value", "get_value");
    }

public:
    void set_value(const StringName& p_value)
    {
        value = p_value;
        update_name();
    }
    StringName get_value()
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
            blackboard_plan->get_property_names_by_type(Variant::STRING,rs);
            blackboard_plan->get_property_names_by_type(Variant::STRING_NAME,rs);
        }
        return rs;
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
        StringName curr = p_blackboard->get_var(propertyName, StringName());
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
    virtual bool is_enable(const Ref<BlackboardPlan>& p_blackboard,bool p_is_include)override
    {
        if(!p_blackboard->has_var(propertyName))
        {
            if(p_is_include)
            {
                return true;
            }
            return false;
        }
        StringName curr = p_blackboard->get_var(propertyName).get_value();
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
    StringName value;
};
