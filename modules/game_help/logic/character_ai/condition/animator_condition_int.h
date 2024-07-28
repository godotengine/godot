#pragma once
#include "../animator_blackboard_set.h"


// int类型
class AnimatorAIStateIntCondition : public AnimatorAIStateConditionBase
{
    GDCLASS(AnimatorAIStateIntCondition,AnimatorAIStateConditionBase)

    static void _bind_methods()
    {
        ClassDB::bind_method(D_METHOD("set_value","value"),&AnimatorAIStateIntCondition::set_value);
        ClassDB::bind_method(D_METHOD("get_value"),&AnimatorAIStateIntCondition::get_value);

        ADD_PROPERTY(PropertyInfo(Variant::INT, "value"), "set_value", "get_value");
    }
public:
    void set_value(int32_t p_value)
    {
        value = p_value;
        update_name();
    }
    int32_t get_value()
    {
        return value;
    }
protected:
    void set_is_value_by_property(bool p_value)
    {
        is_value_by_property = p_value;
        update_name();
    }
    bool get_is_value_by_property()
    {
        return is_value_by_property;
    }

    void set_value_property_name(const StringName& p_name)
    {
        value_property_name = p_name;
        update_name();
    }
    StringName get_value_property_name()
    {
        return value_property_name;
    }
    virtual void update_name()
    {
    }

    virtual Array _get_compare_value() override
    {
        Array ret;
        ret.append(StringName("=="));
        ret.append(StringName(">"));
        ret.append(StringName(">="));
        ret.append(StringName("<"));
        ret.append(StringName("<="));
        ret.append(StringName("!="));
		return ret;
    }
    virtual Array _get_blackbord_propertys() override
    {
        Array rs;
        if(!blackboard_plan.is_null())
        {
            blackboard_plan->get_property_names_by_type(Variant::INT,rs);
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
        int64_t curr = p_blackboard->get_var(propertyName, 0);
        int64_t v = value;
        if(is_value_by_property && p_blackboard->has_var(value_property_name))
        {
            v = p_blackboard->get_var(value_property_name, 0);
        }
        switch (compareType)
        {
            case AnimatorAICompareType::Equal:
                return curr == v;
            case AnimatorAICompareType::Greater:
                return curr > v;
            case AnimatorAICompareType::GreaterEqual:
                return curr >= v;
            case AnimatorAICompareType::Less:
                return curr < v;
            case AnimatorAICompareType::LessEqual:
                return curr <= v;
            case AnimatorAICompareType::NotEqual:
                return curr != v;
        }
        if(p_is_include)
        {
            return true;
        }
        return false;
    }
    virtual bool is_enable(const Ref<BlackboardPlan>& p_blackboard, bool p_is_include)
    {
        if(!p_blackboard->has_var(propertyName))
        {
            if(p_is_include)
            {
                return true;
            }
            return false;
        }
        int64_t curr = p_blackboard->get_var(propertyName).get_value();
        int64_t v = value;
        if(is_value_by_property && p_blackboard->has_var(value_property_name))
        {
            v = p_blackboard->get_var(value_property_name).get_value();
        }
        switch (compareType)
        {
            case AnimatorAICompareType::Equal:
                return curr == v;
            case AnimatorAICompareType::Greater:
                return curr > v;
            case AnimatorAICompareType::GreaterEqual:
                return curr >= v;
            case AnimatorAICompareType::Less:
                return curr < v;
            case AnimatorAICompareType::LessEqual:
                return curr <= v;
            case AnimatorAICompareType::NotEqual:
                return curr != v;
        }
        if(p_is_include)
        {
            return true;
        }
        return false;
    }
    int64_t value;

};
