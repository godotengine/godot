#pragma once
#include "../animator_blackboard_set.h"

// float类型条件
class AnimatorAIStateFloatCondition : public AnimatorAIStateConditionBase
{
    GDCLASS(AnimatorAIStateFloatCondition,AnimatorAIStateConditionBase)

    static void _bind_methods()
    {
        ClassDB::bind_method(D_METHOD("set_value","value"),&AnimatorAIStateFloatCondition::set_value);
        ClassDB::bind_method(D_METHOD("get_value"),&AnimatorAIStateFloatCondition::get_value);

        ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "value"), "set_value", "get_value");
    }

public:
    void set_value(float p_value)
    {
        value = p_value;
        update_name();
    }
    float get_value()
    {
        return value;
    }
protected:
    virtual Array _get_compare_value() override
    {
        Array ret;
        ret.append(StringName(">"));
        ret.append(StringName(">="));
        ret.append(StringName("<"));
        ret.append(StringName("<="));
        return ret;
    }
    virtual void update_name()
    {
    }
    virtual Array _get_blackbord_propertys() override
    {
        Array rs;
        if(!blackboard_plan.is_null())
        {
            blackboard_plan->get_property_names_by_type(Variant::FLOAT,rs);
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
        double curr = p_blackboard->get_var(propertyName, 0.0);
        double v = value;
        if(is_value_by_property && p_blackboard->has_var(value_property_name))
        {
            v = p_blackboard->get_var(value_property_name, 0.0f);
        }
        switch (compareType)
        {
            case Greater:
                return curr > v;
            case AnimatorAICompareType::GreaterEqual:
                return curr >= v;
            case AnimatorAICompareType::Less:
                return curr < v;
            case AnimatorAICompareType::LessEqual:
                return curr <= v;
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
        double curr = p_blackboard->get_var(propertyName).get_value();
        double v = value;
        if(is_value_by_property && p_blackboard->has_var(value_property_name))
        {
            v = p_blackboard->get_var(value_property_name).get_value();
        }
        switch (compareType)
        {
            case Greater:
                return curr > v;
            case AnimatorAICompareType::GreaterEqual:
                return curr >= v;
            case AnimatorAICompareType::Less:
                return curr < v;
            case AnimatorAICompareType::LessEqual:
                return curr <= v;
        }
        if(p_is_include)
        {
            return true;
        }
        return false;
    }
protected:
    double value;

};
