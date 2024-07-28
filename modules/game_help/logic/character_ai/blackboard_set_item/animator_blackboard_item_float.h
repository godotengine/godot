#pragma once

#include "../animator_blackboard_set.h"

// 字符串表达式
class AnimatorBlackboardSetItemFloat : public AnimatorBlackboardSetItemBase
{
    GDCLASS(AnimatorBlackboardSetItemFloat,AnimatorBlackboardSetItemBase);

    static void _bind_methods()
    {
        ClassDB::bind_method(D_METHOD("set_value","value"),&AnimatorBlackboardSetItemFloat::set_value);
        ClassDB::bind_method(D_METHOD("get_value"),&AnimatorBlackboardSetItemFloat::get_value);

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
    virtual void _execute(Blackboard* blackboard) override
    {
        blackboard->set(propertyName,value);
    }

protected:
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
    float value;
};
