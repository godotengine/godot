#pragma once

#include "../animator_blackboard_set.h"

// 字符串表达式
class AnimatorBlackboardSetItemString : public AnimatorBlackboardSetItemBase
{
    GDCLASS(AnimatorBlackboardSetItemString,AnimatorBlackboardSetItemBase);

    static void _bind_methods()
    {
        ClassDB::bind_method(D_METHOD("set_value","value"),&AnimatorBlackboardSetItemString::set_value);
        ClassDB::bind_method(D_METHOD("get_value"),&AnimatorBlackboardSetItemString::get_value);

        ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "value"), "set_value", "get_value");

    }

public:
    void set_value(StringName p_value)
    {
        value = p_value;
        update_name();
    }
    StringName get_value()
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
            blackboard_plan->get_property_names_by_type(Variant::STRING,rs);
            blackboard_plan->get_property_names_by_type(Variant::STRING_NAME,rs);
        }
        return rs;
    }
    StringName value;
};
