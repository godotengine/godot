#pragma once
#include "scene/3d/node_3d.h"
#include "modules/limboai/bt/bt_player.h"
class AnimatorAIStateConditionBase : public RefCounted
{
    GDCLASS(AnimatorAIStateConditionBase, RefCounted)
    static void _bind_methods();

public:
    enum AnimatorAICompareType
    {
        //[LabelText("等于")]
        Equal,

        //[LabelText("小于")]
        Less,
        //[LabelText("小于等于")]
        LessEqual,
        //[LabelText("大于")]
        Greater,
        //[LabelText("大于等于")]
        GreaterEqual,
        //[LabelText("不等于")]
        NotEqual,
    };
    virtual bool is_enable(Blackboard* p_blackboard, bool p_is_include)
    {
        if(p_is_include)
        {
            return true;
        }
        return false;
    }
    virtual void update_name()
    {

    }

    void set_blackbord_property_name(const StringName& p_name) { propertyName = p_name; update_name();}
    StringName get_blackbord_property_name() { return propertyName; }

    void set_compare_type(AnimatorAICompareType p_type) { compareType = p_type; update_name();}
    AnimatorAICompareType get_compare_type() { return compareType; }

    void set_blackboard_plan(const Ref<BlackboardPlan>& p_blackboard_plan) { blackboard_plan = p_blackboard_plan; }

    Array get_compare_value() 
    {
        return _get_compare_value();
    }
    virtual Array _get_compare_value() 
    {
        return Array();
    }
    Array get_blackbord_propertys()
    {
        return _get_blackbord_propertys();
    }
    virtual Array _get_blackbord_propertys()
    {
        return Array();
    }
    void set_compare_type_name(const StringName& p_type)
    {
        if(p_type == "==")
            compareType = AnimatorAICompareType::Equal;
        else if(p_type == "<")
            compareType = AnimatorAICompareType::Less;
        else if(p_type == "<=")
            compareType = AnimatorAICompareType::LessEqual;
        else if(p_type == ">")
            compareType = AnimatorAICompareType::Greater;
        else if(p_type == ">=")
            compareType = AnimatorAICompareType::GreaterEqual;
        else if(p_type == "!=")
            compareType = AnimatorAICompareType::NotEqual;
    }
    StringName get_compare_type_name()
    {
        if(compareType == AnimatorAICompareType::Equal)
            return "==";
        else if(compareType == AnimatorAICompareType::Less)
            return "<";
        else if(compareType == AnimatorAICompareType::LessEqual)
            return "<=";
        else if(compareType == AnimatorAICompareType::Greater)
            return ">";
        else if(compareType == AnimatorAICompareType::GreaterEqual)
            return ">=";
        else if(compareType == AnimatorAICompareType::NotEqual)
            return "!=";
        return "==";
    }
    protected:
    Ref<BlackboardPlan> blackboard_plan;
    StringName propertyName;
    AnimatorAICompareType compareType;

};
// float类型条件
class AnimatorAIStateFloatCondition : public AnimatorAIStateConditionBase
{
    float value;
    virtual Array _get_compare_value() override
    {
        Array ret;
        ret.append(StringName(">"));
        ret.append(StringName(">="));
        ret.append(StringName("<"));
        ret.append(StringName("<="));
        return Array();
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
        float curr = p_blackboard->get_var(propertyName, 0.0f);
        switch (compareType)
        {
            case Greater:
                return curr > value;
            case AnimatorAICompareType::GreaterEqual:
                return curr >= value;
            case AnimatorAICompareType::Less:
                return curr < value;
            case AnimatorAICompareType::LessEqual:
                return curr <= value;
        }
        if(p_is_include)
        {
            return true;
        }
        return false;
    }

};
// int类型
class AnimatorAIStateIntCondition : public AnimatorAIStateConditionBase
{
    virtual Array _get_compare_value() override
    {
        Array ret;
        ret.append(StringName("=="));
        ret.append(StringName(">"));
        ret.append(StringName(">="));
        ret.append(StringName("<"));
        ret.append(StringName("<="));
        ret.append(StringName("!="));
        return Array();
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
        int curr = p_blackboard->get_var(propertyName, 0);
        switch (compareType)
        {
            case AnimatorAICompareType::Equal:
                return curr == value;
            case AnimatorAICompareType::Greater:
                return curr > value;
            case AnimatorAICompareType::GreaterEqual:
                return curr >= value;
            case AnimatorAICompareType::Less:
                return curr < value;
            case AnimatorAICompareType::LessEqual:
                return curr <= value;
            case AnimatorAICompareType::NotEqual:
                return curr != value;
        }
        if(p_is_include)
        {
            return true;
        }
        return false;
    }
    float value;

};
// 字符串表达式
class AnimatorAIStateStringNameCondition : public AnimatorAIStateConditionBase
{
    virtual Array _get_compare_value() override
    {
        Array ret;
        ret.append(StringName("=="));
        ret.append(StringName("!="));
        return Array();
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
    StringName value;
};
// 角色动画的条件
class CharacterAnimatorConditionList : public RefCounted
{
    GDCLASS(CharacterAnimatorConditionList, RefCounted)
    static void _bind_methods()
    {

    }
public:
    
    virtual bool is_enable(Blackboard* p_blackboard,bool p_is_include)
    {
        if(p_is_include)
        {
            for (uint32_t i = 0; i < conditions.size(); ++i)
            {
                if (!conditions[i]->is_enable(p_blackboard,p_is_include))
                {
                    return false;
                }
            }
            return true;
        }
        
        for (uint32_t i = 0; i < conditions.size(); ++i)
        {
            if (conditions[i]->is_enable(p_blackboard,p_is_include))
            {
                return false;
            }
        }
        return true;
    }
    void set_blackboard_plan(const Ref<BlackboardPlan>& p_blackboard_plan) { blackboard_plan = p_blackboard_plan; }

public:
    LocalVector<Ref<AnimatorAIStateConditionBase>> conditions;
    Ref<BlackboardPlan> blackboard_plan;
    
};
// 角色动画的条件
class CharacterAnimatorCondition : public RefCounted
{

    GDCLASS(CharacterAnimatorCondition, RefCounted)
    static void _bind_methods()
    {

    }
    public :
    void set_include_condition(const Ref<CharacterAnimatorConditionList>& p_include_condition) 
    {
         include_condition = p_include_condition; 
    }
    Ref<CharacterAnimatorConditionList> get_include_condition() { return include_condition; }

    void set_exclude_condition(const Ref<CharacterAnimatorConditionList>& p_exclude_condition)
    {
         exclude_condition = p_exclude_condition; 
    }
    Ref<CharacterAnimatorConditionList> get_exclude_condition() { return exclude_condition; }
    void update_blackboard_plan()
    {

    }
public:
    void set_blackboard_plan(const Ref<BlackboardPlan>& p_blackboard_plan) { blackboard_plan = p_blackboard_plan; update_blackboard_plan();}
    virtual bool is_enable(Blackboard* p_blackboard,bool p_is_include)
    {
        if(include_condition.is_valid())
        {
            if(!include_condition->is_enable(p_blackboard,true))
            {
                return false;
            }
        }
        if(exclude_condition.is_valid())
        {
            if(!exclude_condition->is_enable(p_blackboard,false))
            {
                return false;
            }
        }
        return true;
    }
public:
    Ref<CharacterAnimatorConditionList> include_condition;
    Ref<CharacterAnimatorConditionList> exclude_condition;
    Ref<BlackboardPlan> blackboard_plan;
    
};
VARIANT_ENUM_CAST(AnimatorAIStateConditionBase::AnimatorAICompareType)