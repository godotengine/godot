#pragma once
#include "scene/3d/node_3d.h"
#include "modules/limboai/bt/bt_player.h"

// 继承Resource目的主要是为了编辑时方便显示名称
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
    virtual bool is_enable(const Ref<BlackboardPlan>& p_blackboard,bool p_is_include)
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

    void set_blackboard_plan(const Ref<BlackboardPlan>& p_blackboard_plan);
    Ref<BlackboardPlan> get_blackboard_plan() { return blackboard_plan; }

    
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
    void set_property_name(const StringName& p_name) { propertyName = p_name; update_name();}
    StringName get_property_name() { return propertyName; }

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
    void set_compare_type_name(const StringName& p_type);
    StringName get_compare_type_name();
protected:
    Ref<BlackboardPlan> blackboard_plan;
    StringName propertyName;
    AnimatorAICompareType compareType;
    bool is_value_by_property = false;
    StringName value_property_name;

};

// 角色动画的条件
class CharacterAnimatorCondition : public RefCounted
{

    GDCLASS(CharacterAnimatorCondition, RefCounted)
    static void _bind_methods();
public :
    void set_include_condition(const TypedArray<Ref<AnimatorAIStateConditionBase>>& p_include_condition) 
    {
        include_condition.clear();
        for(int32_t i = 0; i < p_include_condition.size(); ++i)
        {
            include_condition.push_back(p_include_condition[i]);
        }
         update_blackboard_plan();
    }
    TypedArray<Ref<AnimatorAIStateConditionBase>> get_include_condition()
     {
        TypedArray<Ref<AnimatorAIStateConditionBase>> rs;
        for(uint32_t i = 0; i < include_condition.size(); ++i)
        {
            rs.push_back(include_condition[i]);
        }
         return rs; 
    }
    void add_include_condition(const Ref<AnimatorAIStateConditionBase>& p_include_condition)
    {
        include_condition.push_back(p_include_condition);
        update_blackboard_plan();
    }
    void remove_include_condition(const Ref<AnimatorAIStateConditionBase>& p_include_condition)
    {
        for(uint32_t i = 0; i < include_condition.size(); ++i)
        {
            if(include_condition[i] == p_include_condition)
            {
                include_condition.remove_at(i);
                break;
            }
        }
        update_blackboard_plan();
    }

    void set_exclude_condition(const TypedArray<Ref<AnimatorAIStateConditionBase>>& p_exclude_condition)
    {
        exclude_condition.clear();
        for(int32_t i = 0; i < p_exclude_condition.size(); ++i)
        {
            exclude_condition.push_back(p_exclude_condition[i]);
        }
         update_blackboard_plan();
    }
    TypedArray<Ref<AnimatorAIStateConditionBase>> get_exclude_condition()
     {
        TypedArray<Ref<AnimatorAIStateConditionBase>> rs;
        for(uint32_t i = 0; i < exclude_condition.size(); ++i)
        {
            rs.push_back(exclude_condition[i]);
        }
         return rs; 
    }
    void add_exclude_condition(const Ref<AnimatorAIStateConditionBase>& p_exclude_condition)
    {
        exclude_condition.push_back(p_exclude_condition);
        update_blackboard_plan();
    }
    void remove_exclude_condition(const Ref<AnimatorAIStateConditionBase>& p_exclude_condition)
    {
        for(uint32_t i = 0; i < exclude_condition.size(); ++i)
        {
            if(exclude_condition[i] == p_exclude_condition)
            {
                exclude_condition.remove_at(i);
                break;
            }
        }
        update_blackboard_plan();
    }
    void update_blackboard_plan()
    {
        for(uint32_t i = 0; i < include_condition.size(); ++i)
        {
			if(include_condition[i].is_valid())
				include_condition[i]->set_blackboard_plan(blackboard_plan);
        }
        for(uint32_t i = 0; i < exclude_condition.size(); ++i)
        {
			if (exclude_condition[i].is_valid())
				exclude_condition[i]->set_blackboard_plan(blackboard_plan);
        }
    }
public:
    void set_blackboard_plan(const Ref<BlackboardPlan>& p_blackboard_plan) { blackboard_plan = p_blackboard_plan; update_blackboard_plan();}
    virtual bool is_enable(Blackboard* p_blackboard)
    {
        if(!is_enable(include_condition,p_blackboard,true))
        {
            return false;
        }
        if(!is_enable(exclude_condition,p_blackboard,false))
        {
            return false;
        }
        return true;
    }
    virtual bool is_enable(const LocalVector<Ref<AnimatorAIStateConditionBase>> & conditions,Blackboard* p_blackboard,bool p_is_include)
    {
        if(p_is_include)
        {
            for (uint32_t i = 0; i < conditions.size(); ++i)
            {
                if(conditions[i].is_null())
                {
                    continue;
                }
                if (!conditions[i]->is_enable(p_blackboard,p_is_include))
                {
                    return false;
                }
            }
            return true;
        }
        
        for (uint32_t i = 0; i < conditions.size(); ++i)
        {
            if(conditions[i].is_null())
            {
                continue;
            }
            if (conditions[i]->is_enable(p_blackboard,p_is_include))
            {
                return false;
            }
        }
        return true;
    }
public:
    LocalVector<Ref<AnimatorAIStateConditionBase>> include_condition;
    LocalVector<Ref<AnimatorAIStateConditionBase>> exclude_condition;
    Ref<BlackboardPlan> blackboard_plan;
};
VARIANT_ENUM_CAST(AnimatorAIStateConditionBase::AnimatorAICompareType)





