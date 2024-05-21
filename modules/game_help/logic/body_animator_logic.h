#pragma once
#include "scene/3d/node_3d.h"
#include "modules/limboai/bt/bt_player.h"
class AnimatorAIStateConditionBase : public Resource
{
    GDCLASS(AnimatorAIStateConditionBase, Resource)
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

    void set_blackboard_plan(const Ref<BlackboardPlan>& p_blackboard_plan);
    Ref<BlackboardPlan> get_blackboard_plan() { return blackboard_plan; }

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

};
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

protected:
    void set_value(float p_value)
    {
        value = p_value;
        update_name();
    }
    float get_value()
    {
        return value;
    }
    virtual Array _get_compare_value() override
    {
        Array ret;
        ret.append(StringName(">"));
        ret.append(StringName(">="));
        ret.append(StringName("<"));
        ret.append(StringName("<="));
        return Array();
    }
    virtual void update_name()
    {
        #if TOOLS_ENABLED
        String nm = String(propertyName)  + " " + get_compare_type_name();
        {
            nm += " " + String::num(value);
        }
        set_name( nm );
        #endif
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
protected:
    double value;
    bool is_value_by_property = false;
    StringName value_property_name;

};
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

protected:
    void set_value(int32_t p_value)
    {
        value = p_value;
        update_name();
    }
    int32_t get_value()
    {
        return value;
    }
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
        #if TOOLS_ENABLED
        String nm = String(propertyName)  + " "+ get_compare_type_name();
        if(is_value_by_property)
        {
            nm += " " + String(value_property_name);
        }
        else
        {
            nm += " " + itos(value);
        }
        set_name( nm );
        #endif
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
        return Array();
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
    int64_t value;
    bool is_value_by_property = false;
    StringName value_property_name;

};
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

protected:
    void set_value(bool p_value)
    {
        value = p_value;
        update_name();
    }
    bool get_value()
    {
        return value;
    }

    virtual void update_name()
    {
        #if TOOLS_ENABLED
        String nm = String(propertyName)  + " " + get_compare_type_name();
        {
            nm += " ";
            nm += (value ? "true" : "false");
        }
        set_name( nm );
        #endif
    }
    virtual Array _get_compare_value() override
    {
        Array ret;
        ret.append(StringName("=="));
        ret.append(StringName("!="));
        return Array();
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

protected:
    void set_value(const StringName& p_value)
    {
        value = p_value;
        update_name();
    }
    StringName get_value()
    {
        return value;
    }

    virtual void update_name()
    {
        #if TOOLS_ENABLED
        String nm = String(propertyName)  + " " + get_compare_type_name();
        {
            nm += " " + value;
        }
        set_name( nm );
        #endif
    }
    virtual Array _get_compare_value() override
    {
        Array ret;
        ret.append(StringName("=="));
        ret.append(StringName("!="));
        return Array();
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
    StringName value;
};
// 角色动画的条件
class CharacterAnimatorConditionList : public RefCounted
{
    GDCLASS(CharacterAnimatorConditionList, RefCounted)
    static void _bind_methods()
    {
        ClassDB::bind_method(D_METHOD("set_conditions","conditions"),&CharacterAnimatorConditionList::set_conditions);
        ClassDB::bind_method(D_METHOD("get_conditions"),&CharacterAnimatorConditionList::get_conditions);

        ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "conditions", PROPERTY_HINT_ARRAY_TYPE,RESOURCE_TYPE_HINT("AnimatorAIStateConditionBase")), "set_conditions", "get_conditions");
    }
public:
    void set_conditions(const Array& p_conditions)
    {
        conditions.clear();
        for(int32_t i = 0; i < p_conditions.size(); ++i)
        {
            conditions.push_back(static_cast<Ref<AnimatorAIStateConditionBase>>(p_conditions[i]));
        }
        update_blackboard_plan();
    }
    Array get_conditions() { 
        Array ret;
        for(uint32_t i = 0; i < conditions.size(); ++i)
        {
            ret.push_back(conditions[i]);
        }
        return ret; 
    }
    
    virtual bool is_enable(Blackboard* p_blackboard,bool p_is_include)
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
    void set_blackboard_plan(const Ref<BlackboardPlan>& p_blackboard_plan) { blackboard_plan = p_blackboard_plan; update_blackboard_plan();}
    void update_blackboard_plan();

public:
    LocalVector<Ref<AnimatorAIStateConditionBase>> conditions;
    Ref<BlackboardPlan> blackboard_plan;
    
};
// 角色动画的条件
class CharacterAnimatorCondition : public RefCounted
{

    GDCLASS(CharacterAnimatorCondition, RefCounted)
    static void _bind_methods();
public :
    void set_include_condition(const Ref<CharacterAnimatorConditionList>& p_include_condition) 
    {
         include_condition = p_include_condition; 
         update_blackboard_plan();
    }
    Ref<CharacterAnimatorConditionList> get_include_condition() { return include_condition; }

    void set_exclude_condition(const Ref<CharacterAnimatorConditionList>& p_exclude_condition)
    {
         exclude_condition = p_exclude_condition; 
         update_blackboard_plan();
    }
    Ref<CharacterAnimatorConditionList> get_exclude_condition() { return exclude_condition; }
    void update_blackboard_plan()
    {
        if(include_condition.is_valid()){
            include_condition->set_blackboard_plan(blackboard_plan);
        }
        if(exclude_condition.is_valid()){
            exclude_condition->set_blackboard_plan(blackboard_plan);
        }
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
class CharacterAnimationLogicNode;
class CharacterAnimationLogicRoot : public RefCounted
{
    GDCLASS(CharacterAnimationLogicRoot,RefCounted)
    static void _bind_methods();
public:
    void sort();
    void set_node_list(const Array& p_node_list)
    {
        for(int32_t i = 0; i < p_node_list.size(); ++i)
        {
            node_list.push_back(p_node_list[i]);
        }
    }
    Array get_node_list() 
    { 
        Array ret;
        for(uint32_t i = 0; i < node_list.size(); ++i)
        {
            ret.push_back(node_list[i]);
        }
        return ret; 
    }

    void set_bt_sort(int id){}
    int get_bt_sort() { return 0; }

public:
    bool is_need_sort = true;
    LocalVector<Ref<CharacterAnimationLogicNode>>  node_list;

};
// 动画逻辑层信息
/*
    每一个层里面可以处理多个动画状态
    每一个状态都有一套动画配置
*/
class CharacterAnimationLogicLayer : public Resource
{
    GDCLASS(CharacterAnimationLogicLayer, Resource);
    static void _bind_methods()
    {
        ClassDB::bind_method(D_METHOD("set_default_state_name","p_default_state_name"),&CharacterAnimationLogicLayer::set_default_state_name);
        ClassDB::bind_method(D_METHOD("get_default_state_name"),&CharacterAnimationLogicLayer::get_default_state_name);

        ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "default_state_name"), "set_default_state_name", "get_default_state_name");
    }
public:

    void set_default_state_name(const StringName& p_default_state_name) { default_state_name = p_default_state_name; }
    StringName get_default_state_name() { return default_state_name; }


public:
    //  默认状态名称
    StringName default_state_name;
    HashMap<StringName, Ref<CharacterAnimationLogicRoot>> state_map;

};
VARIANT_ENUM_CAST(AnimatorAIStateConditionBase::AnimatorAICompareType)