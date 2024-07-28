#pragma once
#include "core/object/ref_counted.h"
#include "modules/limboai/bt/bt_player.h"


// 继承Resource目的主要是为了编辑时方便显示名称
class AnimatorBlackboardSetItemBase : public RefCounted
{
    GDCLASS(AnimatorBlackboardSetItemBase, RefCounted)
    static void _bind_methods()
    {
        ClassDB::bind_method(D_METHOD("set_property_name", "property_name"), &AnimatorBlackboardSetItemBase::set_property_name);
        ClassDB::bind_method(D_METHOD("get_property_name"), &AnimatorBlackboardSetItemBase::get_property_name);

        ClassDB::bind_method(D_METHOD("set_blackboard_plan", "blackboard_plan"), &AnimatorBlackboardSetItemBase::set_blackboard_plan);
        ClassDB::bind_method(D_METHOD("get_blackboard_plan"), &AnimatorBlackboardSetItemBase::get_blackboard_plan);

        ClassDB::bind_method(D_METHOD("set_is_value_by_property", "is_value_by_property"), &AnimatorBlackboardSetItemBase::set_is_value_by_property);
        ClassDB::bind_method(D_METHOD("get_is_value_by_property"), &AnimatorBlackboardSetItemBase::get_is_value_by_property);

        ClassDB::bind_method(D_METHOD("set_value_property_name", "value_property_name"), &AnimatorBlackboardSetItemBase::set_value_property_name);
        ClassDB::bind_method(D_METHOD("get_value_property_name"), &AnimatorBlackboardSetItemBase::get_value_property_name);

        ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "property_name"), "set_property_name", "get_property_name");

        ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "blackboard_plan", PROPERTY_HINT_RESOURCE_TYPE, "BlackboardPlan"), "set_blackboard_plan", "get_blackboard_plan");

        ADD_PROPERTY(PropertyInfo(Variant::BOOL, "is_value_by_property"), "set_is_value_by_property", "get_is_value_by_property");

        ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "value_property_name"), "set_value_property_name", "get_value_property_name");
    }

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
    virtual void update_name()
    {

    }

    void set_blackbord_property_name(const StringName& p_name) { propertyName = p_name; update_name();}
    StringName get_blackbord_property_name() { return propertyName; }


    void set_blackboard_plan(const Ref<BlackboardPlan>& p_blackboard_plan)
    {
        blackboard_plan = p_blackboard_plan;
    }
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

    Array get_blackbord_propertys()
    {
        return _get_blackbord_propertys();
    }
    virtual Array _get_blackbord_propertys()
    {
        return Array();
    }
    virtual void _execute(Blackboard* blackboard)
    {

    }
protected:
    Ref<BlackboardPlan> blackboard_plan;
    StringName propertyName;
    bool is_value_by_property = false;
    StringName value_property_name;

};

// 黑板设置
class AnimatorBlackbordSet : public RefCounted
{
    GDCLASS(AnimatorBlackbordSet, RefCounted)
    static void _bind_methods()
    {
        ClassDB::bind_method(D_METHOD("set_change_list", "change_list"), &AnimatorBlackbordSet::set_change_list);
        ClassDB::bind_method(D_METHOD("get_change_list"), &AnimatorBlackbordSet::get_change_list);

        ClassDB::bind_method(D_METHOD("set_blackboard_plan", "blackboard_plan"), &AnimatorBlackbordSet::set_blackboard_plan);
        ClassDB::bind_method(D_METHOD("get_blackboard_plan"), &AnimatorBlackbordSet::get_blackboard_plan);

        ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "change_list", PROPERTY_HINT_ARRAY_TYPE,RESOURCE_TYPE_HINT("AnimatorBlackboardSetItemBase")), "set_change_list", "get_change_list");
        ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "blackboard_plan", PROPERTY_HINT_RESOURCE_TYPE, "BlackboardPlan"), "set_blackboard_plan", "get_blackboard_plan");
    }
public:
    void set_change_list(TypedArray<Ref<AnimatorBlackboardSetItemBase>> p_list) 
    {
        change_list.clear();
        for(int i=0;i<p_list.size();i++)
            change_list.push_back(p_list[i]);
    }

    TypedArray<AnimatorBlackboardSetItemBase> get_change_list() 
    { 
        TypedArray<Ref<AnimatorBlackboardSetItemBase>> rs;
        for(uint32_t i = 0; i < change_list.size(); ++i)
        {
            rs.push_back(change_list[i]);
        }
        return rs;
    }

    void set_blackboard_plan(const Ref<BlackboardPlan>& p_blackboard_plan)
    {
        blackboard_plan = p_blackboard_plan;
        update_blackboard_plan();
    }
    Ref<BlackboardPlan> get_blackboard_plan() { return blackboard_plan; }

    void update_blackboard_plan()
    {
        for(uint32_t i = 0; i < change_list.size(); ++i)
        {
            if (change_list[i].is_valid())
            {
                change_list[i]->set_blackboard_plan(blackboard_plan);
            }
        }
    }
    void execute(Blackboard* blackboard)
    {
        for(uint32_t i = 0; i < change_list.size(); ++i)
        {
            if (change_list[i].is_valid())
            {
                change_list[i]->_execute(blackboard);
            }
        }
    }
    void add_item(const Ref<AnimatorBlackboardSetItemBase>& p_exclude_condition)
    {
        change_list.push_back(p_exclude_condition);
        update_blackboard_plan();
    }
    void remove_item(const Ref<AnimatorBlackboardSetItemBase>& p_exclude_condition)
    {
        for(uint32_t i = 0; i < change_list.size(); ++i)
        {
            if(change_list[i] == p_exclude_condition)
            {
                change_list.remove_at(i);
                break;
            }
        }
    }
protected:
    LocalVector<Ref<AnimatorBlackboardSetItemBase>> change_list;
    Ref<BlackboardPlan> blackboard_plan;
};
