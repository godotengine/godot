#include "body_animator_logic.h"
#include "body_animator.h"


void AnimatorAIStateConditionBase::set_blackboard_plan(const Ref<BlackboardPlan>& p_blackboard_plan) 
{
        blackboard_plan = p_blackboard_plan; 
}
void AnimatorAIStateConditionBase::set_compare_type_name(const StringName& p_type)
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
StringName AnimatorAIStateConditionBase::get_compare_type_name()
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
void AnimatorAIStateConditionBase::_bind_methods()
{
    
    ClassDB::bind_method(D_METHOD("get_compare_value"), &AnimatorAIStateConditionBase::get_compare_value);
    ClassDB::bind_method(D_METHOD("get_blackbord_propertys"), &AnimatorAIStateConditionBase::get_blackbord_propertys);

    ClassDB::bind_method(D_METHOD("set_compare_type_name", "type"), &AnimatorAIStateConditionBase::set_compare_type_name);
    ClassDB::bind_method(D_METHOD("get_compare_type_name"), &AnimatorAIStateConditionBase::get_compare_type_name);

    ClassDB::bind_method(D_METHOD("set_blackbord_property_name", "name"), &AnimatorAIStateConditionBase::set_blackbord_property_name);
    ClassDB::bind_method(D_METHOD("get_blackbord_property_name"), &AnimatorAIStateConditionBase::get_blackbord_property_name);

    ClassDB::bind_method(D_METHOD("set_compare_type", "type"), &AnimatorAIStateConditionBase::set_compare_type);
    ClassDB::bind_method(D_METHOD("get_compare_type"), &AnimatorAIStateConditionBase::get_compare_type);

    ClassDB::bind_method(D_METHOD("set_blackboard_plan", "plan"), &AnimatorAIStateConditionBase::set_blackboard_plan);
    ClassDB::bind_method(D_METHOD("get_blackboard_plan"), &AnimatorAIStateConditionBase::get_blackboard_plan);

    

    ClassDB::bind_method(D_METHOD("set_is_value_by_property","value"),&AnimatorAIStateFloatCondition::set_is_value_by_property);
    ClassDB::bind_method(D_METHOD("get_is_value_by_property"),&AnimatorAIStateFloatCondition::get_is_value_by_property);

    ClassDB::bind_method(D_METHOD("set_property_name","value"),&AnimatorAIStateFloatCondition::set_property_name);
    ClassDB::bind_method(D_METHOD("get_property_name"),&AnimatorAIStateFloatCondition::get_property_name);

    ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "compare_type_name", PROPERTY_HINT_ENUM_DYNAMIC_LIST,"get_compare_value"), "set_compare_type_name", "get_compare_type_name");
    ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "blackbord_property_name",PROPERTY_HINT_ENUM_DYNAMIC_LIST,"get_blackbord_propertys"), "set_blackbord_property_name", "get_blackbord_property_name");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "is_value_by_property"), "set_is_value_by_property", "get_is_value_by_property");
    ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "property_name",PROPERTY_HINT_ENUM_DYNAMIC_LIST,"get_blackbord_propertys"), "set_property_name", "get_property_name");

    //ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "blackboard_plan", PROPERTY_HINT_RESOURCE_TYPE, "BlackboardPlan"), "set_blackboard_plan", "get_blackboard_plan");
}



void CharacterAnimatorCondition::_bind_methods()
{
    ClassDB::bind_method(D_METHOD("set_include_condition", "include_condition"), &CharacterAnimatorCondition::set_include_condition);
    ClassDB::bind_method(D_METHOD("get_include_condition"), &CharacterAnimatorCondition::get_include_condition);

    ClassDB::bind_method(D_METHOD("set_exclude_condition", "exclude_condition"), &CharacterAnimatorCondition::set_exclude_condition);
    ClassDB::bind_method(D_METHOD("get_exclude_condition"), &CharacterAnimatorCondition::get_exclude_condition);

    ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "include_condition", PROPERTY_HINT_ARRAY_TYPE,RESOURCE_TYPE_HINT("AnimatorAIStateConditionBase")), "set_include_condition", "get_include_condition");
    ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "exclude_condition",PROPERTY_HINT_ARRAY_TYPE, RESOURCE_TYPE_HINT("AnimatorAIStateConditionBase")), "set_exclude_condition", "get_exclude_condition");

}
////////////////////////////////////////// CharacterAnimationLogicRoot /////////////////////////////////////////
void CharacterAnimationLogicRoot::sort()
{
    node_list.sort_custom<CharacterAnimationLogicNode::SortCharacterAnimationLogicNode>();
}

void CharacterAnimationLogicRoot::_bind_methods()
{
    ClassDB::bind_method(D_METHOD("set_node_list","node_list"),&CharacterAnimationLogicRoot::set_node_list);
    ClassDB::bind_method(D_METHOD("get_node_list"),&CharacterAnimationLogicRoot::get_node_list);

    ClassDB::bind_method(D_METHOD("set_bt_sort","id"),&CharacterAnimationLogicRoot::set_bt_sort);
    ClassDB::bind_method(D_METHOD("get_bt_sort"),&CharacterAnimationLogicRoot::get_bt_sort);

    ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "node_list", PROPERTY_HINT_ARRAY_TYPE,RESOURCE_TYPE_HINT("CharacterAnimationLogicNode")), "set_node_list", "get_node_list");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "bt_sort",PROPERTY_HINT_BUTTON,"#FF22AA;Sort;sort"), "set_bt_sort", "get_bt_sort");

}
Ref<CharacterAnimationLogicNode> CharacterAnimationLogicRoot::process_logic(Blackboard* blackboard)
{
    if(is_need_sort)
    {
        sort();
    }
    for(uint32_t i = 0; i < node_list.size(); ++i)
    {
        if(node_list[i].is_valid())
        {
            if(node_list[i]->is_enter(blackboard))
            {
                return node_list[i];
            }
        }
    }
    return Ref<CharacterAnimationLogicNode>();
}

Ref<CharacterAnimationLogicNode> CharacterAnimationLogicLayer::process_logic(StringName p_default_state_name,Blackboard* blackboard)
{
    if(state_map.has(p_default_state_name))
    {
        return state_map[p_default_state_name]->process_logic(blackboard);
    }
    return Ref<CharacterAnimationLogicNode>();
}