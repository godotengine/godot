#include "body_animator_logic.h"
#include "body_animator.h"


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

    ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "compare_type_name", PROPERTY_HINT_ENUM_DYNAMIC_LIST,"get_compare_value"), "set_compare_type_name", "get_compare_type_name");
    ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "blackbord_property_name",PROPERTY_HINT_ENUM_DYNAMIC_LIST,"get_blackbord_propertys"), "set_blackbord_property_name", "get_blackbord_property_name");
}



void CharacterAnimatorCondition::_bind_methods()
{
    ClassDB::bind_method(D_METHOD("set_include_condition", "include_condition"), &CharacterAnimatorCondition::set_include_condition);
    ClassDB::bind_method(D_METHOD("get_include_condition"), &CharacterAnimatorCondition::get_include_condition);

    ClassDB::bind_method(D_METHOD("set_exclude_condition", "exclude_condition"), &CharacterAnimatorCondition::set_exclude_condition);
    ClassDB::bind_method(D_METHOD("get_exclude_condition"), &CharacterAnimatorCondition::get_exclude_condition);

    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "include_condition", PROPERTY_HINT_RESOURCE_TYPE, "CharacterAnimatorConditionList"), "set_include_condition", "get_include_condition");
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "exclude_condition",PROPERTY_HINT_RESOURCE_TYPE, "CharacterAnimatorConditionList"), "set_exclude_condition", "get_exclude_condition");

}
////////////////////////////////////////// CharacterAnimationLogicRoot /////////////////////////////////////////
void CharacterAnimationLogicRoot::sort()
{
    node_list.sort_custom<CharacterAnimationLogicNode::SortCharacterAnimationLogicNode>();
}