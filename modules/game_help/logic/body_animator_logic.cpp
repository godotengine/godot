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


////////////////////////////////////////// CharacterAnimationLogicRoot /////////////////////////////////////////
void CharacterAnimationLogicRoot::sort()
{
    node_list.sort_custom<CharacterAnimationLogicNode::SortCharacterAnimationLogicNode>();
}

//////////////////////////////////////////////// CharacterAnimationLogicNode /////////////////////////////////////////
void CharacterAnimationLogicNode::process(CharacterAnimatorLayer* animator,Blackboard* blackboard, double delta)
{
    if (GDVIRTUAL_IS_OVERRIDDEN(_animation_process)) {
        GDVIRTUAL_CALL(_animation_process, animator,blackboard, delta);
        return ;
    }

}
bool CharacterAnimationLogicNode::check_stop(CharacterAnimatorLayer* animator,Blackboard* blackboard)
{
    if (GDVIRTUAL_IS_OVERRIDDEN(_check_stop)) {
        bool is_stop = false;
        GDVIRTUAL_CALL(_check_stop, animator,blackboard, is_stop);
        return is_stop;
    }
    return true;
}