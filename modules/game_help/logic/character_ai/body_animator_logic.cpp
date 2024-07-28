#include "body_animator_logic.h"
#include "../animator/body_animator.h"



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