#include "character_ai.h"
#include "scene/3d/physics/collision_object_3d.h"

bool CharacterAI_CheckGround::_execute_check(CharacterBodyMain *node, Blackboard* blackboard)
{
    Vector3 start = node->get_global_position() + Vector3(0.0, check_move_height, 0.0);

    Vector3 end = start;
    end.y -= check_move_height + check_max_distance + ground_min_distance;
    PhysicsDirectSpaceState3D::RayParameters ray_params;
    ray_params.from = start;
    ray_params.to = end;
    ray_params.collision_mask = ground_mask;
    auto space_state = node->get_world_3d()->get_direct_space_state();
    if(space_state->intersect_ray(ray_params,result))
    {
        
        float dis = result.position.y - node->get_global_position().y - check_move_height;
        dis = MAX(0,dis);
        blackboard->set_var("is_ground",  dis <= ground_min_distance);
        blackboard->set_var("to_ground_distance", dis);
        blackboard->set_var("ground_pos",result.position);
        blackboard->set_var("ground_normal",result.normal);
        blackboard->set_var("ground_object_id",(int64_t)result.collider_id);
        CollisionObject3D * obj = Object::cast_to<CollisionObject3D>(result.collider);
        if(obj != nullptr)
            blackboard->set_var("ground_collider_layer",obj->get_collision_layer());
    }
    else
    {        
        blackboard->set_var("is_ground",  false);
    }
    return false;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool CharacterAI_CheckEnemy::_execute_check(CharacterBodyMain *node, Blackboard* blackboard)
{
    Ref<CharacterCheckArea3D> area = node->get_check_area_by_name(body_area_name);
    if(area.is_valid())
    {
        // 檢測敵人
        return true;
    }
    return false;
}

















void CharacterAI::execute(CharacterBodyMain *node,Blackboard* blackboard,CharacterAIContext* p_context)
{
    bool is_run_brain = false;
    if(p_context->logic_node.is_valid())
    {
        if(p_context->logic_node->execute(node,blackboard))
        {
            is_run_brain = true;
            p_context->logic_node->exit(node,blackboard);
            p_context->logic_node = Ref<CharacterAILogicNode>();
            p_context->logic_name = StringName();
        }
    }
    else
    {
        is_run_brain = true;
    }
    if(inductor.is_valid())
    {
        inductor->execute(node,blackboard);
    }

    if(is_run_brain && brain.is_valid())
    {
        brain->execute(node,blackboard);
        StringName logic_name = blackboard->get_var("ai/curr_logic_node_name",ident_node_name);
        if(logic_name != StringName() && logic_name != p_context->logic_name)
        {
            if(logic_nodes.has(logic_name))
            {
                if(p_context->logic_node.is_valid())
                {
                    p_context->logic_node->exit(node,blackboard);
                }
                p_context->logic_name = logic_name;
                p_context->logic_node = logic_nodes[logic_name];
                if(p_context->logic_node.is_valid())
                {
                    p_context->logic_node->enter(node,blackboard);
                }
                return;
            }
        }
        
    }
    // 没有判断出任何状态,强制进入休闲状态
    if(p_context->logic_node.is_null())
    {
        p_context->logic_name = ident_node_name;
        if(logic_nodes.has(ident_node_name))
        {
            p_context->logic_node = logic_nodes[ident_node_name];
            if(p_context->logic_node.is_valid())
            {
                p_context->logic_node->enter(node,blackboard);
            }
        }
    }

}
void CharacterAI::_bind_methods()
{
    ClassDB::bind_method(D_METHOD("set_inductor", "inductor"), &CharacterAI::set_inductor);
    ClassDB::bind_method(D_METHOD("get_inductor"), &CharacterAI::get_inductor);

    ClassDB::bind_method(D_METHOD("set_brain", "brain"), &CharacterAI::set_brain);
    ClassDB::bind_method(D_METHOD("get_brain"), &CharacterAI::get_brain);

    ClassDB::bind_method(D_METHOD("set_logic_node", "logic_node"), &CharacterAI::set_logic_node);
    ClassDB::bind_method(D_METHOD("get_logic_node"), &CharacterAI::get_logic_node);

    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "inductor", PROPERTY_HINT_RESOURCE_TYPE, "CharacterAI_Inductor"), "set_inductor", "get_inductor");
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "brain", PROPERTY_HINT_RESOURCE_TYPE, "CharacterAI_Brain"), "set_brain", "get_brain");
    ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "logic_node", PROPERTY_HINT_ARRAY_TYPE, MAKE_RESOURCE_TYPE_HINT("CharacterAILogicNode")), "set_logic_node", "get_logic_node");
    
}
