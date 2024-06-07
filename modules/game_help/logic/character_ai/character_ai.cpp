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