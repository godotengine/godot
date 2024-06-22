#include "character_check_area_3d.h"
#include "body_main.h"


void CharacterCheckArea3D::set_body_main(class CharacterBodyMain* p_mainBody)
{
    if(mainBody == p_mainBody)
    {
        return;
    }
    Area3D * areaCollision = nullptr;
    if(mainBody && areaCollisionID.is_valid())
    {
        areaCollision = Object::cast_to<Area3D>(ObjectDB::get_instance(areaCollisionID));
        if(areaCollision)
        {
            areaCollision->disconnect(SceneStringName(body_entered),callable_mp(this,&CharacterCheckArea3D::on_body_enter_area));
            areaCollision->disconnect(SceneStringName(body_exited),callable_mp(this,&CharacterCheckArea3D::on_body_exit_area));
            areaCollision->queue_free();
        }
        areaCollisionID = ObjectID();

    }
    mainBody = p_mainBody;
    if(mainBody)
    {        
        areaCollision = memnew(Area3D);    
        mainBody->add_child(areaCollision);
        areaCollision->set_owner(mainBody);
        areaCollision->connect(SceneStringName(body_entered),callable_mp(this,&CharacterCheckArea3D::on_body_enter_area));
        areaCollision->connect(SceneStringName(body_exited),callable_mp(this,&CharacterCheckArea3D::on_body_exit_area));
        areaCollision->set_collision_layer(mainBody->get_collision_layer());
        areaCollision->set_collision_mask(collision_check_mask);
        areaCollisionID = areaCollision->get_instance_id();
    }
    if(area_shape.is_valid())
    {
        area_shape->set_link_target(areaCollision);
    }

}
void CharacterCheckArea3D::update_coord()
{
    if(!is_update_coord)
    {
        return;
    }
    boundOtherCharacterByCoord.clear();
    for(auto& node : boundOtherCharacter)
    {
        auto pos = world_pos_to_cell_pos(node.key->get_global_position());
        if(boundOtherCharacterByCoord.has(pos))
        {
            boundOtherCharacterByCoord[pos].push_back(node.value);
        }
        else
        {
            LocalVector<Ref<CharacterCheckArea3DResult>> nodes;
            nodes.push_back(node.value);
            boundOtherCharacterByCoord.insert(pos,nodes);   
        }
    }
    is_update_coord = false;
}

    
void CharacterCheckArea3D::set_area_shape(Ref<CollisionObject3DConnection> p_shape)
{
    if(area_shape == p_shape)
    {
        return;
    }
    if(area_shape.is_valid())
    {
        area_shape->set_link_target(nullptr);
    }
    area_shape = p_shape;
    if(mainBody && areaCollisionID.is_valid() && area_shape.is_valid())
    {
        Area3D* areaCollision = Object::cast_to<Area3D>(ObjectDB::get_instance(areaCollisionID));
        area_shape->set_link_target(areaCollision);
    }
}

void CharacterCheckArea3D::get_bound_other_character_by_angle(TypedArray<CharacterCheckArea3DResult>& _array,float angle)
{
    _array.clear();
    Vector3 curr_pos = mainBody->get_global_transform().origin;
    Vector3 forward = mainBody->get_global_transform().basis.xform(Vector3(0,0,1));
    float ref_angle = Math_PI * 0.5 * angle;
    for(auto& node : boundOtherCharacter)
    {
        node.value->update(curr_pos,forward);
        if(node.value->angle < ref_angle)
        {
            _array.push_back(node.value);
        }
    }
}