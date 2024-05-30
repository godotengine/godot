#include "character_movement.h"


void CharacterMovement::update()
{
    if(target.is_null()) return;
    Object *obj = ObjectDB::get_instance(target);

    Node3D *node = Object::cast_to<Node3D>(obj);
    if(node == nullptr) return;

    node->set_global_transform(world_pos);

}
const Transform3D& CharacterMovement::get_global_transform()
{
    return world_pos;
}
void CharacterMovement::set_attach_target(ObjectID id)
{
    if(id==attach_target) return;

    Transform3D old_world_pos = get_global_transform();
    detach_target();
    if(id.is_valid())
    {
        Object *obj = ObjectDB::get_instance(id);
        Node3D *node = Object::cast_to<Node3D>(obj);
        if(node)
        {
            node->connect(SceneStringName(tree_exiting),Callable(this,"on_attach_target_exit"));
            node->connect(SceneStringName(transform_changed),Callable(this,"on_attack_transform_change"));
            attach_world_pos = node->get_global_transform();
            local_rot = attach_world_pos.inverse() * world_pos;
            attach_target = id;
        }
    }

    attach_target = id;
}
void CharacterMovement::detach_target()
{
    if(attach_target.is_valid())
    {
        Object *obj = ObjectDB::get_instance(attach_target);
        Node3D *node = Object::cast_to<Node3D>(obj);
        if(node)
        {
            node->disconnect(SceneStringName(tree_exiting),Callable(this,"on_attach_target_exit"));
            node->disconnect(SceneStringName(transform_changed),Callable(this,"on_attack_transform_change"));
        }
    }
    attach_target = ObjectID();

}
void CharacterMovement::moveing(const Vector3& pos)
{
    if(!is_move)
    {
        return;
    }
    if(attach_target.is_null())
    {
        local_rot.origin += pos;
    }
    else
    {
        world_pos.origin += pos;
    }
    update();
}
void CharacterMovement::moveing_forward(float dis,bool is_ground )
{
    Vector3 forward = get_global_transform().basis.xform(Vector3(0,0,1));
    if(is_ground)
    {
        forward.y = 0;
        forward = forward.normalized();
    }
    moveing(forward*dis);
}
void CharacterMovement::looking(const Vector3& pos,bool is_ground )
{
    if(attach_target.is_valid())
    {
        Vector3 target_pos = pos;
        if(is_ground)
        {
            target_pos.y = world_pos.origin.y;
        }
        world_pos = world_pos.looking_at(pos,Vector3(0,1,0));
        local_rot = attach_world_pos.inverse() * world_pos;
    }
    else
    {
        world_pos = world_pos.looking_at(pos,Vector3(0,1,0));
    }
    update();
}
void CharacterMovement::moveing_right(float dis,bool is_ground )
{
    Vector3 right = get_global_transform().basis.xform(Vector3(1,0,0));
    if(is_ground)
    {
        right.y = 0;
        right = right.normalized();
    }
    moveing(right*dis);
}
void CharacterMovement::moveing_up(float dis)
{
    moveing(Vector3(0,1,0)*dis);
}


void CharacterMovement::on_attach_target_exit()
{
    Object *obj = ObjectDB::get_instance(attach_target);
    Node3D *node = Object::cast_to<Node3D>(obj);
    attach_world_pos = node->get_global_transform();
    world_pos = attach_world_pos * local_rot;
    attach_target = ObjectID();
}
void CharacterMovement::on_attack_transform_change()
{
    Object *obj = ObjectDB::get_instance(attach_target);
    Node3D *node = Object::cast_to<Node3D>(obj);
    attach_world_pos = node->get_global_transform();
    world_pos = attach_world_pos * local_rot;
}