#include "body_main.h"


void CharacterBodyMain::_bind_methods()
{
    
}

void CharacterBodyMain::clear_all()
{
    if(skeleton)
    {
        memdelete(skeleton);
        skeleton = nullptr;
        
    }
    if(player)
    {
        memdelete(player);
        player = nullptr;
        
    }
    if(tree)
    {
        memdelete(tree);
        tree = nullptr;
    }
}
// 初始化身體
void CharacterBodyMain::init_main_body(String p_skeleton_file_path,StringName p_animation_group)
{
    skeleton_res = p_skeleton_file_path;
    animation_group = p_animation_group;

}
void CharacterBodyMain::load_skeleton()
{
    clear_all();
    Ref<PackedScene> scene = ResourceLoader::load(skeleton_res);
    if(!scene.is_valid())
    {
        return ;
    }
    Node* ins = scene->instantiate(PackedScene::GEN_EDIT_STATE_INSTANCE);
    if (ins == nullptr) {
        return;
    }
    skeleton = Object::cast_to<Skeleton3D>(ins); 
    skeleton->set_owner(this);
    if(skeleton == nullptr)
    {
        memdelete(ins);
        return ;
    }
    skeleton->set_name("Skeleton3D");

    add_child(skeleton);

    // 配置动画信息
    AnimationHelp::setup_animation_tree(this,animation_group);

    player = Object::cast_to<AnimationPlayer>( get_node(NodePath("AnimationPlayer")));
    tree = Object::cast_to<AnimationTree>( get_node(NodePath("AnimationTree")));

}
void CharacterBodyMain::load_mesh(const StringName& part_name,String p_mesh_file_path)
{
    auto old_ins = bodyPart.find(part_name);
    if(old_ins != bodyPart.end())
    {
        old_ins->value.clear();
        bodyPart.remove(old_ins);
    }
    Ref<CharacterBodyPart> mesh = ResourceLoader::load(p_mesh_file_path);
    if(!mesh.is_valid())
    {
        return;
    }
    CharacterBodyPartInstane ins;
    ins.init(this,mesh,skeleton);
    bodyPart.insert(mesh->get_name(),ins);
}