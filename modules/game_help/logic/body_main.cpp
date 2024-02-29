#include "body_main.h"


void CharacterBodyMain::_bind_methods()
{
    
}

// 初始化身體
void CharacterBodyMain::init_main_body(String p_mesh_file_path,StringName p_animation_group)
{
    if(root)
    {
        memdelete(root);
        root = nullptr;
        skeleton = nullptr;
        player = nullptr;
        tree = nullptr;
        
    }

    Ref<PackedScene> scene = ResourceLoader::load(p_mesh_file_path);
    if(!scene.is_valid())
    {
        return ;
    }

    root = Object::cast_to<Node3D>(scene->instantiate()); 
    add_child(root);
    skeleton = Object::cast_to<Skeleton3D>(root->get_node(NodePath("Skeleton3D")));

    // 配置动画信息
    AnimationHelp::setup_animation_tree(root,p_animation_group);

    player = Object::cast_to<AnimationPlayer>( root->get_node(NodePath("AnimationPlayer")));
    tree = Object::cast_to<AnimationTree>( root->get_node(NodePath("AnimationTree")));

}