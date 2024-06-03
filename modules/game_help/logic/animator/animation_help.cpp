#include "animation_help.h"
AnimationManager* AnimationManager::singleton = nullptr;

Ref<Animation> AnimationManager::getNullAnimation()
{
    static Ref<Animation>  nullAnimation = Ref<Animation>(memnew(Animation));
    return nullAnimation;

}
void AnimationManager::_bind_methods()
{
    ClassDB::bind_method(D_METHOD("add_animation_group", "group_name", "animation_tree_path","animation_library_path"), &AnimationManager::add_animation_group);
    ClassDB::bind_method(D_METHOD("add_animation", "group_name", "anim_name","animation_path"), &AnimationManager::add_animation);
    ClassDB::bind_method(D_METHOD("get_animation_nodes","path","is_single_valid","change_animName_to_sceneName"),&AnimationManager::get_animation_nodes);
    ClassDB::bind_method(D_METHOD("save_animation_tree","tree","path"),&AnimationManager::save_animation_tree);

    ClassDB::bind_method(D_METHOD("create_animation_library","animation","name","path"),&AnimationManager::create_animation_library);
    ClassDB::bind_method(D_METHOD("load_animation_tree","anim_parent_node","group_name"),&AnimationManager::load_animation_tree);


}