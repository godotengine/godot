#include "animation_help.h"
AnimationHelp* AnimationHelp::singleton = nullptr;

Ref<Animation> AnimationHelp::getNullAnimation()
{
    static Ref<Animation>  nullAnimation = Ref<Animation>(memnew(Animation));
    return nullAnimation;

}
void AnimationHelp::_bind_methods()
{
    ClassDB::bind_method(D_METHOD("add_animation_group", "group_name", "animation_tree_path","animation_library_path"), &AnimationHelp::add_animation_group);
    ClassDB::bind_method(D_METHOD("add_animation", "group_name", "anim_name","animation_path"), &AnimationHelp::add_animation);
    ClassDB::bind_method(D_METHOD("get_animation_nodes","path","is_single_valid","change_animName_to_sceneName"),&AnimationHelp::get_animation_nodes);
    ClassDB::bind_method(D_METHOD("save_animation_tree","tree","path"),&AnimationHelp::save_animation_tree);

    ClassDB::bind_method(D_METHOD("create_animation_library","animation","name","path"),&AnimationHelp::create_animation_library);
    ClassDB::bind_method(D_METHOD("load_animation_tree","anim_parent_node","group_name"),&AnimationHelp::load_animation_tree);


}