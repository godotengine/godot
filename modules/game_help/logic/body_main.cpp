#include "body_main.h"


void CharacterBodyMain::_bind_methods()
{
    
	ClassDB::bind_method(D_METHOD("restart"), &CharacterBodyMain::restart);


	ClassDB::bind_method(D_METHOD("set_behavior_tree", "behavior_tree"), &CharacterBodyMain::set_behavior_tree);
	ClassDB::bind_method(D_METHOD("get_behavior_tree"), &CharacterBodyMain::get_behavior_tree);
	ClassDB::bind_method(D_METHOD("set_update_mode", "update_mode"), &CharacterBodyMain::set_update_mode);
	ClassDB::bind_method(D_METHOD("get_update_mode"), &CharacterBodyMain::get_update_mode);
	ClassDB::bind_method(D_METHOD("set_blackboard", "blackboard"), &CharacterBodyMain::set_blackboard);
	ClassDB::bind_method(D_METHOD("get_blackboard"), &CharacterBodyMain::get_blackboard);

	ClassDB::bind_method(D_METHOD("set_blackboard_plan", "plan"), &CharacterBodyMain::set_blackboard_plan);
	ClassDB::bind_method(D_METHOD("get_blackboard_plan"), &CharacterBodyMain::get_blackboard_plan);

    


    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "behavior_tree", PROPERTY_HINT_RESOURCE_TYPE, "BehaviorTree"), "set_behavior_tree", "get_behavior_tree");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "update_mode", PROPERTY_HINT_ENUM, "Idle,Physics,Manual"), "set_update_mode", "get_update_mode");
	
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "blackboard", PROPERTY_HINT_NONE, "Blackboard", 0), "set_blackboard", "get_blackboard");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "blackboard_plan", PROPERTY_HINT_RESOURCE_TYPE, "BlackboardPlan", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_EDITOR_INSTANTIATE_OBJECT), "set_blackboard_plan", "get_blackboard_plan");


	ADD_SIGNAL(MethodInfo("behavior_tree_finished", PropertyInfo(Variant::INT, "status")));
	ADD_SIGNAL(MethodInfo("behavior_tree_updated", PropertyInfo(Variant::INT, "status")));

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
void CharacterBodyMain::behavior_tree_finished(int last_status)
{
    emit_signal("behavior_tree_finished", last_status);
}
void CharacterBodyMain::behavior_tree_update(int last_status)
{
    emit_signal("updated", last_status);
}
BTPlayer * CharacterBodyMain::get_bt_player()
{
    if(btPlayer == nullptr)
    {
        btPlayer = memnew(BTPlayer);
        btPlayer->set_owner((Node*)this);
        btPlayer->set_name("BTPlayer");
        btPlayer->connect("behavior_tree_finished", callable_mp(this, &CharacterBodyMain::behavior_tree_finished));
        btPlayer->connect("updated", callable_mp(this, &CharacterBodyMain::behavior_tree_update));
        add_child(btPlayer);
    }
    return btPlayer;
}


void CharacterAnimatorNodeBase::_blend_anmation(CharacterAnimatorLayer *p_layer,int child_count,CharacterAnimationInstance *p_playback_info,float total_weight,const Vector<float> &weight_array)
{
    AnimationMixer::PlaybackInfo * p_playback_info_ptr = p_playback_info->m_ChildAnimationPlaybackArray.ptrw();
    for (uint32_t i = 0; i < child_count; i++)
    {
        float w = weight_array[i] * total_weight;
        if(w > 0.01f)
        {	  
            {
                p_playback_info_ptr[i].weight = w;
                p_playback_info_ptr[i].time = p_playback_info->time;
                p_playback_info_ptr[i].delta = p_playback_info->delta;
                p_playback_info_ptr[i].track_weights = p_playback_info->track_weights;
                p_layer->make_animation_instance(p_playback_info->m_ChildAnimationArray[i], p_playback_info_ptr[i]);
            }
        }
    }
}


void CharacterAnimatorNode1D::process_animation(class CharacterAnimatorLayer *p_layer,CharacterAnimationInstance *p_playback_info,float total_weight,Blackboard *p_blackboard,const StringName & property_name)
{
    if(!p_blackboard->has_var(property_name))
    {
        return;
    }
    float v = p_blackboard->get_var(property_name,0);
    if(p_playback_info->m_WeightArray.size() != m_BlendData.m_ChildCount)
    {
        p_playback_info->m_WeightArray.resize(m_BlendData.m_ChildCount);
        p_playback_info->m_ChildAnimationPlaybackArray.resize(m_BlendData.m_ChildCount);
        p_playback_info->m_ChildAnimationArray.resize(m_BlendData.m_ChildCount);

    }
    GetWeights1d(m_BlendData, p_playback_info->m_WeightArray.ptrw(), v);
    _blend_anmation(p_layer,m_BlendData.m_ChildCount, p_playback_info, total_weight,p_playback_info->m_WeightArray);

}
void CharacterAnimatorNode2D::process_animation(class CharacterAnimatorLayer *p_layer,CharacterAnimationInstance *p_playback_info,float total_weight,Blackboard *p_blackboard,const StringName & property_name)
{
    if(!p_blackboard->has_var(property_name))
    {
        return;
    }
    Vector2 v = p_blackboard->get_var(property_name,0);
    if(p_playback_info->m_WeightArray.size() != m_BlendData.m_ChildCount)
    {
        p_playback_info->m_WeightArray.resize(m_BlendData.m_ChildCount);
        p_playback_info->m_ChildAnimationPlaybackArray.resize(m_BlendData.m_ChildCount);
        p_playback_info->m_ChildAnimationArray.resize(m_BlendData.m_ChildCount);

    }
    if(p_layer->m_TempCropArray.size() < m_BlendData.m_ChildCount)
    {
        p_layer->m_TempCropArray.resize(m_BlendData.m_ChildCount);
        p_layer->m_ChildInputVectorArray.resize(m_BlendData.m_ChildCount);
    }
    if (m_BlendType == SimpleDirectionnal2D)
        GetWeightsSimpleDirectional(m_BlendData, p_playback_info->m_WeightArray.ptrw(), p_layer->m_TempCropArray.ptrw(), p_layer->m_ChildInputVectorArray.ptrw(), v.x, v.y);
    else if (m_BlendType == FreeformDirectionnal2D)
        GetWeightsFreeformDirectional(m_BlendData, p_playback_info->m_WeightArray.ptrw(), p_layer->m_TempCropArray.ptrw(), p_layer->m_ChildInputVectorArray.ptrw(), v.x, v.y);
    else if (m_BlendType == FreeformCartesian2D)
        GetWeightsFreeformCartesian(m_BlendData, p_playback_info->m_WeightArray.ptrw(), p_layer->m_TempCropArray.ptrw(), p_layer->m_ChildInputVectorArray.ptrw(), v.x, v.y);
    else 
        return;

    _blend_anmation(p_layer, m_BlendData.m_ChildCount,p_playback_info, total_weight,p_playback_info->m_WeightArray);

}
// 处理动画
void CharacterAnimatorLayer::_process_animation(double p_delta,float w,bool is_first)
{
    clear_animation_instances();

    // 播放列队的动画

    _blend_process(p_delta,false);


    
}







