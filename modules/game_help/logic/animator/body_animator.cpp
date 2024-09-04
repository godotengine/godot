#include "body_animator.h"
#include "../body_main.h"
#include "../data_table_manager.h"
#include "core/io/json.h"
#include "modules/realtime_retarget/src/retarget_utility.h"

#include "../../unity/unity_animation_import.h"





void CharacterAnimatorLayer::_process_logic(const Ref<Blackboard>& p_playback_info, double p_delta, bool is_first)
{
    if(logic_context.animation_logic.is_null())
    {
        return;
    }
    if(logic_context.curr_name == StringName() )
    {
        logic_context.curr_name = logic_context.animation_logic->get_default_state_name();
    }
    Ref<CharacterAnimationLogicNode>   curr_logic = logic_context.curr_logic;


    bool change_state = false;
    if(curr_logic.is_null() || logic_context.curr_name != logic_context.last_name)
    {
        change_state = true;
    }

    else if(curr_logic.is_valid())
    {
        if(curr_logic->check_stop(this,*p_playback_info))
        {
            change_state = true;
        }
    }

    if(change_state)
    {
        curr_logic = logic_context.animation_logic->process_logic(logic_context.curr_name, *p_playback_info);
    }
    if(curr_logic.is_null())
    {
        return;
    }
    if(curr_logic != logic_context.curr_logic)
    {
        logic_context.curr_logic->process_stop(this,*p_playback_info);
        logic_context.curr_logic = curr_logic;
        logic_context.curr_logic->process_start(this,*p_playback_info);
    }
    logic_context.last_name = logic_context.curr_name;
    curr_logic->process(this,*p_playback_info,p_delta);
}
// 处理动画
void CharacterAnimatorLayer::_process_animator(const Ref<Blackboard> &p_playback_info,double p_delta,bool is_first)
{
    // 处理逻辑节点请求播放的动作
    if(logic_context.curr_animation.is_valid())
    {
        auto anim = logic_context.curr_animation->get_node();
        if(play_animation(anim))
        {
            logic_context.curr_animation_time_length = anim->_get_animation_length();
            logic_context.curr_animation_play_time = 0.0f;
        }
    }
    logic_context.curr_animation_play_time += p_delta;
    
    for(auto& anim : m_AnimationInstances)
    {

        anim.delta = p_delta;
        anim.time += p_delta;
        if(anim.m_PlayState == CharacterAnimationInstance::PS_FadeOut)
        {
            anim.fadeTotalTime += p_delta;
        }
    }
    auto it = m_AnimationInstances.begin();
    float total_weight = 0.0f;
    while(it != m_AnimationInstances.end())
    {
        if(it->get_weight() == 0.0f)
        {
            it = m_AnimationInstances.erase(it);
        }
        else
        {
            total_weight += it->get_weight();
            ++it;
        }
    }
    
    if(m_TotalAnimationWeight.size() < m_AnimationInstances.size())
    {
        m_TotalAnimationWeight.resize(m_AnimationInstances.size());
    }
    for(auto& anim : m_AnimationInstances)
    {
        anim.node->process_animation(this, &anim, it->get_weight() / total_weight, p_playback_info);
    }

}
// 处理动画
void CharacterAnimatorLayer::_process_animation(const Ref<Blackboard> &p_playback_info,double p_delta,bool is_first)
{


	_blend_init();

	cb_begin_animation.call(this, p_delta, false);

	if (_blend_pre_process(p_delta, track_count, track_map)) {
		_blend_capture(p_delta);
		_blend_calc_total_weight();
		_blend_process(p_delta, false);
        // 混合
		layer_blend_apply();
		_blend_post_process();

		cb_end_animation.call(this, p_delta, false);
	};

    
	clear_animation_instances();
}

void CharacterAnimatorLayer::finish_update()
{
    
}

void CharacterAnimatorLayer::layer_blend_apply() {
	// Finally, set the tracks.
	for (const KeyValue<Animation::TypeHash, TrackCache *> &K : track_cache) {
		TrackCache *track = K.value;
		if (!deterministic && Math::is_zero_approx(track->total_weight)) {
			continue;
		}
		switch (track->type) {
			case Animation::TYPE_POSITION_3D: {
#ifndef _3D_DISABLED
				TrackCacheTransform *t = static_cast<TrackCacheTransform *>(track);

				if (t->root_motion) {
					root_motion_position = root_motion_cache.loc;
					root_motion_rotation = root_motion_cache.rot;
					root_motion_scale = root_motion_cache.scale - Vector3(1, 1, 1);
					root_motion_position_accumulator = t->loc;
					root_motion_rotation_accumulator = t->rot;
					root_motion_scale_accumulator = t->scale;
				} else if (t->skeleton_id.is_valid() && t->bone_idx >= 0) {
					Skeleton3D *t_skeleton = Object::cast_to<Skeleton3D>(ObjectDB::get_instance(t->skeleton_id));
					if (!t_skeleton) {
						return;
					}
					if (t->loc_used) {
                        if(config->get_blend_type() == CharacterAnimatorLayerConfig::BT_Blend)
                        {
						    t_skeleton->set_bone_pose_position(t->bone_idx, t_skeleton->get_bone_pose_position(t->bone_idx).lerp(t->loc,blend_weight));
                        }
                        else
                        {
                            t_skeleton->set_bone_pose_position(t->bone_idx, t->loc);
                        }
					}
					if (t->rot_used) {
                        if(config->get_blend_type() == CharacterAnimatorLayerConfig::BT_Blend)
                        {
                            t_skeleton->set_bone_pose_rotation(t->bone_idx, t_skeleton->get_bone_pose_rotation(t->bone_idx).slerp(t->rot,blend_weight));
                        }
                        else
                        {
                            t_skeleton->set_bone_pose_rotation(t->bone_idx, t->rot);
                        }                        
					}
					if (t->scale_used) {
                        if(config->get_blend_type() == CharacterAnimatorLayerConfig::BT_Blend)
                        {
                            t_skeleton->set_bone_pose_scale(t->bone_idx, t_skeleton->get_bone_pose_scale(t->bone_idx).lerp(t->scale,blend_weight));
                        }
                        else
                        {
                            t_skeleton->set_bone_pose_scale(t->bone_idx, t->scale);
                        }
					}

				} else if (!t->skeleton_id.is_valid()) {
					Node3D *t_node_3d = Object::cast_to<Node3D>(ObjectDB::get_instance(t->object_id));
					if (!t_node_3d) {
						return;
					}
					if (t->loc_used) {
                        if(config->get_blend_type() == CharacterAnimatorLayerConfig::BT_Blend)
                        {
                            t_node_3d->set_position(t_node_3d->get_position().lerp(t->loc,blend_weight));
                        }
                        else
                        {
                            t_node_3d->set_position(t->loc);
                        }
					}
					if (t->rot_used) {
                        if(config->get_blend_type() == CharacterAnimatorLayerConfig::BT_Blend)
                        {
                            t_node_3d->set_rotation(t_node_3d->get_rotation().slerp(t->rot.get_euler(),blend_weight));
                        }
                        else
                        {
                            t_node_3d->set_rotation(t->rot.get_euler());
                        }
					}
					if (t->scale_used) {
                        if(config->get_blend_type() == CharacterAnimatorLayerConfig::BT_Blend)
                        {
                            t_node_3d->set_scale(t_node_3d->get_scale().lerp(t->scale,blend_weight));
                        }
                        else
                        {                            
                            t_node_3d->set_scale(t->scale);
                        }
					}
				}
#endif // _3D_DISABLED
			} break;
			case Animation::TYPE_BLEND_SHAPE: {
#ifndef _3D_DISABLED
				TrackCacheBlendShape *t = static_cast<TrackCacheBlendShape *>(track);

				MeshInstance3D *t_mesh_3d = Object::cast_to<MeshInstance3D>(ObjectDB::get_instance(t->object_id));
				if (t_mesh_3d) {
                    if(config->get_blend_type() == CharacterAnimatorLayerConfig::BT_Blend)
                    {
                        t_mesh_3d->set_blend_shape_value(t->shape_index, Math::lerp( t_mesh_3d->get_blend_shape_value(t->shape_index),t->value,blend_weight));
                    }
                    else
                    {
                        t_mesh_3d->set_blend_shape_value(t->shape_index, t->value);
                    }
				}
#endif // _3D_DISABLED
			} break;
			case Animation::TYPE_VALUE: {
				TrackCacheValue *t = static_cast<TrackCacheValue *>(track);

				if (!t->is_variant_interpolatable || (callback_mode_discrete == ANIMATION_CALLBACK_MODE_DISCRETE_DOMINANT && t->use_discrete)) {
					break; // Don't overwrite the value set by UPDATE_DISCRETE.
				}

				// Trim unused elements if init array/string is not blended.
				if (t->value.is_array()) {
					int actual_blended_size = (int)Math::round(Math::abs(t->element_size.operator real_t()));
					if (actual_blended_size < (t->value.operator Array()).size()) {
						real_t abs_weight = Math::abs(track->total_weight);
						if (abs_weight >= 1.0) {
							(t->value.operator Array()).resize(actual_blended_size);
						} else if (t->init_value.is_string()) {
							(t->value.operator Array()).resize(Animation::interpolate_variant((t->init_value.operator String()).length(), actual_blended_size, abs_weight));
						}
					}
				}

				Object *t_obj = ObjectDB::get_instance(t->object_id);
				if (t_obj) {
					t_obj->set_indexed(t->subpath, Animation::cast_from_blendwise(t->value, t->init_value.get_type()));
				}

			} break;
			case Animation::TYPE_AUDIO: {
				TrackCacheAudio *t = static_cast<TrackCacheAudio *>(track);

				// Audio ending process.
				LocalVector<ObjectID> erase_maps;
				for (KeyValue<ObjectID, PlayingAudioTrackInfo> &L : t->playing_streams) {
					PlayingAudioTrackInfo &track_info = L.value;
					float db = Math::linear_to_db(track_info.use_blend ? track_info.volume : 1.0);
					LocalVector<int> erase_streams;
					HashMap<int, PlayingAudioStreamInfo> &map = track_info.stream_info;
					for (const KeyValue<int, PlayingAudioStreamInfo> &M : map) {
						PlayingAudioStreamInfo pasi = M.value;

						bool stop = false;
						if (!t->audio_stream_playback->is_stream_playing(pasi.index)) {
							stop = true;
						}
						if (!track_info.loop) {
							if (!track_info.backward) {
								if (track_info.time < pasi.start) {
									stop = true;
								}
							} else if (track_info.backward) {
								if (track_info.time > pasi.start) {
									stop = true;
								}
							}
						}
						if (pasi.len > 0) {
							double len = 0.0;
							if (!track_info.backward) {
								len = pasi.start > track_info.time ? (track_info.length - pasi.start) + track_info.time : track_info.time - pasi.start;
							} else {
								len = pasi.start < track_info.time ? (track_info.length - track_info.time) + pasi.start : pasi.start - track_info.time;
							}
							if (len > pasi.len) {
								stop = true;
							}
						}
						if (stop) {
							// Time to stop.
							t->audio_stream_playback->stop_stream(pasi.index);
							erase_streams.push_back(M.key);
						} else {
							t->audio_stream_playback->set_stream_volume(pasi.index, db);
						}
					}
					for (uint32_t erase_idx = 0; erase_idx < erase_streams.size(); erase_idx++) {
						map.erase(erase_streams[erase_idx]);
					}
					if (map.size() == 0) {
						erase_maps.push_back(L.key);
					}
				}
				for (uint32_t erase_idx = 0; erase_idx < erase_maps.size(); erase_idx++) {
					t->playing_streams.erase(erase_maps[erase_idx]);
				}
			} break;
			default: {
			} // The rest don't matter.
		}
	}
}
void CharacterAnimatorLayer::play_animation(const Ref<Animation>& p_anim, bool p_is_loop)
{
	Ref<CharacterAnimatorNode1D> anim_node;
	anim_node.instantiate();
}

bool CharacterAnimatorLayer::play_animation(Ref<CharacterAnimatorNodeBase> p_node)
{
    if(p_node.is_null())
    {
        return false;
    }
    if(m_AnimationInstances.size() > 0 && m_AnimationInstances.back()->get().node == p_node)
    {
        return false;
    }
    p_node->_init();
    for(auto& anim : m_AnimationInstances)
    {
        anim.m_PlayState = CharacterAnimationInstance::PS_FadeOut;
    }
    CharacterAnimationInstance ins;
    ins.node = p_node;
    ins.m_PlayState = CharacterAnimationInstance::PS_Play;
    if(config.is_valid() && config->get_mask().is_valid())
    {
        ins.disable_path = config->get_mask()->disable_path;
    }
    m_AnimationInstances.push_back(ins);
    return true;
}


void CharacterAnimatorLayer::play_animation(const StringName& p_node_name)
{
    if(m_Animator == nullptr)
    {
        return;
    }
    logic_context.curr_animation = m_Animator->get_animation_by_name(p_node_name);
}

CharacterAnimatorLayer::~CharacterAnimatorLayer()
{
    if(m_Animator != nullptr)
        m_Animator->on_layer_delete(this);
}

void CharacterAnimatorLayerConfigInstance::editor_play_animation()
{
    if(layer == nullptr)
    {
        return;
    }
    layer->play_animation(play_animation,true);
}

void CharacterAnimatorLayerConfigInstance::set_body(class CharacterBodyMain* p_body)
{
	if (p_body == m_Body)
	{
		return;
	}
	m_Body = p_body;
	if (layer)
	{
		layer->queue_free();
		layer = nullptr;
	}
	auto_init();
}
void CharacterAnimatorLayerConfigInstance::auto_init()
{

	if (m_Body == nullptr || config.is_null())
	{
		return;
	}
	layer = memnew(CharacterAnimatorLayer);
	m_Body->add_child(layer);
	layer->set_owner(m_Body);
	layer->init(m_Body->get_animator().ptr(), config);
}
///////////////

void CharacterAnimator::set_body(class CharacterBodyMain* p_body)
{
     m_Body = p_body;
	 auto it = m_LayerConfigInstanceList.begin();
	 bool is_first = true;
	 while (it != m_LayerConfigInstanceList.end())
	 {
		 Ref< CharacterAnimatorLayerConfigInstance> layer = *it;
		 layer->set_body(m_Body);
		 ++it;
	 }
}

void CharacterAnimator::add_layer(const Ref<CharacterAnimatorLayerConfig>& _mask)
{
    if(_mask.is_null())
    {
        return ;
    }
	Ref< CharacterAnimatorLayerConfigInstance> ins;
	ins.instantiate();
	ins->set_config(_mask);
	ins->set_body(m_Body);
	m_LayerConfigInstanceList.push_back(ins);
}
void CharacterAnimator::update_animator(float delta)
{
    auto it = m_LayerConfigInstanceList.begin();
    bool is_first = true;
    while(it!= m_LayerConfigInstanceList.end())
    {
		Ref< CharacterAnimatorLayerConfigInstance> layer = *it;
        layer->_process_animator(m_Body->get_blackboard(),delta,is_first);
        is_first = false;
        ++it;
    }

}
void CharacterAnimator::update_animation(float delta) {
    auto it = m_LayerConfigInstanceList.begin();
    bool is_first = true;
    while(it!= m_LayerConfigInstanceList.end()) {
		Ref< CharacterAnimatorLayerConfigInstance> layer = *it;
        layer->_process_animation(m_Body->get_blackboard(),delta,is_first);
        is_first = false;
        ++it;
    }
}
void CharacterAnimator::finish_update()
{
    auto it = m_LayerConfigInstanceList.begin();
    bool is_first = true;
    while(it!= m_LayerConfigInstanceList.end())
    {
		Ref< CharacterAnimatorLayerConfigInstance> layer = *it;
		layer->finish_update();
		is_first = false;
        ++it;
    }

}
Ref<CharacterAnimationLibrary::AnimationItem> CharacterAnimator::get_animation_by_name(const StringName& p_name)
{
    if(m_Body == nullptr)
    {
        return Ref<CharacterAnimationLibrary::AnimationItem>();
    }
    auto anim_lib = m_Body->get_animation_library();
    if(anim_lib.is_valid())
    {
        return anim_lib->get_animation_by_name(p_name);
    }

    return Ref<CharacterAnimationLibrary::AnimationItem>();
}


void CharacterAnimator::_bind_methods()
{
	ClassDB::bind_method(D_METHOD("set_animation_layer_arrays", "animation_layer_arrays"), &CharacterAnimator::set_animation_layer_arrays);
    ClassDB::bind_method(D_METHOD("get_animation_layer_arrays"), &CharacterAnimator::get_animation_layer_arrays);

    ADD_PROPERTY(PropertyInfo(Variant::PACKED_STRING_ARRAY, "animation_layer_arrays"), "set_animation_layer_arrays", "get_animation_layer_arrays");
}


//////////////////////////////////////////////// CharacterAnimationLogicNode /////////////////////////////////////////
void CharacterAnimationLogicNode::process_start(CharacterAnimatorLayer* animator,Blackboard* blackboard)
{
    if(start_blackboard_set.is_valid())
    {
        start_blackboard_set->execute(blackboard);
    }
    // 播放动作
    animator->play_animation(player_animation_name);
    if (GDVIRTUAL_IS_OVERRIDDEN(_animation_process_start)) {
        GDVIRTUAL_CALL(_animation_process_start, animator,blackboard);
        return ;
    }
}

void CharacterAnimationLogicNode::process(CharacterAnimatorLayer* animator,Blackboard* blackboard, double delta)
{
    if (GDVIRTUAL_IS_OVERRIDDEN(_animation_process)) {
        GDVIRTUAL_CALL(_animation_process, animator,blackboard, delta);
        return ;
    }

}

void CharacterAnimationLogicNode::process_stop(CharacterAnimatorLayer* animator,Blackboard* blackboard)
{
    if(stop_blackboard_set.is_valid())
    {
        stop_blackboard_set->execute(blackboard);
    }
    if (GDVIRTUAL_IS_OVERRIDDEN(_animation_process_stop)) {
        GDVIRTUAL_CALL(_animation_process_stop, animator,blackboard);
        return ;
    }
}
bool CharacterAnimationLogicNode::check_stop(CharacterAnimatorLayer* animator,Blackboard* blackboard)
{
    auto context = animator->_get_logic_context();
    if(context->time < check_stop_delay_time)
    {
        return false;
    }
    if (GDVIRTUAL_IS_OVERRIDDEN(_check_stop)) {
        bool is_stop = false;
        GDVIRTUAL_CALL(_check_stop, animator,blackboard, is_stop);
        return is_stop;
    }
    if(stop_check_type == Life)
    {
        return (life_time >= context->time);
    }
    else if(stop_check_type == AnimationLengthScale)
    {
        return (context->curr_animation_play_time / context->curr_animation_time_length >= anmation_scale );
    }
    else 
    {
        if(stop_check_condtion.is_valid())
        {
            return stop_check_condtion->is_enable(blackboard);
        }
    }
    return true;
}
    
void CharacterAnimationLogicNode::init_blackboard(Ref<BlackboardPlan> p_blackboard_plan)
{
    Ref<BlackboardPlan> blackboard_plan = p_blackboard_plan;
    if(blackboard_plan.is_null())
    {
        return ;
    }
    if(!blackboard_plan->has_var("OldForward"))
        blackboard_plan->add_var("OldForward",BBVariable(Variant::VECTOR3,Vector3()));

    if(!blackboard_plan->has_var("CurrForward"))
        blackboard_plan->add_var("CurrForward",BBVariable(Variant::VECTOR3,Vector3()));

    if(!blackboard_plan->has_var("MoveTarget"))
        blackboard_plan->add_var("MoveTarget",BBVariable(Variant::VECTOR3,Vector3()));

    if(!blackboard_plan->has_var("CurrState"))
        blackboard_plan->add_var("CurrState",BBVariable(Variant::STRING_NAME,StringName()));

    if(!blackboard_plan->has_var("HorizontalMovement"))
        blackboard_plan->add_var("HorizontalMovement",BBVariable(Variant::FLOAT,0.0f));

    if(!blackboard_plan->has_var("VerticalMovement"))
        blackboard_plan->add_var("VerticalMovement",BBVariable(Variant::FLOAT,0.0f));
    
    if(!blackboard_plan->has_var("Pitch"))
        blackboard_plan->add_var("Pitch",BBVariable(Variant::FLOAT,0.0f));
    if(!blackboard_plan->has_var("Yaw"))
        blackboard_plan->add_var("Yaw",BBVariable(Variant::FLOAT,0.0f));
    if(!blackboard_plan->has_var("Speed"))
        blackboard_plan->add_var("Speed",BBVariable(Variant::FLOAT,0.0f));
    
    // 是否使用能力
    if(!blackboard_plan->has_var("IsAbility"))
        blackboard_plan->add_var("IsAbility",BBVariable(Variant::BOOL,false));
    if(!blackboard_plan->has_var("AbilityIndex"))
        blackboard_plan->add_var("AbilityIndex",BBVariable(Variant::INT,0));
    if(!blackboard_plan->has_var("AbilityIntData"))
        blackboard_plan->add_var("AbilityIntData",BBVariable(Variant::INT,0));
    if(!blackboard_plan->has_var("AbilityFloatData"))
        blackboard_plan->add_var("AbilityFloatData",BBVariable(Variant::FLOAT,0.0f));
    
    if(!blackboard_plan->has_var("IsGround"))
        blackboard_plan->add_var("IsGround",BBVariable(Variant::BOOL,false));
    if(!blackboard_plan->has_var("IsMoving"))
        blackboard_plan->add_var("IsMoving",BBVariable(Variant::BOOL,false));
    
    if(!blackboard_plan->has_var("IsJump"))
        blackboard_plan->add_var("IsJump",BBVariable(Variant::BOOL,false));
    if(!blackboard_plan->has_var("IsCrouch"))
        blackboard_plan->add_var("IsCrouch",BBVariable(Variant::BOOL,false));
    if(!blackboard_plan->has_var("IsAttack"))
        blackboard_plan->add_var("IsAttack",BBVariable(Variant::BOOL,false));
    if(!blackboard_plan->has_var("IsDead"))
        blackboard_plan->add_var("IsDead",BBVariable(Variant::BOOL,false));
    if(!blackboard_plan->has_var("LegIndex"))
        blackboard_plan->add_var("LegIndex",BBVariable(Variant::INT,0));
    if(!blackboard_plan->has_var("IdentityIndex"))
    {
        blackboard_plan->add_var("IdentityIndex",BBVariable(Variant::INT,0));
    }
    
    // AI 大腦更新頻率
    if(!blackboard_plan->has_var("AI_BrainUpdate_Rate"))
        blackboard_plan->add_var("AI_BrainUpdate_Rate",BBVariable(Variant::FLOAT,1.0f));
    
}
void CharacterAnimationLogicNode::_bind_methods()
{
    ClassDB::bind_method(D_METHOD("set_blackboard_plan", "blackboard_plan"), &CharacterAnimationLogicNode::set_blackboard_plan);
    ClassDB::bind_method(D_METHOD("get_blackboard_plan"), &CharacterAnimationLogicNode::get_blackboard_plan);

    ClassDB::bind_method(D_METHOD("set_priority", "priority"), &CharacterAnimationLogicNode::set_priority);
    ClassDB::bind_method(D_METHOD("get_priority"), &CharacterAnimationLogicNode::get_priority);

    ClassDB::bind_method(D_METHOD("set_player_animation_name", "player_animation_name"), &CharacterAnimationLogicNode::set_player_animation_name);
    ClassDB::bind_method(D_METHOD("get_player_animation_name"), &CharacterAnimationLogicNode::get_player_animation_name);

    ClassDB::bind_method(D_METHOD("set_enter_condtion", "enter_condtion"), &CharacterAnimationLogicNode::set_enter_condtion);
    ClassDB::bind_method(D_METHOD("get_enter_condtion"), &CharacterAnimationLogicNode::get_enter_condtion);

    ClassDB::bind_method(D_METHOD("set_start_blackboard_set", "start_blackboard_set"), &CharacterAnimationLogicNode::set_start_blackboard_set);
    ClassDB::bind_method(D_METHOD("get_start_blackboard_set"), &CharacterAnimationLogicNode::get_start_blackboard_set);

    ClassDB::bind_method(D_METHOD("set_stop_blackboard_set", "stop_blackboard_set"), &CharacterAnimationLogicNode::set_stop_blackboard_set);
    ClassDB::bind_method(D_METHOD("get_stop_blackboard_set"), &CharacterAnimationLogicNode::get_stop_blackboard_set);

    ClassDB::bind_method(D_METHOD("set_check_stop_delay_time", "check_stop_delay_time"), &CharacterAnimationLogicNode::set_check_stop_delay_time);
    ClassDB::bind_method(D_METHOD("get_check_stop_delay_time"), &CharacterAnimationLogicNode::get_check_stop_delay_time);

    ClassDB::bind_method(D_METHOD("set_life_time", "life_time"), &CharacterAnimationLogicNode::set_life_time);
    ClassDB::bind_method(D_METHOD("get_life_time"), &CharacterAnimationLogicNode::get_life_time);

    ClassDB::bind_method(D_METHOD("set_stop_check_type", "stop_check_type"), &CharacterAnimationLogicNode::set_stop_check_type);
    ClassDB::bind_method(D_METHOD("get_stop_check_type"), &CharacterAnimationLogicNode::get_stop_check_type);

    ClassDB::bind_method(D_METHOD("set_stop_check_condtion", "stop_check_condtion"), &CharacterAnimationLogicNode::set_stop_check_condtion);
    ClassDB::bind_method(D_METHOD("get_stop_check_condtion"), &CharacterAnimationLogicNode::get_stop_check_condtion);

    ClassDB::bind_method(D_METHOD("set_stop_check_anmation_length_scale", "stop_check_anmation_length_scale"), &CharacterAnimationLogicNode::set_stop_check_anmation_length_scale);
    ClassDB::bind_method(D_METHOD("get_stop_check_anmation_length_scale"), &CharacterAnimationLogicNode::get_stop_check_anmation_length_scale);


    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "blackboard_plan",PROPERTY_HINT_RESOURCE_TYPE, "BlackboardPlan"), "set_blackboard_plan", "get_blackboard_plan");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "priority"), "set_priority", "get_priority");
    ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "player_animation_name"), "set_player_animation_name", "get_player_animation_name");
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "enter_condtion", PROPERTY_HINT_RESOURCE_TYPE, "CharacterAnimatorCondition"), "set_enter_condtion", "get_enter_condtion");
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "start_blackboard_set", PROPERTY_HINT_RESOURCE_TYPE, "AnimatorBlackboardSet"), "set_start_blackboard_set", "get_start_blackboard_set");
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "stop_blackboard_set", PROPERTY_HINT_RESOURCE_TYPE, "AnimatorBlackboardSet"), "set_stop_blackboard_set", "get_stop_blackboard_set");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "check_stop_delay_time"), "set_check_stop_delay_time", "get_check_stop_delay_time");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "life_time"), "set_life_time", "get_life_time");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "stop_check_type",PROPERTY_HINT_ENUM,"Life,PlayCount,Condition,Script"), "set_stop_check_type", "get_stop_check_type");
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "stop_check_condtion", PROPERTY_HINT_RESOURCE_TYPE, "CharacterAnimatorCondition"), "set_stop_check_condtion", "get_stop_check_condtion");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "stop_check_anmation_length_scale"), "set_stop_check_anmation_length_scale", "get_stop_check_anmation_length_scale");


    GDVIRTUAL_BIND(_animation_process_start,"_layer","_blackboard");
    GDVIRTUAL_BIND(_animation_process_stop,"_layer","_blackboard");
    GDVIRTUAL_BIND(_animation_process,"_layer","_blackboard", "_delta");
    GDVIRTUAL_BIND(_check_stop,"_layer","_blackboard");

    BIND_ENUM_CONSTANT(Life);
    BIND_ENUM_CONSTANT(AnimationLengthScale);
    BIND_ENUM_CONSTANT(Condition);
    BIND_ENUM_CONSTANT(Script);

}
