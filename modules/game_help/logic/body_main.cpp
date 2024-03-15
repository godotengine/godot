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
                p_layer->make_animation_instance(m_ChildAnimationArray[i].m_Name, p_playback_info_ptr[i]);
            }
        }
    }
}


void CharacterAnimatorNode1D::process_animation(class CharacterAnimatorLayer *p_layer,CharacterAnimationInstance *p_playback_info,float total_weight,Blackboard *p_blackboard)
{
    if(!p_blackboard->has_var(m_PropertyName))
    {
        return;
    }
    float v = p_blackboard->get_var(m_PropertyName,0);
    if(p_playback_info->m_WeightArray.size() != m_BlendData.m_ChildCount)
    {
        p_playback_info->m_WeightArray.resize(m_BlendData.m_ChildCount);
        p_playback_info->m_ChildAnimationPlaybackArray.resize(m_BlendData.m_ChildCount);
    }
    GetWeights1d(m_BlendData, p_playback_info->m_WeightArray.ptrw(), v);
    _blend_anmation(p_layer,m_BlendData.m_ChildCount, p_playback_info, total_weight,p_playback_info->m_WeightArray);

}
void CharacterAnimatorNode2D::process_animation(class CharacterAnimatorLayer *p_layer,CharacterAnimationInstance *p_playback_info,float total_weight,Blackboard *p_blackboard)
{
    if(!p_blackboard->has_var(m_PropertyName))
    {
        return;
    }
    Vector2 v = p_blackboard->get_var(m_PropertyName,0);
    if(p_playback_info->m_WeightArray.size() != m_BlendData.m_ChildCount)
    {
        p_playback_info->m_WeightArray.resize(m_BlendData.m_ChildCount);
        p_playback_info->m_ChildAnimationPlaybackArray.resize(m_BlendData.m_ChildCount);
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
void CharacterAnimatorLayer::_process_animation(Blackboard *p_playback_info,double p_delta,bool is_first)
{
    
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
                        if(m_BlendType == BT_Blend)
                        {
						    t_skeleton->set_bone_pose_position(t->bone_idx, t_skeleton->get_bone_pose_position(t->bone_idx).lerp(t->loc,blend_weight));
                        }
                        else
                        {
                            t_skeleton->set_bone_pose_position(t->bone_idx, t->loc);
                        }
					}
					if (t->rot_used) {
                        if(m_BlendType == BT_Blend)
                        {
                            t_skeleton->set_bone_pose_rotation(t->bone_idx, t_skeleton->get_bone_pose_rotation(t->bone_idx).slerp(t->rot,blend_weight));
                        }
                        else
                        {
                            t_skeleton->set_bone_pose_rotation(t->bone_idx, t->rot);
                        }                        
					}
					if (t->scale_used) {
                        if(m_BlendType == BT_Blend)
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
                        if(m_BlendType == BT_Blend)
                        {
                            t_node_3d->set_position(t_node_3d->get_position().lerp(t->loc,blend_weight));
                        }
                        else
                        {
                            t_node_3d->set_position(t->loc);
                        }
					}
					if (t->rot_used) {
                        if(m_BlendType == BT_Blend)
                        {
                            t_node_3d->set_rotation(t_node_3d->get_rotation().slerp(t->rot.get_euler(),blend_weight));
                        }
                        else
                        {
                            t_node_3d->set_rotation(t->rot.get_euler());
                        }
					}
					if (t->scale_used) {
                        if(m_BlendType == BT_Blend)
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
                    if(m_BlendType == BT_Blend)
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






