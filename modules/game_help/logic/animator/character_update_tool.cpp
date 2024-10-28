#include "character_update_tool.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/resources/animation.h"
#include "body_animator.h"

void CharacterAnimationUpdateTool::clear_cache(Skeleton3D* t_skeleton,Node* p_parent) {
    if (t_skeleton != nullptr)
    {
        if(t_skeleton->get_instance_id() != skeleton_id) {
            skeleton_id = t_skeleton->get_instance_id();
            context.clear();                
        }

    }
    else
    {
        skeleton_id = ObjectID();
        context.clear();
    }
    skeleton = t_skeleton;
    parent = p_parent;
    animation_instances.clear();
    context.reset();
    human_config = t_skeleton->get_human_config();
    if(human_config.is_valid()) {
        is_human = true;
		human_skeleton.rest(*human_config.ptr());

    }
}
void CharacterAnimationUpdateTool::add_animation_instance(AnimationMixer::AnimationInstance& ai) {
    animation_instances.push_back(ai);
    Ref<Animation> a = ai.animation_data.animation;
    add_animation_cache(ai.animation_data.bone_map, a);
}

void CharacterAnimationUpdateTool::process_animations() {
    for (auto it = animation_instances.begin(); it != animation_instances.end(); ++it) {
        process_anim(*it);
    }
}

void CharacterAnimationUpdateTool::layer_blend_apply(Ref<CharacterAnimatorLayerConfig> config, float blend_weight) {
    Skeleton3D* t_skeleton = Object::cast_to<Skeleton3D>(ObjectDB::get_instance(skeleton_id)); 
    for(auto it = context.bone_cache.begin(); it != context.bone_cache.end(); ++it) {
        AnimationMixer::TrackCacheTransform *t = it->value;
        switch (t->type) {
            case Animation::TYPE_POSITION_3D: {
                if (t->root_motion) {
                    root_motion_position = context.root_motion_cache.loc;
                    root_motion_rotation = context.root_motion_cache.rot;
                    root_motion_scale = context.root_motion_cache.scale - Vector3(1, 1, 1);
                    root_motion_position_accumulator = t->loc;
                    root_motion_rotation_accumulator = t->rot;
                    root_motion_scale_accumulator = t->scale;
                } else if ( t->bone_idx >= 0) {
                    if (!t_skeleton) {
                        return;
                    }
                    if (t->loc_used) {
                        if(t_skeleton->is_human_bone(t->bone_idx)) {
                            t_skeleton->set_bone_pose_position(t->bone_idx, t_skeleton->get_bone_rest(t->bone_idx).origin);
                        } else {
                            if(config->get_blend_type() == CharacterAnimatorLayerConfig::BT_Blend)
                            {
                                t_skeleton->set_bone_pose_position(t->bone_idx, t_skeleton->get_bone_pose_position(t->bone_idx).lerp(t->loc,blend_weight));
                            }
                            else
                            {
                                t_skeleton->set_bone_pose_position(t->bone_idx, t->loc);
                            }

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

                
                }
            } break;
            default: {
            } // The rest don't matter.
        }
    
    }


    for(auto it = context.blend_shape_cache.begin(); it != context.blend_shape_cache.end(); ++it) {
        AnimationMixer::TrackCacheBlendShape *t = it->value;

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
    }
    if(is_human) {
        //if(human_config->human) {
        //    human_config->human->app_dof_to_skeleton(t_skeleton,human_key_frame);
        //}
        if(human_config.is_valid()) {
            human_skeleton.apply(t_skeleton,blend_weight);
            human_skeleton.apply_root_motion(root_motion_position, root_motion_rotation,root_motion_position_add,root_motion_rotation_add,blend_weight);
        }
    }
}

int CharacterAnimationUpdateTool::get_bone_index(const Dictionary& p_bone_map, const NodePath& path) {
    StringName bone_name = path.get_subname(0);
    const Variant* re_name = p_bone_map.getptr(bone_name);
    if (re_name != nullptr) {
        bone_name = *re_name;
    }

    return skeleton->find_bone(bone_name);

}
void CharacterAnimationUpdateTool::add_animation_cache(const Dictionary& bone_map,const Ref<Animation>& p_anim) {
    if(is_human != p_anim->get_is_human_animation()) {
        return;
    }

    if (animation_cache.has(p_anim->get_instance_id())) {
        return;
    }
    animation_cache.insert(p_anim->get_instance_id());
    Ref<Animation> anim = p_anim;
    for (int i = 0; i < anim->get_track_count(); i++) {
        NodePath path = anim->track_get_path(i);
        Animation::TrackType track_src_type = anim->track_get_type(i);

        if(is_human && track_src_type == Animation::TYPE_POSITION_3D && path.get_name(0).begins_with("hm.")) {
            continue;
        }

        if (track_src_type == Animation::TYPE_POSITION_3D || track_src_type == Animation::TYPE_ROTATION_3D || track_src_type == Animation::TYPE_SCALE_3D)
        {
            // 获取骨骼映射
            if (skeleton && path.get_subname_count() == 1) {
                int bone_idx = get_bone_index(bone_map,path);
                if (bone_idx == -1) {
                    continue;
                }
                AnimationMixer::TrackCacheTransform* track_xform = nullptr;// = memnew(TrackCacheTransform);

                if (!context.bone_cache.has(bone_idx)) {
                    track_xform = context.bone_cache[bone_idx];
                    track_xform = memnew(AnimationMixer::TrackCacheTransform);
                    context.bone_cache[bone_idx] = track_xform;
                    track_xform->type = Animation::TYPE_POSITION_3D;
                    track_xform->bone_idx = bone_idx;
                    Transform3D rest = skeleton->get_bone_rest(bone_idx);
                    track_xform->init_loc = rest.origin;
                    track_xform->init_rot = rest.basis.get_rotation_quaternion();
                    track_xform->init_scale = rest.basis.get_scale();
                    track_xform->object_id = skeleton->get_instance_id();
                    track_xform->is_human_bone = skeleton->is_human_bone(bone_idx);
                }
            }
        }
        else if (track_src_type == Animation::TYPE_BLEND_SHAPE) {
            if (context.blend_shape_cache.has(path)) {
                continue;
            }
			if(path.get_name(0).begins_with("hm.")) {
                continue;
            }
            Ref<Resource> resource;
            Vector<StringName> leftover_path;
            Node* child = parent->get_node_and_resource(path, resource, leftover_path);
            if (path.get_subname_count() != 1) {
                //ERR_PRINT(String(anim->get_name()) + ": blend shape track does not contain a blend shape subname:  '" + String(path) + "'.");
                continue;
            }
            MeshInstance3D* mesh_3d = Object::cast_to<MeshInstance3D>(child);

            if (!mesh_3d) {
                //ERR_PRINT(String(anim->get_name()) + ": blend shape track does not point to MeshInstance3D:  '" + String(path) + "'.");
                continue;
            }
            StringName blend_shape_name = path.get_subname(0);
            int blend_shape_idx = mesh_3d->find_blend_shape_by_name(blend_shape_name);
            if (blend_shape_idx == -1) {
                //ERR_PRINT(String(anim->get_name()) + "': blend shape track points to a non-existing name:  '" + String(blend_shape_name) + "'.");
                continue;
            }
            AnimationMixer::TrackCacheBlendShape* track_bshape = memnew(AnimationMixer::TrackCacheBlendShape);

            track_bshape->shape_index = blend_shape_idx;
            track_bshape->object_id = mesh_3d->get_instance_id();
            track_bshape->init_value = 0;
            context.blend_shape_cache[path] = track_bshape;

        }

    }
}

void CharacterAnimationUpdateTool::process_anim(const AnimationMixer::AnimationInstance& ai) {
    Ref<Animation> a = ai.animation_data.animation;
    double time = ai.playback_info.time;
    double delta = ai.playback_info.delta;
    bool seeked = ai.playback_info.seeked;
    Animation::LoopedFlag looped_flag = ai.playback_info.looped_flag;
    bool is_external_seeking = ai.playback_info.is_external_seeking;
    const real_t* track_weights_ptr = ai.playback_info.track_weights.ptr();
    int track_weights_count = ai.playback_info.track_weights.size();
    bool backward = signbit(delta); // This flag is used by the root motion calculates or detecting the end of audio stream.
    bool seeked_backward = signbit(delta);
#ifndef _3D_DISABLED
    bool calc_root = !seeked || is_external_seeking;
#endif // _3D_DISABLED
    const Vector<Animation::Track*> tracks = a->get_tracks();
    Animation::Track* const* tracks_ptr = tracks.ptr();
    real_t a_length = a->get_length();
	temp_anim_skeleton.rest(*human_config.ptr());

    double blend = ai.playback_info.weight;
    int count = tracks.size();
    for (int i = 0; i < count; i++) {
        const Animation::Track* animation_track = tracks_ptr[i];
        if (!animation_track->enabled) {
            continue;
        }
        switch (animation_track->type) {
        case Animation::TYPE_POSITION_3D: {
            StringName name = animation_track->path.get_name(0);
			if (name.begins_with("hm."))
			{
                if(name.begins_with("hm.v.")) {
                    temp_anim_skeleton.set_root_lookat(a,name, i, time, delta);
                }
                else if(name.begins_with("hm.p.")) {
                    temp_anim_skeleton.set_root_position_add(a,name, i, time, delta);
                }
                else {
                    Vector3 loc;
                    Error err = a->try_position_track_interpolate(i, time, &loc);
                    temp_anim_skeleton.set_human_lookat(animation_track->path.get_name(0), loc);

                }
                continue;
			}
            int bone_idx = get_bone_index(ai.animation_data.bone_map, animation_track->path);
            if (bone_idx == -1) {
                continue;
            }
            AnimationMixer::TrackCacheTransform* t = context.bone_cache[bone_idx];
            
            if (t->is_human_bone) {
                continue;
            }            
            if (t->root_motion && calc_root) {
                double prev_time = time - delta;
                if (!backward) {
                    if (Animation::is_less_approx(prev_time, 0)) {
                        switch (a->get_loop_mode()) {
                        case Animation::LOOP_NONE: {
                            prev_time = 0;
                        } break;
                        case Animation::LOOP_LINEAR: {
                            prev_time = Math::fposmod(prev_time, (double)a_length);
                        } break;
                        case Animation::LOOP_PINGPONG: {
                            prev_time = Math::pingpong(prev_time, (double)a_length);
                        } break;
                        default:
                            break;
                        }
                    }
                }
                else {
                    if (Animation::is_greater_approx(prev_time, (double)a_length)) {
                        switch (a->get_loop_mode()) {
                        case Animation::LOOP_NONE: {
                            prev_time = (double)a_length;
                        } break;
                        case Animation::LOOP_LINEAR: {
                            prev_time = Math::fposmod(prev_time, (double)a_length);
                        } break;
                        case Animation::LOOP_PINGPONG: {
                            prev_time = Math::pingpong(prev_time, (double)a_length);
                        } break;
                        default:
                            break;
                        }
                    }
                }
                Vector3 loc[2];
                if (!backward) {
                    if (Animation::is_greater_approx(prev_time, time)) {
                        Error err = a->try_position_track_interpolate(i, prev_time, &loc[0]);
                        if (err != OK) {
                            continue;
                        }
                        a->try_position_track_interpolate(i, (double)a_length, &loc[1]);
                        context.root_motion_cache.loc += (loc[1] - loc[0]) * blend;
                        prev_time = 0;
                    }
                }
                else {
                    if (Animation::is_less_approx(prev_time, time)) {
                        Error err = a->try_position_track_interpolate(i, prev_time, &loc[0]);
                        if (err != OK) {
                            continue;
                        }
                        a->try_position_track_interpolate(i, 0, &loc[1]);
                        context.root_motion_cache.loc += (loc[1] - loc[0]) * blend;
                        prev_time = (double)a_length;
                    }
                }
                Error err = a->try_position_track_interpolate(i, prev_time, &loc[0]);
                if (err != OK) {
                    continue;
                }
                a->try_position_track_interpolate(i, time, &loc[1]);
                context.root_motion_cache.loc += (loc[1] - loc[0]) * blend;
                prev_time = !backward ? 0 : (double)a_length;
            }
            Vector3 loc;
            Error err = a->try_position_track_interpolate(i, time, &loc);
            if (err != OK) {
                continue;
            }
            {
                t->loc = t->loc.lerp(loc, blend);
                t->loc_used = true;
            }
        } break;
        case Animation::TYPE_ROTATION_3D: {
            int bone_idx = get_bone_index(ai.animation_data.bone_map, animation_track->path);
            if (bone_idx == -1) {
                continue;
            }
            AnimationMixer::TrackCacheTransform* t = context.bone_cache[bone_idx];
            if (t->root_motion && calc_root) {
                double prev_time = time - delta;
                if (!backward) {
                    if (Animation::is_less_approx(prev_time, 0)) {
                        switch (a->get_loop_mode()) {
                        case Animation::LOOP_NONE: {
                            prev_time = 0;
                        } break;
                        case Animation::LOOP_LINEAR: {
                            prev_time = Math::fposmod(prev_time, (double)a_length);
                        } break;
                        case Animation::LOOP_PINGPONG: {
                            prev_time = Math::pingpong(prev_time, (double)a_length);
                        } break;
                        default:
                            break;
                        }
                    }
                }
                else {
                    if (Animation::is_greater_approx(prev_time, (double)a_length)) {
                        switch (a->get_loop_mode()) {
                        case Animation::LOOP_NONE: {
                            prev_time = (double)a_length;
                        } break;
                        case Animation::LOOP_LINEAR: {
                            prev_time = Math::fposmod(prev_time, (double)a_length);
                        } break;
                        case Animation::LOOP_PINGPONG: {
                            prev_time = Math::pingpong(prev_time, (double)a_length);
                        } break;
                        default:
                            break;
                        }
                    }
                }
                Quaternion rot[2];
                if (!backward) {
                    if (Animation::is_greater_approx(prev_time, time)) {
                        Error err = a->try_rotation_track_interpolate(i, prev_time, &rot[0]);
                        if (err != OK) {
                            continue;
                        }
                        a->try_rotation_track_interpolate(i, (double)a_length, &rot[1]);
                        context.root_motion_cache.rot = (context.root_motion_cache.rot * Quaternion().slerp(rot[0].inverse() * rot[1], blend)).normalized();
                        prev_time = 0;
                    }
                }
                else {
                    if (Animation::is_less_approx(prev_time, time)) {
                        Error err = a->try_rotation_track_interpolate(i, prev_time, &rot[0]);
                        if (err != OK) {
                            continue;
                        }
                        a->try_rotation_track_interpolate(i, 0, &rot[1]);
                        context.root_motion_cache.rot = (context.root_motion_cache.rot * Quaternion().slerp(rot[0].inverse() * rot[1], blend)).normalized();
                        prev_time = (double)a_length;
                    }
                }
                Error err = a->try_rotation_track_interpolate(i, prev_time, &rot[0]);
                if (err != OK) {
                    continue;
                }
                a->try_rotation_track_interpolate(i, time, &rot[1]);
                context.root_motion_cache.rot = (context.root_motion_cache.rot * Quaternion().slerp(rot[0].inverse() * rot[1], blend)).normalized();
                prev_time = !backward ? 0 : (double)a_length;
            }
            {
                Quaternion rot;
                Error err = a->try_rotation_track_interpolate(i, time, &rot);
                if (err != OK) {
                    continue;
                }
                t->rot = t->rot.slerp(rot, blend).normalized();
                t->rot_used = true;
            }

        } break;
        case Animation::TYPE_SCALE_3D: {
            int bone_idx = get_bone_index(ai.animation_data.bone_map, animation_track->path);
            if (bone_idx == -1) {
                continue;
            }
            AnimationMixer::TrackCacheTransform* t = context.bone_cache[bone_idx];
            if (t->is_human_bone) {
                continue;
            }
            if (t->root_motion && calc_root) {
                double prev_time = time - delta;
                if (!backward) {
                    if (Animation::is_less_approx(prev_time, 0)) {
                        switch (a->get_loop_mode()) {
                        case Animation::LOOP_NONE: {
                            prev_time = 0;
                        } break;
                        case Animation::LOOP_LINEAR: {
                            prev_time = Math::fposmod(prev_time, (double)a_length);
                        } break;
                        case Animation::LOOP_PINGPONG: {
                            prev_time = Math::pingpong(prev_time, (double)a_length);
                        } break;
                        default:
                            break;
                        }
                    }
                }
                else {
                    if (Animation::is_greater_approx(prev_time, (double)a_length)) {
                        switch (a->get_loop_mode()) {
                        case Animation::LOOP_NONE: {
                            prev_time = (double)a_length;
                        } break;
                        case Animation::LOOP_LINEAR: {
                            prev_time = Math::fposmod(prev_time, (double)a_length);
                        } break;
                        case Animation::LOOP_PINGPONG: {
                            prev_time = Math::pingpong(prev_time, (double)a_length);
                        } break;
                        default:
                            break;
                        }
                    }
                }
                Vector3 scale[2];
                if (!backward) {
                    if (Animation::is_greater_approx(prev_time, time)) {
                        Error err = a->try_scale_track_interpolate(i, prev_time, &scale[0]);
                        if (err != OK) {
                            continue;
                        }
                        a->try_scale_track_interpolate(i, (double)a_length, &scale[1]);
                        context.root_motion_cache.scale += (scale[1] - scale[0]) * blend;
                        prev_time = 0;
                    }
                }
                else {
                    if (Animation::is_less_approx(prev_time, time)) {
                        Error err = a->try_scale_track_interpolate(i, prev_time, &scale[0]);
                        if (err != OK) {
                            continue;
                        }
                        a->try_scale_track_interpolate(i, 0, &scale[1]);
                        context.root_motion_cache.scale += (scale[1] - scale[0]) * blend;
                        prev_time = (double)a_length;
                    }
                }
                Error err = a->try_scale_track_interpolate(i, prev_time, &scale[0]);
                if (err != OK) {
                    continue;
                }
                a->try_scale_track_interpolate(i, time, &scale[1]);
                context.root_motion_cache.scale += (scale[1] - scale[0]) * blend;
                prev_time = !backward ? 0 : (double)a_length;
            }
            {
                Vector3 scale;
                Error err = a->try_scale_track_interpolate(i, time, &scale);
                if (err != OK) {
                    continue;
                }
                t->scale = t->scale.lerp(scale, blend);
                t->scale_used = true;
            }
        } break;
        case Animation::TYPE_BLEND_SHAPE: {
            float value;
            Error err = a->try_blend_shape_track_interpolate(i, time, &value);
            //ERR_CONTINUE(err!=OK); //used for testing, should be removed
            if (err != OK) {
                continue;
            }
            AnimationMixer::TrackCacheBlendShape* t = context.blend_shape_cache[animation_track->path];
			if (!context.blend_shape_cache.has(animation_track->path)) {
				continue;
			}
            t->value = Math::lerp( (double)t->value, (double)value, blend);
        } break;
        }

    }
    if(is_human) {
		HumanAnim::HumanAnimmation::retarget(*human_config.ptr(), temp_anim_skeleton);
        human_skeleton.blend(temp_anim_skeleton, blend);
    }
}

void CharacterAnimationUpdateTool::process_human_anim() {
    skeleton->force_update_all_dirty_bones(false);
    int count = 0;
    for(uint32_t i=0;i<animation_instances.size();++i) {
        AnimationMixer::AnimationInstance &ai = animation_instances[i];
    }
    if(count > 0) {
        skeleton->_make_dirty();
    }
}
