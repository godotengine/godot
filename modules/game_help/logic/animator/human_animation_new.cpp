#include "human_animation_new.h"


void HumanBonePostRotation::init(Ref<HumanBoneConfig> p_source_human,Ref<HumanBoneConfig> p_target_human) {
    Basis post_basis;
    root_bone = p_target_human->root_bone;
    for(auto& it : p_target_human->root_bone) {
        
        if(!p_source_human->virtual_pose.has(it)) {
            continue;
        }
        HumanBonePoseOutput& bone_post = post[it];
        BonePose& source_pose = p_source_human->virtual_pose[it];
        BonePose& target_pose = p_target_human->virtual_pose[it]; 
        bone_post.rest_rotation = target_pose.rotation;
        post_basis.set_inverse(bone_post.rest_rotation );

        post_basis.xform(Basis(target_pose.rotation),post_basis);
        bone_post.post_rotation = post_basis;
        bone_post.child_bones = target_pose.child_bones;
        bone_post.global_post_rotation = bone_post.rest_rotation;
        bone_post.global_post_rotation.set_inverse(bone_post.global_post_rotation);
        bone_post.local_post_rotation = bone_post.rest_rotation;
        
        for(auto& cit : target_pose.child_bones) {
            StringName bone_name = cit;
            if(!p_source_human->virtual_pose.has(bone_name)) {
                continue;
            }
            BonePose& child_pose = p_target_human->virtual_pose[bone_name];
            compute_post_rotation(bone_name,p_source_human,p_target_human,source_pose,source_pose.global_pose.basis, child_pose,target_pose.global_pose.basis);
        }
    }
    
}

void HumanBonePostRotation::set_animation_rotation(const Quaternion& p_rotation,StringName p_bone_name) {
    auto it = post.find(p_bone_name);
    if(it == post.end()) {
        return;
    }
    HumanBonePoseOutput& output = it->value;
    output.animation_rotation = p_rotation;
    output.is_set_animation_rotation = true;
}



Ref<Animation> HumanBonePostRotation::build_human_animation(Skeleton3D* p_skeleton,HumanBoneConfig& p_config,Ref<Animation> p_animation,Dictionary & p_bone_map) {
    int key_count = p_animation->get_length() * 100 + 1;
    Vector3 loc,scale;
    Quaternion rot;
    HumanAnim::HumanSkeleton skeleton_config;
    Vector<HashMap<StringName, Quaternion>> animation_rotation;

    //  根节点的位置
    Vector<HashMap<StringName, Vector3>> animation_root_position;
    Vector<HashMap<StringName, Quaternion>> animation_root_rotation;
    animation_rotation.resize(key_count);
    animation_root_position.resize(key_count);
    animation_root_rotation.resize(key_count);
    Vector<Animation::Track*> tracks = p_animation->get_tracks();

    // 获取非人型骨骼的轨迹
    List<Animation::Track*> other_tracks;
    for (int j = 0; j < tracks.size(); j++) {
        Animation::Track* track = tracks[j];
        if (track->type == Animation::TYPE_POSITION_3D) {
            Animation::PositionTrack* track_cache = static_cast<Animation::PositionTrack*>(track);
            NodePath path = track_cache->path;
            StringName bone_name;
            if (path.get_subname_count() == 1) {
                // 获取骨骼映射
                bone_name = path.get_subname(0);
                if (p_bone_map.has(bone_name)) {
                    bone_name = p_bone_map[bone_name];
                }
            }
            if (bone_name != "Root" && !p_config.virtual_pose.has(bone_name)) {
                other_tracks.push_back(track);
                continue;
            }
        }
        else if (track->type == Animation::TYPE_ROTATION_3D) {
            Animation::RotationTrack* track_cache = static_cast<Animation::RotationTrack*>(track);

            NodePath path = track_cache->path;
            StringName bone_name;
            if (path.get_subname_count() == 1) {
                // 获取骨骼映射
                bone_name = path.get_subname(0);
                if (p_bone_map.has(bone_name)) {
                    bone_name = p_bone_map[bone_name];
                }
            }
            if (bone_name != "Root" && !p_config.virtual_pose.has(bone_name)) {
                other_tracks.push_back(track);
                continue;
            }
        }
        else if (track->type == Animation::TYPE_SCALE_3D) {
            Animation::ScaleTrack* track_cache = static_cast<Animation::ScaleTrack*>(track);

            NodePath path = track_cache->path;
            StringName bone_name;
            if (path.get_subname_count() == 1) {
                // 获取骨骼映射
                bone_name = path.get_subname(0);
                if (p_bone_map.has(bone_name)) {
                    bone_name = p_bone_map[bone_name];
                }
            }
            if (bone_name != "Root" && !p_config.virtual_pose.has(bone_name)) {
                other_tracks.push_back(track);
                continue;
            }
        }
        else
        {
            other_tracks.push_back(track);
        }
    }



    for(int i = 0; i < key_count; i++) {
        double time = double(i) / 100.0;
        for(int j = 0; j < tracks.size(); j++) {
            Animation::Track* track = tracks[j];
            if(track->type == Animation::TYPE_POSITION_3D) {
                Animation::PositionTrack* track_cache = static_cast<Animation::PositionTrack*>(track);
                int bone_index = HumanAnim::HumanAnimmation::get_bone_human_index(p_skeleton, p_bone_map,track_cache->path);
                if(bone_index < 0) {
                    continue;
                }
                Error err = p_animation->try_position_track_interpolate(j, time, &loc);
                p_skeleton->set_bone_pose_position(bone_index, loc);
            }
            else if(track->type == Animation::TYPE_ROTATION_3D) {
                Animation::RotationTrack* track_cache = static_cast<Animation::RotationTrack*>(track);
                int bone_index = HumanAnim::HumanAnimmation::get_bone_human_index(p_skeleton, p_bone_map, track_cache->path);
                if(bone_index < 0) {
                    continue;
                }
                Error err = p_animation->try_rotation_track_interpolate(j, time, &rot);
                p_skeleton->set_bone_pose_rotation(bone_index, rot);
            }
            else if(track->type == Animation::TYPE_SCALE_3D) {
                Animation::ScaleTrack* track_cache = static_cast<Animation::ScaleTrack*>(track);
                int bone_index = HumanAnim::HumanAnimmation::get_bone_human_index(p_skeleton, p_bone_map, track_cache->path);
                if(bone_index < 0) {
                    continue;
                }
                Error err = p_animation->try_scale_track_interpolate(j, time, &scale);
                p_skeleton->set_bone_pose_scale(bone_index, scale);
            }
        }
        // 转换骨骼姿势到动画
		HumanAnim::HumanAnimmation::build_skeleton_pose(p_skeleton,p_config,skeleton_config);
        // 存储动画
        animation_rotation.set(i,skeleton_config.bone_global_rotation);
        animation_root_position.set(i,skeleton_config.root_position);
        animation_root_rotation.set(i,skeleton_config.bone_global_rotation);

    }

    Ref<Animation> out_anim;
    out_anim.instantiate();
    out_anim->set_is_human_animation(true);
    if(animation_rotation.size() > 0) {

        auto& keys = animation_rotation[0];

        for(auto& it : keys) {
            int track_index = out_anim->add_track(Animation::TYPE_ROTATION_3D);
            Animation::RotationTrack* track = static_cast<Animation::RotationTrack*>(out_anim->get_track(track_index));
            track->path = String("hm.g.") + it.key;
            track->interpolation = Animation::INTERPOLATION_LINEAR;
            track->rotations.resize(animation_rotation.size());
            for(int i = 0;i < animation_rotation.size();i++) {
                double time = double(i) / 100.0;
                Animation::TKey<Quaternion> key;
                key.time = time;
                key.value = animation_rotation[i][it.key];
                track->rotations.set(i,key);

            }
        }

        auto& root_keys = animation_root_position[0];
        for(auto& it : root_keys) {
            int track_index = out_anim->add_track(Animation::TYPE_POSITION_3D);
            Animation::PositionTrack* track = static_cast<Animation::PositionTrack*>(out_anim->get_track(track_index));
            track->path = String("hm.p.") + it.key;
            track->interpolation = Animation::INTERPOLATION_LINEAR;
            track->positions.resize(animation_root_position.size());
            for(int i = 0;i < animation_root_position.size();i++) {
                double time = double(i) / 100.0;
                Animation::TKey<Vector3> key;
                key.time = time;
                key.value = animation_root_position[i][it.key];
                track->positions.set(i,key);
            }
        }
        // 根节点的朝向
        auto& root_look_keys = animation_root_rotation[0];
        for(auto& it : root_look_keys) {
            int track_index = out_anim->add_track(Animation::TYPE_ROTATION_3D);
            Animation::RotationTrack* track = static_cast<Animation::RotationTrack*>(out_anim->get_track(track_index));
            track->path = String("hm.gr.") + it.key;
            track->interpolation = Animation::INTERPOLATION_LINEAR;
            track->rotations.resize(animation_root_rotation.size());
            for(int i = 0;i < animation_root_rotation.size();i++) {
                double time = double(i) / 100.0;
                Animation::TKey<Quaternion> key;
                key.time = time;
                key.value = animation_root_rotation[i][it.key];
                track->rotations.set(i,key);
            }
        }
        
    }
    // 拷贝轨迹
    for(auto& it : other_tracks) {
        out_anim->add_track_ins(it->duplicate());
    }

    // 
    return out_anim;


}


void HumanBonePostRotation::retarget() {
    Basis global_basis;
    for(auto& it : root_bone) {
        HumanBonePoseOutput& output = post[it];
        output.local_post_rotation = output.global_post_rotation;

        for(auto& cit : output.child_bones) {
            StringName& bone_name = cit;
            HumanBonePoseOutput& child_output = post[bone_name];
            if(!child_output.is_set_animation_rotation) {
                continue;
            }
            retarget(child_output,output,bone_name);
        }
    }
    
}
void HumanBonePostRotation::apply(Skeleton3D *p_skeleton,const HashMap<String, float>& bone_blend_weight,float p_weight) {
    for(auto& it : post) {
        int bone_index = p_skeleton->find_bone(it.key);
        if(!it.value.is_set_animation_rotation) {
            continue;
        }
        if (bone_index >= 0) {
            float weight = 1.0f;
            if(bone_blend_weight.has(it.key)) {
                weight = bone_blend_weight[it.key];
            }
            p_skeleton->set_bone_pose_rotation(bone_index, p_skeleton->get_bone_pose_rotation(bone_index).slerp( it.value.local_post_rotation,p_weight * weight));
        }
    }
}

void HumanBonePostRotation::apply_root_motion(Vector3& p_position,Quaternion& p_rotation,Vector3& p_position_add,Quaternion & p_rotation_add,float p_weight) {

    
    if (root_global_rotation_add.size() > 0) {
        p_rotation_add = p_rotation.slerp(root_global_rotation_add.begin()->value,p_weight);
    }


    if(root_global_move_add.size() > 0) {
        p_position_add = p_position_add.lerp(root_global_move_add.begin()->value,p_weight);
    }
}

static Vector3 compute_lookat_position_add(Ref<Animation> p_animation,int track_index , double time_start, double time_end) {
        Vector3 loc,loc2;
        Error err = p_animation->try_position_track_interpolate(track_index, time_start, &loc);
        err = p_animation->try_position_track_interpolate(track_index, time_end, &loc2);
        return loc2 - loc;
    
}
static Quaternion compute_lookat_rotation_add(Ref<Animation> p_animation,int track_index , double time_start, double time_end) {
        Quaternion rot, rot2;
        Error err = p_animation->try_rotation_track_interpolate(track_index, time_start, &rot);
        err = p_animation->try_rotation_track_interpolate(track_index, time_end, &rot2);
        return rot.inverse() * rot2;
    
}

bool HumanBonePostRotation::apply_animation(Ref<Animation> p_animation,const Animation::Track*  tracks_ptr,int track_index,float time,double delta) {        
    StringName path_name = tracks_ptr->path.get_name(0);        
    if (path_name.begins_with("hm.g.")) {
        HumanAnimationBoneNameMapping * mapping = HumanAnimationBoneNameMapping::get_singleton();
        Quaternion rot;
        Error err = p_animation->try_rotation_track_interpolate(track_index, time, &rot);

        set_animation_rotation(rot,mapping->get_bone_name(path_name));
        
        return true;
    }
    else if(path_name.begins_with("hm.p.")) { 
        if(delta == 0) return true;

        Vector3 q;

        double last_time = time - delta;

        if(delta >= 0) {
            if(last_time < 0) {
                q = compute_lookat_position_add(p_animation,track_index, 0, time) + compute_lookat_position_add(p_animation,track_index, p_animation->get_length() + last_time, p_animation->get_length());
            }
            else {
                q = compute_lookat_position_add(p_animation,track_index, last_time, time);
            }
        } else {
            if(last_time > p_animation->get_length()) {
                q = compute_lookat_position_add(p_animation,track_index, time, p_animation->get_length()) + compute_lookat_position_add(p_animation,track_index, last_time - p_animation->get_length(), 0);
            }
            else {
                q = compute_lookat_position_add(p_animation,track_index, last_time , time);
            }

        }
        StringName name;
        
        HumanAnimationBoneNameMapping * mapping = HumanAnimationBoneNameMapping::get_singleton();
        if(mapping != nullptr) {
            name = mapping->get_bone_name(path_name);
        }
        else {
            name = path_name.substr(5);
        }
        root_global_move_add[name] = q;
        return true;
    }
    else if(path_name.begins_with("hm.gr.")) {
        if(delta == 0) return true;

        Basis q;

        double last_time = time - delta;

        if(delta >= 0) {
            if(last_time < 0) {
                q = compute_lookat_rotation_add(p_animation,track_index, 0, time) * compute_lookat_rotation_add(p_animation,track_index, p_animation->get_length() + last_time, p_animation->get_length());
            }
            else {
                q = compute_lookat_rotation_add(p_animation,track_index, last_time, time);
            }
        } else {
            if(last_time > p_animation->get_length()) {
                q = compute_lookat_rotation_add(p_animation,track_index, time, p_animation->get_length()) * compute_lookat_rotation_add(p_animation,track_index, last_time - p_animation->get_length(), 0);
            }
            else {
                q = compute_lookat_rotation_add(p_animation,track_index, last_time , time);
            }

        }
        StringName name;
        HumanAnimationBoneNameMapping * mapping = HumanAnimationBoneNameMapping::get_singleton();
        if(mapping != nullptr) {
            name = mapping->get_bone_name(path_name);
        }
        else {
            name = path_name.substr(6);
        }
        root_global_rotation_add[name] = q;
        return true;
    }
    return false;
}

void HumanBonePostRotation::compute_post_rotation(StringName p_bone_name, Ref<HumanBoneConfig> p_source_human,Ref<HumanBoneConfig> p_target_human,
    BonePose& source_pose,Basis& source_parent_rotation,BonePose& target_pose,Basis& target_parent_rotation) {
    
    HumanBonePoseOutput& bone_post = post[p_bone_name];
	bone_post.rest_rotation = Basis(target_pose.rotation);
    bone_post.post_rotation = Basis(source_pose.rotation).inverse() * source_parent_rotation.inverse();
    bone_post.post_rotation = bone_post.post_rotation * target_parent_rotation * Basis(target_pose.rotation);

    bone_post.child_bones = target_pose.child_bones;
    bone_post.global_post_rotation = target_parent_rotation * bone_post.rest_rotation;
    bone_post.global_post_rotation.set_inverse(bone_post.global_post_rotation);
    bone_post.local_post_rotation = bone_post.rest_rotation;
    for(auto& it : target_pose.child_bones) {
        StringName& bone_name = it;
        if(!p_source_human->virtual_pose.has(bone_name)) {
            continue;
        }
        BonePose& child_pose = p_target_human->virtual_pose[bone_name];
        compute_post_rotation(bone_name,p_source_human,p_target_human,source_pose,source_pose.global_pose.basis, child_pose,target_pose.global_pose.basis);
    }
}


void HumanBonePostRotation::retarget(HumanBonePoseOutput& output,HumanBonePoseOutput& parent_output, StringName p_bone_name ) {
    if(output.is_set_animation_rotation) {
        output.global_post_rotation = output.animation_rotation * output.post_rotation;
        output.local_post_rotation = parent_output.global_post_rotation_inverse * output.global_post_rotation;
    }
    else {
        output.global_post_rotation = parent_output.global_post_rotation * output.rest_rotation;
        output.local_post_rotation = output.rest_rotation;
    }
    output.global_post_rotation_inverse.set_inverse(output.global_post_rotation);
    
    for(auto& it : output.child_bones) {
        StringName& bone_name = it;
        HumanBonePoseOutput& child_output = post[p_bone_name];
        retarget(child_output,output,bone_name);
    }
    
}

