#include "human_animation_new.h"



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


