#pragma once 

#include "core/io/resource.h"
#include "scene/3d/skeleton_3d.h"

class HumanBonePostRotation : public RefCounted
{
    GDCLASS(HumanBonePostRotation, RefCounted);
public:
    struct HumanBonePoseOutput {
        Basis post_rotation;
        Basis global_post_rotation;
        Basis global_post_rotation_inverse;
        Basis local_post_rotation;
    };
    void init(Ref<HumanBoneConfig> p_source_human,Ref<HumanBoneConfig> p_target_human) {
        Basis post_basis;
        for(auto& it : p_target_human->root_bone) {
            
            if(!p_source_human->virtual_pose.has(it)) {
                continue;
            }
            BonePose& source_pose = p_source_human->virtual_pose[it];
            BonePose& target_pose = p_target_human->virtual_pose[it]; 
            post_basis = Basis(source_pose.rotation).inverse() ;

            post_basis.xform(Basis(target_pose.rotation),post_basis);
            post[it].post_rotation = post_basis;
            
            for(auto& cit : target_pose.child_bones) {
                StringName bone_name = cit;
                if(!p_source_human->virtual_pose.has(bone_name)) {
                    continue;
                }
                BonePose& bone_pose = p_target_human->virtual_pose[bone_name];
                compute_post_rotation(bone_name,p_source_human,p_target_human,source_pose,source_pose.global_pose.basis,bone_pose,target_pose.global_pose.basis);
            }
        }
        
    }
    void retarget(Ref<HumanBoneConfig> p_target_human,HashMap<StringName, Basis>& p_animation_golbal_post_rotation) {
        Basis global_basis;
        for(auto& it : p_target_human->root_bone) {
            if(!p_animation_golbal_post_rotation.has(it)) {
                continue;
            }
            BonePose& target_pose = p_target_human->virtual_pose[it];
            HumanBonePoseOutput& output = post[it];
            Basis& bone_pose = p_animation_golbal_post_rotation[it];
            bone_pose.xform( output.post_rotation,output.global_post_rotation);
            output.global_post_rotation_inverse.set_inverse(output.global_post_rotation);
            output.local_post_rotation = output.global_post_rotation;

            for(auto& cit : target_pose.child_bones) {
                StringName& bone_name = cit;
                if(!p_animation_golbal_post_rotation.has(bone_name)) {
                    continue;
                }
                BonePose& bone_pose = p_target_human->virtual_pose[bone_name];
                retarget(p_target_human,bone_pose,bone_name,it,p_animation_golbal_post_rotation);
            }
        }
        
    }

private:
    void compute_post_rotation(StringName p_bone_name, Ref<HumanBoneConfig> p_source_human,Ref<HumanBoneConfig> p_target_human,
        BonePose& source_pose,Basis& source_parent_rotation,BonePose& target_pose,Basis& target_parent_rotation) {
        
        Basis post_basis = Basis(source_pose.rotation).inverse() * source_parent_rotation.inverse();
        post_basis = post_basis * target_parent_rotation* Basis(target_pose.rotation);
        post[p_bone_name].post_rotation = post_basis;
        for(auto& it : target_pose.child_bones) {
            StringName& bone_name = it;
            if(!p_source_human->virtual_pose.has(bone_name)) {
                continue;
            }
            BonePose& bone_pose = p_target_human->virtual_pose[bone_name];
            compute_post_rotation(bone_name,p_source_human,p_target_human,source_pose,source_pose.global_pose.basis,bone_pose,target_pose.global_pose.basis);
        }
    }


    void retarget(Ref<HumanBoneConfig> p_target_human,BonePose& target_pose, StringName p_bone_name ,StringName p_parent_bone_name,
        HashMap<StringName, Basis>& p_animation_golbal_post_rotation) {
        
        HumanBonePoseOutput& output = post[p_bone_name];
        HumanBonePoseOutput& parent_output = post[p_parent_bone_name];
        output.global_post_rotation = p_animation_golbal_post_rotation[p_bone_name] * output.post_rotation;
        output.global_post_rotation_inverse.set_inverse(output.global_post_rotation);
        output.local_post_rotation = parent_output.global_post_rotation_inverse * output.global_post_rotation;
        
        for(auto& it : target_pose.child_bones) {
            StringName& bone_name = it;
            if(!p_animation_golbal_post_rotation.has(bone_name)) {
                continue;
            }
            BonePose& bone_pose = p_target_human->virtual_pose[bone_name];
            retarget(p_target_human,bone_pose,bone_name,p_bone_name,p_animation_golbal_post_rotation);
        }
        
    }

    HashMap<StringName, HumanBonePoseOutput> post;

};
