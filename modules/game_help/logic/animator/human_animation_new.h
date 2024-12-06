#pragma once 

#include "core/io/resource.h"
#include "scene/3d/skeleton_3d.h"

class HumanBonePostRotation : public RefCounted
{
    GDCLASS(HumanBonePostRotation, RefCounted);
public:
    void init(Ref<HumanBoneConfig> p_source_human,Ref<HumanBoneConfig> p_target_human) {
        
        for(auto& it : p_target_human->root_bone) {
            
            if(!p_source_human->virtual_pose.has(it)) {
                continue;
            }
            BonePose& source_pose = p_source_human->virtual_pose[it];
            BonePose& target_pose = p_target_human->virtual_pose[it]; 
            Basis post_basis = Basis(source_pose.rotation).inverse() ;
            post_basis = post_basis * Basis(target_pose.rotation);
            post_rotation[it] = post_basis;
            
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
    void retarget(Ref<HumanBoneConfig> p_target_human,HashMap<StringName, Basis>& p_golbal_post_rotation,
        HashMap<StringName, Basis>& p_target_global_post_rotation,HashMap<StringName, Basis>& p_target_global_post_rotation_inverse,HashMap<StringName, Basis>& p_target_local_post_rotation) {
        
        for(auto& it : p_target_human->root_bone) {
            if(!p_golbal_post_rotation.has(it)) {
                continue;
            }
            BonePose& target_pose = p_target_human->virtual_pose[it];
            Basis global_basis = p_golbal_post_rotation[it] * post_rotation[it];
			p_target_global_post_rotation[it] = global_basis;
            p_target_global_post_rotation_inverse[it] = global_basis.affine_inverse();
            p_target_local_post_rotation[it] = global_basis;

            for(auto& cit : target_pose.child_bones) {
                StringName bone_name = cit;
                if(!p_golbal_post_rotation.has(bone_name)) {
                    continue;
                }
                BonePose& bone_pose = p_target_human->virtual_pose[bone_name];
                retarget(p_target_human,bone_pose,bone_name,it,p_target_global_post_rotation,p_target_global_post_rotation_inverse,p_target_local_post_rotation);
            }
        }
        
    }

private:
    void compute_post_rotation(StringName p_bone_name, Ref<HumanBoneConfig> p_source_human,Ref<HumanBoneConfig> p_target_human,
        BonePose& source_pose,Basis& source_parent_rotation,BonePose& target_pose,Basis& target_parent_rotation) {
        
        Basis post_basis = Basis(source_pose.rotation).inverse() * source_parent_rotation.inverse();
        post_basis = post_basis * target_parent_rotation* Basis(target_pose.rotation);
        post_rotation[p_bone_name] = post_basis;
        for(auto& it : target_pose.child_bones) {
            StringName bone_name = it;
            if(!p_source_human->virtual_pose.has(bone_name)) {
                continue;
            }
            BonePose& bone_pose = p_target_human->virtual_pose[bone_name];
            compute_post_rotation(bone_name,p_source_human,p_target_human,source_pose,source_pose.global_pose.basis,bone_pose,target_pose.global_pose.basis);
        }
    }


    void retarget(Ref<HumanBoneConfig> p_target_human,BonePose& target_pose, StringName p_bone_name ,StringName p_parent_bone_name
        HashMap<StringName, Basis>& p_golbal_post_rotation,HashMap<StringName, Basis>& p_target_global_post_rotation,
        HashMap<StringName, Basis>& p_target_global_post_rotation_inverse,HashMap<StringName, Basis>& p_target_local_post_rotation) {
        
        
        Basis global_basis = p_golbal_post_rotation[p_bone_name]* post_rotation[p_bone_name];
        p_target_global_post_rotation[p_bone_name] = global_basis;
        p_target_global_post_rotation_inverse[p_bone_name] = global_basis.inverse();
        p_target_local_post_rotation[p_bone_name] = p_target_global_post_rotation_inverse[p_parent_bone_name] * global_basis;
        
        for(auto& it : target_pose.child_bones) {
            StringName bone_name = it;
            if(!p_golbal_post_rotation.has(bone_name)) {
                continue;
            }
            BonePose& bone_pose = p_target_human->virtual_pose[bone_name];
            retarget(p_target_human,bone_pose,bone_name,p_bone_name,p_target_global_post_rotation,p_target_global_post_rotation_inverse,p_target_local_post_rotation);
        }
        
    }

    HashMap<StringName, Basis> post_rotation;

};