#pragma once 

#include "human_animation.h"
#include "core/io/resource.h"
#include "scene/3d/skeleton_3d.h"
#include "scene/resources/animation.h"

class HumanBonePostRotation : public RefCounted
{
    GDCLASS(HumanBonePostRotation, RefCounted);
public:
    struct HumanBonePoseOutput {
        Basis rest_rotation;
        // 子节点信息
	    Vector<StringName> child_bones;

        // 动画的位置
        Basis animation_rotation;
        bool is_set_animation_rotation = false;

        // 变换矩阵
        Basis post_rotation;


        // 计算结果
        Basis global_post_rotation;
        Basis global_post_rotation_inverse;
        Basis local_post_rotation;
    };
    void init(Ref<HumanBoneConfig> p_source_human,Ref<HumanBoneConfig> p_target_human) {
        Basis post_basis;
        root_bone = p_target_human->root_bone;
        for(auto& it : p_target_human->root_bone) {
            
            if(!p_source_human->virtual_pose.has(it)) {
                continue;
            }
            HumanBonePoseOutput& bone_post = post[it];
            BonePose& source_pose = p_source_human->virtual_pose[it];
            BonePose& target_pose = p_target_human->virtual_pose[it]; 
			bone_post.rest_rotation = source_pose.rotation;
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
    void set_animation_rotation(const Quaternion& p_rotation,StringName p_bone_name) {
        auto it = post.find(p_bone_name);
        if(it == post.end()) {
            return;
        }
        HumanBonePoseOutput& output = it->value;
        output.animation_rotation = p_rotation;
        output.is_set_animation_rotation = true;
    }

    bool apply_animation(Ref<Animation> p_animation,Animation::Track* const* tracks_ptr,float time,double delta) ;

    void retarget() {
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

    
    static Ref<Animation> build_human_animation(Skeleton3D* p_skeleton,HumanBoneConfig& p_config,Ref<Animation> p_animation,Dictionary & p_bone_map);
private:
    void compute_post_rotation(StringName p_bone_name, Ref<HumanBoneConfig> p_source_human,Ref<HumanBoneConfig> p_target_human,
        BonePose& source_pose,Basis& source_parent_rotation,BonePose& target_pose,Basis& target_parent_rotation) {
        
        HumanBonePoseOutput& bone_post = post[p_bone_name];
		bone_post.rest_rotation = Basis(source_pose.rotation);
        Basis post_basis = bone_post.rest_rotation.inverse() * source_parent_rotation.inverse();
        post_basis = post_basis * target_parent_rotation* Basis(target_pose.rotation);
        bone_post.post_rotation = post_basis;
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


    void retarget(HumanBonePoseOutput& output,HumanBonePoseOutput& parent_output, StringName p_bone_name ) {
        if(output.is_set_animation_rotation) {
            output.global_post_rotation = output.animation_rotation * output.animation_rotation;
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

    HashMap<StringName, HumanBonePoseOutput> post;
	Vector<StringName> root_bone;
    HashMap<StringName, Vector3> root_global_move_add;
    HashMap<StringName, Quaternion> root_global_rotation_add;

};
