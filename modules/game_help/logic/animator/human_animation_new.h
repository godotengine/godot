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
    void init(Ref<HumanBoneConfig> p_source_human,Ref<HumanBoneConfig> p_target_human);

    void set_animation_rotation(const Quaternion& p_rotation,StringName p_bone_name);

    bool apply_animation(Ref<Animation> p_animation, const Animation::Track* tracks_ptr, int track_index, float time, double delta) ;

    void retarget() ;

    void apply(Skeleton3D *p_skeleton,const HashMap<String, float>& bone_blend_weight,float p_weight);
    void apply_root_motion(Vector3& p_position,Quaternion& p_rotation,Vector3& p_position_add,Quaternion & p_rotation_add,float p_weight);

    
    static Ref<Animation> build_human_animation(Skeleton3D* p_skeleton,HumanBoneConfig& p_config,Ref<Animation> p_animation,Dictionary & p_bone_map);
private:
    void compute_post_rotation(StringName p_bone_name, Ref<HumanBoneConfig> p_source_human,Ref<HumanBoneConfig> p_target_human,
        BonePose& source_pose,Basis& source_parent_rotation,BonePose& target_pose,Basis& target_parent_rotation) ;


    void retarget(HumanBonePoseOutput& output,HumanBonePoseOutput& parent_output, StringName p_bone_name );

    HashMap<StringName, HumanBonePoseOutput> post;
	Vector<StringName> root_bone;
    HashMap<StringName, Vector3> root_global_move_add;
    HashMap<StringName, Quaternion> root_global_rotation_add;

};
