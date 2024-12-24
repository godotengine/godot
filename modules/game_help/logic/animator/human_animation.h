#pragma once

#include "scene/resources/animation.h"
#include "scene/3d/skeleton_3d.h"


namespace HumanAnim
{
    struct HumanAnimationBoneResult{
        
        Vector4 bone_global_lookat;
        Quaternion real_local_pose; 
        Transform3D real_global_pose; 
    };
	// 骨骼配置
    struct HumanSkeleton {
        
        HashMap<StringName, HumanAnimationBoneResult> bone_result;


        HashMap<StringName, Quaternion> bone_global_rotation;
		HashMap<StringName, Vector3> root_position;
		HashMap<StringName, Vector4> root_lookat;



		HashMap<StringName, Basis> root_global_rotation;
        HashMap<StringName, Vector3> root_global_move_add;
        HashMap<StringName, Basis> root_global_rotation_add;

        void rest(HumanBoneConfig& p_config) ;

        void clear();

        void set_human_lookat(StringName p_bone,const Vector3& p_lookat);
		void set_human_roll(StringName p_bone, float p_roll) ;
        static Basis retarget_root_direction(const Vector3& p_start_direction,const Vector3& p_end_direction) ;

        static Basis compute_lookat_rotation_add(Ref<Animation> p_animation,int track_index , double time_start, double time_end) ;

        static Vector3 compute_lookat_position_add(Ref<Animation> p_animation,int track_index , double time_start, double time_end) ;
        void set_root_lookat(Ref<Animation> p_animation,StringName p_bone, int track_index ,double time,double delta);
        void set_root_lookat_roll(Ref<Animation> p_animation,StringName p_bone, float p_roll) ;

        void set_root_position_add(Ref<Animation> p_animation, StringName p_bone, int track_index ,double time,double delta) ;

        void blend(HumanSkeleton& p_other,float p_weight) ;

		void apply_root_motion(Node3D* node);
        static const HashMap<String, float>& get_bone_blend_weight() ;


        void apply(Skeleton3D *p_skeleton,const HashMap<String, float>& bone_blend_weight,float p_weight);

        void apply_root_motion(Vector3& p_position,Quaternion& p_rotation,Vector3& p_position_add,Quaternion & p_rotation_add,float p_weight);

    };

    class HumanAnimmation {
     public:


        /** 
        提取动画姿势:
        把动画文件的骨骼旋转应用到虚拟骨骼,获取到世界位置,河道世界位置
        

        虚拟骨骼:
        长度为1的骨骼,姿势保持和原始骨骼一致
        
        
        应用动画姿势:
        用当前角色的虚拟骨骼从父节点到子节点进行世界空间的Lookat计算
        用虚拟骨骼的世界变换得到局部旋转
        应用虚拟骨骼的局部旋转到真实骨骼
        Hip骨骼的世界朝向应用到角色的朝向,Hip的世界位置应用到角色的位置
        */
        // 构建虚拟姿势
        static void build_virtual_pose(Skeleton3D *p_skeleton,HumanBoneConfig& p_config,HashMap<String, String>& p_human_bone_label) ;
        static void build_virtual_pose_global(HumanBoneConfig& p_config, BonePose& pose, HashMap<String, String>& p_human_bone_label) ;
        
        
        static int get_bone_human_index(Skeleton3D* p_skeleton, Dictionary& p_bone_map,const NodePath& path) ;

        static Ref<Animation> build_human_animation(Skeleton3D* p_skeleton,HumanBoneConfig& p_config,Ref<Animation> p_animation,Dictionary & p_bone_map, bool position_by_hip = false) ;


        // 重定向根骨骼朝向
        static void retarget_root_motion(HumanBoneConfig& p_config,HumanSkeleton& p_skeleton_config) ;

        // 重定向骨骼
        static void retarget(HumanBoneConfig& p_config,HumanSkeleton& p_skeleton_config) ;


        static const HashMap<String, String>& get_bone_label() ;

		// 构建真实姿势
		static void build_skeleton_pose(Skeleton3D* p_skeleton, HumanBoneConfig& p_config, HumanSkeleton& p_skeleton_config, bool position_by_hip = false) ;
     private:
     
        struct SortStringName {
            bool operator()(const StringName &l, const StringName &r) const {
                return l.str() > r.str();
            }
        };
        

        static void build_virtual_pose(Skeleton3D* p_skeleton, HumanBoneConfig& p_config, BonePose& pose, const String& bone_name, HashMap<String, String>& p_human_bone_label) ;
        static void build_skeleton_local_pose(Skeleton3D* p_skeleton, HumanBoneConfig& p_config, BonePose& parent_pose, Transform3D& parent_trans, HumanSkeleton& p_skeleton_config) ;

        static void build_skeleton_lookat(Skeleton3D* p_skeleton,HumanBoneConfig& p_config, BonePose& bone_pose,Transform3D& parent_pose,HumanSkeleton& p_skeleton_config);
        static void retarget(HumanBoneConfig& p_config,BonePose& pose,Transform3D& parent_trans,HumanSkeleton& p_skeleton_config) ;
    

    };

    
}

