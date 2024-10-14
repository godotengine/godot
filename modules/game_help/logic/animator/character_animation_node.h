#pragma once

#include "core/io/resource.h"
#include "scene/resources/animation.h"
#include "scene/main/node.h"
#include "scene/animation/animation_player.h"
#include "scene/animation/animation_tree.h"

#include "../blackboard/blackboard_plan.h"
struct BonePose {
    int bone_index;
    Vector3 position;
    Quaternion rotation;
    Vector3 scale;
    Vector3 forward;
    float length;
    Vector<StringName> child_bones;

    void load(Dictionary& aDict) {
        clear();
        bone_index = aDict["bone_index"];
        position = aDict["position"];
        rotation = aDict["rotation"];
        scale = aDict["scale"];
        forward = aDict["forward"];
        length = aDict["length"];
        Vector<String> child  = aDict["child_bones"];

        for(int i=0; i < child.size(); i++) {
            child_bones.push_back(StringName(child[i]));
        }
    }
    void save(Dictionary& aDict) {

        aDict["bone_index"] = bone_index;
        aDict["position"] = position;
        aDict["rotation"] = rotation;
        aDict["scale"] = scale;
        aDict["forward"] = forward;
        aDict["length"] = length;
        Vector<String> child;
        for(int i=0; i < child_bones.size(); i++) {
            child.push_back(child_bones[i]);
        }
        aDict["child_bones"] = child;
    }
    void clear() {
        child_bones.clear();
        bone_index = -1;
        position = Vector3();
        rotation = Quaternion();
        scale = Vector3();
        forward = Vector3();
        length = 0.0f;
    }
    
};

class HumanConfig : public Resource {
    GDCLASS(HumanConfig, Resource);

    static void _bind_methods() {

        ClassDB::bind_method(D_METHOD("set_data", "data"), &HumanConfig::set_data);
        ClassDB::bind_method(D_METHOD("get_data"), &HumanConfig::get_data);

        ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "data", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_STORAGE), "set_data", "get_data");

    }

public:

    void load(Dictionary& aDict) {
        clear();
        Dictionary pose = aDict["virtual_pose"];
        auto keys = pose.keys();
        for (auto& it : keys) {
			Dictionary dict = pose.get(it, Dictionary());
            virtual_pose[it].load(dict);
        }

        Vector<String> root = aDict["root_bone"];
        for(int i=0; i < root.size(); i++) {
            root_bone.push_back(StringName(root[i]));
        }
        
    }

    void save(Dictionary& aDict) {
        Dictionary pose;
		auto keys = pose.keys();
		for (auto& it : keys) {
			Dictionary dict = pose.get(it, Dictionary());
			virtual_pose[it].save(dict);
            pose[it] = dict;
        }
        aDict["virtual_pose"] = pose;
        Vector<String> root;
        for(int i=0; i < root_bone.size(); i++) {
            root.push_back(root_bone[i]);
        }
        aDict["root_bone"] = root;
    }

    void clear() {
        virtual_pose.clear();
        root_bone.clear();
    }

    void set_data(Dictionary aDict) {
        load(aDict);
    }
    Dictionary get_data() {
        Dictionary dict;
        save(dict);
        return dict;
    }
public:
    // 虚拟姿勢
    HashMap<StringName, BonePose> virtual_pose;

    Vector<StringName> root_bone;
    
};
   

namespace HumanAnim
{
    
 // 骨骼配置
    struct HumanSkeleton {
        
        HashMap<StringName, Transform3D> real_local_pose; 
        HashMap<StringName, Transform3D> real_pose; 

        HashMap<StringName, Vector3> global_lookat;
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
        static void build_virtual_pose(Skeleton3D *p_skeleton,HumanConfig& p_config,HashMap<String, String>& p_human_bone_label) {
            Vector<int> root_bones = p_skeleton->get_root_bones();


            // 人型骨骼的初始的根骨可能带有旋转,所以不能存储根骨
            for(int i=0;i<root_bones.size();i++) {

                StringName bone_name = p_skeleton->get_bone_name(root_bones[i]);
                BonePose pose;
                Transform3D trans = p_skeleton->get_bone_global_rest(root_bones[i]);
                float height = 1.0;
                Vector<int> children = p_skeleton->get_bone_children(root_bones[i]);
                Vector3 bone_foreard = Vector3(0,1,0);
                if(children.size()>0) {
                    bone_foreard = p_skeleton->get_bone_global_rest(children[0]).origin - (trans.origin);
                    height = bone_foreard.length();
                }
                else if(p_skeleton->get_bone_parent(i) >= 0) {
                    bone_foreard = trans.origin - p_skeleton->get_bone_global_rest(p_skeleton->get_bone_parent(i)).origin;
                    height = 1.0;
                }
                pose.position = trans.origin;
                pose.rotation = trans.basis.get_rotation_quaternion();
                float inv_height = 1.0 / height;
                pose.forward = bone_foreard.normalized();
                pose.scale = Vector3(inv_height,inv_height,inv_height);
                pose.length = height;
                pose.bone_index = root_bones[i];
                for(int j=0;j<children.size();j++) {
                    pose.child_bones.push_back(p_skeleton->get_bone_name(children[j]));
                }
                p_config.virtual_pose[bone_name] = pose;

                // 构建所有子骨骼的姿势
                for(int j=0;j<children.size();j++) {
                    build_virtual_pose(p_config,p_skeleton, children[j], p_human_bone_label);
                }
            }

            // 根据骨骼的高度计算虚拟姿势
            for(int i=0;i<p_skeleton->get_bone_count();i++) {
                int parent = p_skeleton->get_bone_parent(i);
                BonePose& parent_pose = p_config.virtual_pose[p_skeleton->get_bone_name(parent)];
                BonePose& child_pose = p_config.virtual_pose[p_skeleton->get_bone_name(i)];
                child_pose.position *= parent_pose.scale;
                
            }

            
        }
        static int get_bone_human_index(Skeleton3D* p_skeleton,const NodePath& path) {
            if (path.get_subname_count() == 1) {
                // 获取骨骼映射
                StringName bone_name = path.get_subname(0);
                return p_skeleton->find_bone(bone_name);
            }
            return -1;

        }

        static Ref<Animation> build_human_animation(Skeleton3D* p_skeleton,HumanConfig& p_config,Ref<Animation> p_animation) {            
            int key_count = p_animation->get_length() * 100 + 1;
            Vector3 loc,scale;
            Quaternion rot;
            HumanSkeleton skeleton_config;
            Vector<HashMap<StringName, Vector3>> animation_lookat;
            animation_lookat.resize(key_count);
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
                    }
                    if (!p_config.virtual_pose.has(bone_name)) {
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
                    }
                    if (!p_config.virtual_pose.has(bone_name)) {
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
                    }
                    if (!p_config.virtual_pose.has(bone_name)) {
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
                        int bone_index = get_bone_human_index(p_skeleton,track_cache->path);
                        if(bone_index < 0) {
                            continue;
                        }
                        Error err = p_animation->try_position_track_interpolate(j, time, &loc);
                        p_skeleton->set_bone_pose_position(bone_index, loc);
                    }
                    else if(track->type == Animation::TYPE_ROTATION_3D) {
                        Animation::RotationTrack* track_cache = static_cast<Animation::RotationTrack*>(track);
                        int bone_index = get_bone_human_index(p_skeleton,track_cache->path);
                        if(bone_index < 0) {
                            continue;
                        }
                        Error err = p_animation->try_rotation_track_interpolate(j, time, &rot);
                        p_skeleton->set_bone_pose_rotation(bone_index, rot);
                    }
                    else if(track->type == Animation::TYPE_SCALE_3D) {
                        Animation::ScaleTrack* track_cache = static_cast<Animation::ScaleTrack*>(track);
                        int bone_index = get_bone_human_index(p_skeleton,track_cache->path);
                        if(bone_index < 0) {
                            continue;
                        }
                        Error err = p_animation->try_scale_track_interpolate(j, time, &scale);
                        p_skeleton->set_bone_pose_scale(bone_index, scale);
                    }
                }
                // 转换骨骼姿势到动画
                build_skeleton_pose(p_skeleton,p_config,skeleton_config);
                // 存储动画
                animation_lookat.set(i,skeleton_config.global_lookat);

            }

            Ref<Animation> out_anim;
            out_anim.instantiate();
            out_anim->set_is_human_animation(true);
            if(animation_lookat.size() > 0) {

                auto& keys = animation_lookat[0];

                for(auto& it : keys) {
                    int track_index = out_anim->add_track(Animation::TYPE_POSITION_3D);
                    Animation::PositionTrack* track = static_cast<Animation::PositionTrack*>(out_anim->get_track(track_index));
                    track->path = String("hm.") + it.key;
                    track->interpolation = Animation::INTERPOLATION_LINEAR;
                    track->positions.resize(animation_lookat.size());
                    for(int i = 0;i < animation_lookat.size();i++) {
                        double time = double(i) / 100.0;
						Animation::TKey<Vector3> key;
						key.time = time;
						key.value = animation_lookat[i][it.key];
                        track->positions.set(i,key);

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

        // 构建真实姿势
        static void build_skeleton_pose(Skeleton3D* p_skeleton,HumanConfig& p_config,HumanSkeleton& p_skeleton_config) {

            for(auto& it : p_config.root_bone) {
                BonePose& pose = p_config.virtual_pose[it];
                Transform3D& trans = p_skeleton_config.real_pose[it];
                trans = p_skeleton->get_bone_global_pose(pose.bone_index);

                build_skeleton_local_pose(p_skeleton,p_config, pose,p_skeleton_config);
            }
            
            for(auto& it : p_config.root_bone) {
                Transform3D& trans = p_skeleton_config.real_pose[it] ;
                Transform3D local_trans;
                build_skeleton_global_lookat(p_config,local_trans,p_skeleton_config);
            }
        }

        // 重定向骨骼
        static void retarget(HumanConfig& p_config,HumanSkeleton& p_skeleton_config) {
            for(auto& it : p_config.root_bone) {
                BonePose& pose = p_config.virtual_pose[it];
                Transform3D& trans = p_skeleton_config.real_pose[it];
                trans.origin = pose.position;
                Basis rot;
                rot.rotate_to_align( pose.forward, p_skeleton_config.global_lookat[it]);
                trans.basis = rot * trans.basis;
                p_skeleton_config.real_local_pose[it] = trans;


                Transform3D local_trans;
                retarget(p_config, pose, local_trans,p_skeleton_config);
            }
            
        }


        static const HashMap<String, String>& get_bone_label() {
            static HashMap<String, String> label_map = {
                {"Hips",L"臀部"},

                {"LeftUpperLeg",L"左上腿"},
                {"RightUpperLeg",L"右上腿"},

                {"LeftLowerLeg",L"左下腿"},
                {"RightLowerLeg",L"右下腿"},

                {"LeftFoot",L"左脚"},
                {"RightFoot",L"右脚"},

                {"Spine",L"脊柱"},
                {"Chest",L"颈部"},
                {"UpperChest",L"上胸部"},
                {"Neck",L"颈部"},
                {"Head",L"头部"},

                {"LeftShoulder",L"左肩"},
                {"RightShoulder",L"右肩"},

                {"LeftUpperArm",L"左上臂"},
                {"RightUpperArm",L"右上臂"},

                {"LeftLowerArm",L"左下臂"},
                {"RightLowerArm",L"右下臂"},

                {"LeftHand",L"左手"},
                {"RightHand",L"右手"},

                {"LeftToes",L"左足"},
                {"RightToes",L"右足"},

                {"LeftEye",L"左眼"},
                {"RightEye",L"右眼"},

                {"Jaw",L"下巴"},

                {"LeftThumbMetacarpal",L"左拇指"},
                {"LeftThumbProximal",L"左拇指近端"},
                {"LeftThumbDistal",L"左拇指远端"},

                {"LeftIndexProximal",L"左食指近端"},
                {"LeftIndexIntermediate",L"左食指中间"},
                {"LeftIndexDistal",L"左食指远端"},

                {"LeftMiddleProximal",L"左中指近端"},
                {"LeftMiddleIntermediate",L"左中指中间"},
                {"LeftMiddleDistal",L"左中指远端"},

                {"LeftRingProximal",L"左无名指近端"},
                {"LeftRingIntermediate",L"左无名指中间"},
                {"LeftRingDistal",L"左无名指远端"},

                {"LeftLittleProximal",L"左小拇指近端"},
                {"LeftLittleIntermediate",L"左小拇指中间"},
                {"LeftLittleDistal",L"左小拇指远端"},

                {"RightThumbMetacarpal",L"右拇指"},
                {"RightThumbProximal",L"右拇指近端"},
                {"RightThumbDistal",L"右拇指远端"},

                {"RightIndexProximal",L"右食指近端"},
                {"RightIndexIntermediate",L"右食指中间"},
                {"RightIndexDistal",L"右食指远端"},

                {"RightMiddleProximal",L"右中指近端"},
                {"RightMiddleIntermediate",L"右中指中间"},
                {"RightMiddleDistal",L"右中指远端"},

                {"RightRingProximal",L"右无名指近端"},
                {"RightRingIntermediate",L"右无名指中间"},
                {"RightRingDistal",L"右无名指远端"},

                {"RightLittleProximal",L"右小拇指近端"},
                {"RightLittleIntermediate",L"右小拇指中间"},
                {"RightLittleDistal",L"右小拇指远端"},

            };
            return label_map;
        }

     private:
        static void build_virtual_pose(HumanConfig& p_config,Skeleton3D* p_skeleton, int bone_index,HashMap<String, String>& p_human_bone_label) {
            
            Vector<int> child_bones = p_skeleton->get_bone_children(bone_index);
            Transform3D parent_trans = p_skeleton->get_bone_global_rest(bone_index);
            parent_trans.invert();
            for(int i=0; i < child_bones.size(); i++) {
                String bone_name = p_skeleton->get_bone_name(child_bones[i]);
                if(!p_human_bone_label.has(bone_name)) {
                    continue;
                }
                BonePose pose;
                Transform3D trans = p_skeleton->get_bone_global_rest(child_bones[i]);
                float height = 1.0;
                Vector<int> children = p_skeleton->get_bone_children(child_bones[i]);
                Vector3 bone_foreard = Vector3(0,1,0);
                if(children.size()>0) {
                    bone_foreard = p_skeleton->get_bone_global_rest(children[0]).origin - (trans.origin);
                    height = bone_foreard.length();
                }
                else if(p_skeleton->get_bone_parent(i) >= 0) {
                    bone_foreard = trans.origin - p_skeleton->get_bone_global_rest(p_skeleton->get_bone_parent(i)).origin;
                    height = 1.0;
                }
                Transform3D local_trans =  trans * parent_trans;
                pose.position = local_trans.origin;
                pose.rotation = local_trans.basis.get_rotation_quaternion();
                float inv_height = 1.0 / height;
                pose.forward = bone_foreard.normalized();
                pose.scale = Vector3(inv_height,inv_height,inv_height);
                pose.length = height;
                pose.bone_index = child_bones[i];
                for(int j=0;j<children.size();j++) {
                    pose.child_bones.push_back(p_skeleton->get_bone_name(children[j]));
                }
                p_config.virtual_pose[bone_name] = pose;
				build_virtual_pose(p_config, p_skeleton,  child_bones[i], p_human_bone_label);
            }
            
        }
        static void build_skeleton_local_pose(Skeleton3D* p_skeleton,HumanConfig& p_config,BonePose& parent_pose,HumanSkeleton& p_skeleton_config) {
            for(auto& it : parent_pose.child_bones) {
                BonePose& pose = p_config.virtual_pose[it];
                Transform3D& trans = p_skeleton_config.real_pose[it];
                trans.origin = pose.position;
                trans.basis = Basis(p_skeleton->get_bone_pose(pose.bone_index).basis.get_rotation_quaternion());

                build_skeleton_local_pose(p_skeleton,p_config, pose,p_skeleton_config);

            }
        }

        static void build_skeleton_global_lookat(HumanConfig& p_config,Transform3D& parent_pose,HumanSkeleton& p_skeleton_config) {

            for(auto& it : p_skeleton_config.real_pose) {
                Transform3D& trans = it.value;
                trans = trans * parent_pose;
                p_skeleton_config.global_lookat[it.key] = trans.xform(p_config.virtual_pose[it.key].position);

            }
            
        }
        static void retarget(HumanConfig& p_config,BonePose& pose,Transform3D& parent_trans,HumanSkeleton& p_skeleton_config) {

            // 重定向骨骼的世界坐标
            Basis rot;

            for(auto& it : pose.child_bones) {
                BonePose& pose = p_config.virtual_pose[it];
                Transform3D& trans = p_skeleton_config.real_pose[it];
                trans.origin = pose.position;
                trans.basis = Basis(pose.rotation);
                trans *= parent_trans;

                rot.rotate_to_align( pose.forward, p_skeleton_config.global_lookat[it] - trans.origin);

                trans.origin = pose.position;
                trans.basis = rot * trans.basis ;
                trans *= parent_trans;
                
                Transform3D& local_trans = p_skeleton_config.real_local_pose[it];
                local_trans.origin = trans.origin;
                local_trans.basis = rot * Basis(pose.rotation) ;
                retarget(p_config,pose,trans,p_skeleton_config);
            }
            
        }
    

    };

    
}



class CharacterBoneMap : public Resource
{
    GDCLASS(CharacterBoneMap, Resource);

    static void _bind_methods()
    {
        ClassDB::bind_method(D_METHOD("set_bone_map", "bone_map"), &CharacterBoneMap::set_bone_map);
        ClassDB::bind_method(D_METHOD("get_bone_map"), &CharacterBoneMap::get_bone_map);

        ClassDB::bind_method(D_METHOD("set_bone_names", "bone_names"), &CharacterBoneMap::set_bone_names);
        ClassDB::bind_method(D_METHOD("get_bone_names"), &CharacterBoneMap::get_bone_names);

        ClassDB::bind_method(D_METHOD("set_human_config", "human_config"), &CharacterBoneMap::set_human_config);
        ClassDB::bind_method(D_METHOD("get_human_config"), &CharacterBoneMap::get_human_config);

        ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "bone_map"), "set_bone_map", "get_bone_map");
        ADD_PROPERTY(PropertyInfo(Variant::PACKED_STRING_ARRAY, "bone_names"), "set_bone_names", "get_bone_names");

        ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "human_config", PROPERTY_HINT_RESOURCE_TYPE, "HumanConfig"), "set_human_config", "get_human_config");

    }

public:
    void set_bone_map(const Dictionary& p_bone_map) { bone_map = p_bone_map; }
    Dictionary get_bone_map() { return bone_map; }
    void set_bone_names(const Vector<String>& p_bone_names) { bone_names = p_bone_names; }
    Vector<String> get_bone_names() { return bone_names; }

    void set_human_config(const Ref<HumanConfig>& p_human_config) { human_config = p_human_config; }
    Ref<HumanConfig> get_human_config() { return human_config; }

	Dictionary bone_map;
    // 动画名称列表,用来处理非人形动画的情况这类动画原始文件没有模型,会被识别成节点,需要靠这个动画节点名称重新映射成骨骼
    Vector<String> bone_names;
    Ref<HumanConfig> human_config;
};


class CharacterAnimationItem : public Resource
{
    GDCLASS(CharacterAnimationItem, Resource);
    static void _bind_methods();

public:

    void set_speed(double p_speed) { speed = p_speed; }
    double get_speed() { return speed; }

    void set_is_clip(bool p_is_clip) { is_clip = p_is_clip; }
    bool get_is_clip() { return is_clip; }

    void set_child_node(const Ref<class CharacterAnimatorNodeBase>& p_child_node) ;
    Ref<class CharacterAnimatorNodeBase> get_child_node();

    void set_animation(Ref<Animation> p_animation) { animation = p_animation; }
    Ref<Animation> get_animation()
    {
        return animation;
    }

    void _init();
    float _get_animation_length();
    void _set_animation_scale_by_length(float p_length);
public:
    Ref<Animation> animation;
    Ref<class CharacterAnimatorNodeBase> child_node;

	double speed = 1.0f;
    float last_using_time = 0;
    bool is_clip = true;
    bool is_init = false;
};
class CharacterAnimatorNodeBase : public Resource
{
    GDCLASS(CharacterAnimatorNodeBase, Resource);
    static void _bind_methods();

public:
    enum LoopType
    {
        LOOP_Once,
        LOOP_ClampCount,
        LOOP_PingPongOnce,
        LOOP_PingPongCount,
    };
	enum BlendType
	{
		SimpleDirectionnal2D = 1,
		FreeformDirectionnal2D = 2,
		FreeformCartesian2D = 3,
	};
	struct Blend1dDataConstant
	{
		LocalVector<float>       position_array;
	};
	struct MotionNeighborList
	{
		uint32_t m_Count = 0;
		LocalVector<uint32_t> m_NeighborArray;
	};
	struct Blend2dDataConstant
	{
		LocalVector<Vector2>             position_array;

		LocalVector<float>               m_ChildMagnitudeArray; // Used by type 2
		LocalVector<Vector2>             m_ChildPairVectorArray; // Used by type 2, (3 TODO)
		LocalVector<float>               m_ChildPairAvgMagInvArray; // Used by type 2
		LocalVector<MotionNeighborList>  m_ChildNeighborListArray; // Used by type 2, (3 TODO)
		bool is_init_precompute = false;

        void reset() {
            int count = position_array.size() * position_array.size();
            m_ChildMagnitudeArray.resize(count);
            m_ChildPairVectorArray.resize(count);
            m_ChildPairAvgMagInvArray.resize(count);
            m_ChildNeighborListArray.resize(count);

        }
		void precompute_freeform(BlendType type);

	};
    void touch() { lastUsingTime = OS::get_singleton()->get_unix_time(); }

    bool is_need_remove(float remove_time) { return OS::get_singleton()->get_unix_time() - lastUsingTime > remove_time; }

    virtual void add_item()
    {
        Ref<CharacterAnimationItem> item;
        item.instantiate();
        animation_arrays.push_back(item);
    }

    virtual void remove_item(int index)
    {
        animation_arrays.remove_at(index);
    }
    virtual void move_up_item(int index)
    {
        if(index > 0)
        {
            animation_arrays.swap(index, index-1);
        }
    }
    virtual void move_down_item(int index)
    {
		int size = (int)animation_arrays.size();
		size -= 1;
        if(index < size)
        {
            animation_arrays.swap(index, index+1);
        }
    }

    void set_item_animation(int index,Ref<Animation> p_animation) { animation_arrays[index]->animation = p_animation; }
    Ref<Animation> get_item_animation(int index) { return animation_arrays[index]->animation; }

    void set_item_animator_node(int index,Ref<CharacterAnimatorNodeBase> p_animator_node) { animation_arrays[index]->child_node = p_animator_node; }
    Ref<CharacterAnimatorNodeBase> get_item_animator_node(int index) { return animation_arrays[index]->child_node; }



public:
    virtual void process_animation(class CharacterAnimatorLayer *p_layer,struct CharacterAnimationInstance *p_playback_info,float total_weight,const Ref<Blackboard> &p_blackboard)
    {

    }
public:
    void _blend_anmation(CharacterAnimatorLayer *p_layer,int child_count,struct CharacterAnimationInstance *p_playback_info,float total_weight,const LocalVector<float> &weight_array,const Ref<Blackboard> &p_blackboard);
    // 统一动画长度
    void _normal_animation_length();
    virtual float _get_animation_length();
    void _set_animation_scale_by_length(float p_length);
	virtual void update_animation_time(struct CharacterAnimationInstance* p_playback_info);

    void set_animation_arrays(TypedArray<CharacterAnimationItem> p_animation_arrays) { 
        animation_arrays.clear();
        for(int i=0;i<p_animation_arrays.size();i++)
        {
            animation_arrays.push_back(p_animation_arrays[i]);
        }
    }
    TypedArray<CharacterAnimationItem> get_animation_arrays() {
        TypedArray<CharacterAnimationItem> rs;
        for(uint32_t i=0;i<animation_arrays.size();i++)
        {
            rs.push_back(animation_arrays[i]);
        }
         return rs; 
    }

    void set_black_board_property(const StringName& p_black_board_property) { black_board_property = p_black_board_property; }
    StringName get_black_board_property() { return black_board_property; }

    void set_black_board_property_y(const StringName& p_black_board_property_y) { black_board_property_y = p_black_board_property_y; }
    StringName get_black_board_property_y() { return black_board_property_y; }
    virtual void _init();

    void set_fade_out_time(float p_fade_out_time) { fade_out_time = p_fade_out_time; }
    float get_fade_out_time() { return fade_out_time; }

    void set_loop(LoopType p_loop) { isLoop = p_loop; }
    LoopType get_loop() { return isLoop; }

    void set_loop_count(int p_loop_count) { loop_count = p_loop_count; }
    int get_loop_count() { return loop_count; }
public:
     

    static float weight_for_index(const float* thresholdArray, uint32_t count, uint32_t index, float blend);
    static void get_weights_simple_directional(const Blend2dDataConstant& blendConstant,
        float* weightArray, int* cropArray, Vector2* workspaceBlendVectors,
        float blendValueX, float blendValueY, bool preCompute = false);
    static  float get_weight_freeform_directional(const Blend2dDataConstant& blendConstant, Vector2* workspaceBlendVectors, int i, int j, Vector2 blendPosition);
    static void get_weights_freeform_directional(const Blend2dDataConstant& blendConstant,
        float* weightArray, int* cropArray, Vector2* workspaceBlendVectors,
        float blendValueX, float blendValueY, bool preCompute = false);
    static void get_weights_freeform_cartesian(const Blend2dDataConstant& blendConstant,
        float* weightArray, int* cropArray, Vector2* workspaceBlendVectors,
        float blendValueX, float blendValueY, bool preCompute = false);
    static void get_weights1d(const Blend1dDataConstant& blendConstant, float* weightArray, float blendValue);
	

	void _add_animation_item(const Ref<CharacterAnimationItem>& p_anim)
	{
		animation_arrays.push_back(p_anim);
	}
    Ref<CharacterAnimationItem> get_animation_item(int index) { return animation_arrays[index]; }

    // 設置黑板
    void set_blackboard_plan(const Ref<BlackboardPlan>& p_blackboard_plan) { blackboard_plan = p_blackboard_plan; }
    virtual Array _get_blackbord_propertys()
    {
        Array rs;
        if(!blackboard_plan.is_null())
        {
            blackboard_plan->get_property_names_by_type(Variant::FLOAT,rs);
        }
        return rs;
    }
protected:
    
    Ref<BlackboardPlan> blackboard_plan;
    LocalVector<Ref<CharacterAnimationItem>>		animation_arrays;
    StringName								black_board_property;
    StringName								black_board_property_y;
    float									fade_out_time = 0.0f;
    float									lastUsingTime = 0.0f;
    LoopType								isLoop = LOOP_Once;
    int										loop_count = 0;


};
class CharacterAnimatorNode1D : public CharacterAnimatorNodeBase
{
    GDCLASS(CharacterAnimatorNode1D, CharacterAnimatorNodeBase);
    
    static void _bind_methods();
public:
	void _set_black_board_property(const StringName& p_black_board_property) { black_board_property = p_black_board_property; }
	StringName _get_black_board_property() { return black_board_property; }
    void add_animation(const Ref<Animation> & p_anim,float p_pos);
    virtual void process_animation(class CharacterAnimatorLayer *p_layer,CharacterAnimationInstance *p_playback_info,float total_weight,const Ref<Blackboard> &p_blackboard) override;

    void set_position_array(Vector<float> p_array) { blend_data.position_array = p_array; }
    Vector<float> get_position_array() { return blend_data.position_array; }

    void set_position(uint32_t p_index, float p_value) {
		if (p_index >= blend_data.position_array.size()) {
			blend_data.position_array.resize(p_index + 1);
		}
		blend_data.position_array[p_index] = p_value;
	}
    float get_position(uint32_t p_index) {
        if(p_index >= blend_data.position_array.size()) {
            return 0;
        }
        return blend_data.position_array[p_index]; 
    }

    
    virtual void add_item()
    {
        Ref<CharacterAnimationItem> item;
        item.instantiate();
        animation_arrays.push_back(item);
        blend_data.position_array.push_back(0.0f);
    }

    virtual void remove_item(int index)
    {
        animation_arrays.remove_at(index);
        blend_data.position_array.remove_at(index);
    }
    virtual void move_up_item(int index)
    {
        if(index > 0)
        {
            animation_arrays.swap(index, index-1);

            blend_data.position_array.swap(index, index-1);
        }
    }
    virtual void move_down_item(int index)
    {
		int size = (int)animation_arrays.size();
		size -= 1;
        if(index < size)
        {
            animation_arrays.swap(index, index + 1);

            blend_data.position_array.swap(index, index + 1);
        }
    }

public:
    Blend1dDataConstant   blend_data;
};
// 顺序播放前面节点,循环播放后面节点
class CharacterAnimatorLoopLast : public CharacterAnimatorNodeBase
{
    GDCLASS(CharacterAnimatorLoopLast, CharacterAnimatorNodeBase);
    static void _bind_methods()
    {

    }
public:
    virtual void process_animation(class CharacterAnimatorLayer *p_layer,CharacterAnimationInstance *p_playback_info,float total_weight,const Ref<Blackboard> &p_blackboard) override;
    virtual float _get_animation_length();

};

class CharacterAnimatorNode2D : public CharacterAnimatorNodeBase
{
    GDCLASS(CharacterAnimatorNode2D, CharacterAnimatorNodeBase);
    static void _bind_methods();
public:
	void _set_black_board_property(const StringName& p_black_board_property) { black_board_property = p_black_board_property; }
	StringName _get_black_board_property() { return black_board_property; }

	void _set_black_board_property_y(const StringName& p_black_board_property_y) { black_board_property_y = p_black_board_property_y; }
	StringName _get_black_board_property_y() { return black_board_property_y; }
public:
    virtual void process_animation(class CharacterAnimatorLayer *p_layer,CharacterAnimationInstance *p_playback_info,float total_weight,const Ref<Blackboard> &p_blackboard) override;

    void set_blend_type(BlendType p_blend_type) { blend_type = (BlendType)p_blend_type; blend_data.is_init_precompute = false;}
    BlendType get_blend_type() { return blend_type; }

    void set_position_array(Vector<Vector2> p_array) {
		blend_data.position_array = p_array;
		blend_data.is_init_precompute = false;
	}
    Vector<Vector2> get_position_array() { return blend_data.position_array; }

	void set_position_x(uint32_t p_index, float p_value) { blend_data.position_array[p_index].x = p_value; blend_data.is_init_precompute = false; }
    float get_position_x(uint32_t p_index) { return blend_data.position_array[p_index].x; }
    void set_position_y(uint32_t p_index, float p_value) { blend_data.position_array[p_index].y = p_value; blend_data.is_init_precompute = false;}
    float get_position_y(uint32_t p_index) { return blend_data.position_array[p_index].y; }

    virtual void add_item()
    {
        Ref<CharacterAnimationItem> item;
        item.instantiate();
        animation_arrays.push_back(item);
        blend_data.position_array.push_back(Vector2(0,0));
		blend_data.is_init_precompute = false;
    }

    virtual void remove_item(int index)
    {
        animation_arrays.remove_at(index);
        blend_data.position_array.remove_at(index);
		blend_data.is_init_precompute = false;
    }
    virtual void move_up_item(int index)
    {
        if(index > 0)
        {
            animation_arrays.swap(index, index-1);

            blend_data.position_array.swap(index, index-1);
			blend_data.is_init_precompute = false;
        }
    }
    virtual void move_down_item(int index)
    {
		int size = (int)animation_arrays.size();
		size -= 1;
        if(index < size)
        {
            animation_arrays.swap(index, index + 1);

            blend_data.position_array.swap(index, index + 1);
			blend_data.is_init_precompute = false;
        }
    }

public:
    BlendType blend_type;
    Blend2dDataConstant blend_data;
};





struct CharacterAnimationInstance
{
	enum PlayState
	{
		PS_None,
		PS_Play,
		PS_FadeOut,
	};
	PlayState m_PlayState = PS_None;
	// 關閉的骨骼
	Dictionary disable_path;
	LocalVector<float> m_WeightArray;
	LocalVector<AnimationMixer::PlaybackInfo> m_ChildAnimationPlaybackArray;
	double delta = 0.0f;
	float time = 0.0f;
	double animation_time_pos = 0.0f;
	float fadeTotalTime = 0.0f;
    int play_index = 0;
	int play_count = 1;
	float get_weight()
	{
		if (m_PlayState == PS_FadeOut)
		{
			if (node->get_fade_out_time() <= 0.0f)
				return 0;
			return MAX(0.0f, 1.0f - fadeTotalTime / node->get_fade_out_time());
		}
		else
		{
			return 1.0f;
		}
	}
	// 动画节点
	Ref<CharacterAnimatorNodeBase> node;
};



VARIANT_ENUM_CAST(CharacterAnimatorNode2D::BlendType)



