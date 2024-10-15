#pragma once

#include "scene/3d/node_3d.h"
#include "scene/3d/skeleton_3d.h"
#include "scene/animation/animation_mixer.h"
#include "scene/3d/human_anim/human.h"
#include "animation_help.h"
#include "human_animation.h"


// 动画遮罩
class CharacterAnimatorMask : public Resource
{
    GDCLASS(CharacterAnimatorMask, Resource);

    static void _bind_methods()
    {
        ClassDB::bind_method(D_METHOD("set_disable_path", "disable_path"), &CharacterAnimatorMask::set_disable_path);
        ClassDB::bind_method(D_METHOD("get_disable_path"), &CharacterAnimatorMask::get_disable_path);

        ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "disable_path"), "set_disable_path", "get_disable_path");
    }

public:
    void set_disable_path(const Dictionary& p_disable_path) { disable_path = p_disable_path; }
    Dictionary get_disable_path() { return disable_path; }
	Dictionary disable_path;
};
// 角色动画层的配置信息
class CharacterAnimatorLayerConfig : public Resource
{
    GDCLASS(CharacterAnimatorLayerConfig, Resource);
    static void _bind_methods()
    {
        ClassDB::bind_method(D_METHOD("set_blend_type", "blend_type"), &CharacterAnimatorLayerConfig::set_blend_type);
        ClassDB::bind_method(D_METHOD("get_blend_type"), &CharacterAnimatorLayerConfig::get_blend_type);
        ClassDB::bind_method(D_METHOD("set_mask", "mask"), &CharacterAnimatorLayerConfig::set_mask);
        ClassDB::bind_method(D_METHOD("get_mask"), &CharacterAnimatorLayerConfig::get_mask);
        ClassDB::bind_method(D_METHOD("set_layer_name", "layer_name"), &CharacterAnimatorLayerConfig::set_layer_name);
        ClassDB::bind_method(D_METHOD("get_layer_name"), &CharacterAnimatorLayerConfig::get_layer_name);

        ADD_PROPERTY(PropertyInfo(Variant::INT, "blend_type"), "set_blend_type", "get_blend_type");
        ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "mask", PROPERTY_HINT_RESOURCE_TYPE, "CharacterAnimatorMask"), "set_mask", "get_mask");
        ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "layer_name"), "set_layer_name", "get_layer_name");

        BIND_ENUM_CONSTANT(BT_Blend);
        BIND_ENUM_CONSTANT(BT_Override);
    }
public:
    enum BlendType
    {
        // 混合
        BT_Blend,
        // 覆盖
        BT_Override,
    };


    void set_mask(const Ref<CharacterAnimatorMask>& p_mask) { _mask = p_mask; }
    Ref<CharacterAnimatorMask> get_mask() { return _mask; }

    void set_layer_name(const StringName& p_layer_name) { layer_name = p_layer_name; }
    StringName get_layer_name() { return layer_name; }

    void set_blend_type(BlendType p_blend_type) { m_BlendType = p_blend_type; }
    BlendType get_blend_type() { return m_BlendType; }
protected:
    // 动画层的名称
    StringName layer_name;
    // 动画遮罩
    Ref<CharacterAnimatorMask> _mask;
    // 混合类型
    BlendType m_BlendType = BT_Blend;
};
struct AnimationCacheContext {

    HashMap<int, AnimationMixer::TrackCacheTransform*> bone_cache;
    HashMap<NodePath, AnimationMixer::TrackCacheBlendShape*> blend_shape_cache;
    AnimationMixer::RootMotionCache root_motion_cache;
    void reset() {
        for (auto it = bone_cache.begin(); it != bone_cache.end(); ++it) {
            it->value->reset();
        }
        for (auto it = blend_shape_cache.begin(); it != blend_shape_cache.end(); ++it) {
            it->value->reset();
        }
        root_motion_cache.loc = Vector3(0,0,0);
        root_motion_cache.rot = Quaternion(0,0,0,1);
        root_motion_cache.scale = Vector3(1,1,1);
    }
    void clear() {
        for (auto it = bone_cache.begin(); it != bone_cache.end(); ++it) {
            memdelete(it->value);
        }
        for (auto it = blend_shape_cache.begin(); it != blend_shape_cache.end(); ++it) {
            memdelete(it->value);
        }
        bone_cache.clear();
        blend_shape_cache.clear();
    }
};
class CharacterAnimationUpdateTool : public RefCounted
{
public:
    void clear_cache(Skeleton3D* t_skeleton,Node* p_parent) ;
    void add_animation_instance(AnimationMixer::AnimationInstance& ai);
    void process_animations() ;

    void layer_blend_apply(Ref<CharacterAnimatorLayerConfig> config, float blend_weight);
protected:

    int get_bone_index(const Dictionary& p_bone_map, const NodePath& path) ;
    void add_animation_cache(const Dictionary& bone_map,const Ref<Animation>& p_anim) ;

    void process_anim(const AnimationMixer::AnimationInstance& ai);
    void process_human_anim() ;
protected:
    Ref<HumanConfig> human_config;
    HumanAnim::HumanSkeleton temp_anim_skeleton;
    HumanAnim::HumanSkeleton human_skeleton;


    HashSet<ObjectID> animation_cache;
    LocalVector<AnimationMixer::AnimationInstance> animation_instances;
    ObjectID skeleton_id;
    Skeleton3D *skeleton = nullptr;
    Node* parent = nullptr;
    AnimationCacheContext context;
    Vector3 root_motion_position = Vector3(0, 0, 0);
    Quaternion root_motion_rotation = Quaternion(0, 0, 0, 1);
    Vector3 root_motion_scale = Vector3(0, 0, 0);

    Vector3 root_motion_position_accumulator = Vector3(0, 0, 0);
    Quaternion root_motion_rotation_accumulator = Quaternion(0, 0, 0, 1);
    Vector3 root_motion_scale_accumulator = Vector3(1, 1, 1);
    bool is_human = false;
};
