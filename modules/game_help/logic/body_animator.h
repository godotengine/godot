#ifndef _BODY_ANIMATOR_H
#define _BODY_ANIMATOR_H

#include "scene/resources/packed_scene.h"
#include "scene/animation/animation_player.h"
#include "scene/animation/animation_tree.h"
#include "scene/3d/node_3d.h"
#include "scene/3d/skeleton_3d.h"
#include "scene/3d/physics/character_body_3d.h"
#include "body_part.h"
#include "animation_help.h"


#include "modules/limboai/bt/bt_player.h"

// 动画遮罩
class CharacterAnimatorMask : public Resource
{
    GDCLASS(CharacterAnimatorMask, Resource);
public:
	Dictionary disable_path;
};
class CharacterAnimatorNodeBase : public Resource
{
    GDCLASS(CharacterAnimatorNodeBase, Resource);

public:
    float fadeOutTime = 0.0f;
    bool isLoop = false;

public:
    virtual void process_animation(class CharacterAnimatorLayer *p_layer,struct CharacterAnimationInstance *p_playback_info,float total_weight,Blackboard *p_blackboard)
    {

    }
public:
    void _blend_anmation(CharacterAnimatorLayer *p_layer,int child_count,struct CharacterAnimationInstance *p_playback_info,float total_weight,const Vector<float> &weight_array,Blackboard *p_blackboard);

    
    struct MotionNeighborList
    {

        MotionNeighborList() : m_Count(0)
        {
        }

        uint32_t m_Count;
        Vector<uint32_t> m_NeighborArray;
    };  
    struct AnimationItem
    {
        StringName m_Name;
        Ref<CharacterAnimatorNodeBase> m_animation_node;
        float m_Speed = 1.0f;
        bool isClip = true;
    };
    Vector<AnimationItem>    m_ChildAnimationArray;
    StringName               m_PropertyName;

 // Constant data for direct blend node types - parameters
    struct BlendDirectDataConstant
    {

        BlendDirectDataConstant() : m_ChildCount(0), m_NormalizedBlendValues(0)
        {
        }

        uint32_t            m_ChildCount;
        Vector<uint32_t> m_ChildBlendEventIDArray;
        bool                m_NormalizedBlendValues;
    };
    struct Blend1dDataConstant
    {

        Blend1dDataConstant() : m_ChildCount(0)
        {
        }

        uint32_t            m_ChildCount;
        Vector<float>    m_ChildThresholdArray;

    };
    struct Blend2dDataConstant
    {

        Blend2dDataConstant()
        {
        }

        uint32_t                m_ChildCount;
        Vector<Vector2>     m_ChildPositionArray;

        uint32_t                m_ChildMagnitudeCount;
        Vector<float>        m_ChildMagnitudeArray; // Used by type 2
        uint32_t                m_ChildPairVectorCount;
        Vector<Vector2>     m_ChildPairVectorArray; // Used by type 2, (3 TODO)
        uint32_t                m_ChildPairAvgMagInvCount;
        Vector<float>        m_ChildPairAvgMagInvArray; // Used by type 2
        uint32_t                        m_ChildNeighborListCount;
        Vector<MotionNeighborList>   m_ChildNeighborListArray; // Used by type 2, (3 TODO)

    };
     

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


};
class CharacterAnimatorNode1D : public CharacterAnimatorNodeBase
{
    GDCLASS(CharacterAnimatorNode1D, CharacterAnimatorNodeBase);
public:
    virtual void process_animation(class CharacterAnimatorLayer *p_layer,CharacterAnimationInstance *p_playback_info,float total_weight,Blackboard *p_blackboard) override;
public:
    Blend1dDataConstant   m_BlendData;
};
class CharacterAnimatorNode2D : public CharacterAnimatorNodeBase
{
    GDCLASS(CharacterAnimatorNode2D, CharacterAnimatorNodeBase);
public:
    enum BlendType
    {
        SimpleDirectionnal2D = 1,
        FreeformDirectionnal2D = 2,
        FreeformCartesian2D = 3,
    };
    virtual void process_animation(class CharacterAnimatorLayer *p_layer,CharacterAnimationInstance *p_playback_info,float total_weight,Blackboard *p_blackboard) override;
public:
    BlendType m_BlendType;
    Blend2dDataConstant m_BlendData;
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
    Vector<float> m_WeightArray;
    Vector<AnimationMixer::PlaybackInfo> m_ChildAnimationPlaybackArray;
    float time = 0.0f;
    float delta = 0.0f;
    float fadeTotalTime = 0.0f;
    float get_weight()
    {
        if(m_PlayState == PS_FadeOut)
        {
            if(node->fadeOutTime <= 0.0f)
                return 0;
            return MAX(0.0f,1.0f - fadeTotalTime / node->fadeOutTime);
        }
        else
        {
            return 1.0f;
        }
    }
    // 动画节点
    Ref<CharacterAnimatorNodeBase> node;
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
// 动画分层
class CharacterAnimatorLayer: public AnimationMixer
{
    GDCLASS(CharacterAnimatorLayer, AnimationMixer);

    List<Ref<CharacterAnimatorNodeBase>> play_list;
public:

    Ref<CharacterAnimatorLayerConfig> config;

    // 处理动画
    void _process_animation(Blackboard *p_playback_info,double p_delta,bool is_first = true);
    void layer_blend_apply() ;
    Vector<Vector2> m_ChildInputVectorArray;
    Vector<int> m_TempCropArray;
    Vector<float> m_TotalAnimationWeight;
    List<CharacterAnimationInstance> m_AnimationInstances;
    float blend_weight = 1.0f;
    class CharacterAnimator* m_Animator = nullptr;
    void init(CharacterAnimator* p_animator,const Ref<CharacterAnimatorLayerConfig>& _config)
    {
         m_Animator = p_animator; 
         config = _config;
    }

    void play_animation(Ref<CharacterAnimatorNodeBase> p_node);
    ~CharacterAnimatorLayer();
};

// 时间线资源,这个主要用来Animation 对角色进行一些操控,比如播放动画,切换角色材质
class CharacterTimelineNode : public Node3D
{
    GDCLASS(CharacterTimelineNode, Node3D);

    class CharacterBodyMain* m_Body = nullptr;
    AnimationPlayer* m_AnimationPlayer = nullptr;
    Ref<Animation>  m_Animation;

    void play_action(StringName p_action_name){}
    
    void set_float_value(StringName p_name,float value){}
};

class CharacterAnimator : public RefCounted
{
    GDCLASS(CharacterAnimator, RefCounted);

    List<CharacterAnimatorLayer*> m_LayerList;
    class CharacterBodyMain* m_Body = nullptr;
public:

    void set_body(class CharacterBodyMain* p_body) { m_Body = p_body; }

    void add_layer(const StringName& name,const Ref<CharacterAnimatorLayerConfig>& _mask);

    void clear_layer();

    void on_layer_delete(CharacterAnimatorLayer *p_layer)
    {
        auto it = m_LayerList.begin();
        while(it != m_LayerList.end())
        {
            if((*it) == p_layer)
            {
                it = m_LayerList.erase(it);
                break;
            }
            else
            {
                ++it;
            }
        }
    }
    ~CharacterAnimator()
    {
        clear_layer();
    }

};
VARIANT_ENUM_CAST(CharacterAnimatorLayerConfig::BlendType)
VARIANT_ENUM_CAST(CharacterAnimatorNode2D::BlendType)
#endif