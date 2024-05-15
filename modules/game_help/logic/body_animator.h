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
class CharacterBoneMap : public Resource
{
    GDCLASS(CharacterBoneMap, Resource);

    static void _bind_methods()
    {
        ClassDB::bind_method(D_METHOD("set_bone_map", "bone_map"), &CharacterBoneMap::set_bone_map);
        ClassDB::bind_method(D_METHOD("get_bone_map"), &CharacterBoneMap::get_bone_map);

        ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "bone_map"), "set_bone_map", "get_bone_map");

        IMP_GODOT_PROPERTY(bool,is_init_skeleton);
        IMP_GODOT_PROPERTY(String, ref_skeleton_file_path);
        IMP_GODOT_PROPERTY(bool, is_by_sekeleton_file);
    }

public:
    void set_bone_map(const Dictionary& p_bone_map) { bone_map = p_bone_map; }
    Dictionary get_bone_map() { return bone_map; }
    void init_skeleton_bone_map();

	Dictionary bone_map;
    DECL_GODOT_PROPERTY(bool,is_init_skeleton,false);
    DECL_GODOT_PROPERTY(String, ref_skeleton_file_path,"");
    DECL_GODOT_PROPERTY(bool, is_by_sekeleton_file,false);
};
class CharacterAnimationItem : public Resource
{
    GDCLASS(CharacterAnimationItem, Resource);
    static void bind_methods();

public:
    void set_animation_name(const String& p_animation_name) { animation_name = p_animation_name; }
    String get_animation_name() { return animation_name; }



    void set_animation_path(const String& p_animation_path) { animation_path = p_animation_path; }
    String get_animation_path() { return animation_path; }


    void set_bone_map_path(const String& p_bone_map_path) { bone_map_path = p_bone_map_path; }
    String get_bone_map_path() { return bone_map_path; }

    void set_speed(float p_speed) { speed = p_speed; }
    float get_speed() { return speed; }

    void set_is_clip(bool p_is_clip) { is_clip = p_is_clip; }
    bool get_is_clip() { return is_clip; }

    void set_child_node(const Ref<class CharacterAnimatorNodeBase>& p_child_node) ;
    Ref<class CharacterAnimatorNodeBase> get_child_node();

    Ref<Animation> get_animation();
    Ref<CharacterBoneMap> get_bone_map();

    void _init();
    float _get_animation_length();
    void _set_animation_scale_by_length(float p_length);


    StringName animation_name;
    // 动画资源路径
    String animation_path;
    // 骨骼映射名称
    String bone_map_path;
    float speed = 1.0f;
    bool is_clip = true;
    Ref<Animation> animation;
    Ref<CharacterBoneMap> bone_map;
    Ref<class CharacterAnimatorNodeBase> child_node;
    bool is_init = false;
};
class CharacterAnimatorNodeBase : public Resource
{
    GDCLASS(CharacterAnimatorNodeBase, Resource);
    static void bind_methods();

public:
    float fadeOutTime = 0.0f;
    bool isLoop = false;

public:
    virtual void process_animation(class CharacterAnimatorLayer *p_layer,struct CharacterAnimationInstance *p_playback_info,float total_weight,const Ref<Blackboard> &p_blackboard)
    {

    }
public:
    void _blend_anmation(CharacterAnimatorLayer *p_layer,int child_count,struct CharacterAnimationInstance *p_playback_info,float total_weight,const Vector<float> &weight_array,const Ref<Blackboard> &p_blackboard);
    // 统一动画长度
    void _normal_animation_length();
    float _get_animation_length();
    void _set_animation_scale_by_length(float p_length);

    void set_animation_arrays(TypedArray<CharacterAnimationItem> p_animation_arrays) { animation_arrays = p_animation_arrays; }
    TypedArray<CharacterAnimationItem> get_animation_arrays() { return animation_arrays; }

    void set_black_board_property(const StringName& p_black_board_property) { black_board_property = p_black_board_property; }
    StringName get_black_board_property() { return black_board_property; }

    void set_black_board_property_y(const StringName& p_black_board_property_y) { black_board_property_y = p_black_board_property_y; }
    StringName get_black_board_property_y() { return black_board_property_y; }
    virtual void _init();

    
    TypedArray<CharacterAnimationItem>    animation_arrays;
    StringName               black_board_property;
    StringName               black_board_property_y;

    struct Blend1dDataConstant
    {

        Blend1dDataConstant() : position_count(0)
        {
        }

        uint32_t            position_count;
        Vector<float>       position_array;
    };
    struct MotionNeighborList
    {

        MotionNeighborList() : m_Count(0)
        {
        }

        uint32_t m_Count;
        Vector<uint32_t> m_NeighborArray;
    };  
    struct Blend2dDataConstant
    {

        Blend2dDataConstant()
        {
        }

        uint32_t                    position_count;
        Vector<Vector2>             position_array;

        uint32_t                    m_ChildMagnitudeCount;
        Vector<float>               m_ChildMagnitudeArray; // Used by type 2
        uint32_t                    m_ChildPairVectorCount;
        Vector<Vector2>             m_ChildPairVectorArray; // Used by type 2, (3 TODO)
        uint32_t                    m_ChildPairAvgMagInvCount;
        Vector<float>               m_ChildPairAvgMagInvArray; // Used by type 2
        uint32_t                    m_ChildNeighborListCount;
        Vector<MotionNeighborList>  m_ChildNeighborListArray; // Used by type 2, (3 TODO)

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
    
    static void bind_methods()
    {
        ClassDB::bind_method(D_METHOD("set_position_count", "count"), &CharacterAnimatorNode1D::set_position_count);
        ClassDB::bind_method(D_METHOD("get_position_count"), &CharacterAnimatorNode1D::get_position_count);
        ClassDB::bind_method(D_METHOD("set_position_array", "array"), &CharacterAnimatorNode1D::set_position_array);
        ClassDB::bind_method(D_METHOD("get_position_array"), &CharacterAnimatorNode1D::get_position_array);

        ADD_PROPERTY(PropertyInfo(Variant::INT, "position_count"), "set_position_count", "get_position_count");
        ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "position_array"), "set_position_array", "get_position_array");
    }
public:
    virtual void process_animation(class CharacterAnimatorLayer *p_layer,CharacterAnimationInstance *p_playback_info,float total_weight,const Ref<Blackboard> &p_blackboard) override;

    void set_position_count(uint32_t p_count) { blend_data.position_count = p_count; }
    uint32_t get_position_count() { return blend_data.position_count; }

    void set_position_array(Vector<float> p_array) { blend_data.position_array = p_array; }
    Vector<float> get_position_array() { return blend_data.position_array; }
public:
    Blend1dDataConstant   blend_data;
};
class CharacterAnimatorNode2D : public CharacterAnimatorNodeBase
{
    GDCLASS(CharacterAnimatorNode2D, CharacterAnimatorNodeBase);
    static void bind_methods()
    {
        ClassDB::bind_method(D_METHOD("set_position_count", "count"), &CharacterAnimatorNode1D::set_position_count);
        ClassDB::bind_method(D_METHOD("get_position_count"), &CharacterAnimatorNode1D::get_position_count);
        ClassDB::bind_method(D_METHOD("set_position_array", "array"), &CharacterAnimatorNode1D::set_position_array);
        ClassDB::bind_method(D_METHOD("get_position_array"), &CharacterAnimatorNode1D::get_position_array);

        ADD_PROPERTY(PropertyInfo(Variant::INT, "position_count"), "set_position_count", "get_position_count");
        ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "position_array"), "set_position_array", "get_position_array");
    }
public:
    enum BlendType
    {
        SimpleDirectionnal2D = 1,
        FreeformDirectionnal2D = 2,
        FreeformCartesian2D = 3,
    };
    virtual void process_animation(class CharacterAnimatorLayer *p_layer,CharacterAnimationInstance *p_playback_info,float total_weight,const Ref<Blackboard> &p_blackboard) override;

    void set_blend_type(BlendType p_blend_type) { blend_type = (BlendType)p_blend_type; }
    BlendType get_blend_type() { return blend_type; }

    void set_position_count(uint32_t p_count) { blend_data.position_count = p_count; }
    uint32_t get_position_count() { return blend_data.position_count; }

    void set_position_array(Vector<Vector2> p_array) { blend_data.position_array = p_array; }
    Vector<Vector2> get_position_array() { return blend_data.position_array; }


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
enum AnimatorAICompareType
{
    //[LabelText("等于")]
    Equal,

    //[LabelText("小于")]
    Less,
    //[LabelText("小于等于")]
    LessEqual,
    //[LabelText("大于")]
    Greater,
    //[LabelText("大于等于")]
    GreaterEqual,
    //[LabelText("不等于")]
    NotEqual,
};
class AnimatorAIStateFloatCondition : public RefCounted
{
    StringName propertyName;
    AnimatorAICompareType compareType;
    float value;

};
// int类型
class AnimatorAIStateIntCondition : public RefCounted
{
    StringName propertyName;
    AnimatorAICompareType compareType;
    float value;

};
// 字符串表达式
class AnimatorAIStateStringNameCondition : public RefCounted
{
    StringName propertyName;
    StringName value;
};
// 角色动画的条件
class CharacterAnimatorConditionList : public RefCounted
{
    // 判断类型
    enum JudgeType
    {
        // 只要一个属性通过就代表通过
        Or,
        // 必须全部满足
        And
    };

public:
    JudgeType judge_type;
    bool is_include = false;
    LocalVector<Ref<AnimatorAIStateFloatCondition>> conditions_float;
    LocalVector<Ref<AnimatorAIStateIntCondition>> conditions_int;
    LocalVector<Ref<AnimatorAIStateStringNameCondition>> conditions_string_names;
    
};
// 角色动画的条件
class CharacterAnimatorCondition : public RefCounted
{

public:
    Ref<CharacterAnimatorConditionList> include_condition;
    Ref<CharacterAnimatorConditionList> exclude_condition;
    
};
class CharacterAnimatorLayer;
enum AnimatorAIStopCheckType
{
    // 固定生命期
    Life,
    PlayCount,
    // 通过检测条件结束
    Condition,
    Script,
};
// 动画逻辑节点
class CharacterAnimationLogicNode : public Resource
{


private:
    
	virtual void animation_process(CharacterAnimatorLayer* animator,Blackboard* blackboard, double delta)const
    {

    }
	virtual bool check_stop(CharacterAnimatorLayer* animator,Blackboard* blackboard)const
    {
        if (GDVIRTUAL_IS_OVERRIDDEN(_check_stop)) {
            bool is_stop = false;
            GDVIRTUAL_CALL(_check_stop, animator,blackboard, is_stop);
            return is_stop;
        }
    }
	GDVIRTUAL3(_animation_process,CharacterAnimatorLayer*,Blackboard*, double)
	GDVIRTUAL2R(bool,_check_stop,CharacterAnimatorLayer*,Blackboard*)


public:
    StringName node_name;
    // 优先级
    int priority = 0;
    // 播放的动画名称
    StringName player_animation_name;
    // 进入条件
    Ref<CharacterAnimatorCondition> enter_condtion;
    Ref<BlackboardPlan> blackboard_plan;
    // 检测结束等待时间
    float check_stop_delay_time = 0.0f;
    // 生命期
    float life_time = 0.0f;
    AnimatorAIStopCheckType stop_check_type;
    
    // 退出检测条件
    Ref<CharacterAnimatorCondition> stop_check_condtion;
};
struct SortCharacterAnimationLogicNode {
	bool operator()(const Ref<CharacterAnimationLogicNode> &l, const Ref<CharacterAnimationLogicNode> &r) const {
		return l.priority > r.priority;
	}
};
class CharacterAnimationLogicRoot : public Resource
{
    void sort()
    {
        node_list.sort_custom<SortCharacterAnimationLogicNode>();
    }
    static int compare_priority(const Ref<CharacterAnimationLogicNode>& p_a, const Ref<CharacterAnimationLogicNode>& p_b)
    {

    }

public:
    bool is_need_sort = true;
    LocalVector<Ref<CharacterAnimationLogicNode>>  node_list;
};
// 动画逻辑层信息
/*
    每一个层里面可以处理多个动画状态
    每一个状态都有一套动画配置
*/
class CharacterAnimationLogicLayer : public Resource
{
    GDCLASS(CharacterAnimationLogicLayer, Resource);

public:

    void set_default_state_name(const StringName& p_default_state_name) { default_state_name = p_default_state_name; }
    StringName get_default_state_name() { return default_state_name; }


public:
    //  默认状态名称
    StringName default_state_name;
    HashMap<StringName, Ref<CharacterAnimationLogicRoot>> state_map;

};
// 动画逻辑上下文
struct CharacterAnimationLogicContext
{
    // 动画逻辑
    Ref<CharacterAnimationLogicLayer> animation_logic;
    // 当前状态名称
    StringName state_name;
    Ref<CharacterAnimationLogicRoot> curr_state_root;
    // 当前处理的逻辑节点
    Ref<CharacterAnimationLogicLayer>   logic;
    // 执行时长
    float time = 0.0f;
    bool is_start = false;

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

// 动画分层
class CharacterAnimatorLayer: public AnimationMixer
{
    GDCLASS(CharacterAnimatorLayer, AnimationMixer);

    List<Ref<CharacterAnimatorNodeBase>> play_list;
public:
    // 黑板信息
    Ref<Blackboard>           blackboard;
    // 逻辑上下文
    CharacterAnimationLogicContext logic_context;
    // 动画掩码
    Ref<CharacterAnimatorLayerConfig> config;

    // 处理动画
    void _process_animation(const Ref<Blackboard> &p_playback_info,double p_delta,bool is_first = true);
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
class CharacterAnimator : public RefCounted
{
    GDCLASS(CharacterAnimator, RefCounted);
    static void _bind_methods();

    List<CharacterAnimatorLayer*> m_LayerList;
    TypedArray<CharacterAnimatorLayerConfig> animation_layer_arrays;
    class CharacterBodyMain* m_Body = nullptr;
    bool is_init = false;
public:

    void set_body(class CharacterBodyMain* p_body);

    void add_layer(const Ref<CharacterAnimatorLayerConfig>& _mask);

    void update_animation(float delta);

    void create_layers();

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
    void set_animation_layer_arrays(TypedArray<CharacterAnimatorLayerConfig> p_animation_layer_arrays)
    {
        animation_layer_arrays = p_animation_layer_arrays;        
        create_layers();
    }
    TypedArray<CharacterAnimatorLayerConfig> get_animation_layer_arrays() { return animation_layer_arrays; }

    ~CharacterAnimator()
    {
        clear_layer();
    }

};
VARIANT_ENUM_CAST(CharacterAnimatorLayerConfig::BlendType)
VARIANT_ENUM_CAST(CharacterAnimatorNode2D::BlendType)
#endif