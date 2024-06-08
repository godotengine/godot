#ifndef _BODY_ANIMATOR_H
#define _BODY_ANIMATOR_H

#include "scene/resources/packed_scene.h"
#include "scene/animation/animation_player.h"
#include "scene/animation/animation_tree.h"
#include "scene/3d/node_3d.h"
#include "scene/3d/skeleton_3d.h"
#include "scene/3d/physics/character_body_3d.h"
#include "../body_part.h"
#include "animation_help.h"
#include "../character_ai/body_animator_logic.h"


#include "modules/limboai/bt/bt_player.h"

class CharacterAnimatorNodeBase;
// 动画库
class CharacterAnimationLibrary : public Resource
{
    GDCLASS(CharacterAnimationLibrary, Resource);

    static void _bind_methods()
    {
        ClassDB::bind_method(D_METHOD("set_animation_library", "animation_library"), &CharacterAnimationLibrary::set_animation_library);
        ClassDB::bind_method(D_METHOD("get_animation_library"), &CharacterAnimationLibrary::get_animation_library);

        ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "animation_library",PROPERTY_HINT_ARRAY_TYPE,"String"), "set_animation_library", "get_animation_library");
    }

public:
    class AnimationItem : public Resource
    {
	public:
        String path;
        Ref<CharacterAnimatorNodeBase> node;
		void load()
		{
			if (is_loaded == 0)
			{
                ResourceLoader::load_threaded_request(path);
                is_loaded = 1;
			}
		}
        Ref<CharacterAnimatorNodeBase> get_node()
        {
            if(is_loaded == 1)
            {
               node = ResourceLoader::load_threaded_get(path);
               is_loaded = 2;
            }
            return node;
        }
        int is_loaded = 0;
    };

public:
    void set_animation_library(const TypedArray<String>& p_animation_library) { animation_library = p_animation_library; init_animation_library();}
    TypedArray<String> get_animation_library() { return animation_library; }
    Ref<CharacterAnimationLibrary::AnimationItem> get_animation_by_name(StringName p_name)
    {
        if(animations.has(p_name))
        {
            return animations[p_name];
        }
        else
        {
            ERR_PRINT(String("not find animation ") +  p_name.operator String().utf8().get_data());
        }
        return Ref<CharacterAnimationLibrary::AnimationItem>();
    }
    void init_animation_library()
    {
        if(is_init)
        {
            return;
        }
        for (int i = 0; i < animation_library.size(); i++)
        {
			Ref<AnimationItem> item;
			item.instantiate();
            item->path = animation_library[i];
            String nm = item->path.get_file().get_basename();
            if(nm.size() > 0)
            {
                animations.insert(nm, item);
            }
        }
        is_init = true;
    }
public:
    bool is_init = false;
    TypedArray<String> animation_library;
    HashMap<StringName, Ref<AnimationItem>> animations;
};

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
    Dictionary bone_reset;
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
    Ref<Animation> retarget_animation;
    Ref<CharacterBoneMap> bone_map;
    Ref<class CharacterAnimatorNodeBase> child_node;
    bool is_init = false;
    float last_using_time = 0;
};
class CharacterAnimatorNodeBase : public Resource
{
    GDCLASS(CharacterAnimatorNodeBase, Resource);
    static void bind_methods();

public:
    enum LoopType
    {
        LOOP_Once,
        LOOP_ClampCount,
        LOOP_PingPongOnce,
        LOOP_PingPongCount,
    };
    void touch() { lastUsingTime = OS::get_singleton()->get_unix_time(); }

    bool is_need_remove(float remove_time) { return OS::get_singleton()->get_unix_time() - lastUsingTime > remove_time; }

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

    void set_fade_out_time(float p_fade_out_time) { fade_out_time = p_fade_out_time; }
    float get_fade_out_time() { return fade_out_time; }

    void set_loop(LoopType p_loop) { isLoop = p_loop; }
    LoopType get_loop() { return isLoop; }

    void set_loop_count(int p_loop_count) { loop_count = p_loop_count; }
    int get_loop_count() { return loop_count; }
public:
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

	void _add_animation_item(const Ref<CharacterAnimationItem>& p_anim)
	{
		animation_arrays.push_back(p_anim);
	}
    protected:
    
    TypedArray<CharacterAnimationItem>    animation_arrays;
    StringName               black_board_property;
    StringName               black_board_property_y;
    float fade_out_time = 0.0f;
    float lastUsingTime = 0.0f;
    LoopType isLoop = LOOP_Once;
    int loop_count = 0;


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

        BIND_ENUM_CONSTANT(SimpleDirectionnal2D);
        BIND_ENUM_CONSTANT(FreeformDirectionnal2D);
        BIND_ENUM_CONSTANT(FreeformCartesian2D);
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
        if(m_PlayState == PlayState::PS_FadeOut)
        {
            if(node->get_fade_out_time() <= 0.0f)
                return 0;
            return MAX(0.0f,1.0f - fadeTotalTime / node->get_fade_out_time());
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
class CharacterAnimatorLayer;

// 时间线资源,这个主要用来Animation 对角色进行一些操控,比如播放动画,切换角色材质
class CharacterTimelineNode : public Node3D
{
    GDCLASS(CharacterTimelineNode, Node3D);
    static void _bind_methods()
    {

    }
    public:

    class CharacterBodyMain* m_Body = nullptr;
    AnimationPlayer* m_AnimationPlayer = nullptr;
    Ref<Animation>  m_Animation;

    void play_action(StringName p_action_name){}
    
    void set_float_value(StringName p_name,float value){}
};
// 动画逻辑上下文
struct CharacterAnimationLogicContext
{
    // 动画逻辑
    Ref<CharacterAnimationLogicLayer> animation_logic;
    // 当前状态名称
    StringName last_name;
    // 当前状态名称
    StringName curr_name;
    Ref<CharacterAnimationLogicRoot> curr_state_root;
    // 当前处理的逻辑节点
    Ref<CharacterAnimationLogicNode>   curr_logic;
    Ref<CharacterAnimationLibrary::AnimationItem> curr_animation;
    // 执行时长
    float time = 0.0f;
    bool is_start = false;
    float curr_animation_play_time = 0.0f;
    float curr_animation_time_length = 0.0f;

};

// 动画分层
class CharacterAnimatorLayer: public AnimationMixer
{
    GDCLASS(CharacterAnimatorLayer, AnimationMixer);
    static void bind_methods()
    {
    }

    List<Ref<CharacterAnimatorNodeBase>> play_list;
public:
    // 黑板信息
    Ref<Blackboard>           blackboard;
    // 逻辑上下文
    CharacterAnimationLogicContext logic_context;
    // 动画掩码
    Ref<CharacterAnimatorLayerConfig> config;

    void _process_logic(const Ref<Blackboard> &p_playback_info,double p_delta,bool is_first = true);
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
    CharacterAnimationLogicContext* _get_logic_context()
    {
        return &logic_context;
    }

    bool play_animation(Ref<CharacterAnimatorNodeBase> p_node);
    void play_animation(const StringName& p_node_name);
    void change_state(const StringName& p_state_name)
    {
        if(logic_context.curr_name == p_state_name)
        {
            logic_context.last_name = logic_context.curr_name;
            logic_context.curr_name = p_state_name;
        }
    }
    ~CharacterAnimatorLayer();
};

// 动画逻辑节点
class CharacterAnimationLogicNode : public Resource
{
    GDCLASS(CharacterAnimationLogicNode,Resource)
    static void _bind_methods();

public:

    enum AnimatorAIStopCheckType
    {
        // 固定生命期
        Life,
        AnimationLengthScale,
        // 通过检测条件结束
        Condition,
        Script
    };
    struct SortCharacterAnimationLogicNode {
        bool operator()(const Ref<CharacterAnimationLogicNode> &l, const Ref<CharacterAnimationLogicNode> &r) const {
            int lp = 0;
            int rp = 0;
            if(l.is_valid()){
                lp = l->priority;
            }
            if(r.is_valid()){
                rp = r->priority;
            }

            return lp > lp;
        }
    };

public:
    void set_blackboard_plan(const Ref<BlackboardPlan>& p_blackboard_plan) 
    {
         blackboard_plan = p_blackboard_plan; 
         init_blackboard(blackboard_plan);
		 update_blackboard_plan();
    }
    Ref<BlackboardPlan> get_blackboard_plan() { return blackboard_plan; }
	void update_blackboard_plan()
	{
		if (enter_condtion.is_valid()) {
			enter_condtion->set_blackboard_plan(blackboard_plan);
		}
		if (stop_check_condtion.is_valid()) {
			stop_check_condtion->set_blackboard_plan(blackboard_plan);
		}

	}


    void set_priority(int p_priority) { priority = p_priority; }
    int get_priority() { return priority; }

    void set_player_animation_name(StringName p_player_animation_name) { player_animation_name = p_player_animation_name; }
    StringName get_player_animation_name() { return player_animation_name; }

	void set_enter_condtion(const Ref<CharacterAnimatorCondition>& p_enter_condtion) { enter_condtion = p_enter_condtion; update_blackboard_plan(); }
    Ref<CharacterAnimatorCondition> get_enter_condtion() { return enter_condtion; }

    void set_check_stop_delay_time(float p_check_stop_delay_time) { check_stop_delay_time = p_check_stop_delay_time; }
    float get_check_stop_delay_time() { return check_stop_delay_time; }

    void set_life_time(float p_life_time) { life_time = p_life_time; }
    float get_life_time() { return life_time; }

    void set_stop_check_type(AnimatorAIStopCheckType p_stop_check_type) { stop_check_type = p_stop_check_type; }
    AnimatorAIStopCheckType get_stop_check_type() { return stop_check_type; }

    void set_stop_check_condtion(const Ref<CharacterAnimatorCondition>& p_stop_check_condtion) { stop_check_condtion = p_stop_check_condtion; update_blackboard_plan();}
    Ref<CharacterAnimatorCondition> get_stop_check_condtion() { return stop_check_condtion; }

    void set_stop_check_anmation_length_scale(float p_stop_check_anmation_length_scale) { anmation_scale = p_stop_check_anmation_length_scale; }
    float get_stop_check_anmation_length_scale() { return anmation_scale; }
    
    static void init_blackboard(Ref<BlackboardPlan> p_blackboard_plan);

public:
    virtual void process_start(CharacterAnimatorLayer* animator,Blackboard* blackboard);
	virtual void process(CharacterAnimatorLayer* animator,Blackboard* blackboard, double delta);
	virtual bool check_stop(CharacterAnimatorLayer* animator,Blackboard* blackboard);
    virtual void process_stop(CharacterAnimatorLayer* animator,Blackboard* blackboard);

public:
    bool is_enter(Blackboard* blackboard)
    {
        if(enter_condtion.is_valid())
        {
            return enter_condtion->is_enable(blackboard);
        }
        return true;
    }

private:
	GDVIRTUAL2(_animation_process_start,CharacterAnimatorLayer*,Blackboard*)
    GDVIRTUAL2(_animation_process_stop,CharacterAnimatorLayer*,Blackboard*)
	GDVIRTUAL3(_animation_process,CharacterAnimatorLayer*,Blackboard*, double)
	GDVIRTUAL2R(bool,_check_stop,CharacterAnimatorLayer*,Blackboard*)


public:
    // 优先级
    int priority = 0;
    // 播放的动画名称
    StringName player_animation_name;
    // 进入条件
    Ref<CharacterAnimatorCondition> enter_condtion;
    Ref<BlackboardPlan> blackboard_plan;
    // 检测结束等待时间
    float check_stop_delay_time = 0.0f;
    AnimatorAIStopCheckType stop_check_type = Life;
    // 生命期
    float life_time = 0.0f;
    float anmation_scale = 1.0f;
    
    // 退出检测条件
    Ref<CharacterAnimatorCondition> stop_check_condtion;
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
    void change_state(const StringName& p_state_name)
    {
        auto it = m_LayerList.begin();
        while(it != m_LayerList.end())
        {
            (*it)->change_state(p_state_name);
            ++it;
        }
    }

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
    Ref<CharacterAnimationLibrary::AnimationItem> get_animation_by_name(const StringName& p_name);
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
VARIANT_ENUM_CAST(CharacterAnimatorNodeBase::LoopType)
VARIANT_ENUM_CAST(CharacterAnimationLogicNode::AnimatorAIStopCheckType)
VARIANT_ENUM_CAST(CharacterAnimatorLayerConfig::BlendType)
VARIANT_ENUM_CAST(CharacterAnimatorNode2D::BlendType)
#endif
