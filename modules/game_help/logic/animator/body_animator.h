#ifndef _BODY_ANIMATOR_H
#define _BODY_ANIMATOR_H

#include "scene/resources/packed_scene.h"
#include "scene/3d/node_3d.h"
#include "scene/3d/skeleton_3d.h"
#include "scene/3d/physics/character_body_3d.h"
#include "animation_help.h"
#include "../character_ai/body_animator_logic.h"
#include "../character_ai/animator_condition.h"
#include "../character_ai/animator_blackboard_set.h"

#include "character_animation_node.h"


class CharacterAnimatorNodeBase;
class CharacterAnimatorLayer;
class CharacterAnimator;
class CharacterAnimationLibraryItem : public RefCounted
{
    GDCLASS(CharacterAnimationLibraryItem, RefCounted);
    static void _bind_methods()
    {
        ClassDB::bind_method(D_METHOD("set_path", "path"), &CharacterAnimationLibraryItem::set_path);
        ClassDB::bind_method(D_METHOD("get_path"), &CharacterAnimationLibraryItem::get_path);

        ClassDB::bind_method(D_METHOD("set_name", "name"), &CharacterAnimationLibraryItem::set_name);
        ClassDB::bind_method(D_METHOD("get_name"), &CharacterAnimationLibraryItem::get_name);

        ADD_PROPERTY(PropertyInfo(Variant::STRING, "path", PROPERTY_HINT_NONE, "",PROPERTY_USAGE_STORAGE), "set_path", "get_path");
        ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "name", PROPERTY_HINT_NONE, "",PROPERTY_USAGE_STORAGE), "set_name", "get_name");

#if TOOLS_ENABLED
        ClassDB::bind_method(D_METHOD("_set_node", "node"), &CharacterAnimationLibraryItem::_set_node);
        ClassDB::bind_method(D_METHOD("_get_node"), &CharacterAnimationLibraryItem::_get_node);
        ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "node", PROPERTY_HINT_RESOURCE_TYPE, "CharacterAnimatorNodeBase", PROPERTY_USAGE_EDITOR), "_set_node", "_get_node");
#endif

    }
public:
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
    void set_path(String p_path) { path = p_path; }
    String get_path() { return path; }

    void set_name(StringName p_name) {
        name = p_name;
    }
    StringName get_name() { return name; }

    void _set_node(Ref<CharacterAnimatorNodeBase> p_node)  {
        if(p_node.is_null())
        {
            return;
        }
        if(p_node->get_path() == "")
        {
            return;
        }
        node = p_node; 
        path = p_node->get_path();
        name = path.get_file().get_basename();
    }
    Ref<CharacterAnimatorNodeBase> _get_node() { return node; }
public:
    Ref<CharacterAnimatorNodeBase> node;
    StringName name;
    String path;
    int is_loaded = 0;
};
// 动画库
class CharacterAnimationLibrary : public Resource
{
    GDCLASS(CharacterAnimationLibrary, Resource);

    static void _bind_methods()
    {
        ClassDB::bind_method(D_METHOD("set_animation_library", "animation_library"), &CharacterAnimationLibrary::set_animation_library);
        ClassDB::bind_method(D_METHOD("get_animation_library"), &CharacterAnimationLibrary::get_animation_library);

        ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "animation_library",PROPERTY_HINT_ARRAY_TYPE,"String"), "set_animation_library", "get_animation_library");

        #if TOOLS_ENABLED


        ADD_GROUP(L"创建动画节点", "editor_");


        ClassDB::bind_method(D_METHOD("set_animator_node_name", "animator_node_name"), &CharacterAnimationLibrary::set_animator_node_name);
        ClassDB::bind_method(D_METHOD("get_animator_node_name"), &CharacterAnimationLibrary::get_animator_node_name);

        ClassDB::bind_method(D_METHOD("set_animator_node_type", "animator_node_type"), &CharacterAnimationLibrary::set_animator_node_type);
        ClassDB::bind_method(D_METHOD("get_animator_node_type"), &CharacterAnimationLibrary::get_animator_node_type);


        ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "editor_animator_node_name"), "set_animator_node_name", "get_animator_node_name");
        ADD_PROPERTY(PropertyInfo(Variant::INT, "editor_animator_node_type", PROPERTY_HINT_ENUM, L"1D,2D,循环最后一个"), "set_animator_node_type", "get_animator_node_type");
        ADD_MEMBER_BUTTON(editor_create_animation_node,L"创建动画节点", CharacterAnimationLibrary);

        #endif
    }

public:

public:
    void set_animation_library(const TypedArray<CharacterAnimationLibraryItem>& p_animation_library) { 
        if(animation_library.size() > 0) {
            return;
        }
        animation_library = p_animation_library;
    }
    TypedArray<CharacterAnimationLibraryItem> get_animation_library() { return animation_library; }


    Ref<CharacterAnimationLibraryItem> get_animation_by_name(StringName p_name)
    {
        if(animations.has(p_name))
        {
            return animations[p_name];
        }
        else
        {
            ERR_PRINT(String("not find animation ") +  p_name.operator String().utf8().get_data());
        }
        return Ref<CharacterAnimationLibraryItem>();
    }
    void init_animation_library()
    {
        if(is_init)
        {
            return;
        }
        for (int i = 0; i < animation_library.size(); i++)
        {
			Ref<CharacterAnimationLibraryItem> item = animation_library[i];
            if(item->get_name()  != StringName())
            {
                animations[item->get_name()] =  item;
            }
        }
        is_init = true;
    }
public:
    enum AnimationNodeType{
        T_CharacterAnimatorNode1D,
        T_CharacterAnimatorNode2D,
        T_CharacterAnimatorLoopLast,
    };
    void set_animator_node_name(String p_animator_node_name) { animator_node_name = p_animator_node_name; }
    String get_animator_node_name() { return animator_node_name; }

    void set_animator_node_type(int p_animator_node_type) { animator_node_type = p_animator_node_type; }
    int get_animator_node_type() { return animator_node_type; }

    String animator_node_name;
    int animator_node_type = T_CharacterAnimatorNode1D;
    DECL_MEMBER_BUTTON(editor_create_animation_node);
public:
    TypedArray<CharacterAnimationLibraryItem> animation_library;
    HashMap<StringName, Ref<CharacterAnimationLibraryItem>> animations;
    bool is_init = false;
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
    Ref<CharacterAnimationLibraryItem> curr_animation;
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
public:

    void _process_logic(const Ref<Blackboard> &p_playback_info,double p_delta,bool is_first = true);
    // 处理动画
    void _process_animator(const Ref<Blackboard> &p_playback_info,double p_delta,bool is_first = true);
    // 处理动画
    void _process_animation(const Ref<Blackboard> &p_playback_info,double p_delta,bool is_first = true);

    void finish_update();
    void layer_blend_apply() ;

    void init(CharacterAnimator* p_animator,const Ref<CharacterAnimatorLayerConfig>& _config)
    {
         m_Animator = p_animator; 
         config = _config;
    }
    CharacterAnimationLogicContext* _get_logic_context()
    {
        return &logic_context;
    }

	void play_animation(const Ref<Animation>& p_anim, bool p_is_loop);
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
public:
    void set_config(const Ref<CharacterAnimatorLayerConfig>& _config) { config = _config; }
    Ref<CharacterAnimatorLayerConfig> get_config() { return config; }
public:
    Vector<Vector2> m_ChildInputVectorArray;
	Vector<int> m_TempCropArray;
protected:
    // 黑板信息
    Ref<Blackboard>           blackboard;
    // 逻辑上下文
    CharacterAnimationLogicContext logic_context;
    // 动画掩码
    Ref<CharacterAnimatorLayerConfig> config;
    List<Ref<CharacterAnimatorNodeBase>> play_list;
    Vector<float> m_TotalAnimationWeight;
    List<CharacterAnimationInstance> m_AnimationInstances;
    class CharacterAnimator* m_Animator = nullptr;
    float blend_weight = 1.0f;
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
        if(p_blackboard_plan == blackboard_plan)
        {
            return;
        }
        if(blackboard_plan.is_valid())
        {
            blackboard_plan->disconnect(LW_NAME(changed), callable_mp(this, &CharacterAnimationLogicNode::_blackboard_changed));
        }
         blackboard_plan = p_blackboard_plan; 
         if(blackboard_plan.is_valid())
         {
             blackboard_plan->connect(LW_NAME(changed), callable_mp(this, &CharacterAnimationLogicNode::_blackboard_changed));
         }
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
        if(start_blackboard_set.is_valid())
        {
            start_blackboard_set->set_blackboard_plan(blackboard_plan);
        }

        if(stop_blackboard_set.is_valid())
        {
            stop_blackboard_set->set_blackboard_plan(blackboard_plan);
        }

	}
    bool get_editor_state() 
    {
        return false; 
    }


    void set_priority(int p_priority) { priority = p_priority; }
    int get_priority() { return priority; }

    void set_player_animation_name(StringName p_player_animation_name) { player_animation_name = p_player_animation_name; }
    StringName get_player_animation_name() { return player_animation_name; }

	void set_enter_condtion(const Ref<CharacterAnimatorCondition>& p_enter_condtion) { enter_condtion = p_enter_condtion; update_blackboard_plan(); }
    Ref<CharacterAnimatorCondition> get_enter_condtion() { return enter_condtion; }

    void set_start_blackboard_set(const Ref<AnimatorBlackboardSet>& p_start_blackboard_set) { start_blackboard_set = p_start_blackboard_set; update_blackboard_plan(); }
    Ref<AnimatorBlackboardSet> get_start_blackboard_set() { return start_blackboard_set; }

    void set_stop_blackboard_set(const Ref<AnimatorBlackboardSet>& p_stop_blackboard_set) { stop_blackboard_set = p_stop_blackboard_set; update_blackboard_plan(); }
    Ref<AnimatorBlackboardSet> get_stop_blackboard_set() { return stop_blackboard_set; }

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

    void _blackboard_changed()
    {
        editor_state_change.call();
    }
    Callable editor_state_change;
public:
    bool is_enter(Blackboard* blackboard)
    {
        if(enter_condtion.is_valid())
        {
            return enter_condtion->is_enable(blackboard);
        }
        return true;
    }
    ~CharacterAnimationLogicNode()
    {
        if(blackboard_plan.is_valid())
        {
            blackboard_plan->disconnect(LW_NAME(changed), callable_mp(this, &CharacterAnimationLogicNode::_blackboard_changed));
        }
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
    // 進入节点设置的黑板
    Ref<AnimatorBlackboardSet> start_blackboard_set;
    // 退出节点设置的黑板
    Ref<AnimatorBlackboardSet> stop_blackboard_set;
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

// 动画层配置实例
class CharacterAnimatorLayerConfigInstance : public RefCounted
{
    GDCLASS(CharacterAnimatorLayerConfigInstance, RefCounted);
    static void _bind_methods()
    {
        ClassDB::bind_method(D_METHOD("set_config", "config"), &CharacterAnimatorLayerConfigInstance::set_config);
        ClassDB::bind_method(D_METHOD("get_config"), &CharacterAnimatorLayerConfigInstance::get_config);

        ClassDB::bind_method(D_METHOD("set_play_animation", "play_animation"), &CharacterAnimatorLayerConfigInstance::set_play_animation);
        ClassDB::bind_method(D_METHOD("get_play_animation"), &CharacterAnimatorLayerConfigInstance::get_play_animation);
        ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "play_animation", PROPERTY_HINT_RESOURCE_TYPE, "Animation"), "set_play_animation", "get_play_animation");

        ADD_MEMBER_BUTTON(editor_play_animation,L"播放动画",CharacterAnimatorLayerConfigInstance);

        ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "config", PROPERTY_HINT_RESOURCE_TYPE, "CharacterAnimatorLayerConfig"), "set_config", "get_config");
    }
public:
	void set_body(class CharacterBodyMain* p_body);
    void change_state(const StringName& p_state_name)
    {
        
    }
	void set_config(const Ref<CharacterAnimatorLayerConfig>& _config)
	{
		config = _config;
		auto_init();
	}
    Ref<CharacterAnimatorLayerConfig> get_config()
    {
        return config;
    }

    void set_play_animation(const Ref<Animation>& p_play_animation)
    {
        play_animation = p_play_animation;
    }

    Ref<Animation> get_play_animation()
    {
        return play_animation;
    }
    
	CharacterAnimatorLayer* get_layer()
	{
		return layer;
	}
	void _process_animator(const Ref<Blackboard>& p_playback_info, double p_delta, bool is_first = true)
	{
		if (layer == nullptr)
		{
			return;
		}

		if (layer->is_active())
		{
			layer->_process_animator(p_playback_info, p_delta, is_first);
		}
	}
	void _process_animation(const Ref<Blackboard>& p_playback_info, double p_delta, bool is_first = true)
	{
		if (layer == nullptr)
		{
			return;
		}

		if (layer->is_active())
		{
			layer->_process_animation(p_playback_info, p_delta, is_first);
		}
	}
	void finish_update()
	{
		if (layer == nullptr)
		{
			return;
		}

		if (layer->is_active())
		{
			layer->finish_update();
		}

	}

protected:
	void auto_init();

protected:
    Ref<Animation> play_animation;
    DECL_MEMBER_BUTTON(editor_play_animation);
protected:
    Ref<CharacterAnimatorLayerConfig> config;
    CharacterAnimatorLayer* layer = nullptr;
    class CharacterBodyMain* m_Body = nullptr;
    bool is_init = false;
};

class CharacterAnimator : public RefCounted
{
    GDCLASS(CharacterAnimator, RefCounted);
    static void _bind_methods();

    List<Ref<CharacterAnimatorLayerConfigInstance>> m_LayerConfigInstanceList;
	class CharacterBodyMain* m_Body = nullptr;
    
public:

    void set_body(class CharacterBodyMain* p_body);

    void add_layer(const Ref<CharacterAnimatorLayerConfig>& _mask);

    void _thread_update_animator(float delta);

    void _thread_update_animation(float delta);

    void finish_update();
    void change_state(const StringName& p_state_name) {
        auto it = m_LayerConfigInstanceList.begin();
        while(it != m_LayerConfigInstanceList.end())
        {
            (*it)->change_state(p_state_name);
            ++it;
        }
    }

    void on_layer_delete(CharacterAnimatorLayer *p_layer) {
        auto it = m_LayerConfigInstanceList.begin();
        while(it != m_LayerConfigInstanceList.end())
        {
            if((*it)->get_layer() == p_layer)
            {
                it = m_LayerConfigInstanceList.erase(it);
                break;
            }
            else
            {
                ++it;
            }
        }
    }
    Ref<CharacterAnimationLibraryItem> get_animation_by_name(const StringName& p_name);
    void set_animation_layer_arrays(TypedArray<CharacterAnimatorLayerConfigInstance> p_animation_layer_arrays) {
		m_LayerConfigInstanceList.clear();
		for (int i = 0; i < p_animation_layer_arrays.size(); ++i) {
			Ref< CharacterAnimatorLayerConfigInstance> ins = p_animation_layer_arrays[i];
			if (ins.is_null()) {
				ins.instantiate();
			}
			ins->set_body(m_Body);
			m_LayerConfigInstanceList.push_back(ins);
		}
    }
    TypedArray<CharacterAnimatorLayerConfigInstance> get_animation_layer_arrays() {
		TypedArray<CharacterAnimatorLayerConfigInstance> rs;
		auto it = m_LayerConfigInstanceList.begin();
		while (it != m_LayerConfigInstanceList.end()) {
			rs.append(*it);
			++it;
		}
		return rs;
	}
    void init() {
        if(m_LayerConfigInstanceList.size() == 0) {
            Ref<CharacterAnimatorLayerConfig> _mask;
            _mask.instantiate();
            add_layer(_mask);
        }
    }

    ~CharacterAnimator() {
    }

};
VARIANT_ENUM_CAST(CharacterAnimatorNodeBase::LoopType)
VARIANT_ENUM_CAST(CharacterAnimationLogicNode::AnimatorAIStopCheckType)
VARIANT_ENUM_CAST(CharacterAnimatorLayerConfig::BlendType)
VARIANT_ENUM_CAST(CharacterAnimationLibrary::AnimationNodeType)
#endif
