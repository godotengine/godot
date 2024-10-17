#pragma once

#include "core/io/resource.h"
#include "scene/resources/animation.h"
#include "scene/main/node.h"
#include "scene/animation/animation_player.h"
#include "scene/animation/animation_tree.h"

#include "human_animation.h"
#include "../blackboard/blackboard_plan.h"  


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

        ClassDB::bind_method(D_METHOD("set_skeleton_path", "skeleton_path"), &CharacterBoneMap::set_skeleton_path);
        ClassDB::bind_method(D_METHOD("get_skeleton_path"), &CharacterBoneMap::get_skeleton_path);

        ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "bone_map"), "set_bone_map", "get_bone_map");
        ADD_PROPERTY(PropertyInfo(Variant::PACKED_STRING_ARRAY, "bone_names"), "set_bone_names", "get_bone_names");

        ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "human_config", PROPERTY_HINT_RESOURCE_TYPE, "HumanConfig"), "set_human_config", "get_human_config");

        ADD_PROPERTY(PropertyInfo(Variant::STRING, "skeleton_path", PROPERTY_HINT_FILE, "*.tscn,*.scn"), "set_skeleton_path", "get_skeleton_path");

    }

public:
    void set_bone_map(const Dictionary& p_bone_map) { bone_map = p_bone_map; }
    Dictionary get_bone_map() { return bone_map; }
    void set_bone_names(const Vector<String>& p_bone_names) { bone_names = p_bone_names; }
    Vector<String> get_bone_names() { return bone_names; }

    void set_human_config(const Ref<HumanConfig>& p_human_config) { human_config = p_human_config; }
    Ref<HumanConfig> get_human_config() { return human_config; }

    void set_skeleton_path(const String& p_skeleton_path) { skeleton_path = p_skeleton_path; }
    String get_skeleton_path() { return skeleton_path; }

	Dictionary bone_map;
    // 动画名称列表,用来处理非人形动画的情况这类动画原始文件没有模型,会被识别成节点,需要靠这个动画节点名称重新映射成骨骼
    Vector<String> bone_names;
    Ref<HumanConfig> human_config;
	String skeleton_path;
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



