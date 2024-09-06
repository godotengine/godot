#pragma once

#include "core/io/resource.h"
#include "scene/resources/animation.h"
#include "scene/main/node.h"
#include "scene/animation/animation_player.h"
#include "scene/animation/animation_tree.h"

#include "modules/limboai/bt/bt_player.h"
class CharacterBoneMap : public Resource
{
    GDCLASS(CharacterBoneMap, Resource);

    static void _bind_methods()
    {
        ClassDB::bind_method(D_METHOD("set_bone_map", "bone_map"), &CharacterBoneMap::set_bone_map);
        ClassDB::bind_method(D_METHOD("get_bone_map"), &CharacterBoneMap::get_bone_map);

        ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "bone_map"), "set_bone_map", "get_bone_map");

    }

public:
    void set_bone_map(const Dictionary& p_bone_map) { bone_map = p_bone_map; }
    Dictionary get_bone_map() { return bone_map; }

	Dictionary bone_map;
};
class CharacterAnimation : public Animation
{
    GDCLASS(CharacterAnimation, Animation);
    static void _bind_methods()
    {
        ClassDB::bind_method(D_METHOD("set_bone_map", "bone_map"), &CharacterAnimation::set_bone_map);
        ClassDB::bind_method(D_METHOD("get_bone_map"), &CharacterAnimation::get_bone_map);
        ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "bone_map", PROPERTY_HINT_RESOURCE_TYPE, "CharacterBoneMap"), "set_bone_map", "get_bone_map");
    }
public:
    void set_bone_map(const Ref<CharacterBoneMap>& p_unity_asset_path) { bone_map = p_unity_asset_path; }
    Ref<CharacterBoneMap> get_bone_map() { return bone_map; }
    Ref<CharacterBoneMap> bone_map;
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

    void set_speed(double p_speed) { speed = p_speed; }
    double get_speed() { return speed; }

    void set_is_clip(bool p_is_clip) { is_clip = p_is_clip; }
    bool get_is_clip() { return is_clip; }

    void set_child_node(const Ref<class CharacterAnimatorNodeBase>& p_child_node) ;
    Ref<class CharacterAnimatorNodeBase> get_child_node();

    void set_animation(Ref<Animation> p_animation) { animation = p_animation; }
    Ref<Animation> get_animation();
    Ref<CharacterBoneMap> get_bone_map();

    void _init();
    float _get_animation_length();
    void _set_animation_scale_by_length(float p_length);
public:
    StringName animation_name;
    // 动画资源路径
    String animation_path;
    // 骨骼映射名称
    String bone_map_path;
    Ref<Animation> animation;
    Ref<CharacterBoneMap> bone_map;
    Ref<class CharacterAnimatorNodeBase> child_node;

	double speed = 1.0f;
    float last_using_time = 0;
    bool is_clip = true;
    bool is_init = false;
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
	struct Blend1dDataConstant
	{
		uint32_t            position_count = 0;
		LocalVector<float>       position_array;
	};
	struct MotionNeighborList
	{
		uint32_t m_Count = 0;
		LocalVector<uint32_t> m_NeighborArray;
	};
	struct Blend2dDataConstant
	{
		uint32_t                    position_count = 0;
		LocalVector<Vector2>             position_array;

		uint32_t                    m_ChildMagnitudeCount = 0;
		LocalVector<float>               m_ChildMagnitudeArray; // Used by type 2
		uint32_t                    m_ChildPairVectorCount = 0;
		LocalVector<Vector2>             m_ChildPairVectorArray; // Used by type 2, (3 TODO)
		uint32_t                    m_ChildPairAvgMagInvCount = 0;
		LocalVector<float>               m_ChildPairAvgMagInvArray; // Used by type 2
		uint32_t                    m_ChildNeighborListCount = 0;
		LocalVector<MotionNeighborList>  m_ChildNeighborListArray; // Used by type 2, (3 TODO)
	};
    void touch() { lastUsingTime = OS::get_singleton()->get_unix_time(); }

    bool is_need_remove(float remove_time) { return OS::get_singleton()->get_unix_time() - lastUsingTime > remove_time; }

public:
    virtual void process_animation(class CharacterAnimatorLayer *p_layer,struct CharacterAnimationInstance *p_playback_info,float total_weight,const Ref<Blackboard> &p_blackboard)
    {

    }
public:
    void _blend_anmation(CharacterAnimatorLayer *p_layer,int child_count,struct CharacterAnimationInstance *p_playback_info,float total_weight,const LocalVector<float> &weight_array,const Ref<Blackboard> &p_blackboard);
    // 统一动画长度
    void _normal_animation_length();
    float _get_animation_length();
    void _set_animation_scale_by_length(float p_length);

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
protected:
    
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
    void add_animation(const Ref<Animation> & p_anim,float p_pos);
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
public:
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
	LocalVector<float> m_WeightArray;
	LocalVector<AnimationMixer::PlaybackInfo> m_ChildAnimationPlaybackArray;
	double delta = 0.0f;
	float time = 0.0f;
	float fadeTotalTime = 0.0f;
	float get_weight()
	{
		if (m_PlayState == PlayState::PS_FadeOut)
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



