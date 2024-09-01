
#include "../unity/unity_link_server.h"
#include "core/io/json.h"
#include "scene/resources/texture.h"
#include "scene/gui/option_button.h"
#include "scene/gui/margin_container.h"
#include "scene/gui/box_container.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/check_button.h"
#include "scene/gui/slider.h"
#include "scene/gui/check_box.h"
#include "scene/gui/separator.h"
#include "scene/resources/texture.h"
#include "scene/gui/label.h"
#include "scene/gui/flow_container.h"
#include "unity_link_server_editor_plugin.h"
#include "../logic/body_main.h"
#include "../logic/data_table_manager.h"



#include "condition_editor.h"
#include "blackboard_set_editor.h"

#if TOOLS_ENABLED
#include "editor/editor_node.h"
#include "editor/editor_properties.h"
#include "editor/editor_properties_array_dict.h"
#include "editor/editor_log.h"
#include "editor/editor_node.h"
#include "editor/editor_settings.h"
#include "editor/dependency_editor.h"
#include "editor/editor_file_system.h"

#include "editor/plugins/editor_plugin.h"
#include "editor/editor_inspector.h"
#include "editor/plugins/skeleton_3d_editor_plugin.h"

#include "../unity/unity_link_server.h"
#include "beehave_graph_editor.h"
#include "body_main_editor.h"
#endif
class CharacterBodyMainLable : public HBoxContainer
{
	GDCLASS(CharacterBodyMainLable, HBoxContainer);
	static void _bind_methods() {
		
	}
public:

	CharacterBodyMainLable()
	{

		label = memnew(Label);
		label->set_h_size_flags(SIZE_EXPAND_FILL);
		label->set_text(L"角色编辑面板");
		label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
		add_child(label);

		run_ai_button = memnew(CheckButton);
		run_ai_button->set_text(L"运行AI");
		run_ai_button->set_tooltip_text(L"运行AI,如果调试行为树,必须关闭此选项");
		run_ai_button->connect("toggled", callable_mp(this, &CharacterBodyMainLable::on_run_ai_toggled));
		add_child(run_ai_button);

	}
	void _notification(int p_what)
	{
		switch (p_what) {
			case NOTIFICATION_ENTER_TREE: {
				if(body_main != nullptr)
				{
					CharacterBodyMain::get_curr_editor_player() = body_main->get_instance_id();
				}
			} break;
			case NOTIFICATION_EXIT_TREE: {
				// 解除绑定黑板设置回调
				if(body_main != nullptr)
				{
					if(CharacterBodyMain::get_curr_editor_player() == body_main->get_instance_id())
					{
						CharacterBodyMain::get_curr_editor_player() = ObjectID();
					}
				}
			} break;
		}
	}
	void set_body_main(CharacterBodyMain* p_body_main)
	{
		body_main = p_body_main;
	}
	void on_run_ai_toggled(bool p_toggled)
	{
		body_main->set_editor_run_ai(p_toggled);
	}
protected:
	CharacterBodyMain* body_main = nullptr;
	Label* label = nullptr;
	CheckButton* run_ai_button = nullptr;
};


#ifdef TOOLS_ENABLED
// 逻辑的根节点分段
class AnimationLogicRootNodeProperty : public EditorPropertyArray
{
	GDCLASS(AnimationLogicRootNodeProperty, EditorPropertyArray);
public:
	void on_reorder_button_gui_input(const Ref<InputEvent> &p_event)
	{
		_reorder_button_gui_input(p_event);
	}
	void on_reorder_button_up()
	{
		_reorder_button_up();
	}
	void on_reorder_button_down(int p_idx)
	{
		_reorder_button_down(p_idx);
	}
	void on_change_type(Object *p_button, int p_slot_index)
	{
		_change_type(p_button,p_slot_index);
	}
	void on_remove_pressed(int p_idx)
	{
		_remove_pressed(p_idx);
	}
	void on_update_state()
	{
		for(uint32_t i=0;i<slots.size();i++)
		{
			EditorPropertyArray::Slot & slot = slots[i];
			if(slot.state_button == nullptr)
			{
				continue;
			}
			Ref<CharacterAnimationLogicNode> node = node_object->get_node(slots[i].index);
			if(node.is_null())
			{
				continue;
			}
			bool rs = node->get_editor_state();
			if(rs)
			{
				slot.state_button->set_pressed(true);
				slot.state_button->set_modulate(Color(0, 0.92549, 0.164706, 1));
			}
			else
			{
				slot.state_button->set_pressed(false);
				slot.state_button->set_modulate(Color(1, 0.255238, 0.196011, 1));
			}
		}
	}
	virtual void _on_clear_slots()
	{
		for(uint32_t i=0;i<slots.size();i++)
		{
			EditorPropertyArray::Slot & slot = slots[i];
			if(slot.state_button == nullptr)
			{
				continue;
			}
			Ref<CharacterAnimationLogicNode> node = node_object->get_node(slots[i].index);
			if(node.is_null())
			{
				continue;
			}
			node->editor_state_change = Callable();
		}
	}
	virtual void _create_new_property_slot() override
	{
		int idx = slots.size();
		HBoxContainer *hbox = memnew(HBoxContainer);

		Button *reorder_button = memnew(Button);
		reorder_button->set_icon(get_editor_theme_icon(SNAME("TripleBar")));
		reorder_button->set_default_cursor_shape(Control::CURSOR_MOVE);
		reorder_button->set_disabled(is_read_only());
		reorder_button->connect(SceneStringName(gui_input), callable_mp(this, &AnimationLogicRootNodeProperty::on_reorder_button_gui_input));
		reorder_button->connect(SNAME("button_up"), callable_mp(this, &AnimationLogicRootNodeProperty::on_reorder_button_up));
		reorder_button->connect(SNAME("button_down"), callable_mp(this, &AnimationLogicRootNodeProperty::on_reorder_button_down).bind(idx));

		hbox->add_child(reorder_button);
		EditorProperty *prop = memnew(EditorPropertyNil);
		hbox->add_child(prop);

		Ref<CharacterAnimationLogicNode> node = node_object->get_node(idx + page_index * page_length);
		bool rs = node->get_editor_state();
		// 增加状态按钮
		CheckButton *state_button = memnew(CheckButton);
		state_button->set_layout_mode(LayoutMode::LAYOUT_MODE_CONTAINER);
		state_button->set_disabled(true);
		state_button->set_focus_mode(FOCUS_NONE);
		if(rs)
		{
			state_button->set_pressed(true);
			state_button->set_modulate(Color(0, 0.92549, 0.164706, 1));
		}
		else{
			state_button->set_pressed(false);
			state_button->set_modulate(Color(1, 0.255238, 0.196011, 1));
		}
		node->editor_state_change = callable_mp(this, &AnimationLogicRootNodeProperty::on_update_state);
		hbox->add_child(state_button);

		bool is_untyped_array = object->get_array().get_type() == Variant::ARRAY && subtype == Variant::NIL;

		if (is_untyped_array) {
			Button *edit_btn = memnew(Button);
			edit_btn->set_icon(get_editor_theme_icon(SNAME("Edit")));
			edit_btn->set_disabled(is_read_only());
			edit_btn->connect(SceneStringName(pressed), callable_mp(this, &AnimationLogicRootNodeProperty::on_change_type).bind(edit_btn, idx));
			hbox->add_child(edit_btn);
		} else {
			Button *remove_btn = memnew(Button);
			remove_btn->set_icon(get_editor_theme_icon(SNAME("Remove")));
			remove_btn->set_disabled(is_read_only());
			remove_btn->connect(SceneStringName(pressed), callable_mp(this, &AnimationLogicRootNodeProperty::on_remove_pressed).bind(idx));
			hbox->add_child(remove_btn);
		}
		property_vbox->add_child(hbox);

		EditorPropertyArray::Slot slot;
		slot.prop = prop;
		slot.object = object;
		slot.container = hbox;
		slot.reorder_button = reorder_button;
		slot.state_button = state_button;
		slot.set_index(idx + page_index * page_length);
		slots.push_back(slot);
	}
public:
	Ref<CharacterAnimationLogicRoot> node_object;
};


#include "../logic/beehave/composites/parallel.h"


class ConditionList_ED 
{

public:
	static bool _parse_condition_property(EditorInspectorPlugin *p_plugin,const Ref<CharacterAnimatorCondition>& object, Variant::Type type, const String& name, PropertyHint hint_type, const String& hint_string, BitField<PropertyUsageFlags> usage_flags, bool wide)
	{
		if(name == "include_condition")
		{
			ConditionSection* section = memnew(ConditionSection);
			section->setup(object,true);
			section->set_category_name("Include Condition");
			TypedArray<Ref<AnimatorAIStateConditionBase>>  condition = object->get_include_condition();
			for(int32_t i = 0; i < condition.size(); ++i)
			{
				Condition_ED* bt = memnew(Condition_ED);
				if(i & 1)
				{
					bt->set_modulate(Color(0.814023, 0.741614, 1, 1));
				}
				bt->setup(section,object,condition[i],true);
				section->add_condition( bt);
			}
			ConditionListButton_ED* bt = memnew(ConditionListButton_ED);
			bt->setup(object,true);
			section->add_condition( bt);
			p_plugin->add_custom_control( section);
			return true;
		}
		if(name == "exclude_condition")
		{
			
			ConditionSection* section = memnew(ConditionSection);
			section->setup(object,false);
			section->set_category_name("Exclude Condition");
			TypedArray<Ref<AnimatorAIStateConditionBase>>  condition = object->get_exclude_condition();
			for(int32_t i = 0; i < condition.size(); ++i)
			{
				Condition_ED* bt = memnew(Condition_ED);
				if(i & 1)
				{
					bt->set_modulate(Color(0.814023, 0.741614, 1, 1));
				}
				bt->setup(section,object,condition[i],false);
				section->add_condition( bt);
			}
			ConditionListButton_ED* bt = memnew(ConditionListButton_ED);
			bt->setup(object,false);
			section->add_condition( bt);
			p_plugin->add_custom_control( section);
			return true;

		}
		return false;
	}

	static bool _parse_blackboard_property(EditorInspectorPlugin *p_plugin,const Ref<AnimatorBlackboardSet>& object, Variant::Type type, const String& name, PropertyHint hint_type, const String& hint_string, BitField<PropertyUsageFlags> usage_flags, bool wide)
	{
		if(name == "change_list")
		{
			BlackbordSetSection* section = memnew(BlackbordSetSection);
			section->setup(object);
			section->set_category_name("Change List");
			TypedArray<Ref<AnimatorBlackboardSetItemBase>>  condition = object->get_change_list();
			for(int32_t i = 0; i < condition.size(); ++i)
			{
				BlackbordSet_ED* bt = memnew(BlackbordSet_ED);
				if(i & 1)
				{
					bt->set_modulate(Color(0.814023, 0.741614, 1, 1));
				}
				bt->setup(object,condition[i]);
				section->add_condition( bt);
			}
			BlackbordSetButtonList_ED* bt = memnew(BlackbordSetButtonList_ED);
			bt->setup(object);
			section->add_condition( bt);
			p_plugin->add_custom_control( section);
			return true;
		}
		return false;
	}
	static bool _parse_node_root_property(EditorInspectorPlugin *p_plugin,const Ref<CharacterAnimationLogicRoot>& object, Variant::Type type, const String& name, PropertyHint hint_type, const String& hint_string, BitField<PropertyUsageFlags> usage_flags, bool wide)
	{
		if(name == "node_list")
		{
			AnimationLogicRootNodeProperty* section = memnew(AnimationLogicRootNodeProperty);
			section->setup(Variant::OBJECT,hint_string);
			section->node_object = object;
			p_plugin->add_custom_control( section);
			return true;

		}
		return false;

	}
	static bool _parse_beehave_tree_property(EditorInspectorPlugin *p_plugin,const Ref<BeehaveTree>& object, Variant::Type type, const String& name, PropertyHint hint_type, const String& hint_string, BitField<PropertyUsageFlags> usage_flags, bool wide)
	{
		if(name == "root_node")
		{
			VBoxContainer* p_select_node_property_vbox = memnew( VBoxContainer);	
			
			BeehaveGraphProperty* section = memnew(BeehaveGraphProperty);
			// 初始化
			if(object->get_root_node().is_null())
			{
				object->set_root_node(memnew(BeehaveCompositeParallel));
			}
			section->setup(object, p_select_node_property_vbox);
			p_plugin->add_custom_control( section);
			p_plugin->add_custom_control( p_select_node_property_vbox);

			return true;

		}
		return false;

	}
	static bool _parse_charater_ai_property(EditorInspectorPlugin *p_plugin,CharacterBodyMain* object, Variant::Type type, const String& name, PropertyHint hint_type, const String& hint_string, BitField<PropertyUsageFlags> usage_flags, bool wide)
	{
		
		if(name == "blackboard_plan")
		{
			CharacterBodyMainLable* lable = memnew(CharacterBodyMainLable);
			lable->set_body_main(object);
			p_plugin->add_custom_control( lable);
			Skeleton3D* skeleton = object->get_skeleton();
			if(skeleton != nullptr)
			{				
				Skeleton3DEditor* skel_editor = memnew(Skeleton3DEditor(nullptr, skeleton));
				p_plugin->add_custom_control( skel_editor);
			}
			return false;
		}
		else if(name == "character_ai")
		{
			CharacterAISection* ai_setion = memnew( CharacterAISection);
			ai_setion->init();
			ai_setion->setup(object);
			p_plugin->add_custom_control( ai_setion);

			return true;

		}
		return false;

	}

};


// 一些自定义的Inspector插件
class GameHelpInspectorPlugin : public EditorInspectorPlugin
{
	GDCLASS(GameHelpInspectorPlugin, EditorInspectorPlugin);
	static void _bind_methods()
	{

	}
	public:
	virtual bool can_handle(Object* p_object) override
	{
		if( Object::cast_to<CharacterAnimatorCondition>(p_object) != nullptr)
		{
			EditorNode::get_log()->add_message(String("GameHelpInspectorPlugin.can_handle") + " :" + p_object->get_class());
			return true;
		}
		if( Object::cast_to<AnimatorBlackboardSet>(p_object) != nullptr)
		{
			EditorNode::get_log()->add_message(String("GameHelpInspectorPlugin.can_handle") + " :" + p_object->get_class());
			return true;
		}
		if( Object::cast_to<CharacterAnimationLogicRoot>(p_object) != nullptr)
		{
			EditorNode::get_log()->add_message(String("GameHelpInspectorPlugin.can_handle") + " :" + p_object->get_class());
			return true;
		}

		if( Object::cast_to<BeehaveTree>(p_object) != nullptr)
		{
			return true;
		}
		if(Object::cast_to<CharacterBodyMain>(p_object) != nullptr)
		{
			CharacterBodyMain* body_main = Object::cast_to<CharacterBodyMain>(p_object);
			body_main->set_editor_run_ai(false);
			body_main->init();
			return true;
		}

		return false;
	}
	bool parse_property(Object *p_object, const Variant::Type p_type, const String &p_path, const PropertyHint p_hint, const String &p_hint_text, const BitField<PropertyUsageFlags> p_usage, const bool p_wide)override {
	
		Ref<CharacterAnimatorCondition> object = Object::cast_to< CharacterAnimatorCondition> (p_object);
		if(object.is_valid())
		{
			return ConditionList_ED::_parse_condition_property(this,object, p_type, p_path, p_hint, p_hint_text, p_usage, p_wide);
		}
		Ref<AnimatorBlackboardSet> set_object = Object::cast_to<AnimatorBlackboardSet> (p_object);
		if (/* condition */set_object.is_valid())
		{
			return ConditionList_ED::_parse_blackboard_property(this,set_object, p_type, p_path, p_hint, p_hint_text, p_usage, p_wide);
		}
		Ref<CharacterAnimationLogicRoot> root_object = Object::cast_to<CharacterAnimationLogicRoot> (p_object);
		if (/* condition */root_object.is_valid())
		{
			return ConditionList_ED::_parse_node_root_property(this,root_object, p_type, p_path, p_hint, p_hint_text, p_usage, p_wide);
		}
		Ref<BeehaveTree> tree_object = Object::cast_to<BeehaveTree>(p_object);
		if (/* condition */tree_object.is_valid())
		{
			return ConditionList_ED::_parse_beehave_tree_property(this,tree_object, p_type, p_path, p_hint, p_hint_text, p_usage, p_wide);
		}
		CharacterBodyMain* body_main = Object::cast_to<CharacterBodyMain>(p_object);
		if(body_main != nullptr)
		{
			
			return ConditionList_ED::_parse_charater_ai_property(this,body_main, p_type, p_path, p_hint, p_hint_text, p_usage, p_wide);
		}
		return false;
	}

};
namespace AnimationToolConst
{
	
	enum class HumanBodyBones {
		Hips = 0,
		LeftUpperLeg = 1,
		RightUpperLeg = 2,
		LeftLowerLeg = 3,
		RightLowerLeg = 4,
		LeftFoot = 5,
		RightFoot = 6,
		Spine = 7,
		Chest = 8,
		UpperChest = 54,
		Neck = 9,
		Head = 10,
		LeftShoulder = 11,
		RightShoulder = 12,
		LeftUpperArm = 13,
		RightUpperArm = 14,
		LeftLowerArm = 15,
		RightLowerArm = 16,
		LeftHand = 17,
		RightHand = 18,
		LeftToes = 19,
		RightToes = 20,
		LeftEye = 21,
		RightEye = 22,
		Jaw = 23,
		LeftThumbProximal = 24,
		LeftThumbIntermediate = 25,
		LeftThumbDistal = 26,
		LeftIndexProximal = 27,
		LeftIndexIntermediate = 28,
		LeftIndexDistal = 29,
		LeftMiddleProximal = 30,
		LeftMiddleIntermediate = 31,
		LeftMiddleDistal = 32,
		LeftRingProximal = 33,
		LeftRingIntermediate = 34,
		LeftRingDistal = 35,
		LeftLittleProximal = 36,
		LeftLittleIntermediate = 37,
		LeftLittleDistal = 38,
		RightThumbProximal = 39,
		RightThumbIntermediate = 40,
		RightThumbDistal = 41,
		RightIndexProximal = 42,
		RightIndexIntermediate = 43,
		RightIndexDistal = 44,
		RightMiddleProximal = 45,
		RightMiddleIntermediate = 46,
		RightMiddleDistal = 47,
		RightRingProximal = 48,
		RightRingIntermediate = 49,
		RightRingDistal = 50,
		RightLittleProximal = 51,
		RightLittleIntermediate = 52,
		RightLittleDistal = 53,
	};

	static HashMap<int64_t,String> utype_to_classname = {
		{0, "Object"},
		{1, "GameObject"},
		{2, "Component"},
		{3, "LevelGameManager"},
		{4, "Transform"},
		{5, "TimeManager"},
		{6, "GlobalGameManager"},
		{8, "Behaviour"},
		{9, "GameManager"},
		{11, "AudioManager"},
		{13, "InputManager"},
		{18, "EditorExtension"},
		{19, "Physics2DSettings"},
		{20, "Camera"},
		{21, "Material"},
		{23, "MeshRenderer"},
		{25, "Renderer"},
		{27, "Texture"},
		{28, "Texture2D"},
		{29, "OcclusionCullingSettings"},
		{30, "GraphicsSettings"},
		{33, "MeshFilter"},
		{41, "OcclusionPortal"},
		{43, "Mesh"},
		{45, "Skybox"},
		{47, "QualitySettings"},
		{48, "Shader"},
		{49, "TextAsset"},
		{50, "Rigidbody2D"},
		{53, "Collider2D"},
		{54, "Rigidbody"},
		{55, "PhysicsManager"},
		{56, "Collider"},
		{57, "Joint"},
		{58, "CircleCollider2D"},
		{59, "HingeJoint"},
		{60, "PolygonCollider2D"},
		{61, "BoxCollider2D"},
		{62, "PhysicsMaterial2D"},
		{64, "MeshCollider"},
		{65, "BoxCollider"},
		{66, "CompositeCollider2D"},
		{68, "EdgeCollider2D"},
		{70, "CapsuleCollider2D"},
		{72, "ComputeShader"},
		{74, "AnimationClip"},
		{75, "ConstantForce"},
		{78, "TagManager"},
		{81, "AudioListener"},
		{82, "AudioSource"},
		{83, "AudioClip"},
		{84, "RenderTexture"},
		{86, "CustomRenderTexture"},
		{89, "Cubemap"},
		{90, "Avatar"},
		{91, "AnimatorController"},
		{93, "RuntimeAnimatorController"},
		{94, "ScriptMapper"},
		{95, "Animator"},
		{96, "TrailRenderer"},
		{98, "DelayedCallManager"},
		{102, "TextMesh"},
		{104, "RenderSettings"},
		{108, "Light"},
		{109, "CGProgram"},
		{110, "BaseAnimationTrack"},
		{111, "Animation"},
		{114, "MonoBehaviour"},
		{115, "MonoScript"},
		{116, "MonoManager"},
		{117, "Texture3D"},
		{118, "NewAnimationTrack"},
		{119, "Projector"},
		{120, "LineRenderer"},
		{121, "Flare"},
		{122, "Halo"},
		{123, "LensFlare"},
		{124, "FlareLayer"},
		{125, "HaloLayer"},
		{126, "NavMeshProjectSettings"},
		{128, "Font"},
		{129, "PlayerSettings"},
		{130, "NamedObject"},
		{134, "PhysicMaterial"},
		{135, "SphereCollider"},
		{136, "CapsuleCollider"},
		{137, "SkinnedMeshRenderer"},
		{138, "FixedJoint"},
		{141, "BuildSettings"},
		{142, "AssetBundle"},
		{143, "CharacterController"},
		{144, "CharacterJoint"},
		{145, "SpringJoint"},
		{146, "WheelCollider"},
		{147, "ResourceManager"},
		{150, "PreloadData"},
		{153, "ConfigurableJoint"},
		{154, "TerrainCollider"},
		{156, "TerrainData"},
		{157, "LightmapSettings"},
		{158, "WebCamTexture"},
		{159, "EditorSettings"},
		{162, "EditorUserSettings"},
		{164, "AudioReverbFilter"},
		{165, "AudioHighPassFilter"},
		{166, "AudioChorusFilter"},
		{167, "AudioReverbZone"},
		{168, "AudioEchoFilter"},
		{169, "AudioLowPassFilter"},
		{170, "AudioDistortionFilter"},
		{171, "SparseTexture"},
		{180, "AudioBehaviour"},
		{181, "AudioFilter"},
		{182, "WindZone"},
		{183, "Cloth"},
		{184, "SubstanceArchive"},
		{185, "ProceduralMaterial"},
		{186, "ProceduralTexture"},
		{187, "Texture2DArray"},
		{188, "CubemapArray"},
		{191, "OffMeshLink"},
		{192, "OcclusionArea"},
		{193, "Tree"},
		{195, "NavMeshAgent"},
		{196, "NavMeshSettings"},
		{198, "ParticleSystem"},
		{199, "ParticleSystemRenderer"},
		{200, "ShaderVariantCollection"},
		{205, "LODGroup"},
		{206, "BlendTree"},
		{207, "Motion"},
		{208, "NavMeshObstacle"},
		{210, "SortingGroup"},
		{212, "SpriteRenderer"},
		{213, "Sprite"},
		{214, "CachedSpriteAtlas"},
		{215, "ReflectionProbe"},
		{218, "Terrain"},
		{220, "LightProbeGroup"},
		{221, "AnimatorOverrideController"},
		{222, "CanvasRenderer"},
		{223, "Canvas"},
		{224, "RectTransform"},
		{225, "CanvasGroup"},
		{226, "BillboardAsset"},
		{227, "BillboardRenderer"},
		{228, "SpeedTreeWindAsset"},
		{229, "AnchoredJoint2D"},
		{230, "Joint2D"},
		{231, "SpringJoint2D"},
		{232, "DistanceJoint2D"},
		{233, "HingeJoint2D"},
		{234, "SliderJoint2D"},
		{235, "WheelJoint2D"},
		{236, "ClusterInputManager"},
		{237, "BaseVideoTexture"},
		{238, "NavMeshData"},
		{240, "AudioMixer"},
		{241, "AudioMixerController"},
		{243, "AudioMixerGroupController"},
		{244, "AudioMixerEffectController"},
		{245, "AudioMixerSnapshotController"},
		{246, "PhysicsUpdateBehaviour2D"},
		{247, "ConstantForce2D"},
		{248, "Effector2D"},
		{249, "AreaEffector2D"},
		{250, "PointEffector2D"},
		{251, "PlatformEffector2D"},
		{252, "SurfaceEffector2D"},
		{253, "BuoyancyEffector2D"},
		{254, "RelativeJoint2D"},
		{255, "FixedJoint2D"},
		{256, "FrictionJoint2D"},
		{257, "TargetJoint2D"},
		{258, "LightProbes"},
		{259, "LightProbeProxyVolume"},
		{271, "SampleClip"},
		{272, "AudioMixerSnapshot"},
		{273, "AudioMixerGroup"},
		{290, "AssetBundleManifest"},
		{300, "RuntimeInitializeOnLoadManager"},
		{310, "UnidotConnectSettings"},
		{319, "AvatarMask"},
		{320, "PlayableDirector"},
		{328, "VideoPlayer"},
		{329, "VideoClip"},
		{330, "ParticleSystemForceField"},
		{331, "SpriteMask"},
		{362, "WorldAnchor"},
		{363, "OcclusionCullingData"},
		{1001, "PrefabInstance"},
		{1002, "EditorExtensionImpl"},
		{1003, "AssetImporter"},
		{1004, "AssetDatabaseV1"},
		{1005, "Mesh3DSImporter"},
		{1006, "TextureImporter"},
		{1007, "ShaderImporter"},
		{1008, "ComputeShaderImporter"},
		{1020, "AudioImporter"},
		{1026, "HierarchyState"},
		{1028, "AssetMetaData"},
		{1029, "DefaultAsset"},
		{1030, "DefaultImporter"},
		{1031, "TextScriptImporter"},
		{1032, "SceneAsset"},
		{1034, "NativeFormatImporter"},
		{1035, "MonoImporter"},
		{1038, "LibraryAssetImporter"},
		{1040, "ModelImporter"},
		{1041, "FBXImporter"},
		{1042, "TrueTypeFontImporter"},
		{1045, "EditorBuildSettings"},
		{1048, "InspectorExpandedState"},
		{1049, "AnnotationManager"},
		{1050, "PluginImporter"},
		{1051, "EditorUserBuildSettings"},
		{1055, "IHVImageFormatImporter"},
		{1101, "AnimatorStateTransition"},
		{1102, "AnimatorState"},
		{1105, "HumanTemplate"},
		{1107, "AnimatorStateMachine"},
		{1108, "PreviewAnimationClip"},
		{1109, "AnimatorTransition"},
		{1110, "SpeedTreeImporter"},
		{1111, "AnimatorTransitionBase"},
		{1112, "SubstanceImporter"},
		{1113, "LightmapParameters"},
		{1120, "LightingDataAsset"},
		{1124, "SketchUpImporter"},
		{1125, "BuildReport"},
		{1126, "PackedAssets"},
		{1127, "VideoClipImporter"},
		{100000, "int"},
		{100001, "bool"},
		{100002, "float"},
		{100003, "MonoObject"},
		{100004, "Collision"},
		{100005, "Vector3f"},
		{100006, "RootMotionData"},
		{100007, "Collision2D"},
		{100008, "AudioMixerLiveUpdateFloat"},
		{100009, "AudioMixerLiveUpdateBool"},
		{100010, "Polygon2D"},
		{100011, "void"},
		{19719996, "TilemapCollider2D"},
		{41386430, "AssetImporterLog"},
		{73398921, "VFXRenderer"},
		{156049354, "Grid"},
		{181963792, "Preset"},
		{277625683, "EmptyObject"},
		{285090594, "IConstraint"},
		{294290339, "AssemblyDefinitionReferenceImporter"},
		{334799969, "SiblingDerived"},
		{367388927, "SubDerived"},
		{369655926, "AssetImportInProgressProxy"},
		{382020655, "PluginBuildInfo"},
		{426301858, "EditorProjectAccess"},
		{468431735, "PrefabImporter"},
		{483693784, "TilemapRenderer"},
		{638013454, "SpriteAtlasDatabase"},
		{641289076, "AudioBuildInfo"},
		{644342135, "CachedSpriteAtlasRuntimeData"},
		{646504946, "RendererFake"},
		{662584278, "AssemblyDefinitionReferenceAsset"},
		{668709126, "BuiltAssetBundleInfoSet"},
		{687078895, "SpriteAtlas"},
		{747330370, "RayTracingShaderImporter"},
		{825902497, "RayTracingShader"},
		{877146078, "PlatformModuleSetup"},
		{895512359, "AimConstraint"},
		{937362698, "VFXManager"},
		{994735392, "VisualEffectSubgraph"},
		{994735403, "VisualEffectSubgraphOperator"},
		{994735404, "VisualEffectSubgraphBlock"},
		{1001480554, "Prefab"},
		{1027052791, "LocalizationImporter"},
		{1091556383, "Derived"},
		{1114811875, "ReferencesArtifactGenerator"},
		{1152215463, "AssemblyDefinitionAsset"},
		{1154873562, "SceneVisibilityState"},
		{1183024399, "LookAtConstraint"},
		{1268269756, "GameObjectRecorder"},
		{1325145578, "LightingDataAssetParent"},
		{1386491679, "PresetManager"},
		{1403656975, "StreamingManager"},
		{1480428607, "LowerResBlitTexture"},
		{1542919678, "StreamingController"},
		{1742807556, "GridLayout"},
		{1766753193, "AssemblyDefinitionImporter"},
		{1773428102, "ParentConstraint"},
		{1803986026, "FakeComponent"},
		{1818360608, "PositionConstraint"},
		{1818360609, "RotationConstraint"},
		{1818360610, "ScaleConstraint"},
		{1839735485, "Tilemap"},
		{1896753125, "PackageManifest"},
		{1896753126, "PackageManifestImporter"},
		{1953259897, "TerrainLayer"},
		{1971053207, "SpriteShapeRenderer"},
		{1977754360, "NativeObjectType"},
		{1995898324, "SerializableManagedHost"},
		{2058629509, "VisualEffectAsset"},
		{2058629510, "VisualEffectImporter"},
		{2058629511, "VisualEffectResource"},
		{2059678085, "VisualEffectObject"},
		{2083052967, "VisualEffect"},
		{2083778819, "LocalizationAsset"},
		{208985858483, "ScriptedImporter"},
	};

	
	LocalVector<String> GodotHumanNames  = {
		"Hips",
		"LeftUpperLeg", "RightUpperLeg",
		"LeftLowerLeg", "RightLowerLeg",
		"LeftFoot", "RightFoot",
		"Spine",
		"Chest",
		"Neck",
		"Head",

		
		"LeftShoulder", "RightShoulder",
		"LeftUpperArm", "RightUpperArm",
		"LeftLowerArm", "RightLowerArm",
		"LeftHand", "RightHand",
		"LeftToes", "RightToes",
		"LeftEye", "RightEye",
		"Jaw",



		"LeftThumbMetacarpal", "LeftThumbProximal", "LeftThumbDistal",
		"LeftIndexProximal", "LeftIndexIntermediate", "LeftIndexDistal",
		"LeftMiddleProximal", "LeftMiddleIntermediate", "LeftMiddleDistal",
		"LeftRingProximal", "LeftRingIntermediate", "LeftRingDistal",
		"LeftLittleProximal", "LeftLittleIntermediate", "LeftLittleDistal",


		"RightThumbMetacarpal", "RightThumbProximal", "RightThumbDistal",
		"RightIndexProximal", "RightIndexIntermediate", "RightIndexDistal",
		"RightMiddleProximal", "RightMiddleIntermediate", "RightMiddleDistal",
		"RightRingProximal", "RightRingIntermediate", "RightRingDistal",
		"RightLittleProximal", "RightLittleIntermediate", "RightLittleDistal",
		"UpperChest",
	};

	LocalVector<String> BoneName = {
		"Hips",
		"LeftUpperLeg", "RightUpperLeg",
		"LeftLowerLeg", "RightLowerLeg",
		"LeftFoot", "RightFoot",
		"Spine",
		"Chest",
		"Neck",
		"Head",

		"LeftShoulder", "RightShoulder",
		"LeftUpperArm", "RightUpperArm",
		"LeftLowerArm", "RightLowerArm",
		"LeftHand", "RightHand",
		"LeftToes", "RightToes",
		"LeftEye", "RightEye",
		"Jaw",

		"Left Thumb Proximal", "Left Thumb Intermediate", "Left Thumb Distal",
		"Left Index Proximal", "Left Index Intermediate", "Left Index Distal",
		"Left Middle Proximal", "Left Middle Intermediate", "Left Middle Distal",
		"Left Ring Proximal", "Left Ring Intermediate", "Left Ring Distal",
		"Left Little Proximal", "Left Little Intermediate", "Left Little Distal",


		"Right Thumb Proximal", "Right Thumb Intermediate", "Right Thumb Distal",
		"Right Index Proximal", "Right Index Intermediate", "Right Index Distal",
		"Right Middle Proximal", "Right Middle Intermediate", "Right Middle Distal",
		"Right Ring Proximal", "Right Ring Intermediate", "Right Ring Distal",
		"Right Little Proximal", "Right Little Intermediate", "Right Little Distal",
		"UpperChest",
	};

	HashMap<String, String> HandBones = {
		{"Left Thumb Proximal"  , "left"}, {"Left Thumb Intermediate" , "left"}, {"Left Thumb Distal" , "left"},
		{"Left Index Proximal" , "left"}, {"Left Index Intermediate" , "left"}, {"Left Index Distal" , "left"},
		{"Left Middle Proximal" , "left"}, {"Left Middle Intermediate" , "left"}, {"Left Middle Distal" , "left"},
		{"Left Ring Proximal" , "left"}, {"Left Ring Intermediate" , "left"}, {"Left Ring Distal" , "left"},
		{"Left Little Proximal" , "left"}, {"Left Little Intermediate" , "left"} , {"Left Little Distal" , "left"},
		{"Right Thumb Proximal" , "right"}, {"Right Thumb Intermediate" , "right"}, {"Right Thumb Distal" , "right"},
		{"Right Index Proximal" , "right"}, {"Right Index Intermediate" , "right"}, {"Right Index Distal" , "right" },
		{"Right Middle Proximal" , "right"}, {"Right Middle Intermediate" , "right"}, {"Right Middle Distal" , "right"},
		{"Right Ring Proximal" , "right"}, {"Right Ring Intermediate" , "right"}, {"Right Ring Distal" , "right"},
		{"Right Little Proximal" , "right"}, {"Right Little Intermediate" , "right"}, {"Right Little Distal" , "right"},
	};

	LocalVector<String> MuscleName = {
		"Spine Front-Back", "Spine Left-Right", "Spine Twist Left-Right",
		"Chest Front-Back", "Chest Left-Right", "Chest Twist Left-Right",
		"UpperChest Front-Back", "UpperChest Left-Right", "UpperChest Twist Left-Right",
		"Neck Nod Down-Up", "Neck Tilt Left-Right", "Neck Turn Left-Right",
		"Head Nod Down-Up", "Head Tilt Left-Right", "Head Turn Left-Right",
		"Left Eye Down-Up", "Left Eye In-Out",
		"Right Eye Down-Up", "Right Eye In-Out",
		"Jaw Close", "Jaw Left-Right",


		"Left Upper Leg Front-Back", "Left Upper Leg In-Out", "Left Upper Leg Twist In-Out",
		"Left Lower Leg Stretch", "Left Lower Leg Twist In-Out",
		"Left Foot Up-Down", "Left Foot Twist In-Out", "Left Toes Up-Down",
		"Right Upper Leg Front-Back", "Right Upper Leg In-Out", "Right Upper Leg Twist In-Out",
		"Right Lower Leg Stretch", "Right Lower Leg Twist In-Out",
		"Right Foot Up-Down", "Right Foot Twist In-Out", "Right Toes Up-Down",


		"Left Shoulder Down-Up", "Left Shoulder Front-Back", "Left Arm Down-Up",
		"Left Arm Front-Back", "Left Arm Twist In-Out",
		"Left Forearm Stretch", "Left Forearm Twist In-Out",
		"Left Hand Down-Up", "Left Hand In-Out",
		"Right Shoulder Down-Up", "Right Shoulder Front-Back", "Right Arm Down-Up",
		"Right Arm Front-Back", "Right Arm Twist In-Out",
		"Right Forearm Stretch", "Right Forearm Twist In-Out",
		"Right Hand Down-Up", "Right Hand In-Out",



		"LeftHand.Thumb.1 Stretched", "LeftHand.Thumb.Spread", "LeftHand.Thumb.2 Stretched", "LeftHand.Thumb.3 Stretched",
		"LeftHand.Index.1 Stretched", "LeftHand.Index.Spread", "LeftHand.Index.2 Stretched", "LeftHand.Index.3 Stretched",
		"LeftHand.Middle.1 Stretched", "LeftHand.Middle.Spread", "LeftHand.Middle.2 Stretched", "LeftHand.Middle.3 Stretched",
		"LeftHand.Ring.1 Stretched", "LeftHand.Ring.Spread", "LeftHand.Ring.2 Stretched", "LeftHand.Ring.3 Stretched",
		"LeftHand.Little.1 Stretched", "LeftHand.Little.Spread", "LeftHand.Little.2 Stretched", "LeftHand.Little.3 Stretched",


		"RightHand.Thumb.1 Stretched", "RightHand.Thumb.Spread", "RightHand.Thumb.2 Stretched", "RightHand.Thumb.3 Stretched",
		"RightHand.Index.1 Stretched", "RightHand.Index.Spread", "RightHand.Index.2 Stretched", "RightHand.Index.3 Stretched",
		"RightHand.Middle.1 Stretched", "RightHand.Middle.Spread", "RightHand.Middle.2 Stretched", "RightHand.Middle.3 Stretched",
		"RightHand.Ring.1 Stretched", "RightHand.Ring.Spread", "RightHand.Ring.2 Stretched", "RightHand.Ring.3 Stretched",
		"RightHand.Little.1 Stretched", "RightHand.Little.Spread", "RightHand.Little.2 Stretched", "RightHand.Little.3 Stretched"
	};


	HashMap<String, String> TraitMapping = {
		{"Left Thumb 1 Stretched"  ,  "LeftHand.Thumb.1 Stretched"},
		{"Left Thumb Spread" ,  "LeftHand.Thumb.Spread"},
		{"Left Thumb 2 Stretched" ,  "LeftHand.Thumb.2 Stretched"},
		{"Left Thumb 3 Stretched" ,  "LeftHand.Thumb.3 Stretched"},
		{"Left Index 1 Stretched" ,  "LeftHand.Index.1 Stretched"},
		{"Left Index Spread" ,  "LeftHand.Index.Spread"},
		{"Left Index 2 Stretched" ,  "LeftHand.Index.2 Stretched"},
		{"Left Index 3 Stretched" ,  "LeftHand.Index.3 Stretched"},
		{"Left Middle 1 Stretched" ,  "LeftHand.Middle.1 Stretched"},
		{"Left Middle Spread" ,  "LeftHand.Middle.Spread"},
		{"Left Middle 2 Stretched" ,  "LeftHand.Middle.2 Stretched"},
		{"Left Middle 3 Stretched" ,  "LeftHand.Middle.3 Stretched"},
		{"Left Ring 1 Stretched" ,  "LeftHand.Ring.1 Stretched"},
		{"Left Ring Spread" ,  "LeftHand.Ring.Spread"},
		{"Left Ring 2 Stretched" ,  "LeftHand.Ring.2 Stretched"},
		{"Left Ring 3 Stretched" ,  "LeftHand.Ring.3 Stretched"},
		{"Left Little 1 Stretched" ,  "LeftHand.Little.1 Stretched"},
		{"Left Little Spread" ,  "LeftHand.Little.Spread"},
		{"Left Little 2 Stretched" ,  "LeftHand.Little.2 Stretched"},
		{"Left Little 3 Stretched" ,  "LeftHand.Little.3 Stretched"},
		{"Right Thumb 1 Stretched" ,  "RightHand.Thumb.1 Stretched"},
		{"Right Thumb Spread" ,  "RightHand.Thumb.Spread"},
		{"Right Thumb 2 Stretched" ,  "RightHand.Thumb.2 Stretched"},
		{"Right Thumb 3 Stretched" ,  "RightHand.Thumb.3 Stretched"},
		{"Right Index 1 Stretched" ,  "RightHand.Index.1 Stretched"},
		{"Right Index Spread" ,  "RightHand.Index.Spread"},
		{"Right Index 2 Stretched" ,  "RightHand.Index.2 Stretched"},
		{"Right Index 3 Stretched" ,  "RightHand.Index.3 Stretched"},
		{"Right Middle 1 Stretched" ,  "RightHand.Middle.1 Stretched"},
		{"Right Middle Spread" ,  "RightHand.Middle.Spread"},
		{"Right Middle 2 Stretched" ,  "RightHand.Middle.2 Stretched"},
		{"Right Middle 3 Stretched" ,  "RightHand.Middle.3 Stretched"},
		{"Right Ring 1 Stretched" ,  "RightHand.Ring.1 Stretched"},
		{"Right Ring Spread" ,  "RightHand.Ring.Spread"},
		{"Right Ring 2 Stretched" ,  "RightHand.Ring.2 Stretched"},
		{"Right Ring 3 Stretched" ,  "RightHand.Ring.3 Stretched"},
		{"Right Little 1 Stretched" ,  "RightHand.Little.1 Stretched"},
		{"Right Little Spread" ,  "RightHand.Little.Spread"},
		{"Right Little 2 Stretched" ,  "RightHand.Little.2 Stretched"},
		{"Right Little 3 Stretched" ,  "RightHand.Little.3 Stretched"},
	};
	LocalVector<String> IKPrefixNames = { "Root", "Motion", "LeftFoot", "RightFoot", "LeftHand", "RightHand" };
	HashMap<String, uint8_t> IKSuffixNames = { {"T.x", 0}, {"T.y" , 1}, {"T.z" , 2}, {"Q.x" , 0}, {"Q.y" , 1}, {"Q.z" , 2}, {"Q.w" , 3} };

	static int32_t BoneCount()
	{
		return BoneName.size();
	}
	static int32_t MuscleCount()
	{
		return MuscleName.size();
	}
	LocalVector<LocalVector<int8_t>> MuscleFromBone = {
		{-1, -1, -1} ,
		{23, 22, 21},{31, 30, 29},
		{25, -1, 24},{33, -1, 32},
		{-1, 27, 26},{-1, 35, 34},
		{2, 1, 0},
		{5, 4, 3},
		{11, 10, 9},
		{14, 13, 12},
		{-1, 38, 37},{-1, 47, 46},
		{41, 40, 39},{50, 49, 48},
		{43, -1, 42},{52, -1, 51},
		{-1, 45, 44},{-1, 54, 53},
		{-1, -1, 28},{-1, -1, 36},
		{-1, 16, 15},{-1, 18, 17},
		{-1, 20, 19},
		{-1, 56, 55},{-1, -1, 57},{-1, -1, 58},
		{-1, 60, 59},{-1, -1, 61},{-1, -1, 62},
		{-1, 64, 63},{-1, -1, 65},{-1, -1, 66},
		{-1, 68, 67},{-1, -1, 69},{-1, -1, 70},
		{-1, 72, 71},{-1, -1, 73},{-1, -1, 74},
		{-1, 76, 75},{-1, -1, 77},{-1, -1, 78},
		{-1, 80, 79},{-1, -1, 81},{-1, -1, 82},
		{-1, 84, 83},{-1, -1, 85},{-1, -1, 86},
		{-1, 88, 87},{-1, -1, 89},{-1, -1, 90},
		{-1, 92, 91},{-1, -1, 93},{-1, -1, 94},
		{8, 7, 6},
	};

	LocalVector<float> MuscleDefaultMax = {
		40, 40, 40, 40, 40, 40, 20, 20, 20, 40, 40, 40, 40, 40, 40,
		15, 20, 15, 20, 10, 10,
		50, 60, 60, 80, 90, 50, 30, 50,
		50, 60, 60, 80, 90, 50, 30, 50,
		30, 15, 100, 100, 90, 80, 90, 80, 40,
		30, 15, 100, 100, 90, 80, 90, 80, 40,
		20, 25, 35, 35, 50, 20, 45, 45, 50, 7.5, 45, 45, 50, 7.5, 45, 45, 50, 20, 45, 45,
		20, 25, 35, 35, 50, 20, 45, 45, 50, 7.5, 45, 45, 50, 7.5, 45, 45, 50, 20, 45, 45,
	};

	LocalVector<float> MuscleDefaultMin = {
		-40, -40, -40, -40, -40, -40, -20, -20, -20, -40, -40, -40, -40, -40, -40,
		-10, -20, -10, -20, -10, -10,
		-90, -60, -60, -80, -90, -50, -30, -50,
		-90, -60, -60, -80, -90, -50, -30, -50,
		-15, -15, -60, -100, -90, -80, -90, -80, -40,
		-15, -15, -60, -100, -90, -80, -90, -80, -40,
		-20, -25, -40, -40, -50, -20, -45, -45, -50, -7.5, -45, -45, -50, -7.5, -45, -45, -50, -20, -45, -45,
		-20, -25, -40, -40, -50, -20, -45, -45, -50, -7.5, -45, -45, -50, -7.5, -45, -45, -50, -20, -45, -45,
	};
	//Right - handed PreQ values for Godot's standard humanoid rig.
	LocalVector<Quaternion> preQ_exported = {
		Quaternion(0, 0, 0, 1), //Hips
		Quaternion(-0.62644, -0.34855, 0.59144, 0.36918), //LeftUpperLeg
		Quaternion(-0.59144, -0.36918, 0.62644, 0.34855), //RightUpperLeg
		Quaternion(-0.69691, 0.04422, -0.7145, 0.04313), //LeftLowerLeg
		Quaternion(-0.7145, 0.04313, -0.69691, 0.04422), //RightLowerLeg
		Quaternion(0.5, -0.5, -0.5, 0.5), //LeftFoot
		Quaternion(0.5, -0.5, -0.5, 0.5), //RightFoot
		Quaternion(0.46815, -0.52994, 0.46815, -0.52994), //Spine
		Quaternion(0.52661, -0.47189, 0.52661, -0.47189), //Chest
		Quaternion(0.46642, -0.5316, 0.46748, -0.5304), //Neck
		Quaternion(-0.5, 0.5, -0.5, 0.5), //Head
		Quaternion(0.03047, -0.00261, -0.9959, -0.08517), //LeftShoulder
		Quaternion(-0.00261, 0.03047, 0.08518, 0.9959), //RightShoulder
		Quaternion(0.505665, -0.400775, 0.749395, -0.148645), //LeftUpperArm
		Quaternion(-0.400775, 0.505665, 0.148645, -0.749395), //RightUpperArm
		Quaternion(-0.998935, 0.046125, 0.001085, 0.000045), //LeftLowerArm
		Quaternion(-0.04613, 0.99894, 0.00005, 0.00108), //RightLowerArm
		Quaternion(-0.02914, -0.029083, 0.707276, -0.705735), //LeftHand
		Quaternion(0.029083, 0.02914, -0.705735, 0.707276), //RightHand
		Quaternion(-0.500002, 0.500002, -0.500002, 0.500002), //LeftToes
		Quaternion(-0.500002, 0.500002, -0.500002, 0.500002), //RightToes
		Quaternion(0.70711, 0, -0.70711, 0), //LeftEye
		Quaternion(0.70711, 0, -0.70711, 0), //RightEye
		Quaternion(0, 0, 0, 1), //Jaw
		Quaternion(-0.957335, 0.251575, 0.073965, 0.121475), //LeftThumbMetacarpal
		Quaternion(-0.435979, 0.550073, -0.413528, 0.579935), //LeftThumbProximal
		Quaternion(-0.435979, 0.550073, -0.413528, 0.579935), //LeftThumbDistal
		Quaternion(-0.70292, 0.26496, 0.53029, -0.39305), //LeftIndexProximal
		Quaternion(-0.67155, 0.2898, 0.5998, -0.32447), //LeftIndexIntermediate
		Quaternion(-0.67155, 0.2898, 0.5998, -0.32447), //LeftIndexDistal
		Quaternion(-0.66667, 0.29261, 0.5877, -0.3529), //LeftMiddleProximal
		Quaternion(-0.660575, 0.278445, 0.633985, -0.290115), //LeftMiddleIntermediate
		Quaternion(-0.660575, 0.278445, 0.633985, -0.290115), //LeftMiddleDistal
		Quaternion(-0.60736, 0.34862, 0.64221, -0.31169), //LeftRingProximal
		Quaternion(-0.629695, 0.327165, 0.621835, -0.331305), //LeftRingIntermediate
		Quaternion(-0.629695, 0.327165, 0.621835, -0.331305), //LeftRingDistal
		Quaternion(-0.584415, 0.369825, 0.660045, -0.293315), //LeftLittleProximal
		Quaternion(-0.608895, 0.338425, 0.641505, -0.321215), //LeftLittleIntermediate
		Quaternion(-0.608895, 0.338425, 0.641505, -0.321215), //LeftLittleDistal
		Quaternion(0.251455, -0.957385, -0.121415, -0.073715), //RightThumbMetacarpal
		Quaternion(0.550156, -0.435959, -0.579765, 0.413695), //RightThumbProximal
		Quaternion(0.550156, -0.435959, -0.579765, 0.413695), //RightThumbDistal
		Quaternion(0.264955, -0.702925, 0.393045, -0.530295), //RightIndexProximal
		Quaternion(0.289805, -0.671545, 0.324465, -0.599805), //RightIndexIntermediate
		Quaternion(0.289805, -0.671545, 0.324465, -0.599805), //RightIndexDistal
		Quaternion(0.292615, -0.666665, 0.352895, -0.587705), //RightMiddleProximal
		Quaternion(0.278455, -0.660575, 0.290125, -0.633985), //RightMiddleIntermediate
		Quaternion(0.278455, -0.660575, 0.290125, -0.633985), //RightMiddleDistal
		Quaternion(0.34862, -0.60736, 0.31169, -0.64221), //RightRingProximal
		Quaternion(0.327165, -0.629695, 0.331295, -0.621845), //RightRingIntermediate
		Quaternion(0.327165, -0.629695, 0.331295, -0.621845), //RightRingDistal
		Quaternion(0.36982, -0.58442, 0.29332, -0.66004), //RightLittleProximal
		Quaternion(0.33843, -0.6089, 0.32123, -0.6415), //RightLittleIntermediate
		Quaternion(0.33843, -0.6089, 0.32123, -0.6415), //RightLittleDistal
		Quaternion(0.56563, -0.42434, 0.56563, -0.42434), //UpperChest
	};

	//Right - handed PostQ values for Godot's standard humanoid rig.
	LocalVector<Quaternion> postQ_inverse_exported = {
		Quaternion(0, 0, 0, 1), //Hips
		Quaternion(0.48977, -0.50952, 0.51876, 0.48105), //LeftUpperLeg
		Quaternion(0.51876, -0.48105, 0.48977, 0.50952), //RightUpperLeg
		Quaternion(-0.51894, 0.48097, 0.50616, 0.49312), //LeftLowerLeg
		Quaternion(-0.50616, 0.49312, 0.51894, 0.48097), //RightLowerLeg
		Quaternion(-0.707107, 0, -0.707107, 0), //LeftFoot
		Quaternion(-0.707107, 0, -0.707107, 0), //RightFoot
		Quaternion(-0.46815, 0.52994, -0.46815, -0.52994), //Spine
		Quaternion(-0.52661, 0.47189, -0.52661, -0.47189), //Chest
		Quaternion(-0.46642, 0.5316, -0.46748, -0.5304), //Neck
		Quaternion(0.5, -0.5, 0.5, 0.5), //Head
		Quaternion(-0.523995, 0.469295, -0.557075, -0.441435), //LeftShoulder
		Quaternion(0.46929, -0.524, -0.44143, -0.55708), //RightShoulder
		Quaternion(0.513635, -0.486185, -0.509345, -0.490275), //LeftUpperArm
		Quaternion(0.486185, -0.513635, 0.490275, 0.509345), //RightUpperArm
		Quaternion(0.519596, -0.479517, -0.520728, -0.478471), //LeftLowerArm
		Quaternion(0.479517, -0.519596, 0.478471, 0.520728), //RightLowerArm
		Quaternion(0.520725, -0.478465, -0.479515, -0.519595), //LeftHand
		Quaternion(0.478465, -0.520725, 0.519595, 0.479515), //RightHand
		Quaternion(-0.500002, 0.500002, 0.500002, 0.500002), //LeftToes
		Quaternion(-0.500002, 0.500002, 0.500002, 0.500002), //RightToes
		Quaternion(-0.500002, 0.500002, 0.500002, 0.500002), //LeftEye
		Quaternion(-0.500002, 0.500002, 0.500002, 0.500002), //RightEye
		Quaternion(0, 0.707107, 0.707107, 0), //Jaw
		Quaternion(0.56005, -0.437881, 0.528429, 0.464077), //LeftThumbMetacarpal
		Quaternion(0.541247, -0.458295, 0.513379, 0.483179), //LeftThumbProximal
		Quaternion(0.541247, -0.458295, 0.513379, 0.483179), //LeftThumbDistal
		Quaternion(0.53845, -0.45868, -0.46056, -0.53625), //LeftIndexProximal
		Quaternion(0.53604, -0.46316, -0.47877, -0.51857), //LeftIndexIntermediate
		Quaternion(0.53604, -0.46316, -0.47877, -0.51857), //LeftIndexDistal
		Quaternion(0.52555, -0.47434, -0.492, -0.50669), //LeftMiddleProximal
		Quaternion(0.536385, -0.463085, -0.514795, -0.482515), //LeftMiddleIntermediate
		Quaternion(0.536385, -0.463085, -0.514795, -0.482515), //LeftMiddleDistal
		Quaternion(0.50517, -0.49482, -0.50264, -0.49731), //LeftRingProximal
		Quaternion(0.494155, -0.505555, -0.487985, -0.511945), //LeftRingIntermediate
		Quaternion(0.494155, -0.505555, -0.487985, -0.511945), //LeftRingDistal
		Quaternion(0.502345, -0.497645, -0.501995, -0.497995), //LeftLittleProximal
		Quaternion(0.47756, -0.52241, -0.50314, -0.49585), //LeftLittleIntermediate
		Quaternion(0.47756, -0.52241, -0.50314, -0.49585), //LeftLittleDistal
		Quaternion(0.437905, -0.559994, -0.463881, -0.528644), //RightThumbMetacarpal
		Quaternion(0.458337, -0.5412, -0.483, -0.513558), //RightThumbProximal
		Quaternion(0.458337, -0.5412, -0.483, -0.513558), //RightThumbDistal
		Quaternion(0.45868, -0.53845, 0.53625, 0.46056), //RightIndexProximal
		Quaternion(0.463165, -0.536035, 0.518565, 0.478775), //RightIndexIntermediate
		Quaternion(0.463165, -0.536035, 0.518565, 0.478775), //RightIndexDistal
		Quaternion(0.47434, -0.52555, 0.50669, 0.492), //RightMiddleProximal
		Quaternion(0.4631, -0.53638, 0.48252, 0.5148), //RightMiddleIntermediate
		Quaternion(0.4631, -0.53638, 0.48252, 0.5148), //RightMiddleDistal
		Quaternion(0.49482, -0.50517, 0.49731, 0.50264), //RightRingProximal
		Quaternion(0.505555, -0.494155, 0.511935, 0.487995), //RightRingIntermediate
		Quaternion(0.505555, -0.494155, 0.511935, 0.487995), //RightRingDistal
		Quaternion(0.49764, -0.50235, 0.498, 0.50199), //RightLittleProximal
		Quaternion(0.52241, -0.47756, 0.49586, 0.50313), //RightLittleIntermediate
		Quaternion(0.52241, -0.47756, 0.49586, 0.50313), //RightLittleDistal
		Quaternion(-0.56563, 0.42434, -0.56563, -0.42434), //UpperChest
	};

	LocalVector<Vector3> Signs = {
		Vector3(+1, +1, +1), //Hips
		Vector3(+1, +1, +1), //LeftUpperLeg
		Vector3(-1, -1, +1), //RightUpperLeg
		Vector3(+1, -1, -1), //LeftLowerLeg
		Vector3(-1, +1, -1), //RightLowerLeg
		Vector3(+1, +1, +1), //LeftFoot
		Vector3(-1, -1, +1), //RightFoot
		Vector3(+1, +1, +1), //Spine
		Vector3(+1, +1, +1), //Chest
		Vector3(+1, +1, +1), //Neck
		Vector3(+1, +1, +1), //Head
		Vector3(+1, +1, -1), //LeftShoulder
		Vector3(-1, +1, +1), //RightShoulder
		Vector3(+1, +1, -1), //LeftUpperArm
		Vector3(-1, +1, +1), //RightUpperArm
		Vector3(+1, +1, -1), //LeftLowerArm
		Vector3(-1, +1, +1), //RightLowerArm
		Vector3(+1, +1, -1), //LeftHand
		Vector3(-1, +1, +1), //RightHand
		Vector3(+1, +1, +1), //LeftToes
		Vector3(-1, -1, +1), //RightToes
		Vector3(-1, +1, -1), //LeftEye
		Vector3(+1, -1, -1), //RightEye
		Vector3(1, 1, 1), //Jaw
		Vector3(+1, -1, +1), //LeftThumbProximal
		Vector3(+1, -1, +1), //LeftThumbIntermediate
		Vector3(+1, -1, +1), //LeftThumbDistal
		Vector3(-1, -1, -1), //LeftIndexProximal
		Vector3(-1, -1, -1), //LeftIndexIntermediate
		Vector3(-1, -1, -1), //LeftIndexDistal
		Vector3(-1, -1, -1), //LeftMiddleProximal
		Vector3(-1, -1, -1), //LeftMiddleIntermediate
		Vector3(-1, -1, -1), //LeftMiddleDistal
		Vector3(+1, +1, -1), //LeftRingProximal
		Vector3(+1, +1, -1), //LeftRingIntermediate
		Vector3(+1, +1, -1), //LeftRingDistal
		Vector3(+1, +1, -1), //LeftLittleProximal
		Vector3(+1, +1, -1), //LeftLittleIntermediate
		Vector3(+1, +1, -1), //LeftLittleDistal
		Vector3(-1, -1, -1), //RightThumbProximal
		Vector3(-1, -1, -1), //RightThumbIntermediate
		Vector3(-1, -1, -1), //RightThumbDistal
		Vector3(+1, -1, +1), //RightIndexProximal
		Vector3(+1, -1, +1), //RightIndexIntermediate
		Vector3(+1, -1, +1), //RightIndexDistal
		Vector3(+1, -1, +1), //RightMiddleProximal
		Vector3(+1, -1, +1), //RightMiddleIntermediate
		Vector3(+1, -1, +1), //RightMiddleDistal
		Vector3(-1, +1, +1), //RightRingProximal
		Vector3(-1, +1, +1), //RightRingIntermediate
		Vector3(-1, +1, +1), //RightRingDistal
		Vector3(-1, +1, +1), //RightLittleProximal
		Vector3(-1, +1, +1), //RightLittleIntermediate
		Vector3(-1, +1, +1), //RightLittleDistal
		Vector3(+1, +1, +1), //UpperChest
	};

	HashMap<int32_t, bool> rootQAffectingBones = {
		{0, true}, //Hips	
		{ 1, true }, //LeftUpperLeg
		//{2, true}, //RightUpperLeg
		{ 7, true }, //Spine
		{ 8, true }, //Chest
		{ 54, true }, //UpperChest
		{ 11, true }, //LeftShoulder
		{ 12, true }, //RightShoulder
		//{13, true}, //LeftUpperArm
		//{14, true}, //RightUpperArm
	};

	HashMap<int32_t, int32_t> extraAffectingBones = {
		{1, 3}, //LeftUpperLeg->LeftLowerLeg
		{3, 5}, //LeftLowerLeg->LeftFoot
		{2, 4}, //RightUpperLeg->RightLowerLeg
		{4, 6}, //RightLowerLeg->RightFoot
		{13, 15}, //LeftUpperArm->LeftLowerArm
		{15, 17}, //LeftLowerArm->LeftHand
		{14, 16}, //RightUpperArm->RightLowerArm
		{16, 18}, //RightLowerArm->RightHand
	};

	HashMap<int32_t, int32_t> extraAffectedByBones  = {
		{3, 1}, //LeftUpperLeg->LeftLowerLeg
		{5, 3}, //LeftLowerLeg->LeftFoot
		{4, 2}, //RightUpperLeg->RightLowerLeg
		{6, 4}, //RightLowerLeg->RightFoot
		{15,13 }, //LeftUpperArm->LeftLowerArm
		{17,15 }, //LeftLowerArm->LeftHand
		{16,14 }, //RightUpperArm->RightLowerArm
		{18,16 }, //RightLowerArm->RightHand
	};

	LocalVector<HumanBodyBones> boneIndexToMono = {// HumanTrait.GetBoneIndexToMono(internal)
		HumanBodyBones::Hips,
		HumanBodyBones::LeftUpperLeg,
		HumanBodyBones::RightUpperLeg,
		HumanBodyBones::LeftLowerLeg,
		HumanBodyBones::RightLowerLeg,
		HumanBodyBones::LeftFoot,
		HumanBodyBones::RightFoot,
		HumanBodyBones::Spine,
		HumanBodyBones::Chest,
		HumanBodyBones::UpperChest,
		HumanBodyBones::Neck,
		HumanBodyBones::Head,
		HumanBodyBones::LeftShoulder,
		HumanBodyBones::RightShoulder,
		HumanBodyBones::LeftUpperArm,
		HumanBodyBones::RightUpperArm,
		HumanBodyBones::LeftLowerArm,
		HumanBodyBones::RightLowerArm,
		HumanBodyBones::LeftHand,
		HumanBodyBones::RightHand,
		HumanBodyBones::LeftToes,
		HumanBodyBones::RightToes,
		HumanBodyBones::LeftEye,
		HumanBodyBones::RightEye,
		HumanBodyBones::Jaw,
		HumanBodyBones::LeftThumbProximal,
		HumanBodyBones::LeftThumbIntermediate,
		HumanBodyBones::LeftThumbDistal,
		HumanBodyBones::LeftIndexProximal,
		HumanBodyBones::LeftIndexIntermediate,
		HumanBodyBones::LeftIndexDistal,
		HumanBodyBones::LeftMiddleProximal,
		HumanBodyBones::LeftMiddleIntermediate,
		HumanBodyBones::LeftMiddleDistal,
		HumanBodyBones::LeftRingProximal,
		HumanBodyBones::LeftRingIntermediate,
		HumanBodyBones::LeftRingDistal,
		HumanBodyBones::LeftLittleProximal,
		HumanBodyBones::LeftLittleIntermediate,
		HumanBodyBones::LeftLittleDistal,
		HumanBodyBones::RightThumbProximal,
		HumanBodyBones::RightThumbIntermediate,
		HumanBodyBones::RightThumbDistal,
		HumanBodyBones::RightIndexProximal,
		HumanBodyBones::RightIndexIntermediate,
		HumanBodyBones::RightIndexDistal,
		HumanBodyBones::RightMiddleProximal,
		HumanBodyBones::RightMiddleIntermediate,
		HumanBodyBones::RightMiddleDistal,
		HumanBodyBones::RightRingProximal,
		HumanBodyBones::RightRingIntermediate,
		HumanBodyBones::RightRingDistal,
		HumanBodyBones::RightLittleProximal,
		HumanBodyBones::RightLittleIntermediate,
		HumanBodyBones::RightLittleDistal,
	};

	LocalVector<int8_t> boneIndexToParent = {// HumanTrait.GetBoneIndexToMono(internal)
		0, //HumanBodyBones.Hips,
		0, //HumanBodyBones.Hips,
		0, //HumanBodyBones.Hips,
		1, //HumanBodyBones.LeftUpperLeg,
		2, //HumanBodyBones.RightUpperLeg,
		3, //HumanBodyBones.LeftLowerLeg,
		4, //HumanBodyBones.RightLowerLeg,
		0, //HumanBodyBones.Hips,
		7, //HumanBodyBones.Spine,
		8, //HumanBodyBones.Chest,
		9, //HumanBodyBones.UpperChest,
		10, //HumanBodyBones.Neck,
		9, //HumanBodyBones.UpperChest,
		9, //HumanBodyBones.UpperChest,
		12, //HumanBodyBones.LeftShoulder,
		13, //HumanBodyBones.RightShoulder,
		14, //HumanBodyBones.LeftUpperArm,
		15, //HumanBodyBones.RightUpperArm,
		16, //HumanBodyBones.LeftLowerArm,
		17, //HumanBodyBones.RightLowerArm,
		5, //HumanBodyBones.LeftFoot,
		6, //HumanBodyBones.RightFoot,
		11, //HumanBodyBones.Head,
		11, //HumanBodyBones.Head,
		11, //HumanBodyBones.Head,
	};

	LocalVector<float> bone_lengths = {
		0.10307032899633, //Hips
		0.42552330971818, //LeftUpperLeg
		0.4255239384901, //RightUpperLeg
		0.42702378815645, //LeftLowerLeg
		0.4270232737067, //RightLowerLeg
		0.13250420705574, //LeftFoot
		0.13250419276546, //RightFoot
		0.09717579233836, //Spine
		0.0882577559016, //Chest
		0.1665140084667, //UpperChest
		0.09364062941212, //Neck
		0.1726370036846, //Head
		0.1039340043854, //LeftShoulder
		0.10393849867558, //RightShoulder
		0.26700124954128, //LeftUpperArm
		0.26700124954128, //RightUpperArm
		0.27167462539484, //LeftLowerArm
		0.27167462539484, //RightLowerArm
		0.0901436357993, //LeftHand
		0.09013411133379, //RightHand
		0.0889776497306, //LeftToes
		0.0889776497306, //RightToes
		0.02, //LeftEye
		0.02, //RightEye
		0.01, //Jaw
	};


	LocalVector<float> human_bone_mass = {
		0.145455, //Hips
		0.121212, //LeftUpperLeg
		0.121212,
		0.0484849, //LeftLowerLeg
		0.0484849,
		0.00969697, //LeftFoot
		0.00969697,
		0.030303, //Spine
		0.145455, //Chest
		0.145455, //UpperChest
		0.0121212, //Neck
		0.0484849, //Head
		0.00606061, //LeftShoulder
		0.00606061,
		0.0242424, //LeftUpperArm
		0.0242424,
		0.0181818, //LeftLowerArm
		0.0181818,
		0.00606061, //LeftHand
		0.00606061,
		0.00242424, //LeftToes
		0.00242424,
		0, //LeftEye
		0,
		0 //Jaw
	};

	LocalVector<Vector3> xbot_positions = {
		Vector3(0, 1, 0), //Vector3(-0, 1, 0.014906),
		Vector3(0.078713, -0.064749, -0.01534),
		Vector3(-0.078713, -0.064749, -0.01534),
		Vector3(0, 0.42551, 0.003327),
		Vector3(0, 0.425511, 0.003317),
		Vector3(0, 0.426025, 0.029196),
		Vector3(0, 0.426025, 0.029188),
		Vector3(-0, 0.097642, 0.001261),
		Vector3(-0, 0.096701, -0.009598),
		Vector3(-0, 0.087269, -0.013171),
		Vector3(0, 0.159882, -0.02413),
		Vector3(0, 0.092236, 0.016159),
		Vector3(0.043831, 0.104972, -0.025203),
		Vector3(-0.043826, 0.104974, -0.025203),
		Vector3(-0.021406, 0.101581, -0.005031),
		Vector3(0.021406, 0.101586, -0.005033),
		Vector3(-0, 0.267001, 0),
		Vector3(0, 0.267001, 0),
		Vector3(0, 0.271675, 0),
		Vector3(0, 0.271675, -0.000001),
		Vector3(0, 0.102715, -0.083708),
		Vector3(0, 0.102715, -0.083708),
		Vector3(0.02, 0.04, 0),
		Vector3(-0.02, 0.04, 0),
		Vector3(0, 0.02, 0),
	};
	static HashMap<String, int32_t> _bone_name_to_index()
	{
		HashMap<String, int32_t> ret;
		for(int i=0;i< BoneName.size();i++)
		{
			ret[BoneName[i]]=i;
		}
		return ret;
	}
	static HashMap<String, int32_t> _muscle_name_to_index()
	{
		HashMap<String, int32_t> ret;
		for(int i=0;i< MuscleName.size();i++)
		{
			ret[MuscleName[i]]=i;
		}
		return ret;
	}
	LocalVector<Vector2i> _muscle_index_to_bone_and_axis()
	{
		LocalVector<Vector2i> ret;
		ret.resize(MuscleFromBone.size());
		for(int i=0;i< MuscleFromBone.size();i++)
		{
			LocalVector<int8_t> bone = MuscleFromBone[i];

			for(int j=0;j<3;j++)
			{
				int32_t muscle_i = bone[j];
				if(muscle_i<0)
				{
					continue;
				}
				ret[muscle_i] = Vector2i(i, j);
			}

		}
		return ret;
	}
	static Quaternion swing_twist(float x, float y, float z )
	{
		float yz = Math::sqrt(y * y + z * z);
		float sinc = Math::abs(yz) < 1e-8 ? sin(yz / 2) / yz : 0.5;
		float swingW = Math::cos(yz / 2);
		float twistW = Math::cos(x / 2);
		float twistX = Math::sin(x / 2);
		return Quaternion(
			swingW * twistX,
			(z * twistX + y * twistW) * sinc,
			(z * twistW - y * twistX) * sinc,
			swingW * twistW);
	}
	static Quaternion getMassQ(LocalVector<Vector3>& humanPositions)
	{
		boneIndexToMono.find(HumanBodyBones::LeftUpperArm);
		Vector3 leftUpperArmT = humanPositions[14];// boneIndexToMono.find(LeftUpperArm)
		Vector3 rightUpperArmT = humanPositions[15]; // boneIndexToMono.find(RightUpperArm)
		Vector3 leftUpperLegT = humanPositions[1]; // boneIndexToMono.find(LeftUpperLeg)
		Vector3 rightUpperLegT = humanPositions[2]; // boneIndexToMono.find(RightUpperLeg)
		// this interpretation of "average left/right hips/shoulders vectors" seems most accurate
		Vector3 x = (leftUpperArmT + leftUpperLegT) - (rightUpperArmT + rightUpperLegT);
		Vector3 y  = (leftUpperArmT + rightUpperArmT) - (leftUpperLegT + rightUpperLegT);
		x = x.normalized();
		y = y.normalized();
		Vector3 z  = x.cross(y).normalized();
		x = y.cross(z);
		return Basis(x, y, z).get_rotation_quaternion();

	}

	static Quaternion get_hips_rotation_delta(LocalVector<Vector3>& humanPositions, Quaternion targetQ)
	{
		Quaternion sourceQ = getMassQ(humanPositions);
		//return Quaternion(targetQ.x, -targetQ.y, -targetQ.z, targetQ.w) * sourceQ.inverse()
		return targetQ * sourceQ.inverse();

	}

	static Vector3 getMassT(LocalVector<Vector3>& humanPositions, LocalVector<Quaternion>& humanRotations )
	{
		float sum = 1.0e-6;
		Vector3 out;
		for (int i = 0; i < humanPositions.size(); ++i)
		{
			// var postQ_inverse : = human_trait.postQ_inverse_exported[i]
			float m_HumanBoneMass = human_bone_mass[i];// # m_HumanBoneMass
			float axisLength = bone_lengths[i];// m_AxesArray.m_Length
			if (m_HumanBoneMass)
			{
				//var centerT : = Vector3(axisLength / 2, 0, 0) # GUESS : mass - center at half bone length
					//centerT = postQ_inverse.inverse() * centerT # Bring centerT from source coords to Godot coords
				Vector3 centerT = Vector3(0, axisLength / 2, 0);
				centerT = humanPositions[i] + humanRotations[i].xform(centerT);
				out += centerT * m_HumanBoneMass;
				sum += m_HumanBoneMass;

			}
		}
		return out / sum;
	}
	static Vector3 get_hips_position(LocalVector<Vector3>& humanPositions, LocalVector<Quaternion>& humanRotations, Quaternion deltaQ, Vector3 targetT)
	{
		Vector3 hipsPosition = humanPositions[0];
		Quaternion hipsRotation = humanRotations[0];
		Vector3 sourceT = getMassT(humanPositions, humanRotations);
		sourceT = deltaQ.xform(sourceT - hipsPosition);
		return targetT - sourceT;
	}

	static Quaternion calculate_humanoid_rotation(int bone_idx, Vector3 muscle_triplet, bool from_postq = false)
	{
		LocalVector<int8_t> muscle_from_bone  = MuscleFromBone[bone_idx];
		for (int i = 0; i < 3; ++i)
		{
			muscle_triplet[i] *= Math::deg_to_rad(
				MuscleDefaultMax[ muscle_triplet[i] < 0 ?
				 - MuscleDefaultMin[muscle_from_bone[i]] : muscle_from_bone[i]])
				* Signs[bone_idx][i];
		}
		Quaternion preQ = preQ_exported[bone_idx];
		if (from_postq)
		{
			preQ = postQ_inverse_exported[bone_idx].inverse().normalized();
		}
		if (!preQ.is_normalized())
		{
			//push_error("preQ is not normalized " + str(bone_idx));
		}
		Quaternion invPostQ = postQ_inverse_exported[bone_idx];
		if (!invPostQ.is_normalized())
		{
			//push_error("invPostQ is not normalized " + str(bone_idx))
		}
		Quaternion swing_res = swing_twist(muscle_triplet.x, -muscle_triplet.y, -muscle_triplet.z);
		if (!swing_res.is_normalized())
		{
			//push_error("swing_res is not normalized " + str(bone_idx) + " " + str(muscle_triplet));

		}
		Quaternion ret = preQ * swing_res * invPostQ;
		if (!ret.is_normalized())
		{
			//push_error("ret is not normalized " + str(bone_idx) + " " + str(muscle_triplet) + " " + str(preQ) + "," + str(swing_res) + "," + str(invPostQ));
		}
		ret = ret.normalized();
		if (!ret.is_normalized())
		{
			//push_error("ret is not normalized " + str(bone_idx) + " " + str(muscle_triplet) + " " + str(preQ) + "," + str(swing_res) + "," + str(invPostQ));
		}
		return ret;
	}
	static String to_unity_bone_path(String path)
	{
		Vector<String> strs = path.split("/", false);
		return "Skeleton3D:" + strs[strs.size() - 1];

	}
	
	class KeyframeIterator: public RefCounted
	{
	public:
		Dictionary curve;
		Array keyframes;
		Dictionary init_key;
		Dictionary final_key;
		Dictionary prev_key;
		Variant prev_slope;
		Dictionary next_key;
		Variant next_slope;
		bool has_slope;
		int key_idx = 0;
		bool is_eof = false;
		bool is_constant = false;
		bool is_mirrored = false;

		float CONSTANT_KEYFRAME_TIMESTAMP = 0.001;

		float timestamp = 0.0;
		void _init(Dictionary p_curve)
		{
			curve = p_curve;
			keyframes = curve["m_Curve"];
			is_mirrored = curve.get("unidot-mirror", false);
			init_key = keyframes[0];
			final_key = keyframes[-1];
			prev_key = init_key;
			next_key = init_key; // if len(keyframes) == 1 else keyframes[1]
			if (prev_key.has("outSlope"))
			{
				has_slope = true; // serializedVersion=3 has inSlope/outSlope while version=2 does not
				// Assets can actually mix and match version 2 and 3 even for related tracks.
				prev_slope = prev_key["outSlope"];
				next_slope = next_key["inSlope"];
			}
			is_constant = false;

		}
		void reset()
		{
			key_idx = 0;
			prev_key = init_key;
			next_key = init_key;
			is_eof = false;
			timestamp = 0.0;
			is_constant = false;

		}
		float get_next_timestamp(float timestep = -1.0)
		{
			if (is_eof)
				return 0.0;
			if (keyframes.size() == 1)
				return 0.0;
			if (is_constant && timestamp <((float) next_key["time"]) - CONSTANT_KEYFRAME_TIMESTAMP)
				// Make a new keyframe with the previous value CONSTANT_KEYFRAME_TIMESTAMP before the next.
				return ((float)next_key["time"]) - CONSTANT_KEYFRAME_TIMESTAMP;
			if (timestep <= 0)
				return next_key["time"];
			else if(timestamp + timestep >= (float)next_key["time"])
				return (float)next_key["time"];
			else
				return timestamp + timestep;

		}
		Variant fixup_strings(Variant val)
		{
			if (val.get_type() == Variant::STRING)
			{
				String t = val;
				val = t.to_float();
			}
			if (is_mirrored)
				//Every value comes through here, so it's a good place to make sure we negate everything
				val = -(float)val;
			return val;

		}
		bool is_finite(float x) const {
			return Math::is_finite(x) ;
		}
		Variant next(float timestep = -1.0)
		{
			if (is_eof)
				return Variant();
			if (prev_slope.get_type() == Variant::STRING)
			{
				String t = prev_slope;
				prev_slope = t.to_float();

			}
			if (next_slope.get_type() == Variant::STRING)
			{
				String t = next_slope;
				next_slope = t.to_float();

			}
			if (prev_slope.get_type() == Variant::FLOAT)
				is_constant = ! (is_finite(prev_slope) && is_finite(next_slope));
				// is_constant = (typeof(key_iter.prev_slope) == TYPE_STRING || typeof(key_iter.next_slope) == TYPE_STRING || is_inf(key_iter.prev_slope) || is_inf(key_iter.next_slope))
			else if (prev_slope.get_type() == Variant::VECTOR3)
			{
				Vector3 t_prev_slope = prev_slope;
				Vector3 t_next_slope = next_slope;
				is_constant = !(is_finite(t_prev_slope.x) && is_finite(t_next_slope.x) && is_finite(t_prev_slope.y) && is_finite(t_next_slope.y) && is_finite(t_prev_slope.z) && is_finite(t_next_slope.z));

			}
			else if (prev_slope.get_type() == Variant::QUATERNION)
			{
				Quaternion t_prev_slope = prev_slope;
				Quaternion t_next_slope = next_slope;
				is_constant = !(is_finite(t_prev_slope.x) && is_finite(t_next_slope.x) && is_finite(t_prev_slope.y) && is_finite(t_next_slope.y) && is_finite(t_prev_slope.z) && is_finite(t_next_slope.z) && is_finite(t_prev_slope.w) && is_finite(t_next_slope.w));

			}

			if (keyframes.size() == 1)
				{
					timestamp = 0.0;
					is_eof = true;
					return fixup_strings(init_key["value"]);
				}
			float constant_end_timestamp = ((float)next_key["time"]) - CONSTANT_KEYFRAME_TIMESTAMP;
			if (is_constant && timestamp < constant_end_timestamp)
			{
				// Make a new keyframe with the previous value CONSTANT_KEYFRAME_TIMESTAMP before the next.
				if (timestep <= 0)
					timestamp = constant_end_timestamp;
				else
					timestamp = MIN(timestamp + timestep, constant_end_timestamp);
				return fixup_strings(prev_key["value"]);

			}
			if (timestep <= 0)
				timestamp = next_key["time"];
			else
				timestamp += timestep;
			if (timestamp >= ((float)next_key["time"]) - CONSTANT_KEYFRAME_TIMESTAMP)
			{
				prev_key = next_key;
				prev_slope = prev_key.get_valid("outSlope");
				timestamp = prev_key["time"];
				key_idx += 1;
				if (key_idx >= keyframes.size())
					is_eof = true;
				else
				{
					next_key = keyframes[key_idx];
					next_slope = next_key.get_valid("inSlope");
				}
				return fixup_strings(prev_key["value"]);

			}
			// Todo: have caller determine desired keyframe depending on slope and accuracy
			// and clip length, to decide whether to use default linear interpolation or add more keyframes.
			// We could also have a setting to use cubic instead of linear for more smoothness but less accuracy.
			// FIXME: Assuming linear interpolation
			if (! (Math::is_equal_approx((float)next_key["time"], (float)prev_key["time"]) && timestamp >= (float)prev_key["time"] && timestamp <= (float)next_key["time"]))
				return Math::lerp(fixup_strings((float)prev_key["value"]), fixup_strings((float)next_key["value"]), (timestamp - (float)prev_key["time"]) / ((float)next_key["time"] - (float)prev_key["time"]));
			return fixup_strings((float)next_key["value"]);

		}








	};
	class LockstepKeyframeiterator : public RefCounted
	{
	public:
		LocalVector<Ref<KeyframeIterator>> kf_iters;

		float timestamp = 0.0;
		bool is_eof = false;
		bool perform_right_handed_position_conversion = false;
		TypedArray<float> results;
		void _init(LocalVector<Ref<KeyframeIterator>>& iters, bool is_position)
		{
			kf_iters = iters;
			results.resize(kf_iters.size());
			if (results.size() == 4)
				results[3] = 1; // normalized quaternion
			if (is_position)
				perform_right_handed_position_conversion = true;

		}
		void reset()
		{
			for(int i=0;i<kf_iters.size();i++)
			{
				kf_iters[i]->reset();
			}
			is_eof = false;
			timestamp = 0.0;
		}
		float get_next_timestamp(float timestep = -1.0)
		{
			Variant next_timestamp;
			for(int i=0;i<kf_iters.size();i++)
			{
				if (kf_iters[i].is_valid())
				{
					Ref<KeyframeIterator> key_iter = kf_iters[i];
					if (key_iter->is_eof)
						continue;
					key_iter->timestamp = timestamp;
					if (key_iter->prev_slope.get_type() == Variant::STRING)
					{
						String t = key_iter->prev_slope;
						key_iter->prev_slope = t.to_float();
					}
					if (key_iter->next_slope.get_type() == Variant::STRING)
					{
						String t = key_iter->next_slope;
						key_iter->next_slope = t.to_float();
					}

					if ((key_iter->prev_slope.get_type() == Variant::FLOAT) && (key_iter->next_slope.get_type() == Variant::FLOAT))
						key_iter->is_constant = Math::is_inf((float)key_iter->prev_slope) || Math::is_inf((float)key_iter->next_slope);
					float this_next_timestamp = key_iter->get_next_timestamp();
					if ((next_timestamp.get_type() != Variant::FLOAT) || (float)next_timestamp > this_next_timestamp)
						next_timestamp = this_next_timestamp;

				}

			}
			if (next_timestamp.get_type() != Variant::FLOAT)
				{
					is_eof = true;
					return 0.0;
				}
			else if( timestep <= 0)
				return next_timestamp;
			else
				return MIN(timestamp + timestep, (float)next_timestamp);

		}
		Variant next(float timestep = -1.0)
		{
			int valid_components = 0;
			int new_eof_components = 0;
			float next_timestamp = get_next_timestamp(timestep);
			if (!is_eof)
			{
				timestamp = next_timestamp;
				for (int i = 0; i < kf_iters.size(); i++)
				{
					if (kf_iters[i].is_valid())
					{
						Ref<KeyframeIterator> key_iter = kf_iters[i];
						if (key_iter->is_eof)
							continue;
						Variant res;
						if (timestep <= 0.0)
							res = key_iter->next(timestamp - key_iter->timestamp);
						else
							res = key_iter->next(timestep);

						results[i] = (float)res;
						valid_components += 1;
						if (key_iter->is_eof)
							new_eof_components += 1;
						key_iter->timestamp = timestamp;

					}

				}
			}
			if (new_eof_components == valid_components)
				is_eof = true;
			if (results.size() == 3)
			{
				if (perform_right_handed_position_conversion)
					return Vector3(-(float)results[0], results[1], results[2]);
				return Vector3(results[0], results[1], results[2]);

			}
			else if (results.size() == 4)
			{
				if (valid_components == 0)
				{//pass # push_error("next() called when all sub-tracks are eof or null")
				}
				else if (Quaternion(results[0], results[1], results[2], results[3]).normalized().is_equal_approx(Quaternion()))
				{
					//pass # push_error("next() valid components " + str(valid_components) + " returned an identity quaternion: " + str(results))
				}
				return Quaternion(results[0], -(float)results[1], -(float)results[2], results[3]).normalized();

			}
			return results;

		}

	};












}

class ResourceFormatUnityAnimation : public ResourceFormatLoader {
public:

	virtual Ref<Resource> load(const String &p_path, const String &p_original_path = "", Error *r_error = nullptr, bool p_use_sub_threads = false, float *r_progress = nullptr, CacheMode p_cache_mode = CACHE_MODE_REUSE) override
	{
		if (r_error) {
			*r_error = ERR_CANT_OPEN;
		}

		Error err;
		Ref<FileAccess> f = FileAccess::open(p_path, FileAccess::READ, &err);
		if(f.is_null())
		{
			return Ref<Resource>();
		}
		String yaml_anim = f->get_as_text();

		Ref<Animation> anim;
		anim.instantiate();
		Ref<JSON> json = DataTableManager::get_singleton()->parse_yaml(yaml_anim);

		Dictionary dict = json->get_data();
		Callable on_load_animation =  DataTableManager::get_singleton()->get_animation_load_cb();
		Dictionary clip = dict["AnimationClip"];
		on_load_animation.call(clip,false,anim);
		anim->optimize();
		print_line(L"ResourceFormatUnityAnimation: 加载Unity动作文件 :" + p_path);	
		return anim;
	}
	virtual void get_recognized_extensions(List<String> *p_extensions) const override
	{		
		p_extensions->push_back("asset");
	}
	virtual bool handles_type(const String &p_type) const override
	{
		return p_type == "Animation";
	}
	virtual String get_resource_type(const String &p_path) const override
	{
		if (p_path.get_extension().to_lower() == "asset") {
			return "Animation";
		}
		return "";
	}

	virtual ~ResourceFormatUnityAnimation() {}
};

class UnityLinkServerEditorPlugin : public EditorPlugin {
	GDCLASS(UnityLinkServerEditorPlugin, EditorPlugin);

	UnityLinkServer server;

	bool started = false;

private:
	void _notification(int p_what);

public:
	UnityLinkServerEditorPlugin();
	void start();
	void stop();
	
	Ref<ResourceFormatUnityAnimation> resource_loader_unity_animation;
};
void UnityLinkServerEditorPluginRegister::initialize()
{
	EditorPlugins::add_by_type<UnityLinkServerEditorPlugin>();
}

UnityLinkServerEditorPlugin::UnityLinkServerEditorPlugin() {
    
	Ref<GameHelpInspectorPlugin> plugin;
	plugin.instantiate();
	
	EditorInspector::add_inspector_plugin(plugin);
	EditorNode::get_log()->add_message("register:GameHelpInspectorPlugin");
}

void UnityLinkServerEditorPlugin::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			start();
		} break;

		case NOTIFICATION_EXIT_TREE: {
			stop();
		} break;

		case NOTIFICATION_INTERNAL_PROCESS: {
			// The main loop can be run again during request processing, which modifies internal state of the protocol.
			// Thus, "polling" is needed to prevent it from parsing other requests while the current one isn't finished.
			if (started ) {
				server.poll();
			}
		} break;

		case EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED: {
		} break;
	}
}

void UnityLinkServerEditorPlugin::start() {
	resource_loader_unity_animation.instantiate();
	//ResourceLoader::add_resource_format_loader(resource_loader_unity_animation);
	server.start() ;
	{
		EditorNode::get_log()->add_message("--- unity link server started port 9010---", EditorLog::MSG_TYPE_EDITOR);
		set_process_internal(true);
		started = true;
	}
	
	
}

void UnityLinkServerEditorPlugin::stop() {
	server.stop();
	started = false;
	EditorNode::get_log()->add_message("--- unity link server stopped ---", EditorLog::MSG_TYPE_EDITOR);
	//ResourceLoader::remove_resource_format_loader(resource_loader_unity_animation);
	resource_loader_unity_animation.unref();
}
#endif
