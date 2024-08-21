
#include "../unity/unity_link_server.h"
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
#include "../unity/unity_link_server.h"
#include "beehave_graph_editor.h"
#endif
class CharacterBodyMainLable : public Label
{
	GDCLASS(CharacterBodyMainLable, Label);
	static void _bind_methods() {
		
	}
public:

	CharacterBodyMainLable()
	{

		this->set_text(L"角色编辑面板");
		set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
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
protected:
	CharacterBodyMain* body_main = nullptr;
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
			HBoxContainer* add_button = memnew(HBoxContainer);		
			
			BeehaveGraphProperty* section = memnew(BeehaveGraphProperty);
			// 初始化
			if(object->get_root_node().is_null())
			{
				object->set_root_node(memnew(BeehaveCompositeParallel));
			}
			section->setup(object, p_select_node_property_vbox,add_button);
			p_plugin->add_custom_control( section);
			p_plugin->add_custom_control( p_select_node_property_vbox);
			p_plugin->add_custom_control( add_button);

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
			
			if(p_path == "update_mode")
			{
				CharacterBodyMainLable* lable = memnew(CharacterBodyMainLable);
				lable->set_body_main(body_main);
				add_custom_control( lable);
			}
			return false;
		}
		return false;
	}

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
}
#endif
