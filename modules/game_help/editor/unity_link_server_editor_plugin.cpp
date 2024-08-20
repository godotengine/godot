
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


#include "modules/game_help/logic/character_ai/blackboard_set_item/animator_blackboard_item_bool.h"
#include "modules/game_help/logic/character_ai/blackboard_set_item/animator_blackboard_item_float.h"
#include "modules/game_help/logic/character_ai/blackboard_set_item/animator_blackboard_item_int.h"
#include "modules/game_help/logic/character_ai/blackboard_set_item/animator_blackboard_item_string.h"

#include "condition_editor.h"

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
class BlackbordSet_ED  : public HBoxContainer
{
	GDCLASS(BlackbordSet_ED, HBoxContainer);
	static void _bind_methods() {
		
	}

public:
	BlackbordSet_ED(){}
	

	void _notification(int p_what) {
		switch (p_what) {
			case NOTIFICATION_ENTER_TREE: {
				// 绑定黑板设置回调
			} break;
			case NOTIFICATION_EXIT_TREE: {
				// 解除绑定黑板设置回调
			} break;
		}
	}
	Ref<Texture2D> get_type_icon(const StringName& p_type)
	{
		#ifdef TOOLS_ENABLED
		return EditorNode::get_singleton()->get_editor_theme()->get_icon(p_type, "EditorIcons");
		#else
		return get_theme()->get_icon(p_type, "EditorIcons");
		#endif
	}
	void setup(Ref<AnimatorBlackboardSet> p_object,Ref<AnimatorBlackboardSetItemBase> p_blackboard_set_item)
	{
		object = p_object;
		blackboard_set_item = p_blackboard_set_item;
		this->set_layout_mode(LayoutMode::LAYOUT_MODE_CONTAINER);


		HSeparator *sep = memnew(HSeparator);
		sep->set_layout_mode(LayoutMode::LAYOUT_MODE_CONTAINER);
		sep->set_custom_minimum_size(Size2(10, 10));
		add_child(sep);


		type_lable = memnew(Button);
		type_lable->set_layout_mode(LayoutMode::LAYOUT_MODE_CONTAINER);
		type_lable->set_custom_minimum_size(Size2(60, 10));
		type_lable->set_v_size_flags(SIZE_FILL);		
		type_lable->set_icon(get_type_icon(Variant::get_type_name(Variant::INT)));
		type_lable->set_disabled(true);
		type_lable->set_focus_mode(FOCUS_NONE);
		type_lable->set_text("Int");
		
		add_child(type_lable);


		sep = memnew(HSeparator);
		sep->set_layout_mode(LayoutMode::LAYOUT_MODE_CONTAINER);
		sep->set_custom_minimum_size(Size2(10, 10));
		add_child(sep);

		// PropertyName
		{
			property_name_bt = memnew(Button);
			property_name_bt->set_name("PropertyName");
			property_name_bt->set_layout_mode(LayoutMode::LAYOUT_MODE_CONTAINER);
			property_name_bt->set_h_size_flags(SIZE_EXPAND_FILL);
			property_name_bt->set_v_size_flags(SIZE_EXPAND_FILL);
			property_name_bt->set_stretch_ratio(0.61f);
			property_name_bt->set_text("Param");
			property_name_bt->connect("pressed", callable_mp(this, &BlackbordSet_ED::_on_property_name_pressed));
			add_child(property_name_bt);

			property_name_list = memnew(PopupMenu);
			property_name_list->set_name("PopupMenu");
			property_name_list->set_auto_translate_mode(AUTO_TRANSLATE_MODE_ALWAYS);
			property_name_list->connect("id_pressed", callable_mp(this, &BlackbordSet_ED::_on_property_name_list_id_pressed));
			property_name_bt->add_child(property_name_list);

		}


		// Value
		{
			value_parent = memnew(MarginContainer);
			value_parent->set_modulate(Color(0.001f, 0.803907f, 0.933103f, 0.9f));
			value_parent->set_name("Value");
			value_parent->set_layout_mode(LayoutMode::LAYOUT_MODE_CONTAINER);
			value_parent->set_h_size_flags(SIZE_EXPAND_FILL);
			value_parent->set_stretch_ratio(0.46f);
			add_child(value_parent);

			// BlackbordProperty
			{
				value_property_bt = memnew(Button);
				value_property_bt->set_name("ValueProperty");
				value_property_bt->set_visible(false);
				value_property_bt->set_layout_mode(LayoutMode::LAYOUT_MODE_CONTAINER);
				value_property_bt->set_v_size_flags(SIZE_EXPAND_FILL);	
				value_property_bt->connect("pressed", callable_mp(this, &BlackbordSet_ED::_on_value_property_pressed));	
				value_parent->add_child(value_property_bt);

				value_property_list = memnew(PopupMenu);
				value_property_list->set_name("PopupMenu");
				value_property_list->set_auto_translate_mode(AUTO_TRANSLATE_MODE_ALWAYS);
				value_property_list->connect("id_pressed", callable_mp(this, &BlackbordSet_ED::_on_value_property_list_id_pressed));
				value_property_bt->add_child(value_property_list);

			}

			value_num = memnew(LineEdit);
			value_num->set_name("ValueNum");
			value_num->set_visible(false);
			value_num->set_layout_mode(LayoutMode::LAYOUT_MODE_CONTAINER);
			value_num->connect("text_submitted", callable_mp(this, &BlackbordSet_ED::_on_value_num_text_changed));
			value_parent->add_child(value_num);

			value_string = memnew(LineEdit);
			value_string->set_name("ValueString");
			value_string->set_visible(false);
			value_string->set_layout_mode(LayoutMode::LAYOUT_MODE_CONTAINER);
			value_string->connect("text_submitted", callable_mp(this, &BlackbordSet_ED::_on_value_string_text_changed));
			value_parent->add_child(value_string);

			value_bool = memnew(CheckButton);
			value_bool->set_name("ValueBool");
			value_bool->set_visible(false);
			value_bool->set_layout_mode(LayoutMode::LAYOUT_MODE_CONTAINER);
			value_bool->connect("toggled", callable_mp(this, &BlackbordSet_ED::_on_value_bool_toggled));
			value_parent->add_child(value_bool);

			value_range = memnew(HSlider);
			value_range->set_name("ValueRange");
			value_range->set_visible(false);
			value_range->set_layout_mode(LayoutMode::LAYOUT_MODE_CONTAINER);
			value_range->set_v_size_flags(SIZE_FILL);
			value_parent->add_child(value_range);


		}

		is_value_property = memnew(CheckBox);
		is_value_property->set_name("IsValueProperty");
		is_value_property->set_layout_mode(LayoutMode::LAYOUT_MODE_CONTAINER);
		is_value_property->set_h_size_flags(SIZE_SHRINK_END);
		is_value_property->set_flat(true);
		is_value_property->connect("toggled", callable_mp(this, &BlackbordSet_ED::_on_value_is_property_toggled));
		add_child(is_value_property);

		
		remove_bt = memnew(Button);
		remove_bt->set_name("Remove");
		remove_bt->set_layout_mode(LayoutMode::LAYOUT_MODE_CONTAINER);
		remove_bt->set_h_size_flags(SIZE_SHRINK_END|SIZE_EXPAND);
		remove_bt->set_stretch_ratio(0);
		remove_bt->set_text(" X ");
		remove_bt->connect("pressed", callable_mp(this, &BlackbordSet_ED::_on_remove_bt_pressed));
		add_child(remove_bt);

		sep = memnew(HSeparator);
		sep->set_layout_mode(LayoutMode::LAYOUT_MODE_CONTAINER);
		sep->set_custom_minimum_size(Size2(20, 10));
		add_child(sep);


		initialize();
	}

	void initialize() {
		property_name_list->clear();
		Array blackbord_propertys = blackboard_set_item->get_blackbord_propertys();
		for(int i=0;i<blackbord_propertys.size();i++)
		{
			property_name_list->add_item(String(blackbord_propertys[i]));
		}
		property_name_bt->set_text(String(blackboard_set_item->get_property_name()));
		
		
		update_state();

		{
			Ref<AnimatorBlackboardSetItemInt> int_blackboard_set_item = blackboard_set_item;
			Ref<AnimatorBlackboardSetItemFloat> float_blackboard_set_item = blackboard_set_item;
			Ref<AnimatorBlackboardSetItemBool> bool_blackboard_set_item = blackboard_set_item;
			Ref<AnimatorBlackboardSetItemString> string_blackboard_set_item = blackboard_set_item;
			if(int_blackboard_set_item.is_valid())
			{
				type_lable->set_icon(get_type_icon(Variant::get_type_name(Variant::INT)));
				type_lable->set_text("Int");
			}
			else if(float_blackboard_set_item.is_valid())
			{
				type_lable->set_icon(get_type_icon(Variant::get_type_name(Variant::FLOAT)));
				type_lable->set_text("Float");
			}
			else if(bool_blackboard_set_item.is_valid())
			{
				type_lable->set_icon(get_type_icon(Variant::get_type_name(Variant::BOOL)));
				type_lable->set_text("Bool");
			}
			else if(string_blackboard_set_item.is_valid())
			{
				type_lable->set_icon(get_type_icon(Variant::get_type_name(Variant::STRING)));
				type_lable->set_text("String");
			}

			bool is_property = blackboard_set_item->get_is_value_by_property();
			is_value_property->set_pressed(is_property);
			if(is_property)
			{
				value_property_list->clear();
				for(int i=0;i<blackbord_propertys.size();i++)
				{
					value_property_list->add_item(String(blackbord_propertys[i]));
				}
				value_property_bt->set_visible(true);
				value_property_bt->set_text(String(blackboard_set_item->get_value_property_name()));
				return;
			}

			if(int_blackboard_set_item.is_valid())
			{
				value_num->set_text(String::num_int64(int_blackboard_set_item->get_value()));
				value_num->set_visible(true);
				return;
			}


			if(float_blackboard_set_item.is_valid())
			{
				value_num->set_text(String::num(float_blackboard_set_item->get_value()));
				value_num->set_visible(true);
				return;
			}
			
			if(bool_blackboard_set_item.is_valid())
			{
				value_bool->set_visible(true);
				value_bool->set_pressed(bool_blackboard_set_item->get_value());
				if(bool_blackboard_set_item->get_value())
				{
					value_bool->set_modulate(Color(0, 0.92549, 0.164706, 1));
				}else
				{
					value_bool->set_modulate(Color(1, 0.255238, 0.196011, 1));
				}
				return;
			}
			

			if(string_blackboard_set_item.is_valid())
			{
				value_string->set_text(string_blackboard_set_item->get_value());
				value_string->set_visible(true);
				return;
			}
				
		}
	}
protected:
	void _on_property_name_pressed() {
		popup_on_target(property_name_list,property_name_bt);		
	}

	void _on_property_name_list_id_pressed(int p_id) {
		property_name_bt->set_text(String(property_name_list->get_item_text(p_id)));
		blackboard_set_item->set_property_name(property_name_list->get_item_text(p_id));
		update_state();
	}



	void _on_value_property_pressed() {
		popup_on_target(value_property_list,value_property_bt);
	}

	void _on_value_property_list_id_pressed(int p_id) {
		value_property_bt->set_text(String(value_property_list->get_item_text(p_id)));
		blackboard_set_item->set_value_property_name(value_property_list->get_item_text(p_id));
		update_state();
	}

	void _on_value_num_text_changed(const String& p_string) {
		Ref<AnimatorBlackboardSetItemInt> int_condition = blackboard_set_item;
		if(int_condition.is_valid())
		{
			int_condition->set_value(p_string.to_int());
			value_num->set_text(String::num_int64(int_condition->get_value()));
			update_state();
			return;
		}
		Ref<AnimatorBlackboardSetItemFloat> float_blackboard_set_item = blackboard_set_item;
		if(float_blackboard_set_item.is_valid())
		{
			float_blackboard_set_item->set_value(p_string.to_float());
			value_num->set_text(String::num(float_blackboard_set_item->get_value()));
			update_state();
			return;
		}
	}

	void _on_value_bool_toggled(bool p_pressed) {
		Ref<AnimatorBlackboardSetItemBool> bool_blackboard_set_item = blackboard_set_item;
		if(bool_blackboard_set_item.is_valid())
		{
			bool_blackboard_set_item->set_value(p_pressed);
			value_bool->set_pressed(bool_blackboard_set_item->get_value());	
			if(p_pressed)
			{
				value_bool->set_modulate(Color(0, 0.92549, 0.164706, 1));
			}else
			{
				value_bool->set_modulate(Color(1, 0.255238, 0.196011, 1));
			}
			update_state();
			return;
		}
	}

	void _on_value_string_text_changed(const String& p_string) {
		Ref<AnimatorBlackboardSetItemString> string_blackboard_set_item = blackboard_set_item;
		if(string_blackboard_set_item.is_valid())
		{
			string_blackboard_set_item->set_value(p_string);
			value_string->set_text(string_blackboard_set_item->get_value());
			update_state();
			return;
		}
	}

	void _on_value_is_property_toggled(bool p_pressed) {
		blackboard_set_item->set_is_value_by_property(p_pressed);
		object->notify_property_list_changed();
	}
	void _on_remove_bt_pressed() {
		object->remove_item(blackboard_set_item);
		object->notify_property_list_changed();
	}
	void update_state()
	{
	}
protected:
	void popup_on_target(PopupMenu *p_menu,Control* p_target) {
		p_menu->reset_size();
		Rect2i usable_rect =  Rect2i(Point2i(0,0), DisplayServer::get_singleton()->window_get_size_with_decorations());
		Rect2i cp_rect = Rect2i(Point2i(0,0), p_target->get_size());

		for(int i = 0; i < 4; i++) {
			if(i > 1)
			{
				cp_rect.position.y = p_target->get_global_position().x - p_target->get_size().y;
			}
			else
			{
				cp_rect.position.y = p_target->get_global_position().y + p_target->get_size().y;
			}
			if(i & 1) {
				cp_rect.position.x = p_target->get_global_position().x ;
			}
			else
			{
				cp_rect.position.x = p_target->get_global_position().x - MAX(0,cp_rect.size.x - p_target->get_size().x);
			}
			if(usable_rect.encloses(cp_rect))
			{
				break;
			}
		}
		Point2i main_window_position = DisplayServer::get_singleton()->window_get_position();
		Point2i popup_position = main_window_position + Point2i(cp_rect.position);
		p_menu->set_position(popup_position);
		p_menu->popup();
		
	}

	Button* type_lable = nullptr;


	Button* property_name_bt = nullptr;
	PopupMenu* property_name_list = nullptr;


	MarginContainer* value_parent = nullptr;
	
	Button* value_property_bt = nullptr;
	PopupMenu* value_property_list = nullptr;

	LineEdit* value_num = nullptr;
	LineEdit* value_string = nullptr;
	CheckButton* value_bool = nullptr;
	HSlider* value_range = nullptr;


	CheckBox* is_value_property = nullptr;
	Button* remove_bt = nullptr;

	Ref<AnimatorBlackboardSetItemBase> blackboard_set_item ;
	Ref<AnimatorBlackboardSet> object ;
};



void BlackbordSetSection::_on_header_pressed() {
	set_collapsed(!is_collapsed());
}

void BlackbordSetSection::set_filter(String p_filter_text) {

}

void BlackbordSetSection::add_condition(Control* p_task_button) {
	tasks_container->add_child(p_task_button);
}

void BlackbordSetSection::set_collapsed(bool p_collapsed) {
	tasks_container->set_visible(!p_collapsed);
#ifdef TOOLS_ENABLED
		object->editor_set_section_unfold("Change List", !p_collapsed);
#endif
	section_header->set_icon(p_collapsed ? theme_cache.arrow_right_icon : theme_cache.arrow_down_icon);
}

bool BlackbordSetSection::is_collapsed() const {
	return !tasks_container->is_visible();
}

void BlackbordSetSection::_do_update_theme_item_cache() {
	theme_cache.arrow_down_icon = get_theme_icon(SNAME("GuiTreeArrowDown"), SNAME("EditorIcons"));
	theme_cache.arrow_right_icon = get_theme_icon(SNAME("GuiTreeArrowRight"), SNAME("EditorIcons"));
}

void BlackbordSetSection::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			section_header->connect(SNAME("pressed"), callable_mp(this, &BlackbordSetSection::_on_header_pressed));
		} break;
		case NOTIFICATION_THEME_CHANGED: {
			_do_update_theme_item_cache();
			section_header->set_icon(is_collapsed() ? theme_cache.arrow_right_icon : theme_cache.arrow_down_icon);
			section_header->add_theme_font_override(SNAME("font"), get_theme_font(SNAME("bold"), SNAME("EditorFonts")));
		} break;
	}
}
void BlackbordSetSection::_bind_methods() {
	ADD_SIGNAL(MethodInfo("task_button_pressed"));
	ADD_SIGNAL(MethodInfo("task_button_rmb"));
}

BlackbordSetSection::BlackbordSetSection() {

	HBoxContainer *hb = memnew(HBoxContainer);
	hb->set_layout_mode(LayoutMode::LAYOUT_MODE_CONTAINER);
	add_child(hb);


	section_header = memnew(Button);
	section_header->set_layout_mode(LayoutMode::LAYOUT_MODE_CONTAINER);
	section_header->set_h_size_flags(SIZE_EXPAND_FILL);
	section_header->set_v_size_flags(SIZE_EXPAND_FILL);
	hb->add_child(section_header);
	

	section_header->set_focus_mode(FOCUS_NONE);

	tasks_container = memnew(VBoxContainer);
	add_child(tasks_container);
}

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
			Button* add_button = memnew(Button);		
			
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
