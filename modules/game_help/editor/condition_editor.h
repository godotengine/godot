#pragma once
#include "scene/gui/box_container.h"
#include "scene/gui/separator.h"
#include "editor/editor_node.h"
#include "../logic/animator/body_animator.h"
#include "../logic/character_ai/body_animator_logic.h"
#include "modules/game_help/logic/character_ai/animator_condition.h"
#include "modules/game_help/logic/character_ai/condition/animator_condition_bool.h"
#include "modules/game_help/logic/character_ai/condition/animator_condition_float.h"
#include "modules/game_help/logic/character_ai/condition/animator_condition_int.h"
#include "modules/game_help/logic/character_ai/condition/animator_condition_string.h"


class ConditionSection : public VBoxContainer {
	GDCLASS(ConditionSection, VBoxContainer);

private:
	struct ThemeCache {
		Ref<Texture2D> arrow_down_icon;
		Ref<Texture2D> arrow_right_icon;
	} theme_cache;

	VBoxContainer *tasks_container;
	Button *section_header;
	CheckButton *state_button;

	Ref<CharacterAnimatorCondition> object;
	bool is_include_condition = false;

	void _on_header_pressed();

protected:
	static void _bind_methods();

	void _notification(int p_what);

	virtual void _do_update_theme_item_cache();

public:
	void setup(Ref<CharacterAnimatorCondition> p_object, bool is_include)
	{

		object = p_object;
		is_include_condition = is_include;
		if(is_include_condition)
		{
#ifdef TOOLS_ENABLED
			set_collapsed(!object->editor_is_section_unfolded("Include Conditions"));
#endif
		}else
		{
#ifdef TOOLS_ENABLED
			set_collapsed(!object->editor_is_section_unfolded("Exclude Conditions"));
#endif
		}
		update_state();
	}
	void set_filter(String p_filter);
	void add_condition(Control* p_task_button);

	void set_collapsed(bool p_collapsed);
	bool is_collapsed() const;

	String get_category_name() const { return section_header->get_text(); }
	void set_category_name(const String &p_cat) { section_header->set_text(p_cat); }
	void update_state();

	ConditionSection();
	~ConditionSection();
};

class Condition_ED  : public HBoxContainer
{
	GDCLASS(Condition_ED, HBoxContainer);
	static void _bind_methods() {
		
	}

public:
	Condition_ED(){}
	

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
	void setup(ConditionSection* p_section ,Ref<CharacterAnimatorCondition> p_object,Ref<AnimatorAIStateConditionBase> p_condition,bool is_include)
	{
		parent_section = p_section;
		object = p_object;
		condition = p_condition;
		is_include_condition = is_include;
		this->set_layout_mode(LayoutMode::LAYOUT_MODE_CONTAINER);


		HSeparator *sep = memnew(HSeparator);
		sep->set_layout_mode(LayoutMode::LAYOUT_MODE_CONTAINER);
		sep->set_custom_minimum_size(Size2(10, 10));
		add_child(sep);

		state_button = memnew(CheckButton);
		state_button->set_disabled(true);
		state_button->set_focus_mode(FOCUS_NONE);
		state_button->set_layout_mode(LayoutMode::LAYOUT_MODE_CONTAINER);
		add_child(state_button);

		sep = memnew(HSeparator);
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
			property_name_bt->connect("pressed", callable_mp(this, &Condition_ED::_on_property_name_pressed));
			add_child(property_name_bt);

			property_name_list = memnew(PopupMenu);
			property_name_list->set_name("PopupMenu");
			property_name_list->set_auto_translate_mode(AUTO_TRANSLATE_MODE_ALWAYS);
			property_name_list->connect("id_pressed", callable_mp(this, &Condition_ED::_on_property_name_list_id_pressed));
			property_name_bt->add_child(property_name_list);

		}

		// Comparation
		{
			comparison_bt = memnew(Button);
			comparison_bt->set_name("Comparison");
			comparison_bt->set_modulate(Color(0.69654f, 0.356956f, 0.0001f, 0.9f));
			comparison_bt->set_layout_mode(LayoutMode::LAYOUT_MODE_CONTAINER);
			comparison_bt->set_h_size_flags(SIZE_EXPAND | SIZE_SHRINK_CENTER);
			comparison_bt->set_v_size_flags(SIZE_EXPAND_FILL);
			comparison_bt->set_stretch_ratio(0.02f);
			comparison_bt->set_text("==");
			comparison_bt->connect("pressed", callable_mp(this, &Condition_ED::_on_comparison_pressed));
			add_child(comparison_bt);

			comparison_list = memnew(PopupMenu);
			comparison_list->set_name("PopupMenu");
			comparison_list->set_auto_translate_mode(AUTO_TRANSLATE_MODE_ALWAYS);
			comparison_list->connect("id_pressed", callable_mp(this, &Condition_ED::_on_comparison_list_id_pressed));
			comparison_bt->add_child(comparison_list);
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
				value_property_bt->connect("pressed", callable_mp(this, &Condition_ED::_on_value_property_pressed));	
				value_parent->add_child(value_property_bt);

				value_property_list = memnew(PopupMenu);
				value_property_list->set_name("PopupMenu");
				value_property_list->set_auto_translate_mode(AUTO_TRANSLATE_MODE_ALWAYS);
				value_property_list->connect("id_pressed", callable_mp(this, &Condition_ED::_on_value_property_list_id_pressed));
				value_property_bt->add_child(value_property_list);

			}

			value_num = memnew(LineEdit);
			value_num->set_name("ValueNum");
			value_num->set_visible(false);
			value_num->set_layout_mode(LayoutMode::LAYOUT_MODE_CONTAINER);
			value_num->connect("text_submitted", callable_mp(this, &Condition_ED::_on_value_num_text_changed));
			value_parent->add_child(value_num);

			value_string = memnew(LineEdit);
			value_string->set_name("ValueString");
			value_string->set_visible(false);
			value_string->set_layout_mode(LayoutMode::LAYOUT_MODE_CONTAINER);
			value_string->connect("text_submitted", callable_mp(this, &Condition_ED::_on_value_string_text_changed));
			value_parent->add_child(value_string);

			value_bool = memnew(CheckButton);
			value_bool->set_name("ValueBool");
			value_bool->set_visible(false);
			value_bool->set_layout_mode(LayoutMode::LAYOUT_MODE_CONTAINER);
			value_bool->connect("toggled", callable_mp(this, &Condition_ED::_on_value_bool_toggled));
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
		is_value_property->connect("toggled", callable_mp(this, &Condition_ED::_on_value_is_property_toggled));
		add_child(is_value_property);

		
		remove_bt = memnew(Button);
		remove_bt->set_name("Remove");
		remove_bt->set_layout_mode(LayoutMode::LAYOUT_MODE_CONTAINER);
		remove_bt->set_h_size_flags(SIZE_SHRINK_END|SIZE_EXPAND);
		remove_bt->set_stretch_ratio(0);
		remove_bt->set_text(" X ");
		remove_bt->connect("pressed", callable_mp(this, &Condition_ED::_on_remove_bt_pressed));
		add_child(remove_bt);

		sep = memnew(HSeparator);
		sep->set_layout_mode(LayoutMode::LAYOUT_MODE_CONTAINER);
		sep->set_custom_minimum_size(Size2(20, 10));
		add_child(sep);


		initialize();
	}

	void initialize() {
		property_name_list->clear();
		Array blackbord_propertys = condition->get_blackbord_propertys();
		for(int i=0;i<blackbord_propertys.size();i++)
		{
			property_name_list->add_item(String(blackbord_propertys[i]));
		}
		property_name_bt->set_text(String(condition->get_property_name()));
		


		comparison_list->clear();
		Array compare_value = condition->get_compare_value();
		for(int i=0;i<compare_value.size();i++)
		{
			comparison_list->add_item(String(compare_value[i]));
		}
		comparison_bt->set_text(String(condition->get_compare_type_name()));
		
		update_state();

		{
			Ref<AnimatorAIStateIntCondition> int_condition = condition;
			Ref<AnimatorAIStateFloatCondition> float_condition = condition;
			Ref<AnimatorAIStateBoolCondition> bool_condition = condition;
			Ref<AnimatorAIStateStringNameCondition> string_condition = condition;
			if(int_condition.is_valid())
			{
				type_lable->set_icon(get_type_icon(Variant::get_type_name(Variant::INT)));
				type_lable->set_text("Int");
			}
			else if(float_condition.is_valid())
			{
				type_lable->set_icon(get_type_icon(Variant::get_type_name(Variant::FLOAT)));
				type_lable->set_text("Float");
			}
			else if(bool_condition.is_valid())
			{
				type_lable->set_icon(get_type_icon(Variant::get_type_name(Variant::BOOL)));
				type_lable->set_text("Bool");
			}
			else if(string_condition.is_valid())
			{
				type_lable->set_icon(get_type_icon(Variant::get_type_name(Variant::STRING)));
				type_lable->set_text("String");
			}

			bool is_property = condition->get_is_value_by_property();
			is_value_property->set_pressed(is_property);
			if(is_property)
			{
				value_property_list->clear();
				for(int i=0;i<blackbord_propertys.size();i++)
				{
					value_property_list->add_item(String(blackbord_propertys[i]));
				}
				value_property_bt->set_visible(true);
				value_property_bt->set_text(String(condition->get_value_property_name()));
				return;
			}

			if(int_condition.is_valid())
			{
				value_num->set_text(String::num_int64(int_condition->get_value()));
				value_num->set_visible(true);
				return;
			}


			if(float_condition.is_valid())
			{
				value_num->set_text(String::num(float_condition->get_value()));
				value_num->set_visible(true);
				return;
			}
			
			if(bool_condition.is_valid())
			{
				value_bool->set_visible(true);
				value_bool->set_pressed(bool_condition->get_value());
				if(bool_condition->get_value())
				{
					value_bool->set_modulate(Color(0, 0.92549, 0.164706, 1));
				}else
				{
					value_bool->set_modulate(Color(1, 0.255238, 0.196011, 1));
				}
				return;
			}
			

			if(string_condition.is_valid())
			{
				value_string->set_text(string_condition->get_value());
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
		condition->set_property_name(property_name_list->get_item_text(p_id));
		update_state();
	}

	void _on_comparison_pressed() {
		popup_on_target(comparison_list,comparison_bt);
	}

	void _on_comparison_list_id_pressed(int p_id) {
		comparison_bt->set_text(String(comparison_list->get_item_text(p_id)));
		condition->set_compare_type_name(comparison_list->get_item_text(p_id));
		update_state();
	}

	void _on_value_property_pressed() {
		popup_on_target(value_property_list,value_property_bt);
	}

	void _on_value_property_list_id_pressed(int p_id) {
		value_property_bt->set_text(String(value_property_list->get_item_text(p_id)));
		condition->set_value_property_name(value_property_list->get_item_text(p_id));
		update_state();
	}

	void _on_value_num_text_changed(const String& p_string) {
		Ref<AnimatorAIStateIntCondition> int_condition = condition;
		if(int_condition.is_valid())
		{
			int_condition->set_value(p_string.to_int());
			value_num->set_text(String::num_int64(int_condition->get_value()));
			update_state();
			return;
		}
		Ref<AnimatorAIStateFloatCondition> float_condition = condition;
		if(float_condition.is_valid())
		{
			float_condition->set_value(p_string.to_float());
			value_num->set_text(String::num(float_condition->get_value()));
			update_state();
			return;
		}
	}

	void _on_value_bool_toggled(bool p_pressed) {
		Ref<AnimatorAIStateBoolCondition> bool_condition = condition;
		if(bool_condition.is_valid())
		{
			bool_condition->set_value(p_pressed);
			value_bool->set_pressed(bool_condition->get_value());
			
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
		Ref<AnimatorAIStateStringNameCondition> string_condition = condition;
		if(string_condition.is_valid())
		{
			string_condition->set_value(p_string);
			value_string->set_text(string_condition->get_value());
			update_state();
			return;
		}
	}

	void _on_value_is_property_toggled(bool p_pressed) {
		condition->set_is_value_by_property(p_pressed);
		object->notify_property_list_changed();
	}
	void _on_remove_bt_pressed() {
		if(is_include_condition)
		{
			object->remove_include_condition(condition);
		}else
		{
			object->remove_exclude_condition(condition);
		}
		object->notify_property_list_changed();
	}
	void update_state()
	{
		bool rs = condition->is_enable(object->blackboard_plan,is_include_condition);
		if(rs)
		{
			state_button->set_pressed(true);
			state_button->set_modulate(Color(0, 0.92549, 0.164706, 1));
		}else
		{
			state_button->set_pressed(false);
			state_button->set_modulate(Color(1, 0.255238, 0.196011, 1));
		}
		if(parent_section)
		{
			parent_section->update_state();
		}
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

	CheckButton* state_button = nullptr;
	Button* type_lable = nullptr;


	Button* property_name_bt = nullptr;
	PopupMenu* property_name_list = nullptr;


	Button* comparison_bt = nullptr;
	PopupMenu* comparison_list = nullptr;

	MarginContainer* value_parent = nullptr;
	
	Button* value_property_bt = nullptr;
	PopupMenu* value_property_list = nullptr;

	LineEdit* value_num = nullptr;
	LineEdit* value_string = nullptr;
	CheckButton* value_bool = nullptr;
	HSlider* value_range = nullptr;


	CheckBox* is_value_property = nullptr;
	Button* remove_bt = nullptr;

	Ref<AnimatorAIStateConditionBase> condition ;
	Ref<CharacterAnimatorCondition> object ;
	ConditionSection* parent_section = nullptr;
	bool is_include_condition = false;
};

class ConditionListButton_ED : public HBoxContainer
{
	GDCLASS(ConditionListButton_ED, HBoxContainer);
	static void _bind_methods()
	{

	}
public:
	ConditionListButton_ED(){}
	
	void setup(Ref<CharacterAnimatorCondition> p_object,bool is_include)
	{

		object = p_object;
		is_include_condition = is_include;
		set_layout_mode(LayoutMode::LAYOUT_MODE_CONTAINER);

		HSeparator *sep = memnew(HSeparator);
		sep->set_layout_mode(LayoutMode::LAYOUT_MODE_CONTAINER);
		sep->set_custom_minimum_size(Size2(20, 10));
		add_child(sep);

		add_int_bt = memnew(Button);
		add_int_bt->set_layout_mode(LayoutMode::LAYOUT_MODE_CONTAINER);
		add_int_bt->set_h_size_flags(SIZE_EXPAND_FILL);
		add_int_bt->set_text("Add Int");
		add_int_bt->connect("pressed", callable_mp(this, &ConditionListButton_ED::_on_add_int_bt_pressed));
		add_int_bt->set_modulate(Color(0.875804, 0.881502, 0.103496, 1));
		add_child(add_int_bt);


		add_float_bt = memnew(Button);
		add_float_bt->set_layout_mode(LayoutMode::LAYOUT_MODE_CONTAINER);
		add_float_bt->set_h_size_flags(SIZE_EXPAND_FILL);
		add_float_bt->set_text("Add Float");
		add_float_bt->connect("pressed", callable_mp(this, &ConditionListButton_ED::_on_add_float_bt_pressed));
		add_child(add_float_bt);


		add_string_bt = memnew(Button);
		add_string_bt->set_layout_mode(LayoutMode::LAYOUT_MODE_CONTAINER);
		add_string_bt->set_h_size_flags(SIZE_EXPAND_FILL);
		add_string_bt->set_text("Add String");
		add_string_bt->connect("pressed", callable_mp(this, &ConditionListButton_ED::_on_add_string_bt_pressed));
		add_string_bt->set_modulate(Color(0.875804, 0.881502, 0.103496, 1));
		add_child(add_string_bt);


		add_bool_bt = memnew(Button);
		add_bool_bt->set_layout_mode(LayoutMode::LAYOUT_MODE_CONTAINER);
		add_bool_bt->set_h_size_flags(SIZE_EXPAND_FILL);
		add_bool_bt->set_text("Add Bool");
		add_bool_bt->connect("pressed", callable_mp(this, &ConditionListButton_ED::_on_add_bool_bt_pressed));
		add_child(add_bool_bt);

		sep = memnew(HSeparator);
		sep->set_layout_mode(LayoutMode::LAYOUT_MODE_CONTAINER);
		sep->set_custom_minimum_size(Size2(20, 10));
		add_child(sep);
	}

	void _on_add_int_bt_pressed()
	{
		Ref<AnimatorAIStateConditionBase> condition = memnew(AnimatorAIStateIntCondition);
		if(is_include_condition)
		{
			object->add_include_condition(condition);
		}else
		{
			object->add_exclude_condition(condition);
		}
		object->notify_property_list_changed();
	}
	void _on_add_float_bt_pressed()
	{
		Ref<AnimatorAIStateConditionBase> condition = memnew(AnimatorAIStateFloatCondition);
		if(is_include_condition)
		{
			object->add_include_condition(condition);
		}else
		{
			object->add_exclude_condition(condition);
		}
		object->notify_property_list_changed();
	}

	void _on_add_string_bt_pressed()
	{
		Ref<AnimatorAIStateConditionBase> condition = memnew(AnimatorAIStateStringNameCondition);
		if(is_include_condition)
		{
			object->add_include_condition(condition);
		}else
		{
			object->add_exclude_condition(condition);
		}
		object->notify_property_list_changed();
	}

	void _on_add_bool_bt_pressed()
	{
		Ref<AnimatorAIStateConditionBase> condition = memnew(AnimatorAIStateBoolCondition);
		if(is_include_condition)
		{
			object->add_include_condition(condition);
		}else
		{
			object->add_exclude_condition(condition);
		}
		object->notify_property_list_changed();
	}


	Button* add_int_bt = nullptr;
	Button* add_float_bt = nullptr;
	Button* add_string_bt = nullptr;
	Button* add_bool_bt = nullptr;
	Ref<CharacterAnimatorCondition> object = nullptr;
	bool is_include_condition = false;
};

void ConditionSection::_on_header_pressed() {
	set_collapsed(!is_collapsed());
}

void ConditionSection::set_filter(String p_filter_text) {

}

void ConditionSection::add_condition(Control* p_task_button) {
	tasks_container->add_child(p_task_button);
}

void ConditionSection::set_collapsed(bool p_collapsed) {
	tasks_container->set_visible(!p_collapsed);
	if(is_include_condition)
	{
#ifdef TOOLS_ENABLED
		object->editor_set_section_unfold("Include Conditions", !p_collapsed);
#endif
	}else
	{
#ifdef TOOLS_ENABLED
		object->editor_set_section_unfold("Exclude Conditions", !p_collapsed);
#endif
	}
	section_header->set_icon(p_collapsed ? theme_cache.arrow_right_icon : theme_cache.arrow_down_icon);
}

bool ConditionSection::is_collapsed() const {
	return !tasks_container->is_visible();
}

void ConditionSection::_do_update_theme_item_cache() {
	theme_cache.arrow_down_icon = get_theme_icon(SNAME("GuiTreeArrowDown"), SNAME("EditorIcons"));
	theme_cache.arrow_right_icon = get_theme_icon(SNAME("GuiTreeArrowRight"), SNAME("EditorIcons"));
}

void ConditionSection::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			section_header->connect(SNAME("pressed"), callable_mp(this, &ConditionSection::_on_header_pressed));
		} break;
		case NOTIFICATION_THEME_CHANGED: {
			_do_update_theme_item_cache();
			section_header->set_icon(is_collapsed() ? theme_cache.arrow_right_icon : theme_cache.arrow_down_icon);
			section_header->add_theme_font_override(SNAME("font"), get_theme_font(SNAME("bold"), SNAME("EditorFonts")));
		} break;
	}
}
void ConditionSection::update_state() {
	bool rs = true;
	if(is_include_condition)
	{
		TypedArray<Ref<AnimatorAIStateConditionBase>>  conditions = object->get_include_condition();
		for(int32_t i = 0; i < conditions.size(); ++i)
		{
			Ref<AnimatorAIStateConditionBase> condition =conditions[i];
			if(condition.is_valid())
			{
				
                if (!condition->is_enable(object->blackboard_plan,is_include_condition))
                {
                    rs =  false;
					break;
                }
			}
		}

	}
	else
	{
		TypedArray<Ref<AnimatorAIStateConditionBase>>  conditions = object->get_exclude_condition();
		for(int32_t i = 0; i < conditions.size(); ++i)
		{
			Ref<AnimatorAIStateConditionBase> condition =conditions[i];
			if(condition.is_valid())
			{
				if (condition->is_enable(object->blackboard_plan,is_include_condition))
				{
					rs =  false;
					break;
				}
			}
		}

	}
	if(rs)
	{
		state_button->set_pressed(true);
		state_button->set_modulate(Color(0, 0.92549, 0.164706, 1));
	}else
	{
		state_button->set_pressed(false);
		state_button->set_modulate(Color(1, 0.255238, 0.196011, 1));
	}
}

void ConditionSection::_bind_methods() {
	ADD_SIGNAL(MethodInfo("task_button_pressed"));
	ADD_SIGNAL(MethodInfo("task_button_rmb"));
}

ConditionSection::ConditionSection() {

	HBoxContainer *hb = memnew(HBoxContainer);
	hb->set_layout_mode(LayoutMode::LAYOUT_MODE_CONTAINER);
	add_child(hb);


	section_header = memnew(Button);
	section_header->set_layout_mode(LayoutMode::LAYOUT_MODE_CONTAINER);
	section_header->set_h_size_flags(SIZE_EXPAND_FILL);
	section_header->set_v_size_flags(SIZE_EXPAND_FILL);
	hb->add_child(section_header);
	
	state_button = memnew(CheckButton);
	state_button->set_layout_mode(LayoutMode::LAYOUT_MODE_CONTAINER);
	state_button->set_disabled(true);
	state_button->set_focus_mode(FOCUS_NONE);
	hb->add_child(state_button);

	section_header->set_focus_mode(FOCUS_NONE);

	tasks_container = memnew(VBoxContainer);
	add_child(tasks_container);
}

ConditionSection::~ConditionSection() {
}


class BlackbordSetSection : public VBoxContainer {
	GDCLASS(BlackbordSetSection, VBoxContainer);

private:
	struct ThemeCache {
		Ref<Texture2D> arrow_down_icon;
		Ref<Texture2D> arrow_right_icon;
	} theme_cache;

	VBoxContainer *tasks_container;
	Button *section_header;

	Ref<AnimatorBlackboardSet> object;

	void _on_header_pressed();

protected:
	static void _bind_methods();

	void _notification(int p_what);

	virtual void _do_update_theme_item_cache();

public:
	void setup(Ref<AnimatorBlackboardSet> p_object)
	{

		object = p_object;
#ifdef TOOLS_ENABLED
			set_collapsed(!object->editor_is_section_unfolded("Change List"));
#endif
	}
	void set_filter(String p_filter);
	void add_condition(Control* p_task_button);

	void set_collapsed(bool p_collapsed);
	bool is_collapsed() const;

	String get_category_name() const { return section_header->get_text(); }
	void set_category_name(const String &p_cat) { section_header->set_text(p_cat); }

	BlackbordSetSection();
};

class BlackbordSetButtonList_ED : public HBoxContainer
{
	GDCLASS(BlackbordSetButtonList_ED, HBoxContainer);
	static void _bind_methods()
	{

	}
public:
	BlackbordSetButtonList_ED(){}
	
	void setup(Ref<AnimatorBlackboardSet> p_object)
	{

		object = p_object;
		set_layout_mode(LayoutMode::LAYOUT_MODE_CONTAINER);

		HSeparator *sep = memnew(HSeparator);
		sep->set_layout_mode(LayoutMode::LAYOUT_MODE_CONTAINER);
		sep->set_custom_minimum_size(Size2(20, 10));
		add_child(sep);

		add_int_bt = memnew(Button);
		add_int_bt->set_layout_mode(LayoutMode::LAYOUT_MODE_CONTAINER);
		add_int_bt->set_h_size_flags(SIZE_EXPAND_FILL);
		add_int_bt->set_text("Add Int");
		add_int_bt->connect("pressed", callable_mp(this, &BlackbordSetButtonList_ED::_on_add_int_bt_pressed));
		add_int_bt->set_modulate(Color(0.875804, 0.881502, 0.103496, 1));
		add_child(add_int_bt);


		add_float_bt = memnew(Button);
		add_float_bt->set_layout_mode(LayoutMode::LAYOUT_MODE_CONTAINER);
		add_float_bt->set_h_size_flags(SIZE_EXPAND_FILL);
		add_float_bt->set_text("Add Float");
		add_float_bt->connect("pressed", callable_mp(this, &BlackbordSetButtonList_ED::_on_add_float_bt_pressed));
		add_child(add_float_bt);


		add_string_bt = memnew(Button);
		add_string_bt->set_layout_mode(LayoutMode::LAYOUT_MODE_CONTAINER);
		add_string_bt->set_h_size_flags(SIZE_EXPAND_FILL);
		add_string_bt->set_text("Add String");
		add_string_bt->connect("pressed", callable_mp(this, &BlackbordSetButtonList_ED::_on_add_string_bt_pressed));
		add_string_bt->set_modulate(Color(0.875804, 0.881502, 0.103496, 1));
		add_child(add_string_bt);


		add_bool_bt = memnew(Button);
		add_bool_bt->set_layout_mode(LayoutMode::LAYOUT_MODE_CONTAINER);
		add_bool_bt->set_h_size_flags(SIZE_EXPAND_FILL);
		add_bool_bt->set_text("Add Bool");
		add_bool_bt->connect("pressed", callable_mp(this, &BlackbordSetButtonList_ED::_on_add_bool_bt_pressed));
		add_child(add_bool_bt);

		sep = memnew(HSeparator);
		sep->set_layout_mode(LayoutMode::LAYOUT_MODE_CONTAINER);
		sep->set_custom_minimum_size(Size2(20, 10));
		add_child(sep);
	}

	void _on_add_int_bt_pressed()
	{
		Ref<AnimatorBlackboardSetItemInt> condition = memnew(AnimatorBlackboardSetItemInt);
		object->add_item(condition);
		object->notify_property_list_changed();
	}
	void _on_add_float_bt_pressed()
	{
		Ref<AnimatorBlackboardSetItemFloat> condition = memnew(AnimatorBlackboardSetItemFloat);
		object->add_item(condition);
		object->notify_property_list_changed();
	}

	void _on_add_string_bt_pressed()
	{
		Ref<AnimatorBlackboardSetItemString> condition = memnew(AnimatorBlackboardSetItemString);
		object->add_item(condition);
		object->notify_property_list_changed();
	}

	void _on_add_bool_bt_pressed()
	{
		Ref<AnimatorBlackboardSetItemBool> condition = memnew(AnimatorBlackboardSetItemBool);
		object->add_item(condition);
		object->notify_property_list_changed();
	}


	Button* add_int_bt = nullptr;
	Button* add_float_bt = nullptr;
	Button* add_string_bt = nullptr;
	Button* add_bool_bt = nullptr;
	Ref<AnimatorBlackboardSet> object = nullptr;
};

