#pragma once
#include "modules/game_help/logic/character_ai/blackboard_set_item/animator_blackboard_item_bool.h"
#include "modules/game_help/logic/character_ai/blackboard_set_item/animator_blackboard_item_float.h"
#include "modules/game_help/logic/character_ai/blackboard_set_item/animator_blackboard_item_int.h"
#include "modules/game_help/logic/character_ai/blackboard_set_item/animator_blackboard_item_string.h"
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
		type_lable->set_button_icon(get_type_icon(Variant::get_type_name(Variant::INT)));
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
				type_lable->set_button_icon(get_type_icon(Variant::get_type_name(Variant::INT)));
				type_lable->set_text("Int");
			}
			else if(float_blackboard_set_item.is_valid())
			{
				type_lable->set_button_icon(get_type_icon(Variant::get_type_name(Variant::FLOAT)));
				type_lable->set_text("Float");
			}
			else if(bool_blackboard_set_item.is_valid())
			{
				type_lable->set_button_icon(get_type_icon(Variant::get_type_name(Variant::BOOL)));
				type_lable->set_text("Bool");
			}
			else if(string_blackboard_set_item.is_valid())
			{
				type_lable->set_button_icon(get_type_icon(Variant::get_type_name(Variant::STRING)));
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
	section_header->set_button_icon(p_collapsed ? theme_cache.arrow_right_icon : theme_cache.arrow_down_icon);
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
			section_header->set_button_icon(is_collapsed() ? theme_cache.arrow_right_icon : theme_cache.arrow_down_icon);
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
