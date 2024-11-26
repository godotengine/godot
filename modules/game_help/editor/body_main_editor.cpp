#include "body_main_editor.h"


void LogicSectionBase::_on_header_pressed() {
	set_collapsed(!is_collapsed());
}

void LogicSectionBase::set_filter(String p_filter_text) {

}

void LogicSectionBase::add_condition(Control* p_task_button) {
	tasks_container->add_child(p_task_button);
}

void LogicSectionBase::set_collapsed(bool p_collapsed) {
	object->editor_set_section_unfold(get_section_unfolded(), !p_collapsed);
	section_header->set_button_icon(p_collapsed ? theme_cache.arrow_right_icon : theme_cache.arrow_down_icon);
    tasks_container->set_visible(!p_collapsed);
    
    on_collapsed_change.call(this, p_collapsed);
}

bool LogicSectionBase::is_collapsed() const {
	return !object->editor_is_section_unfolded(get_section_unfolded());
}

void LogicSectionBase::_do_update_theme_item_cache() {
	theme_cache.arrow_down_icon = get_theme_icon(SNAME("GuiTreeArrowDown"), SNAME("EditorIcons"));
	theme_cache.arrow_right_icon = get_theme_icon(SNAME("GuiTreeArrowRight"), SNAME("EditorIcons"));
}

void LogicSectionBase::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			section_header->connect(SNAME("pressed"), callable_mp(this, &LogicSectionBase::_on_header_pressed));
		} break;
		case NOTIFICATION_THEME_CHANGED: {
			_do_update_theme_item_cache();
			section_header->set_button_icon(is_collapsed() ? theme_cache.arrow_right_icon : theme_cache.arrow_down_icon);
			section_header->add_theme_font_override(SNAME("font"), get_theme_font(SNAME("bold"), SNAME("EditorFonts")));
		} break;
	}
}
void LogicSectionBase::init() {

	HBoxContainer *hb = memnew(HBoxContainer);
	hb->set_layout_mode(LayoutMode::LAYOUT_MODE_CONTAINER);
	add_child(hb);


	section_header = memnew(Button);
	section_header->set_layout_mode(LayoutMode::LAYOUT_MODE_CONTAINER);
	section_header->set_h_size_flags(SIZE_EXPAND_FILL);
	section_header->set_v_size_flags(SIZE_EXPAND_FILL);
    section_header->set_text_alignment(HORIZONTAL_ALIGNMENT_LEFT);
	hb->add_child(section_header);
	

	section_header->set_focus_mode(FOCUS_NONE);

    create_header(hb);

	tasks_container = memnew(VBoxContainer);
	add_child(tasks_container);
    create_child_list(tasks_container);
}

void LogicSectionBase::_bind_methods() {
	ADD_SIGNAL(MethodInfo("task_button_pressed"));
	ADD_SIGNAL(MethodInfo("task_button_rmb"));
}

LogicSectionBase::LogicSectionBase() {

}

LogicSectionBase::~LogicSectionBase() {
}