#pragma once
#include "scene/gui/box_container.h"
#include "scene/gui/separator.h"
#include "scene/gui/check_button.h"
#include "editor/editor_node.h"
#include "editor/editor_properties.h"
#include "modules/game_help/logic/body_main.h"
#include "modules/game_help/logic/character_ai/character_ai.h"







class LogicSectionBase : public VBoxContainer {
	GDCLASS(LogicSectionBase, VBoxContainer);

private:
	struct ThemeCache {
		Ref<Texture2D> arrow_down_icon;
		Ref<Texture2D> arrow_right_icon;
	} theme_cache;
protected:
	VBoxContainer *tasks_container;
	Button *section_header;

	Object* object;

	void _on_header_pressed();

protected:
	static void _bind_methods();

	void _notification(int p_what);

	virtual void _do_update_theme_item_cache();

public:
    void init();
	void setup(Object* p_object)
	{
		object = p_object;
		set_collapsed(!object->editor_is_section_unfolded(get_section_unfolded()));
        update_state();
	}
	void set_filter(String p_filter);
	void add_condition(Control* p_task_button);

	void set_collapsed(bool p_collapsed);
	bool is_collapsed() const;

	String get_category_name() const { return section_header->get_text(); }
	void set_category_name(const String &p_cat) { section_header->set_text(p_cat); }
protected:
    virtual void create_header(HBoxContainer *hb){}
    virtual void create_child_list(VBoxContainer *hb){}
	virtual void update_state(){}
    virtual String get_section_unfolded() const { return "Logic Condition Section"; }
public:
	LogicSectionBase();
	~LogicSectionBase();
};



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
	section_header->set_icon(p_collapsed ? theme_cache.arrow_right_icon : theme_cache.arrow_down_icon);
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
			section_header->set_icon(is_collapsed() ? theme_cache.arrow_right_icon : theme_cache.arrow_down_icon);
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
// 感应器节点
class CharacterInductorConditionSection : public LogicSectionBase {
    GDCLASS(CharacterInductorConditionSection, LogicSectionBase);
public:
    virtual void create_header(HBoxContainer *hb) override
    {

    }
    virtual void create_child_list(VBoxContainer *hb) override{}
	virtual void update_state() override{}
    virtual String get_section_unfolded() const override{ return "AI Logic Condition Section"; }
};


class CharacterLogicNodeConditionSection : public LogicSectionBase {
    GDCLASS(CharacterLogicNodeConditionSection, LogicSectionBase);
public:
    virtual void create_header(HBoxContainer *hb) override
    {
        state_name_edit = memnew(LineEdit);
        state_name_edit->set_custom_minimum_size(Size2(200, 0));
        state_name_edit->connect("text_changed", callable_mp(this, &CharacterLogicNodeConditionSection::_state_name_changed));
        hb->add_child(state_name_edit);

    }
    virtual void create_child_list(VBoxContainer *vb) override {
        
        beehave_tree_property = memnew(EditorResourcePicker);
        beehave_tree_property->connect("resource_changed", callable_mp(this, &CharacterLogicNodeConditionSection::_beehave_tree_property_changed));
        vb->add_child(beehave_tree_property);    

        

		sub_inspector = memnew(EditorInspector);
		
		sub_inspector->set_vertical_scroll_mode(ScrollContainer::SCROLL_MODE_DISABLED);
		sub_inspector->set_use_doc_hints(true);

		sub_inspector->set_sub_inspector(true);
		sub_inspector->set_property_name_style(InspectorDock::get_singleton()->get_property_name_style());

		sub_inspector->connect("property_keyed", callable_mp(this, &CharacterLogicNodeConditionSection::_sub_inspector_property_keyed));
		sub_inspector->connect("resource_selected", callable_mp(this, &CharacterLogicNodeConditionSection::_sub_inspector_resource_selected));
		sub_inspector->connect("object_id_selected", callable_mp(this, &CharacterLogicNodeConditionSection::_sub_inspector_object_id_selected));
		sub_inspector->set_keying(true);
		sub_inspector->set_read_only(false);
		sub_inspector->set_use_folding(true);

		sub_inspector->set_mouse_filter(MOUSE_FILTER_STOP);
		sub_inspector->set_modulate(BeehaveGraphEditor::get_select_color());

        vb->add_child(sub_inspector);

    }
	virtual void update_state() override {
        state_name_edit->set_text(node->get_state_name());
        node = Object::cast_to<CharacterAILogicNode>(object);
    }
    virtual String get_section_unfolded() const override{ return "AI Logic Condition Section"; }
protected:
	void _sub_inspector_property_keyed(const String &p_property, const Variant &p_value, bool p_advance) {
		// The second parameter could be null, causing the event to fire with less arguments, so use the pointer call which preserves it.
		const Variant args[3] = { String("children") + ":" + p_property, p_value, p_advance };
		const Variant *argp[3] = { &args[0], &args[1], &args[2] };
		emit_signalp(SNAME("property_keyed_with_value"), argp, 3);
	}

	void _sub_inspector_resource_selected(const Ref<RefCounted> &p_resource, const String &p_property) {
		emit_signal(SNAME("resource_selected"), String("children") + ":" + p_property, p_resource);
	}

	void _sub_inspector_object_id_selected(int p_id) {
		emit_signal(SNAME("object_id_selected"), "children", p_id);
	}

    void _state_name_changed(const String &p_name) {
        node->set_state_name(p_name);
    }
    void _beehave_tree_property_changed(const Ref<Resource> &p_resource) {
        node->set_tree(p_resource);
        update_state();
    }
protected:
    EditorResourcePicker* beehave_tree_property = nullptr;
    EditorInspector* sub_inspector = nullptr;
    LineEdit* state_name_edit = nullptr;

    Ref<CharacterAILogicNode> node;
};

// 逻辑节点列表
class CharacterLogicNodeListConditionSection : public LogicSectionBase {
    GDCLASS(CharacterLogicNodeListConditionSection, LogicSectionBase);
public:
    virtual void create_header(HBoxContainer *hb) override {
		set_category_name(L"行为树节点列表");
	}
    virtual void create_child_list(VBoxContainer *hb) override{}
	virtual void update_state() override{}
    virtual String get_section_unfolded() const override{ return "AI Logic Condition Section"; }
};

// 大脑节点
class CharacterBrainConditionSection : public LogicSectionBase {
    GDCLASS(CharacterBrainConditionSection, LogicSectionBase);
public:
    virtual void create_header(HBoxContainer *hb) override {
		set_category_name(L"大脑节点");
	}
    virtual void create_child_list(VBoxContainer *hb) override{}
	virtual void update_state() override{}
    virtual String get_section_unfolded() const override{ return "AI Logic Condition Section"; }
protected:
    EditorResourcePicker* beehave_tree_property = nullptr;
    EditorInspector* sub_inspector = nullptr;
};

// 感应器检查节点
class CharacterInductorCheckConditionSection : public LogicSectionBase {
    GDCLASS(CharacterInductorCheckConditionSection, LogicSectionBase);
public:
    virtual void create_header(HBoxContainer *hb) override{}
    virtual void create_child_list(VBoxContainer *hb) override{}
	virtual void update_state() override{}
    virtual String get_section_unfolded() const override{ return "AI Logic Condition Section"; }

protected:
    Button* upmove_button = nullptr;
    Button* downmove_button = nullptr;
    Button* delete_button = nullptr;

    
    EditorResourcePicker* beehave_tree_property = nullptr;
    EditorInspector* sub_inspector = nullptr;
};

// 感应器节点列表
class CharacterInductorListSection : public LogicSectionBase {
    GDCLASS(CharacterInductorListSection, LogicSectionBase);
public:
    virtual void create_header(HBoxContainer *hb) override {

		set_category_name(L"感应器");
	}
    virtual void create_child_list(VBoxContainer *hb) override{}
	virtual void update_state() override{}
    virtual String get_section_unfolded() const override{ return "AI Logic Condition Section"; }
};


class CharacterAISection : public LogicSectionBase {
    GDCLASS(CharacterAISection, LogicSectionBase);
public:
    virtual void create_header(HBoxContainer *hb) override {
        set_category_name(L"人物智能");
    }
    virtual void create_child_list(VBoxContainer *hb) override {
		inductor_list_section = memnew(CharacterInductorListSection);
		inductor_list_section->init();
        hb->add_child(inductor_list_section);

        brain_section = memnew(CharacterBrainConditionSection);
		brain_section->init();
        hb->add_child(brain_section);

        logic_node_list_section = memnew(CharacterLogicNodeListConditionSection);
		logic_node_list_section->init();
        hb->add_child(logic_node_list_section);


    }
	virtual void update_state() override {
		inductor_list_section->setup(object);
		brain_section->setup(object);
		logic_node_list_section->setup(object);
	}
    virtual String get_section_unfolded() const override{ return "AI Logic Condition Section"; }

protected:
    void _add_node_pressed() {
    }

protected:
    CharacterInductorListSection* inductor_list_section = nullptr;
    CharacterBrainConditionSection *brain_section = nullptr;
    CharacterLogicNodeListConditionSection* logic_node_list_section = nullptr;
};

