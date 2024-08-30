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
    Object* get_object() { return object; }
protected:
    virtual void create_header(HBoxContainer *hb){}
    virtual void create_child_list(VBoxContainer *hb){}
	virtual void update_state(){}
    virtual String get_section_unfolded() const { return "Logic Condition Section"; }
public:
	LogicSectionBase();
	~LogicSectionBase();
public:
    Callable on_collapsed_change;
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
// 感应器节点
class CharacterInductorSection : public LogicSectionBase {
    GDCLASS(CharacterInductorSection, LogicSectionBase);
public:
    virtual void create_header(HBoxContainer *hb) override
    {

    }
    virtual void create_child_list(VBoxContainer *hb) override{}
	virtual void update_state() override{}
    virtual String get_section_unfolded() const override{ return "AI Logic Condition Section"; }
};


class CharacterLogicNodeSection : public LogicSectionBase {
    GDCLASS(CharacterLogicNodeSection, LogicSectionBase);
public:
    virtual void create_header(HBoxContainer *hb) override
    {
        state_name_edit = memnew(LineEdit);
        state_name_edit->set_custom_minimum_size(Size2(300, 0));
        state_name_edit->connect("text_changed", callable_mp(this, &CharacterLogicNodeSection::_state_name_changed));
        hb->add_child(state_name_edit);

    }
    virtual void create_child_list(VBoxContainer *vb) override {
    }
	virtual void update_state() override {
		node = Object::cast_to<CharacterAILogicNode>(object);
		if (node.is_valid())
		{
			state_name_edit->set_text(node->get_state_name());
		}
		else
		{
			state_name_edit->set_text(L" ");
		}
		set_category_name(node->get_state_name());
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
		set_category_name(p_name);
    }
    void _beehave_tree_property_changed(const Ref<Resource> &p_resource) {
        node->set_tree(p_resource);
        update_state();
    }
protected:
    LineEdit* state_name_edit = nullptr;

    Ref<CharacterAILogicNode> node;
};

// 逻辑节点列表
class CharacterLogicNodeListSection : public LogicSectionBase {
    GDCLASS(CharacterLogicNodeListSection, LogicSectionBase);
public:
    virtual void create_header(HBoxContainer *hb) override {


		add_node_button = memnew(Button);
		add_node_button->set_text(L" + ");
		add_node_button->connect(SceneStringName(pressed), callable_mp(this, &CharacterLogicNodeListSection::_on_add_node_pressed));
		hb->add_child(add_node_button);


		set_category_name(L"行为树节点列表");
	}
    virtual void create_child_list(VBoxContainer *hb) override {

        // 当前编辑的节点

        node_parent = memnew(HBoxContainer);
        node_parent->set_h_size_flags(SIZE_EXPAND_FILL);
        hb->add_child(node_parent);

        
        
        {
            HSeparator *sep = memnew(HSeparator);
            node_parent->add_child(sep);

			node_list = memnew(VBoxContainer);
			node_list->set_h_size_flags(SIZE_EXPAND_FILL);
            node_parent->add_child(node_list);           

        }
		HBoxContainer* property_parent = memnew(HBoxContainer);
		property_parent->set_h_size_flags(SIZE_EXPAND_FILL);
		hb->add_child(property_parent);

		{
			HSeparator* sep = memnew(HSeparator);
			property_parent->add_child(sep);

			VBoxContainer* pvb = memnew(VBoxContainer);
			pvb->set_h_size_flags(SIZE_EXPAND_FILL);
			property_parent->add_child(pvb);


			beehave_tree_property = memnew(EditorResourcePicker);
			beehave_tree_property->set_base_type("BeehaveTree");
            beehave_tree_property->set_tooltip_text(L"行为树资源");
			beehave_tree_property->connect("resource_changed", callable_mp(this, &CharacterLogicNodeListSection::_beehave_tree_property_changed));
			pvb->add_child(beehave_tree_property);


			sub_inspector = memnew(EditorInspector);
			sub_inspector->set_h_size_flags(SIZE_EXPAND_FILL);

			sub_inspector->set_vertical_scroll_mode(ScrollContainer::SCROLL_MODE_DISABLED);
			sub_inspector->set_use_doc_hints(true);

			sub_inspector->set_sub_inspector(true);
			sub_inspector->set_property_name_style(InspectorDock::get_singleton()->get_property_name_style());

			sub_inspector->connect("property_keyed", callable_mp(this, &CharacterLogicNodeListSection::_sub_inspector_property_keyed));
			sub_inspector->connect("resource_selected", callable_mp(this, &CharacterLogicNodeListSection::_sub_inspector_resource_selected));
			sub_inspector->connect("object_id_selected", callable_mp(this, &CharacterLogicNodeListSection::_sub_inspector_object_id_selected));
			sub_inspector->set_keying(true);
			sub_inspector->set_read_only(false);
			sub_inspector->set_use_folding(false);

			sub_inspector->set_mouse_filter(MOUSE_FILTER_STOP);

			pvb->add_child(sub_inspector);
		}


    }
	virtual void update_state() override {
        CharacterBodyMain* body_main = Object::cast_to<CharacterBodyMain>(object);
		ai_node = body_main->get_character_ai();
        create_node_list();
    }
    virtual String get_section_unfolded() const override{ return "AI Logic Node List Section"; }
protected:
    // 创建节点列表
    void create_node_list() {
        while(node_list->get_child_count() > 0)
        {
			node_list->get_child(0)->queue_free();
			node_list->remove_child(node_list->get_child(0));
        }

        TypedArray<Ref<CharacterAILogicNode>> nodes = ai_node->get_logic_node();

        for(int i = 0; i < nodes.size(); i++) {
            Ref<CharacterAILogicNode> node = nodes[i];
            CharacterLogicNodeSection* section = memnew(CharacterLogicNodeSection);
            section->init();
            section->setup(node.ptr());
            section->on_collapsed_change = callable_mp(this, &CharacterLogicNodeListSection::on_child_node_collapsed);
			node_list->add_child(section);
        }
        
    }
protected:
    void _on_add_node_pressed() {
        
    }
    void on_child_node_collapsed(LogicSectionBase* p_child,bool p_collapsed) {
        // TODO
		curr_node = Object::cast_to<CharacterAILogicNode>(p_child->get_object());
        if(p_collapsed)
        {
            sub_inspector->edit(nullptr);
            beehave_tree_property->set_edited_resource(Ref<RefCounted>());
        }
        else
        {
            sub_inspector->edit(curr_node->get_tree().ptr());
            beehave_tree_property->set_edited_resource(curr_node->get_tree());
			
        }
    }
	void _sub_inspector_property_keyed(const String& p_property, const Variant& p_value, bool p_advance) {
		// The second parameter could be null, causing the event to fire with less arguments, so use the pointer call which preserves it.
		const Variant args[3] = { String("children") + ":" + p_property, p_value, p_advance };
		const Variant* argp[3] = { &args[0], &args[1], &args[2] };
		emit_signalp(SNAME("property_keyed_with_value"), argp, 3);
	}

	void _sub_inspector_resource_selected(const Ref<RefCounted>& p_resource, const String& p_property) {
		emit_signal(SNAME("resource_selected"), String("children") + ":" + p_property, p_resource);
	}

	void _sub_inspector_object_id_selected(int p_id) {
		emit_signal(SNAME("object_id_selected"), "children", p_id);
	}
	void _beehave_tree_property_changed(const Ref<Resource>& p_resource) {
		curr_node->set_tree(p_resource);
		update_state();
	}
protected:
    HBoxContainer* node_parent = nullptr;
	VBoxContainer* node_list = nullptr;
    EditorResourcePicker* beehave_tree_property = nullptr;
    EditorInspector* sub_inspector = nullptr;

    Button* add_node_button = nullptr;

    Ref<CharacterAI> ai_node;
	Ref<CharacterAILogicNode> curr_node;
};

// 大脑节点
class CharacterBrainSection : public LogicSectionBase {
    GDCLASS(CharacterBrainSection, LogicSectionBase);
public:
    virtual void create_header(HBoxContainer *hb) override {
        beehave_tree_property = memnew(EditorResourcePicker);
		beehave_tree_property->set_custom_minimum_size(Vector2(200, 0));
		beehave_tree_property->set_base_type("CharacterAI_Brain");
        beehave_tree_property->connect("resource_changed", callable_mp(this, &CharacterBrainSection::on_brain_changed));
        hb->add_child(beehave_tree_property);    

		set_category_name(L"大脑节点");
	}
    virtual void create_child_list(VBoxContainer *hb) override {

        
        // 当前编辑的节点

        
        

		sub_inspector = memnew(EditorInspector);
		
		sub_inspector->set_vertical_scroll_mode(ScrollContainer::SCROLL_MODE_DISABLED);
		sub_inspector->set_use_doc_hints(true);

		sub_inspector->set_sub_inspector(true);
		sub_inspector->set_property_name_style(InspectorDock::get_singleton()->get_property_name_style());

		sub_inspector->connect("property_keyed", callable_mp(this, &CharacterBrainSection::_sub_inspector_property_keyed));
		sub_inspector->connect("resource_selected", callable_mp(this, &CharacterBrainSection::_sub_inspector_resource_selected));
		sub_inspector->connect("object_id_selected", callable_mp(this, &CharacterBrainSection::_sub_inspector_object_id_selected));
		sub_inspector->set_keying(true);
		sub_inspector->set_read_only(false);
		sub_inspector->set_use_folding(true);

		sub_inspector->set_mouse_filter(MOUSE_FILTER_STOP);
		sub_inspector->set_modulate(BeehaveGraphEditor::get_select_color());

        hb->add_child(sub_inspector);

    }
	virtual void update_state() override {
	}
    virtual String get_section_unfolded() const override{ return "AI Logic Condition Section"; }
	void on_brain_changed(const Ref<Resource>& p_resource)
	{

	}
protected:
	void _sub_inspector_property_keyed(const String& p_property, const Variant& p_value, bool p_advance) {
		// The second parameter could be null, causing the event to fire with less arguments, so use the pointer call which preserves it.
		const Variant args[3] = { String("children") + ":" + p_property, p_value, p_advance };
		const Variant* argp[3] = { &args[0], &args[1], &args[2] };
		emit_signalp(SNAME("property_keyed_with_value"), argp, 3);
	}

	void _sub_inspector_resource_selected(const Ref<RefCounted>& p_resource, const String& p_property) {
		emit_signal(SNAME("resource_selected"), String("children") + ":" + p_property, p_resource);
	}

	void _sub_inspector_object_id_selected(int p_id) {
		emit_signal(SNAME("object_id_selected"), "children", p_id);
	}
protected:
    EditorResourcePicker* beehave_tree_property = nullptr;
    EditorInspector* sub_inspector = nullptr;
};

// 感应器检查节点
class CharacterInductorCheckSection : public LogicSectionBase {
    GDCLASS(CharacterInductorCheckSection, LogicSectionBase);
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
        HBoxContainer *hbc = memnew(HBoxContainer);
        hb->add_child(hbc);
        

		HSeparator *sep = memnew(HSeparator);
        hbc->add_child(sep);

        VBoxContainer *vbc = memnew(VBoxContainer);
        vbc->set_h_size_flags(Control::SIZE_EXPAND_FILL);
        hbc->add_child(vbc);

		inductor_list_section = memnew(CharacterInductorListSection);
        inductor_list_section->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		inductor_list_section->init();
        vbc->add_child(inductor_list_section);

        brain_section = memnew(CharacterBrainSection);
        brain_section->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		brain_section->init();
        vbc->add_child(brain_section);

        logic_node_list_section = memnew(CharacterLogicNodeListSection);
        logic_node_list_section->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		logic_node_list_section->init();
        vbc->add_child(logic_node_list_section);


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
    CharacterBrainSection *brain_section = nullptr;
    CharacterLogicNodeListSection* logic_node_list_section = nullptr;
};

