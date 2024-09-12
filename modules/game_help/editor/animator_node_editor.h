#include "body_main_editor.h"
#include "scene/3d/light_3d.h"
#include "scene/main/viewport.h"
#include "editor/editor_resource_picker.h"
#include "editor/themes/editor_scale.h"
#include "animator_node_preview.h"

class AnimatorNodeItemEditor :public HBoxContainer {
    GDCLASS(AnimatorNodeItemEditor, HBoxContainer);
public:
    AnimatorNodeItemEditor()
    {

    }
    void init(class AnimationNodeSectionBase* p_node_editor,int p_index,bool p_is_show_x, bool p_is_show_y) {
        node_editor = p_node_editor;
        index = p_index;
        is_show_x = p_is_show_x;
        is_show_y = p_is_show_y;


		EditorBottomPanel* p_control = EditorNode::get_bottom_panel();

        move_up_button = memnew(Button);
        move_up_button->set_icon(p_control->get_theme_icon(SNAME("MoveUp"), SNAME("EditorIcons")));
        move_up_button->connect(SceneStringName(pressed), callable_mp(this, &AnimatorNodeItemEditor::_on_move_up_button_pressed));
        add_child(move_up_button);

        move_down_button = memnew(Button);
        move_down_button->set_icon(p_control->get_theme_icon(SNAME("MoveDown"), SNAME("EditorIcons")));
        move_down_button->connect(SceneStringName(pressed), callable_mp(this, &AnimatorNodeItemEditor::_on_move_down_button_pressed));
        add_child(move_down_button);

        is_cilp = memnew(CheckBox);
        is_cilp->set_text(L"动画");
        is_cilp->connect(SceneStringName(toggled), callable_mp(this, &AnimatorNodeItemEditor::_on_is_cilp_toggled));
        is_cilp->set_modulate(Color(0, 0.92549, 0.964706, 1));
        add_child(is_cilp);

        select_animation_picker = memnew(EditorResourcePicker);
        select_animation_picker->set_h_size_flags(SIZE_EXPAND_FILL);
        select_animation_picker->connect("resource_changed", callable_mp(this, &AnimatorNodeItemEditor::_on_animator_item_changed));
		select_animation_picker->set_modulate(Color(1,0.7,0.7));
        select_animation_picker->set_base_type("Animation");
        add_child(select_animation_picker);

        select_animator_node_picker = memnew(EditorResourcePicker);
        select_animator_node_picker->set_h_size_flags(SIZE_EXPAND_FILL);
        select_animator_node_picker->connect("resource_changed", callable_mp(this, &AnimatorNodeItemEditor::_on_animator_node_changed));
		select_animator_node_picker->set_modulate(Color(0.7, 1, 0.7));
	    select_animator_node_picker->set_base_type("CharacterAnimatorNodeBase");
        add_child(select_animator_node_picker);

		int value_size = 100;
		if (is_show_y_input())
		{
			value_size = 50;
		}

        x_velue_editor = memnew(LineEdit);
		x_velue_editor->set_custom_minimum_size(Size2(value_size, 0));
        x_velue_editor->set_visible(is_show_x_input());
        x_velue_editor->connect(SceneStringName(text_changed), callable_mp(this, &AnimatorNodeItemEditor::_on_x_input_changed));
        add_child(x_velue_editor);

        y_velue_editor = memnew(LineEdit);
		y_velue_editor->set_custom_minimum_size(Size2(value_size, 0));
        y_velue_editor->set_visible(is_show_y_input());
        y_velue_editor->connect(SceneStringName(text_changed), callable_mp(this, &AnimatorNodeItemEditor::_on_y_input_changed));
        add_child(y_velue_editor);

        delete_button = memnew(Button);
        delete_button->set_icon(p_control->get_theme_icon(SNAME("Remove"), SNAME("EditorIcons")));
        delete_button->connect(SceneStringName(pressed), callable_mp(this, &AnimatorNodeItemEditor::_on_delete_button_pressed));
        add_child(delete_button);

        update_item_state();
    }
public:
    virtual bool is_show_x_input() { return is_show_x; }
    virtual bool is_show_y_input() { return is_show_y; }
    float get_x_input_value() {
        return x_velue_editor->get_text().to_float();
    }
    float get_y_input_value() {
        return y_velue_editor->get_text().to_float();
    }
    void set_x_input_value(float value) {
        x_velue_editor->set_text(String::num(value));
    }
    void set_y_input_value(float value) {
        y_velue_editor->set_text(String::num(value));
    }
    void update_item_state();
protected:
    void _on_is_cilp_toggled(bool p_pressed);
    void _on_animator_item_changed(const Ref<Resource>& p_resource);
    void _on_animator_node_changed(const Ref<Resource>& p_resource);
    void _on_move_up_button_pressed();
    void _on_move_down_button_pressed();
    void _on_delete_button_pressed();
    void _on_x_input_changed(const String &p_text);
    void _on_y_input_changed(const String &p_text);
protected:
    Button* move_up_button = nullptr;
    Button* move_down_button = nullptr;
    LineEdit* x_velue_editor = nullptr;
    LineEdit* y_velue_editor = nullptr;
    CheckBox* is_cilp = nullptr;
    EditorResourcePicker* select_animation_picker = nullptr;
    EditorResourcePicker* select_animator_node_picker = nullptr;
    Button* delete_button = nullptr;

    class AnimationNodeSectionBase* node_editor = nullptr;
    int index = 0;
    bool is_show_x = false;
    bool is_show_y = false;
};

class AnimationNodeSectionBase : public LogicSectionBase {
    GDCLASS(AnimationNodeSectionBase, LogicSectionBase);
public:
    virtual void create_header(HBoxContainer *hb) override
    {
		set_category_name(L"动画节点列表");
        add_button = memnew(Button);
        add_button->set_text(L" + ");
        add_button->connect(SceneStringName(pressed), callable_mp(this, &AnimationNodeSectionBase::_on_add_node_pressed));
        add_button->set_modulate(Color(0.92549,0, 0.964706, 1));
        hb->add_child(add_button);

    }
    virtual void create_child_list(VBoxContainer *hb) override {

    }
	virtual void update_state() override {


        node = Object::cast_to<CharacterAnimatorNodeBase>(object);
        item_parent = memnew(VBoxContainer);
        item_parent->set_h_size_flags(SIZE_EXPAND_FILL);
        tasks_container->add_child(item_parent);

        
		sub_inspector = memnew(EditorInspector);
		
		sub_inspector->set_vertical_scroll_mode(ScrollContainer::SCROLL_MODE_DISABLED);
		sub_inspector->set_use_doc_hints(true);

		sub_inspector->set_sub_inspector(true);
		sub_inspector->set_property_name_style(InspectorDock::get_singleton()->get_property_name_style());

		sub_inspector->connect("property_keyed", callable_mp(this, &AnimationNodeSectionBase::_sub_inspector_property_keyed));
		sub_inspector->connect("resource_selected", callable_mp(this, &AnimationNodeSectionBase::_sub_inspector_resource_selected));
		sub_inspector->connect("object_id_selected", callable_mp(this, &AnimationNodeSectionBase::_sub_inspector_object_id_selected));
		sub_inspector->set_keying(true);
		sub_inspector->set_read_only(false);
		sub_inspector->set_use_folding(true);

		sub_inspector->set_mouse_filter(MOUSE_FILTER_STOP);
		sub_inspector->set_modulate(BeehaveGraphEditor::get_select_color());
		tasks_container->add_child(sub_inspector);


        // 创建预览窗口
        preview_resource_hb = memnew(HBoxContainer);
        preview_resource_hb->set_h_size_flags(SIZE_EXPAND_FILL);
        tasks_container->add_child(preview_resource_hb);
        {
            prefab_lable = memnew(Label);
            preview_resource_hb->add_child(prefab_lable);
            prefab_lable->set_text(L"预览*预制体:");


            select_prefab_picker = memnew(EditorResourcePicker);
            preview_resource_hb->add_child(select_prefab_picker);
            select_prefab_picker->set_base_type("CharacterBodyPrefab");
            select_prefab_picker->set_h_size_flags(SIZE_EXPAND_FILL);
            select_prefab_picker->connect("resource_changed", callable_mp(this, &AnimationNodeSectionBase::_on_prefab_picker_changed));

            blackbord_lable = memnew(Label);
            preview_resource_hb->add_child(blackbord_lable);
            blackbord_lable->set_text(L"预览*黑板:");


			select_blackbaord_picker = memnew(EditorResourcePicker);
            preview_resource_hb->add_child(select_blackbaord_picker);
			select_blackbaord_picker->set_base_type("BlackboardPlan");
			select_blackbaord_picker->set_h_size_flags(SIZE_EXPAND_FILL);
			select_blackbaord_picker->connect("resource_changed", callable_mp(this, &AnimationNodeSectionBase::_on_blackbord_picker_changed));

        }

        preview = memnew(AnimationNodePreview);
        preview->set_custom_minimum_size(Size2(1, 300) * EDSCALE);
        tasks_container->add_child(preview);

        update_child_item();
    }
    void update_child_item() {
        while(item_parent->get_child_count()) {
			Node* child = item_parent->get_child(0);
            item_parent->remove_child(child);
			child->queue_free();
        }
        item_container.clear();
        TypedArray items = node->get_animation_arrays();
        for(int i=0; i < items.size(); i++) {
            AnimatorNodeItemEditor* item = memnew(AnimatorNodeItemEditor);
            item->init(this,i, is_show_x_input(), is_show_y_input());
            init_node_item(i, item);
            item->set_h_size_flags(SIZE_EXPAND_FILL);
            item_parent->add_child(item);
            item_container.push_back(item);
        }
        preview->stop();
        preview->set_animator_node(node);
        if(CharacterBodyMain::get_current_editor_player() == nullptr) {
            select_prefab_picker->set_edited_resource(AnimationNodePreview::get_globle_preview_prefab());
            select_blackbaord_picker->set_edited_resource(AnimationNodePreview::get_globle_preview_blackboard());
            sub_inspector->edit(AnimationNodePreview::get_globle_preview_blackboard().ptr());
        }
        else {
            preview_resource_hb->set_visible(false);
            sub_inspector->set_visible(false);
        }
    }

    virtual String get_section_unfolded() const override{ return "Animation Item Condition Section"; }
public:
    void stop_animator() {
        preview->stop();
        
    }
    virtual void init_node_item(int p_index,AnimatorNodeItemEditor* item) {

    }
    virtual void update_node_item(int p_index,AnimatorNodeItemEditor* item) {
        
    }
    virtual bool is_show_x_input() { return false; }
    virtual bool is_show_y_input() { return false; }
    void _on_prefab_picker_changed(Ref<Resource> p_resource) {
        if(p_resource.is_valid()) {
			Ref< CharacterBodyPrefab> prefab = Object::cast_to< CharacterBodyPrefab>(p_resource.ptr());
			preview->set_globle_preview_prefab(prefab);
        }
    }
    void _on_blackbord_picker_changed(Ref<Resource> p_resource) {
        if(p_resource.is_valid()) {
			Ref<BlackboardPlan> blackboard = Object::cast_to< BlackboardPlan>(p_resource.ptr());
			preview->set_globle_preview_blackboard(blackboard);
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

public:
    void item_node_move_up(int p_index) {
        node->move_up_item(p_index);
        update_child_item();
    }
    void item_node_move_down(int p_index) {
        
        node->move_down_item(p_index);
        update_child_item();
    }
    void item_set_animation(int p_index,Ref<Animation> p_animation) {
        node->set_item_animation(p_index, p_animation);        
        update_child_item();
    }
    void item_set_animator_node(int p_index,Ref<CharacterAnimatorNodeBase> p_animator_node) {
        node->set_item_animator_node(p_index, p_animator_node);
        update_child_item();
    }
    void item_remove(int p_index) {
        node->remove_item(p_index);
        update_child_item();        
    }
    
    void _on_add_node_pressed() {
        node->add_item();
		update_child_item();
    }
    void _on_move_node_up_pressed(int p_index) {
        node->remove_item(p_index);
        update_child_item();
    }
public:
	CheckBox* is_cilp = nullptr;

	EditorInspector* sub_inspector = nullptr;
	HBoxContainer* preview_resource_hb = nullptr;
    
	Label* prefab_lable = nullptr;
	EditorResourcePicker* select_prefab_picker = nullptr;
	Label* blackbord_lable = nullptr;
	EditorResourcePicker* select_blackbaord_picker = nullptr;


    Button* add_button = nullptr;
    VBoxContainer* item_parent = nullptr;
    List<AnimatorNodeItemEditor*> item_container;
	EditorResourcePicker* select_animation_picker = nullptr;
    AnimationNodePreview* preview = nullptr;
    Ref<CharacterAnimatorNodeBase> node;
};

class AnimationNode2D : public AnimationNodeSectionBase {
    GDCLASS(AnimationNode2D, AnimationNodeSectionBase);
public:
    virtual void init_node_item(int p_index,AnimatorNodeItemEditor* item) {
        Ref<CharacterAnimatorNode2D> node_2d = Object::cast_to<CharacterAnimatorNode2D>(node.ptr());
        if(node_2d.is_valid()) {
            item->set_x_input_value(node_2d->get_position_x(p_index));
            item->set_y_input_value(node_2d->get_position_y(p_index));
        }

    }
    virtual void update_node_item(int p_index,AnimatorNodeItemEditor* item) {
        Ref<CharacterAnimatorNode2D> node_2d = Object::cast_to<CharacterAnimatorNode2D>(node.ptr());
        if(node_2d.is_valid()) {
            node_2d->set_position_x(p_index, item->get_x_input_value());
            node_2d->set_position_y(p_index, item->get_y_input_value());
        }
        
    }
    virtual bool is_show_x_input() { return true; }
    virtual bool is_show_y_input() { return true; }
};

class AnimationNode1D : public AnimationNodeSectionBase {
    GDCLASS(AnimationNode1D, AnimationNodeSectionBase);
public:
    virtual void init_node_item(int p_index,AnimatorNodeItemEditor* item) {
        Ref<CharacterAnimatorNode1D> node_1d = Object::cast_to<CharacterAnimatorNode1D>(node.ptr());
        if(node_1d.is_valid()) {
            item->set_x_input_value(node_1d->get_position(p_index));
        }
    }
    virtual void update_node_item(int p_index,AnimatorNodeItemEditor* item) {
        Ref<CharacterAnimatorNode1D> node_1d = Object::cast_to<CharacterAnimatorNode1D>(node.ptr());
        if(node_1d.is_valid()) {
            node_1d->set_position(p_index, item->get_x_input_value());
        }
        
    }
    virtual bool is_show_x_input() { return true; }
};



void AnimatorNodeItemEditor::update_item_state() {
    bool is_anim_clip = node_editor->node->get_animation_item(index)->get_is_clip();
	is_cilp->set_pressed(is_anim_clip);
    select_animation_picker->set_visible(is_anim_clip);
    select_animator_node_picker->set_visible(!is_anim_clip);
    
    select_animation_picker->set_edited_resource(node_editor->node->get_animation_item(index)->get_animation());    
    select_animator_node_picker->set_edited_resource(node_editor->node->get_animation_item(index)->get_child_node());
    node_editor->stop_animator();

}

void AnimatorNodeItemEditor::_on_is_cilp_toggled(bool p_pressed)
{
    node_editor->node->get_animation_item(index)->set_is_clip(p_pressed);
    update_item_state();
}
void AnimatorNodeItemEditor::_on_animator_item_changed(const Ref<Resource>& p_resource) {
    node_editor->item_set_animation(index, p_resource);
}
void AnimatorNodeItemEditor::_on_animator_node_changed(const Ref<Resource>& p_resource) {
    node_editor->item_set_animator_node(index, p_resource);
}
void AnimatorNodeItemEditor::_on_move_up_button_pressed() {
    node_editor->item_node_move_up(index);
}
void AnimatorNodeItemEditor::_on_move_down_button_pressed() {
    node_editor->item_node_move_down(index);
    
}
void AnimatorNodeItemEditor::_on_delete_button_pressed() {
    node_editor->item_remove(index);
}

void AnimatorNodeItemEditor::_on_x_input_changed(const String &p_text)
{
    node_editor->update_node_item(index, this);
}
void AnimatorNodeItemEditor::_on_y_input_changed(const String &p_text)
{
    node_editor->update_node_item(index, this);    
}



