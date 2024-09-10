#include "body_main_editor.h"
#include "scene/3d/light_3d.h"
#include "scene/main/viewport.h"
#include "editor/editor_resource_picker.h"
#include "editor/themes/editor_scale.h"

class AnimationNodePreview : public SubViewportContainer
{
	float rot_x;
	float rot_y;

	SubViewport *viewport = nullptr;
	CharacterBodyMain *preview_character = nullptr;
	Node3D *rotation = nullptr;
	DirectionalLight3D *light1 = nullptr;
	DirectionalLight3D *light2 = nullptr;
	Camera3D *camera = nullptr;
	Ref<CameraAttributesPractical> camera_attributes;

	Button *light_1_switch = nullptr;
	Button *light_2_switch = nullptr;

	struct ThemeCache {
		Ref<Texture2D> light_1_icon;
		Ref<Texture2D> light_2_icon;
	} theme_cache;

    void _on_light_1_switch_pressed() {
        light1->set_visible(light_1_switch->is_pressed());
    }

    void _on_light_2_switch_pressed() {
        light2->set_visible(light_2_switch->is_pressed());
    }
	void _update_rotation(){
        Transform3D t;
        t.basis.rotate(Vector3(0, 1, 0), -rot_y);
        t.basis.rotate(Vector3(1, 0, 0), -rot_x);
        rotation->set_transform(t);
    }

protected:
	virtual void _update_theme_item_cache() override {
        SubViewportContainer::_update_theme_item_cache();

        theme_cache.light_1_icon = get_editor_theme_icon(SNAME("MaterialPreviewLight1"));
        theme_cache.light_2_icon = get_editor_theme_icon(SNAME("MaterialPreviewLight2"));
    }
	void _notification(int p_what) {
        switch (p_what) {
            case NOTIFICATION_THEME_CHANGED: {
                light_1_switch->set_icon(theme_cache.light_1_icon);
                light_2_switch->set_icon(theme_cache.light_2_icon);
            } break;
        }
    }
	void gui_input(const Ref<InputEvent> &p_event) override{
        ERR_FAIL_COND(p_event.is_null());

        Ref<InputEventMouseMotion> mm = p_event;
        if (mm.is_valid() && (mm->get_button_mask().has_flag(MouseButtonMask::LEFT))) {
            rot_x -= mm->get_relative().y * 0.01;
            rot_y -= mm->get_relative().x * 0.01;

            rot_x = CLAMP(rot_x, -Math_PI / 2, Math_PI / 2);
            _update_rotation();
        }
    }

public:
	void edit(Ref<CharacterBodyPrefab> p_prefab){
        preview_character->set_body_prefab(p_prefab);

        rot_x = Math::deg_to_rad(-15.0);
        rot_y = Math::deg_to_rad(30.0);
        _update_rotation();

        Vector3 ofs = Vector3(0,0,0);
        float m = 1;
        if (m != 0) {
            m = 1.0 / m;
            m *= 0.5;
            Transform3D xform;
            xform.basis.scale(Vector3(m, m, m));
            xform.origin = -xform.basis.xform(ofs); //-ofs*m;
            //xform.origin.z -= aabb.get_longest_axis_size() * 2;
            preview_character->set_transform(xform);
        }
    }
	AnimationNodePreview()
    {
        
        viewport = memnew(SubViewport);
        Ref<World3D> world_3d;
        world_3d.instantiate();
        viewport->set_world_3d(world_3d); // Use own world.
        add_child(viewport);
        viewport->set_disable_input(true);
        viewport->set_msaa_3d(Viewport::MSAA_4X);
        set_stretch(true);
        camera = memnew(Camera3D);
        camera->set_transform(Transform3D(Basis(), Vector3(0, 0, 1.1)));
        camera->set_perspective(45, 0.1, 10);
        viewport->add_child(camera);



        light1 = memnew(DirectionalLight3D);
        light1->set_transform(Transform3D().looking_at(Vector3(-1, -1, -1), Vector3(0, 1, 0)));
        viewport->add_child(light1);

        light2 = memnew(DirectionalLight3D);
        light2->set_transform(Transform3D().looking_at(Vector3(0, 1, 0), Vector3(0, 0, 1)));
        light2->set_color(Color(0.7, 0.7, 0.7));
        viewport->add_child(light2);

        rotation = memnew(Node3D);
        viewport->add_child(rotation);
        preview_character = memnew(CharacterBodyMain);
        rotation->add_child(preview_character);

        set_custom_minimum_size(Size2(1, 150) * EDSCALE);

        VBoxContainer *vb = memnew(VBoxContainer);
        vb->set_h_size_flags(SIZE_EXPAND_FILL);
        add_child(vb);


        HBoxContainer *hb = memnew(HBoxContainer);
        add_child(hb);
        vb->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT, Control::PRESET_MODE_MINSIZE, 2);

        hb->add_spacer();

        VBoxContainer *vb_light = memnew(VBoxContainer);
        hb->add_child(vb_light);

        light_1_switch = memnew(Button);
        light_1_switch->set_theme_type_variation("PreviewLightButton");
        light_1_switch->set_toggle_mode(true);
        light_1_switch->set_pressed(true);
        vb_light->add_child(light_1_switch);
        light_1_switch->connect(SceneStringName(pressed), callable_mp(this, &AnimationNodePreview::_on_light_1_switch_pressed));

        light_2_switch = memnew(Button);
        light_2_switch->set_theme_type_variation("PreviewLightButton");
        light_2_switch->set_toggle_mode(true);
        light_2_switch->set_pressed(true);
        vb_light->add_child(light_2_switch);
        light_2_switch->connect(SceneStringName(pressed), callable_mp(this, &AnimationNodePreview::_on_light_2_switch_pressed));

        rot_x = 0;
        rot_y = 0;
    }

};

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


        // 创建预览窗口
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
    }
    virtual String get_section_unfolded() const override{ return "Animation Item Condition Section"; }
public:

    virtual void init_node_item(int p_index,AnimatorNodeItemEditor* item) {

    }
    virtual void update_node_item(int p_index,AnimatorNodeItemEditor* item) {
        
    }
    virtual bool is_show_x_input() { return false; }
    virtual bool is_show_y_input() { return false; }

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
    Button* add_button = nullptr;
    VBoxContainer* item_parent = nullptr;
    List<AnimatorNodeItemEditor*> item_container;
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
    bool is_clip = node_editor->node->get_animation_item(index)->get_is_clip();
    select_animation_picker->set_visible(is_clip);
    select_animator_node_picker->set_visible(!is_clip);
    
    select_animation_picker->set_edited_resource(node_editor->node->get_animation_item(index)->get_animation());    
    select_animator_node_picker->set_edited_resource(node_editor->node->get_animation_item(index)->get_child_node());

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



