#pragma once

#include "body_main_editor.h"
#include "scene/3d/light_3d.h"
#include "scene/main/viewport.h"
#include "editor/editor_resource_picker.h"
#include "editor/themes/editor_scale.h"

#include "scene/gui/slider.h"

class AnimationNodePreview : public SubViewportContainer
{
	float rot_x;
	float rot_y;
    enum Play_State {
        PS_Play,
        PS_Pause,
        PS_Stop
    };
    Play_State play_state = PS_Stop;

	SubViewport *viewport = nullptr;
	Node3D *rotation = nullptr;
	DirectionalLight3D *light1 = nullptr;
	DirectionalLight3D *light2 = nullptr;
	Camera3D *camera = nullptr;
	Ref<CameraAttributesPractical> camera_attributes;

	Button *light_1_switch = nullptr;
	Button *light_2_switch = nullptr;
    Button *play_button = nullptr;
    Button *pause_button = nullptr;
    Button *stop_button = nullptr;

    Label* time_scale_lablel = nullptr;
    HSlider* time_scale_slider = nullptr;
    Label* animation_time_position_label = nullptr;
    Label* animator_time_label = nullptr;

	Ref<CharacterAnimatorNodeBase> node;
	CharacterBodyMain *preview_character = nullptr;

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

    void _on_play_button_pressed() {
        play();
    }

    void _on_pause_button_pressed() {
        pause();
    }

    void _on_stop_button_pressed() {
        stop();
    }

	void _update_rotation(){
        Transform3D t;
        t.basis.rotate(Vector3(0, 1, 0), -rot_y);
        t.basis.rotate(Vector3(1, 0, 0), -rot_x);
        rotation->set_transform(t);
    }
public:
    void play() {
        play_state = PS_Play;
        preview_character->set_editor_pause_animation(false);
        if(node.is_valid()) {
            preview_character->get_animator()->editor_play_animation(node);
        }
        update_play_state();
    }

    void pause() {
        play_state = PS_Pause;
        preview_character->set_editor_pause_animation(!preview_character->get_editor_pause_animation());
        if(preview_character->get_editor_pause_animation()) {
            pause_button->set_text(L"继续");
            pause_button->set_modulate(Color(1, 0.5, 0.5, 1));
        }
        else
        {
            pause_button->set_text(L"暂停");
            pause_button->set_modulate(Color(1, 1, 1, 1));
        }
        update_play_state();
    }

    void stop() {
        play_state = PS_Stop;
        preview_character->set_editor_pause_animation(false);
        preview_character->get_animator()->editor_stop_animation();
        update_play_state();
    }

public:
	static Ref<CharacterBodyPrefab>& get_globle_preview_prefab() {

		static Ref<CharacterBodyPrefab> prefab;
		return prefab;
	}
	static Ref<BlackboardPlan>& get_globle_preview_blackboard() {

		static Ref<BlackboardPlan> blackboard;
		return blackboard;
	}
	void set_globle_preview_prefab(const Ref<CharacterBodyPrefab>& p_prefab) {

		get_globle_preview_prefab() = p_prefab;
	}
	void set_globle_preview_blackboard(const Ref<BlackboardPlan>& p_blackboard) {
        get_globle_preview_blackboard() = p_blackboard;
        if(node.is_valid()) {
            node->set_blackboard_plan(p_blackboard);
        }
	}

protected:
    void update_play_state() {
        switch (play_state)
        {
        case PS_Play:
            play_button->set_disabled(false);
            pause_button->set_disabled(true);
            stop_button->set_disabled(true);
            break;
        case PS_Pause:
            play_button->set_disabled(true);
            pause_button->set_disabled(false);
            stop_button->set_disabled(false);
            break;
        case PS_Stop:
            play_button->set_disabled(false);
            pause_button->set_disabled(true);
            stop_button->set_disabled(true);
            break;
        }
    }
	virtual void _update_theme_item_cache() override {
        SubViewportContainer::_update_theme_item_cache();


		EditorBottomPanel* p_control = EditorNode::get_bottom_panel();
        theme_cache.light_1_icon = p_control->get_editor_theme_icon(SNAME("MaterialPreviewLight1"));
        theme_cache.light_2_icon = p_control->get_editor_theme_icon(SNAME("MaterialPreviewLight2"));
		light_1_switch->set_icon(theme_cache.light_1_icon);
		light_2_switch->set_icon(theme_cache.light_2_icon);
    }
	void _notification(int p_what) {
        switch (p_what) {
            case NOTIFICATION_ENTER_TREE: {
                Ref<CharacterBodyPrefab> prefab = get_preview_prefab();
                if (prefab.is_valid()) {
                    preview_character->set_body_prefab(prefab);
                }
            }
            break;
        }
    }
    void process(double delta) override {
        Ref<CharacterBodyPrefab> prefab = get_preview_prefab();
        if(preview_character->get_body_prefab() != prefab) {
            edit(prefab);            
        }

		Ref<BlackboardPlan> blackboard = get_preview_blackboard();
		if (preview_character->get_blackboard_plan() != blackboard)
		{
			preview_character->set_blackboard_plan(blackboard);
		}
        
    }
	Ref<BlackboardPlan> get_preview_blackboard() {
		CharacterBodyMain* body_main = CharacterBodyMain::get_current_editor_player();
		if (body_main != nullptr)
		{
			Ref<BlackboardPlan> body_prefab = body_main->get_blackboard_plan();
			if (body_prefab.is_valid())
			{
				return body_prefab;
			}
		}
		return get_globle_preview_blackboard();
	}
	Ref<CharacterBodyPrefab> get_preview_prefab() {
		CharacterBodyMain* body_main = CharacterBodyMain::get_current_editor_player();
		if (body_main != nullptr)
		{
			Ref<CharacterBodyPrefab> body_prefab = body_main->get_body_prefab();
			if (body_prefab.is_valid())
			{
				return body_prefab;
			}
		}
		return get_globle_preview_prefab();
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
	void edit(Ref<CharacterBodyPrefab> p_prefab){
        play_state = PS_Stop;
        update_play_state();
        preview_character->set_body_prefab(p_prefab);

        rot_x = Math::deg_to_rad(-15.0);
        rot_y = Math::deg_to_rad(30.0);
        _update_rotation();
        AABB aabb = preview_character->get_mesh_aabb();
        Vector3 ofs = aabb.get_center();
	    float m = aabb.get_longest_axis_size() * 1.2f;
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
public:
    void set_animator_node(Ref<CharacterAnimatorNodeBase> p_node) {
        node = p_node;
        if(node.is_valid()) {
            node->set_blackboard_plan(get_preview_blackboard());
        }
        stop();
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
        preview_character->init();
        rotation->add_child(preview_character);

        set_custom_minimum_size(Size2(1, 150) * EDSCALE);
        HBoxContainer *root_hb = memnew(HBoxContainer);
        add_child(root_hb);
        root_hb->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT, Control::PRESET_MODE_MINSIZE, 2);

        {
            VBoxContainer *vb = memnew(VBoxContainer);
            root_hb->add_child(vb);

            VBoxContainer *vb_light = memnew(VBoxContainer);
            vb->add_child(vb_light);

            light_1_switch = memnew(Button);
            light_1_switch->set_theme_type_variation("PreviewLightButton");
            light_1_switch->set_toggle_mode(true);
            light_1_switch->set_pressed(true);
            light_1_switch->set_modulate(Color(0.9, 0.9, 1, 1.0));
            vb_light->add_child(light_1_switch);
            light_1_switch->connect(SceneStringName(pressed), callable_mp(this, &AnimationNodePreview::_on_light_1_switch_pressed));

            light_2_switch = memnew(Button);
            light_2_switch->set_theme_type_variation("PreviewLightButton");
            light_2_switch->set_toggle_mode(true);
            light_2_switch->set_pressed(true);
            light_2_switch->set_modulate(Color(0.9, 0.9, 1, 1.0));
            vb_light->add_child(light_2_switch);
            light_2_switch->connect(SceneStringName(pressed), callable_mp(this, &AnimationNodePreview::_on_light_2_switch_pressed));

            play_button = memnew(Button);
            vb_light->add_child(play_button);
            play_button->set_text(L"播放");
            play_button->connect(SceneStringName(pressed), callable_mp(this, &AnimationNodePreview::_on_play_button_pressed));

            pause_button = memnew(Button);
            vb_light->add_child(pause_button);
            pause_button->set_text(L"暂停");
            pause_button->set_disabled(true);
            pause_button->connect(SceneStringName(pressed), callable_mp(this, &AnimationNodePreview::_on_pause_button_pressed));

            stop_button = memnew(Button);
            vb_light->add_child(stop_button);
            stop_button->set_text(L"停止");
            stop_button->set_disabled(true);
            stop_button->connect(SceneStringName(pressed), callable_mp(this, &AnimationNodePreview::_on_stop_button_pressed));

        }

        {
            VBoxContainer *vb = memnew(VBoxContainer);
            root_hb->add_child(vb);
            vb->set_h_size_flags(SIZE_EXPAND_FILL);

            HBoxContainer* hb = memnew(HBoxContainer);
            vb->add_child(hb);
            hb->set_h_size_flags(SIZE_EXPAND_FILL);

            time_scale_lablel = memnew(Label);
            time_scale_lablel->set_text(L"时间缩放:");
            hb->add_child(time_scale_lablel);

            time_scale_slider = memnew(HSlider);
            time_scale_slider->set_h_size_flags(SIZE_EXPAND_FILL);
            hb->add_child(time_scale_slider);
            time_scale_slider->set_min(0);
            time_scale_slider->set_max(5);
            time_scale_slider->set_step(0.01);
            time_scale_slider->set_value(1);
            time_scale_slider->set_allow_greater(true);
            time_scale_slider->set_allow_lesser(true);
            time_scale_slider->set_ticks(10);
            time_scale_slider->set_ticks_on_borders(true);

            hb = memnew(HBoxContainer);
            vb->add_child(hb);
            hb->set_h_size_flags(SIZE_EXPAND_FILL);
            animation_time_position_label = memnew(Label);
            animation_time_position_label->set_text(L"动画时间:");
            hb->add_child(animation_time_position_label);

            animator_time_label = memnew(Label);
            animator_time_label->set_text(L"播放时间:");
            hb->add_child(animator_time_label);

        }

        rot_x = 0;
        rot_y = 0;

        set_process(true);
    }

};
