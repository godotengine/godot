#pragma once

#include "scene/3d/light_3d.h"
#include "scene/main/viewport.h"
#include "editor/editor_resource_picker.h"
#include "editor/themes/editor_scale.h"
#include "editor/filesystem_dock.h"
#include "scene/gui/subviewport_container.h"
#include "editor/editor_properties.h"

#include "scene/gui/slider.h"

class AnimationNodePreview : public SubViewportContainer
{
    GDCLASS(AnimationNodePreview, SubViewportContainer)
    static void _bind_methods() {

        ClassDB::bind_method(D_METHOD("get_animation_groups"),&AnimationNodePreview::get_animation_groups);
        ClassDB::bind_method(D_METHOD("get_animation_tags"),&AnimationNodePreview::get_animation_tags);


        ClassDB::bind_method(D_METHOD("set_resource_group", "group"), &AnimationNodePreview::set_group);
        ClassDB::bind_method(D_METHOD("get_resource_group"), &AnimationNodePreview::get_group);

        ClassDB::bind_method(D_METHOD("set_resource_tag", "tag"), &AnimationNodePreview::set_tag);
        ClassDB::bind_method(D_METHOD("get_resource_tag"), &AnimationNodePreview::get_tag);

        ADD_PROPERTY(PropertyInfo(Variant::STRING, "resource_group"), "set_resource_group", "get_resource_group");
        ADD_PROPERTY(PropertyInfo(Variant::STRING, "resource_tag"), "set_resource_tag", "get_resource_tag");
    }
public:
	float rot_x;
	float rot_y;
    enum class Play_State {
        PS_Play,
        PS_Pause,
        PS_Stop
    };
    enum class PreviewType {
      PT_AnimationNode,
      PT_CharacterBodyPrefab,
      PT_Animation,  
    };
    Play_State play_state = Play_State::PS_Stop;
    PreviewType preview_type = PreviewType::PT_AnimationNode;
    Label *label = nullptr;
	SubViewport *viewport = nullptr;
	Node3D *rotation = nullptr;
	Node3D *charcter_parent = nullptr;
	DirectionalLight3D *light1 = nullptr;
	DirectionalLight3D *light2 = nullptr;
	Camera3D *camera = nullptr;
	Ref<CameraAttributesPractical> camera_attributes;

	Button *light_1_switch = nullptr;
	Button *light_2_switch = nullptr;
    Button *play_button = nullptr;
    Button *pause_button = nullptr;
    Button *stop_button = nullptr;

    Button* drag_button = nullptr;

    Label* time_scale_lablel = nullptr;
    HSlider* time_scale_slider = nullptr;
    Label* animation_time_position_label = nullptr;
    Label* animator_time_label = nullptr;

    EditorTextEnum* group_enum = nullptr;
    EditorTextEnum* tag_enum = nullptr;

	Ref<class CharacterAnimatorNodeBase> node;
    String node_path;
    bool node_path_is_load = false;


    Ref<class Animation> animation;
    String animation_path;
    bool animation_path_is_load = false;


    Ref<class CharacterBodyPrefab> prefab;
    String prefab_path;
    bool prefab_path_is_load = false;


	class CharacterBodyMain *preview_character = nullptr;
    bool preview_animation = true;

    bool show_by_editor_property = false;


	struct ThemeCache {
		Ref<Texture2D> light_1_icon;
		Ref<Texture2D> light_2_icon;
		Ref<Texture2D> drag_icon;
        Ref<Texture2D> play_icon;
        Ref<Texture2D> pause_icon;
        Ref<Texture2D> stop_icon;
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

    void _on_drag_button_pressed() ;
    

	void _update_rotation();

public:
    void play();

	void stop();

    void pause() ;

public:
	static Ref<CharacterBodyPrefab>& get_globle_preview_prefab();
	static Ref<class BlackboardPlan>& get_globle_preview_blackboard();
	void set_globle_preview_prefab(const Ref<CharacterBodyPrefab>& p_prefab) ;
	void set_globle_preview_blackboard(const Ref<BlackboardPlan>& p_blackboard) ;

    void set_group(String p_group) ;
    String get_group();
    void set_tag(String p_tag)  ;
    String get_tag() ;
protected:
    void update_play_state();
	virtual void _update_theme_item_cache() override ;
	void _notification(int p_what);
    Variant get_drag_data_fw(const Point2 &p_point, Control *p_from) ;

    bool can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const ;
    void drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from);
	Ref<BlackboardPlan> get_preview_blackboard() ;
	Ref<CharacterBodyPrefab> get_preview_prefab() ;
	void gui_input(const Ref<InputEvent> &p_event) override;
	void edit(Ref<CharacterBodyPrefab> p_prefab);

    Array get_animation_groups() ;
    Array get_animation_tags();
public:
    void set_prefab(Ref<CharacterBodyPrefab> p_prefab);
    void set_prefab_path(String p_path);
    String get_prefab_path() { return prefab_path; }

    void set_animator_node(Ref<CharacterAnimatorNodeBase> p_node);
	void set_animator_node_path(String p_path);
    String get_animator_node_path() { return node_path; }



    void set_animation(Ref<Animation> p_animation) ;
    void set_animation_path(String p_path) ;
    String get_animation_path() { return animation_path; }

    void set_show_by_editor_property(bool p_show) { show_by_editor_property = p_show; }
	void process(double delta) override;

	AnimationNodePreview();

};
