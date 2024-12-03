#include "animator_node_preview.h"

#include "body_main_editor.h"
#include "../logic/character_manager.h"
void AnimationNodePreview::_on_drag_button_pressed() {
    
    Ref<Resource> res;
    switch (preview_type	)
    {
    case PT_AnimationNode:
        res = node;
        break;
    case PT_CharacterBodyPrefab:
        res = prefab;
        break;
    case PT_Animation:
        res = animation;
        break;
    }
    if(res.is_null()) {
        return ;
    }
    FileSystemDock::get_singleton()->navigate_to_path(res->get_path());
}


void AnimationNodePreview::_update_rotation(){
    Transform3D t;
    t.basis.rotate(Vector3(0, 1, 0), -rot_y);
    t.basis.rotate(Vector3(1, 0, 0), -rot_x);
    rotation->set_transform(t);
}

void AnimationNodePreview::on_visilbe_changed(bool p_visible) {
    
    if(!is_visible_in_tree()) {
        switch (preview_type)
        {
        case PT_AnimationNode:
            node = Ref<CharacterAnimatorNodeBase>();
            node_path_is_load = false;
            break;
        case PT_CharacterBodyPrefab:
            prefab = Ref<CharacterBodyPrefab>();
            prefab_path_is_load = false;
            break;
        case PT_Animation:
            animation = Ref<Animation>();
            animation_path_is_load = false;
            break;
        }

        stop();
    }
    group_enum->update_property();
    tag_enum->update_property();

}
void AnimationNodePreview::play() {
    if(preview_type == PT_CharacterBodyPrefab) {
        return;            
    }
    if(play_state == PS_Play) {
        return;
    }
    play_state = PS_Play;
    preview_character->set_editor_pause_animation(false);
    if(preview_type == PT_Animation) {
        
        if(!animation_path_is_load && animation.is_null()) {
            animation = ResourceLoader::load(animation_path);
            animation_path_is_load = true;
        }
        if(animation.is_valid()) {
            preview_character->get_animator()->editor_play_animation(animation);
        }
        return;
    } else if(preview_type == PT_AnimationNode) {
        if(!node_path_is_load && node.is_null()) {
            node = ResourceLoader::load(node_path);
            node_path_is_load = true;
        }
        if(node.is_valid()) {
            preview_character->get_animator()->editor_play_animation(node);
        }
    } 
    update_play_state();
}

void AnimationNodePreview::pause() {
    if(preview_type == PT_CharacterBodyPrefab) {
        return;            
    }
    if(play_state == PS_Pause) {
        return;
    }
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

void AnimationNodePreview::stop() {
    if(preview_type == PT_CharacterBodyPrefab) {
        return;            
    }
    if(play_state == PS_Stop) {
        return;
    }
    play_state = PS_Stop;
    preview_character->set_editor_pause_animation(false);
    preview_character->get_animator()->editor_stop_animation();
    update_play_state();
}

Ref<CharacterBodyPrefab>& AnimationNodePreview::get_globle_preview_prefab() {

    static Ref<CharacterBodyPrefab> prefab;
    return prefab;
}
Ref<BlackboardPlan>& AnimationNodePreview::get_globle_preview_blackboard() {

    static Ref<BlackboardPlan> blackboard;
    return blackboard;
}
void AnimationNodePreview::set_globle_preview_prefab(const Ref<CharacterBodyPrefab>& p_prefab) {

    get_globle_preview_prefab() = p_prefab;
}
void AnimationNodePreview::set_globle_preview_blackboard(const Ref<BlackboardPlan>& p_blackboard) {
    get_globle_preview_blackboard() = p_blackboard;
    if(node.is_valid()) {
        node->set_blackboard_plan(p_blackboard);
    }
}

void AnimationNodePreview::update_play_state() {
    if(preview_type == PT_CharacterBodyPrefab) {
        play_button->set_visible(false);
        pause_button->set_visible(false);
        stop_button->set_visible(false);
        return;
    }
    play_button->set_visible(true);
    pause_button->set_visible(true);
    stop_button->set_visible(true);
    switch (play_state)
    {
    case PS_Play:
        play_button->set_disabled(true);
        play_button->set_focus_mode(Control::FOCUS_NONE);

        pause_button->set_disabled(false);
        pause_button->set_focus_mode(Control::FOCUS_CLICK);

        stop_button->set_disabled(false);
        stop_button->set_focus_mode(Control::FOCUS_CLICK);
        break;
    case PS_Pause:
        play_button->set_disabled(true);
        play_button->set_focus_mode(Control::FOCUS_NONE);

        pause_button->set_disabled(false);
        pause_button->set_focus_mode(Control::FOCUS_CLICK);

        stop_button->set_disabled(false);
        stop_button->set_focus_mode(Control::FOCUS_CLICK);
        break;
    case PS_Stop:
        play_button->set_disabled(false);
        play_button->set_focus_mode(Control::FOCUS_CLICK);

        pause_button->set_disabled(true);
        pause_button->set_focus_mode(Control::FOCUS_NONE);

        stop_button->set_disabled(true);
        stop_button->set_focus_mode(Control::FOCUS_NONE);
        break;
    }
}
void AnimationNodePreview::_update_theme_item_cache() {
    SubViewportContainer::_update_theme_item_cache();


    EditorBottomPanel* p_control = EditorNode::get_bottom_panel();
    theme_cache.light_1_icon = p_control->get_editor_theme_icon(SNAME("MaterialPreviewLight1"));
    theme_cache.light_2_icon = p_control->get_editor_theme_icon(SNAME("MaterialPreviewLight2"));
    theme_cache.drag_icon = p_control->get_editor_theme_icon(SNAME("ExternalLink"));
    theme_cache.play_icon = p_control->get_editor_theme_icon(SNAME("Play"));
    theme_cache.pause_icon = p_control->get_editor_theme_icon(SNAME("Pause"));
    theme_cache.stop_icon = p_control->get_editor_theme_icon(SNAME("Stop"));
    //light_1_switch->set_button_icon(theme_cache.light_1_icon);
    //light_2_switch->set_button_icon(theme_cache.light_2_icon);
    drag_button->set_button_icon(theme_cache.drag_icon);

    play_button->set_button_icon(theme_cache.play_icon);
    pause_button->set_button_icon(theme_cache.pause_icon);
    stop_button->set_button_icon(theme_cache.stop_icon);
}
void AnimationNodePreview::_notification(int p_what) {
    switch (p_what) {
        case NOTIFICATION_ENTER_TREE: {
            Ref<CharacterBodyPrefab> _prefab = get_preview_prefab();
            if (_prefab.is_valid()) {
                edit(_prefab);
            }
        }
        break;
    }
}
void AnimationNodePreview::process(double delta)  {
    Ref<CharacterBodyPrefab> _prefab = get_preview_prefab();
    if(preview_character->get_body_prefab() != _prefab) {
        edit(_prefab);
        group_enum->update_property();
        tag_enum->update_property();
    }

    if(preview_type != PT_Animation) {
        Ref<BlackboardPlan> blackboard = get_preview_blackboard();
        if (preview_character->get_blackboard_plan() != blackboard)
        {
            preview_character->set_blackboard_plan(blackboard);
        }

    }
    play();
    
}
Variant AnimationNodePreview::get_drag_data_fw(const Point2 &p_point, Control *p_from) {

    Ref<Resource> res;
    switch (preview_type)
    {
    case PT_AnimationNode:
        res = node;
        break;
    case PT_CharacterBodyPrefab:
        res = prefab;
        break;
    case PT_Animation:
        res = animation;
        break;
    }
    if(res.is_null()) {
        return Variant();
    }
    Dictionary drag_data = EditorNode::get_singleton()->drag_resource(res, p_from);
    drag_data["source_picker"] = get_instance_id();
    return drag_data;
}

bool AnimationNodePreview::can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const {
    return false;
}
void AnimationNodePreview::drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from)  {

}
Ref<BlackboardPlan> AnimationNodePreview::get_preview_blackboard() {
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
Ref<CharacterBodyPrefab> AnimationNodePreview::get_preview_prefab() {
    if(preview_type == PT_CharacterBodyPrefab) {
        if(!prefab_path_is_load && prefab.is_null()) {
            prefab = ResourceLoader::load(prefab_path);
            prefab_path_is_load = true;
        }
        return prefab;
    }
    if(preview_type == PT_Animation) {
        if(!animation_path_is_load && animation.is_null()) {
            animation = ResourceLoader::load(animation_path);
            animation_path_is_load = true;
        }
        if(animation.is_valid()) {
            Ref<CharacterBodyPrefab> _prefab = ResourceLoader::load(animation->get_preview_prefab_path());
            if(_prefab.is_valid()) {
                return _prefab;
            }                  
        }         
    }
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
void AnimationNodePreview::gui_input(const Ref<InputEvent> &p_event) {
    ERR_FAIL_COND(p_event.is_null());

    Ref<InputEventMouseMotion> mm = p_event;
    if (mm.is_valid() && (mm->get_button_mask().has_flag(MouseButtonMask::LEFT))) {
        rot_x -= mm->get_relative().y * 0.01;
        rot_y -= mm->get_relative().x * 0.01;

        rot_x = CLAMP(rot_x, -Math_PI / 2, Math_PI / 2);
        _update_rotation();
    }
}
void AnimationNodePreview::edit(Ref<CharacterBodyPrefab> p_prefab){
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
        charcter_parent->set_transform(xform);
    }
	group_enum->update_property();
	tag_enum->update_property();
}

void AnimationNodePreview::set_prefab(Ref<CharacterBodyPrefab> p_prefab) {
    prefab = p_prefab;
    prefab_path = p_prefab->get_path();
    preview_type = PT_CharacterBodyPrefab;

    group_enum->set_visible(true);
    tag_enum->set_visible(false);
}
void AnimationNodePreview::set_prefab_path(String p_path) {
    prefab_path = p_path;
    prefab_path_is_load = false;
    prefab.unref();
    preview_type = PT_CharacterBodyPrefab;
    group_enum->set_visible(true);
    tag_enum->set_visible(false);
}


void AnimationNodePreview::set_animator_node(Ref<CharacterAnimatorNodeBase> p_node) {
    preview_type = PT_AnimationNode;
    node = p_node;
    node_path = p_node->get_path();
    if(node.is_valid()) {
        node->set_blackboard_plan(get_preview_blackboard());
    }
    stop();
    group_enum->set_visible(false);
    tag_enum->set_visible(false);
}
void AnimationNodePreview::set_animator_node_path(String p_path) {
    node_path = p_path;
    node_path_is_load = false;
    node.unref();
    preview_type = PT_AnimationNode;
    if (node.is_valid()) {
        node->set_blackboard_plan(get_preview_blackboard());
    }
    stop();
    group_enum->set_visible(false);
    tag_enum->set_visible(false);
}



void AnimationNodePreview::set_animation(Ref<Animation> p_animation) {
    preview_type = PT_Animation;
    animation = p_animation;
    animation_path = p_animation->get_path();
    if(node.is_valid()) {
        node->set_blackboard_plan(get_preview_blackboard());
    }
    stop();
    group_enum->set_visible(true);
    tag_enum->set_visible(true);
}
void AnimationNodePreview::set_animation_path(String p_path) {
    animation_path = p_path;
    animation_path_is_load = false;
    animation.unref();
    preview_type = PT_Animation;
    if(node.is_valid()) {
        node->set_blackboard_plan(get_preview_blackboard());
    }
    stop();
    group_enum->set_visible(true);
    tag_enum->set_visible(true);
    
}

AnimationNodePreview::AnimationNodePreview()
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
    
    charcter_parent = memnew(Node3D);
    rotation->add_child(charcter_parent);


    preview_character = memnew(CharacterBodyMain);
    preview_character->init();
    charcter_parent->add_child(preview_character);


    set_custom_minimum_size(Size2(1, 150) * EDSCALE);
    HBoxContainer *root_hb = memnew(HBoxContainer);
    root_hb->set_modulate(Color(1, 1, 1, 0.8f));
    add_child(root_hb);
    root_hb->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT, Control::PRESET_MODE_MINSIZE, 2);

    {
        VBoxContainer *vb = memnew(VBoxContainer);
        root_hb->add_child(vb);

        VBoxContainer *vb_light = memnew(VBoxContainer);
        vb->add_child(vb_light);

        // light_1_switch = memnew(Button);
        // light_1_switch->set_theme_type_variation("PreviewLightButton");
        // light_1_switch->set_toggle_mode(true);
        // light_1_switch->set_pressed(true);
        // light_1_switch->set_modulate(Color(0.9, 0.9, 1, 1.0));
        // vb_light->add_child(light_1_switch);
        // light_1_switch->connect(SceneStringName(pressed), callable_mp(this, &AnimationNodePreview::_on_light_1_switch_pressed));

        // light_2_switch = memnew(Button);
        // light_2_switch->set_theme_type_variation("PreviewLightButton");
        // light_2_switch->set_toggle_mode(true);
        // light_2_switch->set_pressed(true);
        // light_2_switch->set_modulate(Color(0.9, 0.9, 1, 1.0));
        // vb_light->add_child(light_2_switch);
        // light_2_switch->connect(SceneStringName(pressed), callable_mp(this, &AnimationNodePreview::_on_light_2_switch_pressed));

        play_button = memnew(Button);
        vb_light->add_child(play_button);
        play_button->set_button_icon(theme_cache.play_icon);
        play_button->set_tooltip_text(L"播放");
        play_button->connect(SceneStringName(pressed), callable_mp(this, &AnimationNodePreview::_on_play_button_pressed));

        pause_button = memnew(Button);
        vb_light->add_child(pause_button);
        pause_button->set_button_icon(theme_cache.pause_icon);
        pause_button->set_tooltip_text(L"暂停");
        pause_button->set_disabled(true);
        pause_button->connect(SceneStringName(pressed), callable_mp(this, &AnimationNodePreview::_on_pause_button_pressed));

        stop_button = memnew(Button);
        vb_light->add_child(stop_button);
        stop_button->set_button_icon(theme_cache.stop_icon);
        stop_button->set_tooltip_text(L"停止");
        stop_button->set_disabled(true);
        stop_button->connect(SceneStringName(pressed), callable_mp(this, &AnimationNodePreview::_on_stop_button_pressed));

        

        animation_time_position_label = memnew(Label);
        animation_time_position_label->set_text(L"ATime:");
        animation_time_position_label->set_h_size_flags(SIZE_EXPAND_FILL);
        vb_light->add_child(animation_time_position_label);

        animator_time_label = memnew(Label);
        animator_time_label->set_text(L"T Time:");
        animator_time_label->set_h_size_flags(SIZE_EXPAND_FILL);
        vb_light->add_child(animator_time_label);

    }

    {
        VBoxContainer *vb = memnew(VBoxContainer);
        root_hb->add_child(vb);
        vb->set_h_size_flags(SIZE_EXPAND_FILL);

        {
            HBoxContainer* hb = memnew(HBoxContainer);
            hb->set_h_size_flags(SIZE_EXPAND_FILL);
            vb->add_child(hb);


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


            drag_button = memnew(Button);
            drag_button->set_custom_minimum_size(Size2(28.0, 28.0) * EDSCALE);
            drag_button->set_button_icon(theme_cache.drag_icon);
            drag_button->set_tooltip_text(L"鼠标左键点击定位资源,按住鼠标左键,拖拽我呀!八格牙路!");
            drag_button->connect(SceneStringName(pressed), callable_mp(this, &AnimationNodePreview::_on_drag_button_pressed));
            SET_DRAG_FORWARDING_GCD(drag_button, AnimationNodePreview);
            hb->add_child(drag_button);
        }

        {
            HBoxContainer* hb = memnew(HBoxContainer);
            hb->set_h_size_flags(SIZE_EXPAND_FILL);
            vb->add_child(hb);


            group_enum = memnew(EditorTextEnum);
            group_enum->set_h_size_flags(SIZE_EXPAND_FILL);
            group_enum->set_object_and_property(this, StringName("group"));
            group_enum->setup({});
            group_enum->set_dynamic(true, "get_animation_groups");
            hb->add_child(group_enum);

            tag_enum = memnew(EditorTextEnum);
            tag_enum->set_h_size_flags(SIZE_EXPAND_FILL);
            tag_enum->set_object_and_property(this, StringName("tag"));
            tag_enum->setup({});
            tag_enum->set_dynamic(true, "get_animation_groups");
            hb->add_child(tag_enum);

        }


    }

    rot_x = 0;
    rot_y = 0;

    set_process(true);
}

Array AnimationNodePreview::get_animation_groups() {
    Array arr;
    CharacterManager::get_singleton()->get_animation_groups(&arr);
    return arr;
}
Array AnimationNodePreview::get_animation_tags() {
    Array arr;
    CharacterManager::get_singleton()->get_animation_tags(&arr);
    return arr;
}

void AnimationNodePreview::set_group(String p_group) {
    if(!group_enum->is_visible_in_tree()) {
        return;
    }
    switch (preview_type)
    {
    case PT_AnimationNode:
        {
        }
        break;
    case PT_CharacterBodyPrefab:   
        if(!prefab_path_is_load && prefab.is_null()) {
            prefab = ResourceLoader::load(prefab_path);
            prefab_path_is_load = true;
        }
        if(prefab.is_valid()) {
            prefab->set_resource_group(p_group);
            ResourceSaver::save(prefab, prefab_path);
        }
        break;
    case PT_Animation:
        {
            if(animation.is_valid()) {
                animation->set_animation_group(p_group);
                ResourceSaver::save(animation, animation_path);
            }
        }
        break;
    
    default:
        break;
    }
}

String AnimationNodePreview::get_group() { 

    switch (preview_type)
    {
    case PT_AnimationNode:
        {
        }
        break;
    case PT_CharacterBodyPrefab:   
        if(prefab.is_valid()) {
            return prefab->get_resource_group();
        }
        break;
    case PT_Animation:
        {
            if(animation.is_valid()) {
                return animation->get_animation_group();
            }
        }
        break;
    
    default:
        break;
    }
    return "";        
}

void AnimationNodePreview::set_tag(String p_tag)  {
    if(!tag_enum->is_visible_in_tree()) {
        return;
    }
    switch (preview_type)
    {
    case PT_AnimationNode:
        {
        }
        break;
    case PT_CharacterBodyPrefab:   
        break;
    case PT_Animation:
        {
            if(animation.is_valid()) {
                animation->set_animation_tag(p_tag);
                ResourceSaver::save(animation, animation_path);
            }
        }
    }
}

String AnimationNodePreview::get_tag() { 
    switch (preview_type)
    {
    case PT_AnimationNode:
        {
        }
        break;
    case PT_CharacterBodyPrefab:   
        break;
    case PT_Animation:
        {
            if(animation.is_valid()) {
                return animation->get_animation_tag();    
            }
        }
    }
    return "";
}


