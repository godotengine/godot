#include "animation_preview_panel.h"


AnimationPreviewPanel::AnimationPreviewPanel() {
    set_h_size_flags(SIZE_EXPAND_FILL);
    set_v_size_flags(SIZE_EXPAND_FILL);

    update_animation_resource = memnew(Button);
    update_animation_resource->set_text(TTR("Update Animation"));
    update_animation_resource->set_h_size_flags(SIZE_EXPAND_FILL);
    update_animation_resource->connect("pressed", callable_mp(this, &AnimationPreviewPanel::_on_update_animation_resource));
    add_child(update_animation_resource);

    revert_show = memnew(CheckBox);
    revert_show->set_text(L"反向选择");
    revert_show->set_tooltip_text(L"反向选择,用来定位哪些没有分类的动画");
    revert_show->set_h_size_flags(SIZE_EXPAND_FILL);
    revert_show->connect("toggled", callable_mp(this, &AnimationPreviewPanel::_on_revert_show_toggled));
    add_child(revert_show);
    
    animation_group_tab = memnew(TabBar);
    animation_group_tab->set_self_modulate(Color(0.958148, 0.603324, 0.533511, 1));
    animation_group_tab->set_tab_alignment(TabBar::ALIGNMENT_CENTER);
    animation_group_tab->set_max_tab_width(80);
    animation_group_tab->connect("tab_changed", callable_mp(this, &AnimationPreviewPanel::_on_tab_changed));
    add_child(animation_group_tab);

    HSeparator* separator = memnew(HSeparator);
    separator->set_self_modulate(Color(0.349727, 0.355482, 0.26278, 1));
    add_child(separator);

    HBoxContainer* hbc = memnew(HBoxContainer);
    hbc->set_v_size_flags(SIZE_EXPAND_FILL);
    add_child(hbc);

    animation_tag_list = memnew(VFlowContainer);
    animation_tag_list->set_modulate(Color(0.895463, 0.702431, 0.0326403, 1));
    animation_tag_list->set_custom_minimum_size(Vector2(80, 0));
    animation_tag_list->set_reverse_fill(true);
    hbc->add_child(animation_tag_list);

    VSeparator* vs = memnew(VSeparator);
    vs->set_self_modulate(Color(0.349727, 0.355482, 0.26278, 1));
    hbc->add_child(vs);

    animation_list = memnew(ItemBox);
    hbc->add_child(animation_list);

}
void AnimationPreviewPanel::_notification(int what) {
    if (what == NOTIFICATION_ENTER_TREE) {
        load_animation_config();
    }
}
void AnimationPreviewPanel::_on_revert_show_toggled(bool pressed) {
    last_inv_select = pressed;
    refresh_animation_list(false);
}
void AnimationPreviewPanel::_on_tab_changed(int p_tab_idx) {
    last_select_group = animation_group_list[p_tab_idx];
    refresh_animation_list(false);

}
void AnimationPreviewPanel::_on_tag_pressed(bool value,String animation_tag) {
    last_select_tag[StringName(animation_tag)] = value;
    refresh_animation_list(false);
}

void AnimationPreviewPanel::_on_update_animation_resource() {
    animations.clear();
    for(auto& it : animation_list_paths) {
        parse_animation_path(it);
    }
    save_animation_config();
    refresh_animation_list();
    
}
void AnimationPreviewPanel::parse_animation_path(String path) {
    if(!DirAccess::exists(path)) {
        return;
    }    
    PackedStringArray files = DirAccess::get_files_at(path);   
    for (int i = 0; i < files.size(); i++) {
        String file = files[i];
        String ext = file.get_extension().to_lower();
        if (ext == "res" || ext == "tres") {
            Ref<Animation> animation = ResourceLoader::load(path.path_join(file));
            if(animation.is_valid() && !animation->get_preview_prefab_path().is_empty()) {
                Ref<AnimationInfo> animation_info = memnew(AnimationInfo);
                animation_info->animation_path = path.path_join(file);
                animation_info->animation_group = animation->get_animation_group();
                animation_info->animation_tag = animation->get_animation_tag();
                animations.push_back(animation_info);
            }
        }
    }
    PackedStringArray sub_dir = DirAccess::get_directories_at(path);
    for (int i = 0; i < sub_dir.size(); i++) {
        parse_animation_path(path.path_join(sub_dir[i]));
    }
}
void AnimationPreviewPanel::load_animation_config() {
    
    Ref<JSON> json = memnew(JSON);
    String config_path = "res://.godot/animation_config.json";
    Ref<FileAccess> f = FileAccess::open(config_path, FileAccess::READ);
    if (f.is_null()) {
        _on_update_animation_resource();
        return;
    }
    String json_str = f->get_as_text();
    json->parse(json_str);

    Dictionary dict = json->get_data();
    Array arr = dict["animations"];
    for (int i = 0; i < arr.size(); i++) {
        Dictionary dict = arr[i];
        Ref<AnimationInfo> animation_info;
        animation_info.instantiate();
        if (animation_info->load(dict)) {
            animations.push_back(animation_info);
        }
    }
    last_select_group = dict["last_select_group"];
    last_select_tag = dict["last_select_tag"];
    refresh_animation_list();
}
void AnimationPreviewPanel::refresh_animation_list(bool update_ui) {

    if(update_ui) {
        animation_group_list.clear();
        animation_tags_list.clear();
        CharacterManager* character_manager = CharacterManager::get_singleton();
        character_manager->get_animation_groups(&animation_group_list);
        character_manager->get_animation_tags(&animation_tags_list);

        while(animation_tag_list->get_child_count() > 0) {
            Node* child = animation_tag_list->get_child(0);
            animation_tag_list->remove_child(child);
            child->queue_free();
        }
        animation_group_tab->clear_tabs();

        for(auto& it : animation_group_list) {
            StringName name = it;
            animation_group_tab->add_tab(name);
        }

        for(auto& it : animation_tags_list) {
            StringName name = it;
            CheckBox* btn = memnew(CheckBox);
            btn->set_text(name.str());
            bool check = last_select_tag.get(name, false);
            btn->set_pressed(check);
            btn->connect("toggled", callable_mp(this, &AnimationPreviewPanel::_on_tag_pressed).bind(name));
            animation_tag_list->add_child(btn);
        }
        revert_show->set_pressed(last_inv_select);
    }

    // 获取动画列表
    curr_show_animations.clear();
    animation_list->clear();
    for(auto& it : animations) {
        if(it->is_show(last_select_group, last_select_tag,last_inv_select)) {
            AnimationNodePreview* preview = memnew(AnimationNodePreview);
            preview->set_animation_path(it->animation_path);
            curr_show_animations.push_back(it);
			animation_list->add_item(preview);
        }
    }
    
    
}
void AnimationPreviewPanel::save_animation_config() {
    Dictionary dict;
    Array arr;
    for(auto& it : animations) {
        Dictionary dict;
        it->save(dict);
        arr.push_back(dict);
    }
    dict["animations"] = arr;
    dict["last_select_group"] = last_select_group;
    dict["last_select_tag"] = last_select_tag;
    String json_str = JSON::stringify(dict);
    Ref<FileAccess> f = FileAccess::open("res://.godot/animation_config.json", FileAccess::WRITE);
    f->store_string(json_str);    

    curr_show_animations.clear();    
    animation_list->clear();

    for(auto& it : animations) {
        if(it->animation_group == last_select_group) {
            curr_show_animations.push_back(it);
        }
    }
    
}
