#include "animation_preview_panel.h"
#include "scene/gui/scroll_container.h"


AnimationPreviewPanel::AnimationPreviewPanel() {
    set_h_size_flags(SIZE_EXPAND_FILL);
    set_v_size_flags(SIZE_EXPAND_FILL);

    update_animation_resource = memnew(Button);
    update_animation_resource->set_text(TTR("Update Animation"));
    update_animation_resource->set_h_size_flags(SIZE_EXPAND_FILL);
    update_animation_resource->connect("pressed", callable_mp(this, &AnimationPreviewPanel::_on_update_animation_resource));
    add_child(update_animation_resource);

    {
        HBoxContainer* hbc = memnew(HBoxContainer);
        add_child(hbc);


		revert_show = memnew(CheckBox);
		revert_show->set_text(L"反向选择");
		revert_show->set_tooltip_text(L"反向选择,用来定位哪些没有分类的动画");
		revert_show->set_h_size_flags(SIZE_EXPAND_FILL);
		revert_show->connect("toggled", callable_mp(this, &AnimationPreviewPanel::_on_revert_show_toggled));
		hbc->add_child(revert_show);

        HSeparator* separator = memnew(HSeparator);
        separator->set_self_modulate(Color(0.349727, 0.355482, 0.26278, 1));
        separator->set_h_size_flags(SIZE_EXPAND_FILL);
        add_child(separator);

        animation_group_tab = memnew(TabBar);
        animation_group_tab->set_self_modulate(Color(0.958148, 0.603324, 0.533511, 1));
        animation_group_tab->set_tab_alignment(TabBar::ALIGNMENT_CENTER);
        animation_group_tab->set_max_tab_width(80);
        animation_group_tab->connect("tab_changed", callable_mp(this, &AnimationPreviewPanel::_on_tab_changed));
        add_child(animation_group_tab);
    }

    HSeparator* separator = memnew(HSeparator);
    separator->set_self_modulate(Color(0.349727, 0.355482, 0.26278, 1));
    add_child(separator);

    HBoxContainer* hbc = memnew(HBoxContainer);
    hbc->set_v_size_flags(SIZE_EXPAND_FILL);
    add_child(hbc);

    {
        ScrollContainer* hsc = memnew(ScrollContainer);
        hsc->set_v_size_flags(SIZE_EXPAND_FILL);
        hsc->set_custom_minimum_size(Vector2(180, 0));
        hbc->add_child(hsc);

        animation_tag_list = memnew(HFlowContainer);
        animation_tag_list->set_modulate(Color(0.895463, 0.702431, 0.0326403, 1));
        animation_tag_list->set_custom_minimum_size(Vector2(80, 0));
        hsc->add_child(animation_tag_list);

    }


    VSeparator* vs = memnew(VSeparator);
    vs->set_self_modulate(Color(0.349727, 0.355482, 0.26278, 1));
    hbc->add_child(vs);

    animation_list = memnew(ItemBox);
    animation_list->set_item_visible_change_callback(callable_mp(this, &AnimationPreviewPanel::on_item_visible_state_change));
    animation_list->set_h_size_flags(SIZE_EXPAND_FILL);
    animation_list->set_v_size_flags(SIZE_EXPAND_FILL); 
    animation_list->set_layout_mode(LAYOUT_MODE_CONTAINER);
    hbc->add_child(animation_list);

    set_process(true);
}
void AnimationPreviewPanel::_notification(int what) {
    if (what == NOTIFICATION_ENTER_TREE) {
        load_animation_config();
    }
    else if (what == NOTIFICATION_PROCESS) {
        if(is_dirty) {
            if(update_preview()) {
                is_dirty = false;
            }
		}
    }
    else if (what == NOTIFICATION_EXIT_TREE) {
        animation_list->clear();
        for(auto& it : unuse_preview_list) {
            it->queue_free();
        }
        unuse_preview_list.clear();
        animations.clear();
        curr_show_animations.clear();
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
        animation_group_tab->set_current_tab_by_name(last_select_group);

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
            curr_show_animations.push_back(it);
			animation_list->add_item(it);
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

void AnimationPreviewPanel::on_item_visible_state_change(ItemBoxItem* item,bool p_visible) {
    Ref<AnimationInfo> animation_info = item->data;
    if(animation_info.is_null()) {
        return;
    }
    if(animation_info->is_visible == p_visible) {
        return;
    }
    animation_info->is_visible = p_visible;
    if(p_visible) {
		if (animation_preview_list.has(item)) {
			return;
		}
        AnimationNodePreview* preview = get_item_preview();
        preview->set_h_size_flags(SIZE_EXPAND_FILL);
        preview->set_v_size_flags(SIZE_EXPAND_FILL);
        animation_preview_list[item] = preview;
    }
    else {
        HashMap<ItemBoxItem*,AnimationNodePreview*>::Iterator it = animation_preview_list.find(item);
        if(it != animation_preview_list.end()) {
            unuse_preview_list.push_back(it->value);
            Node* parent = it->value->get_parent();
            if (parent != nullptr) {
                parent->remove_child(it->value);
            }
            animation_preview_list.remove(it);
        }
		else {
			return;
		}
    }
    is_dirty = true;
}
    
AnimationNodePreview* AnimationPreviewPanel::get_item_preview() {
    if(unuse_preview_list.size() > 0) {
        AnimationNodePreview* preview = unuse_preview_list.front()->get();
        unuse_preview_list.pop_front();
        return preview;
    }
    return memnew(AnimationNodePreview);        
}

bool AnimationPreviewPanel::update_preview() {
    for(auto& it : animation_preview_list) {
        if(it.value->get_parent() != it.key) {
            Ref<AnimationInfo> _data = it.key->data;
            if(_data.is_valid()) {
                it.value->set_animation_path(_data->animation_path);
				it.value->process(0);
                it.key->add_child(it.value);                 
            }
        }
    }
    return true;
}
