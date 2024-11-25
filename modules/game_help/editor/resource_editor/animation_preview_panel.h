#include "../../game_gui/item_box.h"
#include "scene/gui/box_container.h"
#include "scene/gui/tab_bar.h"
#include "scene/gui/separator.h"
#include "scene/gui/button.h"
#include "scene/resources/animation.h"
#include "scene/gui/check_box.h"
#include "core/io/json.h"
#include  "../../logic/body_main.h"


class AnimationPreviewPanel : public VBoxContainer {
    GDCLASS(AnimationPreviewPanel, ItemBox);
public:
    AnimationPreviewPanel() {
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
        add_child(revert_show);
        
        animation_group = memnew(TabBar);
        animation_group->set_self_modulate(Color(0.958148, 0.603324, 0.533511, 1));
        animation_group->set_tab_alignment(TabBar::ALIGNMENT_CENTER);
        animation_group->set_max_tab_width(80);
        animation_group->connect("tab_changed", callable_mp(this, &AnimationPreviewPanel::_on_tab_changed));
        add_child(animation_group);

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
protected:
    void _on_tab_changed(int index) {

    }

    void _on_update_animation_resource() {
        animations.clear();
        for(auto& it : animations_paths) {
            parse_animation_path(it);
        }
        save_animation_config();
        
    }
    void parse_animation_path(String path) {
        if(!DirAccess::exists(path)) {
            return;
        }    
        PackedStringArray files = DirAccess::get_files_at(path);   
        for (int i = 0; i < files.size(); i++) {
            String file = files[i];
            String ext = file.get_extension().to_lower();
            if (ext == "res" || ext == "tres") {
                Ref<Animation> animation = ResourceLoader::load(path.path_join(file));
                if(animation.is_valid()) {
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
    struct AnimationInfo : RefCounted{
        Ref<Animation> animation;
        String animation_path;
        StringName animation_group;
        StringName animation_tag;
        bool load(Dictionary& p_dict) {
            animation_path = p_dict["animation_path"];
            animation_group = p_dict["animation_group"];
            animation_tag = p_dict["animation_tag"];
            return FileAccess::exists(animation_path);
        }
        void save(Dictionary& p_dict) {
            p_dict["animation_path"] = animation_path;
            p_dict["animation_group"] = animation_group;
            p_dict["animation_tag"] = animation_tag;
        }
    }
    void load_animation_config() {
        
        Ref<JSON> json = memnew(JSON);
        String config_path = "res://Assets/public/animation_config.json";
        Ref<FileAccess> f = FileAccess::open(config_path, FileAccess::READ);
        if (!f->is_open()) {
            _on_update_animation_resource();
            return;
        }
        String json_str = f->get_as_text();
        json->parse(json_str);

        Array arr = json->get_data();
        for (int i = 0; i < arr.size(); i++) {
            Dictionary dict = arr[i];
            AnimationInfo animation_info;
            if (animation_info.load(dict)) {
                animations_paths.push_back(animation_info.animation_path);
            }
        }
    }
    void save_animation_config() {
        Array arr;
        for(auto& it : animations_paths) {
            Dictionary dict;
            it->save(dict);
            arr.push_back(dict);
        }
        String json_str = JSON::stringify(arr);
        Ref<FileAccess> f = FileAccess::open("res://Assets/public/animation_config.json", FileAccess::WRITE);
        f->store_string(json_str);        
        
    }
    
protected:
    CheckBox* revert_show = nullptr;
    Button* update_animation_resource = nullptr;
    TabBar* animation_group= nullptr;
    VFlowContainer* animation_tag_list= nullptr;
    ItemBox* animation_list= nullptr;
    List<String> animation_list_paths = {"res://Assets/public/animation/", "res://Assets/public/human_animation/"};
    List<Ref<AnimationInfo>> animations_paths;
    List<Ref<AnimationInfo>> curr_show_animations;
};