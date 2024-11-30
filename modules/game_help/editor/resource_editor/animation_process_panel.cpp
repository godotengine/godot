#include "animation_process_panel.h"
#include "scene/gui/separator.h"
#include "core/io/json.h"



HBoxContainer* AnimationProcessPanel::create_line(Control* control , bool is_side_separator ) {
    HBoxContainer* hb = memnew(HBoxContainer);
    if(is_side_separator) {
        HSeparator* sep = memnew(HSeparator);
        sep->set_h_size_flags(SIZE_EXPAND_FILL);
        hb->add_child(sep);
    }
    hb->add_child(control);

    if(is_side_separator) {
        HSeparator* sep = memnew(HSeparator);
        sep->set_h_size_flags(SIZE_EXPAND_FILL);
        hb->add_child(sep);
    }
    return hb;

}
HBoxContainer* AnimationProcessPanel::create_line(Control* control0 , Control* control1,  bool is_side_separator ) {
    HBoxContainer* hb = memnew(HBoxContainer);
    if(is_side_separator) {
        HSeparator* sep = memnew(HSeparator);
        sep->set_h_size_flags(SIZE_EXPAND_FILL);
        hb->add_child(sep);
    }
    hb->add_child(control0);
    hb->add_child(control1);

    if(is_side_separator) {
        HSeparator* sep = memnew(HSeparator);
        sep->set_h_size_flags(SIZE_EXPAND_FILL);
        hb->add_child(sep);
    }
    return hb;

}

HBoxContainer* AnimationProcessPanel::create_line(Control* control0 , Control* control1, Control* control2,  bool is_side_separator ) {
    HBoxContainer* hb = memnew(HBoxContainer);
    if(is_side_separator) {
        HSeparator* sep = memnew(HSeparator);
        sep->set_h_size_flags(SIZE_EXPAND_FILL);
        hb->add_child(sep);
    }
    hb->add_child(control0);
    hb->add_child(control1);
    hb->add_child(control2);

    if(is_side_separator) {
        HSeparator* sep = memnew(HSeparator);
        sep->set_h_size_flags(SIZE_EXPAND_FILL);
        hb->add_child(sep);
    }
    return hb;

}

HBoxContainer* AnimationProcessPanel::create_line(Control* control0 , Control* control1, Control* control2, Control* control3,  bool is_side_separator ) {
    HBoxContainer* hb = memnew(HBoxContainer);
    if(is_side_separator) {
        HSeparator* sep = memnew(HSeparator);
        sep->set_h_size_flags(SIZE_EXPAND_FILL);
        hb->add_child(sep);
    }
    hb->add_child(control0);
    hb->add_child(control1);
    hb->add_child(control2);
    hb->add_child(control3);

    if(is_side_separator) {
        HSeparator* sep = memnew(HSeparator);
        sep->set_h_size_flags(SIZE_EXPAND_FILL);
        hb->add_child(sep);
    }
    return hb;

}

AnimationProcessPanel::AnimationProcessPanel() {
    load_animation_config();

    HBoxContainer* hb = memnew(HBoxContainer);
    hb->set_h_size_flags(SIZE_EXPAND_FILL);
    add_child(hb);
    {
        VBoxContainer* vb = memnew(VBoxContainer);
        vb->set_custom_minimum_size(Vector2(300, 0));
        hb->add_child(vb);

        property_preview_prefab_path = memnew(EditorPropertyPath);
        property_preview_prefab_path->set_label(L"選擇预制件：");
        property_preview_prefab_path->set_object_and_property(this, SNAME("preview_prefab_path"));
        property_preview_prefab_path->setup({ "res", "tres" }, false, false);
        property_preview_prefab_path->set_h_size_flags(SIZE_EXPAND_FILL);
        vb->add_child(property_preview_prefab_path);

        preview = memnew(AnimationNodePreview);
        preview->set_custom_minimum_size(Vector2(400, 400));
        vb->add_child(preview);
        if(preview_prefab_path != "") {
            preview->set_prefab_path(preview_prefab_path);
        }
    }

    {
        VBoxContainer* vb = memnew(VBoxContainer);
        vb->set_h_size_flags(SIZE_EXPAND_FILL);
        vb->set_custom_minimum_size(Vector2(300, 0));
        hb->add_child(vb);

        {

        
            Label* label = memnew(Label);
            label->set_h_size_flags(SIZE_EXPAND_FILL);
            label->set_text(L"单个动画处理");
            label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
            label->set_modulate(Color(1,0.8,0.7,1));
            vb->add_child(create_line(  label,true));
                                    
            VSeparator* sep = memnew(VSeparator);
            vb->add_child(sep);

            {
                single_path = memnew(EditorPropertyPath);
                single_path->set_label(L"选择动画文件");
                single_path->set_object_and_property(this, SNAME("single_animation_file_path"));
                single_path->setup({ "res", "tres" }, false, false);
                single_path->set_h_size_flags(SIZE_EXPAND_FILL);

                single_animation_group = memnew(EditorPropertyTextEnum);
                single_animation_group->set_label(L"动画组");
                single_animation_group->set_custom_minimum_size(Vector2(300, 0));
                single_animation_group->set_object_and_property(this, "single_animation_group");
                single_animation_group->set_dynamic(true, "get_animation_groups");
                single_animation_group->setup(Vector<String>());

                single_animation_tags = memnew(EditorPropertyTextEnum);
                single_animation_tags->set_label(L"动画标签");
                single_animation_tags->set_custom_minimum_size(Vector2(300, 0));
                single_animation_tags->set_object_and_property(this, "single_animation_tags");
                single_animation_tags->set_dynamic(true, "get_animation_tags");
                single_animation_tags->setup(Vector<String>());

                conver_single_button = memnew(Button);
                conver_single_button->set_text(L"转换");
                conver_single_button->connect(SceneStringName(pressed), callable_mp(this, &AnimationProcessPanel::_on_conver_single_pressed));


                
                vb->add_child(create_line(  single_path,single_animation_group,single_animation_tags));

                vb->add_child(conver_single_button);
            }

        
        }



        {
            Label* label = memnew(Label);
            label->set_h_size_flags(SIZE_EXPAND_FILL);
            label->set_text(L"多个动画处理");
            label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
            label->set_modulate(Color(1,0.8,0.7,1));

            vb->add_child(create_line(label,true));
                                    
            VSeparator* sep = memnew(VSeparator);
            vb->add_child(sep);
            {
                multe_path = memnew(EditorPropertyPath);
                multe_path->set_label(L"选择文件夹");
                multe_path->set_object_and_property(this, "multe_animation_file_path");
                multe_path->set_h_size_flags(SIZE_EXPAND_FILL);
                multe_path->setup(Vector<String>(), true, false);

                multe_animation_group = memnew(EditorPropertyTextEnum);
                multe_animation_group->set_label(L"动画组");
                multe_animation_group->set_custom_minimum_size(Vector2(300, 0));
                multe_animation_group->set_object_and_property(this, "multe_animation_group");
                multe_animation_group->set_dynamic(true, "get_animation_groups");
                multe_animation_group->setup(Vector<String>());

                multe_animation_tags = memnew(EditorPropertyTextEnum);
                multe_animation_tags->set_label(L"动画标签");
                multe_animation_tags->set_custom_minimum_size(Vector2(300, 0));
                multe_animation_tags->set_object_and_property(this, "multe_animation_tags");
                multe_animation_tags->set_dynamic(true, "get_animation_tags");
                multe_animation_tags->setup(Vector<String>());

                conver_multe_button = memnew(Button);
                conver_multe_button->set_text(L"转换");
                conver_multe_button->connect(SceneStringName(pressed), callable_mp(this, &AnimationProcessPanel::_on_conver_multe_pressed));
                
                vb->add_child(create_line(  multe_path,multe_animation_group,multe_animation_tags));

                vb->add_child(conver_multe_button);

            }

            


        }

    }

}
void AnimationProcessPanel::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_preview_prefab_path", "path"), &AnimationProcessPanel::set_preview_prefab_path);
    ClassDB::bind_method(D_METHOD("get_preview_prefab_path"), &AnimationProcessPanel::get_preview_prefab_path);

    ClassDB::bind_method(D_METHOD("get_animation_groups"), &AnimationProcessPanel::get_animation_groups);
    ClassDB::bind_method(D_METHOD("get_animation_tags"), &AnimationProcessPanel::get_animation_tags);

    ClassDB::bind_method(D_METHOD("set_single_animation_file_path", "path"), &AnimationProcessPanel::set_single_animation_file_path);
    ClassDB::bind_method(D_METHOD("get_single_animation_file_path"), &AnimationProcessPanel::get_single_animation_file_path);

    ClassDB::bind_method(D_METHOD("set_single_animation_group", "group"), &AnimationProcessPanel::set_single_animation_group);
    ClassDB::bind_method(D_METHOD("get_single_animation_group"), &AnimationProcessPanel::get_single_animation_group);
    ClassDB::bind_method(D_METHOD("set_single_animation_tags", "tag"), &AnimationProcessPanel::set_single_animation_tags);
    ClassDB::bind_method(D_METHOD("get_single_animation_tags"), &AnimationProcessPanel::get_single_animation_tags);

    ClassDB::bind_method(D_METHOD("set_multe_animation_file_path", "path"), &AnimationProcessPanel::set_multe_animation_file_path);
    ClassDB::bind_method(D_METHOD("get_multe_animation_file_path"), &AnimationProcessPanel::get_multe_animation_file_path);
    ClassDB::bind_method(D_METHOD("set_multe_animation_group", "group"), &AnimationProcessPanel::set_multe_animation_group);
    ClassDB::bind_method(D_METHOD("get_multe_animation_group"), &AnimationProcessPanel::get_multe_animation_group);
    ClassDB::bind_method(D_METHOD("set_multe_animation_tags", "tag"), &AnimationProcessPanel::set_multe_animation_tags);
    ClassDB::bind_method(D_METHOD("get_multe_animation_tags"), &AnimationProcessPanel::get_multe_animation_tags);

    ADD_PROPERTY(PropertyInfo(Variant::STRING, "preview_prefab_path"), "set_preview_prefab_path", "get_preview_prefab_path");

    ADD_PROPERTY(PropertyInfo(Variant::STRING, "single_animation_file_path"), "set_single_animation_file_path", "get_single_animation_file_path");
    ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "single_animation_group"), "set_single_animation_group", "get_single_animation_group");
    ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "single_animation_tags"), "set_single_animation_tags", "get_single_animation_tags");

    ADD_PROPERTY(PropertyInfo(Variant::STRING, "multe_animation_file_path"), "set_multe_animation_file_path", "get_multe_animation_file_path");
    ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "multe_animation_group"), "set_multe_animation_group", "get_multe_animation_group");
    ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "multe_animation_tags"), "set_multe_animation_tags", "get_multe_animation_tags");


}

void AnimationProcessPanel::set_preview_prefab_path(const String& path) {
    preview_prefab_path = path;
    if(preview != nullptr) {
        preview->set_prefab_path(path);
    }
    save_animation_config();
}
String AnimationProcessPanel::get_preview_prefab_path() {
    return preview_prefab_path;
}

void AnimationProcessPanel::set_single_animation_file_path(const String& path) {
    single_animation_file_path = path;
    save_animation_config();
}
String AnimationProcessPanel::get_single_animation_file_path() {
    return single_animation_file_path;
}
void AnimationProcessPanel::set_single_animation_group(const String& group) {
    single_animation_group_name = group;     
    save_animation_config();   
}
StringName AnimationProcessPanel::get_single_animation_group() {
    return single_animation_group_name;
}
void AnimationProcessPanel::set_single_animation_tags(const String& tag) {
    single_animation_tag_name = tag;
    save_animation_config();
    
}
StringName AnimationProcessPanel::get_single_animation_tags() {
    return single_animation_tag_name;
}


void AnimationProcessPanel::set_multe_animation_file_path(const String& path) {
    multe_animation_file_path = path;
    save_animation_config();
}
String AnimationProcessPanel::get_multe_animation_file_path() {
    return multe_animation_file_path;
}
void AnimationProcessPanel::set_multe_animation_group(const String& group) {
    multe_animation_group_name = group;
    save_animation_config();
}
StringName AnimationProcessPanel::get_multe_animation_group() {
    return multe_animation_group_name;
}
void AnimationProcessPanel::set_multe_animation_tags(const String& tag) {
    multe_animation_tag_name = tag;
    save_animation_config();
}
StringName AnimationProcessPanel::get_multe_animation_tags() {
    return multe_animation_tag_name;
}

Array AnimationProcessPanel::get_animation_groups() {
    Array arr;
    CharacterManager::get_singleton()->get_animation_groups(&arr);
    return arr;
}
Array AnimationProcessPanel::get_animation_tags() {
    Array arr;
    CharacterManager::get_singleton()->get_animation_tags(&arr);
    return arr;
}
void AnimationProcessPanel::save_animation_config() {
    String path = "res://.godot/animation_process_panel_config.json";

    Dictionary dict;
    dict["preview_prefab_path"] = preview_prefab_path;

    dict["single_animation_file_path"] = single_animation_file_path;
    dict["single_animation_group"] = single_animation_group_name;
    dict["single_animation_tags"] = single_animation_tag_name;

    dict["multe_animation_file_path"] = multe_animation_file_path;
    dict["multe_animation_group"] = multe_animation_group_name;
    dict["multe_animation_tags"] = multe_animation_tag_name;

    Ref<FileAccess> file = FileAccess::open(path, FileAccess::WRITE);
    file->store_string(JSON::stringify(dict));
    file->close();
}

void AnimationProcessPanel::load_animation_config() {
    String path = "res://.godot/animation_process_panel_config.json";
    Ref<FileAccess> file = FileAccess::open(path, FileAccess::READ);
    if (file == nullptr) {
        return;
    }
    String json = file->get_as_text();
    file->close();
    Dictionary dict = JSON::parse_string(json);

    preview_prefab_path = dict["preview_prefab_path"];

    single_animation_file_path = dict["single_animation_file_path"];
    single_animation_group_name = dict["single_animation_group"];
    single_animation_tag_name = dict["single_animation_tags"];

    multe_animation_file_path = dict["multe_animation_file_path"];
    multe_animation_group_name = dict["multe_animation_group"];
    multe_animation_tag_name = dict["multe_animation_tags"];
}
