#pragma once

#include "scene/gui/box_container.h"
#include "editor/editor_properties.h"
#include "editor/editor_resource_picker.h"
#include "../../logic/character_manager.h"
#include "resource_editor_tool_item.h"

#include "../animator_node_preview.h"

class AnimationProcessPanel : public VBoxContainer {
    GDCLASS(AnimationProcessPanel, VBoxContainer);
    
public:
    HBoxContainer* create_line(Control* control , bool is_side_separator = false) ;
    HBoxContainer* create_line(Control* control0 , Control* control1,  bool is_side_separator = false);

    HBoxContainer* create_line(Control* control0 , Control* control1, Control* control2,  bool is_side_separator = false) ;

    HBoxContainer* create_line(Control* control0 , Control* control1, Control* control2, Control* control3,  bool is_side_separator = false) ;

    AnimationProcessPanel();
    static void _bind_methods() ;
    void set_preview_prefab_path(const String& path) ;
    String get_preview_prefab_path() ;
    void set_single_animation_file_path(const String& path) ;
    String get_single_animation_file_path() ;
    void set_single_animation_group(const String& group) ;
    StringName get_single_animation_group() ;
    void set_single_animation_tags(const String& tag);

    StringName get_single_animation_tags() ;


    void set_multe_animation_file_path(const String& path);
    String get_multe_animation_file_path() ;
    void set_multe_animation_group(const String& group) ;
    StringName get_multe_animation_group() ;
    void set_multe_animation_tags(const String& tag) ;
    StringName get_multe_animation_tags() ;

    Array get_animation_groups() ;
    Array get_animation_tags();
protected:
    void _on_conver_single_pressed() ;
    void _on_conver_multe_pressed() ;

    void save_animation_config() ;

    void load_animation_config() ;

    
    void editor_build_animation_form_path(String p_file_path, const StringName& animation_group, const StringName& animation_tag);

    void editor_convert_animations(String p_file_path, const StringName& animation_group, const StringName& animation_tag);
public:
    String preview_prefab_path;
    EditorPropertyPath* property_preview_prefab_path = nullptr;
    // 预览预制体查看面板
    AnimationNodePreview* preview = nullptr;

    String single_animation_file_path;
    StringName single_animation_group_name;
    StringName single_animation_tag_name;
    EditorPropertyTextEnum* single_animation_group = nullptr;
    EditorPropertyTextEnum* single_animation_tags = nullptr;
    EditorPropertyPath* single_path = nullptr;
    Button* conver_single_button = nullptr;

    String multe_animation_file_path;
    StringName multe_animation_group_name;
    StringName multe_animation_tag_name;
    EditorPropertyPath* multe_path = nullptr;
    EditorPropertyTextEnum* multe_animation_group = nullptr;
    EditorPropertyTextEnum* multe_animation_tags = nullptr;
    Button* conver_multe_button = nullptr;
};
class AnimationProcessPanellItem : public ResourceEditorToolItem {
    GDCLASS(AnimationProcessPanellItem,ResourceEditorToolItem)
    static void _bind_methods() {}

    virtual String get_name() const { return String(L"动画资源导入"); }
    virtual Control *get_control() {

		return Object::cast_to< AnimationProcessPanel>(ObjectDB::get_instance(control_id));
	}
	AnimationProcessPanellItem() {
		AnimationProcessPanel* c = memnew(AnimationProcessPanel);
		control_id = c->get_instance_id();
	}
	~AnimationProcessPanellItem() {
		Control* c = get_control();
		if (c != nullptr) {
			c->queue_free();
		}
	}
	ObjectID control_id;
};
