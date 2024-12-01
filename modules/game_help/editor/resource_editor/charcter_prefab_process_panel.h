#pragma once


#include "scene/gui/box_container.h"
#include "editor/editor_properties.h"
#include "editor/editor_resource_picker.h"
#include "../../logic/character_manager.h"
#include "../../logic/character_shape/character_body_prefab.h"
#include "resource_editor_tool_item.h"
#include "scene_view_panel.h"

class CharacterPrefabProcessPanel : public VBoxContainer {
    GDCLASS(CharacterPrefabProcessPanel, VBoxContainer);
    
public:
    HBoxContainer* create_line(Control* control , bool is_side_separator = false) ;
    HBoxContainer* create_line(Control* control0 , Control* control1,  bool is_side_separator = false);

    HBoxContainer* create_line(Control* control0 , Control* control1, Control* control2,  bool is_side_separator = false) ;

    HBoxContainer* create_line(Control* control0 , Control* control1, Control* control2, Control* control3,  bool is_side_separator = false) ;

    CharacterPrefabProcessPanel();
    static void _bind_methods() ;
    void set_preview_mesh_path(const String& path) ;
    String get_preview_mesh_path() ;
    
    void set_single_charcter_prefab_group(const String& group) ;
    StringName get_single_charcter_prefab_group() ;



    void set_multe_charcter_prefab_file_path(const String& path);
    String get_multe_charcter_prefab_file_path() ;
    void set_multe_charcter_prefab_group(const String& group) ;
    StringName get_multe_charcter_prefab_group() ;

    Array get_charcter_prefab_groups() ;
protected:
    void _on_conver_single_pressed() ;
    void _on_conver_multe_pressed() ;

    void save_charcter_prefab_config() ;

    void load_charcter_prefab_config() ;

    
    void editor_build_prefab_form_path(String p_file_path, const StringName& animation_group);

    void editor_convert_prefab(String p_file_path, const StringName& animation_group);

        
    Ref<CharacterBodyPrefab> build_prefab(const String& mesh_path, const StringName& animation_group,bool p_is_skeleton_human);


public:
    String preview_mesh_path;
    EditorPropertyPath* property_preview_mesh_path = nullptr;
    // 预览预制体查看面板
    SceneViewPanel* preview = nullptr;

    StringName single_charcter_prefab_group_name;
    EditorPropertyTextEnum* single_charcter_prefab_group = nullptr;
    Button* conver_single_button = nullptr;

    String multe_charcter_prefab_file_path;
    StringName multe_charcter_prefab_group_name;
    EditorPropertyPath* multe_path = nullptr;
    EditorPropertyTextEnum* multe_charcter_prefab_group = nullptr;
    Button* conver_multe_button = nullptr;
};
class CharacterPrefabProcessPanellItem : public ResourceEditorToolItem {
    GDCLASS(CharacterPrefabProcessPanellItem,ResourceEditorToolItem)
    static void _bind_methods() {}

    virtual String get_name() const { return String(L"角色预制体导入"); }
    virtual Control *get_control() {

		return Object::cast_to< CharacterPrefabProcessPanel>(ObjectDB::get_instance(control_id));
	}
	CharacterPrefabProcessPanellItem() {
		CharacterPrefabProcessPanel* c = memnew(CharacterPrefabProcessPanel);
		control_id = c->get_instance_id();
	}
	~CharacterPrefabProcessPanellItem() {
		Control* c = get_control();
		if (c != nullptr) {
			c->queue_free();
		}
	}
	ObjectID control_id;
};
