/*************************************************************************/
/*  skeleton_retarget_editor_plugin.h                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#ifndef SKELETON_RETARGET_EDITOR_PLUGIN_H
#define SKELETON_RETARGET_EDITOR_PLUGIN_H

#include "editor/editor_node.h"
#include "editor/editor_plugin.h"
#include "editor/editor_properties.h"
#include "scene/animation/skeleton_retarget.h"
#include "scene/gui/color_rect.h"
#include "scene/gui/dialogs.h"
#include "scene/resources/texture.h"

// Bone picker

class Skeleton3DBonePicker : public AcceptDialog {
	GDCLASS(Skeleton3DBonePicker, AcceptDialog);

	Tree *bones;

public:
	void popup_bones_tree(Skeleton3D *p_skeleton, const Size2i &p_minsize = Size2i());
	bool has_selected_bone();
	StringName get_selected_bone();

protected:
	void _notification(int p_what);
	static void _bind_methods();

private:
	void create_editors();
	void create_bones_tree(Skeleton3D *p_skeleton);

public:
	Skeleton3DBonePicker();
	~Skeleton3DBonePicker();
};

// Base classes

class RetargetEditorForm : public VBoxContainer {
	GDCLASS(RetargetEditorForm, VBoxContainer);

	Button *button_submit;

protected:
	void _notification(int p_what);
	virtual void create_editors();
	virtual void submit();

private:
	void create_button_submit();

public:
	RetargetEditorForm();
	~RetargetEditorForm();
};

class RetargetEditorItem : public VBoxContainer {
	GDCLASS(RetargetEditorItem, VBoxContainer);

	int index = 0;
	VBoxContainer *vbox;
	Button *button_remove;

public:
	VBoxContainer *get_vbox();

protected:
	void _notification(int p_what);
	static void _bind_methods();

private:
	void create_editors();
	void fire_remove();

public:
	RetargetEditorItem(const int p_index);
	~RetargetEditorItem();
};

// Mapper base

class MapperButton : public TextureButton {
	GDCLASS(MapperButton, TextureButton);

	StringName name;

public:
	enum MapperState {
		MAPPER_STATE_UNSET,
		MAPPER_STATE_SET,
		MAPPER_STATE_ERROR
	};

	void set_state(MapperState p_state);
	StringName get_name();

protected:
	void _notification(int p_what);

private:
	bool selected = false;
	MapperState state = MAPPER_STATE_UNSET;

	TextureRect *circle;
	void fetch_textures();

public:
	MapperButton(const StringName &p_name, bool p_selected, MapperState p_state);
	~MapperButton();
};

class RetargetEditorMapperItem : public VBoxContainer {
	GDCLASS(RetargetEditorMapperItem, VBoxContainer);

	Button *button_enable;
	Button *button_remove;

protected:
	int button_id = -1;
	StringName key_name;
	bool enabled;

	VBoxContainer *inputs_vbox;

	void _notification(int p_what);
	static void _bind_methods();
	virtual void _value_changed(const String &p_property, Variant p_value, const String &p_name, bool p_changing);
	virtual void create_editors();

private:
	void create_buttons();
	void fire_remove();
	void fire_enable();

public:
	void assign_button_id(int p_button_id);

	RetargetEditorMapperItem(const StringName &p_name = StringName(), const bool p_enabled = false);
	~RetargetEditorMapperItem();
};

class RetargetEditorMapper : public VBoxContainer {
	GDCLASS(RetargetEditorMapper, VBoxContainer);

public:
	void set_profile(const Ref<RetargetProfile> &p_profile);
	virtual void clear_items();
	virtual void recreate_items();

protected:
	Ref<RetargetProfile> profile;

	EditorInspectorSection *section_unprofiled;

	Vector<RetargetEditorMapperItem *> mapper_items;
	Vector<RetargetEditorMapperItem *> unprofiled_items;

	VBoxContainer *map_vbox;
	VBoxContainer *unprofiled_vbox;

	void _notification(int p_what);
	static void _bind_methods();
	virtual void _value_changed(const String &p_property, Variant p_value, const String &p_name, bool p_changing);

	// For rich profile.
	bool use_rich_profile = false;
	int current_group = 0;
	int current_intermediate_bone = 0;

	AspectRatioContainer *rich_profile_field;
	EditorPropertyEnum *profile_group_selector;
	ColorRect *profile_bg;
	TextureRect *profile_texture;
	Vector<MapperButton *> mapper_buttons;

	void set_current_group(int p_group);
	int get_current_group() const;
	void set_current_intermediate_bone(int p_bone);
	int get_current_intermediate_bone() const;

	void update_group_ids();
	void recreate_rich_editor();

	virtual MapperButton::MapperState get_mapper_state(const StringName &p_bone_name);
	void set_mapper_state(int p_bone, MapperButton::MapperState p_state);

private:
	void create_editors();
	void create_rich_editor();

public:
	RetargetEditorMapper();
	~RetargetEditorMapper();
};

// Retarget profile

class RetargetProfileEditorForm : public RetargetEditorForm {
	GDCLASS(RetargetProfileEditorForm, VBoxContainer);
	EditorPropertyText *key_name;
	StringName prop_key_name = StringName();

protected:
	static void _bind_methods();
	void _value_changed(const String &p_property, Variant p_value, const String &p_name, bool p_changing);
	virtual void create_editors() override;
	virtual void submit() override;

private:
	void set_key_name(const StringName &p_key_name);
	StringName get_key_name() const;

public:
	RetargetProfileEditorForm();
	~RetargetProfileEditorForm();
};

class RetargetProfileEditor : public VBoxContainer {
	GDCLASS(RetargetProfileEditor, VBoxContainer);

	RetargetProfile *retarget_profile;

	Vector<RetargetEditorItem *> intermediate_bones;
	Vector<EditorPropertyText *> intermediate_bone_names;

	VBoxContainer *imb_vbox;
	RetargetProfileEditorForm *imb_form;

protected:
	void _notification(int p_what);
	void _value_changed(const String &p_property, Variant p_value, const String &p_name, bool p_changing);
	void _update_intermediate_bone_property();
	void _add_intermediate_bone(const StringName &p_bone_name);
	void _remove_intermediate_bone(const int p_id);

	void create_editors();
	void recreate_items();
	void clear_items();

public:
	RetargetProfileEditor(RetargetProfile *p_retarget_profile);
	~RetargetProfileEditor();
};

class RetargetRichProfileEditor : public VBoxContainer {
	GDCLASS(RetargetRichProfileEditor, VBoxContainer);

	RetargetRichProfile *retarget_profile;

	EditorInspectorSection *section_grp;
	EditorInspectorSection *section_imb;

	Vector<RetargetEditorItem *> groups;
	Vector<EditorPropertyText *> group_names;
	Vector<EditorPropertyResource *> group_textures;

	VBoxContainer *grp_vbox;
	RetargetProfileEditorForm *grp_form;

	Vector<RetargetEditorItem *> intermediate_bones;
	Vector<EditorPropertyText *> intermediate_bone_names;
	Vector<EditorPropertyVector2 *> intermediate_bone_handle_offsets;
	Vector<EditorPropertyEnum *> intermediate_bone_group_ids;

	VBoxContainer *imb_vbox;
	RetargetProfileEditorForm *imb_form;

protected:
	void _notification(int p_what);
	void _value_changed(const String &p_property, Variant p_value, const String &p_name, bool p_changing);
	void _update_group_ids();
	void _update_group_property();
	void _update_intermediate_bone_property();
	void _add_intermediate_bone(const StringName &p_bone_name);
	void _remove_intermediate_bone(const int p_id);
	void _add_group(const StringName &p_group_name);
	void _remove_group(const int p_id);

	void create_editors();
	void recreate_items();
	void clear_items();

	void redraw();

public:
	RetargetRichProfileEditor(RetargetRichProfile *p_retarget_profile);
	~RetargetRichProfileEditor();
};

class EditorInspectorPluginRetargetProfile : public EditorInspectorPlugin {
	GDCLASS(EditorInspectorPluginRetargetProfile, EditorInspectorPlugin);
	Control *rp_editor;

public:
	virtual bool can_handle(Object *p_object) override;
	virtual void parse_begin(Object *p_object) override;
};

class RetargetProfileEditorPlugin : public EditorPlugin {
	GDCLASS(RetargetProfileEditorPlugin, EditorPlugin);

public:
	virtual String get_name() const override { return "RetargetProfile"; }
	RetargetProfileEditorPlugin();
};

// Retarget source setting

class RetargetBoneOptionMapperItem : public RetargetEditorMapperItem {
	GDCLASS(RetargetBoneOptionMapperItem, RetargetEditorMapperItem);
	RetargetBoneOption *retarget_option;

	PackedStringArray retarget_mode_arr;
	EditorPropertyEnum *retarget_mode;

protected:
	void _notification(int p_what);
	static void _bind_methods();
	virtual void create_editors() override;
	void _update_property();

private:
	virtual void _value_changed(const String &p_property, Variant p_value, const String &p_name, bool p_changing) override;

public:
	RetargetBoneOptionMapperItem(RetargetBoneOption *p_retarget_option, const StringName &p_name, const bool p_enabled);
	~RetargetBoneOptionMapperItem();
};

class RetargetBoneOptionMapper : public RetargetEditorMapper {
	GDCLASS(RetargetBoneOptionMapper, RetargetEditorMapper);
	RetargetBoneOption *retarget_option;

public:
	virtual void clear_items() override;
	virtual void recreate_items() override;

protected:
	void _notification(int p_what);
	void _add_item(const StringName &p_intermediate_bone_name);
	void _remove_item(const StringName &p_intermediate_bone_name);
	void _update_mapper_state();

	virtual MapperButton::MapperState get_mapper_state(const StringName &p_intermediate_bone_name) override;

public:
	RetargetBoneOptionMapper(RetargetBoneOption *p_retarget_option);
	~RetargetBoneOptionMapper();
};

class RetargetBoneOptionEditor : public VBoxContainer {
	GDCLASS(RetargetBoneOptionEditor, VBoxContainer);

	Ref<RetargetProfile> profile;
	RetargetBoneOption *retarget_option;

	RetargetBoneOptionMapper *mapper;

public:
	void set_profile(const Ref<RetargetProfile> &p_profile);
	Ref<RetargetProfile> get_profile() const;
	void fetch_objects();
	void redraw();

protected:
	void _notification(int p_what);

private:
	void clear_editors();
	void create_editors();

public:
	RetargetBoneOptionEditor(RetargetBoneOption *p_retarget_option);
	~RetargetBoneOptionEditor();
};

class EditorInspectorPluginRetargetBoneOption : public EditorInspectorPlugin {
	GDCLASS(EditorInspectorPluginRetargetBoneOption, EditorInspectorPlugin);
	RetargetBoneOptionEditor *rs_editor;

public:
	virtual bool can_handle(Object *p_object) override;
	virtual void parse_begin(Object *p_object) override;
};

class RetargetBoneOptionEditorPlugin : public EditorPlugin {
	GDCLASS(RetargetBoneOptionEditorPlugin, EditorPlugin);

public:
	virtual String get_name() const override { return "RetargetBoneOption"; }
	RetargetBoneOptionEditorPlugin();
};

// Retarget target setting

class RetargetBoneMapMapperItem : public RetargetEditorMapperItem {
	GDCLASS(RetargetBoneMapMapperItem, RetargetEditorMapperItem);
	RetargetBoneMap *retarget_map;

	EditorPropertyText *bone_name;
	Button *button_pick;

protected:
	void _notification(int p_what);
	static void _bind_methods();
	virtual void create_editors() override;
	void _update_property();

private:
	void fire_pick();
	virtual void _value_changed(const String &p_property, Variant p_value, const String &p_name, bool p_changing) override;

public:
	RetargetBoneMapMapperItem(RetargetBoneMap *p_retarget_map, const StringName &p_name, const bool p_enabled);
	~RetargetBoneMapMapperItem();
};

class RetargetBoneMapMapper : public RetargetEditorMapper {
	GDCLASS(RetargetBoneMapMapper, RetargetEditorMapper);
	RetargetBoneMap *retarget_map;

public:
	void set_skeleton(Skeleton3D *p_skeleton);

	virtual void clear_items() override;
	virtual void recreate_items() override;

protected:
	Skeleton3D *skeleton;
	Skeleton3DBonePicker *picker;
	StringName picker_key_name;

	void apply_picker_selection();
	void _add_item(const StringName &p_intermediate_bone_name);
	void _remove_item(const StringName &p_intermediate_bone_name);
	void _pick_bone(const StringName &p_intermediate_bone_name);
	void _update_mapper_state();

	virtual MapperButton::MapperState get_mapper_state(const StringName &p_intermediate_bone_name) override;

	void _notification(int p_what);

public:
	RetargetBoneMapMapper(RetargetBoneMap *p_retarget_map);
	~RetargetBoneMapMapper();
};

class RetargetBoneMapEditor : public VBoxContainer {
	GDCLASS(RetargetBoneMapEditor, VBoxContainer);

	Skeleton3D *skeleton;
	Ref<RetargetProfile> profile;

	RetargetBoneMap *retarget_map;

	RetargetBoneMapMapper *mapper;

public:
	void set_skeleton(Skeleton3D *p_skeleton);
	Skeleton3D *get_skeleton();
	void set_profile(const Ref<RetargetProfile> &p_profile);
	Ref<RetargetProfile> get_profile() const;
	void fetch_objects();
	void redraw();

protected:
	void _notification(int p_what);

private:
	void clear_editors();
	void create_editors();

public:
	RetargetBoneMapEditor(RetargetBoneMap *p_retarget_map);
	~RetargetBoneMapEditor();
};

class EditorInspectorPluginRetargetBoneMap : public EditorInspectorPlugin {
	GDCLASS(EditorInspectorPluginRetargetBoneMap, EditorInspectorPlugin);
	RetargetBoneMapEditor *rt_editor;

public:
	virtual bool can_handle(Object *p_object) override;
	virtual void parse_begin(Object *p_object) override;
};

class RetargetBoneMapEditorPlugin : public EditorPlugin {
	GDCLASS(RetargetBoneMapEditorPlugin, EditorPlugin);

public:
	virtual String get_name() const override { return "RetargetBoneMap"; }
	RetargetBoneMapEditorPlugin();
};

#endif // SKELETON_RETARGET_EDITOR_PLUGIN_H
