/*************************************************************************/
/*  bone_map_editor_plugin.h                                             */
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

#ifndef BONE_MAP_EDITOR_H
#define BONE_MAP_EDITOR_H

#include "editor/editor_node.h"
#include "editor/editor_plugin.h"
#include "editor/editor_properties.h"
#include "scene/3d/skeleton_3d.h"
#include "scene/gui/color_rect.h"
#include "scene/gui/dialogs.h"
#include "scene/resources/bone_map.h"
#include "scene/resources/texture.h"

class BoneMapperButton : public TextureButton {
	GDCLASS(BoneMapperButton, TextureButton);

public:
	enum BoneMapState {
		BONE_MAP_STATE_UNSET,
		BONE_MAP_STATE_SET,
		BONE_MAP_STATE_ERROR
	};

private:
	StringName profile_bone_name;
	bool selected = false;

	TextureRect *circle;

	void fetch_textures();

protected:
	void _notification(int p_what);

public:
	StringName get_profile_bone_name() const;
	void set_state(BoneMapState p_state);

	BoneMapperButton(const StringName p_profile_bone_name, bool p_selected);
	~BoneMapperButton();
};

class BoneMapperItem : public VBoxContainer {
	GDCLASS(BoneMapperItem, VBoxContainer);

	int button_id = -1;
	StringName profile_bone_name;

	PackedStringArray skeleton_bone_names;
	Ref<BoneMap> bone_map;

	EditorPropertyTextEnum *skeleton_bone_selector;

	void _update_property();

protected:
	void _notification(int p_what);
	static void _bind_methods();
	virtual void _value_changed(const String &p_property, Variant p_value, const String &p_name, bool p_changing);
	virtual void create_editor();

public:
	void assign_button_id(int p_button_id);

	BoneMapperItem(Ref<BoneMap> &p_bone_map, PackedStringArray p_skeleton_bone_names, const StringName &p_profile_bone_name = StringName());
	~BoneMapperItem();
};

class BoneMapper : public VBoxContainer {
	GDCLASS(BoneMapper, VBoxContainer);

	Skeleton3D *skeleton;
	Ref<BoneMap> bone_map;

	Vector<BoneMapperItem *> bone_mapper_items;

	VBoxContainer *mapper_item_vbox;
	HSeparator *separator;

	int current_group_idx = 0;
	int current_bone_idx = -1;

	AspectRatioContainer *bone_mapper_field;
	EditorPropertyEnum *profile_group_selector;
	ColorRect *profile_bg;
	TextureRect *profile_texture;
	Vector<BoneMapperButton *> bone_mapper_buttons;

	void create_editor();
	void recreate_editor();
	void clear_items();
	void recreate_items();
	void update_group_idx();
	void _update_state();

protected:
	void _notification(int p_what);
	static void _bind_methods();
	virtual void _value_changed(const String &p_property, Variant p_value, const String &p_name, bool p_changing);

public:
	void set_current_group_idx(int p_group_idx);
	int get_current_group_idx() const;
	void set_current_bone_idx(int p_bone_idx);
	int get_current_bone_idx() const;

	BoneMapper(Skeleton3D *p_skeleton, Ref<BoneMap> &p_bone_map);
	~BoneMapper();
};

class BoneMapEditor : public VBoxContainer {
	GDCLASS(BoneMapEditor, VBoxContainer);

	Skeleton3D *skeleton;
	Ref<BoneMap> bone_map;
	BoneMapper *bone_mapper;

	void fetch_objects();
	void clear_editors();
	void create_editors();

protected:
	void _notification(int p_what);

public:
	BoneMapEditor(Ref<BoneMap> &p_bone_map);
	~BoneMapEditor();
};

class EditorInspectorPluginBoneMap : public EditorInspectorPlugin {
	GDCLASS(EditorInspectorPluginBoneMap, EditorInspectorPlugin);
	BoneMapEditor *editor;

public:
	virtual bool can_handle(Object *p_object) override;
	virtual void parse_begin(Object *p_object) override;
};

class BoneMapEditorPlugin : public EditorPlugin {
	GDCLASS(BoneMapEditorPlugin, EditorPlugin);

public:
	virtual String get_name() const override { return "BoneMap"; }
	BoneMapEditorPlugin();
};

#endif // BONE_MAP_EDITOR_H
