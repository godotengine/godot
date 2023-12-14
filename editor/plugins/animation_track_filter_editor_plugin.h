/**************************************************************************/
/*  animation_track_filter_editor_plugin.h                                */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#ifndef ANIMATION_TRACK_FILTER_EDITOR_PLUGIN_H
#define ANIMATION_TRACK_FILTER_EDITOR_PLUGIN_H

#include "editor/editor_inspector.h"
#include "editor/editor_plugin.h"
#include "scene/gui/dialogs.h"

class AnimationTrackFilter;
class AnimationNode;
class EditorResourcePicker;
class Tree;
class AnimationMixer;
class TreeItem;

class AnimationTrackFilterEditDialog : public AcceptDialog {
	GDCLASS(AnimationTrackFilterEditDialog, AcceptDialog)

	class TrackItem : public HBoxContainer {
		GDCLASS(TrackItem, HBoxContainer)

		NodePath track_path;

		SpinBox *amount = nullptr;
		TextureRect *icon = nullptr;
		Label *text = nullptr;

		void _amount_changed(double p_value);

	protected:
		static void _bind_methods();

	public:
		void set_editable(bool p_editable);
		void set_icon(const Ref<Texture2D> &p_icon);
		void set_text(const String &p_text);
		void set_amount(float p_amount);
		void set_track_path(const NodePath &p_track_path);
		void set_as_uneditable_track(bool p_editable);
		float get_amount() const;

		NodePath get_track_path() const;
		void set_amount_changed_callback();

		TrackItem();
	};

	Ref<StyleBox> focus_style;
	Control *tree_focus_rect;

	Tree *filter_tree;
	bool read_only = false;
	bool updating = false;

	HBoxContainer *tool_bar = nullptr;
	AnimationMixer *anim_player = nullptr;
	Ref<AnimationTrackFilter> filter;

	void _tree_draw();
	void _tree_item_draw(TreeItem *p_item, const Rect2 &p_rect) const;
	void _track_item_amount_changed(const NodePath &p_track_path, double p_amount);

	void hide_invisiable_track_items();
	TrackItem *get_or_create_track_items_and_setup_tree_item(TreeItem *p_item);

	HashMap<TreeItem *, TrackItem *> track_items;

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	void set_read_only(bool p_read_only);
	bool is_read_only() const;
	bool update_filters(class AnimationMixer *p_player, const Ref<AnimationTrackFilter> &p_filter);

	HBoxContainer *get_tool_bar() const;

	AnimationTrackFilterEditDialog();
};

class AnimationNodeFilterTracksEditor : public EditorProperty {
	GDCLASS(AnimationNodeFilterTracksEditor, EditorProperty);

	EditorResourcePicker *picker = nullptr;

	Button *edit_btn = nullptr;
	AnimationTrackFilterEditDialog *edit_dialog = nullptr;

	void _edit_requested();

	bool update_filters();

public:
	virtual void update_property() override;

	AnimationNodeFilterTracksEditor();
};

class AnimationTrackFilterEditorInspectorPlugin : public EditorInspectorPlugin {
	GDCLASS(AnimationTrackFilterEditorInspectorPlugin, EditorInspectorPlugin);

public:
	virtual bool can_handle(Object *p_object) override;
	virtual bool parse_property(Object *p_object, const Variant::Type p_type, const String &p_path, const PropertyHint p_hint, const String &p_hint_text, const BitField<PropertyUsageFlags> p_usage, const bool p_wide = false) override;
};

class AnimationTrackFilterEditorPlugin : public EditorPlugin {
	GDCLASS(AnimationTrackFilterEditorPlugin, EditorPlugin);

public:
	AnimationTrackFilterEditorPlugin();
};

#endif // ANIMATION_TRACK_FILTER_EDITOR_PLUGIN_H