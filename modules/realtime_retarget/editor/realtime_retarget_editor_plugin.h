/*************************************************************************/
/*  realtime_retarget_editor_plugin.h                                    */
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

#ifndef REALTIME_RETARGET_EDITOR_PLUGIN_H
#define REALTIME_RETARGET_EDITOR_PLUGIN_H

#include "editor/plugins/editor_plugin.h"
#include "editor/editor_properties_array_dict.h"
#include "../src/retarget_animation_player.h"

class RetargetAnimationPlayerEditor : public VBoxContainer {
	GDCLASS(RetargetAnimationPlayerEditor, VBoxContainer);

	RetargetAnimationPlayer *rap = nullptr;

	Button *warning = nullptr;
	AcceptDialog *warning_dialog = nullptr;

	Label *animation_name = nullptr;
	EditorPropertyDictionary *meta_inspector = nullptr;

	Vector<String> state_names; // To convert enum to string.

	void _update_editor(String p_animation_name = "");
	void _warning_pressed();

protected:
	void _notification(int p_what);

public:
	RetargetAnimationPlayerEditor(RetargetAnimationPlayer *p_retarget_animation_player);
	~RetargetAnimationPlayerEditor();
};

class EditorInspectorPluginRetargetAnimationPlayer : public EditorInspectorPlugin {
	GDCLASS(EditorInspectorPluginRetargetAnimationPlayer, EditorInspectorPlugin);
	RetargetAnimationPlayerEditor *editor = nullptr;

public:
	virtual bool can_handle(Object *p_object) override;
	virtual void parse_begin(Object *p_object) override;
};

class RetargetAnimationPlayerEditorPlugin : public EditorPlugin {
	GDCLASS(RetargetAnimationPlayerEditorPlugin, EditorPlugin);

	EditorInspectorPluginRetargetAnimationPlayer *rap_plugin = nullptr;

public:
	bool has_main_screen() const override { return false; }
	virtual bool handles(Object *p_object) const override;

	virtual String get_name() const override { return "RetargetAnimationPlayer"; }

	RetargetAnimationPlayerEditorPlugin();
};

class RealtimeRetargetEditorPlugin : public EditorPlugin {
	GDCLASS(RealtimeRetargetEditorPlugin, EditorPlugin);

public:
	RealtimeRetargetEditorPlugin();
};

#endif // REALTIME_RETARGET_EDITOR_PLUGIN_H
