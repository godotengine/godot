/**************************************************************************/
/*  import_pipeline_editor_plugin.h                                       */
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

#ifndef IMPORT_PIPELINE_EDITOR_PLUGIN_H
#define IMPORT_PIPELINE_EDITOR_PLUGIN_H

#include "core/io/resource_importer.h"
#include "editor/editor_file_dialog.h"
#include "editor/editor_plugin.h"
#include "editor/editor_quick_open.h"
#include "editor/editor_scale.h"
#include "editor/import/import_pipeline.h"
#include "editor/import/import_pipeline_step.h"
#include "editor/plugins/node_3d_editor_gizmos.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/graph_edit.h"
#include "scene/gui/popup_menu.h"
#include "scene/gui/rich_text_label.h"
#include "scene/gui/spin_box.h"
#include "scene/gui/split_container.h"

class NodeData;

//todo: reorder functions (with cpp)
class ImportPipelineEditor : public Control {
	GDCLASS(ImportPipelineEditor, Control);
	friend class NodeData;

	static ImportPipelineEditor *singleton;

	EditorFileDialog *source_dialog;
	EditorFileDialog *load_dialog;
	EditorFileDialog *script_dialog;
	ConfirmationDialog *overwrite_dialog;
	SubViewport *sub_viewport;

	PopupMenu *add_menu;
	int plugins_state = 0;
	GraphEdit *graph;
	Label *name_label;

	Ref<ImportPipeline> pipeline_data;
	enum PipelineState {
		PIPELINE_NOTHING,
		PIPELINE_SAVED,
		PIPELINE_UNSAVED,
	} pipeline_state = PIPELINE_NOTHING;

	String path;
	String load_path;

	EditorInspector *preview_inspector;
	String preview_node;
	int preview_idx;
	Ref<Resource> preview_resource;

	EditorInspector *settings_inspector;
	String settings_node;

	HashMap<String, NodeData *> steps;
	String importer_node_name;

	void _reset_pipeline();

	StringName _create_importer_node(Vector2 p_position, const String &p_path);
	StringName _create_overwritter_node(Vector2 p_position);
	StringName _create_loader_node(Vector2 p_position, const String &p_path);
	StringName _create_saver_node(Vector2 p_positionm, const String &p_name);
	StringName _create_node(Ref<ImportPipelineStep> p_step, Vector2 p_position, bool p_closable = true);

	void _import_plugins_changed();
	void _connection_request(const String &p_from, int p_from_index, const String &p_to, int p_to_index);
	void _node_selected(Node *p_node);
	void _node_deselected(Node *p_node);
	void _result_button_pressed(StringName p_node, int p_idx);
	Ref<Resource> _get_result(StringName p_node, int p_idx);
	void _remove_node(StringName p_node);
	void _create_step(int p_idx, PopupMenu *p_menu);
	void _create_script_step(const String &p_path);
	void _create_special_step(int p_idx);
	void _create_add_popup(Vector2 position);
	void _update_node(StringName p_node);
	bool _has_cycle(const String &current, const String &target);
	void _update_preview();
	void _update_settings(const String &p_node);
	void _settings_property_changed(const String &p_name);
	void _add_source(const String &p_path);
	Vector2 _get_creation_position();
	void _save();
	void _load();
	void _save_with_path(const String &p_path);
	void _node_moved();

protected:
	void _notification(int p_what);

public:
	static ImportPipelineEditor *get_singleton() { return singleton; }

	void change(const String &p_path);

	Dictionary get_state() const;
	void set_state(const Dictionary &p_state);

	ImportPipelineEditor();
	~ImportPipelineEditor();
};

class ImportPipelineEditorPlugin : public EditorPlugin {
	GDCLASS(ImportPipelineEditorPlugin, EditorPlugin);

public:
	virtual String get_name() const override { return "ImportPipeline"; }
	bool has_main_screen() const override { return true; }
	void make_visible(bool p_visible) override;

	virtual void edit(Object *p_object) override;
	virtual bool handles(Object *p_object) const override;

	Dictionary get_state() const override;
	void set_state(const Dictionary &p_state) override;

	ImportPipelineEditorPlugin();
};

#endif // IMPORT_PIPELINE_EDITOR_PLUGIN_H
