/*************************************************************************/
/*  shader_file_editor_plugin.h                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef SHADER_FILE_EDITOR_PLUGIN_H
#define SHADER_FILE_EDITOR_PLUGIN_H

#include "editor/code_editor.h"
#include "editor/editor_plugin.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/rich_text_label.h"
#include "scene/gui/tab_container.h"
#include "scene/gui/text_edit.h"
#include "scene/main/timer.h"
#include "servers/rendering/rendering_device_binds.h"

class ShaderFileEditor : public PanelContainer {
	GDCLASS(ShaderFileEditor, PanelContainer);

	Ref<RDShaderFile> shader_file;

	HBoxContainer *stage_hb;
	ItemList *versions;
	Button *stages[RD::SHADER_STAGE_MAX];
	RichTextLabel *error_text;

	void _update_version(const StringName &p_version_txt, const RenderingDevice::ShaderStage p_stage);
	void _version_selected(int p_stage);
	void _editor_settings_changed();

	void _update_options();
	void _shader_changed();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	static ShaderFileEditor *singleton;
	void edit(const Ref<RDShaderFile> &p_shader);

	ShaderFileEditor(EditorNode *p_node);
};

class ShaderFileEditorPlugin : public EditorPlugin {
	GDCLASS(ShaderFileEditorPlugin, EditorPlugin);

	ShaderFileEditor *shader_editor;
	EditorNode *editor;
	Button *button;

public:
	virtual String get_name() const override { return "ShaderFile"; }
	bool has_main_screen() const override { return false; }
	virtual void edit(Object *p_object) override;
	virtual bool handles(Object *p_object) const override;
	virtual void make_visible(bool p_visible) override;

	ShaderFileEditor *get_shader_editor() const { return shader_editor; }

	ShaderFileEditorPlugin(EditorNode *p_node);
	~ShaderFileEditorPlugin();
};

#endif // SHADER_FILE_EDITOR_PLUGIN_H
