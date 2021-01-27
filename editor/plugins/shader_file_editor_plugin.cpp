/*************************************************************************/
/*  shader_file_editor_plugin.cpp                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "shader_file_editor_plugin.h"

#include "core/io/resource_loader.h"
#include "core/io/resource_saver.h"
#include "core/os/keyboard.h"
#include "core/os/os.h"
#include "editor/editor_node.h"
#include "editor/editor_scale.h"
#include "editor/editor_settings.h"
#include "editor/property_editor.h"
#include "servers/display_server.h"
#include "servers/rendering/shader_types.h"

/*** SHADER SCRIPT EDITOR ****/

/*** SCRIPT EDITOR ******/

void ShaderFileEditor::_update_version(const StringName &p_version_txt, const RD::ShaderStage p_stage) {
}

void ShaderFileEditor::_version_selected(int p_option) {
	int c = versions->get_current();
	StringName version_txt = versions->get_item_metadata(c);

	RD::ShaderStage stage = RD::SHADER_STAGE_MAX;
	int first_found = -1;

	Ref<RDShaderBytecode> bytecode = shader_file->get_bytecode(version_txt);
	ERR_FAIL_COND(bytecode.is_null());

	for (int i = 0; i < RD::SHADER_STAGE_MAX; i++) {
		if (bytecode->get_stage_bytecode(RD::ShaderStage(i)).is_empty() && bytecode->get_stage_compile_error(RD::ShaderStage(i)) == String()) {
			stages[i]->set_icon(Ref<Texture2D>());
			continue;
		}

		Ref<Texture2D> icon;
		if (bytecode->get_stage_compile_error(RD::ShaderStage(i)) != String()) {
			icon = get_theme_icon("ImportFail", "EditorIcons");
		} else {
			icon = get_theme_icon("ImportCheck", "EditorIcons");
		}
		stages[i]->set_icon(icon);

		if (first_found == -1) {
			first_found = i;
		}

		if (stages[i]->is_pressed()) {
			stage = RD::ShaderStage(i);
			break;
		}
	}

	error_text->clear();

	if (stage == RD::SHADER_STAGE_MAX) { //need to change stage, does not have it
		if (first_found == -1) {
			error_text->add_text(TTR("No valid shader stages found."));
			return; //well you did not put any stage I guess?
		}
		stages[first_found]->set_pressed(true);
		stage = RD::ShaderStage(first_found);
	}

	String error = bytecode->get_stage_compile_error(stage);

	error_text->push_font(get_theme_font("source", "EditorFonts"));

	if (error == String()) {
		error_text->add_text(TTR("Shader stage compiled without errors."));
	} else {
		error_text->add_text(error);
	}
}

void ShaderFileEditor::_update_options() {
	ERR_FAIL_COND(shader_file.is_null());

	if (shader_file->get_base_error() != String()) {
		stage_hb->hide();
		versions->hide();
		error_text->clear();
		error_text->push_font(get_theme_font("source", "EditorFonts"));
		error_text->add_text(vformat(TTR("File structure for '%s' contains unrecoverable errors:\n\n"), shader_file->get_path().get_file()));
		error_text->add_text(shader_file->get_base_error());
		return;
	}

	stage_hb->show();
	versions->show();

	int c = versions->get_current();
	//remember current
	versions->clear();
	Vector<StringName> version_list = shader_file->get_version_list();

	if (c >= version_list.size()) {
		c = version_list.size() - 1;
	}
	if (c < 0) {
		c = 0;
	}

	StringName current_version;

	for (int i = 0; i < version_list.size(); i++) {
		String title = version_list[i];
		if (title == "") {
			title = "default";
		}

		Ref<Texture2D> icon;

		Ref<RDShaderBytecode> bytecode = shader_file->get_bytecode(version_list[i]);
		ERR_FAIL_COND(bytecode.is_null());

		bool failed = false;
		for (int j = 0; j < RD::SHADER_STAGE_MAX; j++) {
			String error = bytecode->get_stage_compile_error(RD::ShaderStage(j));
			if (error != String()) {
				failed = true;
			}
		}

		if (failed) {
			icon = get_theme_icon("ImportFail", "EditorIcons");
		} else {
			icon = get_theme_icon("ImportCheck", "EditorIcons");
		}

		versions->add_item(title, icon);
		versions->set_item_metadata(i, version_list[i]);

		if (i == c) {
			versions->select(i);
			current_version = version_list[i];
		}
	}

	if (version_list.size() == 0) {
		for (int i = 0; i < RD::SHADER_STAGE_MAX; i++) {
			stages[i]->set_disabled(true);
		}
		return;
	}

	Ref<RDShaderBytecode> bytecode = shader_file->get_bytecode(current_version);
	ERR_FAIL_COND(bytecode.is_null());
	int first_valid = -1;
	int current = -1;
	for (int i = 0; i < RD::SHADER_STAGE_MAX; i++) {
		Vector<uint8_t> bc = bytecode->get_stage_bytecode(RD::ShaderStage(i));
		String error = bytecode->get_stage_compile_error(RD::ShaderStage(i));
		bool disable = error == String() && bc.is_empty();
		stages[i]->set_disabled(disable);
		if (!disable) {
			if (stages[i]->is_pressed()) {
				current = i;
			}
			first_valid = i;
		}
	}

	if (current == -1 && first_valid != -1) {
		stages[first_valid]->set_pressed(true);
	}

	_version_selected(0);
}

void ShaderFileEditor::_notification(int p_what) {
	if (p_what == NOTIFICATION_WM_WINDOW_FOCUS_IN) {
		if (is_visible_in_tree() && shader_file.is_valid()) {
			_update_options();
		}
	}
}

void ShaderFileEditor::_editor_settings_changed() {
	if (is_visible_in_tree() && shader_file.is_valid()) {
		_update_options();
	}
}

void ShaderFileEditor::_bind_methods() {
}

void ShaderFileEditor::edit(const Ref<RDShaderFile> &p_shader) {
	if (p_shader.is_null()) {
		if (shader_file.is_valid()) {
			shader_file->disconnect("changed", callable_mp(this, &ShaderFileEditor::_shader_changed));
		}
		return;
	}

	if (shader_file == p_shader) {
		return;
	}

	shader_file = p_shader;

	if (shader_file.is_valid()) {
		shader_file->connect("changed", callable_mp(this, &ShaderFileEditor::_shader_changed));
	}

	_update_options();
}

void ShaderFileEditor::_shader_changed() {
	if (is_visible_in_tree()) {
		_update_options();
	}
}

ShaderFileEditor *ShaderFileEditor::singleton = nullptr;

ShaderFileEditor::ShaderFileEditor(EditorNode *p_node) {
	singleton = this;
	HSplitContainer *main_hs = memnew(HSplitContainer);

	add_child(main_hs);

	versions = memnew(ItemList);
	versions->connect("item_selected", callable_mp(this, &ShaderFileEditor::_version_selected));
	versions->set_custom_minimum_size(Size2i(200 * EDSCALE, 0));
	main_hs->add_child(versions);

	VBoxContainer *main_vb = memnew(VBoxContainer);
	main_vb->set_h_size_flags(SIZE_EXPAND_FILL);
	main_hs->add_child(main_vb);

	static const char *stage_str[RD::SHADER_STAGE_MAX] = {
		"Vertex",
		"Fragment",
		"TessControl",
		"TessEval",
		"Compute"
	};

	stage_hb = memnew(HBoxContainer);
	main_vb->add_child(stage_hb);

	Ref<ButtonGroup> bg;
	bg.instance();
	for (int i = 0; i < RD::SHADER_STAGE_MAX; i++) {
		Button *button = memnew(Button(stage_str[i]));
		button->set_toggle_mode(true);
		button->set_focus_mode(FOCUS_NONE);
		stage_hb->add_child(button);
		stages[i] = button;
		button->set_button_group(bg);
		button->connect("pressed", callable_mp(this, &ShaderFileEditor::_version_selected), varray(i));
	}

	error_text = memnew(RichTextLabel);
	error_text->set_v_size_flags(SIZE_EXPAND_FILL);
	main_vb->add_child(error_text);
}

void ShaderFileEditorPlugin::edit(Object *p_object) {
	RDShaderFile *s = Object::cast_to<RDShaderFile>(p_object);
	shader_editor->edit(s);
}

bool ShaderFileEditorPlugin::handles(Object *p_object) const {
	RDShaderFile *shader = Object::cast_to<RDShaderFile>(p_object);
	return shader != nullptr;
}

void ShaderFileEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		button->show();
		editor->make_bottom_panel_item_visible(shader_editor);

	} else {
		button->hide();
		if (shader_editor->is_visible_in_tree()) {
			editor->hide_bottom_panel();
		}
	}
}

ShaderFileEditorPlugin::ShaderFileEditorPlugin(EditorNode *p_node) {
	editor = p_node;
	shader_editor = memnew(ShaderFileEditor(p_node));

	shader_editor->set_custom_minimum_size(Size2(0, 300) * EDSCALE);
	button = editor->add_bottom_panel_item(TTR("ShaderFile"), shader_editor);
	button->hide();
}

ShaderFileEditorPlugin::~ShaderFileEditorPlugin() {
}
