/**************************************************************************/
/*  shader_editor_plugin.cpp                                              */
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

#include "shader_editor_plugin.h"

#include "editor/editor_node.h"
#include "editor/editor_scale.h"
#include "editor/editor_undo_redo_manager.h"
#include "editor/filesystem_dock.h"
#include "editor/inspector_dock.h"
#include "editor/plugins/text_shader_editor.h"
#include "editor/plugins/visual_shader_editor_plugin.h"
#include "editor/shader_create_dialog.h"
#include "scene/gui/item_list.h"
#include "scene/gui/texture_rect.h"

void ShaderEditorPlugin::_update_shader_list() {
	shader_list->clear();
	for (EditedShader &edited_shader : edited_shaders) {
		Ref<Resource> shader = edited_shader.shader;
		if (shader.is_null()) {
			shader = edited_shader.shader_inc;
		}

		String path = shader->get_path();
		String text = path.get_file();
		if (text.is_empty()) {
			// This appears for newly created built-in shaders before saving the scene.
			text = TTR("[unsaved]");
		} else if (shader->is_built_in()) {
			const String &shader_name = shader->get_name();
			if (!shader_name.is_empty()) {
				text = vformat("%s (%s)", shader_name, text.get_slice("::", 0));
			}
		}

		bool unsaved = false;
		if (edited_shader.shader_editor) {
			unsaved = edited_shader.shader_editor->is_unsaved();
		}
		// TODO: Handle visual shaders too.

		if (unsaved) {
			text += "(*)";
		}

		String _class = shader->get_class();
		if (!shader_list->has_theme_icon(_class, SNAME("EditorIcons"))) {
			_class = "TextFile";
		}
		Ref<Texture2D> icon = shader_list->get_theme_icon(_class, SNAME("EditorIcons"));

		shader_list->add_item(text, icon);
		shader_list->set_item_tooltip(shader_list->get_item_count() - 1, path);
	}

	if (shader_tabs->get_tab_count()) {
		shader_list->select(shader_tabs->get_current_tab());
	}

	for (int i = FILE_SAVE; i < FILE_MAX; i++) {
		file_menu->get_popup()->set_item_disabled(file_menu->get_popup()->get_item_index(i), edited_shaders.is_empty());
	}

	_update_shader_list_status();
}

void ShaderEditorPlugin::_update_shader_list_status() {
	for (int i = 0; i < shader_list->get_item_count(); i++) {
		TextShaderEditor *se = Object::cast_to<TextShaderEditor>(shader_tabs->get_tab_control(i));
		if (se) {
			if (se->was_compilation_successful()) {
				shader_list->set_item_tag_icon(i, Ref<Texture2D>());
			} else {
				shader_list->set_item_tag_icon(i, shader_list->get_theme_icon(SNAME("Error"), SNAME("EditorIcons")));
			}
		}
	}
}

void ShaderEditorPlugin::_move_shader_tab(int p_from, int p_to) {
	if (p_from == p_to) {
		return;
	}
	EditedShader es = edited_shaders[p_from];
	edited_shaders.remove_at(p_from);
	edited_shaders.insert(p_to, es);
	shader_tabs->move_child(shader_tabs->get_tab_control(p_from), p_to);
	_update_shader_list();
}

void ShaderEditorPlugin::edit(Object *p_object) {
	if (!p_object) {
		return;
	}

	EditedShader es;

	ShaderInclude *si = Object::cast_to<ShaderInclude>(p_object);
	if (si != nullptr) {
		for (uint32_t i = 0; i < edited_shaders.size(); i++) {
			if (edited_shaders[i].shader_inc.ptr() == si) {
				shader_tabs->set_current_tab(i);
				shader_list->select(i);
				return;
			}
		}
		es.shader_inc = Ref<ShaderInclude>(si);
		es.shader_editor = memnew(TextShaderEditor);
		es.shader_editor->edit(si);
		shader_tabs->add_child(es.shader_editor);
		es.shader_editor->connect("validation_changed", callable_mp(this, &ShaderEditorPlugin::_update_shader_list));
	} else {
		Shader *s = Object::cast_to<Shader>(p_object);
		for (uint32_t i = 0; i < edited_shaders.size(); i++) {
			if (edited_shaders[i].shader.ptr() == s) {
				shader_tabs->set_current_tab(i);
				shader_list->select(i);
				return;
			}
		}
		es.shader = Ref<Shader>(s);
		Ref<VisualShader> vs = es.shader;
		if (vs.is_valid()) {
			es.visual_shader_editor = memnew(VisualShaderEditor);
			shader_tabs->add_child(es.visual_shader_editor);
			es.visual_shader_editor->edit(vs.ptr());
		} else {
			es.shader_editor = memnew(TextShaderEditor);
			shader_tabs->add_child(es.shader_editor);
			es.shader_editor->edit(s);
			es.shader_editor->connect("validation_changed", callable_mp(this, &ShaderEditorPlugin::_update_shader_list));
		}
	}

	shader_tabs->set_current_tab(shader_tabs->get_tab_count() - 1);
	edited_shaders.push_back(es);
	_update_shader_list();
}

bool ShaderEditorPlugin::handles(Object *p_object) const {
	return Object::cast_to<Shader>(p_object) != nullptr || Object::cast_to<ShaderInclude>(p_object) != nullptr;
}

void ShaderEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		EditorNode::get_singleton()->make_bottom_panel_item_visible(main_split);
	}
}

void ShaderEditorPlugin::selected_notify() {
}

TextShaderEditor *ShaderEditorPlugin::get_shader_editor(const Ref<Shader> &p_for_shader) {
	for (EditedShader &edited_shader : edited_shaders) {
		if (edited_shader.shader == p_for_shader) {
			return edited_shader.shader_editor;
		}
	}
	return nullptr;
}

VisualShaderEditor *ShaderEditorPlugin::get_visual_shader_editor(const Ref<Shader> &p_for_shader) {
	for (EditedShader &edited_shader : edited_shaders) {
		if (edited_shader.shader == p_for_shader) {
			return edited_shader.visual_shader_editor;
		}
	}
	return nullptr;
}

void ShaderEditorPlugin::save_external_data() {
	for (EditedShader &edited_shader : edited_shaders) {
		if (edited_shader.shader_editor) {
			edited_shader.shader_editor->save_external_data();
		}
	}
	_update_shader_list();
}

void ShaderEditorPlugin::apply_changes() {
	for (EditedShader &edited_shader : edited_shaders) {
		if (edited_shader.shader_editor) {
			edited_shader.shader_editor->apply_shaders();
		}
	}
}

void ShaderEditorPlugin::_shader_selected(int p_index) {
	if (edited_shaders[p_index].shader_editor) {
		edited_shaders[p_index].shader_editor->validate_script();
	}
	shader_tabs->set_current_tab(p_index);
	shader_list->select(p_index);
}

void ShaderEditorPlugin::_shader_list_clicked(int p_item, Vector2 p_local_mouse_pos, MouseButton p_mouse_button_index) {
	if (p_mouse_button_index == MouseButton::MIDDLE) {
		_close_shader(p_item);
	}
}

void ShaderEditorPlugin::_close_shader(int p_index) {
	ERR_FAIL_INDEX(p_index, shader_tabs->get_tab_count());
	Control *c = shader_tabs->get_tab_control(p_index);
	memdelete(c);
	edited_shaders.remove_at(p_index);
	_update_shader_list();
	EditorUndoRedoManager::get_singleton()->clear_history(); // To prevent undo on deleted graphs.
}

void ShaderEditorPlugin::_resource_saved(Object *obj) {
	// May have been renamed on save.
	for (EditedShader &edited_shader : edited_shaders) {
		if (edited_shader.shader.ptr() == obj) {
			_update_shader_list();
			return;
		}
	}
}

void ShaderEditorPlugin::_menu_item_pressed(int p_index) {
	switch (p_index) {
		case FILE_NEW: {
			String base_path = FileSystemDock::get_singleton()->get_current_path().get_base_dir();
			shader_create_dialog->config(base_path.path_join("new_shader"), false, false, 0);
			shader_create_dialog->popup_centered();
		} break;
		case FILE_NEW_INCLUDE: {
			String base_path = FileSystemDock::get_singleton()->get_current_path().get_base_dir();
			shader_create_dialog->config(base_path.path_join("new_shader"), false, false, 2);
			shader_create_dialog->popup_centered();
		} break;
		case FILE_OPEN: {
			InspectorDock::get_singleton()->open_resource("Shader");
		} break;
		case FILE_OPEN_INCLUDE: {
			InspectorDock::get_singleton()->open_resource("ShaderInclude");
		} break;
		case FILE_SAVE: {
			int index = shader_tabs->get_current_tab();
			ERR_FAIL_INDEX(index, shader_tabs->get_tab_count());
			if (edited_shaders[index].shader.is_valid()) {
				EditorNode::get_singleton()->save_resource(edited_shaders[index].shader);
			} else {
				EditorNode::get_singleton()->save_resource(edited_shaders[index].shader_inc);
			}
			if (edited_shaders[index].shader_editor) {
				edited_shaders[index].shader_editor->tag_saved_version();
			}
		} break;
		case FILE_SAVE_AS: {
			int index = shader_tabs->get_current_tab();
			ERR_FAIL_INDEX(index, shader_tabs->get_tab_count());
			String path;
			if (edited_shaders[index].shader.is_valid()) {
				path = edited_shaders[index].shader->get_path();
				if (!path.is_resource_file()) {
					path = "";
				}
				EditorNode::get_singleton()->save_resource_as(edited_shaders[index].shader, path);
			} else {
				path = edited_shaders[index].shader_inc->get_path();
				if (!path.is_resource_file()) {
					path = "";
				}
				EditorNode::get_singleton()->save_resource_as(edited_shaders[index].shader_inc, path);
			}
			if (edited_shaders[index].shader_editor) {
				edited_shaders[index].shader_editor->tag_saved_version();
			}
		} break;
		case FILE_INSPECT: {
			int index = shader_tabs->get_current_tab();
			ERR_FAIL_INDEX(index, shader_tabs->get_tab_count());
			if (edited_shaders[index].shader.is_valid()) {
				EditorNode::get_singleton()->push_item(edited_shaders[index].shader.ptr());
			} else {
				EditorNode::get_singleton()->push_item(edited_shaders[index].shader_inc.ptr());
			}
		} break;
		case FILE_CLOSE: {
			_close_shader(shader_tabs->get_current_tab());
		} break;
	}
}

void ShaderEditorPlugin::_shader_created(Ref<Shader> p_shader) {
	EditorNode::get_singleton()->push_item(p_shader.ptr());
}

void ShaderEditorPlugin::_shader_include_created(Ref<ShaderInclude> p_shader_inc) {
	EditorNode::get_singleton()->push_item(p_shader_inc.ptr());
}

Variant ShaderEditorPlugin::get_drag_data_fw(const Point2 &p_point, Control *p_from) {
	if (shader_list->get_item_count() == 0) {
		return Variant();
	}

	int idx = shader_list->get_item_at_position(p_point);
	if (idx < 0) {
		return Variant();
	}

	HBoxContainer *drag_preview = memnew(HBoxContainer);
	String preview_name = shader_list->get_item_text(idx);
	Ref<Texture2D> preview_icon = shader_list->get_item_icon(idx);

	if (!preview_icon.is_null()) {
		TextureRect *tf = memnew(TextureRect);
		tf->set_texture(preview_icon);
		tf->set_stretch_mode(TextureRect::STRETCH_KEEP_CENTERED);
		drag_preview->add_child(tf);
	}
	Label *label = memnew(Label(preview_name));
	drag_preview->add_child(label);
	main_split->set_drag_preview(drag_preview);

	Dictionary drag_data;
	drag_data["type"] = "shader_list_element";
	drag_data["shader_list_element"] = idx;

	return drag_data;
}

bool ShaderEditorPlugin::can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const {
	Dictionary d = p_data;
	if (!d.has("type")) {
		return false;
	}

	if (String(d["type"]) == "shader_list_element") {
		return true;
	}

	if (String(d["type"]) == "files") {
		Vector<String> files = d["files"];

		if (files.size() == 0) {
			return false;
		}

		for (int i = 0; i < files.size(); i++) {
			String file = files[i];
			if (ResourceLoader::exists(file, "Shader")) {
				Ref<Shader> shader = ResourceLoader::load(file);
				if (shader.is_valid()) {
					return true;
				}
			}
		}
		return false;
	}

	return false;
}

void ShaderEditorPlugin::drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) {
	if (!can_drop_data_fw(p_point, p_data, p_from)) {
		return;
	}

	Dictionary d = p_data;
	if (!d.has("type")) {
		return;
	}

	if (String(d["type"]) == "shader_list_element") {
		int idx = d["shader_list_element"];
		int new_idx = shader_list->get_item_at_position(p_point);
		_move_shader_tab(idx, new_idx);
		return;
	}

	if (String(d["type"]) == "files") {
		Vector<String> files = d["files"];

		for (int i = 0; i < files.size(); i++) {
			String file = files[i];
			if (!ResourceLoader::exists(file, "Shader")) {
				continue;
			}

			Ref<Resource> res = ResourceLoader::load(file);
			if (res.is_valid()) {
				edit(res.ptr());
			}
		}
	}
}

ShaderEditorPlugin::ShaderEditorPlugin() {
	main_split = memnew(HSplitContainer);

	VBoxContainer *vb = memnew(VBoxContainer);

	HBoxContainer *file_hb = memnew(HBoxContainer);
	vb->add_child(file_hb);
	file_menu = memnew(MenuButton);
	file_menu->set_text(TTR("File"));
	file_menu->get_popup()->add_item(TTR("New Shader"), FILE_NEW);
	file_menu->get_popup()->add_item(TTR("New Shader Include"), FILE_NEW_INCLUDE);
	file_menu->get_popup()->add_separator();
	file_menu->get_popup()->add_item(TTR("Load Shader File"), FILE_OPEN);
	file_menu->get_popup()->add_item(TTR("Load Shader Include File"), FILE_OPEN_INCLUDE);
	file_menu->get_popup()->add_item(TTR("Save File"), FILE_SAVE);
	file_menu->get_popup()->add_item(TTR("Save File As"), FILE_SAVE_AS);
	file_menu->get_popup()->add_separator();
	file_menu->get_popup()->add_item(TTR("Open File in Inspector"), FILE_INSPECT);
	file_menu->get_popup()->add_separator();
	file_menu->get_popup()->add_item(TTR("Close File"), FILE_CLOSE);
	file_menu->get_popup()->connect("id_pressed", callable_mp(this, &ShaderEditorPlugin::_menu_item_pressed));
	file_hb->add_child(file_menu);

	for (int i = FILE_SAVE; i < FILE_MAX; i++) {
		file_menu->get_popup()->set_item_disabled(file_menu->get_popup()->get_item_index(i), true);
	}

	shader_list = memnew(ItemList);
	shader_list->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	vb->add_child(shader_list);
	shader_list->connect("item_selected", callable_mp(this, &ShaderEditorPlugin::_shader_selected));
	shader_list->connect("item_clicked", callable_mp(this, &ShaderEditorPlugin::_shader_list_clicked));
	SET_DRAG_FORWARDING_GCD(shader_list, ShaderEditorPlugin);

	main_split->add_child(vb);
	vb->set_custom_minimum_size(Size2(200, 300) * EDSCALE);

	shader_tabs = memnew(TabContainer);
	shader_tabs->set_tabs_visible(false);
	shader_tabs->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	main_split->add_child(shader_tabs);
	Ref<StyleBoxEmpty> empty;
	empty.instantiate();
	shader_tabs->add_theme_style_override("panel", empty);

	button = EditorNode::get_singleton()->add_bottom_panel_item(TTR("Shader Editor"), main_split);

	// Defer connect because Editor class is not in the binding system yet.
	EditorNode::get_singleton()->call_deferred("connect", "resource_saved", callable_mp(this, &ShaderEditorPlugin::_resource_saved), CONNECT_DEFERRED);

	shader_create_dialog = memnew(ShaderCreateDialog);
	vb->add_child(shader_create_dialog);
	shader_create_dialog->connect("shader_created", callable_mp(this, &ShaderEditorPlugin::_shader_created));
	shader_create_dialog->connect("shader_include_created", callable_mp(this, &ShaderEditorPlugin::_shader_include_created));
}

ShaderEditorPlugin::~ShaderEditorPlugin() {
}
