/*************************************************************************/
/*  editor_native_shader_source_visualizer.cpp                           */
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

#include "editor_native_shader_source_visualizer.h"

#include "scene/gui/text_edit.h"

void EditorNativeShaderSourceVisualizer::_inspect_shader(RID p_shader) {
	if (versions) {
		memdelete(versions);
		versions = nullptr;
	}

	RS::ShaderNativeSourceCode nsc = RS::get_singleton()->shader_get_native_source_code(p_shader);

	versions = memnew(TabContainer);
	versions->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	versions->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	for (int i = 0; i < nsc.versions.size(); i++) {
		TabContainer *vtab = memnew(TabContainer);
		vtab->set_name("Version " + itos(i));
		vtab->set_v_size_flags(Control::SIZE_EXPAND_FILL);
		vtab->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		versions->add_child(vtab);
		for (int j = 0; j < nsc.versions[i].stages.size(); j++) {
			TextEdit *vtext = memnew(TextEdit);
			vtext->set_editable(false);
			vtext->set_name(nsc.versions[i].stages[j].name);
			vtext->set_text(nsc.versions[i].stages[j].code);
			vtext->set_v_size_flags(Control::SIZE_EXPAND_FILL);
			vtext->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			vtab->add_child(vtext);
		}
	}
	add_child(versions);
	popup_centered_ratio();
}

void EditorNativeShaderSourceVisualizer::_bind_methods() {
	ClassDB::bind_method("_inspect_shader", &EditorNativeShaderSourceVisualizer::_inspect_shader);
}
EditorNativeShaderSourceVisualizer::EditorNativeShaderSourceVisualizer() {
	add_to_group("_native_shader_source_visualizer");
	set_title(TTR("Native Shader Source Inspector"));
}
