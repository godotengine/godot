/*************************************************************************/
/*  gi_probe_editor_plugin.cpp                                           */
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

#include "gi_probe_editor_plugin.h"

void GIProbeEditorPlugin::_bake() {
	if (gi_probe) {
		if (gi_probe->get_probe_data().is_null()) {
			String path = get_tree()->get_edited_scene_root()->get_filename();
			if (path == String()) {
				path = "res://" + gi_probe->get_name() + "_data.res";
			} else {
				String ext = path.get_extension();
				path = path.get_basename() + "." + gi_probe->get_name() + "_data.res";
			}
			probe_file->set_current_path(path);
			probe_file->popup_file_dialog();
			return;
		}
		gi_probe->bake();
	}
}

void GIProbeEditorPlugin::edit(Object *p_object) {
	GIProbe *s = Object::cast_to<GIProbe>(p_object);
	if (!s) {
		return;
	}

	gi_probe = s;
}

bool GIProbeEditorPlugin::handles(Object *p_object) const {
	return p_object->is_class("GIProbe");
}

void GIProbeEditorPlugin::_notification(int p_what) {
	if (p_what == NOTIFICATION_PROCESS) {
		if (!gi_probe) {
			return;
		}

		const Vector3i size = gi_probe->get_estimated_cell_size();
		String text = vformat(String::utf8("%d × %d × %d"), size.x, size.y, size.z);
		int data_size = 4;
		if (GLOBAL_GET("rendering/quality/gi_probes/anisotropic")) {
			data_size += 4;
		}
		const double size_mb = size.x * size.y * size.z * data_size / (1024.0 * 1024.0);
		text += " - " + vformat(TTR("VRAM Size: %s MB"), String::num(size_mb, 2));

		if (bake_info->get_text() == text) {
			return;
		}

		// Color the label depending on the estimated performance level.
		Color color;
		if (size_mb <= 16.0 + CMP_EPSILON) {
			// Fast.
			color = bake_info->get_theme_color("success_color", "Editor");
		} else if (size_mb <= 64.0 + CMP_EPSILON) {
			// Medium.
			color = bake_info->get_theme_color("warning_color", "Editor");
		} else {
			// Slow.
			color = bake_info->get_theme_color("error_color", "Editor");
		}
		bake_info->add_theme_color_override("font_color", color);

		bake_info->set_text(text);
	}
}

void GIProbeEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		bake_hb->show();
		set_process(true);
	} else {
		bake_hb->hide();
		set_process(false);
	}
}

EditorProgress *GIProbeEditorPlugin::tmp_progress = nullptr;

void GIProbeEditorPlugin::bake_func_begin(int p_steps) {
	ERR_FAIL_COND(tmp_progress != nullptr);

	tmp_progress = memnew(EditorProgress("bake_gi", TTR("Bake GI Probe"), p_steps));
}

void GIProbeEditorPlugin::bake_func_step(int p_step, const String &p_description) {
	ERR_FAIL_COND(tmp_progress == nullptr);
	tmp_progress->step(p_description, p_step, false);
}

void GIProbeEditorPlugin::bake_func_end() {
	ERR_FAIL_COND(tmp_progress == nullptr);
	memdelete(tmp_progress);
	tmp_progress = nullptr;
}

void GIProbeEditorPlugin::_giprobe_save_path_and_bake(const String &p_path) {
	probe_file->hide();
	if (gi_probe) {
		gi_probe->bake();
		ERR_FAIL_COND(gi_probe->get_probe_data().is_null());
		ResourceSaver::save(p_path, gi_probe->get_probe_data(), ResourceSaver::FLAG_CHANGE_PATH);
	}
}

void GIProbeEditorPlugin::_bind_methods() {
}

GIProbeEditorPlugin::GIProbeEditorPlugin(EditorNode *p_node) {
	editor = p_node;
	bake_hb = memnew(HBoxContainer);
	bake_hb->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	bake_hb->hide();
	bake = memnew(Button);
	bake->set_flat(true);
	bake->set_icon(editor->get_gui_base()->get_theme_icon("Bake", "EditorIcons"));
	bake->set_text(TTR("Bake GI Probe"));
	bake->connect("pressed", callable_mp(this, &GIProbeEditorPlugin::_bake));
	bake_hb->add_child(bake);
	bake_info = memnew(Label);
	bake_info->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	bake_info->set_clip_text(true);
	bake_hb->add_child(bake_info);

	add_control_to_container(CONTAINER_SPATIAL_EDITOR_MENU, bake_hb);
	gi_probe = nullptr;
	probe_file = memnew(EditorFileDialog);
	probe_file->set_file_mode(EditorFileDialog::FILE_MODE_SAVE_FILE);
	probe_file->add_filter("*.res");
	probe_file->connect("file_selected", callable_mp(this, &GIProbeEditorPlugin::_giprobe_save_path_and_bake));
	get_editor_interface()->get_base_control()->add_child(probe_file);
	probe_file->set_title(TTR("Select path for GIProbe Data File"));

	GIProbe::bake_begin_function = bake_func_begin;
	GIProbe::bake_step_function = bake_func_step;
	GIProbe::bake_end_function = bake_func_end;
}

GIProbeEditorPlugin::~GIProbeEditorPlugin() {
}
