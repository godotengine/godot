/*************************************************************************/
/*  gpu_particles_collision_sdf_editor_plugin.cpp                        */
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

#include "gpu_particles_collision_sdf_editor_plugin.h"

void GPUParticlesCollisionSDFEditorPlugin::_bake() {
	if (col_sdf) {
		if (col_sdf->get_texture().is_null() || !col_sdf->get_texture()->get_path().is_resource_file()) {
			String path = get_tree()->get_edited_scene_root()->get_filename();
			if (path == String()) {
				path = "res://" + col_sdf->get_name() + "_data.exr";
			} else {
				String ext = path.get_extension();
				path = path.get_basename() + "." + col_sdf->get_name() + "_data.exr";
			}
			probe_file->set_current_path(path);
			probe_file->popup_file_dialog();
			return;
		}

		_sdf_save_path_and_bake(col_sdf->get_texture()->get_path());
	}
}

void GPUParticlesCollisionSDFEditorPlugin::edit(Object *p_object) {
	GPUParticlesCollisionSDF *s = Object::cast_to<GPUParticlesCollisionSDF>(p_object);
	if (!s) {
		return;
	}

	col_sdf = s;
}

bool GPUParticlesCollisionSDFEditorPlugin::handles(Object *p_object) const {
	return p_object->is_class("GPUParticlesCollisionSDF");
}

void GPUParticlesCollisionSDFEditorPlugin::_notification(int p_what) {
	if (p_what == NOTIFICATION_PROCESS) {
		if (!col_sdf) {
			return;
		}

		const Vector3i size = col_sdf->get_estimated_cell_size();
		String text = vformat(String::utf8("%d × %d × %d"), size.x, size.y, size.z);
		int data_size = 2;

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

void GPUParticlesCollisionSDFEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		bake_hb->show();
		set_process(true);
	} else {
		bake_hb->hide();
		set_process(false);
	}
}

EditorProgress *GPUParticlesCollisionSDFEditorPlugin::tmp_progress = nullptr;

void GPUParticlesCollisionSDFEditorPlugin::bake_func_begin(int p_steps) {
	ERR_FAIL_COND(tmp_progress != nullptr);

	tmp_progress = memnew(EditorProgress("bake_sdf", TTR("Bake SDF"), p_steps));
}

void GPUParticlesCollisionSDFEditorPlugin::bake_func_step(int p_step, const String &p_description) {
	ERR_FAIL_COND(tmp_progress == nullptr);
	tmp_progress->step(p_description, p_step, false);
}

void GPUParticlesCollisionSDFEditorPlugin::bake_func_end() {
	ERR_FAIL_COND(tmp_progress == nullptr);
	memdelete(tmp_progress);
	tmp_progress = nullptr;
}

void GPUParticlesCollisionSDFEditorPlugin::_sdf_save_path_and_bake(const String &p_path) {
	probe_file->hide();
	if (col_sdf) {
		Ref<Image> bake_img = col_sdf->bake();
		if (bake_img.is_null()) {
			EditorNode::get_singleton()->show_warning("Bake Error.");
			return;
		}

		Ref<ConfigFile> config;

		config.instance();
		if (FileAccess::exists(p_path + ".import")) {
			config->load(p_path + ".import");
		}

		config->set_value("remap", "importer", "3d_texture");
		config->set_value("remap", "type", "StreamTexture3D");
		if (!config->has_section_key("params", "compress/mode")) {
			config->set_value("params", "compress/mode", 3); //user may want another compression, so leave it be
		}
		config->set_value("params", "compress/channel_pack", 1);
		config->set_value("params", "mipmaps/generate", false);
		config->set_value("params", "slices/horizontal", 1);
		config->set_value("params", "slices/vertical", bake_img->get_meta("depth"));

		config->save(p_path + ".import");

		Error err = bake_img->save_exr(p_path, false);
		ERR_FAIL_COND(err);
		ResourceLoader::import(p_path);
		Ref<Texture> t = ResourceLoader::load(p_path); //if already loaded, it will be updated on refocus?
		ERR_FAIL_COND(t.is_null());

		col_sdf->set_texture(t);
	}
}

void GPUParticlesCollisionSDFEditorPlugin::_bind_methods() {
}

GPUParticlesCollisionSDFEditorPlugin::GPUParticlesCollisionSDFEditorPlugin(EditorNode *p_node) {
	editor = p_node;
	bake_hb = memnew(HBoxContainer);
	bake_hb->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	bake_hb->hide();
	bake = memnew(Button);
	bake->set_flat(true);
	bake->set_icon(editor->get_gui_base()->get_theme_icon("Bake", "EditorIcons"));
	bake->set_text(TTR("Bake SDF"));
	bake->connect("pressed", callable_mp(this, &GPUParticlesCollisionSDFEditorPlugin::_bake));
	bake_hb->add_child(bake);
	bake_info = memnew(Label);
	bake_info->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	bake_info->set_clip_text(true);
	bake_hb->add_child(bake_info);

	add_control_to_container(CONTAINER_SPATIAL_EDITOR_MENU, bake_hb);
	col_sdf = nullptr;
	probe_file = memnew(EditorFileDialog);
	probe_file->set_file_mode(EditorFileDialog::FILE_MODE_SAVE_FILE);
	probe_file->add_filter("*.exr");
	probe_file->connect("file_selected", callable_mp(this, &GPUParticlesCollisionSDFEditorPlugin::_sdf_save_path_and_bake));
	get_editor_interface()->get_base_control()->add_child(probe_file);
	probe_file->set_title(TTR("Select path for SDF Texture"));

	GPUParticlesCollisionSDF::bake_begin_function = bake_func_begin;
	GPUParticlesCollisionSDF::bake_step_function = bake_func_step;
	GPUParticlesCollisionSDF::bake_end_function = bake_func_end;
}

GPUParticlesCollisionSDFEditorPlugin::~GPUParticlesCollisionSDFEditorPlugin() {
}
