/*************************************************************************/
/*  voxel_gi_editor_plugin.cpp                                           */
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

#include "voxel_gi_editor_plugin.h"

void VoxelGIEditorPlugin::_bake() {
	if (voxel_gi) {
		if (voxel_gi->get_probe_data().is_null()) {
			String path = get_tree()->get_edited_scene_root()->get_scene_file_path();
			if (path == String()) {
				path = "res://" + voxel_gi->get_name() + "_data.res";
			} else {
				String ext = path.get_extension();
				path = path.get_basename() + "." + voxel_gi->get_name() + "_data.res";
			}
			probe_file->set_current_path(path);
			probe_file->popup_file_dialog();
			return;
		}
		voxel_gi->bake();
	}
}

void VoxelGIEditorPlugin::edit(Object *p_object) {
	VoxelGI *s = Object::cast_to<VoxelGI>(p_object);
	if (!s) {
		return;
	}

	voxel_gi = s;
}

bool VoxelGIEditorPlugin::handles(Object *p_object) const {
	return p_object->is_class("VoxelGI");
}

void VoxelGIEditorPlugin::_notification(int p_what) {
	if (p_what == NOTIFICATION_PROCESS) {
		if (!voxel_gi) {
			return;
		}

		// Set information tooltip on the Bake button. This information is useful
		// to optimize performance (video RAM size) and reduce light leaking (individual cell size).

		const Vector3i size = voxel_gi->get_estimated_cell_size();

		const Vector3 extents = voxel_gi->get_extents();

		const int data_size = 4;
		const double size_mb = size.x * size.y * size.z * data_size / (1024.0 * 1024.0);
		// Add a qualitative measurement to help the user assess whether a VoxelGI node is using a lot of VRAM.
		String size_quality;
		if (size_mb < 16.0) {
			size_quality = TTR("Low");
		} else if (size_mb < 64.0) {
			size_quality = TTR("Moderate");
		} else {
			size_quality = TTR("High");
		}

		String text;
		text += vformat(TTR("Subdivisions: %s"), vformat(String::utf8("%d × %d × %d"), size.x, size.y, size.z)) + "\n";
		text += vformat(TTR("Cell size: %s"), vformat(String::utf8("%.3f × %.3f × %.3f"), extents.x / size.x, extents.y / size.y, extents.z / size.z)) + "\n";
		text += vformat(TTR("Video RAM size: %s MB (%s)"), String::num(size_mb, 2), size_quality);

		// Only update the tooltip when needed to avoid constant redrawing.
		if (bake->get_tooltip(Point2()) == text) {
			return;
		}

		bake->set_tooltip(text);
	}
}

void VoxelGIEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		bake_hb->show();
		set_process(true);
	} else {
		bake_hb->hide();
		set_process(false);
	}
}

EditorProgress *VoxelGIEditorPlugin::tmp_progress = nullptr;

void VoxelGIEditorPlugin::bake_func_begin(int p_steps) {
	ERR_FAIL_COND(tmp_progress != nullptr);

	tmp_progress = memnew(EditorProgress("bake_gi", TTR("Bake GI Probe"), p_steps));
}

void VoxelGIEditorPlugin::bake_func_step(int p_step, const String &p_description) {
	ERR_FAIL_COND(tmp_progress == nullptr);
	tmp_progress->step(p_description, p_step, false);
}

void VoxelGIEditorPlugin::bake_func_end() {
	ERR_FAIL_COND(tmp_progress == nullptr);
	memdelete(tmp_progress);
	tmp_progress = nullptr;
}

void VoxelGIEditorPlugin::_voxel_gi_save_path_and_bake(const String &p_path) {
	probe_file->hide();
	if (voxel_gi) {
		voxel_gi->bake();
		ERR_FAIL_COND(voxel_gi->get_probe_data().is_null());
		ResourceSaver::save(p_path, voxel_gi->get_probe_data(), ResourceSaver::FLAG_CHANGE_PATH);
	}
}

void VoxelGIEditorPlugin::_bind_methods() {
}

VoxelGIEditorPlugin::VoxelGIEditorPlugin(EditorNode *p_node) {
	editor = p_node;
	bake_hb = memnew(HBoxContainer);
	bake_hb->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	bake_hb->hide();
	bake = memnew(Button);
	bake->set_flat(true);
	bake->set_icon(editor->get_gui_base()->get_theme_icon(SNAME("Bake"), SNAME("EditorIcons")));
	bake->set_text(TTR("Bake GI Probe"));
	bake->connect("pressed", callable_mp(this, &VoxelGIEditorPlugin::_bake));
	bake_hb->add_child(bake);

	add_control_to_container(CONTAINER_SPATIAL_EDITOR_MENU, bake_hb);
	voxel_gi = nullptr;
	probe_file = memnew(EditorFileDialog);
	probe_file->set_file_mode(EditorFileDialog::FILE_MODE_SAVE_FILE);
	probe_file->add_filter("*.res");
	probe_file->connect("file_selected", callable_mp(this, &VoxelGIEditorPlugin::_voxel_gi_save_path_and_bake));
	get_editor_interface()->get_base_control()->add_child(probe_file);
	probe_file->set_title(TTR("Select path for VoxelGI Data File"));

	VoxelGI::bake_begin_function = bake_func_begin;
	VoxelGI::bake_step_function = bake_func_step;
	VoxelGI::bake_end_function = bake_func_end;
}

VoxelGIEditorPlugin::~VoxelGIEditorPlugin() {
}
