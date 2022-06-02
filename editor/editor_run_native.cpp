/*************************************************************************/
/*  editor_run_native.cpp                                                */
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

#include "editor_run_native.h"

#include "editor_export.h"
#include "editor_node.h"
#include "editor_scale.h"

void EditorRunNative::_notification(int p_what) {
	if (p_what == NOTIFICATION_ENTER_TREE) {
		for (int i = 0; i < EditorExport::get_singleton()->get_export_platform_count(); i++) {
			Ref<EditorExportPlatform> eep = EditorExport::get_singleton()->get_export_platform(i);
			if (eep.is_null()) {
				continue;
			}
			Ref<ImageTexture> icon = eep->get_run_icon();
			if (!icon.is_null()) {
				Ref<Image> im = icon->get_data();
				im = im->duplicate();
				im->clear_mipmaps();
				if (!im->empty()) {
					im->resize(16 * EDSCALE, 16 * EDSCALE);
					Ref<ImageTexture> small_icon;
					small_icon.instance();
					small_icon->create_from_image(im, 0);
					MenuButton *mb = memnew(MenuButton);
					mb->get_popup()->connect("id_pressed", this, "_run_native", varray(i));
					mb->connect("pressed", this, "_run_native", varray(-1, i));
					mb->set_icon(small_icon);
					add_child(mb);
					menus[i] = mb;
				}
			}
		}
	}

	if (p_what == NOTIFICATION_PROCESS) {
		bool changed = EditorExport::get_singleton()->poll_export_platforms() || first;

		if (changed) {
			for (Map<int, MenuButton *>::Element *E = menus.front(); E; E = E->next()) {
				Ref<EditorExportPlatform> eep = EditorExport::get_singleton()->get_export_platform(E->key());
				MenuButton *mb = E->get();
				int dc = eep->get_options_count();

				if (dc == 0) {
					mb->hide();
				} else {
					mb->get_popup()->clear();
					mb->show();
					if (dc == 1) {
						mb->set_tooltip(eep->get_option_tooltip(0));
					} else {
						mb->set_tooltip(eep->get_options_tooltip());
						for (int i = 0; i < dc; i++) {
							mb->get_popup()->add_icon_item(eep->get_option_icon(i), eep->get_option_label(i));
							mb->get_popup()->set_item_tooltip(mb->get_popup()->get_item_count() - 1, eep->get_option_tooltip(i));
						}
					}
				}
			}

			first = false;
		}
	}
}

void EditorRunNative::_run_native(int p_idx, int p_platform) {
	if (!EditorNode::get_singleton()->ensure_main_scene(true)) {
		resume_idx = p_idx;
		resume_platform = p_platform;
		return;
	}

	Ref<EditorExportPlatform> eep = EditorExport::get_singleton()->get_export_platform(p_platform);
	ERR_FAIL_COND(eep.is_null());

	if (p_idx == -1) {
		if (eep->get_options_count() == 1) {
			menus[p_platform]->get_popup()->hide();
			p_idx = 0;
		} else {
			return;
		}
	}

	Ref<EditorExportPreset> preset;

	for (int i = 0; i < EditorExport::get_singleton()->get_export_preset_count(); i++) {
		Ref<EditorExportPreset> ep = EditorExport::get_singleton()->get_export_preset(i);
		if (ep->is_runnable() && ep->get_platform() == eep) {
			preset = ep;
			break;
		}
	}

	if (preset.is_null()) {
		EditorNode::get_singleton()->show_warning(TTR("No runnable export preset found for this platform.\nPlease add a runnable preset in the Export menu or define an existing preset as runnable."));
		return;
	}

	emit_signal("native_run");

	int flags = 0;
	if (deploy_debug_remote) {
		flags |= EditorExportPlatform::DEBUG_FLAG_REMOTE_DEBUG;
	}
	if (deploy_dumb) {
		flags |= EditorExportPlatform::DEBUG_FLAG_DUMB_CLIENT;
	}
	if (debug_collisions) {
		flags |= EditorExportPlatform::DEBUG_FLAG_VIEW_COLLISONS;
	}
	if (debug_navigation) {
		flags |= EditorExportPlatform::DEBUG_FLAG_VIEW_NAVIGATION;
	}
	if (debug_shader_fallbacks) {
		flags |= EditorExportPlatform::DEBUG_FLAG_SHADER_FALLBACKS;
	}

	eep->clear_messages();
	Error err = eep->run(preset, p_idx, flags);
	result_dialog_log->clear();
	if (eep->fill_log_messages(result_dialog_log, err)) {
		result_dialog->popup_centered_ratio(0.5);
	}
}

void EditorRunNative::resume_run_native() {
	_run_native(resume_idx, resume_platform);
}

void EditorRunNative::_bind_methods() {
	ClassDB::bind_method("_run_native", &EditorRunNative::_run_native);

	ADD_SIGNAL(MethodInfo("native_run"));
}

void EditorRunNative::set_deploy_dumb(bool p_enabled) {
	deploy_dumb = p_enabled;
}

bool EditorRunNative::is_deploy_dumb_enabled() const {
	return deploy_dumb;
}

void EditorRunNative::set_deploy_debug_remote(bool p_enabled) {
	deploy_debug_remote = p_enabled;
}

bool EditorRunNative::is_deploy_debug_remote_enabled() const {
	return deploy_debug_remote;
}

void EditorRunNative::set_debug_collisions(bool p_debug) {
	debug_collisions = p_debug;
}

bool EditorRunNative::get_debug_collisions() const {
	return debug_collisions;
}

void EditorRunNative::set_debug_navigation(bool p_debug) {
	debug_navigation = p_debug;
}

bool EditorRunNative::get_debug_navigation() const {
	return debug_navigation;
}

void EditorRunNative::set_debug_shader_fallbacks(bool p_debug) {
	debug_shader_fallbacks = p_debug;
}

bool EditorRunNative::get_debug_shader_fallbacks() const {
	return debug_shader_fallbacks;
}

EditorRunNative::EditorRunNative() {
	result_dialog = memnew(AcceptDialog);
	result_dialog->set_title(TTR("Project Run"));
	result_dialog_log = memnew(RichTextLabel);
	result_dialog_log->set_custom_minimum_size(Size2(300, 80) * EDSCALE);
	result_dialog->add_child(result_dialog_log);

	add_child(result_dialog);
	result_dialog->hide();

	set_process(true);
	first = true;
	deploy_dumb = false;
	deploy_debug_remote = false;
	debug_collisions = false;
	debug_navigation = false;
	debug_shader_fallbacks = false;
	resume_idx = 0;
	resume_platform = 0;
}
