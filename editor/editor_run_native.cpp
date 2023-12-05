/**************************************************************************/
/*  editor_run_native.cpp                                                 */
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

#include "editor_run_native.h"

#include "editor/editor_node.h"
#include "editor/editor_scale.h"
#include "editor/editor_settings.h"
#include "editor/export/editor_export.h"
#include "editor/export/editor_export_platform.h"
#include "scene/resources/image_texture.h"

void EditorRunNative::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED: {
			remote_debug->set_icon(get_editor_theme_icon(SNAME("PlayRemote")));
		} break;

		case NOTIFICATION_PROCESS: {
			bool changed = EditorExport::get_singleton()->poll_export_platforms() || first;

			if (changed) {
				PopupMenu *popup = remote_debug->get_popup();
				popup->clear();
				for (int i = 0; i < EditorExport::get_singleton()->get_export_platform_count(); i++) {
					Ref<EditorExportPlatform> eep = EditorExport::get_singleton()->get_export_platform(i);
					if (eep.is_null()) {
						continue;
					}
					int dc = MIN(eep->get_options_count(), 9000);
					if (dc > 0) {
						popup->add_icon_item(eep->get_run_icon(), eep->get_name(), -1);
						popup->set_item_disabled(-1, true);
						for (int j = 0; j < dc; j++) {
							popup->add_icon_item(eep->get_option_icon(j), eep->get_option_label(j), 10000 * i + j);
							popup->set_item_tooltip(-1, eep->get_option_tooltip(j));
							popup->set_item_indent(-1, 2);
						}
					}
				}
				if (popup->get_item_count() == 0) {
					remote_debug->set_disabled(true);
					remote_debug->set_tooltip_text(TTR("No Remote Debug export presets configured."));
				} else {
					remote_debug->set_disabled(false);
					remote_debug->set_tooltip_text(TTR("Remote Debug"));
				}

				first = false;
			}
		} break;
	}
}

Error EditorRunNative::start_run_native(int p_id) {
	if (p_id < 0) {
		return OK;
	}

	int platform = p_id / 10000;
	int idx = p_id % 10000;

	if (!EditorNode::get_singleton()->ensure_main_scene(true)) {
		resume_id = p_id;
		return OK;
	}

	Ref<EditorExportPlatform> eep = EditorExport::get_singleton()->get_export_platform(platform);
	ERR_FAIL_COND_V(eep.is_null(), ERR_UNAVAILABLE);

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
		return ERR_UNAVAILABLE;
	}

	emit_signal(SNAME("native_run"), preset);

	int flags = 0;

	bool deploy_debug_remote = is_deploy_debug_remote_enabled();
	bool deploy_dumb = EditorSettings::get_singleton()->get_project_metadata("debug_options", "run_file_server", false);
	bool debug_collisions = EditorSettings::get_singleton()->get_project_metadata("debug_options", "run_debug_collisions", false);
	bool debug_navigation = EditorSettings::get_singleton()->get_project_metadata("debug_options", "run_debug_navigation", false);

	if (deploy_debug_remote) {
		flags |= EditorExportPlatform::DEBUG_FLAG_REMOTE_DEBUG;
	}
	if (deploy_dumb) {
		flags |= EditorExportPlatform::DEBUG_FLAG_DUMB_CLIENT;
	}
	if (debug_collisions) {
		flags |= EditorExportPlatform::DEBUG_FLAG_VIEW_COLLISIONS;
	}
	if (debug_navigation) {
		flags |= EditorExportPlatform::DEBUG_FLAG_VIEW_NAVIGATION;
	}

	eep->clear_messages();
	Error err = eep->run(preset, idx, flags);
	result_dialog_log->clear();
	if (eep->fill_log_messages(result_dialog_log, err)) {
		if (eep->get_worst_message_type() >= EditorExportPlatform::EXPORT_MESSAGE_ERROR) {
			result_dialog->popup_centered_ratio(0.5);
		}
	}
	return err;
}

void EditorRunNative::resume_run_native() {
	start_run_native(resume_id);
}

void EditorRunNative::_bind_methods() {
	ADD_SIGNAL(MethodInfo("native_run", PropertyInfo(Variant::OBJECT, "preset", PROPERTY_HINT_RESOURCE_TYPE, "EditorExportPreset")));
}

bool EditorRunNative::is_deploy_debug_remote_enabled() const {
	return EditorSettings::get_singleton()->get_project_metadata("debug_options", "run_deploy_remote_debug", false);
}

EditorRunNative::EditorRunNative() {
	remote_debug = memnew(MenuButton);
	remote_debug->get_popup()->connect("id_pressed", callable_mp(this, &EditorRunNative::start_run_native));
	remote_debug->set_tooltip_text(TTR("Remote Debug"));
	remote_debug->set_disabled(true);

	add_child(remote_debug);

	result_dialog = memnew(AcceptDialog);
	result_dialog->set_title(TTR("Project Run"));
	result_dialog_log = memnew(RichTextLabel);
	result_dialog_log->set_custom_minimum_size(Size2(300, 80) * EDSCALE);
	result_dialog->add_child(result_dialog_log);

	add_child(result_dialog);
	result_dialog->hide();

	set_process(true);
}
