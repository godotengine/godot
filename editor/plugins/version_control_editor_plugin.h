/*************************************************************************/
/*  version_control_editor_plugin.h                                      */
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

#ifndef VERSION_CONTROL_EDITOR_PLUGIN_H
#define VERSION_CONTROL_EDITOR_PLUGIN_H

#include "editor/editor_plugin.h"
#include "editor/editor_vcs_interface.h"
#include "scene/gui/check_button.h"
#include "scene/gui/container.h"
#include "scene/gui/file_dialog.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/rich_text_label.h"
#include "scene/gui/separator.h"
#include "scene/gui/tab_container.h"
#include "scene/gui/text_edit.h"
#include "scene/gui/tree.h"

class VersionControlEditorPlugin : public EditorPlugin {
	GDCLASS(VersionControlEditorPlugin, EditorPlugin)

	static VersionControlEditorPlugin *singleton;

	// Settings UI
	ConfirmationDialog *metadata_dialog = nullptr;
	OptionButton *metadata_selection = nullptr;
	AcceptDialog *set_up_dialog = nullptr;
	CheckButton *toggle_vcs_choice = nullptr;
	OptionButton *set_up_choice = nullptr;
	HSeparator *set_up_hs = nullptr;
	VBoxContainer *vcs_plugin_settings = nullptr;

	void _create_vcs_metadata_files();
	void _toggle_vcs_integration(bool p_toggled);

	void _instantiate_plugin_and_ui(String p_plugin_name);
	void _destroy_plugin_and_ui();

	bool _assign_plugin_singleton(String p_plugin_name);

protected:
	static void _bind_methods();

public:
	static VersionControlEditorPlugin *get_singleton();

	void popup_vcs_metadata_dialog();
	void popup_vcs_set_up_dialog(const Control *p_gui_base);

	List<StringName> fetch_available_vcs_plugin_names();

	VersionControlEditorPlugin();
	~VersionControlEditorPlugin();
};

#endif // VERSION_CONTROL_EDITOR_PLUGIN_H
