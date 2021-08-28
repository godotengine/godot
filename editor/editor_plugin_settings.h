/*************************************************************************/
/*  editor_plugin_settings.h                                             */
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

#ifndef EDITORPLUGINSETTINGS_H
#define EDITORPLUGINSETTINGS_H

#include "core/object/undo_redo.h"
#include "editor/plugin_config_dialog.h"
#include "editor_data.h"
#include "property_editor.h"
#include "scene/gui/dialogs.h"

class EditorPluginSettings : public VBoxContainer {
	GDCLASS(EditorPluginSettings, VBoxContainer);

	enum {
		BUTTON_PLUGIN_EDIT
	};

	PluginConfigDialog *plugin_config_dialog;
	Button *create_plugin;
	Button *update_list;
	Tree *plugin_list;
	bool updating;

	void _plugin_activity_changed();
	void _create_clicked();
	void _cell_button_pressed(Object *p_item, int p_column, int p_id);

	static Vector<String> _get_plugins(const String &p_dir);

protected:
	void _notification(int p_what);

	static void _bind_methods();

public:
	void update_plugins();

	EditorPluginSettings();
};

#endif // EDITORPLUGINSETTINGS_H
