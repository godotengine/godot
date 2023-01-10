/**************************************************************************/
/*  import_defaults_editor.h                                              */
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

#ifndef IMPORT_DEFAULTS_EDITOR_H
#define IMPORT_DEFAULTS_EDITOR_H

#include "core/undo_redo.h"
#include "editor/editor_data.h"
#include "editor/editor_plugin_settings.h"
#include "editor/editor_sectioned_inspector.h"
#include "editor_autoload_settings.h"
#include "scene/gui/center_container.h"
#include "scene/gui/option_button.h"

class ImportDefaultsEditorSettings;

class ImportDefaultsEditor : public VBoxContainer {
	GDCLASS(ImportDefaultsEditor, VBoxContainer)

	OptionButton *importers;
	Button *save_defaults;
	Button *reset_defaults;

	EditorInspector *inspector;

	ImportDefaultsEditorSettings *settings;

	void _update_importer();
	void _importer_selected(int p_index);

	void _reset();
	void _save();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void clear();

	ImportDefaultsEditor();
	~ImportDefaultsEditor();
};

#endif // IMPORT_DEFAULTS_EDITOR_H
