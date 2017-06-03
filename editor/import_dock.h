/*************************************************************************/
/*  import_dock.h                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifndef IMPORTDOCK_H
#define IMPORTDOCK_H

#include "editor_file_system.h"
#include "io/resource_import.h"
#include "property_editor.h"
#include "scene/gui/box_container.h"
#include "scene/gui/option_button.h"
#include "scene/gui/popup_menu.h"

class ImportDockParameters;
class ImportDock : public VBoxContainer {
	GDCLASS(ImportDock, VBoxContainer)

	LineEdit *imported;
	OptionButton *import_as;
	MenuButton *preset;
	PropertyEditor *import_opts;

	List<PropertyInfo> properties;
	Map<StringName, Variant> property_values;

	Button *import;

	ImportDockParameters *params;

	void _preset_selected(int p_idx);

	void _reimport();

protected:
	static void _bind_methods();

public:
	void set_edit_path(const String &p_path);
	void set_edit_multiple_paths(const Vector<String> &p_paths);
	void clear();

	ImportDock();
	~ImportDock();
};

#endif // IMPORTDOCK_H
