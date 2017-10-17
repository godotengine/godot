/*************************************************************************/
/*  export_template_manager.h                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
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
#ifndef EXPORT_TEMPLATE_MANAGER_H
#define EXPORT_TEMPLATE_MANAGER_H

#include "editor/editor_settings.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/file_dialog.h"
#include "scene/gui/scroll_container.h"

class ExportTemplateVersion;

class ExportTemplateManager : public ConfirmationDialog {
	GDCLASS(ExportTemplateManager, ConfirmationDialog)

	ScrollContainer *installed_scroll;
	VBoxContainer *installed_vb;
	HBoxContainer *current_hb;
	FileDialog *template_open;

	ConfirmationDialog *remove_confirm;
	String to_remove;

	void _update_template_list();

	void _download_template(const String &p_version);
	void _uninstall_template(const String &p_version);
	void _uninstall_template_confirm();

	virtual void ok_pressed();
	void _install_from_file(const String &p_file);

protected:
	static void _bind_methods();

public:
	void popup_manager();

	ExportTemplateManager();
};

#endif // EXPORT_TEMPLATE_MANAGER_H
