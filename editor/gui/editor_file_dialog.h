/**************************************************************************/
/*  editor_file_dialog.h                                                  */
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

#pragma once

#include "scene/gui/file_dialog.h"

class DependencyRemoveDialog;

class EditorFileDialog : public FileDialog {
	GDCLASS(EditorFileDialog, FileDialog);

	DependencyRemoveDialog *dependency_remove_dialog = nullptr;

protected:
	virtual void _item_menu_id_pressed(int p_option) override;
	virtual void _dir_contents_changed() override;

	virtual bool _should_use_native_popup() const override;
	virtual bool _should_hide_file(const String &p_file) const override;
	virtual Color _get_folder_color(const String &p_path) const override;

	static void _bind_methods();
	void _validate_property(PropertyInfo &p_property) const;
	void _notification(int p_what);

public:
#ifndef DISABLE_DEPRECATED
	void add_side_menu(Control *p_menu, const String &p_title = "") { ERR_FAIL_MSG("add_side_menu() is kept for compatibility and does nothing. For similar functionality, you can show another dialog after file dialog."); }
	void set_disable_overwrite_warning(bool p_disable) { set_customization_flag_enabled(CUSTOMIZATION_OVERWRITE_WARNING, !p_disable); }
	bool is_overwrite_warning_disabled() const { return !is_customization_flag_enabled(CUSTOMIZATION_OVERWRITE_WARNING); }
#endif
};
