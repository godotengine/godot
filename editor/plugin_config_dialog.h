/*************************************************************************/
/*  plugin_config_dialog.h                                               */
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

#ifndef PLUGIN_CONFIG_DIALOG_H
#define PLUGIN_CONFIG_DIALOG_H

#include "scene/gui/check_box.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/option_button.h"
#include "scene/gui/text_edit.h"
#include "scene/gui/texture_rect.h"

class PluginConfigDialog : public ConfirmationDialog {
	GDCLASS(PluginConfigDialog, ConfirmationDialog);

	LineEdit *name_edit;
	LineEdit *subfolder_edit;
	TextEdit *desc_edit;
	LineEdit *author_edit;
	LineEdit *version_edit;
	OptionButton *script_option_edit;
	LineEdit *script_edit;
	CheckBox *active_edit;

	TextureRect *name_validation;
	TextureRect *subfolder_validation;
	TextureRect *script_validation;

	bool _edit_mode;

	void _clear_fields();
	void _on_confirmed();
	void _on_cancelled();
	void _on_language_changed(const int p_language);
	void _on_required_text_changed(const String &p_text);

	static String _to_absolute_plugin_path(const String &p_plugin_name);

protected:
	virtual void _notification(int p_what);
	static void _bind_methods();

public:
	void config(const String &p_config_path);

	PluginConfigDialog();
	~PluginConfigDialog();
};

#endif // PLUGIN_CONFIG_DIALOG_H
