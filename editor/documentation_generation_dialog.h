/*************************************************************************/
/*  documentation_generation_dialog.h                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef DOCUMENTATION_GENERATION_DIALOG_H
#define DOCUMENTATION_GENERATION_DIALOG_H

#include "editor/editor_file_dialog.h"

#include "editor/doc_data.h"
#include "scene/gui/check_box.h"
#include "scene/gui/grid_container.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/option_button.h"
#include "scene/gui/panel_container.h"

class DocumentationGenerationDialog : public ConfirmationDialog {
	GDCLASS(DocumentationGenerationDialog, ConfirmationDialog);

	GridContainer *gc;
	Label *input_error_label;
	Label *target_error_label;
	Label *language_not_supported;
	PanelContainer *status_panel;
	OptionButton *language_menu;
	LineEdit *input_dir_path;
	Button *input_dir_button;
	LineEdit *target_dir_path;
	Button *target_dir_button;
	OptionButton *output_format;
	CheckBox *ignore_invalid_scripts;
	CheckBox *exclude_private;
	CheckBox *documented_only;

	AcceptDialog *alert;
	EditorFileDialog *file_browse;

	bool is_browsing_input = false;
	bool is_input_path_valid = false;
	bool is_target_path_valid = false;
	bool is_language_supported = false;
	int default_language = -1;

	DocData::ClassDoc _apply_options_filter(const DocData::ClassDoc &p_class);
	bool _generate(const String &p_input, const String &p_target, bool p_recursive_call = false);

	void _lang_changed(int p_lang);
	void _path_changed(const String &p_path, bool p_is_input);
	void _path_entered(const String &p_path, bool p_is_input);
	void _browse_path(bool p_is_input);
	void _dir_selected(const String &p_dir);
	void _update_dialog();
	virtual void ok_pressed() override;

protected:
	void _theme_changed();
	void _notification(int p_what);
	static void _bind_methods();

public:
	enum Options {
		OPT_UNKNOWN = -1,
		EXCLUDE_PRIVATE = 1 << 0,
		DOCUMENTED_ONLY = 1 << 1,
		IGNORE_INVALID = 1 << 2,
	};

	enum OutputFormats {
		FMT_UNKNOWN = -1,
		FMT_XML = 0,
		FMT_JSON = 1,
	};

	void config(const String &p_input_dir, const String &p_output_dir, int p_output_format = FMT_UNKNOWN, int p_options = OPT_UNKNOWN);
	DocumentationGenerationDialog();
};

#endif //  DOCUMENTATION_GENERATION_DIALOG_H
