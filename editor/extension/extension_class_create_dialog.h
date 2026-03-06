/**************************************************************************/
/*  extension_class_create_dialog.h                                       */
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

#include "editor/extension/editor_extension_source_code_plugin.h"
#include "editor/inspector/editor_inspector.h"
#include "scene/gui/check_box.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/option_button.h"

class CreateDialog;
class EditorFileDialog;
class EditorValidationPanel;
class LineEdit;

class ExtensionClassCreateDialog : public ConfirmationDialog {
	GDCLASS(ExtensionClassCreateDialog, ConfirmationDialog);

	enum {
		MSG_ID_CLASS,
		MSG_ID_INHERITED_CLASS,
		MSG_ID_PATH,
		MSG_ID_TEMPLATE,
		MSG_ID_TEMPLATE_OPTIONS,
	};

	struct PathEdit {
		Label *label = nullptr;
		HBoxContainer *hb = nullptr;
		LineEdit *line_edit = nullptr;
		Button *button = nullptr;
	};

	enum TemplateLocation {
		TEMPLATE_LOCATION_PLUGIN,
		TEMPLATE_LOCATION_EDITOR,
		TEMPLATE_LOCATION_PROJECT,
	};

	class TemplateOptions : public Object {
		GDCLASS(TemplateOptions, Object);

		struct VariantContainer {
			PropertyInfo info;
			Variant variant;
			Variant initial;
		};

	private:
		HashMap<StringName, VariantContainer> options;

	protected:
		bool _set(const StringName &p_name, const Variant &p_value);
		bool _get(const StringName &p_name, Variant &r_ret) const;
		void _get_property_list(List<PropertyInfo> *p_list) const;
		bool _property_can_revert(const StringName &p_name) const;
		bool _property_get_revert(const StringName &p_name, Variant &r_property) const;

	public:
		void config(TypedArray<Dictionary> p_options);
	};

	bool initialized = false;

	GridContainer *gc = nullptr;
	OptionButton *language_menu = nullptr;
	LineEdit *class_name_edit = nullptr;
	LineEdit *inherited_class_name_edit = nullptr;
	Button *inherited_class_search_button = nullptr;
	OptionButton *template_menu = nullptr;
	CheckBox *use_templates = nullptr;
	Label *template_options_label = nullptr;
	EditorInspector *template_options_inspector = nullptr;
	TemplateOptions *template_options_object = nullptr;
	LocalVector<PathEdit> path_edits;

	EditorValidationPanel *validation_panel = nullptr;
	AcceptDialog *alert = nullptr;
	CreateDialog *select_inherited_class_dialog = nullptr;
	EditorFileDialog *select_path_dialog = nullptr;

	int default_language = -1;
	bool is_using_templates = true;

	int path_edit_insert_index = 0;
	int browsing_path_index = 0;

	const String VALIDATION_SCOPE_CLASS_NAME = "class_name";
	const String VALIDATION_SCOPE_INHERITED_CLASS_NAME = "inherited_class_name";
	const String VALIDATION_SCOPE_TEMPLATE_OPTIONS = "template_options";
	const String VALIDATION_SCOPE_PATH = "path";
	ValidationContext *validation_context = nullptr;

	Ref<EditorExtensionSourceCodePlugin> selected_source_code_plugin;

	String base_type;
	String base_path;

	void _language_changed(int p_language_index);
	void _class_name_changed(const String &p_class_name);
	void _inherited_class_name_changed(const String &p_class_name);
	void _use_template_pressed();
	void _template_changed(int p_template_index);
	void _template_option_changed(const String &p_option_name);
	void _path_changed(int p_path_index, const String &p_path);
	void _path_changed_bind(const String &p_path, int p_path_index);

	void _validate_class_name(ValidationContext *p_validation_context, const String &p_class_name);
	void _validate_inherited_class_name(ValidationContext *p_validation_context, const String &p_class_name);
	void _validate_template_option(ValidationContext *p_validation_context, const String &p_template_id, const String &p_option_name, const Variant &p_value);
	void _validate_path(ValidationContext *p_validation_context, int p_path_index, const String &p_path);

	void _browse_inherited_class();
	void _inherited_class_selected();

	void _browse_path(int p_path_index);
	void _path_selected(const String &p_path);

	virtual void ok_pressed() override;

	void _create_new();

	String _get_template_location_label(const TemplateLocation &p_template_location) const;
	String _get_template_display_name(const String &p_template_id_or_path, const TemplateLocation &p_template_location) const;
	PackedStringArray _get_user_template_files(const String &p_base_class_name, const String &p_root_path) const;
	PackedStringArray _get_templates_from_location(const TemplateLocation &p_template_location, const String &p_base_class_name) const;

	void _update_language_menu();
	void _update_template_menu();
	void _update_template_label(const String &p_template_name, const String &p_template_description);
	void _update_path_edits();
	void _adjust_paths();
	void _update_validation_messages(const String &p_scope, int p_msg_id, const String &p_ok_message = "");
	void _update_dialog();
	void _focus_on_class_name_edit();

	PathEdit _create_path_edit(int p_path_index);
	void _update_path_edit(int p_path_index, const PathEdit &p_path_edit);
	void _free_path_edit(const PathEdit &p_path_edit);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void config(const String &p_base_name, const String &p_base_path);
	ExtensionClassCreateDialog();
	~ExtensionClassCreateDialog();
};
