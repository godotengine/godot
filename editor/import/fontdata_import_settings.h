/*************************************************************************/
/*  fontdata_import_settings.h                                           */
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

#ifndef FONTDATA_IMPORT_SETTINGS_H
#define FONTDATA_IMPORT_SETTINGS_H

#include "editor/editor_file_dialog.h"
#include "editor/editor_inspector.h"

#include "editor/import/resource_importer_fontdata.h"

#include "scene/gui/dialogs.h"
#include "scene/gui/item_list.h"
#include "scene/gui/option_button.h"
#include "scene/gui/split_container.h"
#include "scene/gui/subviewport_container.h"
#include "scene/gui/tree.h"

#include "scene/resources/font.h"
#include "servers/text_server.h"

class FontDataImportSettingsData;

class FontDataImportSettings : public ConfirmationDialog {
	GDCLASS(FontDataImportSettings, ConfirmationDialog)

	enum ItemButton {
		BUTTON_ADD_VAR,
		BUTTON_REMOVE_VAR,
	};

	static FontDataImportSettings *singleton;

	String base_path;

	Ref<FontDataImportSettingsData> import_settings_data;
	List<ResourceImporter::ImportOption> options_variations;
	List<ResourceImporter::ImportOption> options_general;

	Button *add_var = nullptr;
	Button *add_lang = nullptr;
	Button *add_script = nullptr;

	PopupMenu *menu_langs = nullptr;
	PopupMenu *menu_scripts = nullptr;

	EditorInspector *inspector_general = nullptr;
	EditorInspector *inspector_vars = nullptr;

	Tree *vars_list = nullptr;
	TreeItem *vars_list_root = nullptr;

	Tree *lang_list = nullptr;
	TreeItem *lang_list_root = nullptr;

	Tree *script_list = nullptr;
	TreeItem *script_list_root = nullptr;

	Tree *glyph_table = nullptr;
	Tree *glyph_tree = nullptr;
	TreeItem *glyph_root = nullptr;

	Label *label_general = nullptr;
	Label *label_vars = nullptr;
	Label *label_langs = nullptr;
	Label *label_script = nullptr;
	Label *label_glyph = nullptr;

	Ref<Font> font_preview;

	Set<char32_t> selected_chars;

	void _variation_selected();
	void _variation_add();
	void _variation_remove(Object *p_item, int p_column, int p_id);

	void _lang_add();
	void _lang_add_item(int p_option);
	void _lang_remove(Object *p_item, int p_column, int p_id);

	void _script_add();
	void _script_add_item(int p_option);
	void _script_remove(Object *p_item, int p_column, int p_id);

	void _range_edited();
	void _range_selected();

	void _add_glyph_range_item(uint32_t p_start, uint32_t p_end, const String &p_name);

	bool _char_update(uint32_t p_char);
	void _range_update(uint32_t p_start, uint32_t p_end, bool p_select);

	void _edit_range(uint32_t p_start, uint32_t p_end);
	void _glyph_selected();

	void _re_import();

	String _pad_zeros(const String &p_hex) const;

protected:
	void _notification(int p_what);

public:
	void open_settings(const String &p_path);
	static FontDataImportSettings *get_singleton();

	FontDataImportSettings();
};

#endif // FONTDATA_IMPORT_SETTINGS_H
