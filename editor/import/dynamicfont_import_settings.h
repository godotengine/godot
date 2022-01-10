/*************************************************************************/
/*  dynamicfont_import_settings.h                                        */
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

#ifndef FONTDATA_IMPORT_SETTINGS_H
#define FONTDATA_IMPORT_SETTINGS_H

#include "editor/editor_file_dialog.h"
#include "editor/editor_inspector.h"

#include "editor/import/resource_importer_dynamicfont.h"

#include "scene/gui/dialogs.h"
#include "scene/gui/item_list.h"
#include "scene/gui/option_button.h"
#include "scene/gui/split_container.h"
#include "scene/gui/subviewport_container.h"
#include "scene/gui/tab_container.h"
#include "scene/gui/text_edit.h"
#include "scene/gui/tree.h"

#include "scene/resources/font.h"
#include "servers/text_server.h"

class DynamicFontImportSettingsData;

class DynamicFontImportSettings : public ConfirmationDialog {
	GDCLASS(DynamicFontImportSettings, ConfirmationDialog)
	friend class DynamicFontImportSettingsData;

	enum ItemButton {
		BUTTON_ADD_VAR,
		BUTTON_REMOVE_VAR,
	};

	static DynamicFontImportSettings *singleton;

	String base_path;

	Ref<DynamicFontImportSettingsData> import_settings_data;
	List<ResourceImporter::ImportOption> options_variations;
	List<ResourceImporter::ImportOption> options_general;

	// Root layout
	Label *label_warn = nullptr;
	TabContainer *main_pages = nullptr;

	// Page 1 layout: Rendering Options
	Label *page1_description = nullptr;
	Label *font_preview_label = nullptr;
	EditorInspector *inspector_general = nullptr;

	void _main_prop_changed(const String &p_edited_property);

	// Page 2 layout: Configurations
	Label *page2_description = nullptr;
	Label *label_vars = nullptr;
	Button *add_var = nullptr;
	Tree *vars_list = nullptr;
	TreeItem *vars_list_root = nullptr;
	EditorInspector *inspector_vars = nullptr;

	void _variation_add();
	void _variation_selected();
	void _variation_remove(Object *p_item, int p_column, int p_id);
	void _variation_changed(const String &p_edited_property);
	void _variations_validate();

	// Page 3 layout: Text to select glyphs
	Label *page3_description = nullptr;
	Label *label_glyphs = nullptr;
	TextEdit *text_edit = nullptr;
	LineEdit *ftr_edit = nullptr;
	LineEdit *lang_edit = nullptr;

	void _change_text_opts();
	void _glyph_text_selected();
	void _glyph_clear();

	// Page 4 layout: Character map
	Label *page4_description = nullptr;
	Tree *glyph_table = nullptr;
	Tree *glyph_tree = nullptr;
	TreeItem *glyph_root = nullptr;

	void _glyph_selected();
	void _range_edited();
	void _range_selected();
	void _edit_range(int32_t p_start, int32_t p_end);
	bool _char_update(int32_t p_char);
	void _range_update(int32_t p_start, int32_t p_end);

	// Page 5 layout: Metadata override
	Label *page5_description = nullptr;
	Button *add_lang = nullptr;
	Button *add_script = nullptr;
	Button *add_ot = nullptr;

	PopupMenu *menu_langs = nullptr;
	PopupMenu *menu_scripts = nullptr;
	PopupMenu *menu_ot = nullptr;
	PopupMenu *menu_ot_ss = nullptr;
	PopupMenu *menu_ot_cv = nullptr;
	PopupMenu *menu_ot_cu = nullptr;

	Tree *lang_list = nullptr;
	TreeItem *lang_list_root = nullptr;
	Label *label_langs = nullptr;

	Tree *script_list = nullptr;
	TreeItem *script_list_root = nullptr;
	Label *label_script = nullptr;

	Tree *ot_list = nullptr;
	TreeItem *ot_list_root = nullptr;
	Label *label_ot = nullptr;

	void _lang_add();
	void _lang_add_item(int p_option);
	void _lang_remove(Object *p_item, int p_column, int p_id);

	void _script_add();
	void _script_add_item(int p_option);
	void _script_remove(Object *p_item, int p_column, int p_id);

	void _ot_add();
	void _ot_add_item(int p_option);
	void _ot_remove(Object *p_item, int p_column, int p_id);

	// Common

	void _add_glyph_range_item(int32_t p_start, int32_t p_end, const String &p_name);

	Ref<Font> font_preview;
	Ref<Font> font_main;

	Set<char32_t> selected_chars;
	Set<int32_t> selected_glyphs;

	void _re_import();

	String _pad_zeros(const String &p_hex) const;

protected:
	void _notification(int p_what);

public:
	void open_settings(const String &p_path);
	static DynamicFontImportSettings *get_singleton();

	DynamicFontImportSettings();
};

#endif // FONTDATA_IMPORT_SETTINGS_H
