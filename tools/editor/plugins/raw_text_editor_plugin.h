/*************************************************************************/
/*  raw_text_editor_plugin.h                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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
#ifndef RAWTEXT_EDITOR_PLUGIN_H
#define RAWTEXT_EDITOR_PLUGIN_H

#include "tools/editor/code_editor.h"
#include "tools/editor/editor_plugin.h"
#include "scene/gui/tab_container.h"
#include "scene/gui/text_edit.h"
#include "scene/gui/label.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/option_button.h"
#include "scene/main/timer.h"
#include "scene/resources/raw_text.h"

struct HighlightConfig {
	enum {
		LANG_NONE,
		LANG_CPP,
		LANG_CSHARP,
		LANG_CSS,
		LANG_JAVASCRIPT,
		LANG_JSON,
		LANG_PYTHON,
		LANG_SHELL,
		LANG_TOML,
		LANG_XML,
		LANG_YAML,
	};
	String line_comment;
	String multi_line_comment_beg;
	String multi_line_comment_end;
	List<String> string_delimiters;
	List<String> keywords;
	bool filled;
};

class RawTextSourceEditor : public CodeTextEditor {

	OBJ_TYPE( RawTextSourceEditor, CodeTextEditor );

	friend class RawTextEditor;

	Ref<RawText> text;

protected:
	String language;
	virtual void _validate_script();
	static void _bind_methods();
	virtual void _load_theme_settings();
	HighlightConfig get_highlight(const String& extension);

public:

	Ref<RawText>& get_edited_text();
	void set_edited_text(const Ref<RawText>& p_text);
	void set_highlight_language(const String& lang);
	RawTextSourceEditor();
	virtual ~RawTextSourceEditor() {}
};


class RawTextEditor : public Control {

	OBJ_TYPE(RawTextEditor, Control );

	enum {

		EDIT_UNDO,
		EDIT_REDO,
		EDIT_CUT,
		EDIT_COPY,
		EDIT_PASTE,
		EDIT_SELECT_ALL,
		SEARCH_FIND,
		SEARCH_FIND_NEXT,
		SEARCH_FIND_PREV,
		SEARCH_REPLACE,
		SEARCH_GOTO_LINE,
	};
	mutable Ref<RawText> text;
	MenuButton *edit_menu;
	MenuButton *search_menu;
	MenuButton *settings_menu;
	Label *title_label;
	OptionButton *lang_option;
	uint64_t idle;

	GotoLineDialog *goto_line_dialog;
	ConfirmationDialog *erase_tab_confirm;

	TextureButton *close;

	RawTextSourceEditor *text_editor;

	void _menu_option(int p_optin);
	void _close_callback();
	void _editor_settings_changed();
	void _update_lang_option();
	void _update_title();

protected:
	void _text_changed();
	void _highlight_selected(int);
	void _notification(int p_what);
	static void _bind_methods();
public:

	void apply_text();

	void ensure_select_current();
	void edit(const Ref<RawText>& p_text);

	Dictionary get_state() const;
	void set_state(const Dictionary& p_state);
	void clear();

	virtual Size2 get_minimum_size() const { return Size2(0,300); }
	void save_external_data();

	RawTextEditor();
};

class RawTextEditorPlugin : public EditorPlugin {

	OBJ_TYPE( RawTextEditorPlugin, EditorPlugin );

	RawTextEditor *text_editor;
	EditorNode *editor;
public:

	virtual String get_name() const { return "RawText"; }
	bool has_main_screen() const { return false; }
	virtual void edit(Object *p_node);
	virtual bool handles(Object *p_node) const;
	virtual void make_visible(bool p_visible);
	virtual void selected_notify();

	Dictionary get_state() const;
	virtual void set_state(const Dictionary& p_state);
	virtual void clear();

	virtual void save_external_data();
	virtual void apply_changes();

	RawTextEditorPlugin(EditorNode *p_node, bool p_2d);
	~RawTextEditorPlugin();

};
#endif
