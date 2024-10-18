/**************************************************************************/
/*  editor_quick_open_dialog.h                                            */
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

#ifndef EDITOR_QUICK_OPEN_DIALOG_H
#define EDITOR_QUICK_OPEN_DIALOG_H

#include "core/string/fuzzy_search.h"
#include "core/templates/oa_hash_map.h"
#include "scene/gui/dialogs.h"

class Button;
class CenterContainer;
class CheckButton;
class EditorFileSystemDirectory;
class LineEdit;
class HFlowContainer;
class MarginContainer;
class PanelContainer;
class ScrollContainer;
class StringName;
class Texture2D;
class TextureRect;
class VBoxContainer;

class QuickOpenResultItem;

enum class QuickOpenDisplayMode {
	GRID,
	LIST,
};

struct QuickOpenResultCandidate {
	String file_name;
	String file_directory;

	Ref<Texture2D> thumbnail;
	Ref<FuzzySearchResult> result;
};

class QuickOpenResultContainer : public VBoxContainer {
	GDCLASS(QuickOpenResultContainer, VBoxContainer)

public:
	void init(const Vector<StringName> &p_base_types);
	void handle_search_box_input(const Ref<InputEvent> &p_ie);
	void set_query_and_update(const String &p_query);
	void update_results();

	bool has_nothing_selected() const;
	String get_selected() const;

	void save_selected_item();
	void cleanup();

	QuickOpenResultContainer();
	~QuickOpenResultContainer();

protected:
	void _notification(int p_what);

private:
	static const int SHOW_ALL_FILES_THRESHOLD = 30;

	Ref<FuzzySearch> fuzzy_search;
	Vector<StringName> base_types;
	Vector<String> filepaths;
	OAHashMap<String, StringName> filetypes;
	Vector<QuickOpenResultCandidate> candidates;

	OAHashMap<StringName, List<QuickOpenResultCandidate>> selected_history;

	String query;
	int selection_index = -1;
	int num_visible_results = 0;
	int max_total_results = 0;

	bool showing_history = false;
	bool never_opened = true;

	QuickOpenDisplayMode content_display_mode = QuickOpenDisplayMode::LIST;
	Vector<QuickOpenResultItem *> result_items;

	ScrollContainer *scroll_container = nullptr;
	VBoxContainer *list = nullptr;
	HFlowContainer *grid = nullptr;

	PanelContainer *panel_container = nullptr;
	CenterContainer *no_results_container = nullptr;
	Label *no_results_label = nullptr;

	Label *file_details_path = nullptr;
	Button *display_mode_toggle = nullptr;
	CheckButton *include_addons_toggle = nullptr;
	CheckButton *fuzzy_search_toggle = nullptr;

	OAHashMap<StringName, Ref<Texture2D>> file_type_icons;

	static QuickOpenDisplayMode get_adaptive_display_mode(const Vector<StringName> &p_base_types);

	void _create_initial_results();
	void _find_filepaths_in_folder(EditorFileSystemDirectory *p_directory, bool p_include_addons);

	void _score_and_sort_candidates();
	void _update_result_items(int p_new_visible_results_count, int p_new_selection_index);

	void _move_selection_index(Key p_key);
	void _select_item(int p_index);

	void _item_input(const Ref<InputEvent> &p_ev, int p_index);

	void _set_display_mode(QuickOpenDisplayMode p_display_mode);
	void _toggle_display_mode();
	void _toggle_include_addons(bool p_pressed);
	void _toggle_fuzzy_search(bool p_pressed);

	static void _bind_methods();
};

class QuickOpenResultGridItem : public VBoxContainer {
	GDCLASS(QuickOpenResultGridItem, VBoxContainer)

public:
	QuickOpenResultGridItem();

	Ref<FuzzySearchResult> result;

	void set_content(const QuickOpenResultCandidate &p_candidate);
	Vector<Rect2i> get_search_highlights();
	void reset();
	void highlight_item(const Color &p_color);
	void remove_highlight();

private:
	TextureRect *thumbnail = nullptr;
	Label *name = nullptr;
};

class QuickOpenResultListItem : public HBoxContainer {
	GDCLASS(QuickOpenResultListItem, HBoxContainer)

public:
	QuickOpenResultListItem();

	Ref<FuzzySearchResult> result;

	void set_content(const QuickOpenResultCandidate &p_candidate);
	Vector<Rect2i> get_search_highlights();
	void reset();
	void highlight_item(const Color &p_color);
	void remove_highlight();

protected:
	void _notification(int p_what);

private:
	static const int CONTAINER_MARGIN = 8;

	MarginContainer *image_container = nullptr;
	VBoxContainer *text_container = nullptr;

	TextureRect *thumbnail = nullptr;
	Label *name = nullptr;
	Label *path = nullptr;
};

class QuickOpenResultItem : public HBoxContainer {
	GDCLASS(QuickOpenResultItem, HBoxContainer)

public:
	QuickOpenResultItem();

	bool enable_highlights = true;

	void set_content(const QuickOpenResultCandidate &p_candidate);
	void set_display_mode(QuickOpenDisplayMode p_display_mode);
	void reset();
	void highlight_item(bool p_enabled);
	void draw_search_highlights();

protected:
	void _notification(int p_what);

private:
	QuickOpenResultListItem *list_item = nullptr;
	QuickOpenResultGridItem *grid_item = nullptr;

	Ref<StyleBox> selected_stylebox;
	Ref<StyleBox> hovering_stylebox;
	Color highlighted_font_color;

	bool is_hovering = false;
	bool is_selected = false;
	bool dirty_highlights = false;

	void _set_enabled(bool p_enabled);
};

class EditorQuickOpenDialog : public AcceptDialog {
	GDCLASS(EditorQuickOpenDialog, AcceptDialog);

public:
	void popup_dialog(const Vector<StringName> &p_base_types, const Callable &p_item_selected_callback);
	EditorQuickOpenDialog();

protected:
	virtual void cancel_pressed() override;
	virtual void ok_pressed() override;

private:
	static String get_dialog_title(const Vector<StringName> &p_base_types);

	LineEdit *search_box = nullptr;
	QuickOpenResultContainer *container = nullptr;

	Callable item_selected_callback;

	void _search_box_text_changed(const String &p_query);
};

#endif // EDITOR_QUICK_OPEN_DIALOG_H
