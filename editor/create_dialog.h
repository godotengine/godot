/*************************************************************************/
/*  create_dialog.h                                                      */
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

#ifndef CREATE_DIALOG_H
#define CREATE_DIALOG_H

#include "editor_help.h"
#include "scene/gui/button.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/item_list.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/tree.h"

class CreateDialogCandidate {
	String type;
	String wb_chars;
	bool is_preferred_type = false;
	bool in_favorites = false;
	bool in_recent = false;

	String _compute_word_boundary_characters() const;

	float _word_score(const String &p_word, const String &p_query) const;

public:
	bool is_valid(const String &p_query) const;
	float compute_score(const String &p_query) const;

	String get_type() const { return type; }

	CreateDialogCandidate() {}
	CreateDialogCandidate(const String &p_type, const bool p_is_preferred_type, const bool in_favorites, const bool in_recent);
};

class FavoriteList : public Tree {
	GDCLASS(FavoriteList, Tree);

	Vector<String> favorites;

	String icon_fallback;
	bool favorites_changed = false;

	Variant _get_drag_data_fw(const Point2 &p_point, Control *p_from);
	bool _can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from);
	void _drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from);

	void _update_tree();

protected:
	void _notification(int p_what) {}
	static void _bind_methods();

public:
	void load_favorites(const String &p_file_id, const String &p_icon_fallback);
	bool toggle_favorite(const String &p_type);
	bool has_favorite(const String &p_type) { return favorites.has(p_type); }
	bool save_favorites(const String &p_file_id);

	FavoriteList();
};

class HistoryList : public ItemList {
	GDCLASS(HistoryList, ItemList);

	Set<String> history;

public:
	void load_history(const String &p_file_id, const String &p_icon_fallback);
	void save_to_history(const String &p_file_id, const String &p_item);
	bool has_history(const String &p_type) const { return history.has(p_type); };
	void clear_history();

	HistoryList();
};

class CreateDialog : public ConfirmationDialog {
	GDCLASS(CreateDialog, ConfirmationDialog);

	LineEdit *search_box;
	Button *favorite;
	Tree *result_tree;
	EditorHelpBit *help_bit;

	FavoriteList *favorite_list;
	HistoryList *history_list;

	String base_type;
	String icon_fallback;
	String preferred_search_result_type;

	Vector<CreateDialogCandidate> candidates;
	Set<StringName> blacklisted_types;

	HashMap<String, TreeItem *> result_tree_types;
	HashMap<String, String> custom_type_parents;
	HashMap<String, int> custom_type_indices;

	Vector<CreateDialogCandidate> _compute_candidates();
	bool _should_hide_type(const String &p_type) const;

	void _update_result_tree();
	void _add_type(const String &p_type, bool p_cpp_type);
	TreeItem *_create_type(const String &p_type, TreeItem *p_parent_type, const bool p_cpp_type);
	void _select_type(const String &p_type);

	String _top_result(const Vector<CreateDialogCandidate> p_candidates, const String &p_query) const;

	void _search_box_input(const Ref<InputEvent> &p_ie);
	void _text_changed(const String &p_newtext);
	void _item_selected();
	void _favorite_toggled();
	void _hide_requested();
	void _confirmed();
	void _history_selected(int p_idx);
	void _history_activated(int p_idx);
	void _favorite_selected();
	void _favorite_activated();

	virtual void cancel_pressed() override;

	void _cleanup();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void popup_create(bool p_dont_clear, bool p_replace_mode = false, const String &p_select_type = "Node");
	Variant instance_selected();
	String get_selected_type();

	void set_base_type(const String &p_base) { base_type = p_base; }
	String get_base_type() const { return base_type; }

	void set_preferred_search_result_type(const String &p_preferred_type) { preferred_search_result_type = p_preferred_type; }

	CreateDialog();
};

#endif
