/*************************************************************************/
/*  create_dialog.h                                                      */
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

#ifndef CREATE_DIALOG_H
#define CREATE_DIALOG_H

#include "editor_help.h"
#include "scene/gui/button.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/item_list.h"
#include "scene/gui/label.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/tree.h"

class CreateDialog : public ConfirmationDialog {
	GDCLASS(CreateDialog, ConfirmationDialog);

	Vector<String> favorite_list;
	Tree *favorites;
	Tree *recent;

	Button *favorite;
	LineEdit *search_box;
	Tree *search_options;
	HashMap<String, TreeItem *> search_options_types;
	HashMap<String, RES> search_loaded_scripts;
	bool is_replace_mode;
	String base_type;
	String preferred_search_result_type;
	EditorHelpBit *help_bit;
	List<StringName> type_list;
	Set<StringName> type_blacklist;

	void _item_selected();

	void _update_search();
	void _update_favorite_list();
	void _save_favorite_list();
	void _favorite_toggled();

	void _history_selected();
	void _favorite_selected();

	void _history_activated();
	void _favorite_activated();

	void _sbox_input(const Ref<InputEvent> &p_ie);

	void _confirmed();
	void _text_changed(const String &p_newtext);

	void add_type(const String &p_type, HashMap<String, TreeItem *> &p_types, TreeItem *p_root, TreeItem **to_select);

	void select_type(const String &p_type);

	Variant get_drag_data_fw(const Point2 &p_point, Control *p_from);
	bool can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const;
	void drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from);

	bool _is_class_disabled_by_feature_profile(const StringName &p_class);
	bool _is_type_prefered(const String &type);

protected:
	void _notification(int p_what);
	static void _bind_methods();

	void _save_and_update_favorite_list();

public:
	Variant instance_selected();
	String get_selected_type();

	void set_base_type(const String &p_base);
	String get_base_type() const;

	void set_preferred_search_result_type(const String &p_preferred_type);
	String get_preferred_search_result_type();

	void popup_create(bool p_dont_clear, bool p_replace_mode = false, const String &p_select_type = "Node");

	CreateDialog();
};

#endif // CREATE_DIALOG_H
