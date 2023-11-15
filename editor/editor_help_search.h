/**************************************************************************/
/*  editor_help_search.h                                                  */
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

#ifndef EDITOR_HELP_SEARCH_H
#define EDITOR_HELP_SEARCH_H

#include "core/templates/rb_map.h"
#include "editor/code_editor.h"
#include "editor/editor_help.h"
#include "editor/editor_plugin.h"
#include "scene/gui/option_button.h"
#include "scene/gui/tree.h"

class EditorHelpSearch : public ConfirmationDialog {
	GDCLASS(EditorHelpSearch, ConfirmationDialog);

	enum SearchFlags {
		SEARCH_CLASSES = 1 << 0,
		SEARCH_CONSTRUCTORS = 1 << 1,
		SEARCH_METHODS = 1 << 2,
		SEARCH_OPERATORS = 1 << 3,
		SEARCH_SIGNALS = 1 << 4,
		SEARCH_CONSTANTS = 1 << 5,
		SEARCH_PROPERTIES = 1 << 6,
		SEARCH_THEME_ITEMS = 1 << 7,
		SEARCH_ANNOTATIONS = 1 << 8,
		SEARCH_ALL = SEARCH_CLASSES | SEARCH_CONSTRUCTORS | SEARCH_METHODS | SEARCH_OPERATORS | SEARCH_SIGNALS | SEARCH_CONSTANTS | SEARCH_PROPERTIES | SEARCH_THEME_ITEMS | SEARCH_ANNOTATIONS,
		SEARCH_CASE_SENSITIVE = 1 << 29,
		SEARCH_SHOW_HIERARCHY = 1 << 30
	};

	LineEdit *search_box = nullptr;
	Button *case_sensitive_button = nullptr;
	Button *hierarchy_button = nullptr;
	OptionButton *filter_combo = nullptr;
	Tree *results_tree = nullptr;
	bool old_search = false;
	String old_term;

	class Runner;
	Ref<Runner> search;

	void _update_results();

	void _search_box_gui_input(const Ref<InputEvent> &p_event);
	void _search_box_text_changed(const String &p_text);
	void _filter_combo_item_selected(int p_option);
	void _confirmed();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void popup_dialog();
	void popup_dialog(const String &p_term);

	EditorHelpSearch();
};

class EditorHelpSearch::Runner : public RefCounted {
	enum Phase {
		PHASE_MATCH_CLASSES_INIT,
		PHASE_MATCH_CLASSES,
		PHASE_CLASS_ITEMS_INIT,
		PHASE_CLASS_ITEMS,
		PHASE_MEMBER_ITEMS_INIT,
		PHASE_MEMBER_ITEMS,
		PHASE_SELECT_MATCH,
		PHASE_MAX
	};
	int phase = 0;

	struct ClassMatch {
		DocData::ClassDoc *doc = nullptr;
		bool name = false;
		Vector<DocData::MethodDoc *> constructors;
		Vector<DocData::MethodDoc *> methods;
		Vector<DocData::MethodDoc *> operators;
		Vector<DocData::MethodDoc *> signals;
		Vector<DocData::ConstantDoc *> constants;
		Vector<DocData::PropertyDoc *> properties;
		Vector<DocData::ThemeItemDoc *> theme_properties;
		Vector<DocData::MethodDoc *> annotations;

		bool required() {
			return name || methods.size() || signals.size() || constants.size() || properties.size() || theme_properties.size() || annotations.size();
		}
	};

	Control *ui_service = nullptr;
	Tree *results_tree = nullptr;
	String term;
	Vector<String> terms;
	int search_flags;

	Color disabled_color;

	HashMap<String, DocData::ClassDoc>::Iterator iterator_doc;
	HashMap<String, ClassMatch> matches;
	HashMap<String, ClassMatch>::Iterator iterator_match;
	TreeItem *root_item = nullptr;
	HashMap<String, TreeItem *> class_items;
	TreeItem *matched_item = nullptr;
	float match_highest_score = 0;

	bool _is_class_disabled_by_feature_profile(const StringName &p_class);

	bool _slice();
	bool _phase_match_classes_init();
	bool _phase_match_classes();
	bool _phase_class_items_init();
	bool _phase_class_items();
	bool _phase_member_items_init();
	bool _phase_member_items();
	bool _phase_select_match();

	String _build_method_tooltip(const DocData::ClassDoc *p_class_doc, const DocData::MethodDoc *p_doc) const;

	void _match_method_name_and_push_back(Vector<DocData::MethodDoc> &p_methods, Vector<DocData::MethodDoc *> *r_match_methods);
	bool _all_terms_in_name(String name);
	bool _match_string(const String &p_term, const String &p_string) const;
	void _match_item(TreeItem *p_item, const String &p_text);
	TreeItem *_create_class_hierarchy(const ClassMatch &p_match);
	TreeItem *_create_class_item(TreeItem *p_parent, const DocData::ClassDoc *p_doc, bool p_gray);
	TreeItem *_create_method_item(TreeItem *p_parent, const DocData::ClassDoc *p_class_doc, const String &p_text, const DocData::MethodDoc *p_doc);
	TreeItem *_create_signal_item(TreeItem *p_parent, const DocData::ClassDoc *p_class_doc, const DocData::MethodDoc *p_doc);
	TreeItem *_create_annotation_item(TreeItem *p_parent, const DocData::ClassDoc *p_class_doc, const String &p_text, const DocData::MethodDoc *p_doc);
	TreeItem *_create_constant_item(TreeItem *p_parent, const DocData::ClassDoc *p_class_doc, const DocData::ConstantDoc *p_doc);
	TreeItem *_create_property_item(TreeItem *p_parent, const DocData::ClassDoc *p_class_doc, const DocData::PropertyDoc *p_doc);
	TreeItem *_create_theme_property_item(TreeItem *p_parent, const DocData::ClassDoc *p_class_doc, const DocData::ThemeItemDoc *p_doc);
	TreeItem *_create_member_item(TreeItem *p_parent, const String &p_class_name, const String &p_icon, const String &p_name, const String &p_text, const String &p_type, const String &p_metatype, const String &p_tooltip, bool is_deprecated, bool is_experimental);

public:
	bool work(uint64_t slot = 100000);

	Runner(Control *p_icon_service, Tree *p_results_tree, const String &p_term, int p_search_flags);
};

#endif // EDITOR_HELP_SEARCH_H
