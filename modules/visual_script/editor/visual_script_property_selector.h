/*************************************************************************/
/*  visual_script_property_selector.h                                    */
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

#ifndef VISUALSCRIPT_PROPERTYSELECTOR_H
#define VISUALSCRIPT_PROPERTYSELECTOR_H

#include "../visual_script.h"
#include "editor/editor_help.h"
#include "editor/property_editor.h"
#include "scene/gui/rich_text_label.h"

class VisualScriptPropertySelector : public ConfirmationDialog {
	GDCLASS(VisualScriptPropertySelector, ConfirmationDialog);

	enum SearchFlags {
		SEARCH_CLASSES = 1 << 0,
		SEARCH_CONSTRUCTORS = 1 << 1,
		SEARCH_METHODS = 1 << 2,
		SEARCH_OPERATORS = 1 << 3,
		SEARCH_SIGNALS = 1 << 4,
		SEARCH_CONSTANTS = 1 << 5,
		SEARCH_PROPERTIES = 1 << 6,
		SEARCH_THEME_ITEMS = 1 << 7,
		SEARCH_VISUAL_SCRIPT_NODES = 1 << 8,
		SEARCH_ALL = SEARCH_CLASSES | SEARCH_CONSTRUCTORS | SEARCH_METHODS | SEARCH_OPERATORS | SEARCH_SIGNALS | SEARCH_CONSTANTS | SEARCH_PROPERTIES | SEARCH_THEME_ITEMS,
		SEARCH_CASE_SENSITIVE = 1 << 29,
		SEARCH_SHOW_HIERARCHY = 1 << 30,
	};

	enum ScopeFlags {
		SCOPE_BASE = 1 << 0,
		SCOPE_INHERITERS = 1 << 1,
		SCOPE_UNRELATED = 1 << 2,
		SCOPE_RELATED = SCOPE_BASE | SCOPE_INHERITERS,
		SCOPE_ALL = SCOPE_BASE | SCOPE_INHERITERS | SCOPE_UNRELATED
	};

	LineEdit *search_box;

	Button *case_sensitive_button;
	Button *hierarchy_button;

	Button *search_visual_script_nodes;
	Button *search_classes;
	Button *search_operators;

	Button *search_methods;
	Button *search_signals;
	Button *search_constants;
	Button *search_properties;
	Button *search_theme_items;

	OptionButton *scope_combo;
	Tree *results_tree;

	class SearchRunner;
	Ref<SearchRunner> search_runner;

	void _update_icons();

	void _sbox_input(const Ref<InputEvent> &p_ie);
	void _update_results_i(int p_int);
	void _update_results_s(String p_string);
	void _update_results();

	void _confirmed();
	void _item_selected();
	void _hide_requested();

	EditorHelpBit *help_bit;

	bool properties;
	bool visual_script_generic;
	bool connecting;
	String selected;
	Variant::Type type;
	String base_type;
	String base_script;
	ObjectID script;
	Object *instance;
	bool virtuals_only;
	VBoxContainer *vbox;

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void select_method_from_base_type(const String &p_base, const bool p_virtuals_only = false, const bool p_connecting = true, bool clear_text = true);
	void select_from_base_type(const String &p_base, const String &p_base_script = "", bool p_virtuals_only = false, const bool p_connecting = true, bool clear_text = true);
	void select_from_script(const Ref<Script> &p_script, const bool p_connecting = true, bool clear_text = true);
	void select_from_basic_type(Variant::Type p_type, const bool p_connecting = true, bool clear_text = true);
	void select_from_action(const String &p_type, const bool p_connecting = true, bool clear_text = true);
	void select_from_instance(Object *p_instance, const bool p_connecting = true, bool clear_text = true);
	void select_from_visual_script(const Ref<Script> &p_script, bool clear_text = true);

	void show_window(float p_screen_ratio);

	VisualScriptPropertySelector();
};

class VisualScriptPropertySelector::SearchRunner : public RefCounted {
	enum Phase {
		PHASE_INIT,
		PHASE_MATCH_CLASSES_INIT,
		PHASE_NODE_CLASSES_INIT,
		PHASE_NODE_CLASSES_BUILD,
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
		DocData::ClassDoc *doc;
		bool name = false;
		String category = "";
		Vector<DocData::MethodDoc *> constructors;
		Vector<DocData::MethodDoc *> methods;
		Vector<DocData::MethodDoc *> operators;
		Vector<DocData::MethodDoc *> signals;
		Vector<DocData::ConstantDoc *> constants;
		Vector<DocData::PropertyDoc *> properties;
		Vector<DocData::ThemeItemDoc *> theme_properties;

		bool required() {
			return name || methods.size() || signals.size() || constants.size() || properties.size() || theme_properties.size();
		}
	};

	VisualScriptPropertySelector *selector_ui;
	Control *ui_service;
	Tree *results_tree;
	String term;
	int search_flags;
	int scope_flags;

	Ref<Texture2D> empty_icon;
	Color disabled_color;

	Map<String, DocData::ClassDoc>::Element *iterator_doc = nullptr;
	Map<String, ClassMatch> matches;
	Map<String, ClassMatch>::Element *iterator_match = nullptr;
	TreeItem *root_item = nullptr;
	Map<String, TreeItem *> class_items;
	TreeItem *matched_item = nullptr;
	float match_highest_score = 0;

	Map<String, DocData::ClassDoc> combined_docs;
	List<String> vs_nodes;

	bool _is_class_disabled_by_feature_profile(const StringName &p_class);
	bool _is_class_disabled_by_scope(const StringName &p_class);

	bool _slice();
	bool _phase_init();
	bool _phase_match_classes_init();
	bool _phase_node_classes_init();
	bool _phase_node_classes_build();
	bool _phase_match_classes();
	bool _phase_class_items_init();
	bool _phase_class_items();
	bool _phase_member_items_init();
	bool _phase_member_items();
	bool _phase_select_match();

	bool _match_string(const String &p_term, const String &p_string) const;
	bool _match_visual_script(DocData::ClassDoc &class_doc);
	bool _match_is_hidden(DocData::ClassDoc &class_doc);
	void _match_item(TreeItem *p_item, const String &p_text);
	void _add_class_doc(String class_name, String inherits, String category);
	DocData::MethodDoc _get_method_doc(MethodInfo method_info);
	TreeItem *_create_class_hierarchy(const ClassMatch &p_match);
	TreeItem *_create_class_item(TreeItem *p_parent, const DocData::ClassDoc *p_doc, bool p_gray);
	TreeItem *_create_method_item(TreeItem *p_parent, const DocData::ClassDoc *p_class_doc, const String &p_text, const DocData::MethodDoc *p_doc);
	TreeItem *_create_signal_item(TreeItem *p_parent, const DocData::ClassDoc *p_class_doc, const DocData::MethodDoc *p_doc);
	TreeItem *_create_constant_item(TreeItem *p_parent, const DocData::ClassDoc *p_class_doc, const DocData::ConstantDoc *p_doc);
	TreeItem *_create_property_item(TreeItem *p_parent, const DocData::ClassDoc *p_class_doc, const DocData::PropertyDoc *p_doc);
	TreeItem *_create_theme_property_item(TreeItem *p_parent, const DocData::ClassDoc *p_class_doc, const DocData::ThemeItemDoc *p_doc);
	TreeItem *_create_member_item(TreeItem *p_parent, const String &p_class_name, const String &p_icon, const String &p_name, const String &p_text, const String &p_type, const String &p_metatype, const String &p_tooltip, const String &p_description);

public:
	bool work(uint64_t slot = 100000);

	SearchRunner(VisualScriptPropertySelector *p_selector_ui, Tree *p_results_tree);
};

#endif // VISUALSCRIPT_PROPERTYSELECTOR_H
