/**************************************************************************/
/*  translation_viewer_plugin.cpp                                         */
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

#include "translation_viewer_plugin.h"

#include "core/string/optimized_translation.h"
#include "core/string/translation.h"
#include "editor/editor_node.h"
#include "editor/editor_string_names.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/foldable_container.h"
#include "scene/gui/label.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/margin_container.h"
#include "scene/gui/text_edit.h"
#include "scene/gui/tree.h"

static TextEdit *_create_text_view() {
	TextEdit *text_edit = memnew(TextEdit);
	text_edit->set_editable(false);
	text_edit->set_line_wrapping_mode(TextEdit::LINE_WRAPPING_BOUNDARY);
	text_edit->set_fit_content_height_enabled(true);
	return text_edit;
}

static Control *_add_details_section(Control *p_parent, const String &p_title, Control *p_content) {
	FoldableContainer *fc = memnew(FoldableContainer);
	fc->set_title(p_title);
	fc->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	fc->add_child(p_content);
	p_parent->add_child(fc);
	return fc;
}

static void _set_label_autowrap(Label *p_label) {
	// Workaround a known bug that causes autowrap labels to be extremely tall.
	p_label->set_custom_minimum_size(Vector2(100 * EDSCALE, 0));
	p_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
}

static String _get_first_line(const String &p_text) {
	int newline_index = p_text.find_char('\n');
	if (newline_index == -1) {
		return p_text;
	}
	return p_text.left(newline_index);
}

void TranslationViewerDialog::_on_translation_changed() {
	Ref<OptimizedTranslation> ot = translation;
	if (ot.is_valid()) {
		info_label->set_text(TTR("OptimizedTranslation does not support viewing messages."));
	} else {
		int message_count = translation.is_valid() ? translation->get_message_count() : 0;
		info_label->set_text(vformat(TTRN("%d message found.", "%d messages found.", message_count), message_count));
	}

	keys.clear();

	if (translation.is_valid() && ot.is_null()) {
		List<Translation::MessageKey> raw_keys;
		translation->get_message_list(&raw_keys);

		keys.reserve(raw_keys.size());
		for (const Translation::MessageKey &key : raw_keys) {
			keys.push_back({ key.msgctxt, key.msgid });
		}
	}

	_update_messages_list(false);
}

void TranslationViewerDialog::_update_messages_list(bool p_keep_selection) {
	int prev_selected_index = -1;
	if (p_keep_selection) {
		TreeItem *selected = messages_list->get_selected();
		if (selected) {
			prev_selected_index = selected->get_metadata(0);
		}
	}

	// Filter first to determine whether to show the context column.
	const String &filter_text = filter_edit->get_text();
	bool context_column_needed = false;
	LocalVector<unsigned int> indices;
	indices.reserve(keys.size());
	for (unsigned int i = 0; i < keys.size(); i++) {
		const KeyEntry &key = keys[i];
		if (!filter_text.is_empty() && !key.msgid.containsn(filter_text) && !key.msgctxt.containsn(filter_text)) {
			continue;
		}
		if (!key.msgctxt.is_empty()) {
			context_column_needed = true;
		}
		indices.push_back(i);
	}

	TreeItem *root = messages_list->get_root();
	root->clear_children();

	if (context_column_needed) {
		messages_list->set_columns(2);
		messages_list->set_column_expand(1, false);
	} else {
		messages_list->set_columns(1);
	}

	TreeItem *selected_item = nullptr;
	for (unsigned int i : indices) {
		const KeyEntry &key = keys[i];

		TreeItem *item = root->create_child();
		item->set_text(0, _get_first_line(key.msgid));
		item->set_metadata(0, i);
		if (!key.msgctxt.is_empty()) {
			item->set_text_alignment(1, HORIZONTAL_ALIGNMENT_RIGHT);
			item->set_text_overrun_behavior(1, TextServer::OVERRUN_NO_TRIMMING);
			item->set_text(1, _get_first_line(key.msgctxt));
			item->set_tooltip_text(1, TTRC("Context"));
		}

		if (prev_selected_index != -1 && (unsigned int)prev_selected_index == i) {
			selected_item = item;
		}
	}

	if (selected_item == nullptr && root->get_child_count() > 0) {
		selected_item = root->get_child(0);
	}
	if (selected_item) {
		messages_list->set_selected(selected_item, 0);
		messages_list->scroll_to_item(selected_item);
	}

	_update_details_view();
}

void TranslationViewerDialog::_update_details_view() {
	TreeItem *selected = messages_list->get_selected();
	if (!selected) {
		details_info_label->show();
		details_scroll->hide();
		return;
	}

	unsigned int key_index = selected->get_metadata(0);
	ERR_FAIL_UNSIGNED_INDEX(key_index, keys.size());
	const KeyEntry &key = keys[key_index];

	msgid_view->set_text(key.msgid);

	msgctxt_section->set_visible(!key.msgctxt.is_empty());
	msgctxt_view->set_text(key.msgctxt);

	const String &msgid_plural = translation->get_hint(key.msgid, key.msgctxt, Translation::HINT_PLURAL);
	msgid_plural_section->set_visible(!msgid_plural.is_empty());
	msgid_plural_view->set_text(msgid_plural);

	if (msgid_plural.is_empty()) {
		_set_msgstrs({ translation->get_message(key.msgid, key.msgctxt) });
	} else {
		_set_msgstrs(translation->get_plural_forms(key.msgid, key.msgctxt));
	}

	const String &comments = translation->get_hint(key.msgid, key.msgctxt, Translation::HINT_COMMENTS);
	comments_section->set_visible(!comments.is_empty());
	comments_view->set_text(comments);

	const String &locations = translation->get_hint(key.msgid, key.msgctxt, Translation::HINT_LOCATIONS);
	locations_section->set_visible(!locations.is_empty());
	locations_view->set_text(locations);

	details_info_label->hide();
	details_scroll->show();
}

void TranslationViewerDialog::_set_msgstrs(const Vector<String> &p_msgstrs) {
	for (int i = msgstrs_view->get_child_count(); i < p_msgstrs.size(); i++) {
		msgstrs_view->add_child(_create_text_view());
	}
	for (int i = 0; i < msgstrs_view->get_child_count(); i++) {
		TextEdit *text_edit = Object::cast_to<TextEdit>(msgstrs_view->get_child(i));
		ERR_CONTINUE(text_edit == nullptr); // Should not happen, but just in case.

		if (i < p_msgstrs.size()) {
			text_edit->set_text(p_msgstrs[i]);
			text_edit->show();
		} else {
			text_edit->hide();
		}
	}
}

void TranslationViewerDialog::edit(Ref<Translation> p_translation) {
	if (translation == p_translation) {
		return;
	}
	if (translation.is_valid()) {
		translation->disconnect_changed(callable_mp(this, &TranslationViewerDialog::_on_translation_changed));
	}
	translation = p_translation;
	if (translation.is_valid()) {
		translation->connect_changed(callable_mp(this, &TranslationViewerDialog::_on_translation_changed));
	}

	filter_edit->set_text(String());
	_on_translation_changed();
}

void TranslationViewerDialog::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED: {
			filter_edit->set_right_icon(get_theme_icon(SNAME("Search"), EditorStringName(EditorIcons)));
		} break;
	}
}

TranslationViewerDialog::TranslationViewerDialog() {
	set_title(TTRC("Translation Viewer"));

	VBoxContainer *main_vb = memnew(VBoxContainer);
	add_child(main_vb);

	info_label = memnew(Label);
	info_label->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	_set_label_autowrap(info_label);
	main_vb->add_child(info_label);

	HSplitContainer *split = memnew(HSplitContainer);
	split->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	main_vb->add_child(split);

	VBoxContainer *list_vb = memnew(VBoxContainer);
	list_vb->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	list_vb->set_stretch_ratio(3);
	split->add_child(list_vb);

	messages_list = memnew(Tree);
	messages_list->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	messages_list->set_tooltip_auto_translate_mode(AUTO_TRANSLATE_MODE_ALWAYS);
	messages_list->set_auto_tooltip(false);
	messages_list->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	messages_list->set_select_mode(Tree::SELECT_ROW);
	messages_list->set_hide_root(true);
	messages_list->create_item();
	messages_list->connect(SceneStringName(item_selected), callable_mp(this, &TranslationViewerDialog::_update_details_view));
	list_vb->add_child(messages_list);

	filter_edit = memnew(LineEdit);
	filter_edit->set_clear_button_enabled(true);
	filter_edit->set_placeholder(TTRC("Filter Messages"));
	filter_edit->connect(SceneStringName(text_changed), callable_mp(this, &TranslationViewerDialog::_update_messages_list).bind(true).unbind(1));
	list_vb->add_child(filter_edit);

	MarginContainer *details_view = memnew(MarginContainer);
	details_view->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	details_view->set_stretch_ratio(2);
	split->add_child(details_view);

	details_info_label = memnew(Label);
	details_info_label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	details_info_label->set_vertical_alignment(VERTICAL_ALIGNMENT_CENTER);
	details_info_label->set_text(TTRC("Select a message to see details."));
	_set_label_autowrap(details_info_label);
	details_view->add_child(details_info_label);

	details_scroll = memnew(ScrollContainer);
	details_scroll->set_horizontal_scroll_mode(ScrollContainer::SCROLL_MODE_DISABLED);
	details_view->add_child(details_scroll);

	VBoxContainer *details_vb = memnew(VBoxContainer);
	details_vb->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	details_vb->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	details_scroll->add_child(details_vb);

	msgctxt_view = _create_text_view();
	msgctxt_section = _add_details_section(details_vb, TTRC("Context"), msgctxt_view);

	msgid_view = _create_text_view();
	_add_details_section(details_vb, TTRC("Source Text"), msgid_view);

	msgid_plural_view = _create_text_view();
	msgid_plural_section = _add_details_section(details_vb, TTRC("Source Text Plural"), msgid_plural_view);

	msgstrs_view = memnew(VBoxContainer);
	_add_details_section(details_vb, TTRC("Translated Text"), msgstrs_view);

	comments_view = _create_text_view();
	comments_section = _add_details_section(details_vb, TTRC("Comments"), comments_view);

	locations_view = _create_text_view();
	locations_section = _add_details_section(details_vb, TTRC("Locations"), locations_view);
}

////////////////////////

void EditorInspectorPluginTranslation::_on_view_messages_pressed(Object *p_object) {
	Ref<Translation> translation = Object::cast_to<Translation>(p_object);
	if (translation.is_null()) {
		return;
	}

	if (!dialog) {
		dialog = memnew(TranslationViewerDialog);
		EditorNode::get_singleton()->get_gui_base()->add_child(dialog);
	}
	dialog->edit(translation);
	dialog->popup_centered_clamped(Size2(800, 600) * EDSCALE);
}

bool EditorInspectorPluginTranslation::can_handle(Object *p_object) {
	return Object::cast_to<Translation>(p_object) != nullptr;
}

void EditorInspectorPluginTranslation::parse_end(Object *p_object) {
	Button *button = memnew(EditorInspectorActionButton(TTRC("View Messages"), SNAME("FileList")));
	button->connect(SceneStringName(pressed), callable_mp(this, &EditorInspectorPluginTranslation::_on_view_messages_pressed).bind(p_object));
	add_custom_control(button);
}

////////////////////////

TranslationViewerPlugin::TranslationViewerPlugin() {
	Ref<EditorInspectorPluginTranslation> inspector_plugin;
	inspector_plugin.instantiate();
	add_inspector_plugin(inspector_plugin);
}
