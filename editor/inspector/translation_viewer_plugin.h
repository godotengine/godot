/**************************************************************************/
/*  translation_viewer_plugin.h                                           */
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

#include "editor/inspector/editor_inspector.h"
#include "editor/plugins/editor_plugin.h"
#include "scene/gui/dialogs.h"

class Label;
class ScrollContainer;
class TextEdit;
class Translation;
class Tree;
class VBoxContainer;

class TranslationViewerDialog : public AcceptDialog {
	GDCLASS(TranslationViewerDialog, AcceptDialog);

	Ref<Translation> translation;

	// Filtering needs String operations. Avoid converting from StringName every time.
	struct KeyEntry {
		String msgctxt;
		String msgid;
	};
	LocalVector<KeyEntry> keys;

	Label *info_label = nullptr;

	Tree *messages_list = nullptr;
	LineEdit *filter_edit = nullptr;

	Label *details_info_label = nullptr;
	ScrollContainer *details_scroll = nullptr;

	TextEdit *msgctxt_view = nullptr;
	TextEdit *msgid_view = nullptr;
	TextEdit *msgid_plural_view = nullptr;
	VBoxContainer *msgstrs_view = nullptr;
	TextEdit *comments_view = nullptr;
	TextEdit *locations_view = nullptr;

	Control *msgctxt_section = nullptr;
	Control *msgid_plural_section = nullptr;
	Control *comments_section = nullptr;
	Control *locations_section = nullptr;

	void _on_translation_changed();

	void _update_messages_list(bool p_keep_selection);
	void _update_details_view();

	void _set_msgstrs(const Vector<String> &p_msgstrs);

protected:
	void _notification(int p_what);

public:
	void edit(Ref<Translation> p_translation);

	TranslationViewerDialog();
};

class EditorInspectorPluginTranslation : public EditorInspectorPlugin {
	GDCLASS(EditorInspectorPluginTranslation, EditorInspectorPlugin);

	TranslationViewerDialog *dialog = nullptr;

	void _on_view_messages_pressed(Object *p_object);

public:
	virtual bool can_handle(Object *p_object) override;
	virtual void parse_end(Object *p_object) override;
};

class TranslationViewerPlugin : public EditorPlugin {
	GDCLASS(TranslationViewerPlugin, EditorPlugin);

public:
	virtual String get_plugin_name() const override { return "TranslationViewer"; }

	TranslationViewerPlugin();
};
