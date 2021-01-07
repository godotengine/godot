/*************************************************************************/
/*  font_editor_plugin.h                                                 */
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

#ifndef FONT_EDITOR_PLUGIN_H
#define FONT_EDITOR_PLUGIN_H

#include "editor/editor_node.h"
#include "editor/editor_plugin.h"
#include "scene/resources/font.h"
#include "scene/resources/text_line.h"

class FontDataPreview : public Control {
	GDCLASS(FontDataPreview, Control);

protected:
	void _notification(int p_what);
	static void _bind_methods();

	Ref<TextLine> line;

public:
	virtual Size2 get_minimum_size() const override;

	void set_data(const Ref<FontData> &p_data);

	FontDataPreview();
};

/*************************************************************************/

class FontDataEditor : public EditorProperty {
	GDCLASS(FontDataEditor, EditorProperty);

	LineEdit *le = nullptr;
	CheckBox *chk = nullptr;
	Button *button = nullptr;

	void toggle_lang(bool p_pressed);
	void toggle_script(bool p_pressed);
	void add_lang();
	void add_script();
	void remove_lang();
	void remove_script();

protected:
	void _notification(int p_what);

	static void _bind_methods();

public:
	virtual Size2 get_minimum_size() const override;
	virtual void update_property() override;

	void init_lang_add();
	void init_lang_edit();
	void init_script_add();
	void init_script_edit();

	FontDataEditor();
};

/*************************************************************************/

class EditorInspectorPluginFont : public EditorInspectorPlugin {
	GDCLASS(EditorInspectorPluginFont, EditorInspectorPlugin);

public:
	virtual bool can_handle(Object *p_object) override;
	virtual void parse_begin(Object *p_object) override;
	virtual bool parse_property(Object *p_object, Variant::Type p_type, const String &p_path, PropertyHint p_hint, const String &p_hint_text, int p_usage, bool p_wide) override;
};

/*************************************************************************/

class FontEditorPlugin : public EditorPlugin {
	GDCLASS(FontEditorPlugin, EditorPlugin);

public:
	FontEditorPlugin(EditorNode *p_node);

	virtual String get_name() const override { return "Font"; }
};

#endif // FONT_EDITOR_PLUGIN_H
