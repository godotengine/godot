/*************************************************************************/
/*  font_editor_plugin.h                                                 */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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

#include "scene/resources/font.h"
#include "scene/gui/texture_frame.h"
#include "scene/gui/option_button.h"
#include "tools/editor/editor_node.h"


class FontEditor : public Control {

	OBJ_TYPE( FontEditor, Control );

	Panel *panel;
	LineEdit *font_size;
	//TextureFrame *tframe; //for debug
	Label *label;
	LineEdit *preview_text;
	FileDialog *file;
	FileDialog* _source_file;
	FileDialog* _export_file;
	OptionButton *import_option;

	Ref<Font> font;

	Map<CharType, int> import_chars;

	void _export_fnt(const String& p_name, Ref<Font> p_font);
	void _export_fnt_pressed();
	void _export_fnt_accept(const String& p_file);

	void _import_ttf(const String& p_string);
	void _import_fnt(const String& p_string);
	void _preview_text_changed(const String& p_text);

	void _add_source();
	void _add_source_accept(const String& p_file);

	void _import_accept(const String&);
	void _import();
protected:
	static void _bind_methods();
public:

	void edit(const Ref<Font>& p_font);

	FontEditor();
};



class FontEditorPlugin : public EditorPlugin {

	OBJ_TYPE( FontEditorPlugin, EditorPlugin );

	FontEditor *font_editor;
	EditorNode *editor;

public:

	virtual String get_name() const { return "Font"; }
	bool has_main_screen() const { return false; }
	virtual void edit(Object *p_node);
	virtual bool handles(Object *p_node) const;
	virtual void make_visible(bool p_visible);

	FontEditorPlugin(EditorNode *p_node);

};


#endif // FONT_EDITOR_PLUGIN_H
