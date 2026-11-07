/**************************************************************************/
/*  editor_about.h                                                        */
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

#include "scene/gui/dialogs.h"

class CreditsRoll;
class ItemList;
class Label;
class RichTextLabel;
class TextureRect;
class Tree;

/**
 * NOTE: Do not assume the EditorNode singleton to be available in this class' methods.
 * EditorAbout is also used from the project manager where EditorNode isn't initialized.
 */
class EditorAbout : public AcceptDialog {
	GDSOFTCLASS(EditorAbout, AcceptDialog);

private:
	enum SectionFlags {
		FLAG_SINGLE_COLUMN = 1 << 0,
		FLAG_ALLOW_WEBSITE = 1 << 1,
		FLAG_EASTER_EGG = 1 << 2,
	};

	void _license_tree_selected();
	void _credits_visibility_changed();
	void _item_activated(int p_idx, ItemList *p_il);
	void _item_list_resized(ItemList *p_il);
	Label *_create_section(Control *p_parent, const String &p_name, const char *const *p_src, BitField<SectionFlags> p_flags = 0);

	Label *_about_text_label = nullptr;
	Label *_project_manager_label = nullptr;
	Tree *_tpl_tree = nullptr;
	RichTextLabel *license_text_label = nullptr;
	RichTextLabel *_tpl_text = nullptr;
	TextureRect *_logo = nullptr;
	Vector<ItemList *> name_lists;
	CreditsRoll *credits_roll = nullptr;

protected:
	void _notification(int p_what);

public:
	EditorAbout();
};
