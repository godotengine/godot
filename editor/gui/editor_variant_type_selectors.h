/**************************************************************************/
/*  editor_variant_type_selectors.h                                       */
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

#include "core/variant/variant.h"
#include "scene/gui/option_button.h"
#include "scene/gui/popup_menu.h"

class EditorVariantTypeOptionButton : public OptionButton {
	GDCLASS(EditorVariantTypeOptionButton, OptionButton);

	void _update_menu_icons();

protected:
	void _notification(int p_what);

public:
	Variant::Type get_selected_type() const;

	void populate(const LocalVector<Variant::Type> &p_disabled_types, const HashMap<Variant::Type, String> &p_renames = {});
};

class EditorVariantTypePopupMenu : public PopupMenu {
	GDCLASS(EditorVariantTypePopupMenu, PopupMenu);

	bool remove_item = false;
	bool icons_dirty = true;

	void _populate();
	void _update_menu_icons();

protected:
	void _notification(int p_what);
	virtual void _popup_base(const Rect2i &p_bounds = Rect2i()) override;

public:
	EditorVariantTypePopupMenu(bool p_remove_item);
};
