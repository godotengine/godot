/*************************************************************************/
/*  item_list_editor_plugin.h                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef ITEM_LIST_EDITOR_PLUGIN_H
#define ITEM_LIST_EDITOR_PLUGIN_H

#include "canvas_item_editor_plugin.h"
#include "editor/editor_inspector.h"
#include "editor/editor_node.h"
#include "editor/editor_plugin.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/option_button.h"
#include "scene/gui/popup_menu.h"

class ItemListPlugin : public Object {
	GDCLASS(ItemListPlugin, Object);

protected:
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

public:
	enum Flags {

		FLAG_ICON = 1,
		FLAG_CHECKABLE = 2,
		FLAG_ID = 4,
		FLAG_ENABLE = 8,
		FLAG_SEPARATOR = 16
	};

	virtual void set_object(Object *p_object) = 0;
	virtual bool handles(Object *p_object) const = 0;

	virtual int get_flags() const = 0;

	virtual void set_item_text(int p_idx, const String &p_text) {}
	virtual String get_item_text(int p_idx) const { return ""; };

	virtual void set_item_icon(int p_idx, const Ref<Texture2D> &p_tex) {}
	virtual Ref<Texture2D> get_item_icon(int p_idx) const { return Ref<Texture2D>(); };

	virtual void set_item_checkable(int p_idx, bool p_check) {}
	virtual void set_item_radio_checkable(int p_idx, bool p_check) {}
	virtual bool is_item_checkable(int p_idx) const { return false; };
	virtual bool is_item_radio_checkable(int p_idx) const { return false; };

	virtual void set_item_checked(int p_idx, bool p_checked) {}
	virtual bool is_item_checked(int p_idx) const { return false; };

	virtual void set_item_enabled(int p_idx, int p_enabled) {}
	virtual bool is_item_enabled(int p_idx) const { return false; };

	virtual void set_item_id(int p_idx, int p_id) {}
	virtual int get_item_id(int p_idx) const { return -1; };

	virtual void set_item_separator(int p_idx, bool p_separator) {}
	virtual bool is_item_separator(int p_idx) const { return false; };

	virtual void add_item() = 0;
	virtual int get_item_count() const = 0;
	virtual void erase(int p_idx) = 0;

	ItemListPlugin() {}
};

///////////////////////////////////////////////////////////////

class ItemListOptionButtonPlugin : public ItemListPlugin {
	GDCLASS(ItemListOptionButtonPlugin, ItemListPlugin);

	OptionButton *ob;

public:
	virtual void set_object(Object *p_object) override;
	virtual bool handles(Object *p_object) const override;
	virtual int get_flags() const override;

	virtual void set_item_text(int p_idx, const String &p_text) override { ob->set_item_text(p_idx, p_text); }
	virtual String get_item_text(int p_idx) const override { return ob->get_item_text(p_idx); }

	virtual void set_item_icon(int p_idx, const Ref<Texture2D> &p_tex) override { ob->set_item_icon(p_idx, p_tex); }
	virtual Ref<Texture2D> get_item_icon(int p_idx) const override { return ob->get_item_icon(p_idx); }

	virtual void set_item_enabled(int p_idx, int p_enabled) override { ob->set_item_disabled(p_idx, !p_enabled); }
	virtual bool is_item_enabled(int p_idx) const override { return !ob->is_item_disabled(p_idx); }

	virtual void set_item_id(int p_idx, int p_id) override { ob->set_item_id(p_idx, p_id); }
	virtual int get_item_id(int p_idx) const override { return ob->get_item_id(p_idx); }

	virtual void add_item() override;
	virtual int get_item_count() const override;
	virtual void erase(int p_idx) override;

	ItemListOptionButtonPlugin();
};

class ItemListPopupMenuPlugin : public ItemListPlugin {
	GDCLASS(ItemListPopupMenuPlugin, ItemListPlugin);

	PopupMenu *pp;

public:
	virtual void set_object(Object *p_object) override;
	virtual bool handles(Object *p_object) const override;
	virtual int get_flags() const override;

	virtual void set_item_text(int p_idx, const String &p_text) override { pp->set_item_text(p_idx, p_text); }
	virtual String get_item_text(int p_idx) const override { return pp->get_item_text(p_idx); }

	virtual void set_item_icon(int p_idx, const Ref<Texture2D> &p_tex) override { pp->set_item_icon(p_idx, p_tex); }
	virtual Ref<Texture2D> get_item_icon(int p_idx) const override { return pp->get_item_icon(p_idx); }

	virtual void set_item_checkable(int p_idx, bool p_check) override { pp->set_item_as_checkable(p_idx, p_check); }
	virtual void set_item_radio_checkable(int p_idx, bool p_check) override { pp->set_item_as_radio_checkable(p_idx, p_check); }
	virtual bool is_item_checkable(int p_idx) const override { return pp->is_item_checkable(p_idx); }
	virtual bool is_item_radio_checkable(int p_idx) const override { return pp->is_item_radio_checkable(p_idx); }

	virtual void set_item_checked(int p_idx, bool p_checked) override { pp->set_item_checked(p_idx, p_checked); }
	virtual bool is_item_checked(int p_idx) const override { return pp->is_item_checked(p_idx); }

	virtual void set_item_enabled(int p_idx, int p_enabled) override { pp->set_item_disabled(p_idx, !p_enabled); }
	virtual bool is_item_enabled(int p_idx) const override { return !pp->is_item_disabled(p_idx); }

	virtual void set_item_id(int p_idx, int p_id) override { pp->set_item_id(p_idx, p_id); }
	virtual int get_item_id(int p_idx) const override { return pp->get_item_id(p_idx); }

	virtual void set_item_separator(int p_idx, bool p_separator) override { pp->set_item_as_separator(p_idx, p_separator); }
	virtual bool is_item_separator(int p_idx) const override { return pp->is_item_separator(p_idx); }

	virtual void add_item() override;
	virtual int get_item_count() const override;
	virtual void erase(int p_idx) override;

	ItemListPopupMenuPlugin();
};

///////////////////////////////////////////////////////////////

class ItemListItemListPlugin : public ItemListPlugin {
	GDCLASS(ItemListItemListPlugin, ItemListPlugin);

	ItemList *pp;

public:
	virtual void set_object(Object *p_object) override;
	virtual bool handles(Object *p_object) const override;
	virtual int get_flags() const override;

	virtual void set_item_text(int p_idx, const String &p_text) override { pp->set_item_text(p_idx, p_text); }
	virtual String get_item_text(int p_idx) const override { return pp->get_item_text(p_idx); }

	virtual void set_item_icon(int p_idx, const Ref<Texture2D> &p_tex) override { pp->set_item_icon(p_idx, p_tex); }
	virtual Ref<Texture2D> get_item_icon(int p_idx) const override { return pp->get_item_icon(p_idx); }

	virtual void set_item_enabled(int p_idx, int p_enabled) override { pp->set_item_disabled(p_idx, !p_enabled); }
	virtual bool is_item_enabled(int p_idx) const override { return !pp->is_item_disabled(p_idx); }

	virtual void add_item() override;
	virtual int get_item_count() const override;
	virtual void erase(int p_idx) override;

	ItemListItemListPlugin();
};

///////////////////////////////////////////////////////////////

class ItemListEditor : public HBoxContainer {
	GDCLASS(ItemListEditor, HBoxContainer);

	Node *item_list;

	Button *toolbar_button;

	AcceptDialog *dialog;
	EditorInspector *property_editor;
	Tree *tree;
	Button *add_button;
	Button *del_button;

	int selected_idx;

	Vector<ItemListPlugin *> item_plugins;

	void _edit_items();

	void _add_pressed();
	void _delete_pressed();

	void _node_removed(Node *p_node);

protected:
	void _notification(int p_notification);
	static void _bind_methods();

public:
	void edit(Node *p_item_list);
	bool handles(Object *p_object) const;
	void add_plugin(ItemListPlugin *p_plugin) { item_plugins.push_back(p_plugin); }
	ItemListEditor();
	~ItemListEditor();
};

class ItemListEditorPlugin : public EditorPlugin {
	GDCLASS(ItemListEditorPlugin, EditorPlugin);

	ItemListEditor *item_list_editor;
	EditorNode *editor;

public:
	virtual String get_name() const override { return "ItemList"; }
	bool has_main_screen() const override { return false; }
	virtual void edit(Object *p_object) override;
	virtual bool handles(Object *p_object) const override;
	virtual void make_visible(bool p_visible) override;

	ItemListEditorPlugin(EditorNode *p_node);
	~ItemListEditorPlugin();
};

#endif // ITEM_LIST_EDITOR_PLUGIN_H
