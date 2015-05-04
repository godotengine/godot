/*************************************************************************/
/*  item_list_editor_plugin.h                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2015 Juan Linietsky, Ariel Manzur.                 */
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

#include "tools/editor/editor_plugin.h"
#include "tools/editor/editor_node.h"
#include "scene/gui/option_button.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/popup_menu.h"
#include "scene/gui/spin_box.h"

/**
	@author Juan Linietsky <reduzio@gmail.com>
*/


class ItemListPlugin : public Object {

	OBJ_TYPE(ItemListPlugin,Object);

protected:

	bool _set(const StringName& p_name, const Variant& p_value);
	bool _get(const StringName& p_name,Variant &r_ret) const;
	void _get_property_list( List<PropertyInfo> *p_list) const;


public:

	enum Flags {

		FLAG_ICON=1,
		FLAG_CHECKABLE=2,
		FLAG_ACCEL=4,
		FLAG_ID=8,
		FLAG_ENABLE=16,
		FLAG_SEPARATOR=32
	};

	virtual void set_object(Object *p_object)=0;

	virtual bool handles(Object *p_object) const=0;

	virtual int get_flags() const=0;

	virtual void set_item_text(int p_idx,const String& p_text){}
	virtual void set_item_icon(int p_idx,const Ref<Texture>& p_tex){}
	virtual void set_item_checkable(int p_idx,bool p_check){}
	virtual void set_item_checked(int p_idx,bool p_checked){}
	virtual void set_item_accel(int p_idx,int p_accel){}
	virtual void set_item_enabled(int p_idx,int p_enabled){}
	virtual void set_item_id(int p_idx,int p_id){}
	virtual void set_item_separator(int p_idx,bool p_separator){}


	virtual String get_item_text(int p_idx) const{ return ""; };
	virtual Ref<Texture> get_item_icon(int p_idx) const{ return Ref<Texture>(); };
	virtual bool is_item_checkable(int p_idx) const{ return false; };
	virtual bool is_item_checked(int p_idx) const{ return false; };
	virtual int get_item_accel(int p_idx) const{ return 0; };
	virtual bool is_item_enabled(int p_idx) const{ return false; };
	virtual int get_item_id(int p_idx) const{ return -1; };
	virtual bool is_item_separator(int p_idx) const{ return false; };

	virtual void add_item()=0;
	virtual int get_item_count() const=0;
	virtual void erase(int p_idx)=0;

	ItemListPlugin() {}
};

class ItemListEditor : public Control {

	OBJ_TYPE(ItemListEditor, Control );

	Node *item_list;

	enum {

		MENU_EDIT_ITEMS,
		MENU_CLEAR
	};

	AcceptDialog *dialog;

	PropertyEditor *prop_editor;

	MenuButton * options;
	int selected_idx;

	Button *add_button;
	Button *del_button;


//	FileDialog *emission_file_dialog;
	void _menu_option(int);

	Vector<ItemListPlugin*> item_plugins;

	void _node_removed(Node *p_node);
	void _add_pressed();
	void _delete_pressed();
protected:

	void _notification(int p_notification);

	static void _bind_methods();
public:

	void edit(Node *p_item_list);
	bool handles(Object *p_object) const;
	void add_plugin(ItemListPlugin* p_plugin) { item_plugins.push_back(p_plugin); }
	ItemListEditor();
	~ItemListEditor();
};

class ItemListEditorPlugin : public EditorPlugin {

	OBJ_TYPE( ItemListEditorPlugin, EditorPlugin );

	ItemListEditor *item_list_editor;
	EditorNode *editor;

public:

	virtual String get_name() const { return "ItemList"; }
	bool has_main_screen() const { return false; }
	virtual void edit(Object *p_node);
	virtual bool handles(Object *p_node) const;
	virtual void make_visible(bool p_visible);

	ItemListEditorPlugin(EditorNode *p_node);
	~ItemListEditorPlugin();

};

#endif // ITEM_LIST_EDITOR_PLUGIN_H
