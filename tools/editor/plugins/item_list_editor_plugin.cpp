/*************************************************************************/
/*  item_list_editor_plugin.cpp                                          */
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
#include "item_list_editor_plugin.h"

#include "io/resource_loader.h"


bool ItemListPlugin::_set(const StringName& p_name, const Variant& p_value) {

	String name = p_name;
	int idx = name.get_slice("/",0).to_int();
	String what=name.get_slice("/",1);

	if (what=="text")
		set_item_text(idx,p_value);
	else if (what=="icon")
		set_item_icon(idx,p_value);
	else if (what=="checkable")
		set_item_checkable(idx,p_value);
	else if (what=="checked")
		set_item_checked(idx,p_value);
	else if (what=="enabled")
		set_item_enabled(idx,p_value);
	else if (what=="accel")
		set_item_accel(idx,p_value);
	else if (what=="id")
		set_item_id(idx,p_value);
	else if (what=="separator")
		set_item_separator(idx,p_value);
	else
		return false;

	return true;
}

bool ItemListPlugin::_get(const StringName& p_name,Variant &r_ret) const {
	String name = p_name;
	int idx = name.get_slice("/",0).to_int();
	String what=name.get_slice("/",1);

	if (what=="text")
		r_ret=get_item_text(idx);
	else if (what=="icon")
		r_ret=get_item_icon(idx);
	else if (what=="checkable")
		r_ret=is_item_checkable(idx);
	else if (what=="checked")
		r_ret=is_item_checked(idx);
	else if (what=="enabled")
		r_ret=is_item_enabled(idx);
	else if (what=="accel")
		r_ret=get_item_accel(idx);
	else if (what=="id")
		r_ret=get_item_id(idx);
	else if (what=="separator")
		r_ret=is_item_separator(idx);
	else
		return false;

	return true;
}
void ItemListPlugin::_get_property_list( List<PropertyInfo> *p_list) const {

	for(int i=0;i<get_item_count();i++) {

		String base=itos(i)+"/";

		p_list->push_back( PropertyInfo(Variant::STRING,base+"text") );
		p_list->push_back( PropertyInfo(Variant::OBJECT,base+"icon",PROPERTY_HINT_RESOURCE_TYPE,"Texture") );
		if (get_flags()&FLAG_CHECKABLE) {

			p_list->push_back( PropertyInfo(Variant::BOOL,base+"checkable") );
			p_list->push_back( PropertyInfo(Variant::BOOL,base+"checked") );

		}
		if (get_flags()&FLAG_ENABLE) {

			p_list->push_back( PropertyInfo(Variant::BOOL,base+"enabled") );

		}
		if (get_flags()&FLAG_ACCEL) {

			p_list->push_back( PropertyInfo(Variant::INT,base+"accel",PROPERTY_HINT_KEY_ACCEL) );

		}
		if (get_flags()&FLAG_ID) {

			p_list->push_back( PropertyInfo(Variant::INT,base+"id",PROPERTY_HINT_RANGE,"-1,4096") );

		}
		if (get_flags()&FLAG_SEPARATOR) {

			p_list->push_back( PropertyInfo(Variant::BOOL,base+"separator") );

		}
	}
}

void ItemListEditor::_node_removed(Node *p_node) {

	if(p_node==item_list) {
		item_list=NULL;
		hide();
		dialog->hide();
	}


}

void ItemListEditor::_delete_pressed() {

	String p = prop_editor->get_selected_path();

	if (p.find("/")!=-1) {

		if (selected_idx<0 || selected_idx>=item_plugins.size())
			return;

		item_plugins[selected_idx]->erase(p.get_slice("/",0).to_int());;
	}

}

void ItemListEditor::_add_pressed() {

	if (selected_idx<0 || selected_idx>=item_plugins.size())
		return;

	item_plugins[selected_idx]->add_item();
}

void ItemListEditor::_notification(int p_notification) {

	if (p_notification==NOTIFICATION_ENTER_TREE) {

		add_button->set_icon(get_icon("Add","EditorIcons"));
		del_button->set_icon(get_icon("Del","EditorIcons"));
	}
}


void ItemListEditor::_menu_option(int p_option) {


	switch(p_option) {

		case MENU_EDIT_ITEMS: {

			dialog->popup_centered_ratio();
		} break;
	}
}


void ItemListEditor::edit(Node *p_item_list) {

	item_list=p_item_list;

	for(int i=0;i<item_plugins.size();i++) {
		if (item_plugins[i]->handles(p_item_list)) {

			item_plugins[i]->set_object(p_item_list);
			prop_editor->edit(item_plugins[i]);
			selected_idx=i;
			return;
		}
	}

	selected_idx=-1;

	prop_editor->edit(NULL);

}


void ItemListEditor::_bind_methods() {

	ObjectTypeDB::bind_method("_menu_option",&ItemListEditor::_menu_option);
	ObjectTypeDB::bind_method("_add_button",&ItemListEditor::_add_pressed);
	ObjectTypeDB::bind_method("_delete_button",&ItemListEditor::_delete_pressed);

	//ObjectTypeDB::bind_method("_populate",&ItemListEditor::_populate);

}

bool ItemListEditor::handles(Object *p_object) const {
	return false;
	for(int i=0;i<item_plugins.size();i++)  {
		if (item_plugins[i]->handles(p_object)) {
			return true;
		}
	}

	return false;

}
ItemListEditor::ItemListEditor() {

	selected_idx=-1;
	options = memnew( MenuButton );
	add_child(options);
	options->set_area_as_parent_rect();

	options->set_text("Items");
	options->get_popup()->add_item("Edit Items",MENU_EDIT_ITEMS);
	//options->get_popup()->add_item("Clear",MENU_CLEAR);

	options->get_popup()->connect("item_pressed", this,"_menu_option");

	dialog = memnew( AcceptDialog );
	add_child( dialog );



	HBoxContainer *hbc = memnew( HBoxContainer );

	dialog->add_child(hbc);
	dialog->set_child_rect(hbc);

	prop_editor = memnew( PropertyEditor );

	hbc->add_child(prop_editor);
	prop_editor->set_h_size_flags(SIZE_EXPAND_FILL);

	VBoxContainer *vbc = memnew( VBoxContainer );
	hbc->add_child(vbc);

	add_button = memnew( Button );
	//add_button->set_text("Add");
	add_button->connect("pressed",this,"_add_button");
	vbc->add_child(add_button);

	del_button = memnew( Button );
	//del_button->set_text("Del");
	del_button->connect("pressed",this,"_delete_button");
	vbc->add_child(del_button);

	dialog->set_title("Item List");
	prop_editor->hide_top_label();



}


void ItemListEditorPlugin::edit(Object *p_object) {

	item_list_editor->edit(p_object->cast_to<Node>());
}

bool ItemListEditorPlugin::handles(Object *p_object) const {

	return item_list_editor->handles(p_object);
}

void ItemListEditorPlugin::make_visible(bool p_visible) {

	if (p_visible) {
		item_list_editor->show();
	} else {

		item_list_editor->hide();
		item_list_editor->edit(NULL);
	}

}


ItemListEditor::~ItemListEditor() {

	for(int i=0;i<item_plugins.size();i++)
		memdelete( item_plugins[i] );
}

///////////////////////// PLUGINS /////////////////////////////
///////////////////////// PLUGINS /////////////////////////////
///////////////////////// PLUGINS /////////////////////////////
///////////////////////// PLUGINS /////////////////////////////
///////////////////////// PLUGINS /////////////////////////////


class ItemListOptionButtonPlugin : public ItemListPlugin {

	OBJ_TYPE(ItemListOptionButtonPlugin,ItemListPlugin);

	OptionButton *ob;
public:

	virtual void set_object(Object *p_object) { ob = p_object->cast_to<OptionButton>(); }

	virtual bool handles(Object *p_object) const { return p_object->cast_to<OptionButton>()!=NULL; }

	virtual int get_flags() const { return FLAG_ICON|FLAG_ID|FLAG_ENABLE; }

	virtual void set_item_text(int p_idx,const String& p_text){ ob->set_item_text(p_idx,p_text);}
	virtual void set_item_icon(int p_idx,const Ref<Texture>& p_tex){ ob->set_item_icon(p_idx,p_tex);}
	virtual void set_item_enabled(int p_idx,int p_enabled){ ob->set_item_disabled(p_idx,!p_enabled);}
	virtual void set_item_id(int p_idx,int p_id){ ob->set_item_ID(p_idx,p_id);}


	virtual String get_item_text(int p_idx) const{ return ob->get_item_text(p_idx); };
	virtual Ref<Texture> get_item_icon(int p_idx) const{ return ob->get_item_icon(p_idx); };
	virtual bool is_item_enabled(int p_idx) const{ return !ob->is_item_disabled(p_idx); };
	virtual int get_item_id(int p_idx) const{ return ob->get_item_ID(p_idx); };

	virtual void add_item() { ob->add_item( "New Item "+itos(ob->get_item_count())); _change_notify();}
	virtual int get_item_count() const { return ob->get_item_count(); }
	virtual void erase(int p_idx) { ob->remove_item(p_idx); _change_notify();}


	ItemListOptionButtonPlugin() { ob=NULL; }
};

class ItemListPopupMenuPlugin : public ItemListPlugin {

	OBJ_TYPE(ItemListPopupMenuPlugin,ItemListPlugin);

	PopupMenu *pp;
public:

	virtual void set_object(Object *p_object) {
		if (p_object->cast_to<MenuButton>())
			pp = p_object->cast_to<MenuButton>()->get_popup();
		else
			pp = p_object->cast_to<PopupMenu>();
	}

	virtual bool handles(Object *p_object) const { return p_object->cast_to<PopupMenu>()!=NULL || p_object->cast_to<MenuButton>()!=NULL; }

	virtual int get_flags() const { return FLAG_ICON|FLAG_ID|FLAG_ENABLE|FLAG_CHECKABLE|FLAG_SEPARATOR|FLAG_ACCEL; }

	virtual void set_item_text(int p_idx,const String& p_text){ pp->set_item_text(p_idx,p_text); }
	virtual void set_item_icon(int p_idx,const Ref<Texture>& p_tex){ pp->set_item_icon(p_idx,p_tex);}
	virtual void set_item_checkable(int p_idx,bool p_check){ pp->set_item_as_checkable(p_idx,p_check);}
	virtual void set_item_checked(int p_idx,bool p_checked){ pp->set_item_checked(p_idx,p_checked);}
	virtual void set_item_accel(int p_idx,int p_accel){ pp->set_item_accelerator(p_idx,p_accel);}
	virtual void set_item_enabled(int p_idx,int p_enabled){ pp->set_item_disabled(p_idx,!p_enabled);}
	virtual void set_item_id(int p_idx,int p_id){ pp->set_item_ID(p_idx,p_idx);}
	virtual void set_item_separator(int p_idx,bool p_separator){ pp->set_item_as_separator(p_idx,p_separator);}


	virtual String get_item_text(int p_idx) const{ return pp->get_item_text(p_idx); };
	virtual Ref<Texture> get_item_icon(int p_idx) const{ return pp->get_item_icon(p_idx); };
	virtual bool is_item_checkable(int p_idx) const{ return pp->is_item_checkable(p_idx);  };
	virtual bool is_item_checked(int p_idx) const{ return pp->is_item_checked(p_idx); };
	virtual int get_item_accel(int p_idx) const{ return pp->get_item_accelerator(p_idx); };
	virtual bool is_item_enabled(int p_idx) const{ return !pp->is_item_disabled(p_idx);  };
	virtual int get_item_id(int p_idx) const{ return pp->get_item_ID(p_idx);  };
	virtual bool is_item_separator(int p_idx) const{ return pp->is_item_separator(p_idx); };



	virtual void add_item() { pp->add_item( "New Item "+itos(pp->get_item_count())); _change_notify();}
	virtual int get_item_count() const { return pp->get_item_count(); }
	virtual void erase(int p_idx) { pp->remove_item(p_idx); _change_notify();}


	ItemListPopupMenuPlugin() { pp=NULL; }
};






ItemListEditorPlugin::ItemListEditorPlugin(EditorNode *p_node) {

	editor=p_node;
	item_list_editor = memnew( ItemListEditor );
	editor->get_viewport()->add_child(item_list_editor);

//	item_list_editor->set_anchor(MARGIN_LEFT,Control::ANCHOR_END);
//	item_list_editor->set_anchor(MARGIN_RIGHT,Control::ANCHOR_END);
	item_list_editor->set_margin(MARGIN_LEFT,180);
	item_list_editor->set_margin(MARGIN_RIGHT,230);
	item_list_editor->set_margin(MARGIN_TOP,0);
	item_list_editor->set_margin(MARGIN_BOTTOM,10);


	item_list_editor->hide();
	item_list_editor->add_plugin( memnew( ItemListOptionButtonPlugin) );
	item_list_editor->add_plugin( memnew( ItemListPopupMenuPlugin) );
}


ItemListEditorPlugin::~ItemListEditorPlugin()
{
}


