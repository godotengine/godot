/*************************************************************************/
/*  call_dialog.cpp                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "call_dialog.h"

#if 0
#include "class_db.h"
#include "print_string.h"
#include "scene/gui/label.h"


class CallDialogParams : public Object {

	GDCLASS( CallDialogParams, Object );
public:

	bool _set(const StringName& p_name, const Variant& p_value) {

		values[p_name]=p_value;
                return true;
	}

	bool _get(const StringName& p_name,Variant &r_ret) const {

            if (values.has(p_name)) {
                        r_ret=values[p_name];
                        return true;
                    }
                return false;
	}

	void _get_property_list( List<PropertyInfo> *p_list) const {

		for(int i=0;i<method.arguments.size();i++)
			p_list->push_back(method.arguments[i]);
	}

	MethodInfo method;
	HashMap<String,Variant> values;

	CallDialogParams() {}
};


void CallDialog::_notification(int p_what) {

	if (p_what==NOTIFICATION_READY) {

		call->connect("pressed", this,"_call");
		cancel->connect("pressed", this,"_cancel");
		//filter->get_path()->connect("text_changed", this,"_text_changed");
		_update_method_list();
	}

	if (p_what==NOTIFICATION_EXIT_TREE) {

		call->disconnect("pressed", this,"_call");
		cancel->disconnect("pressed", this,"_cancel");

		//filter->get_path()->connect("text_changed", this,"_text_changed");
		_update_method_list();
	}

	if (p_what==NOTIFICATION_DRAW) {

		RID ci = get_canvas_item();
		get_stylebox("panel","PopupMenu")->draw(ci,Rect2(Point2(),get_size()));
	}
}


void CallDialog::_call() {

	if (!tree->get_selected())
		return;

	TreeItem* item=tree->get_selected();
	ERR_FAIL_COND(!item);
	int idx=item->get_metadata(0);
	ERR_FAIL_INDEX(idx,methods.size());
	MethodInfo &m = methods[idx];

	Variant args[VARIANT_ARG_MAX];

	for(int i=0;i<VARIANT_ARG_MAX;i++) {

		if (i>=m.arguments.size())
			continue;

		if (call_params->values.has(m.arguments[i].name))
			args[i]=call_params->values[m.arguments[i].name];
	}

	Variant ret = object->call(m.name,args[0],args[1],args[2],args[3],args[4]);
	if (ret.get_type()!=Variant::NIL)
		return_value->set_text(ret);
	else
		return_value->set_text("");
}

void CallDialog::_cancel() {

	hide();
}


void CallDialog::_item_selected() {

	TreeItem* item=tree->get_selected();
	ERR_FAIL_COND(!item);

	if (item->get_metadata(0).get_type()==Variant::NIL) {

		call->set_disabled(true);
		return;
	}

	call->set_disabled(false);

	int idx=item->get_metadata(0);
	ERR_FAIL_INDEX(idx,methods.size());

	MethodInfo &m = methods[idx];

	call_params->values.clear();
	call_params->method=m;

	property_editor->edit(call_params);
	property_editor->update_tree();


}

void CallDialog::_update_method_list() {

	tree->clear();
	if (!object)
		return;

	TreeItem *root = tree->create_item();

	List<MethodInfo> method_list;
	object->get_method_list(&method_list);
	method_list.sort();
	methods.clear();

	List<String> inheritance_list;

	String type = object->get_class();

	while(type!="") {
		inheritance_list.push_back( type );
		type=ClassDB::get_parent_class(type);
	}

	TreeItem *selected_item=NULL;

	for(int i=0;i<inheritance_list.size();i++) {

		String type=inheritance_list[i];
		String parent_type=ClassDB::get_parent_class(type);

		TreeItem *type_item=NULL;

		List<MethodInfo>::Element *N,*E=method_list.front();

		while(E) {

			N=E->next();

			if (parent_type!="" && ClassDB::get_method(parent_type,E->get().name)!=NULL) {
				E=N;
				continue;
			}

			if (!type_item) {
				type_item=tree->create_item(root);
				type_item->set_text(0,type);
				if (has_icon(type,"EditorIcons"))
					type_item->set_icon(0,get_icon(type,"EditorIcons"));
			}

			TreeItem *method_item = tree->create_item(type_item);
			method_item->set_text(0,E->get().name);
			method_item->set_metadata(0,methods.size());
			if (E->get().name==selected)
				selected_item=method_item;
			methods.push_back( E->get() );

			method_list.erase(E);
			E=N;
		}
	}



	if (selected_item)
		selected_item->select(0);
}

void CallDialog::_bind_methods() {

	ClassDB::bind_method("_call",&CallDialog::_call);
	ClassDB::bind_method("_cancel",&CallDialog::_cancel);
	ClassDB::bind_method("_item_selected", &CallDialog::_item_selected);

}

void CallDialog::set_object(Object *p_object,StringName p_selected) {

	object=p_object;
	selected=p_selected;
	property_editor->edit(NULL);
	call->set_disabled(true);
	return_value->clear();

	_update_method_list();
	method_label->set_text(vformat(TTR("Method List For '%s':"),p_object->get_class()));
}

CallDialog::CallDialog() {

	object=NULL;

	call = memnew( Button );
	call->set_anchor( MARGIN_LEFT, ANCHOR_END );
	call->set_anchor( MARGIN_TOP, ANCHOR_END );
	call->set_anchor( MARGIN_RIGHT, ANCHOR_END );
	call->set_anchor( MARGIN_BOTTOM, ANCHOR_END );
	call->set_begin( Point2( 70, 29 ) );
	call->set_end( Point2( 15, 15 ) );
	call->set_text(TTR("Call"));

	add_child(call);

	cancel = memnew( Button );
	cancel->set_anchor( MARGIN_TOP, ANCHOR_END );
	cancel->set_anchor( MARGIN_BOTTOM, ANCHOR_END );
	cancel->set_begin( Point2( 15, 29 ) );
	cancel->set_end( Point2( 70, 15 ) );
	cancel->set_text(TTR("Close"));

	add_child(cancel);

	tree = memnew( Tree );

	tree->set_anchor( MARGIN_BOTTOM, ANCHOR_END );
	tree->set_begin( Point2( 20,50 ) );
	tree->set_margin(MARGIN_BOTTOM, 44 );
	tree->set_margin(MARGIN_RIGHT, 0.5 );
	tree->set_select_mode( Tree::SELECT_ROW );
	add_child(tree);

	tree->connect("item_selected", this,"_item_selected");
	tree->set_hide_root(true);

	property_editor = memnew( PropertyEditor );

	property_editor->set_anchor_and_margin( MARGIN_RIGHT, ANCHOR_END, 15 );
	property_editor->set_anchor_and_margin( MARGIN_TOP, ANCHOR_BEGIN, 50 );
	//property_editor->set_anchor_and_margin( MARGIN_LEFT, ANCHOR_RATIO, 0.55 );
	property_editor->set_anchor_and_margin( MARGIN_BOTTOM, ANCHOR_END, 90 );
	property_editor->get_scene_tree()->set_hide_root( true );
	property_editor->hide_top_label();

	add_child(property_editor);
	method_label = memnew( Label );
	method_label->set_pos( Point2( 15,25) );
	method_label->set_text(TTR("Method List:"));

	add_child(method_label);

	Label *label = memnew( Label );
	//label->set_anchor_and_margin( MARGIN_LEFT, ANCHOR_RATIO, 0.53 );
	label->set_anchor_and_margin( MARGIN_TOP, ANCHOR_BEGIN, 25 );
	label->set_text(TTR("Arguments:"));

	add_child(label);

	return_label = memnew( Label );
	//return_label->set_anchor_and_margin( MARGIN_LEFT, ANCHOR_RATIO, 0.53 );
	return_label->set_anchor_and_margin( MARGIN_TOP, ANCHOR_END, 85 );
	return_label->set_text(TTR("Return:"));

	add_child(return_label);

	return_value = memnew( LineEdit );
	//return_value->set_anchor_and_margin( MARGIN_LEFT, ANCHOR_RATIO, 0.55 );
	return_value->set_anchor_and_margin( MARGIN_RIGHT, ANCHOR_END, 15 );
	return_value->set_anchor_and_margin( MARGIN_TOP, ANCHOR_END, 65 );

	add_child(return_value);

	/*
	label = memnew( Label );
	label->set_anchor( MARGIN_TOP, ANCHOR_END );
	label->set_anchor( MARGIN_BOTTOM, ANCHOR_END );

	label->set_begin( Point2( 15,54) );
	label->set_end( Point2( 16,44) );
	label->set_text("Parameters:");

	add_child(label);
	*/


	call_params = memnew( CallDialogParams );
	set_as_toplevel(true);
}


CallDialog::~CallDialog()
{
	memdelete(call_params);
}
#endif
