/*************************************************************************/
/*  array_property_edit.cpp                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
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
#include "array_property_edit.h"

#include "editor_node.h"

#define ITEMS_PER_PAGE 100

Variant ArrayPropertyEdit::get_array() const{

	Object*o = ObjectDB::get_instance(obj);
	if (!o)
		return Array();
	Variant arr=o->get(property);
	if (!arr.is_array()) {
		Variant::CallError ce;
		arr=Variant::construct(default_type,NULL,0,ce);
	}
	return arr;
}

void ArrayPropertyEdit::_notif_change() {
	_change_notify();
}
void ArrayPropertyEdit::_notif_changev(const String& p_v) {

	_change_notify(p_v.utf8().get_data());
}

void ArrayPropertyEdit::_set_size(int p_size) {

	Variant arr = get_array();
	arr.call("resize",p_size);
	Object*o = ObjectDB::get_instance(obj);
	if (!o)
		return;

	o->set(property,arr);

}

void ArrayPropertyEdit::_set_value(int p_idx,const Variant& p_value) {

	Variant arr = get_array();
	arr.set(p_idx,p_value);
	Object*o = ObjectDB::get_instance(obj);
	if (!o)
		return;

	o->set(property,arr);
}

bool ArrayPropertyEdit::_set(const StringName& p_name, const Variant& p_value){

	String pn=p_name;

	if (pn.begins_with("array/")) {

		if (pn=="array/size") {

			Variant arr = get_array();
			int size = arr.call("size");

			int newsize=p_value;
			if (newsize==size)
				return true;

			UndoRedo *ur = EditorNode::get_singleton()->get_undo_redo();
			ur->create_action(TTR("Resize Array"));
			ur->add_do_method(this,"_set_size",newsize);
			ur->add_undo_method(this,"_set_size",size);
			if (newsize<size) {
				for(int i=newsize;i<size;i++) {
					ur->add_undo_method(this,"_set_value",i,arr.get(i));

				}
			} else if (newsize>size && size) {

				Variant init;
				Variant::CallError ce;
				init = Variant::construct(arr.get(size-1).get_type(),NULL,0,ce);
				for(int i=size;i<newsize;i++) {
					ur->add_do_method(this,"_set_value",i,init);
				}

			}
			ur->add_do_method(this,"_notif_change");
			ur->add_undo_method(this,"_notif_change");
			ur->commit_action();
			return true;
		}
		if (pn=="array/page") {
			page=p_value;
			_change_notify();
			return true;
		}

	} else if (pn.begins_with("indices")) {

		if (pn.find("_")!=-1) {
			//type
			int idx=pn.get_slicec('/',1).get_slicec('_',0).to_int();

			int type = p_value;

			Variant arr = get_array();

			Variant value = arr.get(idx);
			if (value.get_type()!=type && type>=0 && type<Variant::VARIANT_MAX) {
				Variant::CallError ce;
				Variant new_value=Variant::construct(Variant::Type(type),NULL,0,ce);
				UndoRedo *ur = EditorNode::get_singleton()->get_undo_redo();

				ur->create_action(TTR("Change Array Value Type"));
				ur->add_do_method(this,"_set_value",idx,new_value);
				ur->add_undo_method(this,"_set_value",idx,value);
				ur->add_do_method(this,"_notif_change");
				ur->add_undo_method(this,"_notif_change");
				ur->commit_action();

			}
			return true;

		} else {
			int idx=pn.get_slicec('/',1).to_int();
			Variant arr = get_array();

			Variant value = arr.get(idx);
			UndoRedo *ur = EditorNode::get_singleton()->get_undo_redo();

			ur->create_action(TTR("Change Array Value"));
			ur->add_do_method(this,"_set_value",idx,p_value);
			ur->add_undo_method(this,"_set_value",idx,value);
			ur->add_do_method(this,"_notif_changev",p_name);
			ur->add_undo_method(this,"_notif_changev",p_name);
			ur->commit_action();
			return true;
		}
	}

	return false;
}

bool ArrayPropertyEdit::_get(const StringName& p_name,Variant &r_ret) const {

	Variant arr = get_array();
	//int size = arr.call("size");

	String pn=p_name;
	if (pn.begins_with("array/")) {

		if (pn=="array/size") {
			r_ret=arr.call("size");
			return true;
		}
		if (pn=="array/page") {
			r_ret=page;
			return true;
		}
	} else if (pn.begins_with("indices")) {

		if (pn.find("_")!=-1) {
			//type
			int idx=pn.get_slicec('/',1).get_slicec('_',0).to_int();
			bool valid;
			r_ret=arr.get(idx,&valid);
			if (valid)
				r_ret=r_ret.get_type();
			return valid;

		} else {
			int idx=pn.get_slicec('/',1).to_int();
			bool valid;
			r_ret=arr.get(idx,&valid);
			return valid;
		}
	}

	return false;
}

void ArrayPropertyEdit::_get_property_list( List<PropertyInfo> *p_list) const{

	Variant arr = get_array();
	int size = arr.call("size");

	p_list->push_back( PropertyInfo(Variant::INT,"array/size",PROPERTY_HINT_RANGE,"0,100000,1") );
	int pages = size/ITEMS_PER_PAGE;
	if (pages>0)
		p_list->push_back( PropertyInfo(Variant::INT,"array/page",PROPERTY_HINT_RANGE,"0,"+itos(pages)+",1") );

	int offset=page*ITEMS_PER_PAGE;

	int items=MIN(size-offset,ITEMS_PER_PAGE);


	for(int i=0;i<items;i++) {

		Variant v=arr.get(i+offset);
		if (arr.get_type()==Variant::ARRAY) {
			p_list->push_back(PropertyInfo(Variant::INT,"indices/"+itos(i+offset)+"_type",PROPERTY_HINT_ENUM,vtypes));
		}
		if (arr.get_type()!=Variant::ARRAY || v.get_type()!=Variant::NIL) {
			PropertyInfo pi(v.get_type(),"indices/"+itos(i+offset));
			if (v.get_type()==Variant::OBJECT) {
				pi.hint=PROPERTY_HINT_RESOURCE_TYPE;
				pi.hint_string="Resource";
			}
			p_list->push_back(pi);
		}
	}

}

void ArrayPropertyEdit::edit(Object* p_obj,const StringName& p_prop,Variant::Type p_deftype) {

	page=0;
	property=p_prop;
	obj=p_obj->get_instance_ID();
	default_type=p_deftype;

}

Node *ArrayPropertyEdit::get_node() {

	Object *o = ObjectDB::get_instance(obj);
	if (!o)
		return NULL;

	return o->cast_to<Node>();
}

void ArrayPropertyEdit::_bind_methods() {

	ClassDB::bind_method(_MD("_set_size"),&ArrayPropertyEdit::_set_size);
	ClassDB::bind_method(_MD("_set_value"),&ArrayPropertyEdit::_set_value);
	ClassDB::bind_method(_MD("_notif_change"),&ArrayPropertyEdit::_notif_change);
	ClassDB::bind_method(_MD("_notif_changev"),&ArrayPropertyEdit::_notif_changev);
}

ArrayPropertyEdit::ArrayPropertyEdit()
{
	page=0;
	for(int i=0;i<Variant::VARIANT_MAX;i++) {

		if (i>0)
			vtypes+=",";
		vtypes+=Variant::get_type_name( Variant::Type(i) );
	}
	default_type=Variant::NIL;

}
