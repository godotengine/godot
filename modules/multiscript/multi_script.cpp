/*************************************************************************/
/*  multi_script.cpp                                                     */
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
#include "multi_script.h"

bool MultiScriptInstance::set(const StringName& p_name, const Variant& p_value) {

	ScriptInstance **sarr = instances.ptr();
	int sc = instances.size();

	for(int i=0;i<sc;i++) {

		if (!sarr[i])
			continue;

		bool found = sarr[i]->set(p_name,p_value);
		if (found)
			return true;
	}

	if (String(p_name).begins_with("script_")) {
		bool valid;
		owner->set(p_name,p_value,&valid);
		return valid;
	}
	return false;

}

bool MultiScriptInstance::get(const StringName& p_name, Variant &r_ret) const{

	ScriptInstance **sarr = instances.ptr();
	int sc = instances.size();

	for(int i=0;i<sc;i++) {

		if (!sarr[i])
			continue;

		bool found = sarr[i]->get(p_name,r_ret);
		if (found)
			return true;
	}
	if (String(p_name).begins_with("script_")) {
		bool valid;
		r_ret=owner->get(p_name,&valid);
		return valid;
	}
	return false;

}
void MultiScriptInstance::get_property_list(List<PropertyInfo> *p_properties) const{

	ScriptInstance **sarr = instances.ptr();
	int sc = instances.size();


	Set<String> existing;

	for(int i=0;i<sc;i++) {

		if (!sarr[i])
			continue;

		List<PropertyInfo> pl;
		sarr[i]->get_property_list(&pl);

		for(List<PropertyInfo>::Element *E=pl.front();E;E=E->next()) {

			if (existing.has(E->get().name))
				continue;

			p_properties->push_back(E->get());
			existing.insert(E->get().name);
		}
	}

	p_properties->push_back( PropertyInfo(Variant::NIL,"Scripts",PROPERTY_HINT_NONE,String(),PROPERTY_USAGE_CATEGORY) );

	for(int i=0;i<owner->scripts.size();i++) {

		p_properties->push_back( PropertyInfo(Variant::OBJECT,"script_"+String::chr('a'+i),PROPERTY_HINT_RESOURCE_TYPE,"Script",PROPERTY_USAGE_EDITOR) );

	}

	if (owner->scripts.size()<25) {

		p_properties->push_back( PropertyInfo(Variant::OBJECT,"script_"+String::chr('a'+(owner->scripts.size())),PROPERTY_HINT_RESOURCE_TYPE,"Script",PROPERTY_USAGE_EDITOR) );
	}

}

void MultiScriptInstance::get_method_list(List<MethodInfo> *p_list) const{

	ScriptInstance **sarr = instances.ptr();
	int sc = instances.size();


	Set<StringName> existing;

	for(int i=0;i<sc;i++) {

		if (!sarr[i])
			continue;

		List<MethodInfo> ml;
		sarr[i]->get_method_list(&ml);

		for(List<MethodInfo>::Element *E=ml.front();E;E=E->next()) {

			if (existing.has(E->get().name))
				continue;

			p_list->push_back(E->get());
			existing.insert(E->get().name);
		}
	}

}
bool MultiScriptInstance::has_method(const StringName& p_method) const{

	ScriptInstance **sarr = instances.ptr();
	int sc = instances.size();

	for(int i=0;i<sc;i++) {

		if (!sarr[i])
			continue;

		if (sarr[i]->has_method(p_method))
			return true;
	}

	return false;

}

Variant MultiScriptInstance::call(const StringName& p_method,const Variant** p_args,int p_argcount,Variant::CallError &r_error) {

	ScriptInstance **sarr = instances.ptr();
	int sc = instances.size();

	for(int i=0;i<sc;i++) {

		if (!sarr[i])
			continue;

		Variant r = sarr[i]->call(p_method,p_args,p_argcount,r_error);
		if (r_error.error==Variant::CallError::CALL_OK)
			return r;
		else if (r_error.error!=Variant::CallError::CALL_ERROR_INVALID_METHOD)
			return r;
	}

	r_error.error=Variant::CallError::CALL_ERROR_INVALID_METHOD;
	return Variant();

}

void MultiScriptInstance::call_multilevel(const StringName& p_method,const Variant** p_args,int p_argcount){

	ScriptInstance **sarr = instances.ptr();
	int sc = instances.size();

	for(int i=0;i<sc;i++) {

		if (!sarr[i])
			continue;

		sarr[i]->call_multilevel(p_method,p_args,p_argcount);
	}


}
void MultiScriptInstance::notification(int p_notification){

	ScriptInstance **sarr = instances.ptr();
	int sc = instances.size();

	for(int i=0;i<sc;i++) {

		if (!sarr[i])
			continue;

		sarr[i]->notification(p_notification);
	}

}


Ref<Script> MultiScriptInstance::get_script() const {

	return owner;
}

ScriptLanguage *MultiScriptInstance::get_language() {

	return MultiScriptLanguage::get_singleton();
}

MultiScriptInstance::~MultiScriptInstance() {

	owner->remove_instance(object);
}


///////////////////


bool MultiScript::is_tool() const {

	for(int i=0;i<scripts.size();i++) {

		if (scripts[i]->is_tool())
			return true;
	}

	return false;
}

bool MultiScript::_set(const StringName& p_name, const Variant& p_value) {

	_THREAD_SAFE_METHOD_

	String s = String(p_name);
	if (s.begins_with("script_")) {

		int idx = s[7];
		if (idx==0)
			return false;
		idx-='a';

		ERR_FAIL_COND_V(idx<0,false);

		Ref<Script> s = p_value;

		if (idx<scripts.size()) {


			if (s.is_null())
				remove_script(idx);
			else
				set_script(idx,s);
		} else if (idx==scripts.size()) {
			if (s.is_null())
				return false;
			add_script(s);
		} else
			return false;

		return true;
	}

	return false;
}

bool MultiScript::_get(const StringName& p_name,Variant &r_ret) const{

	_THREAD_SAFE_METHOD_

	String s = String(p_name);
	if (s.begins_with("script_")) {

		int idx = s[7];
		if (idx==0)
			return false;
		idx-='a';

		ERR_FAIL_COND_V(idx<0,false);

		if (idx<scripts.size()) {

			r_ret=get_script(idx);
			return true;
		} else if (idx==scripts.size()) {
			r_ret=Ref<Script>();
			return true;
		}
	}

	return false;
}
void MultiScript::_get_property_list( List<PropertyInfo> *p_list) const{

	_THREAD_SAFE_METHOD_

	for(int i=0;i<scripts.size();i++) {

		p_list->push_back( PropertyInfo(Variant::OBJECT,"script_"+String::chr('a'+i),PROPERTY_HINT_RESOURCE_TYPE,"Script") );

	}

	if (scripts.size()<25) {

		p_list->push_back( PropertyInfo(Variant::OBJECT,"script_"+String::chr('a'+(scripts.size())),PROPERTY_HINT_RESOURCE_TYPE,"Script") );
	}
}

void MultiScript::set_script(int p_idx,const Ref<Script>& p_script ) {

	_THREAD_SAFE_METHOD_

	ERR_FAIL_INDEX(p_idx,scripts.size());
	ERR_FAIL_COND( p_script.is_null() );

	scripts[p_idx]=p_script;
	Ref<Script> s=p_script;

	for (Map<Object*,MultiScriptInstance*>::Element *E=instances.front();E;E=E->next()) {


		MultiScriptInstance*msi=E->get();
		ScriptInstance *si = msi->instances[p_idx];
		if (si) {
			msi->instances[p_idx]=NULL;
			memdelete(si);
		}

		if (p_script->can_instance())
			msi->instances[p_idx]=s->instance_create(msi->object);

	}


}


Ref<Script> MultiScript::get_script(int p_idx) const{

	_THREAD_SAFE_METHOD_

	ERR_FAIL_INDEX_V(p_idx,scripts.size(),Ref<Script>());

	return scripts[p_idx];

}
void MultiScript::add_script(const Ref<Script>& p_script){

	_THREAD_SAFE_METHOD_
	ERR_FAIL_COND( p_script.is_null() );
	scripts.push_back(p_script);
	Ref<Script> s=p_script;

	for (Map<Object*,MultiScriptInstance*>::Element *E=instances.front();E;E=E->next()) {


		MultiScriptInstance*msi=E->get();

		if (p_script->can_instance())
			msi->instances.push_back( s->instance_create(msi->object) );
		else
			msi->instances.push_back(NULL);

		msi->object->_change_notify();

	}


	_change_notify();
}


void MultiScript::remove_script(int p_idx) {

	_THREAD_SAFE_METHOD_

	ERR_FAIL_INDEX(p_idx,scripts.size());

	scripts.remove(p_idx);

	for (Map<Object*,MultiScriptInstance*>::Element *E=instances.front();E;E=E->next()) {


		MultiScriptInstance*msi=E->get();
		ScriptInstance *si = msi->instances[p_idx];
		msi->instances.remove(p_idx);
		if (si) {
			memdelete(si);
		}

		msi->object->_change_notify();
	}


}


void MultiScript::remove_instance(Object *p_object) {

	_THREAD_SAFE_METHOD_
	instances.erase(p_object);
}

bool MultiScript::can_instance() const {

	return true;
}

StringName MultiScript::get_instance_base_type() const {

	return StringName();
}
ScriptInstance* MultiScript::instance_create(Object *p_this) {

	_THREAD_SAFE_METHOD_
	MultiScriptInstance *msi = memnew( MultiScriptInstance );
	msi->object=p_this;
	msi->owner=this;
	for(int i=0;i<scripts.size();i++) {

		ScriptInstance *si;

		if (scripts[i]->can_instance())
			si = scripts[i]->instance_create(p_this);
		else
			si=NULL;

		msi->instances.push_back(si);
	}

	instances[p_this]=msi;
	p_this->_change_notify();
	return msi;
}
bool MultiScript::instance_has(const Object *p_this) const {

	_THREAD_SAFE_METHOD_
	return instances.has((Object*)p_this);
}

bool MultiScript::has_source_code() const {

	return false;
}
String MultiScript::get_source_code() const {

	return "";
}
void MultiScript::set_source_code(const String& p_code) {


}
Error MultiScript::reload() {

	for(int i=0;i<scripts.size();i++)
		scripts[i]->reload();

	return OK;
}

String MultiScript::get_node_type() const {

	return "";
}

void MultiScript::_bind_methods() {


}

ScriptLanguage *MultiScript::get_language() const {

	return MultiScriptLanguage::get_singleton();
}


///////////////

MultiScript::MultiScript() {
}


MultiScriptLanguage *MultiScriptLanguage::singleton=NULL;
