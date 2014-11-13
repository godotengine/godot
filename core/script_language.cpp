/*************************************************************************/
/*  script_language.cpp                                                  */
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
#include "script_language.h"

ScriptLanguage *ScriptServer::_languages[MAX_LANGUAGES];
int ScriptServer::_language_count=0;

bool ScriptServer::scripting_enabled=true;


void Script::_notification( int p_what) {

	if (p_what==NOTIFICATION_POSTINITIALIZE) {

		if (ScriptDebugger::get_singleton())
			ScriptDebugger::get_singleton()->set_break_language(get_language());
	}
}

void Script::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("can_instance"),&Script::can_instance);
	//ObjectTypeDB::bind_method(_MD("instance_create","base_object"),&Script::instance_create);
	ObjectTypeDB::bind_method(_MD("instance_has","base_object"),&Script::instance_has);
	ObjectTypeDB::bind_method(_MD("has_source_code"),&Script::has_source_code);
	ObjectTypeDB::bind_method(_MD("get_source_code"),&Script::get_source_code);
	ObjectTypeDB::bind_method(_MD("set_source_code","source"),&Script::set_source_code);
	ObjectTypeDB::bind_method(_MD("reload"),&Script::reload);

}

void ScriptServer::set_scripting_enabled(bool p_enabled) {

	scripting_enabled=p_enabled;
}

bool ScriptServer::is_scripting_enabled() {
	
	return scripting_enabled;
}


int ScriptServer::get_language_count() {
	
	return _language_count;
	
}

ScriptLanguage* ScriptServer::get_language(int p_idx) {

	ERR_FAIL_INDEX_V(p_idx,_language_count,NULL);
	
	return _languages[p_idx];
}

void ScriptServer::register_language(ScriptLanguage *p_language) {
	
	ERR_FAIL_COND( _language_count >= MAX_LANGUAGES );
	_languages[_language_count++]=p_language;
}

void ScriptServer::init_languages() {

	for(int i=0;i<_language_count;i++) {
		_languages[i]->init();
	}
}

Variant ScriptInstance::call(const StringName& p_method,VARIANT_ARG_DECLARE) {

	VARIANT_ARGPTRS;
	int argc=0;
	for(int i=0;i<VARIANT_ARG_MAX;i++) {
		if (argptr[i]->get_type()==Variant::NIL)
			break;
		argc++;
	}

	Variant::CallError error;
	return call(p_method,argptr,argc,error);
}


void ScriptInstance::call_multilevel(const StringName& p_method,const Variant** p_args,int p_argcount) {
	Variant::CallError ce;
	call(p_method,p_args,p_argcount,ce); // script may not support multilevel calls
}

void ScriptInstance::call_multilevel_reversed(const StringName& p_method,const Variant** p_args,int p_argcount) {
	Variant::CallError ce;
	call(p_method,p_args,p_argcount,ce); // script may not support multilevel calls
}

void ScriptInstance::call_multilevel(const StringName& p_method,VARIANT_ARG_DECLARE) {

	VARIANT_ARGPTRS;
	int argc=0;
	for(int i=0;i<VARIANT_ARG_MAX;i++) {
		if (argptr[i]->get_type()==Variant::NIL)
			break;
		argc++;
	}

	Variant::CallError error;
	call_multilevel(p_method,argptr,argc);
}

ScriptInstance::~ScriptInstance() {


}

void ScriptLanguage::frame() {


}

ScriptDebugger * ScriptDebugger::singleton=NULL;


void ScriptDebugger::set_lines_left(int p_left) {

	lines_left=p_left;
}

int ScriptDebugger::get_lines_left() const {

	return lines_left;
}

void ScriptDebugger::set_depth(int p_depth) {

	depth=p_depth;
}

int ScriptDebugger::get_depth() const {

	return depth;
}

void ScriptDebugger::insert_breakpoint(int p_line, const StringName& p_source) {

	if (!breakpoints.has(p_line))
		breakpoints[p_line]=Set<StringName>();
	breakpoints[p_line].insert(p_source);

}

void ScriptDebugger::remove_breakpoint(int p_line, const StringName& p_source) {

	if (!breakpoints.has(p_line))
		return;

	breakpoints[p_line].erase(p_source);
	if (breakpoints[p_line].size()==0)
		breakpoints.erase(p_line);
}
bool ScriptDebugger::is_breakpoint(int p_line,const StringName& p_source) const {

	if (!breakpoints.has(p_line))
		return false;
	return breakpoints[p_line].has(p_source);

}
bool ScriptDebugger::is_breakpoint_line(int p_line) const {

	return breakpoints.has(p_line);

}


String ScriptDebugger::breakpoint_find_source(const String& p_source) const {

	return p_source;
}

void ScriptDebugger::clear_breakpoints() {

	breakpoints.clear();
}

void ScriptDebugger::idle_poll() {


}


void ScriptDebugger::line_poll() {


}

void ScriptDebugger::set_break_language(ScriptLanguage *p_lang) {

	break_lang=p_lang;
}

ScriptLanguage* ScriptDebugger::get_break_language() const{

	return break_lang;
}


ScriptDebugger::ScriptDebugger() {

	singleton=this;
	lines_left=-1;
	depth=-1;
	break_lang=NULL;
}


bool PlaceHolderScriptInstance::set(const StringName& p_name, const Variant& p_value) {

	if (values.has(p_name)) {
		values[p_name]=p_value;
		return true;
	}
	return false;
}
bool PlaceHolderScriptInstance::get(const StringName& p_name, Variant &r_ret) const {

	if (values.has(p_name)) {
		r_ret=values[p_name];
		return true;
	} return false;
}

void PlaceHolderScriptInstance::get_property_list(List<PropertyInfo> *p_properties) const {

	for(const List<PropertyInfo>::Element *E=properties.front();E;E=E->next()) {
		p_properties->push_back(E->get());
	}
}

void PlaceHolderScriptInstance::update(const List<PropertyInfo> &p_properties,const Map<StringName,Variant>& p_values) {


	Set<StringName> new_values;
	for(const List<PropertyInfo>::Element *E=p_properties.front();E;E=E->next()) {

		StringName n = E->get().name;
		new_values.insert(n);

		if (!values.has(n) || values[n].get_type()!=E->get().type) {

			if (p_values.has(n))
				values[n]=p_values[n];
		}
	}

	properties=p_properties;
	List<StringName> to_remove;

	for(Map<StringName,Variant>::Element *E=values.front();E;E=E->next()) {

		if (!new_values.has(E->key()))
			to_remove.push_back(E->key());
	}

	while(to_remove.size()) {

		values.erase(to_remove.front()->get());
		to_remove.pop_front();
	}

	if (owner && owner->get_script_instance()==this) {

		owner->_change_notify();
	}
	//change notify
}

PlaceHolderScriptInstance::PlaceHolderScriptInstance(ScriptLanguage *p_language, Ref<Script> p_script,Object *p_owner) {

	language=p_language;
	script=p_script;
	owner=p_owner;
}

PlaceHolderScriptInstance::~PlaceHolderScriptInstance() {

	if (script.is_valid()) {
		script->_placeholder_erased(this);
	}
}
