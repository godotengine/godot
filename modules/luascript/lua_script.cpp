/*************************************************************************/
/*  lua_script.cpp                                                       */
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
#include <stdlib.h>
#include <string.h>
#include "lua_script.h"
#include "globals.h"
#include "global_constants.h"
//#include "gd_compiler.h"
#include "os/file_access.h"

/* TODO:

   *populate globals
   *do checks as close to debugger as possible (but don't do debugger)
   *const check plz
   *check arguments and default arguments in GDFunction
   -get property list in instance?
   *missing opcodes
   -const checks
   -make thread safe
 */
static void l_push_variant(lua_State *L, const Variant& var);


LuaNativeClass::LuaNativeClass(const StringName& p_name) {

	name=p_name;
}

/*void LuaNativeClass::call_multilevel(const StringName& p_method,const Variant** p_args,int p_argcount){


}*/


bool LuaNativeClass::_get(const StringName& p_name,Variant &r_ret) const {

	bool ok;
	int v = ObjectTypeDB::get_integer_constant(name, p_name, &ok);

	if (ok) {
		r_ret=v;
		return true;
	} else {
		return false;
	}
}


void LuaNativeClass::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("new"),&LuaNativeClass::_new);
}

Variant LuaNativeClass::_new() {

	Object *o = instance();
	if (!o) {
		ERR_EXPLAIN("Class type: '"+String(name)+"' is not instantiable.");
		ERR_FAIL_COND_V(!o,Variant());
	}

	Reference *ref = o->cast_to<Reference>();
	if (ref) {
		return REF(ref);
	} else {
		return o;
	}

}

Object *LuaNativeClass::instance() {

	return ObjectTypeDB::instance(name);
}



LuaInstance* LuaScript::_create_instance(const Variant** p_args,int p_argcount,Object *p_owner,bool p_isref) {

	/* STEP 1, CREATE */
	LuaInstance* instance = memnew( LuaInstance );
	instance->base_ref=p_isref;
//	instance->members.resize(member_indices.size());
	instance->script=Ref<LuaScript>(this);
	instance->owner=p_owner;
	instance->owner->set_script_instance(instance);

	/* STEP 2, INITIALIZE AND CONSRTUCT */
	instances.insert(instance->owner);

    if(instance->init() != OK)
    {
		instance->script=Ref<LuaScript>();
		instances.erase(p_owner);
		memdelete(instance);
		ERR_FAIL_V(NULL); //error consrtucting
    }

	//@TODO make thread safe
	return instance;

}

Variant LuaScript::_new(const Variant** p_args,int p_argcount,Variant::CallError& r_error) {

	/* STEP 1, CREATE */

	r_error.error=Variant::CallError::CALL_OK;
	REF ref;
	Object *owner=NULL;

	LuaScript *_baseptr=this;
	while (_baseptr->_base) {
		_baseptr=_baseptr->_base;
	}

	if (_baseptr->native.ptr()) {
		owner=_baseptr->native->instance();
	} else {
		owner=memnew( Reference ); //by default, no base means use reference
	}

	Reference *r=owner->cast_to<Reference>();
	if (r) {
		ref=REF(r);
	}


	LuaInstance* instance = _create_instance(p_args,p_argcount,owner,r!=NULL);
	if (!instance) {
		if (ref.is_null()) {
			memdelete(owner); //no owner, sorry
		}
		return Variant();
	}

	if (ref.is_valid()) {		
		return ref;
	} else {
		return owner;
	}
}

bool LuaScript::can_instance() const {

	return valid; //any script in LuaScript can instance
}

StringName LuaScript::get_instance_base_type() const {

	if (native.is_valid())
		return native->get_name();
	if (base.is_valid())
		return base->get_instance_base_type();
	return StringName();
}

//struct _GDScriptMemberSort {
//
//	int index;
//	StringName name;
//	_FORCE_INLINE_ bool operator<(const _GDScriptMemberSort& p_member) const { return index < p_member.index; }
//
//};
//

#ifdef TOOLS_ENABLED


void LuaScript::_placeholder_erased(PlaceHolderScriptInstance *p_placeholder) {

	placeholders.erase(p_placeholder);
}

void LuaScript::_update_placeholder(PlaceHolderScriptInstance *p_placeholder) {


	List<PropertyInfo> plist;
	LuaScript *scr=this;

	Map<StringName,Variant> default_values;
	while(scr) {
//
//		Vector<_GDScriptMemberSort> msort;
//		for(Map<StringName,PropertyInfo>::Element *E=scr->member_info.front();E;E=E->next()) {
//
//			_GDScriptMemberSort ms;
//			ERR_CONTINUE(!scr->member_indices.has(E->key()));
//			ms.index=scr->member_indices[E->key()];
//			ms.name=E->key();
//
//			msort.push_back(ms);
//
//		}
//
//		msort.sort();
//		msort.invert();
//		for(int i=0;i<msort.size();i++) {
//
//			plist.push_front(scr->member_info[msort[i].name]);
//			if (scr->member_default_values.has(msort[i].name))
//				default_values[msort[i].name]=scr->member_default_values[msort[i].name];
//			else {
//				Variant::CallError err;
//				default_values[msort[i].name]=Variant::construct(scr->member_info[msort[i].name].type,NULL,0,err);
//			}
//		}

		scr=scr->_base;
	}


	p_placeholder->update(plist,default_values);

}
#endif
ScriptInstance* LuaScript::instance_create(Object *p_this) {

	if (!tool && !ScriptServer::is_scripting_enabled()) {

#ifdef TOOLS_ENABLED

		//instance a fake script for editing the values
		//plist.invert();

		/*print_line("CREATING PLACEHOLDER");
		for(List<PropertyInfo>::Element *E=plist.front();E;E=E->next()) {
			print_line(E->get().name);
		}*/
		PlaceHolderScriptInstance *si = memnew( PlaceHolderScriptInstance(LuaScriptLanguage::get_singleton(),Ref<Script>(this),p_this) );
		placeholders.insert(si);
		_update_placeholder(si);
		return si;
#else
		return NULL;
#endif
	}

	LuaScript *top=this;
	while(top->_base)
		top=top->_base;

	if (top->native.is_valid()) {
		if (!ObjectTypeDB::is_type(p_this->get_type_name(),top->native->get_name())) {

			if (ScriptDebugger::get_singleton()) {
				LuaScriptLanguage::get_singleton()->debug_break_parse(get_path(),0,"Script inherits from native type '"+String(top->native->get_name())+"', so it can't be instanced in object of type: '"+p_this->get_type()+"'");
			}
			ERR_EXPLAIN("Script inherits from native type '"+String(top->native->get_name())+"', so it can't be instanced in object of type: '"+p_this->get_type()+"'");
			ERR_FAIL_V(NULL);

		}
	}

	return _create_instance(NULL,0,p_this,p_this->cast_to<Reference>());

}
bool LuaScript::instance_has(const Object *p_this) const {

	return instances.has((Object*)p_this);
}

bool LuaScript::has_source_code() const {

	return source!="";
}
String LuaScript::get_source_code() const {

	return source;
}
void LuaScript::set_source_code(const String& p_code) {

	source=p_code;

}

//void LuaScript::_set_subclass_path(Ref<LuaScript>& p_sc,const String& p_path) {
//
//	p_sc->path=p_path;
//	for(Map<StringName,Ref<LuaScript> >::Element *E=p_sc->subclasses.front();E;E=E->next()) {
//
//		_set_subclass_path(E->get(),p_path);
//	}
//}

void LuaScript::reportError(const char *fmt, ...) const
{
    char buf[2048];

    LuaScriptLanguage *lang = LuaScriptLanguage::get_singleton();
    MutexLock(lang->get_lock());

    lua_State *L = lang->get_state();
    lua_getglobal(L, "debug");
    lua_getfield(L, -1, "traceback");
    lua_pcall(L, 0, 1, 0);
    const char *trace_back = lua_tostring(L, -1);
    lua_pop(L, 2);

    va_list argp;
    va_start(argp, fmt);
    vsnprintf(buf, sizeof(buf), fmt, argp);
    va_end(argp);

    int error_line = -1;
    const char *ls = strstr(buf, ".lua\"]:");
    while(ls != NULL && ls[0] != '\0')
    {
        ls = strchr(ls, ':');
        if(ls == NULL)
            break;
        error_line = atoi(ls + 1);
        if(error_line > 0)
            break;
        ls ++;
    }

    String path = get_path();

    if (ScriptDebugger::get_singleton()) {
		LuaScriptLanguage::get_singleton()->debug_break_parse(path, error_line, (String(buf) + "\n" + trace_back));
    }
    _err_print_error("LuaScript::reportError",path.empty()?"built-in":(const char*)path.utf8().get_data(),error_line, ((String(buf) + "\n" + trace_back).utf8()));
}

int LuaScript::l_meta_gc(lua_State *L)
{
    LuaScriptLanguage *lang = LuaScriptLanguage::get_singleton();
    MutexLock(lang->get_lock());
    return 0;
}

int LuaScript::l_meta_index(lua_State *L)
{
    LuaScriptLanguage *lang = LuaScriptLanguage::get_singleton();
    MutexLock(lang->get_lock());

    const char *key = luaL_checkstring(L, 2);

    LuaScriptLanguage *lsl = ::LuaScriptLanguage::get_singleton();
    if(lsl->get_global_map().has(key))
    {
        int idx = lsl->get_global_map()[key];
        Variant& var = lsl->get_global_array()[idx];
        LuaInstance::l_push_variant(L, var);
        return 1;
    }

    if(LuaInstance::l_push_bultins_ctor(L, key))
        return 1;

    // get from globals
    lua_getglobal(L, "_G");
    lua_pushvalue(L, 2);
    lua_gettable(L, -2);
    lua_insert(L, -2);
    lua_pop(L, 1);
    return 1;
}

int LuaScript::l_extends(lua_State *L)
{
    LuaScriptLanguage *lang = LuaScriptLanguage::get_singleton();
    MutexLock(lang->get_lock());

    LuaScript *self = (LuaScript *) lua_touserdata(L, lua_upvalueindex(1));
    const char *base = luaL_checkstring(L, 1);
    // todo: check base type~~~
    //  extends "Node" -- engine class
    //  extends "res://class.lua" -- script base class

	//if not found, try engine classes
	if (!LuaScriptLanguage::get_singleton()->get_global_map().has(base)) {
        self->reportError("Unknown class: '%s'", base);
		return 0;
	}

	int base_idx = LuaScriptLanguage::get_singleton()->get_global_map()[base];
    Ref<LuaScript> script;
    Ref<LuaNativeClass> native;
    
    native = LuaScriptLanguage::get_singleton()->get_global_array()[base_idx];
	if (!native.is_valid()) {
        self->reportError("Global '%s' not a class : %s", base, self->name.utf8().get_data());
		return 0;
	}

	if (script.is_valid()) {

		self->base=script;
		self->_base=script->base.ptr();
	//	self->member_indices=script->member_indices;

	} else if (native.is_valid()) {

		self->native=native;
	} else {
        self->reportError("Could not determine inheritance : %s", self->name.utf8().get_data());
		return 0;
	}

    lua_rawgeti(L, LUA_REGISTRYINDEX, self->ref);
    return 1;
}

Error LuaScript::reload() {

    LuaScriptLanguage *lang = LuaScriptLanguage::get_singleton();
    MutexLock(lang->get_lock());

    ERR_FAIL_COND_V(instances.size(),ERR_ALREADY_IN_USE);
    reset();

	String basedir=path;

	if (basedir=="")
		basedir=get_path();

	if (basedir!="")
		basedir=basedir.get_base_dir();

    CharString code = source.utf8();

    lua_State *L = LuaScriptLanguage::get_singleton()->get_state();
    int top = lua_gettop(L);
    if(luaL_loadbuffer(L, code.get_data(), code.length(), path.utf8()))
    {
        const char *err = lua_tostring(L, -1);
        reportError("Parse Error: %s", err);
		ERR_FAIL_V(ERR_PARSE_ERROR);
    }
    // new object's mtable(method table)
    lua_newtable(L);
    // make a ref to mtable
    lua_pushvalue(L, -1);
    ref = luaL_ref(L, LUA_REGISTRYINDEX);
    {
        lua_pushlightuserdata(L, this);
        lua_pushcclosure(L, l_extends, 1);
        lua_setfield(L, -2, "extends");

        lua_newtable(L);
        luaL_reg meta_methods[] = {
            { "__index", l_meta_index },
            { "__gc", l_meta_gc },
            { NULL, NULL },
        };
        luaL_register(L, NULL, meta_methods);
        lua_setmetatable(L, -2);
    }
    lua_setfenv(L, -2);
    //lua_pushcfunction(L, LuaScriptLanguage::panic);
    //lua_insert(L, -2);
    if(lua_pcall(L, 0, 0, 0/*-2*/))
    {
        const char *err = lua_tostring(L, -1);
        reportError("Execute Error: %s", err);
		ERR_FAIL_V(ERR_SCRIPT_FAILED);
    }
    lua_settop(L, top);

	valid=true;

//	for(Map<StringName,Ref<LuaScript> >::Element *E=subclasses.front();E;E=E->next()) {
//
//		_set_subclass_path(E->get(),path);
//	}

#ifdef TOOLS_ENABLED
	for (Set<PlaceHolderScriptInstance*>::Element *E=placeholders.front();E;E=E->next()) {

		_update_placeholder(E->get());
	}
#endif
	return OK;
}

String LuaScript::get_node_type() const {

	return ""; // ?
}

ScriptLanguage *LuaScript::get_language() const {

	return LuaScriptLanguage::get_singleton();
}


//Variant LuaScript::call(const StringName& p_method,const Variant** p_args,int p_argcount,Variant::CallError &r_error) {
//
//
//	LuaScript *top=this;
//	while(top) {
//
//		Map<StringName,GDFunction>::Element *E=top->member_functions.find(p_method);
//		if (E) {
//
//			if (!E->get().is_static()) {
//				WARN_PRINT(String("Can't call non-static function: '"+String(p_method)+"' in script.").utf8().get_data());
//			}
//
//			return E->get().call(NULL,p_args,p_argcount,r_error);
//		}
//		top=top->_base;
//	}
//
//	//none found, regular
//
//	return Script::call(p_method,p_args,p_argcount,r_error);
//
//}
//
//bool LuaScript::_get(const StringName& p_name,Variant &r_ret) const {
//
//	{
//
//
//		const LuaScript *top=this;
//		while(top) {
//
//			{
//				const Map<StringName,Variant>::Element *E=top->constants.find(p_name);
//				if (E) {
//
//					r_ret= E->get();
//					return true;
//				}
//			}
//
//			{
//				const Map<StringName,Ref<LuaScript> >::Element *E=subclasses.find(p_name);
//				if (E) {
//
//					r_ret=E->get();
//					return true;
//				}
//			}
//			top=top->_base;
//		}
//
//		if (p_name==LuaScriptLanguage::get_singleton()->strings._script_source) {
//
//			r_ret=get_source_code();
//			return true;
//		}
//	}
//
//
//
//	return false;
//
//}
//bool LuaScript::_set(const StringName& p_name, const Variant& p_value) {
//
//	if (p_name==LuaScriptLanguage::get_singleton()->strings._script_source) {
//
//		set_source_code(p_value);
//		reload();
//	} else
//		return false;
//
//	return true;
//}
//
//void LuaScript::_get_property_list(List<PropertyInfo> *p_properties) const {
//
//	p_properties->push_back( PropertyInfo(Variant::STRING,"script/source",PROPERTY_HINT_NONE,"",PROPERTY_USAGE_NOEDITOR) );
//}


void LuaScript::_bind_methods()
{
	ObjectTypeDB::bind_native_method(METHOD_FLAGS_DEFAULT,"new",&LuaScript::_new,MethodInfo("new"));	
}

Error LuaScript::load_source_code(const String& p_path) {


	DVector<uint8_t> sourcef;
	Error err;
	FileAccess *f=FileAccess::open(p_path,FileAccess::READ,&err);
	if (err) {

		ERR_FAIL_COND_V(err,err);
	}

	int len = f->get_len();
	sourcef.resize(len+1);
	DVector<uint8_t>::Write w = sourcef.write();
	int r = f->get_buffer(w.ptr(),len);
	f->close();
	memdelete(f);
	ERR_FAIL_COND_V(r!=len,ERR_CANT_OPEN);
	w[len]=0;

	String s;
	if (s.parse_utf8((const char*)w.ptr())) {

		ERR_EXPLAIN("Script '"+p_path+"' contains invalid unicode (utf-8), so it was not loaded. Please ensure that scripts are saved in valid utf-8 unicode.");
		ERR_FAIL_V(ERR_INVALID_DATA);
	}

	source=s;
	path=p_path;
	return OK;

}


//const Map<StringName,GDFunction>& LuaScript::debug_get_member_functions() const {
//
//	return member_functions;
//}



//StringName LuaScript::debug_get_member_by_index(int p_idx) const {
//
//
//	for(const Map<StringName,int>::Element *E=member_indices.front();E;E=E->next()) {
//
//		if (E->get()==p_idx)
//			return E->key();
//	}
//
//	return "<error>";
//}


//Ref<LuaScript> LuaScript::get_base() const {
//
//	return base;
//}

void LuaScript::reset()
{
    LuaScriptLanguage *lang = LuaScriptLanguage::get_singleton();
    MutexLock(lang->get_lock());

    if(ref != LUA_NOREF)
    {
        lua_State *L = LuaScriptLanguage::get_singleton()->get_state();
        lua_rawgeti(L, LUA_REGISTRYINDEX, ref);
        lua_pushnil(L);
        lua_setmetatable(L, -2);
        // remove 'extends' function
        lua_pushnil(L);
        lua_setfield(L, -2, "extends");
        lua_pop(L, 1);
        // unref from lua
        luaL_unref(L, LUA_REGISTRYINDEX, ref);
        ref = LUA_NOREF;
    }
    valid=false;
}

LuaScript::LuaScript() {
    valid=false;
//	subclass_count=0;
//	initializer=NULL;
	_base=NULL;
//	_owner=NULL;
    ref=LUA_NOREF;
	tool=false;
}

LuaScript::~LuaScript()
{
    reset();
}

/************* SCRIPT LANGUAGE **************/
/************* SCRIPT LANGUAGE **************/
/************* SCRIPT LANGUAGE **************/
/************* SCRIPT LANGUAGE **************/
/************* SCRIPT LANGUAGE **************/

LuaScriptLanguage *LuaScriptLanguage::singleton=NULL;


String LuaScriptLanguage::get_name() const {

	return "LuaScript";
}

/* LANGUAGE FUNCTIONS */

void LuaScriptLanguage::_add_global(const StringName& p_name,const Variant& p_value) {


	if (globals.has(p_name)) {
		//overwrite existing
		global_array[globals[p_name]]=p_value;
		return;
	}
	globals[p_name]=global_array.size();
	global_array.push_back(p_value);
	_global_array=global_array.ptr();
}

void LuaScriptLanguage::init() {

    // setup lua instance object's metamethods
    LuaInstance::setup();

	//populate global constants
	int gcc=GlobalConstants::get_global_constant_count();
	for(int i=0;i<gcc;i++) {

		_add_global(StaticCString::create(GlobalConstants::get_global_constant_name(i)),GlobalConstants::get_global_constant_value(i));
	}

	_add_global(StaticCString::create("PI"),Math_PI);

	//populate native classes

	List<String> class_list;
	ObjectTypeDB::get_type_list(&class_list);
	for(List<String>::Element *E=class_list.front();E;E=E->next()) {

		StringName n = E->get();
		String s = String(n);
		if (s.begins_with("_"))
			n=s.substr(1,s.length());

		if (globals.has(n))
			continue;
		Ref<LuaNativeClass> nc = memnew( LuaNativeClass(E->get()) );
		_add_global(n,nc);
	}

	//populate singletons

	List<Globals::Singleton> singletons;
	Globals::get_singleton()->get_singletons(&singletons);
	for(List<Globals::Singleton>::Element *E=singletons.front();E;E=E->next()) {

		_add_global(E->get().name,E->get().ptr);
	}
}

String LuaScriptLanguage::get_type() const {

	return "LuaScript";
}
String LuaScriptLanguage::get_extension() const {

	return "lua";
}
Error LuaScriptLanguage::execute_file(const String& p_path)  {

	// ??
	return OK;
}
void LuaScriptLanguage::finish()  {


}


void LuaScriptLanguage::frame() {

//	print_line("calls: "+itos(calls));
	calls=0;
}

/* EDITOR FUNCTIONS */
void LuaScriptLanguage::get_reserved_words(List<String> *p_words) const  {

	static const char *_reserved_words[]={
        "and",
        "break",
        "do",
        "else",
        "elseif",
        "end",
        "false",
        "for",
        "function",
        "if",
        "in",
        "local",
        "nil",
        "not",
        "or",
        "repeat",
        "return",
        "then",
        "true",
        "until",
        "while",
	0};


	const char **w=_reserved_words;


	while (*w) {

		p_words->push_back(*w);
		w++;
	}

	//for(int i=0;i<GDFunctions::FUNC_MAX;i++) {
	//	p_words->push_back(GDFunctions::get_func_name(GDFunctions::Function(i)));
	//}

}

static void *l_alloc(void *ud, void *ptr, size_t osize, size_t nsize)
{
    if (nsize == 0)
    {
        if (ptr)
            memfree(ptr);
        return NULL;
    }
    else
    {
        if (ptr)
            return memrealloc(ptr, nsize);
        return memalloc(nsize);
    }
    return NULL;
}

int LuaScriptLanguage::panic(lua_State *L)
{
    LuaScriptLanguage *lang = LuaScriptLanguage::get_singleton();
    MutexLock(lang->get_lock());

    const char *s = lua_tostring(L, -1);
    fputs("PANIC: unprotected error in call to Lua API (", stderr);
    fputs(s ? s : "?", stderr);
    fputc(')', stderr); fputc('\n', stderr);
    fflush(stderr);
    return 0;
}

LuaScriptLanguage::LuaScriptLanguage() {

	calls=0;
	ERR_FAIL_COND(singleton);
	singleton=this;
	strings._init = StaticCString::create("_init");
	strings._notification = StaticCString::create("_notification");
	strings._set= StaticCString::create("_set");
	strings._get= StaticCString::create("_get");
	strings._get_property_list= StaticCString::create("_get_property_list");
	strings._script_source=StaticCString::create("script/source");
	_debug_parse_err_line=-1;
	_debug_parse_err_file="";

    _debug_call_stack_pos=0;
    int dmcs=GLOBAL_DEF("debug/script_max_call_stack",1024);
    if (ScriptDebugger::get_singleton()) {
        //debugging enabled!

	_debug_max_call_stack = dmcs;
        if (_debug_max_call_stack<1024)
            _debug_max_call_stack=1024;
	_call_stack = memnew_arr( CallLevel, _debug_max_call_stack+1 );

    } else {
        _debug_max_call_stack=0;
        _call_stack=NULL;
    }
    L = lua_newstate(l_alloc, NULL);
    lua_atpanic(L, panic);
    luaL_openlibs(L);
    lock = Mutex::create();
}


LuaScriptLanguage::~LuaScriptLanguage() {

    if (_call_stack)  {
        memdelete_arr(_call_stack);
    }
    lock->lock();
    if (L) {
        lua_close(L);
    }
    lock->unlock();
    memdelete(lock);
    singleton=NULL;
}

/*************** RESOURCE ***************/

RES ResourceFormatLoaderLuaScript::load(const String &p_path,const String& p_original_path) {

	LuaScript *script = memnew( LuaScript );

	Ref<LuaScript> scriptres(script);

    // lua does not need load_byte_code
    //  bytecode also stored in source
	Error err = script->load_source_code(p_path);

	if (err!=OK) {

		ERR_FAIL_COND_V(err!=OK, RES());
	}

	script->set_script_path(p_original_path); // script needs this.
	script->set_path(p_original_path);
	script->set_name(p_path.get_file());

	script->reload();

	return scriptres;
}
void ResourceFormatLoaderLuaScript::get_recognized_extensions(List<String> *p_extensions) const {

	p_extensions->push_back("lua");
	p_extensions->push_back("luac");
}

bool ResourceFormatLoaderLuaScript::handles_type(const String& p_type) const {

	return (p_type=="Script" || p_type=="LuaScript");
}

String ResourceFormatLoaderLuaScript::get_resource_type(const String &p_path) const {

	String el = p_path.extension().to_lower();
	if (el=="lua" || el=="luac")
		return "LuaScript";
	return "";
}


Error ResourceFormatSaverLuaScript::save(const String &p_path,const RES& p_resource,uint32_t p_flags) {

	Ref<LuaScript> sqscr = p_resource;
	ERR_FAIL_COND_V(sqscr.is_null(),ERR_INVALID_PARAMETER);

	String source = sqscr->get_source_code();

	Error err;
	FileAccess *file = FileAccess::open(p_path,FileAccess::WRITE,&err);


	if (err) {

		ERR_FAIL_COND_V(err,err);
	}

	file->store_string(source);

	file->close();
	memdelete(file);
	return OK;
}

void ResourceFormatSaverLuaScript::get_recognized_extensions(const RES& p_resource,List<String> *p_extensions) const {

	if (p_resource->cast_to<LuaScript>()) {
		p_extensions->push_back("lua");
	}

}
bool ResourceFormatSaverLuaScript::recognize(const RES& p_resource) const {

	return p_resource->cast_to<LuaScript>()!=NULL;
}
