/*************************************************************************/
/*  object.cpp                                                           */
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
#include "object.h"
#include "print_string.h"
#include "object_type_db.h"
#include "script_language.h"
#include "message_queue.h"
#include "core_string_names.h"
#include "translation.h"

#ifdef DEBUG_ENABLED

struct _ObjectDebugLock {

	Object *obj;

	_ObjectDebugLock(Object *p_obj) {
		obj=p_obj;
		obj->_lock_index.ref();
	}
	~_ObjectDebugLock() {
		obj->_lock_index.unref();
	}
};

#define OBJ_DEBUG_LOCK _ObjectDebugLock _debug_lock(this);

#else

#define OBJ_DEBUG_LOCK

#endif

Array convert_property_list(const List<PropertyInfo> * p_list) {

	Array va;
	for (const List<PropertyInfo>::Element *E=p_list->front();E;E=E->next()) {

		const PropertyInfo &pi = E->get();
		Dictionary d;
		d["name"]=pi.name;
		d["type"]=pi.type;
		d["hint"]=pi.hint;
		d["hint_string"]=pi.hint_string;
		d["usage"]=pi.usage;
		va.push_back(d);
	}

	return va;
}

MethodInfo::MethodInfo() {

	id=0;
	flags=METHOD_FLAG_NORMAL;
}

MethodInfo::MethodInfo(const String& p_name) {

	id=0;
	name=p_name;
	flags=METHOD_FLAG_NORMAL;
}
MethodInfo::MethodInfo(const String& p_name, const PropertyInfo& p_param1) {

	id=0;
	name=p_name;
	arguments.push_back( p_param1 );
	flags=METHOD_FLAG_NORMAL;
}
MethodInfo::MethodInfo(const String& p_name, const PropertyInfo& p_param1,const PropertyInfo& p_param2) {

	id=0;
	name=p_name;
	arguments.push_back( p_param1 );
	arguments.push_back( p_param2 );
	flags=METHOD_FLAG_NORMAL;
}

MethodInfo::MethodInfo(const String& p_name, const PropertyInfo& p_param1,const PropertyInfo& p_param2,const PropertyInfo& p_param3) {

	id=0;
	name=p_name;
	arguments.push_back( p_param1 );
	arguments.push_back( p_param2 );
	arguments.push_back( p_param3 );
	flags=METHOD_FLAG_NORMAL;
}

MethodInfo::MethodInfo(const String& p_name, const PropertyInfo& p_param1,const PropertyInfo& p_param2,const PropertyInfo& p_param3,const PropertyInfo& p_param4) {

	id=0;
	name=p_name;
	arguments.push_back( p_param1 );
	arguments.push_back( p_param2 );
	arguments.push_back( p_param3 );
	arguments.push_back( p_param4 );
	flags=METHOD_FLAG_NORMAL;
}

MethodInfo::MethodInfo(const String& p_name, const PropertyInfo& p_param1,const PropertyInfo& p_param2,const PropertyInfo& p_param3,const PropertyInfo& p_param4,const PropertyInfo& p_param5) {
	id=0;
	name=p_name;
	arguments.push_back( p_param1 );
	arguments.push_back( p_param2 );
	arguments.push_back( p_param3 );
	arguments.push_back( p_param4 );
	arguments.push_back( p_param5 );
	flags=METHOD_FLAG_NORMAL;
}

MethodInfo::MethodInfo(Variant::Type ret) {

	id=0;
	flags=METHOD_FLAG_NORMAL;
	return_val.type=ret;
}

MethodInfo::MethodInfo(Variant::Type ret,const String& p_name) {

	id=0;
	name=p_name;
	flags=METHOD_FLAG_NORMAL;
	return_val.type=ret;
}
MethodInfo::MethodInfo(Variant::Type ret,const String& p_name, const PropertyInfo& p_param1) {

	id=0;
	name=p_name;
	arguments.push_back( p_param1 );
	flags=METHOD_FLAG_NORMAL;
	return_val.type=ret;
}
MethodInfo::MethodInfo(Variant::Type ret,const String& p_name, const PropertyInfo& p_param1,const PropertyInfo& p_param2) {

	id=0;
	name=p_name;
	arguments.push_back( p_param1 );
	arguments.push_back( p_param2 );
	flags=METHOD_FLAG_NORMAL;
	return_val.type=ret;
}

MethodInfo::MethodInfo(Variant::Type ret,const String& p_name, const PropertyInfo& p_param1,const PropertyInfo& p_param2,const PropertyInfo& p_param3) {

	id=0;
	name=p_name;
	arguments.push_back( p_param1 );
	arguments.push_back( p_param2 );
	arguments.push_back( p_param3 );
	flags=METHOD_FLAG_NORMAL;
	return_val.type=ret;
}

MethodInfo::MethodInfo(Variant::Type ret,const String& p_name, const PropertyInfo& p_param1,const PropertyInfo& p_param2,const PropertyInfo& p_param3,const PropertyInfo& p_param4) {

	id=0;
	name=p_name;
	arguments.push_back( p_param1 );
	arguments.push_back( p_param2 );
	arguments.push_back( p_param3 );
	arguments.push_back( p_param4 );
	flags=METHOD_FLAG_NORMAL;
	return_val.type=ret;
}

MethodInfo::MethodInfo(Variant::Type ret,const String& p_name, const PropertyInfo& p_param1,const PropertyInfo& p_param2,const PropertyInfo& p_param3,const PropertyInfo& p_param4,const PropertyInfo& p_param5) {
	id=0;
	name=p_name;
	arguments.push_back( p_param1 );
	arguments.push_back( p_param2 );
	arguments.push_back( p_param3 );
	arguments.push_back( p_param4 );
	arguments.push_back( p_param5 );
	flags=METHOD_FLAG_NORMAL;
	return_val.type=ret;
}

Object::Connection::operator Variant() const {

	Dictionary d;
	d["source"]=source;
	d["signal"]=signal;
	d["target"]=target;
	d["method"]=method;
	d["flags"]=flags;
	d["binds"]=binds;
	return d;
}

bool Object::Connection::operator<(const Connection& p_conn) const {

	if (source==p_conn.source) {

		if (signal == p_conn.signal) {


			if (target == p_conn.target) {

				return method < p_conn.method;
			} else {

				return target < p_conn.target;
			}
		} else
			return signal < p_conn.signal;
	} else {
		return source<p_conn.source;
	}
}
Object::Connection::Connection(const Variant& p_variant) {

	Dictionary d=p_variant;
	if (d.has("source"))
		source=d["source"];
	if (d.has("signal"))
		signal=d["signal"];
	if (d.has("target"))
		target=d["target"];
	if (d.has("method"))
		method=d["method"];
	if (d.has("flags"))
		flags=d["flags"];
	if (d.has("binds"))
		binds=d["binds"];
}


bool Object::_predelete() {
	
	_predelete_ok=1;
	notification(NOTIFICATION_PREDELETE,true);
	return _predelete_ok;

}

void Object::_postinitialize() {
	
	_initialize_typev();
	notification(NOTIFICATION_POSTINITIALIZE);
	
}

void Object::get_valid_parents_static(List<String> *p_parents) {
	
	
}
void Object::_get_valid_parents_static(List<String> *p_parents) {
	
	
}
#if 0
//old style set, deprecated

void Object::set(const String& p_name, const Variant& p_value) {

	_setv(p_name,p_value);
	
	//if (!_use_builtin_script())
//		return;

	bool success;
	ObjectTypeDB::set_property(this,p_name,p_value,success);
	if (success) {
		return;
	}

	if (p_name=="__meta__") {
		metadata=p_value;
	} else if (p_name=="script/script") {
		set_script(p_value);
	} else if (script_instance) {
		script_instance->set(p_name,p_value);
	}
	
	
}
#endif

void Object::set(const StringName& p_name, const Variant& p_value, bool *r_valid) {

#ifdef TOOLS_ENABLED

	_edited=true;
#endif
	if (script_instance) {

		if (script_instance->set(p_name,p_value)) {
			if (r_valid)
				*r_valid=true;
			return;
		}

	}

	//try built-in setgetter
	{
		if (ObjectTypeDB::set_property(this,p_name,p_value)) {
			if (r_valid)
				*r_valid=true;
			return;
		}
	}


	if (p_name==CoreStringNames::get_singleton()->_script) {
		set_script(p_value);
		if (r_valid)
			*r_valid=true;
		return;

	} else if (p_name==CoreStringNames::get_singleton()->_meta) {
		//set_meta(p_name,p_value);
		metadata=p_value;
		if (r_valid)
			*r_valid=true;
		return;
	} else {
		//something inside the object... :|		
		bool success = _setv(p_name,p_value);
		if (success) {
			if (r_valid)
				*r_valid=true;
			return;
		}
		setvar(p_name,p_value,r_valid);
	}

}

Variant Object::get(const StringName& p_name, bool *r_valid) const{


	Variant ret;

	if (script_instance) {

		if (script_instance->get(p_name,ret)) {
			if (r_valid)
				*r_valid=true;
			return ret;
		}

	}


	//try built-in setgetter
	{
		if (ObjectTypeDB::get_property(const_cast<Object*>(this),p_name,ret)) {
			if (r_valid)
				*r_valid=true;
			return ret;
		}
	}


	if (p_name==CoreStringNames::get_singleton()->_script) {
		ret = get_script();
		if (r_valid)
			*r_valid=true;
		return ret;

	} else if (p_name==CoreStringNames::get_singleton()->_meta) {
		ret = metadata;
		if (r_valid)
			*r_valid=true;
		return ret;
	} else {
		//something inside the object... :|
		bool success = _getv(p_name,ret);
		if (success) {
			if (r_valid)
				*r_valid=true;
			return ret;
		}
		//if nothing else, use getvar
		return getvar(p_name,r_valid);
	}


}


#if 0
//old style get, deprecated
Variant Object::get(const String& p_name) const {

	Variant ret=_getv(p_name);
	if (ret.get_type()!=Variant::NIL)
		return ret;
		
	bool success;
	ObjectTypeDB::get_property(const_cast<Object*>(this),p_name,ret,success);
	if (success) {
		return ret;
	}

	if (p_name=="__meta__")
		return metadata;
	else if (p_name=="script/script")
		return script;
		
	if (script_instance) {
		return script_instance->get(p_name);
	}
	
	return Variant();

}
#endif

void Object::get_property_list(List<PropertyInfo> *p_list,bool p_reversed) const {

	if (script_instance && p_reversed) {
		p_list->push_back( PropertyInfo(Variant::NIL,"Script Variables",PROPERTY_HINT_NONE,String(),PROPERTY_USAGE_CATEGORY));
		script_instance->get_property_list(p_list);
	}

	_get_property_listv(p_list,p_reversed);
	
	if (!_use_builtin_script())
		return;
		
	if (!is_type("Script")) // can still be set, but this is for userfriendlyness
		p_list->push_back( PropertyInfo( Variant::OBJECT, "script/script", PROPERTY_HINT_RESOURCE_TYPE, "Script",PROPERTY_USAGE_DEFAULT|PROPERTY_USAGE_STORE_IF_NONZERO));
	if (!metadata.empty())
		p_list->push_back( PropertyInfo( Variant::DICTIONARY, "__meta__", PROPERTY_HINT_NONE,"",PROPERTY_USAGE_NOEDITOR|PROPERTY_USAGE_STORE_IF_NONZERO));
	if (script_instance && !p_reversed) {
		p_list->push_back( PropertyInfo(Variant::NIL,"Script Variables",PROPERTY_HINT_NONE,String(),PROPERTY_USAGE_CATEGORY));
		script_instance->get_property_list(p_list);
	}	
	
}
void Object::get_method_list(List<MethodInfo> *p_list) const {

	ObjectTypeDB::get_method_list(get_type_name(),p_list);
	if (script_instance) {
		script_instance->get_method_list(p_list);
	}	
}



Variant Object::_call_bind(const Variant** p_args, int p_argcount, Variant::CallError& r_error) {

	if (p_argcount<1) {
		r_error.error=Variant::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
		r_error.argument=0;
		return Variant();
	}

	if (p_args[0]->get_type()!=Variant::STRING) {
		r_error.error=Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
		r_error.argument=0;
		r_error.expected=Variant::STRING;
		return Variant();
	}

	StringName method = *p_args[0];

	return call(method,&p_args[1],p_argcount-1,r_error);


}

Variant Object::_call_deferred_bind(const Variant** p_args, int p_argcount, Variant::CallError& r_error) {

	if (p_argcount<1) {
		r_error.error=Variant::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
		r_error.argument=0;
		return Variant();
	}

	if (p_args[0]->get_type()!=Variant::STRING) {
		r_error.error=Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
		r_error.argument=0;
		r_error.expected=Variant::STRING;
		return Variant();
	}

	r_error.error=Variant::CallError::CALL_OK;

	StringName signal = *p_args[0];

	Variant v[VARIANT_ARG_MAX];


	for(int i=0;i<MIN(5,p_argcount-1);i++) {

		v[i]=*p_args[i+1];
	}

	call_deferred(signal,v[0],v[1],v[2],v[3],v[4]);
	return Variant();

}

#if 0
Variant Object::_call_bind(const StringName& p_name, const Variant& p_arg1, const Variant& p_arg2, const Variant& p_arg3, const Variant& p_arg4) {

	ERR_FAIL_COND_V(p_argcount<1,Variant());

	return call(p_name, p_arg1, p_arg2, p_arg3, p_arg4);
};




void Object::_call_deferred_bind(const StringName& p_name, const Variant& p_arg1, const Variant& p_arg2, const Variant& p_arg3, const Variant& p_arg4) {

	call_deferred(p_name, p_arg1, p_arg2, p_arg3, p_arg4);
};
#endif
#ifdef DEBUG_ENABLED
static bool _test_call_error(const StringName& p_func,const Variant::CallError& error) {


	switch(error.error) {

		case Variant::CallError::CALL_OK:
			return true;
		case Variant::CallError::CALL_ERROR_INVALID_METHOD:
			return false;
		case Variant::CallError::CALL_ERROR_INVALID_ARGUMENT: {

			ERR_EXPLAIN("Error Calling Function: "+String(p_func)+" - Invalid type for argument "+itos(error.argument)+", expected "+Variant::get_type_name(error.expected));
			ERR_FAIL_V(true);
		} break;
		case Variant::CallError::CALL_ERROR_TOO_MANY_ARGUMENTS: {

			ERR_EXPLAIN("Error Calling Function: "+String(p_func)+" - Too many arguments, expected "+itos(error.argument));
			ERR_FAIL_V(true);

		} break;
		case Variant::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS: {

			ERR_EXPLAIN("Error Calling Function: "+String(p_func)+" - Too few arguments, expected "+itos(error.argument));
			ERR_FAIL_V(true);

		} break;
		case Variant::CallError::CALL_ERROR_INSTANCE_IS_NULL: {} //?

	}

	return true;
}
#else

#define _test_call_error(m_str,m_err) ((m_err.error==Variant::CallError::CALL_ERROR_INVALID_METHOD)?false:true)

#endif

void Object::call_multilevel(const StringName& p_method,const Variant** p_args,int p_argcount) {


	if (p_method==CoreStringNames::get_singleton()->_free) {
#ifdef DEBUG_ENABLED
		if (cast_to<Reference>()) {
			ERR_EXPLAIN("Can't 'free' a reference.");
			ERR_FAIL();
			return;
		}


		if (_lock_index.get()>1) {
			ERR_EXPLAIN("Object is locked and can't be freed.");
			ERR_FAIL();
			return;
		}
#endif

		//must be here, must be before everything,
		memdelete(this);
		return;
	}

	//Variant ret;
	OBJ_DEBUG_LOCK

	Variant::CallError error;

	if (script_instance) {
		script_instance->call_multilevel(p_method,p_args,p_argcount);
		//_test_call_error(p_method,error);

	}

	MethodBind *method=ObjectTypeDB::get_method(get_type_name(),p_method);

	if (method) {

		method->call(this,p_args,p_argcount,error);
		_test_call_error(p_method,error);
	}

}

void Object::call_multilevel_reversed(const StringName& p_method,const Variant** p_args,int p_argcount) {


	MethodBind *method=ObjectTypeDB::get_method(get_type_name(),p_method);

	Variant::CallError error;
	OBJ_DEBUG_LOCK

	if (method) {

		method->call(this,p_args,p_argcount,error);
		_test_call_error(p_method,error);
	}

	//Variant ret;



	if (script_instance) {
		script_instance->call_multilevel_reversed(p_method,p_args,p_argcount);
		//_test_call_error(p_method,error);

	}

}

bool Object::has_method(const StringName& p_method) const {

	if (p_method==CoreStringNames::get_singleton()->_free) {
		return true;
	}


	if (script_instance && script_instance->has_method(p_method)) {
		return true;
	}

	MethodBind *method=ObjectTypeDB::get_method(get_type_name(),p_method);

	if (method) {
		return true;
	}

	return false;
}


Variant Object::getvar(const Variant& p_key, bool *r_valid) const {

	if (r_valid)
		*r_valid=false;
	return Variant();
}
void Object::setvar(const Variant& p_key, const Variant& p_value,bool *r_valid) {

	if (r_valid)
		*r_valid=false;
}


Variant Object::callv(const StringName& p_method,const Array& p_args) {

	if (p_args.size()==0) {
		return call(p_method);
	}

	Vector<Variant> args;
	args.resize(p_args.size());
	Vector<const Variant*> argptrs;
	argptrs.resize(p_args.size());

	for(int i=0;i<p_args.size();i++) {
		args[i]=p_args[i];
		argptrs[i]=&args[i];
	}

	Variant::CallError ce;
	return call(p_method,argptrs.ptr(),p_args.size(),ce);

}

Variant Object::call(const StringName& p_name, VARIANT_ARG_DECLARE) {
#if 0
	if (p_name==CoreStringNames::get_singleton()->_free) {
#ifdef DEBUG_ENABLED
		if (cast_to<Reference>()) {
			ERR_EXPLAIN("Can't 'free' a reference.");
			ERR_FAIL_V(Variant());
		}
#endif
		//must be here, must be before everything,
		memdelete(this);
		return Variant();
	}

	VARIANT_ARGPTRS;

	int argc=0;
	for(int i=0;i<VARIANT_ARG_MAX;i++) {
		if (argptr[i]->get_type()==Variant::NIL)
			break;
		argc++;
	}

	Variant::CallError error;

	Variant ret;

	if (script_instance) {
		ret = script_instance->call(p_name,argptr,argc,error);
		if (_test_call_error(p_name,error))
			return ret;
	}

	MethodBind *method=ObjectTypeDB::get_method(get_type_name(),p_name);

	if (method) {


		Variant ret = method->call(this,argptr,argc,error);
		if (_test_call_error(p_name,error))
			return ret;

		return ret;
	} else {

	}

	return Variant();
#else

	VARIANT_ARGPTRS;

	int argc=0;
	for(int i=0;i<VARIANT_ARG_MAX;i++) {
		if (argptr[i]->get_type()==Variant::NIL)
			break;
		argc++;
	}

	Variant::CallError error;

	Variant ret = call(p_name,argptr,argc,error);
	return ret;

#endif

}

void Object::call_multilevel(const StringName& p_name, VARIANT_ARG_DECLARE) {
#if 0
	if (p_name==CoreStringNames::get_singleton()->_free) {
#ifdef DEBUG_ENABLED
		if (cast_to<Reference>()) {
			ERR_EXPLAIN("Can't 'free' a reference.");
			ERR_FAIL();
			return;
		}
#endif
		//must be here, must be before everything,
		memdelete(this);
		return;
	}

	VARIANT_ARGPTRS;

	int argc=0;
	for(int i=0;i<VARIANT_ARG_MAX;i++) {
		if (argptr[i]->get_type()==Variant::NIL)
			break;
		argc++;
	}

	Variant::CallError error;

	if (script_instance) {
		script_instance->call(p_name,argptr,argc,error);
		_test_call_error(p_name,error);

	}

	MethodBind *method=ObjectTypeDB::get_method(get_type_name(),p_name);

	if (method) {

		method->call(this,argptr,argc,error);
		_test_call_error(p_name,error);

	}

#else

	VARIANT_ARGPTRS;

	int argc=0;
	for(int i=0;i<VARIANT_ARG_MAX;i++) {
		if (argptr[i]->get_type()==Variant::NIL)
			break;
		argc++;
	}

	//Variant::CallError error;
	call_multilevel(p_name,argptr,argc);

#endif

}



Variant Object::call(const StringName& p_method,const Variant** p_args,int p_argcount,Variant::CallError &r_error) {

	if (p_method==CoreStringNames::get_singleton()->_free) {
		//free must be here, before anything, always ready
#ifdef DEBUG_ENABLED
		if (p_argcount!=0) {
			r_error.argument=0;
			r_error.error=Variant::CallError::CALL_ERROR_TOO_MANY_ARGUMENTS;
			return Variant();
		}
		if (cast_to<Reference>()) {
			r_error.argument=0;
			r_error.error=Variant::CallError::CALL_ERROR_INVALID_METHOD;
			ERR_EXPLAIN("Can't 'free' a reference.");
			ERR_FAIL_V(Variant());
		}

		if (_lock_index.get()>1) {
			r_error.argument=0;
			r_error.error=Variant::CallError::CALL_ERROR_INVALID_METHOD;
			ERR_EXPLAIN("Object is locked and can't be freed.");
			ERR_FAIL_V(Variant());

		}

#endif
		//must be here, must be before everything,
		memdelete(this);
		r_error.error=Variant::CallError::CALL_OK;
		return Variant();
	}

	Variant ret;
	OBJ_DEBUG_LOCK
	if (script_instance) {
		ret = script_instance->call(p_method,p_args,p_argcount,r_error);
		//force jumptable
		switch(r_error.error) {

			case Variant::CallError::CALL_OK:
				return ret;
			case Variant::CallError::CALL_ERROR_INVALID_METHOD:
				break;
			case Variant::CallError::CALL_ERROR_INVALID_ARGUMENT:
			case Variant::CallError::CALL_ERROR_TOO_MANY_ARGUMENTS:
			case Variant::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS:
				return ret;
			case Variant::CallError::CALL_ERROR_INSTANCE_IS_NULL: {}

		}
	}

	MethodBind *method=ObjectTypeDB::get_method(get_type_name(),p_method);

	if (method) {

		ret=method->call(this,p_args,p_argcount,r_error);
	} else {
		r_error.error=Variant::CallError::CALL_ERROR_INVALID_METHOD;
	}

	return ret;
}


void Object::notification(int p_notification,bool p_reversed) {
	

	_notificationv(p_notification,p_reversed);
	
	if (script_instance) {
		script_instance->notification(p_notification);
	}
}

void Object::_changed_callback(Object *p_changed,const char *p_prop) {
	
		
}

void Object::add_change_receptor( Object *p_receptor ) {
	
	change_receptors.insert(p_receptor);
}

void Object::remove_change_receptor( Object *p_receptor ) {

	change_receptors.erase(p_receptor);
}

void Object::property_list_changed_notify() {

	_change_notify();
}

void Object::cancel_delete() {

	_predelete_ok=true;
}

void Object::set_script(const RefPtr& p_script) {

	if (script==p_script)
		return;

	if (script_instance) {
		memdelete(script_instance);
		script_instance=NULL;
	}
	
	script=p_script;	
	Ref<Script> s(script);

	if (!s.is_null() && s->can_instance() ) {
		OBJ_DEBUG_LOCK
		script_instance = s->instance_create(this);

	}

	_change_notify("script/script");
	emit_signal(CoreStringNames::get_singleton()->script_changed);

}

void Object::set_script_instance(ScriptInstance *p_instance) {

	if (script_instance==p_instance)
		return;

	if (script_instance)
		memdelete(script_instance);

	script_instance=p_instance;

	script=p_instance->get_script().get_ref_ptr();
}

RefPtr Object::get_script() const {

	return script;
}

bool Object::has_meta(const String& p_name) const {

	return metadata.has(p_name);
}

void Object::set_meta(const String& p_name, const Variant& p_value ) {

	if (p_value.get_type() == Variant::NIL) {
		metadata.erase(p_name);
		return;
	};

	metadata[p_name]=p_value;
}

Variant Object::get_meta(const String& p_name) const {

	ERR_FAIL_COND_V(!metadata.has(p_name),Variant());
	return metadata[p_name];
}

Array Object::_get_property_list_bind() const {

	List<PropertyInfo> lpi;
	get_property_list(&lpi);
	return convert_property_list(&lpi);
}
DVector<String> Object::_get_meta_list_bind() const {

	DVector<String> _metaret;

	List<Variant> keys;
	metadata.get_key_list(&keys);
	for(List<Variant>::Element *E=keys.front();E;E=E->next()) {

		_metaret.push_back(E->get());
	}

	return _metaret;
}
void Object::get_meta_list(List<String> *p_list) const {

	List<Variant> keys;
	metadata.get_key_list(&keys);
	for(List<Variant>::Element *E=keys.front();E;E=E->next()) {

		p_list->push_back(E->get());
	}
}

void Object::add_user_signal(const MethodInfo& p_signal) {

	ERR_FAIL_COND(p_signal.name=="");
	ERR_FAIL_COND( ObjectTypeDB::has_signal(get_type_name(),p_signal.name ) );
	ERR_FAIL_COND(signal_map.has(p_signal.name));
	Signal s;
	s.user=p_signal;
	signal_map[p_signal.name]=s;
}

struct _ObjectSignalDisconnectData {

	StringName signal;
	Object *target;
	StringName method;

};

#if 0
void Object::_emit_signal(const StringName& p_name,const Array& p_pargs){

	Variant args[VARIANT_ARG_MAX];

	int count = p_pargs.size();

	for(int i=0;i<count;i++) {
		args[i]=p_pargs[i];
	}

	emit_signal(p_name,VARIANT_ARGS_FROM_ARRAY(args));
}

#endif

Variant Object::_emit_signal(const Variant** p_args, int p_argcount, Variant::CallError& r_error) {


	r_error.error=Variant::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;

	ERR_FAIL_COND_V(p_argcount<1,Variant());
	if (p_args[0]->get_type()!=Variant::STRING) {
		r_error.error=Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
		r_error.argument=0;
		r_error.expected=Variant::STRING;
		ERR_FAIL_COND_V(p_args[0]->get_type()!=Variant::STRING,Variant());
	}

	r_error.error=Variant::CallError::CALL_OK;

	StringName signal = *p_args[0];

	Variant v[VARIANT_ARG_MAX];


	for(int i=0;i<MIN(5,p_argcount-1);i++) {

		v[i]=*p_args[i+1];
	}

	emit_signal(signal,v[0],v[1],v[2],v[3],v[4]);
	return Variant();
}



void Object::emit_signal(const StringName& p_name,VARIANT_ARG_DECLARE) {

	if (_block_signals)
		return; //no emit, signals blocked

	Signal *s = signal_map.getptr(p_name);
	if (!s) {
		return;
	}


	List<_ObjectSignalDisconnectData> disconnect_data;


	//copy on write will ensure that disconnecting the signal or even deleting the object will not affect the signal calling.
	//this happens automatically and will not change the performance of calling.
	//awesome, isn't it?
	VMap<Signal::Target,Signal::Slot> slot_map = s->slot_map;

	int ssize = slot_map.size();

	OBJ_DEBUG_LOCK

	for(int i=0;i<ssize;i++) {

		const Connection &c = slot_map.getv(i).conn;
		VARIANT_ARGPTRS

		Object *target;
#ifdef DEBUG_ENABLED
		target = ObjectDB::get_instance(slot_map.getk(i)._id);
		ERR_CONTINUE(!target);
#else
		target=c.target;
#endif


		int bind_count=c.binds.size();
		int bind=0;

		for(int i=0;bind < bind_count && i<VARIANT_ARG_MAX;i++) {

			if (argptr[i]->get_type()==Variant::NIL) {
				argptr[i]=&c.binds[bind];
				bind++;
			}
		}

		if (c.flags&CONNECT_DEFERRED) {
			MessageQueue::get_singleton()->push_call(target->get_instance_ID(),c.method,VARIANT_ARGPTRS_PASS);
		} else {
			target->call( c.method, VARIANT_ARGPTRS_PASS );
		}

		if (c.flags&CONNECT_ONESHOT) {
			_ObjectSignalDisconnectData dd;
			dd.signal=p_name;
			dd.target=target;
			dd.method=c.method;
			disconnect_data.push_back(dd);
		}

	}

#if 0

	//old (deprecated and dangerous code)
	s->lock++;
	for( Map<Signal::Target,Signal::Slot>::Element *E = s->slot_map.front();E;E=E->next() ) {

		const Signal::Target& t = E->key();
		const Signal::Slot& s = E->get();
		const Connection &c = s.cE->get();

		VARIANT_ARGPTRS

		int bind_count=c.binds.size();
		int bind=0;

		for(int i=0;bind < bind_count && i<VARIANT_ARG_MAX;i++) {

			if (argptr[i]->get_type()==Variant::NIL) {
				argptr[i]=&c.binds[bind];
				bind++;
			}
		}

		if (c.flags&CONNECT_DEFERRED) {
			MessageQueue::get_singleton()->push_call(t._id,t.method,VARIANT_ARGPTRS_PASS);
		} else {
			Object *obj = ObjectDB::get_instance(t._id);
			ERR_CONTINUE(!obj); //yeah this should always be here
			obj->call( t.method, VARIANT_ARGPTRS_PASS );
		}

		if (c.flags&CONNECT_ONESHOT) {
			_ObjectSignalDisconnectData dd;
			dd.signal=p_name;
			dd.target=ObjectDB::get_instance(t._id);
			dd.method=t.method;
			disconnect_data.push_back(dd);
		}

	}



	s->lock--;
#endif
	while (!disconnect_data.empty()) {

		const _ObjectSignalDisconnectData &dd = disconnect_data.front()->get();
		disconnect(dd.signal,dd.target,dd.method);
		disconnect_data.pop_front();
	}

}


void Object::_add_user_signal(const String& p_name, const Array& p_args) {

	// this version of add_user_signal is meant to be used from scripts or external apis
	// without access to ADD_SIGNAL in bind_methods
	// added events are per instance, as opposed to the other ones, which are global



	MethodInfo mi;
	mi.name=p_name;

	for(int i=0;i<p_args.size();i++) {

		Dictionary d=p_args[i];
		PropertyInfo param;

		if (d.has("name"))
			param.name=d["name"];
		if (d.has("type"))
			param.type=(Variant::Type)(int)d["type"];

		mi.arguments.push_back(param);
	}

	add_user_signal(mi);

}
#if 0
void Object::_emit_signal(const StringName& p_name,const Array& p_pargs){

	Variant args[VARIANT_ARG_MAX];

	int count = p_pargs.size();

	for(int i=0;i<count;i++) {
		args[i]=p_pargs[i];
	}

	emit_signal(p_name,VARIANT_ARGS_FROM_ARRAY(args));
}

#endif
Array Object::_get_signal_list() const{

	return Array();
}
Array Object::_get_signal_connection_list(const String& p_signal) const{

	return Array();
}


void Object::get_signal_list(List<MethodInfo> *p_signals ) const {

	ObjectTypeDB::get_signal_list(get_type_name(),p_signals);
	//find maybe usersignals?
	const StringName *S=NULL;

	while((S=signal_map.next(S))) {

		if (signal_map[*S].user.name!="") {
			//user signal
			p_signals->push_back(signal_map[*S].user);
		}
	}
}

void Object::get_signal_connection_list(const StringName& p_signal,List<Connection> *p_connections) const {

	const Signal *s=signal_map.getptr(p_signal);
	if (!s)
		return; //nothing

	for(int i=0;i<s->slot_map.size();i++)
		p_connections->push_back(s->slot_map.getv(i).conn);

}


Error Object::connect(const StringName& p_signal, Object *p_to_object, const StringName& p_to_method,const Vector<Variant>& p_binds,uint32_t p_flags) {

	ERR_FAIL_NULL_V(p_to_object,ERR_INVALID_PARAMETER);

	Signal *s = signal_map.getptr(p_signal);
	if (!s) {
		bool signal_is_valid = ObjectTypeDB::has_signal(get_type_name(),p_signal);
		if (!signal_is_valid) {
			ERR_EXPLAIN("Attempt to connect to unexisting signal: "+p_signal);
			ERR_FAIL_COND_V(!signal_is_valid,ERR_INVALID_PARAMETER);
		}
		signal_map[p_signal]=Signal();
		s=&signal_map[p_signal];
	}

	Signal::Target target(p_to_object->get_instance_ID(),p_to_method);
	if (s->slot_map.has(target)) {
		ERR_EXPLAIN("Signal '"+p_signal+"'' already connected to given method '"+p_to_method+"' in that object.");
		ERR_FAIL_COND_V(s->slot_map.has(target),ERR_INVALID_PARAMETER);
	}

	Signal::Slot slot;

	Connection conn;
	conn.source=this;
	conn.target=p_to_object;
	conn.method=p_to_method;
	conn.signal=p_signal;
	conn.flags=p_flags;
	conn.binds=p_binds;
	slot.conn=conn;
	slot.cE=p_to_object->connections.push_back(conn);
	s->slot_map[target]=slot;

	return OK;
}

bool Object::is_connected(const StringName& p_signal, Object *p_to_object, const StringName& p_to_method) const {

	ERR_FAIL_NULL_V(p_to_object,false);
	const Signal *s = signal_map.getptr(p_signal);
	if (!s) {
		bool signal_is_valid = ObjectTypeDB::has_signal(get_type_name(),p_signal);
		if (signal_is_valid)
			return false;
		ERR_EXPLAIN("Unexisting signal: "+p_signal);
		ERR_FAIL_COND_V(!s,false);
	}

	Signal::Target target(p_to_object->get_instance_ID(),p_to_method);

	return s->slot_map.has(target);
	//const Map<Signal::Target,Signal::Slot>::Element *E = s->slot_map.find(target);
	//return (E!=NULL);

}

void Object::disconnect(const StringName& p_signal, Object *p_to_object, const StringName& p_to_method) {

	ERR_FAIL_NULL(p_to_object);
	Signal *s = signal_map.getptr(p_signal);
	if (!s) {
		ERR_EXPLAIN("Unexisting signal: "+p_signal);
		ERR_FAIL_COND(!s);
	}
	if (s->lock>0) {
		ERR_EXPLAIN("Attempt to disconnect signal '"+p_signal+"' while emitting (locks: "+itos(s->lock)+")");
		ERR_FAIL_COND(s->lock>0);
	}

	Signal::Target target(p_to_object->get_instance_ID(),p_to_method);

	if (!s->slot_map.has(target)) {
		ERR_EXPLAIN("Disconnecting unexisting signal '"+p_signal+"', slot: "+itos(target._id)+":"+target.method);
		ERR_FAIL();
	}
	int prev = p_to_object->connections.size();
	p_to_object->connections.erase(s->slot_map[target].cE);
	s->slot_map.erase(target);

	if (s->slot_map.empty() && ObjectTypeDB::has_signal(get_type_name(),p_signal )) {
		//not user signal, delete
		signal_map.erase(p_signal);
	}
}


void Object::_set_bind(const String& p_set,const Variant& p_value) {

	set(p_set,p_value);
}

Variant Object::_get_bind(const String& p_name) const {

	return get(p_name);
}

void Object::initialize_type() {

	static bool initialized=false;
	if (initialized)
		return;
	ObjectTypeDB::_add_type<Object>();
	_bind_methods();
	initialized=true;
}

StringName Object::XL_MESSAGE(const StringName& p_message) const {

	if (!_can_translate || !TranslationServer::get_singleton())
		return p_message;

	return TranslationServer::get_singleton()->translate(p_message);
}

StringName Object::tr(const StringName& p_message) const {

	return XL_MESSAGE(p_message);

}

void Object::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("get_type"),&Object::get_type);
	ObjectTypeDB::bind_method(_MD("is_type","type"),&Object::is_type);
	ObjectTypeDB::bind_method(_MD("set","property","value"),&Object::_set_bind);
	ObjectTypeDB::bind_method(_MD("get","property"),&Object::_get_bind);
	ObjectTypeDB::bind_method(_MD("get_property_list"),&Object::_get_property_list_bind);
	ObjectTypeDB::bind_method(_MD("notification","what"),&Object::notification,DEFVAL(false));
	ObjectTypeDB::bind_method(_MD("get_instance_ID"),&Object::get_instance_ID);

	ObjectTypeDB::bind_method(_MD("set_script","script:Script"),&Object::set_script);
	ObjectTypeDB::bind_method(_MD("get_script:Script"),&Object::get_script);

	ObjectTypeDB::bind_method(_MD("set_meta","name","value"),&Object::set_meta);
	ObjectTypeDB::bind_method(_MD("get_meta","name","value"),&Object::get_meta);
	ObjectTypeDB::bind_method(_MD("has_meta","name"),&Object::has_meta);
	ObjectTypeDB::bind_method(_MD("get_meta_list"),&Object::_get_meta_list_bind);

	//todo reimplement this per language so all 5 arguments can be called

//	ObjectTypeDB::bind_method(_MD("call","method","arg1","arg2","arg3","arg4"),&Object::_call_bind,DEFVAL(Variant()),DEFVAL(Variant()),DEFVAL(Variant()),DEFVAL(Variant()));
//	ObjectTypeDB::bind_method(_MD("call_deferred","method","arg1","arg2","arg3","arg4"),&Object::_call_deferred_bind,DEFVAL(Variant()),DEFVAL(Variant()),DEFVAL(Variant()),DEFVAL(Variant()));

	ObjectTypeDB::bind_method(_MD("add_user_signal","signal","arguments"),&Object::_add_user_signal,DEFVAL(Array()));
//	ObjectTypeDB::bind_method(_MD("emit_signal","signal","arguments"),&Object::_emit_signal,DEFVAL(Array()));


	{
		MethodInfo mi;
		mi.name="emit_signal";
		mi.arguments.push_back( PropertyInfo( Variant::STRING, "signal"));
		Vector<Variant> defargs;
		for(int i=0;i<VARIANT_ARG_MAX;i++) {
			mi.arguments.push_back( PropertyInfo( Variant::NIL, "arg"+itos(i)));
			defargs.push_back(Variant());
		}


		ObjectTypeDB::bind_native_method(METHOD_FLAGS_DEFAULT,"emit_signal",&Object::_emit_signal,mi,defargs);
	}

	{
		MethodInfo mi;
		mi.name="call";
		mi.arguments.push_back( PropertyInfo( Variant::STRING, "method"));
		Vector<Variant> defargs;
		for(int i=0;i<10;i++) {
			mi.arguments.push_back( PropertyInfo( Variant::NIL, "arg"+itos(i)));
			defargs.push_back(Variant());
		}


		ObjectTypeDB::bind_native_method(METHOD_FLAGS_DEFAULT,"call",&Object::_call_bind,mi,defargs);
	}

	{
		MethodInfo mi;
		mi.name="call_deferred";
		mi.arguments.push_back( PropertyInfo( Variant::STRING, "method"));
		Vector<Variant> defargs;
		for(int i=0;i<VARIANT_ARG_MAX;i++) {
			mi.arguments.push_back( PropertyInfo( Variant::NIL, "arg"+itos(i)));
			defargs.push_back(Variant());
		}


		ObjectTypeDB::bind_native_method(METHOD_FLAGS_DEFAULT,"call_deferred",&Object::_call_deferred_bind,mi,defargs);
	}

	ObjectTypeDB::bind_method(_MD("callv:var","method","arg_array"),&Object::callv);

	ObjectTypeDB::bind_method(_MD("has_method"),&Object::has_method);

	ObjectTypeDB::bind_method(_MD("get_signal_list"),&Object::_get_signal_list);

	ObjectTypeDB::bind_method(_MD("connect","signal","target:Object","method","binds","flags"),&Object::connect,DEFVAL(Array()),DEFVAL(0));
	ObjectTypeDB::bind_method(_MD("disconnect","signal","target:Object","method"),&Object::disconnect);
	ObjectTypeDB::bind_method(_MD("is_connected","signal","target:Object","method"),&Object::is_connected);

	ObjectTypeDB::bind_method(_MD("set_block_signals","enable"),&Object::set_block_signals);
	ObjectTypeDB::bind_method(_MD("is_blocking_signals"),&Object::is_blocking_signals);
	ObjectTypeDB::bind_method(_MD("set_message_translation","enable"),&Object::set_message_translation);
	ObjectTypeDB::bind_method(_MD("can_translate_messages"),&Object::can_translate_messages);
	ObjectTypeDB::bind_method(_MD("property_list_changed_notify"),&Object::property_list_changed_notify);

	ObjectTypeDB::bind_method(_MD("XL_MESSAGE","message"),&Object::XL_MESSAGE);
	ObjectTypeDB::bind_method(_MD("tr","message"),&Object::tr);

	ADD_SIGNAL( MethodInfo("script_changed"));

	BIND_VMETHOD( MethodInfo("_notification",PropertyInfo(Variant::INT,"what")) );
	BIND_VMETHOD( MethodInfo("_set",PropertyInfo(Variant::STRING,"property"),PropertyInfo(Variant::NIL,"value")) );
#ifdef TOOLS_ENABLED
	MethodInfo miget("_get",PropertyInfo(Variant::STRING,"property") );
	miget.return_val.name="var";
	BIND_VMETHOD( miget );

	MethodInfo plget("_get_property_list");

	plget.return_val.type=Variant::ARRAY;
	BIND_VMETHOD( plget );

#endif
	BIND_VMETHOD( MethodInfo("_init") );



	BIND_CONSTANT( NOTIFICATION_POSTINITIALIZE );
	BIND_CONSTANT( NOTIFICATION_PREDELETE );

	BIND_CONSTANT( CONNECT_DEFERRED );
	BIND_CONSTANT( CONNECT_PERSIST );
	BIND_CONSTANT( CONNECT_ONESHOT );

}

void Object::call_deferred(const StringName& p_method,VARIANT_ARG_DECLARE) {

	MessageQueue::get_singleton()->push_call(this,p_method,VARIANT_ARG_PASS);
}

void Object::set_block_signals(bool p_block) {

	_block_signals=p_block;
}

bool Object::is_blocking_signals() const{

	return _block_signals;
}

void Object::get_translatable_strings(List<String> *p_strings) const {

	List<PropertyInfo> plist;
	get_property_list(&plist);

	for(List<PropertyInfo>::Element *E=plist.front();E;E=E->next()) {

		if (!(E->get().usage&PROPERTY_USAGE_INTERNATIONALIZED))
			continue;

		String text = get( E->get().name );

		if (text=="")
			continue;

		p_strings->push_back(text);
	}

}

#ifdef TOOLS_ENABLED
void Object::set_edited(bool p_edited) {

	_edited=p_edited;
}

bool Object::is_edited() const {

	return _edited;

}
#endif

Object::Object() {
	

	_block_signals=false;
	_predelete_ok=0;
	_instance_ID=0;
	_instance_ID = ObjectDB::add_instance(this);
	_can_translate=true;
	script_instance=NULL;
#ifdef TOOLS_ENABLED

	_edited=false;
#endif

#ifdef DEBUG_ENABLED
	_lock_index.init(1);
#endif



}


Object::~Object() {



	if (script_instance)
		memdelete(script_instance);
	script_instance=NULL;


	List<Connection> sconnections;
	const StringName *S=NULL;

	while((S=signal_map.next(S))) {

		Signal *s=&signal_map[*S];

		ERR_EXPLAIN("Attempt to delete an object in the middle of a signal emission from it");
		ERR_CONTINUE(s->lock>0);

		for(int i=0;i<s->slot_map.size();i++) {

			sconnections.push_back(s->slot_map.getv(i).conn);
		}
	}

	for(List<Connection>::Element *E=sconnections.front();E;E=E->next()) {

		Connection &c = E->get();
		ERR_CONTINUE(c.source!=this); //bug?

		this->disconnect(c.signal,c.target,c.method);
	}

	while(connections.size()) {

		Connection c = connections.front()->get();
		c.source->disconnect(c.signal,c.target,c.method);
	}

	ObjectDB::remove_instance(this);
	_instance_ID=0;
	_predelete_ok=2;

}



bool predelete_handler(Object *p_object) {
	
	return p_object->_predelete();
}

void postinitialize_handler(Object *p_object) {
	
	p_object->_postinitialize();
}

HashMap<uint32_t,Object*> ObjectDB::instances;
uint32_t ObjectDB::instance_counter=1;
HashMap<Object*,ObjectID,ObjectDB::ObjectPtrHash> ObjectDB::instance_checks;
uint32_t ObjectDB::add_instance(Object *p_object) {

	GLOBAL_LOCK_FUNCTION;
	ERR_FAIL_COND_V( p_object->get_instance_ID()!=0, 0 );
	instances[++instance_counter]=p_object;
#ifdef DEBUG_ENABLED
	instance_checks[p_object]=instance_counter;
#endif
	return instance_counter;
}

void ObjectDB::remove_instance(Object *p_object) {

	GLOBAL_LOCK_FUNCTION;
	instances.erase( p_object->get_instance_ID() );
#ifdef DEBUG_ENABLED
	instance_checks.erase(p_object);
#endif
}
Object *ObjectDB::get_instance(uint32_t p_instance_ID) {

	GLOBAL_LOCK_FUNCTION;
	Object**obj=instances.getptr(p_instance_ID);
	if (!obj)
		return NULL;
	return *obj;
}

void ObjectDB::debug_objects(DebugFunc p_func) {

	GLOBAL_LOCK_FUNCTION;

	const uint32_t *K=NULL;
	while((K=instances.next(K))) {

		p_func(instances[*K]);
	}
}


void Object::get_argument_options(const StringName& p_function,int p_idx,List<String>*r_options) const {


}

int ObjectDB::get_object_count() {

	GLOBAL_LOCK_FUNCTION;
	return instances.size();

}

void ObjectDB::cleanup() {


	GLOBAL_LOCK_FUNCTION;
	if (instances.size()) {
	
		WARN_PRINT("ObjectDB Instances still exist!");		
	}
	instances.clear();
	instance_checks.clear();
}
