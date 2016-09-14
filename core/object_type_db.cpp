/*************************************************************************/
/*  object_type_db.cpp                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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
#include "object_type_db.h"
#include "os/mutex.h"

#ifdef NO_THREADS

#define OBJTYPE_LOCK

#else

#define OBJTYPE_LOCK MutexLock _mutex_lock_(lock);

#endif

#ifdef DEBUG_METHODS_ENABLED

ParamDef::ParamDef(const Variant& p_variant) { used=true; val=p_variant; }


MethodDefinition _MD(const char* p_name) {

	MethodDefinition md;
	md.name=StaticCString::create(p_name);
	return md;
}

MethodDefinition _MD(const char* p_name,const char *p_arg1) {

	MethodDefinition md;
	md.name=StaticCString::create(p_name);
	md.args.push_back(StaticCString::create(p_arg1));
	return md;
}

MethodDefinition _MD(const char* p_name,const char *p_arg1,const char *p_arg2) {

	MethodDefinition md;
	md.name=StaticCString::create(p_name);
	md.args.resize(2);
	md.args[0]=StaticCString::create(p_arg1);
	md.args[1]=StaticCString::create(p_arg2);
	return md;
}

MethodDefinition _MD(const char* p_name,const char *p_arg1,const char *p_arg2,const char *p_arg3) {

	MethodDefinition md;
	md.name=StaticCString::create(p_name);
	md.args.resize(3);
	md.args[0]=StaticCString::create(p_arg1);
	md.args[1]=StaticCString::create(p_arg2);
	md.args[2]=StaticCString::create(p_arg3);
	return md;
}

MethodDefinition _MD(const char* p_name,const char *p_arg1,const char *p_arg2,const char *p_arg3,const char *p_arg4) {

	MethodDefinition md;
	md.name=StaticCString::create(p_name);
	md.args.resize(4);
	md.args[0]=StaticCString::create(p_arg1);
	md.args[1]=StaticCString::create(p_arg2);
	md.args[2]=StaticCString::create(p_arg3);
	md.args[3]=StaticCString::create(p_arg4);
	return md;
}

MethodDefinition _MD(const char* p_name,const char *p_arg1,const char *p_arg2,const char *p_arg3,const char *p_arg4,const char *p_arg5) {

	MethodDefinition md;
	md.name=StaticCString::create(p_name);
	md.args.resize(5);
	md.args[0]=StaticCString::create(p_arg1);
	md.args[1]=StaticCString::create(p_arg2);
	md.args[2]=StaticCString::create(p_arg3);
	md.args[3]=StaticCString::create(p_arg4);
	md.args[4]=StaticCString::create(p_arg5);
	return md;
}


MethodDefinition _MD(const char* p_name,const char *p_arg1,const char *p_arg2,const char *p_arg3,const char *p_arg4,const char *p_arg5,const char *p_arg6) {

	MethodDefinition md;
	md.name=StaticCString::create(p_name);
	md.args.resize(6);
	md.args[0]=StaticCString::create(p_arg1);
	md.args[1]=StaticCString::create(p_arg2);
	md.args[2]=StaticCString::create(p_arg3);
	md.args[3]=StaticCString::create(p_arg4);
	md.args[4]=StaticCString::create(p_arg5);
	md.args[5]=StaticCString::create(p_arg6);
	return md;
}

MethodDefinition _MD(const char* p_name,const char *p_arg1,const char *p_arg2,const char *p_arg3,const char *p_arg4,const char *p_arg5,const char *p_arg6,const char *p_arg7) {

	MethodDefinition md;
	md.name=StaticCString::create(p_name);
	md.args.resize(7);
	md.args[0]=StaticCString::create(p_arg1);
	md.args[1]=StaticCString::create(p_arg2);
	md.args[2]=StaticCString::create(p_arg3);
	md.args[3]=StaticCString::create(p_arg4);
	md.args[4]=StaticCString::create(p_arg5);
	md.args[5]=StaticCString::create(p_arg6);
	md.args[6]=StaticCString::create(p_arg7);
	return md;
}

MethodDefinition _MD(const char* p_name,const char *p_arg1,const char *p_arg2,const char *p_arg3,const char *p_arg4,const char *p_arg5,const char *p_arg6,const char *p_arg7,const char *p_arg8) {

	MethodDefinition md;
	md.name=StaticCString::create(p_name);
	md.args.resize(8);
	md.args[0]=StaticCString::create(p_arg1);
	md.args[1]=StaticCString::create(p_arg2);
	md.args[2]=StaticCString::create(p_arg3);
	md.args[3]=StaticCString::create(p_arg4);
	md.args[4]=StaticCString::create(p_arg5);
	md.args[5]=StaticCString::create(p_arg6);
	md.args[6]=StaticCString::create(p_arg7);
	md.args[7]=StaticCString::create(p_arg8);
	return md;
}

MethodDefinition _MD(const char* p_name,const char *p_arg1,const char *p_arg2,const char *p_arg3,const char *p_arg4,const char *p_arg5,const char *p_arg6,const char *p_arg7,const char *p_arg8,const char *p_arg9) {

	MethodDefinition md;
	md.name=StaticCString::create(p_name);
	md.args.resize(9);
	md.args[0]=StaticCString::create(p_arg1);
	md.args[1]=StaticCString::create(p_arg2);
	md.args[2]=StaticCString::create(p_arg3);
	md.args[3]=StaticCString::create(p_arg4);
	md.args[4]=StaticCString::create(p_arg5);
	md.args[5]=StaticCString::create(p_arg6);
	md.args[6]=StaticCString::create(p_arg7);
	md.args[7]=StaticCString::create(p_arg8);
	md.args[8]=StaticCString::create(p_arg9);
	return md;
}

MethodDefinition _MD(const char* p_name,const char *p_arg1,const char *p_arg2,const char *p_arg3,const char *p_arg4,const char *p_arg5,const char *p_arg6,const char *p_arg7,const char *p_arg8,const char *p_arg9,const char *p_arg10) {

	MethodDefinition md;
	md.name=StaticCString::create(p_name);
	md.args.resize(10);
	md.args[0]=StaticCString::create(p_arg1);
	md.args[1]=StaticCString::create(p_arg2);
	md.args[2]=StaticCString::create(p_arg3);
	md.args[3]=StaticCString::create(p_arg4);
	md.args[4]=StaticCString::create(p_arg5);
	md.args[5]=StaticCString::create(p_arg6);
	md.args[6]=StaticCString::create(p_arg7);
	md.args[7]=StaticCString::create(p_arg8);
	md.args[8]=StaticCString::create(p_arg9);
	md.args[9]=StaticCString::create(p_arg10);
	return md;
}


#endif


ObjectTypeDB::APIType ObjectTypeDB::current_api=API_CORE;

void ObjectTypeDB::set_current_api(APIType p_api) {

	current_api=p_api;
}

HashMap<StringName,ObjectTypeDB::TypeInfo,StringNameHasher> ObjectTypeDB::types;
HashMap<StringName,StringName,StringNameHasher> ObjectTypeDB::resource_base_extensions;
HashMap<StringName,StringName,StringNameHasher> ObjectTypeDB::compat_types;

ObjectTypeDB::TypeInfo::TypeInfo() {

	creation_func=NULL;
	inherits_ptr=NULL;
	disabled=false;
}
ObjectTypeDB::TypeInfo::~TypeInfo() {


}


bool ObjectTypeDB::is_type(const StringName &p_type,const StringName& p_inherits) {

	OBJTYPE_LOCK;

	StringName inherits=p_type;

	while (inherits.operator String().length()) {

		if (inherits==p_inherits)
			return true;
		inherits=type_inherits_from(inherits);
	}

	return false;
}
void ObjectTypeDB::get_type_list( List<StringName> *p_types) {

	OBJTYPE_LOCK;

	const StringName *k=NULL;

	while((k=types.next(k))) {

		p_types->push_back(*k);
	}

	p_types->sort();
}


void ObjectTypeDB::get_inheriters_from( const StringName& p_type,List<StringName> *p_types) {

	OBJTYPE_LOCK;

	const StringName *k=NULL;

	while((k=types.next(k))) {

		if (*k!=p_type && is_type(*k,p_type))
			p_types->push_back(*k);
	}

}

StringName ObjectTypeDB::type_inherits_from(const StringName& p_type) {

	OBJTYPE_LOCK;

	TypeInfo *ti = types.getptr(p_type);
	ERR_FAIL_COND_V(!ti,"");
	return ti->inherits;
}

ObjectTypeDB::APIType ObjectTypeDB::get_api_type(const StringName &p_type) {

	OBJTYPE_LOCK;

	TypeInfo *ti = types.getptr(p_type);
	ERR_FAIL_COND_V(!ti,API_NONE);
	return ti->api;
}

uint64_t ObjectTypeDB::get_api_hash(APIType p_api) {

#ifdef DEBUG_METHODS_ENABLED

	uint64_t hash = hash_djb2_one_64(HashMapHahserDefault::hash(VERSION_FULL_NAME));

	List<StringName> names;

	const StringName *k=NULL;

	while((k=types.next(k))) {

		names.push_back(*k);
	}
	//must be alphabetically sorted for hash to compute
	names.sort_custom<StringName::AlphCompare>();

	for (List<StringName>::Element *E=names.front();E;E=E->next()) {

		TypeInfo *t = types.getptr(E->get());
		ERR_FAIL_COND_V(!t,0);
		if (t->api!=p_api)
			continue;
		hash = hash_djb2_one_64(t->name.hash(),hash);
		hash = hash_djb2_one_64(t->inherits.hash(),hash);

		{ //methods

			List<StringName> snames;

			k=NULL;

			while((k=t->method_map.next(k))) {

				snames.push_back(*k);
			}

			snames.sort_custom<StringName::AlphCompare>();

			for (List<StringName>::Element *F=snames.front();F;F=F->next()) {

				MethodBind *mb = t->method_map[F->get()];
				hash = hash_djb2_one_64( mb->get_name().hash(), hash);
				hash = hash_djb2_one_64( mb->get_argument_count(), hash);
				hash = hash_djb2_one_64( mb->get_argument_type(-1), hash); //return

				for(int i=0;i<mb->get_argument_count();i++) {
					hash = hash_djb2_one_64( mb->get_argument_info(i).type, hash );
					hash = hash_djb2_one_64( mb->get_argument_info(i).name.hash(), hash );
					hash = hash_djb2_one_64( mb->get_argument_info(i).hint, hash );
					hash = hash_djb2_one_64( mb->get_argument_info(i).hint_string.hash(), hash );
				}

				hash = hash_djb2_one_64( mb->get_default_argument_count(), hash);

				for(int i=0;i<mb->get_default_argument_count();i++) {
					//hash should not change, i hope for tis
					Variant da = mb->get_default_argument(i);
					hash = hash_djb2_one_64( da.hash(), hash );
				}

				hash = hash_djb2_one_64( mb->get_hint_flags(), hash);

			}
		}


		{ //constants

			List<StringName> snames;

			k=NULL;

			while((k=t->constant_map.next(k))) {

				snames.push_back(*k);
			}

			snames.sort_custom<StringName::AlphCompare>();

			for (List<StringName>::Element *F=snames.front();F;F=F->next()) {

				hash = hash_djb2_one_64(F->get().hash(), hash);
				hash = hash_djb2_one_64( t->constant_map[F->get()], hash);
			}
		}


		{ //signals

			List<StringName> snames;

			k=NULL;

			while((k=t->signal_map.next(k))) {

				snames.push_back(*k);
			}

			snames.sort_custom<StringName::AlphCompare>();

			for (List<StringName>::Element *F=snames.front();F;F=F->next()) {

				MethodInfo &mi = t->signal_map[F->get()];
				hash = hash_djb2_one_64( F->get().hash(), hash);
				for(int i=0;i<mi.arguments.size();i++) {
					hash = hash_djb2_one_64( mi.arguments[i].type, hash);
				}
			}
		}

		{ //properties

			List<StringName> snames;

			k=NULL;

			while((k=t->property_setget.next(k))) {

				snames.push_back(*k);
			}

			snames.sort_custom<StringName::AlphCompare>();

			for (List<StringName>::Element *F=snames.front();F;F=F->next()) {

				PropertySetGet *psg=t->property_setget.getptr(F->get());

				hash = hash_djb2_one_64( F->get().hash(), hash);
				hash = hash_djb2_one_64( psg->setter.hash(), hash);
				hash = hash_djb2_one_64( psg->getter.hash(), hash);

			}
		}

		//property list
		for (List<PropertyInfo>::Element *F=t->property_list.front();F;F=F->next()) {

			hash = hash_djb2_one_64( F->get().name.hash(), hash);
			hash = hash_djb2_one_64( F->get().type, hash);
			hash = hash_djb2_one_64( F->get().hint, hash);
			hash = hash_djb2_one_64( F->get().hint_string.hash(), hash);
			hash = hash_djb2_one_64( F->get().usage, hash);
		}


	}


	return hash;
#else
	return 0;
#endif

}

bool ObjectTypeDB::type_exists(const StringName &p_type) {

	OBJTYPE_LOCK;
	return types.has(p_type);
}

void ObjectTypeDB::add_compatibility_type(const StringName& p_type,const StringName& p_fallback) {

	compat_types[p_type]=p_fallback;
}

Object *ObjectTypeDB::instance(const StringName &p_type) {

	TypeInfo *ti;
	{
		OBJTYPE_LOCK;
		ti=types.getptr(p_type);
		if (!ti || ti->disabled || !ti->creation_func) {
			if (compat_types.has(p_type)) {
				ti=types.getptr(compat_types[p_type]);
			}
		}
		ERR_FAIL_COND_V(!ti,NULL);
		ERR_FAIL_COND_V(ti->disabled,NULL);
		ERR_FAIL_COND_V(!ti->creation_func,NULL);
	}

	return ti->creation_func();
}
bool ObjectTypeDB::can_instance(const StringName &p_type) {

	OBJTYPE_LOCK;

	TypeInfo *ti = types.getptr(p_type);
	ERR_FAIL_COND_V(!ti,false);
	return (!ti->disabled && ti->creation_func!=NULL);
}


void ObjectTypeDB::_add_type2(const StringName& p_type, const StringName& p_inherits) {

	OBJTYPE_LOCK;

	StringName name = p_type;

	ERR_FAIL_COND(types.has(name));

	types[name]=TypeInfo();
	TypeInfo &ti=types[name];
	ti.name=name;
	ti.inherits=p_inherits;
	ti.api=current_api;

	if (ti.inherits) {

		ERR_FAIL_COND( !types.has(ti.inherits) ); //it MUST be registered.
		ti.inherits_ptr = &types[ti.inherits];

	} else {
		ti.inherits_ptr=NULL;
	}


}

void ObjectTypeDB::get_method_list(StringName p_type,List<MethodInfo> *p_methods,bool p_no_inheritance) {


	OBJTYPE_LOCK;

	TypeInfo *type=types.getptr(p_type);

	while(type) {

		if (type->disabled) {

			if (p_no_inheritance)
				break;

			type=type->inherits_ptr;
			continue;
		}

#ifdef DEBUG_METHODS_ENABLED

		for( List<MethodInfo>::Element *E=type->virtual_methods.front();E;E=E->next()) {

			p_methods->push_back(E->get());
		}

		for( List<StringName>::Element *E=type->method_order.front();E;E=E->next()) {

			MethodBind *method=type->method_map.get(E->get());
			MethodInfo minfo;
			minfo.name=E->get();
			minfo.id=method->get_method_id();

			for (int i=0;i<method->get_argument_count();i++) {

				//Variant::Type t=method->get_argument_type(i);

				minfo.arguments.push_back(method->get_argument_info(i));
			}


			if (method->get_argument_type(-1)!=Variant::NIL) {
				minfo.return_val=method->get_argument_info(-1);
			}

			minfo.flags=method->get_hint_flags();
			p_methods->push_back(minfo);
		}



#else

		const StringName *K=NULL;

		while((K=type->method_map.next(K))) {

			MethodBind*m = type->method_map[*K];
			MethodInfo mi;
			mi.name=m->get_name();
			p_methods->push_back(mi);
		}


#endif

		if (p_no_inheritance)
			break;

		type=type->inherits_ptr;
	}

}


MethodBind *ObjectTypeDB::get_method(StringName p_type, StringName p_name) {

	OBJTYPE_LOCK;

	TypeInfo *type=types.getptr(p_type);

	while(type) {

		MethodBind **method=type->method_map.getptr(p_name);
		if (method && *method)
			return *method;
		type=type->inherits_ptr;
	}
	return NULL;
}


void ObjectTypeDB::bind_integer_constant(const StringName& p_type, const StringName &p_name, int p_constant) {

	OBJTYPE_LOCK;

	TypeInfo *type=types.getptr(p_type);
	if (!type) {

		ERR_FAIL_COND(!type);
	}

	if (type->constant_map.has(p_name)) {

		ERR_FAIL();
	}

	type->constant_map[p_name]=p_constant;
#ifdef DEBUG_METHODS_ENABLED
	type->constant_order.push_back(p_name);
#endif

}

void ObjectTypeDB::get_integer_constant_list(const StringName& p_type, List<String> *p_constants, bool p_no_inheritance) {

	OBJTYPE_LOCK;

	TypeInfo *type=types.getptr(p_type);

	while(type) {

#ifdef DEBUG_METHODS_ENABLED
		for(List<StringName>::Element *E=type->constant_order.front();E;E=E->next())
			p_constants->push_back(E->get());
#else
		const StringName *K=NULL;

		while((K=type->constant_map.next(K))) {
			p_constants->push_back(*K);
		}

#endif
		if (p_no_inheritance)
			break;

		type=type->inherits_ptr;
	}

}


int ObjectTypeDB::get_integer_constant(const StringName& p_type, const StringName &p_name, bool *p_success) {

	OBJTYPE_LOCK;


	TypeInfo *type=types.getptr(p_type);

	while(type) {


		int *constant=type->constant_map.getptr(p_name);
		if (constant) {

			if (p_success)
				*p_success=true;
			return *constant;
		}

		type=type->inherits_ptr;
	}

	if (p_success)
		*p_success=false;

	return 0;
}

void ObjectTypeDB::add_signal(StringName p_type,const MethodInfo& p_signal) {

	TypeInfo *type=types.getptr(p_type);
	ERR_FAIL_COND(!type);

	TypeInfo *check=type;
	StringName sname = p_signal.name;
#ifdef DEBUG_METHODS_ENABLED

	while(check) {
		if (check->signal_map.has(sname)) {
			ERR_EXPLAIN("Type "+String(p_type)+" already has signal: "+String(sname));
			ERR_FAIL();
		}
		check=check->inherits_ptr;
	}
#endif

	type->signal_map[sname]=p_signal;

}

void ObjectTypeDB::get_signal_list(StringName p_type,List<MethodInfo> *p_signals,bool p_no_inheritance) {

	TypeInfo *type=types.getptr(p_type);
	ERR_FAIL_COND(!type);

	TypeInfo *check=type;

	while(check) {

		const StringName *S=NULL;
		while((S=check->signal_map.next(S))) {

			p_signals->push_back(check->signal_map[*S]);
		}

		if (p_no_inheritance)
			return;

		check=check->inherits_ptr;
	}


}

bool ObjectTypeDB::has_signal(StringName p_type,StringName p_signal) {

	TypeInfo *type=types.getptr(p_type);
	TypeInfo *check=type;
	while(check) {
		if (check->signal_map.has(p_signal))
			return true;
		check=check->inherits_ptr;
	}

	return false;
}

bool ObjectTypeDB::get_signal(StringName p_type,StringName p_signal,MethodInfo *r_signal) {

	TypeInfo *type=types.getptr(p_type);
	TypeInfo *check=type;
	while(check) {
		if (check->signal_map.has(p_signal)) {
			if (r_signal) {
				*r_signal=check->signal_map[p_signal];
			}
			return true;
		}
		check=check->inherits_ptr;
	}

	return false;
}

void ObjectTypeDB::add_property(StringName p_type,const PropertyInfo& p_pinfo, const StringName& p_setter, const StringName& p_getter, int p_index) {


	TypeInfo *type=types.getptr(p_type);
	ERR_FAIL_COND(!type);

	MethodBind *mb_set=NULL;
	if (p_setter) {
		mb_set = get_method(p_type,p_setter);
#ifdef DEBUG_METHODS_ENABLED
		if (!mb_set) {
			ERR_EXPLAIN("Invalid Setter: "+p_type+"::"+p_setter+" for property: "+p_pinfo.name);
			ERR_FAIL_COND(!mb_set);
		} else {
			int exp_args=1+(p_index>=0?1:0);
			if (mb_set->get_argument_count()!=exp_args) {
				ERR_EXPLAIN("Invalid Function for Setter: "+p_type+"::"+p_setter+" for property: "+p_pinfo.name);
				ERR_FAIL();

			}
		}
#endif
	}

	MethodBind *mb_get=NULL;
	if (p_getter) {

		MethodBind *mb_get = get_method(p_type,p_getter);
#ifdef DEBUG_METHODS_ENABLED

		if (!mb_get) {
			ERR_EXPLAIN("Invalid Getter: "+p_type+"::"+p_getter+" for property: "+p_pinfo.name);
			ERR_FAIL_COND(!mb_get);
		} else {

			int exp_args=0+(p_index>=0?1:0);
			if (mb_get->get_argument_count()!=exp_args) {
				ERR_EXPLAIN("Invalid Function for Getter: "+p_type+"::"+p_getter+" for property: "+p_pinfo.name);
				ERR_FAIL();

			}

		}
#endif
	}



#ifdef DEBUG_METHODS_ENABLED

	if (type->property_setget.has(p_pinfo.name)) {
		ERR_EXPLAIN("Object already has property: "+p_type);
		ERR_FAIL();
	}
#endif
	type->property_list.push_back(p_pinfo);

	PropertySetGet psg;
	psg.setter=p_setter;
	psg.getter=p_getter;
	psg._setptr=mb_set;
	psg._getptr=mb_get;
	psg.index=p_index;
	psg.type=p_pinfo.type;

	type->property_setget[p_pinfo.name]=psg;

}


void ObjectTypeDB::get_property_list(StringName p_type, List<PropertyInfo> *p_list, bool p_no_inheritance,const Object *p_validator) {

	TypeInfo *type=types.getptr(p_type);
	TypeInfo *check=type;
	while(check) {

		for(List<PropertyInfo>::Element *E=check->property_list.front();E;E=E->next()) {


			if (p_validator) {
				PropertyInfo pi = E->get();
				p_validator->_validate_property(pi);
				p_list->push_back(pi);
			} else {
				p_list->push_back(E->get());
			}
		}

		if (p_no_inheritance)
			return ;
		check=check->inherits_ptr;
	}

}
bool ObjectTypeDB::set_property(Object* p_object,const StringName& p_property, const Variant& p_value,bool *r_valid) {


	TypeInfo *type=types.getptr(p_object->get_type_name());
	TypeInfo *check=type;
	while(check) {
		const PropertySetGet *psg = check->property_setget.getptr(p_property);
		if (psg) {

			if (!psg->setter) {
				if (r_valid)
					*r_valid=false;
				return true; //return true but do nothing
			}

			Variant::CallError ce;

			if (psg->index>=0) {
				Variant index=psg->index;
				const Variant* arg[2]={&index,&p_value};
//				p_object->call(psg->setter,arg,2,ce);
				if (psg->_setptr) {
					psg->_setptr->call(p_object,arg,2,ce);
				} else {
					p_object->call(psg->setter,arg,2,ce);
				}


			} else {
				const Variant* arg[1]={&p_value};
				if (psg->_setptr) {
					psg->_setptr->call(p_object,arg,1,ce);
				} else {
					p_object->call(psg->setter,arg,1,ce);
				}
			}

			if (r_valid)
				*r_valid=ce.error==Variant::CallError::CALL_OK;

			return true;
		}

		check=check->inherits_ptr;
	}

	return false;
}
bool ObjectTypeDB::get_property(Object* p_object,const StringName& p_property, Variant& r_value) {

	TypeInfo *type=types.getptr(p_object->get_type_name());
	TypeInfo *check=type;
	while(check) {
		const PropertySetGet *psg = check->property_setget.getptr(p_property);
		if (psg) {
			if (!psg->getter)
				return true; //return true but do nothing

			if (psg->index>=0) {
				Variant index=psg->index;
				const Variant* arg[1]={&index};
				Variant::CallError ce;
				r_value = p_object->call(psg->getter,arg,1,ce);

			} else {

				Variant::CallError ce;
				if (psg->_getptr) {

					r_value = psg->_getptr->call(p_object,NULL,0,ce);
				} else {
					r_value = p_object->call(psg->getter,NULL,0,ce);
				}
			}
			return true;
		}

		const int *c =check->constant_map.getptr(p_property);
		if (c) {

			r_value=*c;
			return true;
		}
		//if (check->constant_map.fin)

		check=check->inherits_ptr;
	}

	return false;
}

Variant::Type ObjectTypeDB::get_property_type(const StringName& p_type, const StringName& p_property,bool *r_is_valid) {

	TypeInfo *type=types.getptr(p_type);
	TypeInfo *check=type;
	while(check) {
		const PropertySetGet *psg = check->property_setget.getptr(p_property);
		if (psg) {

			if (r_is_valid)
				*r_is_valid=true;

			return psg->type;
		}

		check=check->inherits_ptr;
	}
	if (r_is_valid)
		*r_is_valid=false;

	return Variant::NIL;

}


void ObjectTypeDB::set_method_flags(StringName p_type,StringName p_method,int p_flags) {

	TypeInfo *type=types.getptr(p_type);
	TypeInfo *check=type;
	ERR_FAIL_COND(!check);
	ERR_FAIL_COND(!check->method_map.has(p_method));
	check->method_map[p_method]->set_hint_flags(p_flags);


}

bool ObjectTypeDB::has_method(StringName p_type,StringName p_method,bool p_no_inheritance) {

	TypeInfo *type=types.getptr(p_type);
	TypeInfo *check=type;
	while(check) {
		if (check->method_map.has(p_method))
			return true;
		if (p_no_inheritance)
			return false;
		check=check->inherits_ptr;
	}

	return false;

}

bool ObjectTypeDB::get_setter_and_type_for_property(const StringName& p_class, const StringName& p_prop, StringName& r_class, StringName& r_setter) {

	TypeInfo *type=types.getptr(p_class);
	TypeInfo *check=type;
	while(check) {

		if (check->property_setget.has(p_prop)) {
			r_class=check->name;
			r_setter=check->property_setget[p_prop].setter;
			return true;
		}

		check=check->inherits_ptr;
	}

	return false;

}

#ifdef DEBUG_METHODS_ENABLED
MethodBind* ObjectTypeDB::bind_methodfi(uint32_t p_flags, MethodBind *p_bind , const MethodDefinition &method_name, const Variant **p_defs, int p_defcount) {
	StringName mdname=method_name.name;
#else
MethodBind* ObjectTypeDB::bind_methodfi(uint32_t p_flags, MethodBind *p_bind , const char *method_name, const Variant **p_defs, int p_defcount) {
	StringName mdname=StaticCString::create(method_name);
#endif


	StringName rettype;
	if (mdname.operator String().find(":")!=-1) {
		rettype = mdname.operator String().get_slice(":",1);
		mdname = mdname.operator String().get_slice(":",0);
	}


	OBJTYPE_LOCK;
	ERR_FAIL_COND_V(!p_bind,NULL);
	p_bind->set_name(mdname);

	String instance_type=p_bind->get_instance_type();

	TypeInfo *type=types.getptr(instance_type);
	if (!type) {
		ERR_PRINTS("Couldn't bind method '"+mdname+"' for instance: "+instance_type);
		memdelete(p_bind);
		ERR_FAIL_COND_V(!type,NULL);
	}

	if (type->method_map.has(mdname)) {
		memdelete(p_bind);
		// overloading not supported
		ERR_EXPLAIN("Method already bound: "+instance_type+"::"+mdname);
		ERR_FAIL_V(NULL);
	}
#ifdef DEBUG_METHODS_ENABLED
	p_bind->set_argument_names(method_name.args);
	p_bind->set_return_type(rettype);
	type->method_order.push_back(mdname);
#endif
	type->method_map[mdname]=p_bind;


	Vector<Variant> defvals;

	defvals.resize(p_defcount);
	for(int i=0;i<p_defcount;i++) {

		defvals[i]=*p_defs[p_defcount-i-1];
	}

	p_bind->set_default_arguments(defvals);
	p_bind->set_hint_flags(p_flags);
	return p_bind;

}

void ObjectTypeDB::add_virtual_method(const StringName& p_type, const MethodInfo& p_method , bool p_virtual) {
	ERR_FAIL_COND(!types.has(p_type));

#ifdef DEBUG_METHODS_ENABLED
	MethodInfo mi=p_method;
	if (p_virtual)
		mi.flags|=METHOD_FLAG_VIRTUAL;
	types[p_type].virtual_methods.push_back(mi);

#endif

}

void ObjectTypeDB::get_virtual_methods(const StringName& p_type, List<MethodInfo> * p_methods , bool p_no_inheritance) {

	ERR_FAIL_COND(!types.has(p_type));

#ifdef DEBUG_METHODS_ENABLED

	TypeInfo *type=types.getptr(p_type);
	TypeInfo *check=type;
	while(check) {

		for(List<MethodInfo>::Element *E=check->virtual_methods.front();E;E=E->next()) {
			p_methods->push_back(E->get());
		}

		if (p_no_inheritance)
			return;
		check=check->inherits_ptr;
	}

#endif

}

void ObjectTypeDB::set_type_enabled(StringName p_type,bool p_enable) {

	ERR_FAIL_COND(!types.has(p_type));
	types[p_type].disabled=!p_enable;
}

bool ObjectTypeDB::is_type_enabled(StringName p_type) {

	TypeInfo *ti=types.getptr(p_type);
	if (!ti || !ti->creation_func) {
		if (compat_types.has(p_type)) {
			ti=types.getptr(compat_types[p_type]);
		}
	}

	ERR_FAIL_COND_V(!ti,false);
	return !ti->disabled;
}

StringName ObjectTypeDB::get_category(const StringName& p_node) {

	ERR_FAIL_COND_V(!types.has(p_node),StringName());
#ifdef DEBUG_ENABLED
	return types[p_node].category;
#else
	return StringName();
#endif
}

void ObjectTypeDB::add_resource_base_extension(const StringName& p_extension,const StringName& p_type) {

	if (resource_base_extensions.has(p_extension))
		return;

	resource_base_extensions[p_extension]=p_type;
}

void ObjectTypeDB::get_resource_base_extensions(List<String> *p_extensions) {

	const StringName *K=NULL;

	while((K=resource_base_extensions.next(K))) {

		p_extensions->push_back(*K);
	}
}

void ObjectTypeDB::get_extensions_for_type(const StringName& p_type,List<String> *p_extensions) {

	const StringName *K=NULL;

	while((K=resource_base_extensions.next(K))) {
		StringName cmp = resource_base_extensions[*K];
		if (is_type(cmp,p_type))
			p_extensions->push_back(*K);
	}
}


Mutex *ObjectTypeDB::lock=NULL;

void ObjectTypeDB::init() {

#ifndef NO_THREADS

	lock = Mutex::create();
#endif
}

void ObjectTypeDB::cleanup() {


#ifndef NO_THREADS

	memdelete(lock);
#endif

	//OBJTYPE_LOCK; hah not here

	const StringName *k=NULL;

	while((k=types.next(k))) {

		TypeInfo &ti=types[*k];

		const StringName *m=NULL;
		while((m=ti.method_map.next(m))) {

			memdelete( ti.method_map[*m] );
		}
	}
	types.clear();
	resource_base_extensions.clear();
	compat_types.clear();
}

//
