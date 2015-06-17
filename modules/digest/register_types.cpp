/*************************************************/
/*  register_script_types.cpp                    */
/*************************************************/
/*            This file is part of:              */
/*                GODOT ENGINE                   */
/*************************************************/
/*       Source code within this file is:        */
/*  (c) 2007-2010 Juan Linietsky, Ariel Manzur   */
/*             All Rights Reserved.              */
/*************************************************/
#include "register_types.h"
#include "core/globals.h"
#include "snappy.h"
#include "sproto.h"
#include "crypt/crypt.h"

#include "core/io/resource_loader.h"
#include "os/file_access.h"

extern "C" {
#include "sproto/sproto.h"
};

class ResourceFormatLoaderSproto : public ResourceFormatLoader {
public:

	virtual RES load(const String &p_path, const String& p_original_path = "") {

		Sproto *res = memnew(Sproto);
		Ref<Sproto> ref(res);

		Vector<uint8_t> bytecode = FileAccess::get_file_as_array(p_path);

		sproto *proto = sproto_create(bytecode.ptr(), bytecode.size());
		ERR_FAIL_COND_V(proto == NULL, RES());

		res->set_proto(proto);
		res->set_path(p_path);
		return ref;
	}

	virtual void get_recognized_extensions(List<String> *p_extensions) const {

		p_extensions->push_back("spb");
	}

	virtual bool handles_type(const String& p_type) const {

		return p_type=="Sproto";
	}

	virtual String get_resource_type(const String &p_path) const {

		String el = p_path.extension().to_lower();
		if (el=="spb")
			return "Sproto";
		return "";
	}
};

static ResourceFormatLoaderSproto *resource_loader_sproto = NULL;
static Snappy *_snappy = NULL;
static Crypt *_crypt = NULL;

void register_digest_types() {

	ObjectTypeDB::register_type<Snappy>();
	_snappy = memnew(Snappy);
	Globals::get_singleton()->add_singleton( Globals::Singleton("Snappy",_snappy ) );
	ObjectTypeDB::register_type<Crypt>();
	_crypt = memnew(Crypt);
	Globals::get_singleton()->add_singleton( Globals::Singleton("Crypt",_crypt ) );


	ObjectTypeDB::register_type<Sproto>();
	resource_loader_sproto = memnew( ResourceFormatLoaderSproto );
	ResourceLoader::add_resource_format_loader(resource_loader_sproto);
}

void unregister_digest_types() {

	if (resource_loader_sproto)
		memdelete(resource_loader_sproto);

	memdelete( _snappy );
	memdelete( _crypt );
}
