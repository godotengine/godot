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
#include "core/reference.h"
#include "core/globals.h"

#include "snappy/snappy-c.h"

#ifdef MODULE_DIGEST_ENABLED

class Snappy : public Reference {

	OBJ_TYPE(Snappy,Reference);

protected:

	static void _bind_methods();

public:

	ByteArray compress(const ByteArray& p_input);
	ByteArray uncompress(const ByteArray& p_input);

	Snappy() {};
};

static Snappy *_snappy = NULL;

ByteArray Snappy::compress(const ByteArray& p_input) {

	ByteArray::Read r = p_input.read();
	size_t max_compressed_length = snappy_max_compressed_length(p_input.size());

	ByteArray output;
	output.resize(max_compressed_length);
	ByteArray::Write w = output.write();

	size_t compressed_length;
	if(snappy_compress((const char *) r.ptr(), p_input.size(), (char *) w.ptr(), &compressed_length) == SNAPPY_OK) {

		w = ByteArray::Write();
		output.resize(compressed_length);
		return output;
	}

	return ByteArray();
}

ByteArray Snappy::uncompress(const ByteArray& p_input) {

	ByteArray::Read r = p_input.read();
	size_t uncompressed_length;
	if(snappy_uncompressed_length((const char *) r.ptr(), p_input.size(), &uncompressed_length) == SNAPPY_OK) {
		
		ByteArray output;
		output.resize(uncompressed_length);
		ByteArray::Write w = output.write();

		if(::snappy_uncompress((const char *) r.ptr(), p_input.size(), (char *) w.ptr(), &uncompressed_length) == SNAPPY_OK)
			return output;
	}

	return ByteArray();
}

void Snappy::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("compress"),&Snappy::compress);
	ObjectTypeDB::bind_method(_MD("uncompress"),&Snappy::uncompress);
}

void register_digest_types() {

	ObjectTypeDB::register_type<Snappy>();
	_snappy = memnew(Snappy);
	Globals::get_singleton()->add_singleton( Globals::Singleton("Snappy",_snappy ) );
}

void unregister_digest_types() {

	memdelete( _snappy );
}

#endif
