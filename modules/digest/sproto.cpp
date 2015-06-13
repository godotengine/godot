/*************************************************************************/
/*  sproto.cpp                                                          */
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
#include "sproto.h"
extern "C" {
#include "sproto/sproto.h"
};

void Sproto::dump() {

	ERR_FAIL_COND(proto == NULL);
	sproto_dump(proto);
}

int Sproto::proto_tag(const String& p_type) {

	ERR_FAIL_COND_V(proto == NULL, -1);
	return sproto_prototag(proto, p_type.utf8().get_data());
}

String Sproto::proto_name(int p_tag) {

	ERR_FAIL_COND_V(proto == NULL, "");
	return sproto_protoname(proto, p_tag);
}

void Sproto::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("dump"),&Sproto::dump);
	ObjectTypeDB::bind_method(_MD("get_default","type"),&Sproto::get_default);
	ObjectTypeDB::bind_method(_MD("encode","type","dict"),&Sproto::encode);
	ObjectTypeDB::bind_method(_MD("decode","type","stream","use_default"),&Sproto::decode,false);

	ObjectTypeDB::bind_method(_MD("proto_tag","type"),&Sproto::proto_tag);
	ObjectTypeDB::bind_method(_MD("proto_name","tag"),&Sproto::proto_name);
	ObjectTypeDB::bind_method(_MD("proto_get_default","type","what"),&Sproto::proto_get_default);
	ObjectTypeDB::bind_method(_MD("proto_encode","type","what","dict"),&Sproto::proto_encode);
	ObjectTypeDB::bind_method(_MD("proto_decode","type","what","stream","use_default"),&Sproto::proto_decode,false);

	BIND_CONSTANT(REQUEST);
	BIND_CONSTANT(RESPONSE);
}

Sproto::Sproto()
	: proto(NULL)
{
}

Sproto::~Sproto() {

	if(proto != NULL)
		sproto_release(proto);
}
