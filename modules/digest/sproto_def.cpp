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

static int encode_default(const struct sproto_arg *args) {

	Dictionary& dict = *(Dictionary *) args->ud;
	Variant value;
	if(args->index > 0) {
		if(args->mainindex > 0)
			value = Dictionary(true);
		else
			value = Array(true);
	} else {
		switch(args->type) {
		case SPROTO_TINTEGER:
			value = 0;
			break;
		case SPROTO_TREAL:
			value = 0.0f;
			break;
		case SPROTO_TBOOLEAN:
			value = false;
			break;
		case SPROTO_TSTRING:
			value = "";
			break;
		case SPROTO_TSTRUCT:
			Dictionary sub(true);
			sub["__type"] = sproto_name(args->subtype);
			value = sub;
			char dummy[32];
			sproto_encode(args->subtype, dummy, sizeof(dummy), encode_default, &sub);
			break;
		}
	}
	dict[args->tagname] = value;
	return 0;
}

Dictionary Sproto::get_default(const String& p_type) {

	ERR_FAIL_COND_V(proto == NULL, Variant());
	struct sproto_type *st = sproto_type(proto, p_type.utf8().get_data());
	ERR_FAIL_COND_V(st == NULL, Variant());
	// 32 is enough for dummy buffer, because ldefault encode nothing but the header.
	char dummy[32];
	Dictionary dict(true);
	dict["__type"] = p_type;
	int ret = sproto_encode(st, dummy, sizeof(dummy), encode_default, &dict);
	ERR_FAIL_COND_V(ret < 0, Variant());

	return dict;
}

Dictionary Sproto::proto_get_default(const String& p_type, Proto p_what) {

	ERR_FAIL_COND_V(proto == NULL, Variant());
	int tag = sproto_prototag(proto, p_type.utf8().get_data());
	ERR_FAIL_COND_V(tag == -1, Variant());
	struct sproto_type *st = sproto_protoquery(proto, tag, p_what);
	ERR_FAIL_COND_V(st == NULL, Variant());

	// 32 is enough for dummy buffer, because ldefault encode nothing but the header.
	char dummy[32];
	Dictionary dict(true);
	dict["__type"] = p_type;
	int ret = sproto_encode(st, dummy, sizeof(dummy), encode_default, &dict);
	ERR_FAIL_COND_V(ret < 0, Variant());

	return dict;
}
