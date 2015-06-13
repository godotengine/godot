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
#include <stdlib.h>
#include <memory.h>
#include "sproto.h"
extern "C" {
#include "sproto/sproto.h"
};

struct encode_ud {
	encode_ud() : st(NULL), source_tag(NULL) {}
	struct sproto_type *st;
	Variant value;
	const char *source_tag;
	Variant source;
	int deep;
};
static int encode_callback(const struct sproto_arg *args) {

	struct encode_ud *self = (struct encode_ud *) args->ud;
	Variant& value = self->value;
	Variant source = self->source;

	if (args->index > 0) {
		if (args->tagname != self->source_tag) {
			// a new array
			self->source_tag = args->tagname;
			bool r_valid;
			source = value.get(args->tagname, &r_valid);
			ERR_FAIL_COND_V(!r_valid, 0);
			if(source.get_type() == Variant::NIL)
				return 0;
			if(source.get_type() != Variant::DICTIONARY && source.get_type() != Variant::ARRAY) {
				ERR_EXPLAIN(String(args->tagname)
					+ "("
					+ String::num(args->tagid)
					+ ") should be a dict/array (Is a "
					+ source.get_type_name(source.get_type())
					+ ")"
				);
				ERR_FAIL_V(0);
			}
			self->source = source;
		}
		int index = args->index - 1;
		if (args->mainindex >= 0) {
			// todo: check the key is equal to mainindex value
			if(source.get_type() == Variant::DICTIONARY) {
				Dictionary dict = source;
				if(index >= dict.size())
					return 0;
				const Variant *K=NULL;
				while((K=dict.next(K))) {
					if(index-- == 0) {
						source = dict[*K];
						break;
					}
				}
			}
			else if(source.get_type() == Variant::ARRAY) {
				Array array = source;
				if(index >= array.size())
					return 0;
				source = array[index];
			}
		} else {
			if(source.get_type() == Variant::DICTIONARY) {
				Dictionary dict = source;
				if(!dict.has(index))
					return 0;
				source = dict[index];
			}
			else if(source.get_type() == Variant::ARRAY) {
				Array array = source;
				if(index >= array.size())
					return 0;
				source = array[index];
			}
		}
	} else {
		if(value.get_type() != Variant::DICTIONARY && value.get_type() != Variant::ARRAY) {
			ERR_EXPLAIN(String(args->tagname)
				+ "("
				+ String::num(args->tagid)
				+ ") should be a dict/array (Is a "
				+ value.get_type_name(value.get_type())
				+ ")"
			);
			ERR_FAIL_V(0);
		}
		if(value.get_type() == Variant::DICTIONARY) {
			Dictionary dict = value;
			if(!dict.has(args->tagname))
				return 0;
			source = dict[args->tagname];
		}
		else if(value.get_type() == Variant::ARRAY) {
			Array array = value;
			int idx = atoi(args->tagname);
			if(idx >= array.size())
				return 0;
			source = array[idx];
		}
	}

	if(source.get_type() == Variant::NIL)
		return 0;
	switch (args->type) {
	case SPROTO_TINTEGER: {
		if(source.get_type() != Variant::REAL && source.get_type() != Variant::INT) {
			ERR_EXPLAIN(String(args->tagname)
				+ "("
				+ String::num(args->tagid)
				+ ") should be a int/real (Is a "
				+ source.get_type_name(source.get_type())
				+ ")"
			);
			ERR_FAIL_V(0);
		}
		long v = (double) source;
		// notice: godot only support 32bit integer
		long vh = v >> 31;
		if (vh == 0 || vh == -1) {
			if(args->length < 4)
				return -1;
			*(unsigned int *)args->value = (unsigned int)v;
			//printf("%s -> %d bytes\n", args->tagname, 4);
			return 4;
		}
		else {
			*(unsigned long *)args->value = (unsigned long)v;
			if(args->length < 8)
				return -1;
			//printf("%s -> %d bytes\n", args->tagname, 8);
			return 8;
		}
	}
	case SPROTO_TREAL: {
		if(source.get_type() != Variant::REAL && source.get_type() != Variant::INT) {
			ERR_EXPLAIN(String(args->tagname)
				+ "("
				+ String::num(args->tagid)
				+ ") should be a int/real (Is a "
				+ source.get_type_name(source.get_type())
				+ ")"
			);
			ERR_FAIL_V(0);
		}
		double v = source;
		*(double *)args->value = v;
		//printf("%s -> %d bytes\n", args->tagname, 8);
		return 8;
	}
	case SPROTO_TBOOLEAN: {
		if(source.get_type() != Variant::BOOL) {
			ERR_EXPLAIN(String(args->tagname)
				+ "("
				+ String::num(args->tagid)
				+ ") should be a bool (Is a "
				+ source.get_type_name(source.get_type())
				+ ")"
			);
			ERR_FAIL_V(0);
		}
		bool v = source;
		*(int *)args->value = v;
		//printf("%s -> %d bytes\n", args->tagname, 4);
		return 4;
	}
	case SPROTO_TSTRING: {
		if(source.get_type() != Variant::STRING) {
			ERR_EXPLAIN(String(args->tagname)
				+ "("
				+ String::num(args->tagid)
				+ ") should be a string (Is a "
				+ source.get_type_name(source.get_type())
				+ ")"
			);
			ERR_FAIL_V(0);
		}
		String v = source;
		CharString utf8 = v.utf8();
		size_t sz = utf8.length();
		if(sz > args->length)
			return -1;
		memcpy(args->value, utf8.get_data(), sz);
		//printf("%s -> %d bytes\n", args->tagname, sz + 1);
		return sz + 1;	// The length of empty string is 1.
	}
	case SPROTO_TSTRUCT: {
		struct encode_ud sub;
		sub.value = source;
		int r;
		if(source.get_type() != Variant::DICTIONARY && source.get_type() != Variant::ARRAY) {
			ERR_EXPLAIN(String(args->tagname)
				+ "("
				+ String::num(args->tagid)
				+ ") should be a dict/array (Is a "
				+ source.get_type_name(source.get_type())
				+ ")"
			);
			ERR_FAIL_V(0);
		}
		sub.st = args->subtype;
		sub.deep = self->deep + 1;
		r = sproto_encode(args->subtype, args->value, args->length, encode_callback, &sub);
		//printf("%s -> %d bytes\n", args->tagname, r);
		return r;
	}
	default:
		ERR_EXPLAIN("Invalid field type: " + String::num(args->type));
		ERR_FAIL_V(0);
	}
	return 0;
}

static ByteArray _encode(struct sproto_type *p_st, const Dictionary& p_dict) {

	ByteArray output;
	output.resize(1024);
	ByteArray::Write w = output.write();

	encode_ud self;
	self.value = p_dict;
	self.st = p_st;
	for (;;) {
		self.deep = 0;
		int r = sproto_encode(self.st, w.ptr(), output.size(), encode_callback, &self);
		w = ByteArray::Write();
		if (r<0) {
			output.resize(output.size() * 2);
			ByteArray::Write w = output.write();
		} else {
			output.resize(r);
			break;
		}
	}
	return output;
}

ByteArray Sproto::encode(const String& p_type, const Dictionary& p_dict) {

	ERR_FAIL_COND_V(proto == NULL, ByteArray());
	struct sproto_type *st = sproto_type(proto, p_type.utf8().get_data());
	ERR_FAIL_COND_V(st == NULL, ByteArray());

	return _encode(st, p_dict);
}

ByteArray Sproto::proto_encode(const String& p_type, Proto p_what, const Dictionary& p_dict) {

	ERR_FAIL_COND_V(proto == NULL, ByteArray());
	int tag = sproto_prototag(proto, p_type.utf8().get_data());
	ERR_FAIL_COND_V(tag == -1, ByteArray());
	struct sproto_type *st = sproto_protoquery(proto, tag, p_what);
	ERR_FAIL_COND_V(st == NULL, ByteArray());

	return _encode(st, p_dict);
}
