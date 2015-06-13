/*************************************************************************/
/*  sproto.h                                                             */
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
#ifndef SPROTO_H
#define SPROTO_H

extern "C" {
struct sproto;
struct sproto_type;
};

#include "core/resource.h"

class Sproto : public Resource {
	OBJ_TYPE(Sproto,Resource);

	sproto* proto;

	Dictionary _decode(struct sproto_type *p_st, const String& p_type, const ByteArray& p_stream, bool p_use_default);

protected:
	static void _bind_methods();

public:
	Sproto();
	~Sproto();

	void set_proto(sproto* p_proto) { proto = p_proto; }

	void dump();
	Dictionary get_default(const String& p_type);
	ByteArray encode(const String& p_type, const Dictionary& p_dict);
	Dictionary decode(const String& p_type, const ByteArray& p_stream, bool p_use_default = false);

	enum Proto {
		REQUEST,
		RESPONSE,
	};

	int proto_tag(const String& p_type);
	String proto_name(int p_tag);
	Dictionary proto_get_default(const String& p_type, Proto p_what);
	ByteArray proto_encode(const String& p_type, Proto p_what, const Dictionary& p_dict);
	Dictionary proto_decode(const String& p_type, Proto p_what, const ByteArray& p_stream, bool p_use_default = false);
};

VARIANT_ENUM_CAST(Sproto::Proto);

#endif // SPROTO_H
