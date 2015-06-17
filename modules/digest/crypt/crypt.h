/*************************************************************************/
/*  crypt.h                                                              */
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
#ifndef CRYPT_H
#define CRYPT_H

#include "core/reference.h"

class Crypt : public Object {

	OBJ_TYPE(Crypt,Object);

protected:

	static void _bind_methods();

public:

	ByteArray hashkey(const ByteArray& p_key);
	ByteArray randomkey();
	ByteArray desencode(const ByteArray& p_key, const ByteArray& p_text);
	ByteArray desdecode(const ByteArray& p_key, const ByteArray& p_text);
	ByteArray hexencode(const ByteArray& p_raw);
	ByteArray hexdecode(const ByteArray& p_hex);
	ByteArray hmac64(const ByteArray& p_x, const ByteArray& p_y);
	ByteArray dhexchange(const ByteArray& p_raw);
	ByteArray dhsecret(const ByteArray& p_x, const ByteArray& p_y);
	ByteArray base64encode(const ByteArray& p_raw);
	ByteArray base64decode(const ByteArray& p_base64);
	ByteArray sha1(const ByteArray& p_raw);
	ByteArray hmac_sha1(const ByteArray& p_key, const ByteArray& p_text);
	ByteArray hmac_hash(const ByteArray& p_key, const ByteArray& p_text);

	Crypt() {};
};


#endif // CRYPT_H
