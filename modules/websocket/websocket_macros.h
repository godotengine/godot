/*************************************************************************/
/*  websocket_macros.h                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef WEBSOCKETMACTOS_H
#define WEBSOCKETMACTOS_H

// Defaults per peer buffers, 1024 packets with a shared 65536 bytes payload.
#define DEF_PKT_SHIFT 10
#define DEF_BUF_SHIFT 16

/* clang-format off */
#define GDCICLASS(CNAME) \
public:\
	static CNAME *(*_create)();\
\
	static Ref<CNAME > create_ref() {\
\
		if (!_create)\
			return Ref<CNAME >();\
		return Ref<CNAME >(_create());\
	}\
\
	static CNAME *create() {\
\
		if (!_create)\
			return nullptr;\
		return _create();\
	}\
protected:\

#define GDCINULL(CNAME) \
CNAME *(*CNAME::_create)() = nullptr;

#define GDCIIMPL(IMPNAME, CNAME) \
public:\
	static CNAME *_create() { return memnew(IMPNAME); }\
	static void make_default() { CNAME::_create = IMPNAME::_create; }\
protected:\
/* clang-format on */

#endif // WEBSOCKETMACTOS_H
