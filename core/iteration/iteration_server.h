/**************************************************************************/
/* iteration_server.h                                                     */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#ifndef GODOT_ITERATION_SERVER_H
#define GODOT_ITERATION_SERVER_H

#include "core/error/error_list.h"
#include "core/object/class_db.h"
#include "core/object/object.h"
#include "core/variant/binder_common.h"

class CustomIterator;
class Main;

class IterationServer : public Object {
	GDCLASS(IterationServer, Object)

	friend class Main;

	enum {
		MAX_ITERATORS = 64
	};

	static CustomIterator *_iterators[MAX_ITERATORS];
	static int _iterator_count;

protected:
	static void _bind_methods();
	IterationServer(){};
	~IterationServer(){};

public:
	enum IteratorType {
		ITERATOR_TYPE_UNSET = 0,
		ITERATOR_TYPE_SEPARATE = 1 << 0,
		ITERATOR_TYPE_MIXED = 1 << 1
	};

	static int get_iterator_count() { return _iterator_count; }
	static CustomIterator *get_iterator(int p_idx);
	static Error register_iterator(CustomIterator *p_iterator);
	static Error unregister_iterator(const CustomIterator *p_iterator);
	static bool is_iterator_enabled_for_process(const String &name);
	static bool is_iterator_enabled_for_process(const CustomIterator *p_iterator);
	static bool is_iterator_enabled_for_physics(const String &name);
	static bool is_iterator_enabled_for_physics(const CustomIterator *p_iterator);
	static bool is_iterator_enabled_for_audio(const String &name);
	static bool is_iterator_enabled_for_audio(const CustomIterator *p_iterator);
	static bool is_iterator_enabled_for_mixed(const String &name);
	static bool is_iterator_enabled_for_mixed(const CustomIterator *p_iterator);
};

VARIANT_ENUM_CAST(IterationServer::IteratorType);

#endif //GODOT_ITERATION_SERVER_H
