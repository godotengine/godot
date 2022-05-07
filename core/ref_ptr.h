/*************************************************************************/
/*  ref_ptr.h                                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef REF_PTR_H
#define REF_PTR_H
/**
	@author Juan Linietsky <reduzio@gmail.com>
 * This class exists to workaround a limitation in C++ but keep the design OK.
 * It's basically an opaque container of a Reference reference, so Variant can use it.
*/

#include "core/rid.h"

class RefPtr {
	enum {

		DATASIZE = sizeof(void *) //*4 -ref was shrunk
	};

	mutable char data[DATASIZE]; // too much probably, virtual class + pointer
public:
	bool is_null() const;
	void operator=(const RefPtr &p_other);
	bool operator==(const RefPtr &p_other) const;
	bool operator!=(const RefPtr &p_other) const;
	RID get_rid() const;
	void unref();
	_FORCE_INLINE_ void *get_data() const { return data; }
	RefPtr(const RefPtr &p_other);
	RefPtr();
	~RefPtr();
};

#endif // REF_PTR_H
