/**************************************************************************/
/*  rid.h                                                                 */
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

#ifndef RID_H
#define RID_H

#include "core/typedefs.h"

class RID_AllocBase;

class RID {
	friend class RID_AllocBase;
	uint64_t _id = 0;

public:
	_ALWAYS_INLINE_ bool operator==(const RID &p_rid) const {
		return _id == p_rid._id;
	}
	_ALWAYS_INLINE_ bool operator<(const RID &p_rid) const {
		return _id < p_rid._id;
	}
	_ALWAYS_INLINE_ bool operator<=(const RID &p_rid) const {
		return _id <= p_rid._id;
	}
	_ALWAYS_INLINE_ bool operator>(const RID &p_rid) const {
		return _id > p_rid._id;
	}
	_ALWAYS_INLINE_ bool operator>=(const RID &p_rid) const {
		return _id >= p_rid._id;
	}
	_ALWAYS_INLINE_ bool operator!=(const RID &p_rid) const {
		return _id != p_rid._id;
	}
	_ALWAYS_INLINE_ bool is_valid() const { return _id != 0; }
	_ALWAYS_INLINE_ bool is_null() const { return _id == 0; }

	_ALWAYS_INLINE_ uint32_t get_local_index() const { return _id & 0xFFFFFFFF; }

	static _ALWAYS_INLINE_ RID from_uint64(uint64_t p_id) {
		RID _rid;
		_rid._id = p_id;
		return _rid;
	}
	_ALWAYS_INLINE_ uint64_t get_id() const { return _id; }

	_ALWAYS_INLINE_ RID() {}
};

#endif // RID_H
