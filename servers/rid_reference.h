/**************************************************************************/
/*  rid_reference.h                                                       */
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

#ifndef RID_REFERENCES_H
#define RID_REFERENCES_H

#include "core/os/mutex.h"
#include "core/pooled_list.h"
#include "core/rid.h"

class RIDReferences {
public:
	enum ServerType {
		SERVER_PHYSICS,
		SERVER_COUNT,
	};
	template <RIDReferences::ServerType>
	friend class RIDRef;

private:
	struct RIDReference {
		RID rid;
	};
	struct Server {
		TrackedPooledList<RIDReference> _pool;
		Mutex _mutex;
	};

	static Server _servers[SERVER_COUNT];

	static uint32_t request_reference(ServerType p_server_type);
	static void free_reference(ServerType p_server_type, uint32_t p_id);
	static void set_reference(ServerType p_server_type, uint32_t p_id, RID p_rid);
	static RID get_reference(ServerType p_server_type, uint32_t p_id);

public:
	static void notify_free_RID(ServerType p_server_type, RID p_rid);
};

template <RIDReferences::ServerType TYPE>
class RIDRef {
	uint32_t id = UINT32_MAX;

public:
	bool is_valid() const { return id != UINT32_MAX; }
	void set(RID p_rid) {
		if (p_rid.is_valid()) {
			if (!is_valid()) {
				id = RIDReferences::request_reference(TYPE);
			}
			RIDReferences::set_reference(TYPE, id, p_rid);
		} else {
			if (is_valid()) {
				RIDReferences::free_reference(TYPE, id);
				id = UINT32_MAX;
			}
		}
	}
	RID get() const {
		if (is_valid()) {
			return RIDReferences::get_reference(TYPE, id);
		}
		return RID();
	}
	~RIDRef() {
		if (is_valid()) {
			RIDReferences::free_reference(TYPE, id);
			id = UINT32_MAX;
		}
	}
};

#endif // RID_REFERENCES_H
