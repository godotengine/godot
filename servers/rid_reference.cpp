/**************************************************************************/
/*  rid_reference.cpp                                                     */
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

#include "rid_reference.h"

RIDReferences::Server RIDReferences::_servers[SERVER_COUNT];

uint32_t RIDReferences::request_reference(ServerType p_server_type) {
	uint32_t id = UINT32_MAX;
	MutexLock(_servers[p_server_type]._mutex);
	RIDReference *ref = _servers[p_server_type]._pool.request(id);
	ref->rid = RID();
	return id;
}

void RIDReferences::free_reference(ServerType p_server_type, uint32_t p_id) {
	MutexLock(_servers[p_server_type]._mutex);
	_servers[p_server_type]._pool.free(p_id);
}

void RIDReferences::set_reference(ServerType p_server_type, uint32_t p_id, RID p_rid) {
	MutexLock(_servers[p_server_type]._mutex);
	_servers[p_server_type]._pool[p_id].rid = p_rid;
}

RID RIDReferences::get_reference(ServerType p_server_type, uint32_t p_id) {
	MutexLock(_servers[p_server_type]._mutex);
	return _servers[p_server_type]._pool[p_id].rid;
}

// The important function, when deleting a RID, Null out any references to it,
// so no dangling references will be used.
// Note that this assumes single threaded use from client to server.
// If another thread frees a RID BEFORE the first thread uses it, then a dangling
// RID can still be used in the server.
void RIDReferences::notify_free_RID(ServerType p_server_type, RID p_rid) {
	ERR_FAIL_COND(p_rid == RID());
	MutexLock(_servers[p_server_type]._mutex);
	uint32_t active_size = _servers[p_server_type]._pool.active_size();
	for (uint32_t n = 0; n < active_size; n++) {
		RIDReference &ref = _servers[p_server_type]._pool.get_active(n);
		if (ref.rid == p_rid) {
			ref.rid = RID();
		}
	}
}
