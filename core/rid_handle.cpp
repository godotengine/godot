/*************************************************************************/
/*  rid_handle.cpp                                                       */
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

#include "rid_handle.h"

#ifdef RID_HANDLES_ENABLED
#include "core/os/memory.h"
#include "core/print_string.h"
#include "core/ustring.h"

// This define will flag up an error when get() or getptr() is called with a NULL object.
// These calls should more correctly be made as get_or_null().
// #define RID_HANDLE_FLAG_NULL_GETS

RID_Database g_rid_database;

RID_Data::~RID_Data() {
}

void RID_OwnerBase::init_rid() {
	// NOOP
}

RID_OwnerBase::~RID_OwnerBase() {
	_shutdown = true;
}

void RID_OwnerBase::_rid_print(const char *pszType, String sz, const RID &p_rid) {
	String tn = pszType;
	print_line(tn + " : " + sz + " " + itos(p_rid._id) + " [ " + itos(p_rid._revision) + " ]");
}

RID_Database::RID_Database() {
	// request first element so the handles start from 1
	uint32_t dummy;
	_pool.request(dummy);
}

RID_Database::~RID_Database() {
}

String RID_Database::_rid_to_string(const RID &p_rid, const PoolElement &p_pe) {
	String s = "RID id=" + itos(p_rid._id);
	s += " [ rev " + itos(p_rid._revision) + " ], ";

	s += "PE [ rev " + itos(p_pe.revision) + " ] ";
#ifdef RID_HANDLE_ALLOCATION_TRACKING_ENABLED
	if (p_pe.filename) {
		s += String(p_pe.filename).get_file() + " ";
	}
	s += "line " + itos(p_pe.line_number);

	if (p_pe.previous_filename) {
		s += " ( prev ";
		s += String(p_pe.previous_filename).get_file() + " ";
		s += "line " + itos(p_pe.previous_line_number) + " )";
	}
#endif
	return s;
}

void RID_Database::preshutdown() {
#ifdef RID_HANDLE_ALLOCATION_TRACKING_ENABLED
	for (uint32_t n = 0; n < _pool.active_size(); n++) {
		uint32_t id = _pool.get_active_id(n);

		// ignore zero as it is a dummy
		if (!id) {
			continue;
		}

		PoolElement &pe = _pool[id];

		if (pe.data) {
			if (pe.data->_owner) {
				const char *tn = pe.data->_owner->get_typename();

				// does it already exist?
				bool found = false;
				for (unsigned int i = 0; i < _owner_names.size(); i++) {
					if (_owner_names[i] == tn) {
						found = true;
						pe.owner_name_id = i;
					}
				}

				if (!found) {
					pe.owner_name_id = _owner_names.size();
					_owner_names.push_back(tn);
				}
			}
		}
	}
#endif
}

void RID_Database::register_leak(uint32_t p_line_number, uint32_t p_owner_name_id, const char *p_filename) {
	// does the leak exist already?
	for (unsigned int n = 0; n < _leaks.size(); n++) {
		Leak &leak = _leaks[n];
		if ((leak.line_number == p_line_number) && (leak.filename == p_filename) && (leak.owner_name_id == p_owner_name_id)) {
			leak.num_objects_leaked += 1;
			return;
		}
	}

	Leak leak;
	leak.filename = p_filename;
	leak.line_number = p_line_number;
	leak.owner_name_id = p_owner_name_id;
	leak.num_objects_leaked = 1;
	_leaks.push_back(leak);
}

void RID_Database::shutdown() {
	// free the first dummy element, so we don't report a false leak
	_pool.free(0);

	// print leaks
	if (_pool.active_size()) {
		ERR_PRINT("RID_Database leaked " + itos(_pool.active_size()) + " objects at exit.");

#ifdef RID_HANDLE_ALLOCATION_TRACKING_ENABLED
		for (uint32_t n = 0; n < _pool.active_size(); n++) {
			const PoolElement &pe = _pool.get_active(n);

			register_leak(pe.line_number, pe.owner_name_id, pe.filename);
		}
#endif
	}

#ifdef RID_HANDLE_ALLOCATION_TRACKING_ENABLED
	for (uint32_t n = 0; n < _leaks.size(); n++) {
		const Leak &leak = _leaks[n];

		const char *tn = "RID_Owner unknown";
		if (_owner_names.size()) {
			tn = _owner_names[leak.owner_name_id];
		}

		const char *fn = "Filename unknown";
		if (leak.filename) {
			fn = leak.filename;
		}

		_err_print_error(tn, fn, leak.line_number, itos(leak.num_objects_leaked) + " RID objects leaked");
	}
#endif

	_shutdown = true;
}

void RID_Database::handle_make_rid(RID &r_rid, RID_Data *p_data, RID_OwnerBase *p_owner) {
	ERR_FAIL_COND_MSG(_shutdown, "RID_Database make_rid use after shutdown.");
	ERR_FAIL_COND_MSG(!p_data, "RID_Database make_rid, data is empty.");
	bool data_was_empty = true;

	_mutex.lock();
	PoolElement *pe = _pool.request(r_rid._id);
	data_was_empty = !pe->data;

	pe->data = p_data;
	p_data->_owner = p_owner;
	pe->revision = pe->revision + 1;
	r_rid._revision = pe->revision;

#ifdef RID_HANDLE_ALLOCATION_TRACKING_ENABLED
	// make a note of the previous allocation - this isn't super necessary
	// but can pinpoint source allocations when dangling RIDs occur.
	pe->previous_filename = pe->filename;
	pe->previous_line_number = pe->line_number;

	pe->line_number = 0;
	pe->filename = nullptr;
#endif

	_mutex.unlock();

	ERR_FAIL_COND_MSG(!data_was_empty, "RID_Database make_rid, previous data was not empty.");
}

RID_Data *RID_Database::handle_get(const RID &p_rid) {
	RID_Data *data = handle_get_or_null(p_rid);
#ifdef RID_HANDLE_FLAG_NULL_GETS
	ERR_FAIL_COND_V_MSG(!data, nullptr, "RID_Database get is NULL");
#endif
	return data;
}

RID_Data *RID_Database::handle_getptr(const RID &p_rid) {
	RID_Data *data = handle_get_or_null(p_rid);
#ifdef RID_HANDLE_FLAG_NULL_GETS
	ERR_FAIL_COND_V_MSG(!data, nullptr, "RID_Database getptr is NULL");
#endif
	return data;
}

// Note, no locks used in the getters.
// Godot 4.x does use locks in the getters, but it is arguably overkill because even though
// the pointer returned will be correct (i.e. it has not been replaced during this call),
// it can be invalidated during the client code use. (There may also be an internal reason why
// locks are needed in 4.x, as the database is different.)
// An example of a "safer" way to do this kind of thing is object level locking,
// (but that has complications of its own), or atomic object changes.

RID_Data *RID_Database::handle_get_or_null(const RID &p_rid) {
	if (p_rid.is_valid()) {
		ERR_FAIL_COND_V_MSG(_shutdown, nullptr, "RID_Database get_or_null after shutdown.");

		// The if statement is to allow breakpointing without a recompile.
		if (p_rid._id >= _pool.pool_reserved_size()) {
			ERR_FAIL_COND_V_MSG(p_rid._id >= _pool.pool_reserved_size(), nullptr, "RID_Database get_or_null, RID id was outside pool size.");
		}

		const PoolElement &pe = _pool[p_rid._id];
		if (pe.revision != p_rid._revision) {
			ERR_FAIL_COND_V_MSG(pe.revision != p_rid._revision, nullptr, "RID get_or_null, revision is incorrect, possible dangling RID. " + _rid_to_string(p_rid, pe));
		}

		return pe.data;
	}
	return nullptr;
}

bool RID_Database::handle_owns(const RID &p_rid) const {
	ERR_FAIL_COND_V_MSG(_shutdown, false, "RID_Database owns after shutdown.");

	if (!p_rid.is_valid()) {
		return false;
	}

	if (p_rid._id >= _pool.pool_reserved_size()) {
		return false;
	}

	const PoolElement &pe = _pool[p_rid._id];
	if (pe.revision != p_rid._revision) {
		return false;
	}

	if (!pe.data) {
		return false;
	}

	return true;
}

void RID_Database::handle_free(const RID &p_rid) {
	ERR_FAIL_COND_MSG(_shutdown, "RID_Database free after shutdown.");
	bool revision_correct = true;

	ERR_FAIL_COND_MSG(p_rid._id >= _pool.pool_reserved_size(), "RID_Database free, RID id was outside pool size.");
	_mutex.lock();
	PoolElement &pe = _pool[p_rid._id];
	revision_correct = pe.revision == p_rid._revision;

	// mark the data as zero, which indicates unused element
	if (revision_correct) {
		pe.data->_owner = nullptr;
		pe.data = nullptr;
		_pool.free(p_rid._id);
	}

	_mutex.unlock();

	ERR_FAIL_COND_MSG(!revision_correct, "RID_Database free, revision is incorrect, object possibly freed more than once.");
}

RID RID_Database::prime(const RID &p_rid, int p_line_number, const char *p_filename) {
#ifdef RID_HANDLE_ALLOCATION_TRACKING_ENABLED
	if (p_rid.is_valid()) {
		ERR_FAIL_COND_V_MSG(_shutdown, p_rid, "RID_Database prime after shutdown.");
		ERR_FAIL_COND_V_MSG(p_rid._id >= _pool.pool_reserved_size(), p_rid, "RID_Database prime, RID id was outside pool size.");

		PoolElement &pe = _pool[p_rid._id];
		ERR_FAIL_COND_V_MSG(pe.revision != p_rid._revision, p_rid, "RID_Database prime, revision is incorrect, object possibly freed before use.");

		// no threading checks as it the tracking info doesn't matter if there is a race condition
		pe.line_number = p_line_number;
		pe.filename = p_filename;
	}
#endif
	return p_rid;
}

#endif // RID_HANDLES_ENABLED
