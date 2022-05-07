/*************************************************************************/
/*  rid_handle.h                                                         */
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

#ifndef RID_HANDLE_H
#define RID_HANDLE_H

#include "core/list.h"
#include "core/os/mutex.h"
#include "core/pooled_list.h"
#include "core/safe_refcount.h"
#include "core/typedefs.h"

#include <typeinfo>

// SCONS parameters:
// rids=pointers (default)
// rids=handles
// rids=tracked_handles (handles plus allocation tracking)

// Defines RID_HANDLES_ENABLED and RID_HANDLE_ALLOCATION_TRACKING_ENABLED
// should be defined from Scons as required following the above convention.

// RID_PRIME is the macro which stores line numbers and file in the RID_Database.
// It will be a NOOP if tracking is off.
#ifdef RID_HANDLE_ALLOCATION_TRACKING_ENABLED
#define RID_PRIME(a) g_rid_database.prime(a, __LINE__, __FILE__)
#else
#define RID_PRIME(a) a
#endif

// All the handle code can be compiled out if they are not in use.
#ifdef RID_HANDLES_ENABLED

// This define will print out each make_rid and free_rid. Useful for debugging.
// #define RID_HANDLE_PRINT_LIFETIMES

class RID_OwnerBase;
class RID_Database;

class RID_Data {
	friend class RID_OwnerBase;
	friend class RID_Database;

	RID_OwnerBase *_owner;
	uint32_t _id;

public:
	uint32_t get_id() const { return _id; }

	virtual ~RID_Data();
};

class RID_Handle {
public:
	union {
		struct {
			uint32_t _id;
			uint32_t _revision;
		};
		uint64_t _handle_data;
	};

	RID_Handle() {
		_handle_data = 0;
	}

	bool operator==(const RID_Handle &p_rid) const {
		return _handle_data == p_rid._handle_data;
	}
	bool operator<(const RID_Handle &p_rid) const {
		return _handle_data < p_rid._handle_data;
	}
	bool operator<=(const RID_Handle &p_rid) const {
		return _handle_data <= p_rid._handle_data;
	}
	bool operator>(const RID_Handle &p_rid) const {
		return _handle_data > p_rid._handle_data;
	}
	bool operator!=(const RID_Handle &p_rid) const {
		return _handle_data != p_rid._handle_data;
	}
	bool is_valid() const { return _id != 0; }

	uint32_t get_id() const { return _id ? _handle_data : 0; }
};

class RID : public RID_Handle {
};

class RID_Database {
	struct PoolElement {
		RID_Data *data;
		uint32_t revision;
#ifdef RID_HANDLE_ALLOCATION_TRACKING_ENABLED
		// current allocation
		uint16_t line_number;
		uint16_t owner_name_id;
		const char *filename;

		// previous allocation (allows identifying dangling RID source allocations)
		const char *previous_filename;
		uint32_t previous_line_number;
#endif
	};

	struct Leak {
		uint16_t line_number;
		uint16_t owner_name_id;
		const char *filename;
		uint32_t num_objects_leaked;
	};

	// The pooled list zeros on first request .. this is important
	// so that we initialize the revision to zero. Other than that, it
	// is treated as a POD type.
	TrackedPooledList<PoolElement, uint32_t, true, true> _pool;
	bool _shutdown = false;
	mutable Mutex _mutex;

	// This is purely for printing the leaks at the end, as RID_Owners may be
	// destroyed before the RID_Database is shutdown, so the RID_Data may be invalid
	// by this point, and we still want to have a record of the owner names.
	// The owner names should part of the binary, thus the pointers should still be valid.
	// They were retrieved using typeid(T).name()
	LocalVector<const char *> _owner_names;
	LocalVector<Leak> _leaks;

	void register_leak(uint32_t p_line_number, uint32_t p_owner_name_id, const char *p_filename);
	String _rid_to_string(const RID &p_rid, const PoolElement &p_pe) const;

public:
	RID_Database();
	~RID_Database();

	// Called to record the owner names before RID_Owners are destroyed
	void preshutdown();

	// Called after destroying RID_Owners to detect leaks
	void shutdown();

	// Prepare a RID for memory tracking
	RID prime(const RID &p_rid, int p_line_number, const char *p_filename);

	void handle_make_rid(RID &r_rid, RID_Data *p_data, RID_OwnerBase *p_owner);
	RID_Data *handle_get(const RID &p_rid) const;
	RID_Data *handle_getptr(const RID &p_rid) const;
	RID_Data *handle_get_or_null(const RID &p_rid) const;

	bool handle_is_owner(const RID &p_rid, const RID_OwnerBase *p_owner) const;
	void handle_free(const RID &p_rid);
};

extern RID_Database g_rid_database;

class RID_OwnerBase {
protected:
	bool _is_owner(const RID &p_rid) const {
		return g_rid_database.handle_is_owner(p_rid, this);
	}

	void _rid_print(const char *pszType, String sz, const RID &p_rid);

	const char *_typename = nullptr;
	bool _shutdown = false;

public:
	virtual void get_owned_list(List<RID> *p_owned) = 0;
	const char *get_typename() const { return _typename; }

	static void init_rid();
	virtual ~RID_OwnerBase();
};

template <class T>
class RID_Owner : public RID_OwnerBase {
public:
	RID make_rid(T *p_data) {
		RID rid;
		g_rid_database.handle_make_rid(rid, p_data, this);

#ifdef RID_HANDLE_PRINT_LIFETIMES
		_rid_print(_typename, "make_rid", rid);
#endif
		return rid;
	}

	T *get(const RID &p_rid) {
		return static_cast<T *>(g_rid_database.handle_get(p_rid));
	}

	T *getornull(const RID &p_rid) {
		return static_cast<T *>(g_rid_database.handle_get_or_null(p_rid));
	}

	T *getptr(const RID &p_rid) {
		return static_cast<T *>(g_rid_database.handle_getptr(p_rid));
	}

	bool owns(const RID &p_rid) const {
		return _is_owner(p_rid);
	}

	void free(RID p_rid) {
#ifdef RID_HANDLE_PRINT_LIFETIMES
		_rid_print(_typename, "free_rid", p_rid);
#endif
		g_rid_database.handle_free(p_rid);
	}

	void get_owned_list(List<RID> *p_owned){
#ifdef DEBUG_ENABLED

#endif
	}

	RID_Owner() {
		_typename = typeid(T).name();
	}
};

#endif // RID_HANDLES_ENABLED

#endif // RID_HANDLE_H
