/*************************************************************************/
/*  rid.h                                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifndef RID_H
#define RID_H

#include "hash_map.h"
#include "list.h"
#include "os/memory.h"
#include "safe_refcount.h"
#include "typedefs.h"

/**
	@author Juan Linietsky <reduzio@gmail.com>
*/

class RID_OwnerBase;

typedef uint32_t ID;

class RID {
	friend class RID_OwnerBase;
	ID _id;
	RID_OwnerBase *owner;

public:
	_FORCE_INLINE_ ID get_id() const { return _id; }
	bool operator==(const RID &p_rid) const {

		return _id == p_rid._id;
	}
	_FORCE_INLINE_ bool operator<(const RID &p_rid) const {

		return _id < p_rid._id;
	}
	_FORCE_INLINE_ bool operator<=(const RID &p_rid) const {

		return _id <= p_rid._id;
	}
	_FORCE_INLINE_ bool operator>(const RID &p_rid) const {

		return _id > p_rid._id;
	}
	bool operator!=(const RID &p_rid) const {

		return _id != p_rid._id;
	}
	_FORCE_INLINE_ bool is_valid() const { return _id > 0; }

	operator const void *() const {
		return is_valid() ? this : 0;
	};

	_FORCE_INLINE_ RID() {
		_id = 0;
		owner = 0;
	}
};

class RID_OwnerBase {
protected:
	friend class RID;
	void set_id(RID &p_rid, ID p_id) const { p_rid._id = p_id; }
	void set_ownage(RID &p_rid) const { p_rid.owner = const_cast<RID_OwnerBase *>(this); }
	ID new_ID();

public:
	virtual bool owns(const RID &p_rid) const = 0;
	virtual void get_owned_list(List<RID> *p_owned) const = 0;

	static void init_rid();

	virtual ~RID_OwnerBase() {}
};

template <class T, bool thread_safe = false>
class RID_Owner : public RID_OwnerBase {
public:
	typedef void (*ReleaseNotifyFunc)(void *user, T *p_data);

private:
	Mutex *mutex;
	mutable HashMap<ID, T *> id_map;

public:
	RID make_rid(T *p_data) {

		if (thread_safe) {
			mutex->lock();
		}

		ID id = new_ID();
		id_map[id] = p_data;
		RID rid;
		set_id(rid, id);
		set_ownage(rid);

		if (thread_safe) {
			mutex->unlock();
		}

		return rid;
	}

	_FORCE_INLINE_ T *get(const RID &p_rid) {

		if (thread_safe) {
			mutex->lock();
		}

		T **elem = id_map.getptr(p_rid.get_id());

		if (thread_safe) {
			mutex->unlock();
		}

		ERR_FAIL_COND_V(!elem, NULL);

		return *elem;
	}

	virtual bool owns(const RID &p_rid) const {

		if (thread_safe) {
			mutex->lock();
		}

		T **elem = id_map.getptr(p_rid.get_id());

		if (thread_safe) {
			mutex->lock();
		}

		return elem != NULL;
	}

	virtual void free(RID p_rid) {

		if (thread_safe) {
			mutex->lock();
		}
		ERR_FAIL_COND(!owns(p_rid));
		id_map.erase(p_rid.get_id());
	}
	virtual void get_owned_list(List<RID> *p_owned) const {

		if (thread_safe) {
			mutex->lock();
		}

		const ID *id = NULL;
		while ((id = id_map.next(id))) {

			RID rid;
			set_id(rid, *id);
			set_ownage(rid);
			p_owned->push_back(rid);
		}

		if (thread_safe) {
			mutex->lock();
		}
	}
	RID_Owner() {

		if (thread_safe) {

			mutex = Mutex::create();
		}
	}

	~RID_Owner() {

		if (thread_safe) {

			memdelete(mutex);
		}
	}
};

#endif
