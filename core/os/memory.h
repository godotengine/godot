/*************************************************************************/
/*  memory.h                                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
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
#ifndef MEMORY_H
#define MEMORY_H

#include <stddef.h>
#include "safe_refcount.h"
#include "os/memory_pool_dynamic.h"



/**
	@author Juan Linietsky <reduzio@gmail.com>
*/

#ifndef PAD_ALIGN
#define PAD_ALIGN 16 //must always be greater than this at much
#endif


class MID {

	struct Data {

		SafeRefCount refcount;
		MemoryPoolDynamic::ID id;
	};

	mutable Data *data;



	void ref(Data *p_data) {

		if (data==p_data)
			return;
		unref();

		if (p_data && p_data->refcount.ref())
			data=p_data;
	}

friend class MID_Lock;

	inline void lock() {

		if (data && data->id!=MemoryPoolDynamic::INVALID_ID)
			MemoryPoolDynamic::get_singleton()->lock(data->id);
	}
	inline void unlock() {

		if (data && data->id!=MemoryPoolDynamic::INVALID_ID)
			MemoryPoolDynamic::get_singleton()->unlock(data->id);

	}

	inline void * get() {

		if (data && data->id!=MemoryPoolDynamic::INVALID_ID)
			return MemoryPoolDynamic::get_singleton()->get(data->id);

		return NULL;
	}


	void unref();
	Error _resize(size_t p_size);

friend class Memory;

	MID(MemoryPoolDynamic::ID p_id);
public:

	bool is_valid() const { return data; }
	operator bool() const { return data; }


	size_t get_size() const { return (data && data->id!=MemoryPoolDynamic::INVALID_ID) ? MemoryPoolDynamic::get_singleton()->get_size(data->id) : 0; }
	Error resize(size_t p_size) { return _resize(p_size); }
	inline void operator=(const MID& p_mid) { ref( p_mid.data ); }
	inline bool is_locked() const { return (data && data->id!=MemoryPoolDynamic::INVALID_ID) ? MemoryPoolDynamic::get_singleton()->is_locked(data->id) : false; }
	inline MID(const MID& p_mid) { data=NULL; ref( p_mid.data ); }
	inline MID() { data = NULL; }
	~MID() { unref(); }
};


class MID_Lock {

	MID mid;

public:

	void *data() { return mid.get(); }

	void operator=(const MID_Lock& p_lock ) { mid.unlock(); mid = p_lock.mid; mid.lock(); }
	inline MID_Lock(const MID& p_mid) { mid=p_mid; mid.lock(); }
	inline MID_Lock(const MID_Lock& p_lock) { mid=p_lock.mid; mid.lock(); }
	MID_Lock() {}
	~MID_Lock() { mid.unlock(); }
};


class Memory{

	Memory();
#ifdef DEBUG_ENABLED
	static size_t mem_usage;
	static size_t max_usage;
#endif

	static size_t alloc_count;

public:

	static void * alloc_static(size_t p_bytes,bool p_pad_align=false);
	static void * realloc_static(void *p_memory,size_t p_bytes,bool p_pad_align=false);
	static void free_static(void *p_ptr,bool p_pad_align=false);

	static size_t get_mem_available();
	static size_t get_mem_usage();
	static size_t get_mem_max_usage();


	static MID alloc_dynamic(size_t p_bytes, const char *p_descr="");
	static Error realloc_dynamic(MID p_mid,size_t p_bytes);

	static size_t get_dynamic_mem_available();
	static size_t get_dynamic_mem_usage();

};

class DefaultAllocator {
public:
	_FORCE_INLINE_ static void *alloc(size_t p_memory) { return Memory::alloc_static(p_memory, false); }
	_FORCE_INLINE_ static void free(void *p_ptr) { return Memory::free_static(p_ptr,false); }

};


void * operator new(size_t p_size,const char *p_description); ///< operator new that takes a description and uses MemoryStaticPool
void * operator new(size_t p_size,void* (*p_allocfunc)(size_t p_size)); ///< operator new that takes a description and uses MemoryStaticPool

void * operator new(size_t p_size,void *p_pointer,size_t check, const char *p_description); ///< operator new that takes a description and uses a pointer to the preallocated memory

#define memalloc(m_size) Memory::alloc_static(m_size)
#define memrealloc(m_mem,m_size) Memory::realloc_static(m_mem,m_size)
#define memfree(m_size) Memory::free_static(m_size)


#ifdef DEBUG_MEMORY_ENABLED
#define dynalloc(m_size) Memory::alloc_dynamic(m_size, __FILE__ ":" __STR(__LINE__) ", type: DYNAMIC")
#define dynrealloc(m_mem,m_size) m_mem.resize(m_size)

#else

#define dynalloc(m_size) Memory::alloc_dynamic(m_size)
#define dynrealloc(m_mem,m_size) m_mem.resize(m_size)

#endif


_ALWAYS_INLINE_ void postinitialize_handler(void *) {}


template<class T>
_ALWAYS_INLINE_ T *_post_initialize(T *p_obj) {

	postinitialize_handler(p_obj);
	return p_obj;
}

#define memnew(m_class) _post_initialize(new("") m_class)

_ALWAYS_INLINE_ void * operator new(size_t p_size,void *p_pointer,size_t check, const char *p_description) {
//	void *failptr=0;
//	ERR_FAIL_COND_V( check < p_size , failptr); /** bug, or strange compiler, most likely */

	return p_pointer;
}


#define memnew_allocator(m_class,m_allocator) _post_initialize(new(m_allocator::alloc) m_class)
#define memnew_placement(m_placement,m_class) _post_initialize(new(m_placement,sizeof(m_class),"") m_class)


_ALWAYS_INLINE_ bool predelete_handler(void *) { return true; }

template<class T>
void memdelete(T *p_class) {

	if (!predelete_handler(p_class))
		return; // doesn't want to be deleted
	p_class->~T();
	Memory::free_static(p_class,false);
}

template<class T,class A>
void memdelete_allocator(T *p_class) {

	if (!predelete_handler(p_class))
		return; // doesn't want to be deleted
	p_class->~T();
	A::free(p_class);
}

#define memdelete_notnull(m_v) { if (m_v) memdelete(m_v); }

#define memnew_arr( m_class, m_count ) memnew_arr_template<m_class>(m_count)


template<typename T>
T* memnew_arr_template(size_t p_elements,const char *p_descr="") {

	if (p_elements==0)
		return 0;
	/** overloading operator new[] cannot be done , because it may not return the real allocated address (it may pad the 'element count' before the actual array). Because of that, it must be done by hand. This is the
	same strategy used by std::vector, and the DVector class, so it should be safe.*/

	size_t len = sizeof(T) * p_elements;
	uint64_t *mem = (uint64_t*)Memory::alloc_static( len , true );
	T *failptr=0; //get rid of a warning
	ERR_FAIL_COND_V( !mem, failptr );
	*(mem-1)=p_elements;

	T* elems = (T*)mem;

	/* call operator new */
	for (size_t i=0;i<p_elements;i++) {
		new(&elems[i],sizeof(T),p_descr) T;
	}

	return (T*)mem;
}

/**
 * Wonders of having own array functions, you can actually check the length of
 * an allocated-with memnew_arr() array
 */

template<typename T>
size_t memarr_len(const T *p_class) {

	uint64_t* ptr = (uint64_t*)p_class;
	return *(ptr-1);
}

template<typename T>
void memdelete_arr(T *p_class) {

	uint64_t* ptr = (uint64_t*)p_class;

	uint64_t elem_count = *(ptr-1);

	for (uint64_t i=0;i<elem_count;i++) {

		p_class[i].~T();
	};
	Memory::free_static(ptr,true);
}


struct _GlobalNil {

	int color;
	_GlobalNil *right;
	_GlobalNil *left;
	_GlobalNil *parent;

	_GlobalNil();

};

struct _GlobalNilClass {

	static _GlobalNil _nil;
};



#endif
