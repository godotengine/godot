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
#include "os/memory_pool_static.h"


/**
	@author Juan Linietsky <reduzio@gmail.com>
*/

class MID {

	struct Data {

		SafeRefCount refcount;
		MemoryPoolDynamic::ID id;
	};

	mutable Data *data;

	void unref() {

		if (!data)
			return;
		if (data->refcount.unref()) {

			if (data->id!=MemoryPoolDynamic::INVALID_ID)
				MemoryPoolDynamic::get_singleton()->free(data->id);
			MemoryPoolStatic::get_singleton()->free(data);
		}

		data=NULL;
	}

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

	Error _resize(size_t p_size) {

		if (p_size==0 && (!data || data->id==MemoryPoolDynamic::INVALID_ID))
				return OK;
		if (p_size && !data) {
			// create data because we'll need it
			data = (Data*)MemoryPoolStatic::get_singleton()->alloc(sizeof(Data),"MID::Data");
			ERR_FAIL_COND_V( !data,ERR_OUT_OF_MEMORY );
			data->refcount.init();
			data->id=MemoryPoolDynamic::INVALID_ID;
		}

		if (p_size==0 && data && data->id==MemoryPoolDynamic::INVALID_ID) {

			MemoryPoolDynamic::get_singleton()->free(data->id);
			data->id=MemoryPoolDynamic::INVALID_ID;
		}

		if (p_size>0) {

		 	if (data->id==MemoryPoolDynamic::INVALID_ID) {

				data->id=MemoryPoolDynamic::get_singleton()->alloc(p_size,"Unnamed MID");
				ERR_FAIL_COND_V( data->id==MemoryPoolDynamic::INVALID_ID, ERR_OUT_OF_MEMORY );

			} else {

				MemoryPoolDynamic::get_singleton()->realloc(data->id,p_size);
				ERR_FAIL_COND_V( data->id==MemoryPoolDynamic::INVALID_ID, ERR_OUT_OF_MEMORY );

			}
		}

		return OK;
	}
friend class Memory;

	MID(MemoryPoolDynamic::ID p_id) {

		data = (Data*)MemoryPoolStatic::get_singleton()->alloc(sizeof(Data),"MID::Data");
		data->refcount.init();
		data->id=p_id;
	}
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
public:

	static void * alloc_static(size_t p_bytes,const char *p_descr="");
	static void * realloc_static(void *p_memory,size_t p_bytes);
	static void free_static(void *p_ptr);
	static size_t get_static_mem_available();
	static size_t get_static_mem_usage();
	static size_t get_static_mem_max_usage();
	static void dump_static_mem_to_file(const char* p_file);

	static MID alloc_dynamic(size_t p_bytes, const char *p_descr="");
	static Error realloc_dynamic(MID p_mid,size_t p_bytes);

	static size_t get_dynamic_mem_available();
	static size_t get_dynamic_mem_usage();

};

template<class T>
struct MemAalign {
	static _FORCE_INLINE_ int get_align() { return DEFAULT_ALIGNMENT; }
};

class DefaultAllocator {
public:
	_FORCE_INLINE_ static void *alloc(size_t p_memory) { return Memory::alloc_static(p_memory, ""); }
	_FORCE_INLINE_ static void free(void *p_ptr) { return Memory::free_static(p_ptr); }

};


void * operator new(size_t p_size,const char *p_description); ///< operator new that takes a description and uses MemoryStaticPool
void * operator new(size_t p_size,void* (*p_allocfunc)(size_t p_size)); ///< operator new that takes a description and uses MemoryStaticPool

void * operator new(size_t p_size,void *p_pointer,size_t check, const char *p_description); ///< operator new that takes a description and uses a pointer to the preallocated memory

#ifdef DEBUG_MEMORY_ENABLED

#define memalloc(m_size) Memory::alloc_static(m_size, __FILE__ ":" __STR(__LINE__) ", memalloc.")
#define memrealloc(m_mem,m_size) Memory::realloc_static(m_mem,m_size)
#define memfree(m_size) Memory::free_static(m_size)

#else

#define memalloc(m_size) Memory::alloc_static(m_size)
#define memrealloc(m_mem,m_size) Memory::realloc_static(m_mem,m_size)
#define memfree(m_size) Memory::free_static(m_size)

#endif

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

#ifdef DEBUG_MEMORY_ENABLED

#define memnew(m_class) _post_initialize(new(__FILE__ ":" __STR(__LINE__) ", memnew type: " __STR(m_class)) m_class)

#else

#define memnew(m_class) _post_initialize(new("") m_class)

#endif

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
	Memory::free_static(p_class);
}

template<class T,class A>
void memdelete_allocator(T *p_class) {

	if (!predelete_handler(p_class))
		return; // doesn't want to be deleted
	p_class->~T();
	A::free(p_class);
}

#define memdelete_notnull(m_v) { if (m_v) memdelete(m_v); }
#ifdef DEBUG_MEMORY_ENABLED

#define memnew_arr( m_class, m_count ) memnew_arr_template<m_class>(m_count,__FILE__ ":" __STR(__LINE__) ", memnew_arr type: " _STR(m_class))

#else

#define memnew_arr( m_class, m_count ) memnew_arr_template<m_class>(m_count)

#endif

template<typename T>
T* memnew_arr_template(size_t p_elements,const char *p_descr="") {

	if (p_elements==0)
		return 0;
	/** overloading operator new[] cannot be done , because it may not return the real allocated address (it may pad the 'element count' before the actual array). Because of that, it must be done by hand. This is the
	same strategy used by std::vector, and the DVector class, so it should be safe.*/

	size_t len = sizeof(T) * p_elements;
	unsigned int *mem = (unsigned int*)Memory::alloc_static( len + MAX(sizeof(size_t), DEFAULT_ALIGNMENT), p_descr );
	T *failptr=0; //get rid of a warning
	ERR_FAIL_COND_V( !mem, failptr );
	*mem=p_elements;
	mem = (unsigned int *)( ((uint8_t*)mem) + MAX(sizeof(size_t), DEFAULT_ALIGNMENT));
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

	uint8_t* ptr = ((uint8_t*)p_class) - MAX(sizeof(size_t), DEFAULT_ALIGNMENT);
	return *(size_t*)ptr;
}

template<typename T>
void memdelete_arr(T *p_class) {

	unsigned int * elems = (unsigned int*)(((uint8_t*)p_class) - MAX(sizeof(size_t), DEFAULT_ALIGNMENT));

	for (unsigned int i=0;i<*elems;i++) {

		p_class[i].~T();
	};
	Memory::free_static(elems);
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
