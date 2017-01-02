/*************************************************************************/
/*  dvector.h                                                            */
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
#ifndef DVECTOR_H
#define DVECTOR_H

#include "os/memory.h"


/**
	@author Juan Linietsky <reduzio@gmail.com>
*/


extern Mutex* dvector_lock;

template<class T>
class DVector {

	mutable MID mem;


	void copy_on_write() {

		if (!mem.is_valid())
			return;

		if (dvector_lock)
			dvector_lock->lock();

		MID_Lock lock( mem );


		if ( *(int*)lock.data()  == 1 ) {
			// one reference, means no refcount changes
			if (dvector_lock)
				dvector_lock->unlock();
			return;
		}

		MID new_mem= dynalloc( mem.get_size() );

		if (!new_mem.is_valid()) {

			if (dvector_lock)
				dvector_lock->unlock();
			ERR_FAIL_COND( new_mem.is_valid() ); // out of memory
		}

		MID_Lock dst_lock( new_mem );

		int *rc = (int*)dst_lock.data();

		*rc=1;

		T * dst = (T*)(rc + 1 );

		T * src =(T*) ((int*)lock.data() + 1 );

		int count = (mem.get_size() - sizeof(int)) / sizeof(T);

		for (int i=0;i<count;i++) {

			memnew_placement( &dst[i], T(src[i]) );
		}

		(*(int*)lock.data())--;

		// unlock all
		dst_lock=MID_Lock();
		lock=MID_Lock();

		mem=new_mem;

		if (dvector_lock)
			dvector_lock->unlock();

	}

	void reference( const DVector& p_dvector ) {

		unreference();

		if (dvector_lock)
			dvector_lock->lock();

		if (!p_dvector.mem.is_valid()) {

			if (dvector_lock)
				dvector_lock->unlock();
			return;
		}

		MID_Lock lock(p_dvector.mem);

		int * rc = (int*)lock.data();
		(*rc)++;

		lock = MID_Lock();
		mem=p_dvector.mem;

		if (dvector_lock)
			dvector_lock->unlock();

	}


	void unreference() {

		if (dvector_lock)
			dvector_lock->lock();

		if (!mem.is_valid()) {

			if (dvector_lock)
				dvector_lock->unlock();
			return;
		}

		MID_Lock lock(mem);

		int * rc = (int*)lock.data();
		(*rc)--;

		if (*rc==0) {
			// no one else using it, destruct

			T * t= (T*)(rc+1);
			int count = (mem.get_size() - sizeof(int)) / sizeof(T);

			for (int i=0;i<count;i++) {

				t[i].~T();
			}

		}


		lock = MID_Lock();

		mem = MID ();

		if (dvector_lock)
			dvector_lock->unlock();

	}

public:

	class Read {
	friend class DVector;
		MID_Lock lock;
		const T * mem;
	public:

		_FORCE_INLINE_ const T& operator[](int p_index) const { return mem[p_index]; }
		_FORCE_INLINE_ const T *ptr() const { return mem; }

		Read() { mem=NULL; }
	};

	class Write {
	friend class DVector;
		MID_Lock lock;
		T * mem;
	public:

		_FORCE_INLINE_ T& operator[](int p_index) { return mem[p_index]; }
		_FORCE_INLINE_ T *ptr() { return mem; }

		Write() { mem=NULL; }
	};


	Read read() const {

		Read r;
		if (mem.is_valid()) {
			r.lock = MID_Lock( mem );
			r.mem = (const T*)((int*)r.lock.data()+1);
		}
		return r;
	}
	Write write() {

		Write w;
		if (mem.is_valid()) {
			copy_on_write();
			w.lock = MID_Lock( mem );
			w.mem = (T*)((int*)w.lock.data()+1);
		}
		return w;
	}

	template<class MC>
	void fill_with(const MC& p_mc) {


		int c=p_mc.size();
		resize(c);
		Write w=write();
		int idx=0;
		for(const typename MC::Element *E=p_mc.front();E;E=E->next()) {

			w[idx++]=E->get();
		}
	}


	void remove(int p_index) {

		int s = size();
		ERR_FAIL_INDEX(p_index, s);
		Write w = write();
		for (int i=p_index; i<s-1; i++) {

			w[i]=w[i+1];
		};
		w = Write();
		resize(s-1);
	}

	inline int size() const;
	T get(int p_index) const;
	void set(int p_index, const T& p_val);
	void push_back(const T& p_val);
	void append(const T& p_val) { push_back(p_val); }
	void append_array(const DVector<T>& p_arr) {
		int ds = p_arr.size();
		if (ds==0)
			return;
		int bs = size();
		resize( bs + ds);
		Write w = write();
		Read r = p_arr.read();
		for(int i=0;i<ds;i++)
			w[bs+i]=r[i];
	}

	DVector<T> subarray(int p_from, int p_to) {

		if (p_from<0) {
			p_from=size()+p_from;
		}
		if (p_to<0) {
			p_to=size()+p_to;
		}
		if (p_from<0 || p_from>=size()) {
			DVector<T>& aux=*((DVector<T>*)0); // nullreturn
			ERR_FAIL_COND_V(p_from<0 || p_from>=size(),aux)
		}
		if (p_to<0 || p_to>=size()) {
			DVector<T>& aux=*((DVector<T>*)0); // nullreturn
			ERR_FAIL_COND_V(p_to<0 || p_to>=size(),aux)
		}

		DVector<T> slice;
		int span=1 + p_to - p_from;
		slice.resize(span);
		Read r = read();
		Write w = slice.write();
		for (int i=0; i<span; ++i) {
			w[i] = r[p_from+i];
		}

		return slice;
	}

	Error insert(int p_pos,const T& p_val) {

		int s=size();
		ERR_FAIL_INDEX_V(p_pos,s+1,ERR_INVALID_PARAMETER);
		resize(s+1);
		{
			Write w = write();
			for (int i=s;i>p_pos;i--)
				w[i]=w[i-1];
			w[p_pos]=p_val;
		}

		return OK;
	}


	bool is_locked() const { return mem.is_locked(); }

	inline const T operator[](int p_index) const;

	Error resize(int p_size);

	void invert();

	void operator=(const DVector& p_dvector) { reference(p_dvector); }
	DVector() {}
	DVector(const DVector& p_dvector) { reference(p_dvector); }
	~DVector() { unreference(); }

};

template<class T>
int DVector<T>::size() const {

	return mem.is_valid() ? ((mem.get_size() - sizeof(int)) / sizeof(T) ) : 0;
}

template<class T>
T DVector<T>::get(int p_index) const {

	return operator[](p_index);
}

template<class T>
void DVector<T>::set(int p_index, const T& p_val) {

	if (p_index<0 || p_index>=size()) {
		ERR_FAIL_COND(p_index<0 || p_index>=size());
	}

	Write w = write();
	w[p_index]=p_val;
}

template<class T>
void DVector<T>::push_back(const T& p_val) {

	resize( size() + 1 );
	set( size() -1, p_val );
}

template<class T>
const T DVector<T>::operator[](int p_index) const {

	if (p_index<0 || p_index>=size()) {
		T& aux=*((T*)0); //nullreturn
		ERR_FAIL_COND_V(p_index<0 || p_index>=size(),aux);
	}

	Read r = read();

	return r[p_index];
}


template<class T>
Error DVector<T>::resize(int p_size) {

	if (dvector_lock)
		dvector_lock->lock();

	bool same = p_size==size();

	if (dvector_lock)
		dvector_lock->unlock();
	// no further locking is necesary because we are supposed to own the only copy of this (using copy on write)

	if (same)
		return OK;

	if (p_size == 0 ) {

		unreference();
		return OK;
	}


	copy_on_write(); // make it unique

	ERR_FAIL_COND_V( mem.is_locked(), ERR_LOCKED ); // if after copy on write, memory is locked, fail.

	if (p_size > size() ) {

		int oldsize=size();

		MID_Lock lock;

		if (oldsize==0) {

			mem = dynalloc( p_size * sizeof(T) + sizeof(int) );
			lock=MID_Lock(mem);
			int *rc = ((int*)lock.data());
			*rc=1;

		} else {

			if (dynrealloc( mem, p_size * sizeof(T) + sizeof(int) )!=OK ) {

				ERR_FAIL_V(ERR_OUT_OF_MEMORY); // out of memory
			}

			lock=MID_Lock(mem);
		}




		T *t = (T*)((int*)lock.data() + 1);

		for (int i=oldsize;i<p_size;i++) {

			memnew_placement(&t[i], T );
		}

		lock = MID_Lock(); // clear
	} else {

		int oldsize=size();

		MID_Lock lock(mem);


		T *t = (T*)((int*)lock.data() + 1);

		for (int i=p_size;i<oldsize;i++) {

			t[i].~T();
		}

		lock = MID_Lock(); // clear

		if (dynrealloc( mem, p_size * sizeof(T) + sizeof(int) )!=OK ) {

			ERR_FAIL_V(ERR_OUT_OF_MEMORY); // wtf error
		}


	}

	return OK;
}

template<class T>
void DVector<T>::invert() {
	T temp;
	Write w = write();
	int s = size();
	int half_s = s/2;

	for(int i=0;i<half_s;i++) {
		temp = w[i];
		w[i] = w[s-i-1];
		w[s-i-1] = temp;
	}
}

#endif
