/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2006 Erwin Coumans  http://continuousphysics.com/Bullet/

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#ifndef BT_OBJECT_ARRAY__
#define BT_OBJECT_ARRAY__

#include "btScalar.h"  // has definitions like SIMD_FORCE_INLINE
#include "btAlignedAllocator.h"

///If the platform doesn't support placement new, you can disable BT_USE_PLACEMENT_NEW
///then the btAlignedObjectArray doesn't support objects with virtual methods, and non-trivial constructors/destructors
///You can enable BT_USE_MEMCPY, then swapping elements in the array will use memcpy instead of operator=
///see discussion here: http://continuousphysics.com/Bullet/phpBB2/viewtopic.php?t=1231 and
///http://www.continuousphysics.com/Bullet/phpBB2/viewtopic.php?t=1240

#define BT_USE_PLACEMENT_NEW 1
//#define BT_USE_MEMCPY 1 //disable, because it is cumbersome to find out for each platform where memcpy is defined. It can be in <memory.h> or <string.h> or otherwise...
#define BT_ALLOW_ARRAY_COPY_OPERATOR  // enabling this can accidently perform deep copies of data if you are not careful

#ifdef BT_USE_MEMCPY
#include <memory.h>
#include <string.h>
#endif  //BT_USE_MEMCPY

#ifdef BT_USE_PLACEMENT_NEW
#include <new>  //for placement new
#endif          //BT_USE_PLACEMENT_NEW

// The register keyword is deprecated in C++11 so don't use it.
#if __cplusplus > 199711L
#define BT_REGISTER
#else
#define BT_REGISTER register
#endif

///The btAlignedObjectArray template class uses a subset of the stl::vector interface for its methods
///It is developed to replace stl::vector to avoid portability issues, including STL alignment issues to add SIMD/SSE data
template <typename T>
//template <class T>
class btAlignedObjectArray
{
	btAlignedAllocator<T, 16> m_allocator;

	int m_size;
	int m_capacity;
	T* m_data;
	//PCK: added this line
	bool m_ownsMemory;

#ifdef BT_ALLOW_ARRAY_COPY_OPERATOR
public:
	SIMD_FORCE_INLINE btAlignedObjectArray<T>& operator=(const btAlignedObjectArray<T>& other)
	{
		copyFromArray(other);
		return *this;
	}
#else   //BT_ALLOW_ARRAY_COPY_OPERATOR
private:
	SIMD_FORCE_INLINE btAlignedObjectArray<T>& operator=(const btAlignedObjectArray<T>& other);
#endif  //BT_ALLOW_ARRAY_COPY_OPERATOR

protected:
	SIMD_FORCE_INLINE int allocSize(int size)
	{
		return (size ? size * 2 : 1);
	}
	SIMD_FORCE_INLINE void copy(int start, int end, T* dest) const
	{
		int i;
		for (i = start; i < end; ++i)
#ifdef BT_USE_PLACEMENT_NEW
			new (&dest[i]) T(m_data[i]);
#else
			dest[i] = m_data[i];
#endif  //BT_USE_PLACEMENT_NEW
	}

	SIMD_FORCE_INLINE void init()
	{
		//PCK: added this line
		m_ownsMemory = true;
		m_data = 0;
		m_size = 0;
		m_capacity = 0;
	}
	SIMD_FORCE_INLINE void destroy(int first, int last)
	{
		int i;
		for (i = first; i < last; i++)
		{
			m_data[i].~T();
		}
	}

	SIMD_FORCE_INLINE void* allocate(int size)
	{
		if (size)
			return m_allocator.allocate(size);
		return 0;
	}

	SIMD_FORCE_INLINE void deallocate()
	{
		if (m_data)
		{
			//PCK: enclosed the deallocation in this block
			if (m_ownsMemory)
			{
				m_allocator.deallocate(m_data);
			}
			m_data = 0;
		}
	}

public:
	btAlignedObjectArray()
	{
		init();
	}

	~btAlignedObjectArray()
	{
		clear();
	}

	///Generally it is best to avoid using the copy constructor of an btAlignedObjectArray, and use a (const) reference to the array instead.
	btAlignedObjectArray(const btAlignedObjectArray& otherArray)
	{
		init();

		int otherSize = otherArray.size();
		resize(otherSize);
		otherArray.copy(0, otherSize, m_data);
	}

	/// return the number of elements in the array
	SIMD_FORCE_INLINE int size() const
	{
		return m_size;
	}

	SIMD_FORCE_INLINE const T& at(int n) const
	{
		btAssert(n >= 0);
		btAssert(n < size());
		return m_data[n];
	}

	SIMD_FORCE_INLINE T& at(int n)
	{
		btAssert(n >= 0);
		btAssert(n < size());
		return m_data[n];
	}

	SIMD_FORCE_INLINE const T& operator[](int n) const
	{
		btAssert(n >= 0);
		btAssert(n < size());
		return m_data[n];
	}

	SIMD_FORCE_INLINE T& operator[](int n)
	{
		btAssert(n >= 0);
		btAssert(n < size());
		return m_data[n];
	}

	///clear the array, deallocated memory. Generally it is better to use array.resize(0), to reduce performance overhead of run-time memory (de)allocations.
	SIMD_FORCE_INLINE void clear()
	{
		destroy(0, size());

		deallocate();

		init();
	}

	SIMD_FORCE_INLINE void pop_back()
	{
		btAssert(m_size > 0);
		m_size--;
		m_data[m_size].~T();
	}

	///resize changes the number of elements in the array. If the new size is larger, the new elements will be constructed using the optional second argument.
	///when the new number of elements is smaller, the destructor will be called, but memory will not be freed, to reduce performance overhead of run-time memory (de)allocations.
	SIMD_FORCE_INLINE void resizeNoInitialize(int newsize)
	{
		if (newsize > size())
		{
			reserve(newsize);
		}
		m_size = newsize;
	}

	SIMD_FORCE_INLINE void resize(int newsize, const T& fillData = T())
	{
		const BT_REGISTER int curSize = size();

		if (newsize < curSize)
		{
			for (int i = newsize; i < curSize; i++)
			{
				m_data[i].~T();
			}
		}
		else
		{
			if (newsize > curSize)
			{
				reserve(newsize);
			}
#ifdef BT_USE_PLACEMENT_NEW
			for (int i = curSize; i < newsize; i++)
			{
				new (&m_data[i]) T(fillData);
			}
#endif  //BT_USE_PLACEMENT_NEW
		}

		m_size = newsize;
	}
	SIMD_FORCE_INLINE T& expandNonInitializing()
	{
		const BT_REGISTER int sz = size();
		if (sz == capacity())
		{
			reserve(allocSize(size()));
		}
		m_size++;

		return m_data[sz];
	}

	SIMD_FORCE_INLINE T& expand(const T& fillValue = T())
	{
		const BT_REGISTER int sz = size();
		if (sz == capacity())
		{
			reserve(allocSize(size()));
		}
		m_size++;
#ifdef BT_USE_PLACEMENT_NEW
		new (&m_data[sz]) T(fillValue);  //use the in-place new (not really allocating heap memory)
#endif

		return m_data[sz];
	}

	SIMD_FORCE_INLINE void push_back(const T& _Val)
	{
		const BT_REGISTER int sz = size();
		if (sz == capacity())
		{
			reserve(allocSize(size()));
		}

#ifdef BT_USE_PLACEMENT_NEW
		new (&m_data[m_size]) T(_Val);
#else
		m_data[size()] = _Val;
#endif  //BT_USE_PLACEMENT_NEW

		m_size++;
	}

	/// return the pre-allocated (reserved) elements, this is at least as large as the total number of elements,see size() and reserve()
	SIMD_FORCE_INLINE int capacity() const
	{
		return m_capacity;
	}

	SIMD_FORCE_INLINE void reserve(int _Count)
	{  // determine new minimum length of allocated storage
		if (capacity() < _Count)
		{  // not enough room, reallocate
			T* s = (T*)allocate(_Count);

			copy(0, size(), s);

			destroy(0, size());

			deallocate();

			//PCK: added this line
			m_ownsMemory = true;

			m_data = s;

			m_capacity = _Count;
		}
	}

	class less
	{
	public:
		bool operator()(const T& a, const T& b) const
		{
			return (a < b);
		}
	};

	template <typename L>
	void quickSortInternal(const L& CompareFunc, int lo, int hi)
	{
		//  lo is the lower index, hi is the upper index
		//  of the region of array a that is to be sorted
		int i = lo, j = hi;
		T x = m_data[(lo + hi) / 2];

		//  partition
		do
		{
			while (CompareFunc(m_data[i], x))
				i++;
			while (CompareFunc(x, m_data[j]))
				j--;
			if (i <= j)
			{
				swap(i, j);
				i++;
				j--;
			}
		} while (i <= j);

		//  recursion
		if (lo < j)
			quickSortInternal(CompareFunc, lo, j);
		if (i < hi)
			quickSortInternal(CompareFunc, i, hi);
	}

	template <typename L>
	void quickSort(const L& CompareFunc)
	{
		//don't sort 0 or 1 elements
		if (size() > 1)
		{
			quickSortInternal(CompareFunc, 0, size() - 1);
		}
	}

	///heap sort from http://www.csse.monash.edu.au/~lloyd/tildeAlgDS/Sort/Heap/
	template <typename L>
	void downHeap(T* pArr, int k, int n, const L& CompareFunc)
	{
		/*  PRE: a[k+1..N] is a heap */
		/* POST:  a[k..N]  is a heap */

		T temp = pArr[k - 1];
		/* k has child(s) */
		while (k <= n / 2)
		{
			int child = 2 * k;

			if ((child < n) && CompareFunc(pArr[child - 1], pArr[child]))
			{
				child++;
			}
			/* pick larger child */
			if (CompareFunc(temp, pArr[child - 1]))
			{
				/* move child up */
				pArr[k - 1] = pArr[child - 1];
				k = child;
			}
			else
			{
				break;
			}
		}
		pArr[k - 1] = temp;
	} /*downHeap*/

	void swap(int index0, int index1)
	{
#ifdef BT_USE_MEMCPY
		char temp[sizeof(T)];
		memcpy(temp, &m_data[index0], sizeof(T));
		memcpy(&m_data[index0], &m_data[index1], sizeof(T));
		memcpy(&m_data[index1], temp, sizeof(T));
#else
		T temp = m_data[index0];
		m_data[index0] = m_data[index1];
		m_data[index1] = temp;
#endif  //BT_USE_PLACEMENT_NEW
	}

	template <typename L>
	void heapSort(const L& CompareFunc)
	{
		/* sort a[0..N-1],  N.B. 0 to N-1 */
		int k;
		int n = m_size;
		for (k = n / 2; k > 0; k--)
		{
			downHeap(m_data, k, n, CompareFunc);
		}

		/* a[1..N] is now a heap */
		while (n >= 1)
		{
			swap(0, n - 1); /* largest of a[0..n-1] */

			n = n - 1;
			/* restore a[1..i-1] heap */
			downHeap(m_data, 1, n, CompareFunc);
		}
	}

	///non-recursive binary search, assumes sorted array
	int findBinarySearch(const T& key) const
	{
		int first = 0;
		int last = size() - 1;

		//assume sorted array
		while (first <= last)
		{
			int mid = (first + last) / 2;  // compute mid point.
			if (key > m_data[mid])
				first = mid + 1;  // repeat search in top half.
			else if (key < m_data[mid])
				last = mid - 1;  // repeat search in bottom half.
			else
				return mid;  // found it. return position /////
		}
		return size();  // failed to find key
	}

	int findLinearSearch(const T& key) const
	{
		int index = size();
		int i;

		for (i = 0; i < size(); i++)
		{
			if (m_data[i] == key)
			{
				index = i;
				break;
			}
		}
		return index;
	}

	// If the key is not in the array, return -1 instead of 0,
	// since 0 also means the first element in the array.
	int findLinearSearch2(const T& key) const
	{
		int index = -1;
		int i;

		for (i = 0; i < size(); i++)
		{
			if (m_data[i] == key)
			{
				index = i;
				break;
			}
		}
		return index;
	}

	void removeAtIndex(int index)
	{
		if (index < size())
		{
			swap(index, size() - 1);
			pop_back();
		}
	}
	void remove(const T& key)
	{
		int findIndex = findLinearSearch(key);
		removeAtIndex(findIndex);
	}

	//PCK: whole function
	void initializeFromBuffer(void* buffer, int size, int capacity)
	{
		clear();
		m_ownsMemory = false;
		m_data = (T*)buffer;
		m_size = size;
		m_capacity = capacity;
	}

	void copyFromArray(const btAlignedObjectArray& otherArray)
	{
		int otherSize = otherArray.size();
		resize(otherSize);
		otherArray.copy(0, otherSize, m_data);
	}
};

#endif  //BT_OBJECT_ARRAY__
