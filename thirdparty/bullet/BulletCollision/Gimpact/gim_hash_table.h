#ifndef GIM_HASH_TABLE_H_INCLUDED
#define GIM_HASH_TABLE_H_INCLUDED
/*! \file gim_trimesh_data.h
\author Francisco Leon Najera
*/
/*
-----------------------------------------------------------------------------
This source file is part of GIMPACT Library.

For the latest info, see http://gimpact.sourceforge.net/

Copyright (c) 2006 Francisco Leon Najera. C.C. 80087371.
email: projectileman@yahoo.com

 This library is free software; you can redistribute it and/or
 modify it under the terms of EITHER:
   (1) The GNU Lesser General Public License as published by the Free
       Software Foundation; either version 2.1 of the License, or (at
       your option) any later version. The text of the GNU Lesser
       General Public License is included with this library in the
       file GIMPACT-LICENSE-LGPL.TXT.
   (2) The BSD-style license that is included with this library in
       the file GIMPACT-LICENSE-BSD.TXT.
   (3) The zlib/libpng license that is included with this library in
       the file GIMPACT-LICENSE-ZLIB.TXT.

 This library is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the files
 GIMPACT-LICENSE-LGPL.TXT, GIMPACT-LICENSE-ZLIB.TXT and GIMPACT-LICENSE-BSD.TXT for more details.

-----------------------------------------------------------------------------
*/

#include "gim_radixsort.h"

#define GIM_INVALID_HASH 0xffffffff  //!< A very very high value
#define GIM_DEFAULT_HASH_TABLE_SIZE 380
#define GIM_DEFAULT_HASH_TABLE_NODE_SIZE 4
#define GIM_HASH_TABLE_GROW_FACTOR 2

#define GIM_MIN_RADIX_SORT_SIZE 860  //!< calibrated on a PIII

template <typename T>
struct GIM_HASH_TABLE_NODE
{
	GUINT m_key;
	T m_data;
	GIM_HASH_TABLE_NODE()
	{
	}

	GIM_HASH_TABLE_NODE(const GIM_HASH_TABLE_NODE& value)
	{
		m_key = value.m_key;
		m_data = value.m_data;
	}

	GIM_HASH_TABLE_NODE(GUINT key, const T& data)
	{
		m_key = key;
		m_data = data;
	}

	bool operator<(const GIM_HASH_TABLE_NODE<T>& other) const
	{
		///inverse order, further objects are first
		if (m_key < other.m_key) return true;
		return false;
	}

	bool operator>(const GIM_HASH_TABLE_NODE<T>& other) const
	{
		///inverse order, further objects are first
		if (m_key > other.m_key) return true;
		return false;
	}

	bool operator==(const GIM_HASH_TABLE_NODE<T>& other) const
	{
		///inverse order, further objects are first
		if (m_key == other.m_key) return true;
		return false;
	}
};

///Macro for getting the key
class GIM_HASH_NODE_GET_KEY
{
public:
	template <class T>
	inline GUINT operator()(const T& a)
	{
		return a.m_key;
	}
};

///Macro for comparing the key and the element
class GIM_HASH_NODE_CMP_KEY_MACRO
{
public:
	template <class T>
	inline int operator()(const T& a, GUINT key)
	{
		return ((int)(a.m_key - key));
	}
};

///Macro for comparing Hash nodes
class GIM_HASH_NODE_CMP_MACRO
{
public:
	template <class T>
	inline int operator()(const T& a, const T& b)
	{
		return ((int)(a.m_key - b.m_key));
	}
};

//! Sorting for hash table
/*!
switch automatically between quicksort and radixsort
*/
template <typename T>
void gim_sort_hash_node_array(T* array, GUINT array_count)
{
	if (array_count < GIM_MIN_RADIX_SORT_SIZE)
	{
		gim_heap_sort(array, array_count, GIM_HASH_NODE_CMP_MACRO());
	}
	else
	{
		memcopy_elements_func cmpfunc;
		gim_radix_sort(array, array_count, GIM_HASH_NODE_GET_KEY(), cmpfunc);
	}
}

// Note: assumes long is at least 32 bits.
#define GIM_NUM_PRIME 28

static const GUINT gim_prime_list[GIM_NUM_PRIME] =
	{
		53ul, 97ul, 193ul, 389ul, 769ul,
		1543ul, 3079ul, 6151ul, 12289ul, 24593ul,
		49157ul, 98317ul, 196613ul, 393241ul, 786433ul,
		1572869ul, 3145739ul, 6291469ul, 12582917ul, 25165843ul,
		50331653ul, 100663319ul, 201326611ul, 402653189ul, 805306457ul,
		1610612741ul, 3221225473ul, 4294967291ul};

inline GUINT gim_next_prime(GUINT number)
{
	//Find nearest upper prime
	GUINT result_ind = 0;
	gim_binary_search(gim_prime_list, 0, (GIM_NUM_PRIME - 2), number, result_ind);

	// inv: result_ind < 28
	return gim_prime_list[result_ind];
}

//! A compact hash table implementation
/*!
A memory aligned compact hash table that coud be treated as an array.
It could be a simple sorted array without the overhead of the hash key bucked, or could
be a formely hash table with an array of keys.
You can use switch_to_hashtable() and switch_to_sorted_array for saving space or increase speed.
</br>

<ul>
<li> if node_size = 0, then this container becomes a simple sorted array allocator. reserve_size is used for reserve memory in m_nodes.
When the array size reaches the size equivalent to 'min_hash_table_size', then it becomes a hash table by calling check_for_switching_to_hashtable.
<li> If node_size != 0, then this container becomes a hash table for ever
</ul>

*/
template <class T>
class gim_hash_table
{
protected:
	typedef GIM_HASH_TABLE_NODE<T> _node_type;

	//!The nodes
	//array< _node_type, SuperAllocator<_node_type> > m_nodes;
	gim_array<_node_type> m_nodes;
	//SuperBufferedArray< _node_type > m_nodes;
	bool m_sorted;

	///Hash table data management. The hash table has the indices to the corresponding m_nodes array
	GUINT* m_hash_table;  //!<
	GUINT m_table_size;   //!<
	GUINT m_node_size;    //!<
	GUINT m_min_hash_table_size;

	//! Returns the cell index
	inline GUINT _find_cell(GUINT hashkey)
	{
		_node_type* nodesptr = m_nodes.pointer();
		GUINT start_index = (hashkey % m_table_size) * m_node_size;
		GUINT end_index = start_index + m_node_size;

		while (start_index < end_index)
		{
			GUINT value = m_hash_table[start_index];
			if (value != GIM_INVALID_HASH)
			{
				if (nodesptr[value].m_key == hashkey) return start_index;
			}
			start_index++;
		}
		return GIM_INVALID_HASH;
	}

	//! Find the avaliable cell for the hashkey, and return an existing cell if it has the same hash key
	inline GUINT _find_avaliable_cell(GUINT hashkey)
	{
		_node_type* nodesptr = m_nodes.pointer();
		GUINT avaliable_index = GIM_INVALID_HASH;
		GUINT start_index = (hashkey % m_table_size) * m_node_size;
		GUINT end_index = start_index + m_node_size;

		while (start_index < end_index)
		{
			GUINT value = m_hash_table[start_index];
			if (value == GIM_INVALID_HASH)
			{
				if (avaliable_index == GIM_INVALID_HASH)
				{
					avaliable_index = start_index;
				}
			}
			else if (nodesptr[value].m_key == hashkey)
			{
				return start_index;
			}
			start_index++;
		}
		return avaliable_index;
	}

	//! reserves the memory for the hash table.
	/*!
    \pre hash table must be empty
    \post reserves the memory for the hash table, an initializes all elements to GIM_INVALID_HASH.
    */
	inline void _reserve_table_memory(GUINT newtablesize)
	{
		if (newtablesize == 0) return;
		if (m_node_size == 0) return;

		//Get a Prime size

		m_table_size = gim_next_prime(newtablesize);

		GUINT datasize = m_table_size * m_node_size;
		//Alloc the data buffer
		m_hash_table = (GUINT*)gim_alloc(datasize * sizeof(GUINT));
	}

	inline void _invalidate_keys()
	{
		GUINT datasize = m_table_size * m_node_size;
		for (GUINT i = 0; i < datasize; i++)
		{
			m_hash_table[i] = GIM_INVALID_HASH;  // invalidate keys
		}
	}

	//! Clear all memory for the hash table
	inline void _clear_table_memory()
	{
		if (m_hash_table == NULL) return;
		gim_free(m_hash_table);
		m_hash_table = NULL;
		m_table_size = 0;
	}

	//! Invalidates the keys (Assigning GIM_INVALID_HASH to all) Reorders the hash keys
	inline void _rehash()
	{
		_invalidate_keys();

		_node_type* nodesptr = m_nodes.pointer();
		for (GUINT i = 0; i < (GUINT)m_nodes.size(); i++)
		{
			GUINT nodekey = nodesptr[i].m_key;
			if (nodekey != GIM_INVALID_HASH)
			{
				//Search for the avaliable cell in buffer
				GUINT index = _find_avaliable_cell(nodekey);

				if (m_hash_table[index] != GIM_INVALID_HASH)
				{  //The new index is alreade used... discard this new incomming object, repeated key
					btAssert(m_hash_table[index] == nodekey);
					nodesptr[i].m_key = GIM_INVALID_HASH;
				}
				else
				{
					//;
					//Assign the value for alloc
					m_hash_table[index] = i;
				}
			}
		}
	}

	//! Resize hash table indices
	inline void _resize_table(GUINT newsize)
	{
		//Clear memory
		_clear_table_memory();
		//Alloc the data
		_reserve_table_memory(newsize);
		//Invalidate keys and rehash
		_rehash();
	}

	//! Destroy hash table memory
	inline void _destroy()
	{
		if (m_hash_table == NULL) return;
		_clear_table_memory();
	}

	//! Finds an avaliable hash table cell, and resizes the table if there isn't space
	inline GUINT _assign_hash_table_cell(GUINT hashkey)
	{
		GUINT cell_index = _find_avaliable_cell(hashkey);

		if (cell_index == GIM_INVALID_HASH)
		{
			//rehashing
			_resize_table(m_table_size + 1);
			GUINT cell_index = _find_avaliable_cell(hashkey);
			btAssert(cell_index != GIM_INVALID_HASH);
		}
		return cell_index;
	}

	//! erase by index in hash table
	inline bool _erase_by_index_hash_table(GUINT index)
	{
		if (index >= m_nodes.size()) return false;
		if (m_nodes[index].m_key != GIM_INVALID_HASH)
		{
			//Search for the avaliable cell in buffer
			GUINT cell_index = _find_cell(m_nodes[index].m_key);

			btAssert(cell_index != GIM_INVALID_HASH);
			btAssert(m_hash_table[cell_index] == index);

			m_hash_table[cell_index] = GIM_INVALID_HASH;
		}

		return this->_erase_unsorted(index);
	}

	//! erase by key in hash table
	inline bool _erase_hash_table(GUINT hashkey)
	{
		if (hashkey == GIM_INVALID_HASH) return false;

		//Search for the avaliable cell in buffer
		GUINT cell_index = _find_cell(hashkey);
		if (cell_index == GIM_INVALID_HASH) return false;

		GUINT index = m_hash_table[cell_index];
		m_hash_table[cell_index] = GIM_INVALID_HASH;

		return this->_erase_unsorted(index);
	}

	//! insert an element in hash table
	/*!
    If the element exists, this won't insert the element
    \return the index in the array of the existing element,or GIM_INVALID_HASH if the element has been inserted
    If so, the element has been inserted at the last position of the array.
    */
	inline GUINT _insert_hash_table(GUINT hashkey, const T& value)
	{
		if (hashkey == GIM_INVALID_HASH)
		{
			//Insert anyway
			_insert_unsorted(hashkey, value);
			return GIM_INVALID_HASH;
		}

		GUINT cell_index = _assign_hash_table_cell(hashkey);

		GUINT value_key = m_hash_table[cell_index];

		if (value_key != GIM_INVALID_HASH) return value_key;  // Not overrited

		m_hash_table[cell_index] = m_nodes.size();

		_insert_unsorted(hashkey, value);
		return GIM_INVALID_HASH;
	}

	//! insert an element in hash table.
	/*!
    If the element exists, this replaces the element.
    \return the index in the array of the existing element,or GIM_INVALID_HASH if the element has been inserted
    If so, the element has been inserted at the last position of the array.
    */
	inline GUINT _insert_hash_table_replace(GUINT hashkey, const T& value)
	{
		if (hashkey == GIM_INVALID_HASH)
		{
			//Insert anyway
			_insert_unsorted(hashkey, value);
			return GIM_INVALID_HASH;
		}

		GUINT cell_index = _assign_hash_table_cell(hashkey);

		GUINT value_key = m_hash_table[cell_index];

		if (value_key != GIM_INVALID_HASH)
		{  //replaces the existing
			m_nodes[value_key] = _node_type(hashkey, value);
			return value_key;  // index of the replaced element
		}

		m_hash_table[cell_index] = m_nodes.size();

		_insert_unsorted(hashkey, value);
		return GIM_INVALID_HASH;
	}

	///Sorted array data management. The hash table has the indices to the corresponding m_nodes array
	inline bool _erase_sorted(GUINT index)
	{
		if (index >= (GUINT)m_nodes.size()) return false;
		m_nodes.erase_sorted(index);
		if (m_nodes.size() < 2) m_sorted = false;
		return true;
	}

	//! faster, but unsorted
	inline bool _erase_unsorted(GUINT index)
	{
		if (index >= m_nodes.size()) return false;

		GUINT lastindex = m_nodes.size() - 1;
		if (index < lastindex && m_hash_table != 0)
		{
			GUINT hashkey = m_nodes[lastindex].m_key;
			if (hashkey != GIM_INVALID_HASH)
			{
				//update the new position of the last element
				GUINT cell_index = _find_cell(hashkey);
				btAssert(cell_index != GIM_INVALID_HASH);
				//new position of the last element which will be swaped
				m_hash_table[cell_index] = index;
			}
		}
		m_nodes.erase(index);
		m_sorted = false;
		return true;
	}

	//! Insert in position ordered
	/*!
    Also checks if it is needed to transform this container to a hash table, by calling check_for_switching_to_hashtable
    */
	inline void _insert_in_pos(GUINT hashkey, const T& value, GUINT pos)
	{
		m_nodes.insert(_node_type(hashkey, value), pos);
		this->check_for_switching_to_hashtable();
	}

	//! Insert an element in an ordered array
	inline GUINT _insert_sorted(GUINT hashkey, const T& value)
	{
		if (hashkey == GIM_INVALID_HASH || size() == 0)
		{
			m_nodes.push_back(_node_type(hashkey, value));
			return GIM_INVALID_HASH;
		}
		//Insert at last position
		//Sort element

		GUINT result_ind = 0;
		GUINT last_index = m_nodes.size() - 1;
		_node_type* ptr = m_nodes.pointer();

		bool found = gim_binary_search_ex(
			ptr, 0, last_index, result_ind, hashkey, GIM_HASH_NODE_CMP_KEY_MACRO());

		//Insert before found index
		if (found)
		{
			return result_ind;
		}
		else
		{
			_insert_in_pos(hashkey, value, result_ind);
		}
		return GIM_INVALID_HASH;
	}

	inline GUINT _insert_sorted_replace(GUINT hashkey, const T& value)
	{
		if (hashkey == GIM_INVALID_HASH || size() == 0)
		{
			m_nodes.push_back(_node_type(hashkey, value));
			return GIM_INVALID_HASH;
		}
		//Insert at last position
		//Sort element
		GUINT result_ind;
		GUINT last_index = m_nodes.size() - 1;
		_node_type* ptr = m_nodes.pointer();

		bool found = gim_binary_search_ex(
			ptr, 0, last_index, result_ind, hashkey, GIM_HASH_NODE_CMP_KEY_MACRO());

		//Insert before found index
		if (found)
		{
			m_nodes[result_ind] = _node_type(hashkey, value);
		}
		else
		{
			_insert_in_pos(hashkey, value, result_ind);
		}
		return result_ind;
	}

	//! Fast insertion in m_nodes array
	inline GUINT _insert_unsorted(GUINT hashkey, const T& value)
	{
		m_nodes.push_back(_node_type(hashkey, value));
		m_sorted = false;
		return GIM_INVALID_HASH;
	}

public:
	/*!
        <li> if node_size = 0, then this container becomes a simple sorted array allocator. reserve_size is used for reserve memory in m_nodes.
        When the array size reaches the size equivalent to 'min_hash_table_size', then it becomes a hash table by calling check_for_switching_to_hashtable.
        <li> If node_size != 0, then this container becomes a hash table for ever
        </ul>
    */
	gim_hash_table(GUINT reserve_size = GIM_DEFAULT_HASH_TABLE_SIZE,
				   GUINT node_size = GIM_DEFAULT_HASH_TABLE_NODE_SIZE,
				   GUINT min_hash_table_size = GIM_INVALID_HASH)
	{
		m_hash_table = NULL;
		m_table_size = 0;
		m_sorted = false;
		m_node_size = node_size;
		m_min_hash_table_size = min_hash_table_size;

		if (m_node_size != 0)
		{
			if (reserve_size != 0)
			{
				m_nodes.reserve(reserve_size);
				_reserve_table_memory(reserve_size);
				_invalidate_keys();
			}
			else
			{
				m_nodes.reserve(GIM_DEFAULT_HASH_TABLE_SIZE);
				_reserve_table_memory(GIM_DEFAULT_HASH_TABLE_SIZE);
				_invalidate_keys();
			}
		}
		else if (reserve_size != 0)
		{
			m_nodes.reserve(reserve_size);
		}
	}

	~gim_hash_table()
	{
		_destroy();
	}

	inline bool is_hash_table()
	{
		if (m_hash_table) return true;
		return false;
	}

	inline bool is_sorted()
	{
		if (size() < 2) return true;
		return m_sorted;
	}

	bool sort()
	{
		if (is_sorted()) return true;
		if (m_nodes.size() < 2) return false;

		_node_type* ptr = m_nodes.pointer();
		GUINT siz = m_nodes.size();
		gim_sort_hash_node_array(ptr, siz);
		m_sorted = true;

		if (m_hash_table)
		{
			_rehash();
		}
		return true;
	}

	bool switch_to_hashtable()
	{
		if (m_hash_table) return false;
		if (m_node_size == 0) m_node_size = GIM_DEFAULT_HASH_TABLE_NODE_SIZE;
		if (m_nodes.size() < GIM_DEFAULT_HASH_TABLE_SIZE)
		{
			_resize_table(GIM_DEFAULT_HASH_TABLE_SIZE);
		}
		else
		{
			_resize_table(m_nodes.size() + 1);
		}

		return true;
	}

	bool switch_to_sorted_array()
	{
		if (m_hash_table == NULL) return true;
		_clear_table_memory();
		return sort();
	}

	//!If the container reaches the
	bool check_for_switching_to_hashtable()
	{
		if (this->m_hash_table) return true;

		if (!(m_nodes.size() < m_min_hash_table_size))
		{
			if (m_node_size == 0)
			{
				m_node_size = GIM_DEFAULT_HASH_TABLE_NODE_SIZE;
			}

			_resize_table(m_nodes.size() + 1);
			return true;
		}
		return false;
	}

	inline void set_sorted(bool value)
	{
		m_sorted = value;
	}

	//! Retrieves the amount of keys.
	inline GUINT size() const
	{
		return m_nodes.size();
	}

	//! Retrieves the hash key.
	inline GUINT get_key(GUINT index) const
	{
		return m_nodes[index].m_key;
	}

	//! Retrieves the value by index
	/*!
    */
	inline T* get_value_by_index(GUINT index)
	{
		return &m_nodes[index].m_data;
	}

	inline const T& operator[](GUINT index) const
	{
		return m_nodes[index].m_data;
	}

	inline T& operator[](GUINT index)
	{
		return m_nodes[index].m_data;
	}

	//! Finds the index of the element with the key
	/*!
    \return the index in the array of the existing element,or GIM_INVALID_HASH if the element has been inserted
    If so, the element has been inserted at the last position of the array.
    */
	inline GUINT find(GUINT hashkey)
	{
		if (m_hash_table)
		{
			GUINT cell_index = _find_cell(hashkey);
			if (cell_index == GIM_INVALID_HASH) return GIM_INVALID_HASH;
			return m_hash_table[cell_index];
		}
		GUINT last_index = m_nodes.size();
		if (last_index < 2)
		{
			if (last_index == 0) return GIM_INVALID_HASH;
			if (m_nodes[0].m_key == hashkey) return 0;
			return GIM_INVALID_HASH;
		}
		else if (m_sorted)
		{
			//Binary search
			GUINT result_ind = 0;
			last_index--;
			_node_type* ptr = m_nodes.pointer();

			bool found = gim_binary_search_ex(ptr, 0, last_index, result_ind, hashkey, GIM_HASH_NODE_CMP_KEY_MACRO());

			if (found) return result_ind;
		}
		return GIM_INVALID_HASH;
	}

	//! Retrieves the value associated with the index
	/*!
    \return the found element, or null
    */
	inline T* get_value(GUINT hashkey)
	{
		GUINT index = find(hashkey);
		if (index == GIM_INVALID_HASH) return NULL;
		return &m_nodes[index].m_data;
	}

	/*!
    */
	inline bool erase_by_index(GUINT index)
	{
		if (index > m_nodes.size()) return false;

		if (m_hash_table == NULL)
		{
			if (is_sorted())
			{
				return this->_erase_sorted(index);
			}
			else
			{
				return this->_erase_unsorted(index);
			}
		}
		else
		{
			return this->_erase_by_index_hash_table(index);
		}
		return false;
	}

	inline bool erase_by_index_unsorted(GUINT index)
	{
		if (index > m_nodes.size()) return false;

		if (m_hash_table == NULL)
		{
			return this->_erase_unsorted(index);
		}
		else
		{
			return this->_erase_by_index_hash_table(index);
		}
		return false;
	}

	/*!

    */
	inline bool erase_by_key(GUINT hashkey)
	{
		if (size() == 0) return false;

		if (m_hash_table)
		{
			return this->_erase_hash_table(hashkey);
		}
		//Binary search

		if (is_sorted() == false) return false;

		GUINT result_ind = find(hashkey);
		if (result_ind != GIM_INVALID_HASH)
		{
			return this->_erase_sorted(result_ind);
		}
		return false;
	}

	void clear()
	{
		m_nodes.clear();

		if (m_hash_table == NULL) return;
		GUINT datasize = m_table_size * m_node_size;
		//Initialize the hashkeys.
		GUINT i;
		for (i = 0; i < datasize; i++)
		{
			m_hash_table[i] = GIM_INVALID_HASH;  // invalidate keys
		}
		m_sorted = false;
	}

	//! Insert an element into the hash
	/*!
    \return If GIM_INVALID_HASH, the object has been inserted succesfully. Else it returns the position
    of the existing element.
    */
	inline GUINT insert(GUINT hashkey, const T& element)
	{
		if (m_hash_table)
		{
			return this->_insert_hash_table(hashkey, element);
		}
		if (this->is_sorted())
		{
			return this->_insert_sorted(hashkey, element);
		}
		return this->_insert_unsorted(hashkey, element);
	}

	//! Insert an element into the hash, and could overrite an existing object with the same hash.
	/*!
    \return If GIM_INVALID_HASH, the object has been inserted succesfully. Else it returns the position
    of the replaced element.
    */
	inline GUINT insert_override(GUINT hashkey, const T& element)
	{
		if (m_hash_table)
		{
			return this->_insert_hash_table_replace(hashkey, element);
		}
		if (this->is_sorted())
		{
			return this->_insert_sorted_replace(hashkey, element);
		}
		this->_insert_unsorted(hashkey, element);
		return m_nodes.size();
	}

	//! Insert an element into the hash,But if this container is a sorted array, this inserts it unsorted
	/*!
    */
	inline GUINT insert_unsorted(GUINT hashkey, const T& element)
	{
		if (m_hash_table)
		{
			return this->_insert_hash_table(hashkey, element);
		}
		return this->_insert_unsorted(hashkey, element);
	}
};

#endif  // GIM_CONTAINERS_H_INCLUDED
