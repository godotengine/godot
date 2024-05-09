/*
 *  Copyright 2008-2018 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/vector_base.h>
#include <thrust/detail/copy.h>
#include <thrust/detail/overlapped_copy.h>
#include <thrust/equal.h>
#include <thrust/distance.h>
#include <thrust/advance.h>
#include <thrust/detail/type_traits.h>
#include <thrust/detail/minmax.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/temporary_array.h>

#include <stdexcept>

THRUST_NAMESPACE_BEGIN

namespace detail
{

template<typename T, typename Alloc>
  vector_base<T,Alloc>
    ::vector_base(void)
      :m_storage(),
       m_size(0)
{
  ;
} // end vector_base::vector_base()

template<typename T, typename Alloc>
  vector_base<T,Alloc>
    ::vector_base(const Alloc &alloc)
      :m_storage(alloc),
       m_size(0)
{
  ;
} // end vector_base::vector_base()

template<typename T, typename Alloc>
  vector_base<T,Alloc>
    ::vector_base(size_type n)
      :m_storage(),
       m_size(0)
{
  default_init(n);
} // end vector_base::vector_base()

template<typename T, typename Alloc>
  vector_base<T,Alloc>
    ::vector_base(size_type n, const Alloc &alloc)
      :m_storage(alloc),
       m_size(0)
{
  default_init(n);
} // end vector_base::vector_base()

template<typename T, typename Alloc>
  vector_base<T,Alloc>
    ::vector_base(size_type n, const value_type &value)
      :m_storage(),
       m_size(0)
{
  fill_init(n,value);
} // end vector_base::vector_base()

template<typename T, typename Alloc>
  vector_base<T,Alloc>
    ::vector_base(size_type n, const value_type &value, const Alloc &alloc)
      :m_storage(alloc),
       m_size(0)
{
  fill_init(n,value);
} // end vector_base::vector_base()

template<typename T, typename Alloc>
  vector_base<T,Alloc>
    ::vector_base(const vector_base &v)
      :m_storage(copy_allocator_t(), v.m_storage),
       m_size(0)
{
  range_init(v.begin(), v.end());
} // end vector_base::vector_base()

template<typename T, typename Alloc>
  vector_base<T,Alloc>
    ::vector_base(const vector_base &v, const Alloc &alloc)
      :m_storage(alloc),
       m_size(0)
{
  range_init(v.begin(), v.end());
} // end vector_base::vector_base()

#if THRUST_CPP_DIALECT >= 2011
  template<typename T, typename Alloc>
    vector_base<T,Alloc>
      ::vector_base(vector_base &&v)
        :m_storage(copy_allocator_t(), v.m_storage),
         m_size(0)
  {
    *this = std::move(v);
  } //end vector_base::vector_base()
#endif

template<typename T, typename Alloc>
  vector_base<T,Alloc> &
    vector_base<T,Alloc>
      ::operator=(const vector_base &v)
{
  if(this != &v)
  {
    m_storage.destroy_on_allocator_mismatch(v.m_storage, begin(), end());
    m_storage.deallocate_on_allocator_mismatch(v.m_storage);

    m_storage.propagate_allocator(v.m_storage);

    assign(v.begin(), v.end());
  } // end if

  return *this;
} // end vector_base::operator=()

#if THRUST_CPP_DIALECT >= 2011
  template<typename T, typename Alloc>
    vector_base<T,Alloc> &
      vector_base<T,Alloc>
        ::operator=(vector_base &&v)
  {
    m_storage.destroy(begin(), end());
    m_storage = std::move(v.m_storage);
    m_size = std::move(v.m_size);

    v.m_storage = contiguous_storage<T,Alloc>(copy_allocator_t(), m_storage);
    v.m_size = 0;

    return *this;
  } // end vector_base::operator=()
#endif

template<typename T, typename Alloc>
  template<typename OtherT, typename OtherAlloc>
    vector_base<T,Alloc>
      ::vector_base(const vector_base<OtherT,OtherAlloc> &v)
        :m_storage(),
         m_size(0)
{
  range_init(v.begin(), v.end());
} // end vector_base::vector_base()

template<typename T, typename Alloc>
  template<typename OtherT, typename OtherAlloc>
    vector_base<T,Alloc> &
      vector_base<T,Alloc>
        ::operator=(const vector_base<OtherT,OtherAlloc> &v)
{
  assign(v.begin(), v.end());

  return *this;
} // end vector_base::operator=()

template<typename T, typename Alloc>
  template<typename OtherT, typename OtherAlloc>
    vector_base<T,Alloc>
      ::vector_base(const std::vector<OtherT,OtherAlloc> &v)
        :m_storage(),
         m_size(0)
{
  range_init(v.begin(), v.end());
} // end vector_base::vector_base()

template<typename T, typename Alloc>
  template<typename OtherT, typename OtherAlloc>
    vector_base<T,Alloc> &
      vector_base<T,Alloc>
        ::operator=(const std::vector<OtherT,OtherAlloc> &v)
{
  assign(v.begin(), v.end());

  return *this;
} // end vector_base::operator=()

  template<typename T, typename Alloc>
    vector_base<T,Alloc>
      ::vector_base(std::initializer_list<T> il)
        :m_storage(),
         m_size(0)
  {
    range_init(il.begin(), il.end());
  } // end vector_base::vector_base()

  template<typename T, typename Alloc>
  vector_base<T,Alloc>
    ::vector_base(std::initializer_list<T> il, const Alloc &alloc)
    :m_storage(alloc),
      m_size(0)
  {
    range_init(il.begin(), il.end());
  } // end vector_base::vector_base()

  template<typename T, typename Alloc>
    vector_base<T,Alloc> &
      vector_base<T,Alloc>
      ::operator=(std::initializer_list<T> il)
  {
    assign(il.begin(), il.end());

    return *this;
  } // end vector_base::operator=()

template<typename T, typename Alloc>
  template<typename IteratorOrIntegralType>
    void vector_base<T,Alloc>
      ::init_dispatch(IteratorOrIntegralType n,
                      IteratorOrIntegralType value,
                      true_type)
{
  fill_init(n,value);
} // end vector_base::init_dispatch()

template<typename T, typename Alloc>
  void vector_base<T,Alloc>
    ::default_init(size_type n)
{
  if(n > 0)
  {
    m_storage.allocate(n);
    m_size = n;

    m_storage.default_construct_n(begin(), size());
  } // end if
} // end vector_base::default_init()

template<typename T, typename Alloc>
  void vector_base<T,Alloc>
    ::fill_init(size_type n, const T &x)
{
  if(n > 0)
  {
    m_storage.allocate(n);
    m_size = n;

    m_storage.uninitialized_fill_n(begin(), size(), x);
  } // end if
} // end vector_base::fill_init()

template<typename T, typename Alloc>
  template<typename InputIterator>
    void vector_base<T,Alloc>
      ::init_dispatch(InputIterator first,
                      InputIterator last,
                      false_type)
{
  range_init(first, last);
} // end vector_base::init_dispatch()

template<typename T, typename Alloc>
  template<typename InputIterator>
    void vector_base<T,Alloc>
      ::range_init(InputIterator first,
                   InputIterator last)
{
  range_init(first, last,
    typename thrust::iterator_traversal<InputIterator>::type());
} // end vector_base::range_init()

template<typename T, typename Alloc>
  template<typename InputIterator>
    void vector_base<T,Alloc>
      ::range_init(InputIterator first,
                   InputIterator last,
                   thrust::incrementable_traversal_tag)
{
  for(; first != last; ++first)
    push_back(*first);
} // end vector_base::range_init()

template<typename T, typename Alloc>
  template<typename ForwardIterator>
    void vector_base<T,Alloc>
      ::range_init(ForwardIterator first,
                   ForwardIterator last,
                   thrust::random_access_traversal_tag)
{
  size_type new_size = thrust::distance(first, last);

  allocate_and_copy(new_size, first, last, m_storage);
  m_size    = new_size;
} // end vector_base::range_init()

template<typename T, typename Alloc>
  template<typename InputIterator>
    vector_base<T,Alloc>
      ::vector_base(InputIterator first,
                    InputIterator last)
        :m_storage(),
         m_size(0)
{
  // check the type of InputIterator: if it's an integral type,
  // we need to interpret this call as (size_type, value_type)
  typedef thrust::detail::is_integral<InputIterator> Integer;

  init_dispatch(first, last, Integer());
} // end vector_base::vector_base()

template<typename T, typename Alloc>
  template<typename InputIterator>
    vector_base<T,Alloc>
      ::vector_base(InputIterator first,
                    InputIterator last,
                    const Alloc &alloc)
        :m_storage(alloc),
         m_size(0)
{
  // check the type of InputIterator: if it's an integral type,
  // we need to interpret this call as (size_type, value_type)
  typedef thrust::detail::is_integral<InputIterator> Integer;

  init_dispatch(first, last, Integer());
} // end vector_base::vector_base()

template<typename T, typename Alloc>
  void vector_base<T,Alloc>
    ::resize(size_type new_size)
{
  if(new_size < size())
  {
    iterator new_end = begin();
    thrust::advance(new_end, new_size);
    erase(new_end, end());
  } // end if
  else
  {
    append(new_size - size());
  } // end else
} // end vector_base::resize()

template<typename T, typename Alloc>
  void vector_base<T,Alloc>
    ::resize(size_type new_size, const value_type &x)
{
  if(new_size < size())
  {
    iterator new_end = begin();
    thrust::advance(new_end, new_size);
    erase(new_end, end());
  } // end if
  else
  {
    insert(end(), new_size - size(), x);
  } // end else
} // end vector_base::resize()

template<typename T, typename Alloc>
  typename vector_base<T,Alloc>::size_type
    vector_base<T,Alloc>
      ::size(void) const
{
  return m_size;
} // end vector_base::size()

template<typename T, typename Alloc>
  typename vector_base<T,Alloc>::size_type
    vector_base<T,Alloc>
      ::max_size(void) const
{
  return m_storage.max_size();
} // end vector_base::max_size()

template<typename T, typename Alloc>
  void vector_base<T,Alloc>
    ::reserve(size_type n)
{
  if(n > capacity())
  {
    // compute the new capacity after the allocation
    size_type new_capacity = n;

    // do not exceed maximum storage
    new_capacity = thrust::min THRUST_PREVENT_MACRO_SUBSTITUTION <size_type>(new_capacity, max_size());

    // create new storage
    storage_type new_storage(copy_allocator_t(), m_storage, new_capacity);

    // record how many constructors we invoke in the try block below
    iterator new_end = new_storage.begin();

    try
    {
      // construct copy all elements into the newly allocated storage
      new_end = m_storage.uninitialized_copy(begin(), end(), new_storage.begin());
    } // end try
    catch(...)
    {
      // something went wrong, so destroy & deallocate the new storage
      new_storage.destroy(new_storage.begin(), new_end);
      new_storage.deallocate();

      // rethrow
      throw;
    } // end catch

    // call destructors on the elements in the old storage
    m_storage.destroy(begin(), end());

    // record the vector's new state
    m_storage.swap(new_storage);
  } // end if
} // end vector_base::reserve()

template<typename T, typename Alloc>
  typename vector_base<T,Alloc>::size_type
    vector_base<T,Alloc>
      ::capacity(void) const
{
  return m_storage.size();
} // end vector_base::capacity()

template<typename T, typename Alloc>
  void vector_base<T,Alloc>
    ::shrink_to_fit(void)
{
  // use the swap trick
  vector_base(*this).swap(*this);
} // end vector_base::shrink_to_fit()

template<typename T, typename Alloc>
  typename vector_base<T,Alloc>::reference
    vector_base<T,Alloc>
      ::operator[](const size_type n)
{
  return m_storage[n];
} // end vector_base::operator[]

template<typename T, typename Alloc>
  typename vector_base<T,Alloc>::const_reference
    vector_base<T,Alloc>
      ::operator[](const size_type n) const
{
  return m_storage[n];
} // end vector_base::operator[]

template<typename T, typename Alloc>
  typename vector_base<T,Alloc>::iterator
    vector_base<T,Alloc>
      ::begin(void)
{
  return m_storage.begin();
} // end vector_base::begin()

template<typename T, typename Alloc>
  typename vector_base<T,Alloc>::const_iterator
    vector_base<T,Alloc>
      ::begin(void) const
{
  return m_storage.begin();
} // end vector_base::begin()

template<typename T, typename Alloc>
  typename vector_base<T,Alloc>::const_iterator
    vector_base<T,Alloc>
      ::cbegin(void) const
{
  return begin();
} // end vector_base::cbegin()

template<typename T, typename Alloc>
  typename vector_base<T,Alloc>::reverse_iterator
    vector_base<T,Alloc>
      ::rbegin(void)
{
  return reverse_iterator(end());
} // end vector_base::rbegin()

template<typename T, typename Alloc>
  typename vector_base<T,Alloc>::const_reverse_iterator
    vector_base<T,Alloc>
      ::rbegin(void) const
{
  return const_reverse_iterator(end());
} // end vector_base::rbegin()

template<typename T, typename Alloc>
  typename vector_base<T,Alloc>::const_reverse_iterator
    vector_base<T,Alloc>
      ::crbegin(void) const
{
  return rbegin();
} // end vector_base::crbegin()

template<typename T, typename Alloc>
  typename vector_base<T,Alloc>::iterator
    vector_base<T,Alloc>
      ::end(void)
{
  iterator result = begin();
  thrust::advance(result, size());
  return result;
} // end vector_base::end()

template<typename T, typename Alloc>
  typename vector_base<T,Alloc>::const_iterator
    vector_base<T,Alloc>
      ::end(void) const
{
  const_iterator result = begin();
  thrust::advance(result, size());
  return result;
} // end vector_base::end()

template<typename T, typename Alloc>
  typename vector_base<T,Alloc>::const_iterator
    vector_base<T,Alloc>
      ::cend(void) const
{
  return end();
} // end vector_base::cend()

template<typename T, typename Alloc>
  typename vector_base<T,Alloc>::reverse_iterator
    vector_base<T,Alloc>
      ::rend(void)
{
  return reverse_iterator(begin());
} // end vector_base::rend()

template<typename T, typename Alloc>
  typename vector_base<T,Alloc>::const_reverse_iterator
    vector_base<T,Alloc>
      ::rend(void) const
{
  return const_reverse_iterator(begin());
} // end vector_base::rend()

template<typename T, typename Alloc>
  typename vector_base<T,Alloc>::const_reverse_iterator
    vector_base<T,Alloc>
      ::crend(void) const
{
  return rend();
} // end vector_base::crend()

template<typename T, typename Alloc>
  typename vector_base<T,Alloc>::const_reference
    vector_base<T,Alloc>
      ::front(void) const
{
  return *begin();
} // end vector_base::front()

template<typename T, typename Alloc>
  typename vector_base<T,Alloc>::reference
    vector_base<T,Alloc>
      ::front(void)
{
  return *begin();
} // end vector_base::front()

template<typename T, typename Alloc>
  typename vector_base<T,Alloc>::const_reference
    vector_base<T,Alloc>
      ::back(void) const
{
  const_iterator ptr_to_back = end();
  --ptr_to_back;
  return *ptr_to_back;
} // end vector_base::vector_base

template<typename T, typename Alloc>
  typename vector_base<T,Alloc>::reference
    vector_base<T,Alloc>
      ::back(void)
{
  iterator ptr_to_back = end();
  --ptr_to_back;
  return *ptr_to_back;
} // end vector_base::vector_base

template<typename T, typename Alloc>
  typename vector_base<T,Alloc>::pointer
    vector_base<T,Alloc>
      ::data(void)
{
  return pointer(&front());
} // end vector_base::data()

template<typename T, typename Alloc>
  typename vector_base<T,Alloc>::const_pointer
    vector_base<T,Alloc>
      ::data(void) const
{
  return const_pointer(&front());
} // end vector_base::data()

template<typename T, typename Alloc>
  vector_base<T,Alloc>
    ::~vector_base(void)
{
  // destroy every living thing
  if (!empty())
    m_storage.destroy(begin(),end());
} // end vector_base::~vector_base()

template<typename T, typename Alloc>
  void vector_base<T,Alloc>
    ::clear(void)
{
  erase(begin(), end());
} // end vector_base::~vector_dev()

template<typename T, typename Alloc>
  bool vector_base<T,Alloc>
    ::empty(void) const
{
  return size() == 0;
} // end vector_base::empty();

template<typename T, typename Alloc>
  void vector_base<T,Alloc>
    ::push_back(const value_type &x)
{
  insert(end(), x);
} // end vector_base::push_back()

template<typename T, typename Alloc>
  void vector_base<T,Alloc>
    ::pop_back(void)
{
  iterator e = end();
  iterator ptr_to_back = e;
  --ptr_to_back;
  m_storage.destroy(ptr_to_back, e);
  --m_size;
} // end vector_base::pop_back()

template<typename T, typename Alloc>
  typename vector_base<T,Alloc>::iterator vector_base<T,Alloc>
    ::erase(iterator pos)
{
  iterator end = pos;
  ++end;
  return erase(pos,end);
} // end vector_base::erase()

template<typename T, typename Alloc>
  typename vector_base<T,Alloc>::iterator vector_base<T,Alloc>
    ::erase(iterator first, iterator last)
{
  // overlap copy the range [last,end()) to first
  // XXX this copy only potentially overlaps
  iterator i = thrust::detail::overlapped_copy(last, end(), first);

  // destroy everything after i
  m_storage.destroy(i, end());

  // modify our size
  m_size -= (last - first);

  // return an iterator pointing to the position of the first element
  // following the erased range
  return first;
} // end vector_base::erase()

template<typename T, typename Alloc>
  void vector_base<T,Alloc>
    ::swap(vector_base &v)
{
  thrust::swap(m_storage,  v.m_storage);
  thrust::swap(m_size,     v.m_size);
} // end vector_base::swap()

template<typename T, typename Alloc>
  void vector_base<T,Alloc>
    ::assign(size_type n, const T &x)
{
  fill_assign(n, x);
} // end vector_base::assign()

template<typename T, typename Alloc>
  template<typename InputIterator>
    void vector_base<T,Alloc>
      ::assign(InputIterator first, InputIterator last)
{
  // we could have received assign(n, x), so disambiguate on the
  // type of InputIterator
  typedef typename thrust::detail::is_integral<InputIterator> integral;

  assign_dispatch(first, last, integral());
} // end vector_base::assign()

template<typename T, typename Alloc>
  typename vector_base<T,Alloc>::allocator_type
    vector_base<T,Alloc>
      ::get_allocator(void) const
{
  return m_storage.get_allocator();
} // end vector_base::get_allocator()

template<typename T, typename Alloc>
  typename vector_base<T,Alloc>::iterator
    vector_base<T,Alloc>
      ::insert(iterator position, const T &x)
{
  // find the index of the insertion
  size_type index = thrust::distance(begin(), position);

  // make the insertion
  insert(position, 1, x);

  // return an iterator pointing back to position
  iterator result = begin();
  thrust::advance(result, index);
  return result;
} // end vector_base::insert()

template<typename T, typename Alloc>
  void vector_base<T,Alloc>
    ::insert(iterator position, size_type n, const T &x)
{
  fill_insert(position, n, x);
} // end vector_base::insert()

template<typename T, typename Alloc>
  template<typename InputIterator>
    void vector_base<T,Alloc>
      ::insert(iterator position, InputIterator first, InputIterator last)
{
  // we could have received insert(position, n, x), so disambiguate on the
  // type of InputIterator
  typedef typename thrust::detail::is_integral<InputIterator> integral;

  insert_dispatch(position, first, last, integral());
} // end vector_base::insert()

template<typename T, typename Alloc>
  template<typename InputIterator>
    void vector_base<T,Alloc>
      ::assign_dispatch(InputIterator first, InputIterator last, false_type)
{
  range_assign(first, last);
} // end vector_base::assign_dispatch()

template<typename T, typename Alloc>
  template<typename Integral>
    void vector_base<T,Alloc>
      ::assign_dispatch(Integral n, Integral x, true_type)
{
  fill_assign(n, x);
} // end vector_base::assign_dispatch()

template<typename T, typename Alloc>
  template<typename InputIterator>
    void vector_base<T,Alloc>
      ::insert_dispatch(iterator position, InputIterator first, InputIterator last, false_type)
{
  copy_insert(position, first, last);
} // end vector_base::insert_dispatch()

template<typename T, typename Alloc>
  template<typename Integral>
    void vector_base<T,Alloc>
      ::insert_dispatch(iterator position, Integral n, Integral x, true_type)
{
  fill_insert(position, n, x);
} // end vector_base::insert_dispatch()

template<typename T, typename Alloc>
  template<typename ForwardIterator>
    void vector_base<T,Alloc>
      ::copy_insert(iterator position,
                    ForwardIterator first,
                    ForwardIterator last)
{
  if(first != last)
  {
    // how many new elements will we create?
    const size_type num_new_elements = thrust::distance(first, last);
    if(capacity() - size() >= num_new_elements)
    {
      // we've got room for all of them
      // how many existing elements will we displace?
      const size_type num_displaced_elements = end() - position;
      iterator old_end = end();

      if(num_displaced_elements > num_new_elements)
      {
        // construct copy n displaced elements to new elements
        // following the insertion
        m_storage.uninitialized_copy(end() - num_new_elements, end(), end());

        // extend the size
        m_size += num_new_elements;

        // copy num_displaced_elements - num_new_elements elements to existing elements
        // this copy overlaps
        const size_type copy_length = (old_end - num_new_elements) - position;
        thrust::detail::overlapped_copy(position, old_end - num_new_elements, old_end - copy_length);

        // finally, copy the range to the insertion point
        thrust::copy(first, last, position);
      } // end if
      else
      {
        ForwardIterator mid = first;
        thrust::advance(mid, num_displaced_elements);

        // construct copy new elements at the end of the vector
        m_storage.uninitialized_copy(mid, last, end());

        // extend the size
        m_size += num_new_elements - num_displaced_elements;

        // construct copy the displaced elements
        m_storage.uninitialized_copy(position, old_end, end());

        // extend the size
        m_size += num_displaced_elements;

        // copy to elements which already existed
        thrust::copy(first, mid, position);
      } // end else
    } // end if
    else
    {
      const size_type old_size = size();

      // compute the new capacity after the allocation
      size_type new_capacity = old_size + thrust::max THRUST_PREVENT_MACRO_SUBSTITUTION (old_size, num_new_elements);

      // allocate exponentially larger new storage
      new_capacity = thrust::max THRUST_PREVENT_MACRO_SUBSTITUTION <size_type>(new_capacity, 2 * capacity());

      // do not exceed maximum storage
      new_capacity = thrust::min THRUST_PREVENT_MACRO_SUBSTITUTION <size_type>(new_capacity, max_size());

      if(new_capacity > max_size())
      {
        throw std::length_error("insert(): insertion exceeds max_size().");
      } // end if

      storage_type new_storage(copy_allocator_t(), m_storage, new_capacity);

      // record how many constructors we invoke in the try block below
      iterator new_end = new_storage.begin();

      try
      {
        // construct copy elements before the insertion to the beginning of the newly
        // allocated storage
        new_end = m_storage.uninitialized_copy(begin(), position, new_storage.begin());

        // construct copy elements to insert
        new_end = m_storage.uninitialized_copy(first, last, new_end);

        // construct copy displaced elements from the old storage to the new storage
        // remember [position, end()) refers to the old storage
        new_end = m_storage.uninitialized_copy(position, end(), new_end);
      } // end try
      catch(...)
      {
        // something went wrong, so destroy & deallocate the new storage
        m_storage.destroy(new_storage.begin(), new_end);
        new_storage.deallocate();

        // rethrow
        throw;
      } // end catch

      // call destructors on the elements in the old storage
      m_storage.destroy(begin(), end());

      // record the vector's new state
      m_storage.swap(new_storage);
      m_size = old_size + num_new_elements;
    } // end else
  } // end if
} // end vector_base::copy_insert()

template<typename T, typename Alloc>
  void vector_base<T,Alloc>
    ::append(size_type n)
{
  if(n != 0)
  {
    if(capacity() - size() >= n)
    {
      // we've got room for all of them

      // default construct new elements at the end of the vector
      m_storage.default_construct_n(end(), n);

      // extend the size
      m_size += n;
    } // end if
    else
    {
      const size_type old_size = size();

      // compute the new capacity after the allocation
      size_type new_capacity = old_size + thrust::max THRUST_PREVENT_MACRO_SUBSTITUTION (old_size, n);

      // allocate exponentially larger new storage
      new_capacity = thrust::max THRUST_PREVENT_MACRO_SUBSTITUTION <size_type>(new_capacity, 2 * capacity());

      // do not exceed maximum storage
      new_capacity = thrust::min THRUST_PREVENT_MACRO_SUBSTITUTION <size_type>(new_capacity, max_size());

      // create new storage
      storage_type new_storage(copy_allocator_t(), m_storage, new_capacity);

      // record how many constructors we invoke in the try block below
      iterator new_end = new_storage.begin();

      try
      {
        // construct copy all elements into the newly allocated storage
        new_end = m_storage.uninitialized_copy(begin(), end(), new_storage.begin());

        // construct new elements to insert
        new_storage.default_construct_n(new_end, n);
        new_end += n;
      } // end try
      catch(...)
      {
        // something went wrong, so destroy & deallocate the new storage
        new_storage.destroy(new_storage.begin(), new_end);
        new_storage.deallocate();

        // rethrow
        throw;
      } // end catch

      // call destructors on the elements in the old storage
      m_storage.destroy(begin(), end());

      // record the vector's new state
      m_storage.swap(new_storage);
      m_size    = old_size + n;
    } // end else
  } // end if
} // end vector_base::append()

template<typename T, typename Alloc>
  void vector_base<T,Alloc>
    ::fill_insert(iterator position, size_type n, const T &x)
{
  if(n != 0)
  {
    if(capacity() - size() >= n)
    {
      // we've got room for all of them
      // how many existing elements will we displace?
      const size_type num_displaced_elements = end() - position;
      iterator old_end = end();

      if(num_displaced_elements > n)
      {
        // construct copy n displaced elements to new elements
        // following the insertion
        m_storage.uninitialized_copy(end() - n, end(), end());

        // extend the size
        m_size += n;

        // copy num_displaced_elements - n elements to existing elements
        // this copy overlaps
        const size_type copy_length = (old_end - n) - position;
        thrust::detail::overlapped_copy(position, old_end - n, old_end - copy_length);

        // finally, fill the range to the insertion point
        thrust::fill_n(position, n, x);
      } // end if
      else
      {
        // construct new elements at the end of the vector
        m_storage.uninitialized_fill_n(end(), n - num_displaced_elements, x);

        // extend the size
        m_size += n - num_displaced_elements;

        // construct copy the displaced elements
        m_storage.uninitialized_copy(position, old_end, end());

        // extend the size
        m_size += num_displaced_elements;

        // fill to elements which already existed
        thrust::fill(position, old_end, x);
      } // end else
    } // end if
    else
    {
      const size_type old_size = size();

      // compute the new capacity after the allocation
      size_type new_capacity = old_size + thrust::max THRUST_PREVENT_MACRO_SUBSTITUTION (old_size, n);

      // allocate exponentially larger new storage
      new_capacity = thrust::max THRUST_PREVENT_MACRO_SUBSTITUTION <size_type>(new_capacity, 2 * capacity());

      // do not exceed maximum storage
      new_capacity = thrust::min THRUST_PREVENT_MACRO_SUBSTITUTION <size_type>(new_capacity, max_size());

      if(new_capacity > max_size())
      {
        throw std::length_error("insert(): insertion exceeds max_size().");
      } // end if

      storage_type new_storage(copy_allocator_t(), m_storage, new_capacity);

      // record how many constructors we invoke in the try block below
      iterator new_end = new_storage.begin();

      try
      {
        // construct copy elements before the insertion to the beginning of the newly
        // allocated storage
        new_end = m_storage.uninitialized_copy(begin(), position, new_storage.begin());

        // construct new elements to insert
        m_storage.uninitialized_fill_n(new_end, n, x);
        new_end += n;

        // construct copy displaced elements from the old storage to the new storage
        // remember [position, end()) refers to the old storage
        new_end = m_storage.uninitialized_copy(position, end(), new_end);
      } // end try
      catch(...)
      {
        // something went wrong, so destroy & deallocate the new storage
        m_storage.destroy(new_storage.begin(), new_end);
        new_storage.deallocate();

        // rethrow
        throw;
      } // end catch

      // call destructors on the elements in the old storage
      m_storage.destroy(begin(), end());

      // record the vector's new state
      m_storage.swap(new_storage);
      m_size    = old_size + n;
    } // end else
  } // end if
} // end vector_base::fill_insert()

template<typename T, typename Alloc>
  template<typename InputIterator>
    void vector_base<T,Alloc>
      ::range_assign(InputIterator first,
                     InputIterator last)
{
  // dispatch on traversal
  range_assign(first, last,
    typename thrust::iterator_traversal<InputIterator>::type());
} // end range_assign()

template<typename T, typename Alloc>
  template<typename InputIterator>
    void vector_base<T,Alloc>
      ::range_assign(InputIterator first,
                     InputIterator last,
                     thrust::incrementable_traversal_tag)
{
  iterator current(begin());

  // assign to elements which already exist
  for(; first != last && current != end(); ++current, ++first)
  {
    *current = *first;
  } // end for

  // either just the input was exhausted or both
  // the input and vector elements were exhausted
  if(first == last)
  {
    // if we exhausted the input, erase leftover elements
    erase(current, end());
  } // end if
  else
  {
    // insert the rest of the input at the end of the vector
    insert(end(), first, last);
  } // end else
} // end vector_base::range_assign()

template<typename T, typename Alloc>
  template<typename RandomAccessIterator>
    void vector_base<T,Alloc>
      ::range_assign(RandomAccessIterator first,
                     RandomAccessIterator last,
                     thrust::random_access_traversal_tag)
{
  const size_type n = thrust::distance(first, last);

  if(n > capacity())
  {
    storage_type new_storage(copy_allocator_t(), m_storage);
    allocate_and_copy(n, first, last, new_storage);

    // call destructors on the elements in the old storage
    m_storage.destroy(begin(), end());

    // record the vector's new state
    m_storage.swap(new_storage);
    m_size = n;
  } // end if
  else if(size() >= n)
  {
    // we can already accomodate the new range
    iterator new_end = thrust::copy(first, last, begin());

    // destroy the elements we don't need
    m_storage.destroy(new_end, end());

    // update size
    m_size = n;
  } // end else if
  else
  {
    // range fits inside allocated storage, but some elements
    // have not been constructed yet

    // XXX TODO we could possibly implement this with one call
    // to transform rather than copy + uninitialized_copy

    // copy to elements which already exist
    RandomAccessIterator mid = first;
    thrust::advance(mid, size());
    thrust::copy(first, mid, begin());

    // uninitialize_copy to elements which must be constructed
    m_storage.uninitialized_copy(mid, last, end());

    // update size
    m_size = n;
  } // end else
} // end vector_base::assign()

template<typename T, typename Alloc>
  void vector_base<T,Alloc>
    ::fill_assign(size_type n, const T &x)
{
  if(n > capacity())
  {
    // XXX we should also include a copy of the allocator:
    // vector_base<T,Alloc> temp(n, x, get_allocator());
    vector_base<T,Alloc> temp(n, x);
    temp.swap(*this);
  } // end if
  else if(n > size())
  {
    // fill to existing elements
    thrust::fill(begin(), end(), x);

    // construct uninitialized elements
    m_storage.uninitialized_fill_n(end(), n - size(), x);

    // adjust size
    m_size += (n - size());
  } // end else if
  else
  {
    // fill to existing elements
    iterator new_end = thrust::fill_n(begin(), n, x);

    // erase the elements after the fill
    erase(new_end, end());
  } // end else
} // end vector_base::fill_assign()

template<typename T, typename Alloc>
  template<typename ForwardIterator>
    void vector_base<T,Alloc>
      ::allocate_and_copy(size_type requested_size,
                          ForwardIterator first, ForwardIterator last,
                          storage_type &new_storage)
{
  if(requested_size == 0)
  {
    new_storage.deallocate();
    return;
  } // end if

  // allocate exponentially larger new storage
  size_type allocated_size = thrust::max<size_type>(requested_size, 2 * capacity());

  // do not exceed maximum storage
  allocated_size = thrust::min<size_type>(allocated_size, max_size());

  if(requested_size > allocated_size)
  {
    throw std::length_error("assignment exceeds max_size().");
  } // end if

  new_storage.allocate(allocated_size);

  try
  {
    // construct the range to the newly allocated storage
    m_storage.uninitialized_copy(first, last, new_storage.begin());
  } // end try
  catch(...)
  {
    // something went wrong, so destroy & deallocate the new storage
    // XXX seems like this destroys too many elements -- should just be last - first instead of requested_size
    iterator new_storage_end = new_storage.begin();
    thrust::advance(new_storage_end, requested_size);
    m_storage.destroy(new_storage.begin(), new_storage_end);
    new_storage.deallocate();

    // rethrow
    throw;
  } // end catch
} // end vector_base::allocate_and_copy()


} // end detail

template<typename T, typename Alloc>
  void swap(detail::vector_base<T,Alloc> &a,
            detail::vector_base<T,Alloc> &b)
{
  a.swap(b);
} // end swap()



namespace detail
{

// iterator tags match
template <typename InputIterator1, typename InputIterator2>
bool vector_equal(InputIterator1 first1, InputIterator1 last1,
                  InputIterator2 first2,
                  thrust::detail::true_type)
{
  return thrust::equal(first1, last1, first2);
}

// iterator tags differ
template <typename InputIterator1, typename InputIterator2>
bool vector_equal(InputIterator1 first1, InputIterator1 last1,
                  InputIterator2 first2,
                  thrust::detail::false_type)
{
  typename thrust::iterator_difference<InputIterator1>::type n = thrust::distance(first1,last1);

  typedef typename thrust::iterator_system<InputIterator1>::type FromSystem1;
  typedef typename thrust::iterator_system<InputIterator2>::type FromSystem2;

  // bring both ranges to the host system
  // note that these copies are no-ops if the range is already convertible to the host system
  FromSystem1 from_system1;
  FromSystem2 from_system2;
  thrust::host_system_tag to_system;
  thrust::detail::move_to_system<InputIterator1, FromSystem1, thrust::host_system_tag> rng1(from_system1, to_system, first1, last1);
  thrust::detail::move_to_system<InputIterator2, FromSystem2, thrust::host_system_tag> rng2(from_system2, to_system, first2, first2 + n);

  return thrust::equal(rng1.begin(), rng1.end(), rng2.begin());
}

template <typename InputIterator1, typename InputIterator2>
bool vector_equal(InputIterator1 first1, InputIterator1 last1,
                  InputIterator2 first2)
{
  typedef typename thrust::iterator_system<InputIterator1>::type system1;
  typedef typename thrust::iterator_system<InputIterator2>::type system2;

  // dispatch on the sameness of the two systems
  return vector_equal(first1, last1, first2,
    thrust::detail::is_same<system1,system2>());
}

} // end namespace detail




template<typename T1, typename Alloc1,
         typename T2, typename Alloc2>
bool operator==(const detail::vector_base<T1,Alloc1>& lhs,
                const detail::vector_base<T2,Alloc2>& rhs)
{
    return lhs.size() == rhs.size() && detail::vector_equal(lhs.begin(), lhs.end(), rhs.begin());
}

template<typename T1, typename Alloc1,
         typename T2, typename Alloc2>
bool operator==(const detail::vector_base<T1,Alloc1>& lhs,
                const std::vector<T2,Alloc2>&         rhs)
{
    return lhs.size() == rhs.size() && detail::vector_equal(lhs.begin(), lhs.end(), rhs.begin());
}

template<typename T1, typename Alloc1,
         typename T2, typename Alloc2>
bool operator==(const std::vector<T1,Alloc1>&         lhs,
                const detail::vector_base<T2,Alloc2>& rhs)
{
    return lhs.size() == rhs.size() && detail::vector_equal(lhs.begin(), lhs.end(), rhs.begin());
}

template<typename T1, typename Alloc1,
         typename T2, typename Alloc2>
bool operator!=(const detail::vector_base<T1,Alloc1>& lhs,
                const detail::vector_base<T2,Alloc2>& rhs)
{
    return !(lhs == rhs);
}

template<typename T1, typename Alloc1,
         typename T2, typename Alloc2>
bool operator!=(const detail::vector_base<T1,Alloc1>& lhs,
                const std::vector<T2,Alloc2>&         rhs)
{
    return !(lhs == rhs);
}

template<typename T1, typename Alloc1,
         typename T2, typename Alloc2>
bool operator!=(const std::vector<T1,Alloc1>&         lhs,
                const detail::vector_base<T2,Alloc2>& rhs)
{
    return !(lhs == rhs);
}

THRUST_NAMESPACE_END

