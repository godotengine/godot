/*
 *  Copyright 2008-2013 NVIDIA Corporation
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

/*! \file 
 *  \brief A reference to an object which resides in memory associated with the
 *  device system.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/device_ptr.h>
#include <thrust/detail/type_traits.h>
#include <thrust/detail/reference.h>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup memory_management Memory Management
 *  \{
 */

/*! \p device_reference acts as a reference-like object to an object stored in device memory.
 *  \p device_reference is not intended to be used directly; rather, this type
 *  is the result of deferencing a \p device_ptr. Similarly, taking the address of
 *  a \p device_reference yields a \p device_ptr.
 *
 *  \p device_reference may often be used from host code in place of operations defined on
 *  its associated \c value_type. For example, when \p device_reference refers to an
 *  arithmetic type, arithmetic operations on it are legal:
 *
 *  \code
 *  #include <thrust/device_vector.h>
 *
 *  int main(void)
 *  {
 *    thrust::device_vector<int> vec(1, 13);
 *
 *    thrust::device_reference<int> ref_to_thirteen = vec[0];
 *
 *    int x = ref_to_thirteen + 1;
 *
 *    // x is 14
 *
 *    return 0;
 *  }
 *  \endcode
 *
 *  Similarly, we can print the value of \c ref_to_thirteen in the above code by using an
 *  \c iostream:
 *
 *  \code
 *  #include <thrust/device_vector.h>
 *  #include <iostream>
 *
 *  int main(void)
 *  {
 *    thrust::device_vector<int> vec(1, 13);
 *
 *    thrust::device_reference<int> ref_to_thirteen = vec[0];
 *
 *    std::cout << ref_to_thirteen << std::endl;
 *
 *    // 13 is printed
 *
 *    return 0;
 *  }
 *  \endcode
 *
 *  Of course, we needn't explicitly create a \p device_reference in the previous
 *  example, because one is returned by \p device_vector's bracket operator. A more natural
 *  way to print the value of a \p device_vector element might be:
 *
 *  \code
 *  #include <thrust/device_vector.h>
 *  #include <iostream>
 *
 *  int main(void)
 *  {
 *    thrust::device_vector<int> vec(1, 13);
 *
 *    std::cout << vec[0] << std::endl;
 *
 *    // 13 is printed
 *
 *    return 0;
 *  }
 *  \endcode
 *
 *  These kinds of operations should be used sparingly in performance-critical code, because
 *  they imply a potentially expensive copy between host and device space.
 *
 *  Some operations which are possible with regular objects are impossible with their
 *  corresponding \p device_reference objects due to the requirements of the C++ language. For
 *  example, because the member access operator cannot be overloaded, member variables and functions
 *  of a referent object cannot be directly accessed through its \p device_reference.
 *
 *  The following code, which generates a compiler error, illustrates:
 *
 *  \code
 *  #include <thrust/device_vector.h>
 *
 *  struct foo
 *  {
 *    int x;
 *  };
 *
 *  int main(void)
 *  {
 *    thrust::device_vector<foo> foo_vec(1);
 *
 *    thrust::device_reference<foo> foo_ref = foo_vec[0];
 *
 *    foo_ref.x = 13; // ERROR: x cannot be accessed through foo_ref
 *
 *    return 0;
 *  }
 *  \endcode
 *
 *  Instead, a host space copy must be created to access \c foo's \c x member:
 *
 *  \code
 *  #include <thrust/device_vector.h>
 *
 *  struct foo
 *  {
 *    int x;
 *  };
 *
 *  int main(void)
 *  {
 *    thrust::device_vector<foo> foo_vec(1);
 *
 *    // create a local host-side foo object
 *    foo host_foo;
 *    host_foo.x = 13;
 *
 *    thrust::device_reference<foo> foo_ref = foo_vec[0];
 *
 *    foo_ref = host_foo;
 *
 *    // foo_ref's x member is 13
 *
 *    return 0;
 *  }
 *  \endcode
 *
 *  Another common case where a \p device_reference cannot directly be used in place of
 *  its referent object occurs when passing them as parameters to functions like \c printf
 *  which have varargs parameters. Because varargs parameters must be Plain Old Data, a
 *  \p device_reference to a POD type requires a cast when passed to \c printf:
 *
 *  \code
 *  #include <stdio.h>
 *  #include <thrust/device_vector.h>
 *
 *  int main(void)
 *  {
 *    thrust::device_vector<int> vec(1,13);
 *
 *    // vec[0] must be cast to int when passing to printf
 *    printf("%d\n", (int) vec[0]);
 *
 *    return 0;
 *  }
 *  \endcode
 *
 *  \see device_ptr
 *  \see device_vector
 */
template<typename T>
  class device_reference
    : public thrust::reference<
               T,
               thrust::device_ptr<T>,
               thrust::device_reference<T>
             >
{
  private:
    typedef thrust::reference<
      T,
      thrust::device_ptr<T>,
      thrust::device_reference<T>
    > super_t;

  public:
    /*! The type of the value referenced by this type of \p device_reference.
     */
    typedef typename super_t::value_type value_type;

    /*! The type of the expression <tt>&ref</tt>, where <tt>ref</tt> is a \p device_reference.
     */
    typedef typename super_t::pointer    pointer;

    /*! This copy constructor accepts a const reference to another
     *  \p device_reference. After this \p device_reference is constructed,
     *  it shall refer to the same object as \p other.
     *
     *  \param other A \p device_reference to copy from.
     *
     *  The following code snippet demonstrates the semantics of this
     *  copy constructor.
     *
     *  \code
     *  #include <thrust/device_vector.h>
     *  #include <assert.h>
     *  ...
     *  thrust::device_vector<int> v(1,0);
     *  thrust::device_reference<int> ref = v[0];
     *
     *  // ref equals the object at v[0]
     *  assert(ref == v[0]);
     *
     *  // the address of ref equals the address of v[0]
     *  assert(&ref == &v[0]);
     *
     *  // modifying v[0] modifies ref
     *  v[0] = 13;
     *  assert(ref == 13);
     *  \endcode
     *
     *  \note This constructor is templated primarily to allow initialization of
     *  <tt>device_reference<const T></tt> from <tt>device_reference<T></tt>.
     */
    template<typename OtherT>
    __host__ __device__
    device_reference(const device_reference<OtherT> &other,
                     typename thrust::detail::enable_if_convertible<
                       typename device_reference<OtherT>::pointer,
                       pointer
                     >::type * = 0)
      : super_t(other)
    {}

    /*! This copy constructor initializes this \p device_reference
     *  to refer to an object pointed to by the given \p device_ptr. After
     *  this \p device_reference is constructed, it shall refer to the
     *  object pointed to by \p ptr.
     *
     *  \param ptr A \p device_ptr to copy from.
     *
     *  The following code snippet demonstrates the semantic of this
     *  copy constructor.
     *
     *  \code
     *  #include <thrust/device_vector.h>
     *  #include <assert.h>
     *  ...
     *  thrust::device_vector<int> v(1,0);
     *  thrust::device_ptr<int> ptr = &v[0];
     *  thrust::device_reference<int> ref(ptr);
     *
     *  // ref equals the object pointed to by ptr
     *  assert(ref == *ptr);
     *
     *  // the address of ref equals ptr
     *  assert(&ref == ptr);
     *
     *  // modifying *ptr modifies ref
     *  *ptr = 13;
     *  assert(ref == 13);
     *  \endcode
     */
    __host__ __device__
    explicit device_reference(const pointer &ptr)
      : super_t(ptr)
    {}

    /*! This assignment operator assigns the value of the object referenced by
     *  the given \p device_reference to the object referenced by this
     *  \p device_reference.
     *
     *  \param other The \p device_reference to assign from.
     *  \return <tt>*this</tt>
     */
    template<typename OtherT>
    __host__ __device__
    device_reference &operator=(const device_reference<OtherT> &other)
    {
      return super_t::operator=(other);
    }

    /*! Assignment operator assigns the value of the given value to the
     *  value referenced by this \p device_reference.
     *
     *  \param x The value to assign from.
     *  \return <tt>*this</tt>
     */
    __host__ __device__
    device_reference &operator=(const value_type &x)
    {
      return super_t::operator=(x);
    }

// declare these members for the purpose of Doxygenating them
// they actually exist in a derived-from class
#if 0
    /*! Address-of operator returns a \p device_ptr pointing to the object
     *  referenced by this \p device_reference. It does not return the
     *  address of this \p device_reference.
     *
     *  \return A \p device_ptr pointing to the object this
     *  \p device_reference references.
     */
    __host__ __device__
    pointer operator&(void) const;

    /*! Conversion operator converts this \p device_reference to T
     *  by returning a copy of the object referenced by this
     *  \p device_reference.
     *
     *  \return A copy of the object referenced by this \p device_reference.
     */
    __host__ __device__
    operator value_type (void) const;

    /*! swaps the value this \p device_reference references with another.
     *  \p other The other \p device_reference with which to swap.
     */
    __host__ __device__
    void swap(device_reference &other);

    /*! Prefix increment operator increments the object referenced by this
     *  \p device_reference.
     *
     *  \return <tt>*this</tt>
     *
     *  The following code snippet demonstrates the semantics of
     *  \p device_reference's prefix increment operator.
     *
     *  \code
     *  #include <thrust/device_vector.h>
     *  #include <assert.h>
     *  ...
     *  thrust::device_vector<int> v(1,0);
     *  thrust::device_ptr<int> ptr = &v[0];
     *  thrust::device_reference<int> ref(ptr);
     *
     *  // ref equals 0
     *  assert(ref == 0);
     *
     *  // the object pointed to by ptr equals 1
     *  assert(*ptr == 1);
     *
     *  // v[0] equals 1
     *  assert(v[0] == 1);
     *
     *  // increment ref
     *  ++ref;
     *
     *  // ref equals 1
     *  assert(ref == 1);
     *
     *  // the object pointed to by ptr equals 1
     *  assert(*ptr == 1);
     *
     *  // v[0] equals 1
     *  assert(v[0] == 1);
     *  \endcode
     *
     *  \note The increment executes as if it were executed on the host.
     *  This may change in a later version.
     */
    device_reference &operator++(void);

    /*! Postfix increment operator copies the object referenced by this
     *  \p device_reference, increments the object referenced by this
     *  \p device_reference, and returns the copy.
     *
     *  \return A copy of the object referenced by this \p device_reference
     *          before being incremented.
     *
     *  The following code snippet demonstrates the semantics of
     *  \p device_reference's postfix increment operator.
     *
     *  \code
     *  #include <thrust/device_vector.h>
     *  #include <assert.h>
     *  ...
     *  thrust::device_vector<int> v(1,0);
     *  thrust::device_ptr<int> ptr = &v[0];
     *  thrust::device_reference<int> ref(ptr);
     *
     *  // ref equals 0
     *  assert(ref == 0);
     *
     *  // the object pointed to by ptr equals 0
     *  assert(*ptr == 0);
     *
     *  // v[0] equals 0
     *  assert(v[0] == 0);
     *
     *  // increment ref
     *  int x = ref++;
     *
     *  // x equals 0
     *  assert(x == 0)
     *
     *  // ref equals 1
     *  assert(ref == 1);
     *
     *  // the object pointed to by ptr equals 1
     *  assert(*ptr == 1);
     *
     *  // v[0] equals 1
     *  assert(v[0] == 1);
     *  \endcode
     *
     *  \note The increment executes as if it were executed on the host.
     *  This may change in a later version.
     */
    value_type operator++(int);

    /*! Addition assignment operator add-assigns the object referenced by this
     *  \p device_reference and returns this \p device_reference.
     *
     *  \param rhs The right hand side of the add-assignment.
     *  \return <tt>*this</tt>.
     *
     *  The following code snippet demonstrates the semantics of
     *  \p device_reference's addition assignment operator.
     *
     *  \code
     *  #include <thrust/device_vector.h>
     *  #include <assert.h>
     *  ...
     *  thrust::device_vector<int> v(1,0);
     *  thrust::device_ptr<int> ptr = &v[0];
     *  thrust::device_reference<int> ref(ptr);
     *
     *  // ref equals 0
     *  assert(ref == 0);
     *
     *  // the object pointed to by ptr equals 0
     *  assert(*ptr == 0);
     *
     *  // v[0] equals 0
     *  assert(v[0] == 0);
     *
     *  // add-assign ref
     *  ref += 5;
     *
     *  // ref equals 5
     *  assert(ref == 5);
     *
     *  // the object pointed to by ptr equals 5
     *  assert(*ptr == 5);
     *
     *  // v[0] equals 5
     *  assert(v[0] == 5);
     *  \endcode
     *
     *  \note The add-assignment executes as as if it were executed on the host.
     *  This may change in a later version.
     */
    device_reference &operator+=(const T &rhs);

    /*! Prefix decrement operator decrements the object referenced by this
     *  \p device_reference.
     *
     *  \return <tt>*this</tt>
     *
     *  The following code snippet demonstrates the semantics of
     *  \p device_reference's prefix decrement operator.
     *
     *  \code
     *  #include <thrust/device_vector.h>
     *  #include <assert.h>
     *  ...
     *  thrust::device_vector<int> v(1,0);
     *  thrust::device_ptr<int> ptr = &v[0];
     *  thrust::device_reference<int> ref(ptr);
     *
     *  // ref equals 0
     *  assert(ref == 0);
     *
     *  // the object pointed to by ptr equals 0
     *  assert(*ptr == 0);
     *
     *  // v[0] equals 0
     *  assert(v[0] == 0);
     *
     *  // decrement ref
     *  --ref;
     *
     *  // ref equals -1
     *  assert(ref == -1);
     *
     *  // the object pointed to by ptr equals -1
     *  assert(*ptr == -1);
     *
     *  // v[0] equals -1
     *  assert(v[0] == -1);
     *  \endcode
     *
     *  \note The decrement executes as if it were executed on the host.
     *  This may change in a later version.
     */
    device_reference &operator--(void);

    /*! Postfix decrement operator copies the object referenced by this
     *  \p device_reference, decrements the object referenced by this
     *  \p device_reference, and returns the copy.
     *
     *  \return A copy of the object referenced by this \p device_reference
     *          before being decremented.
     *
     *  The following code snippet demonstrates the semantics of
     *  \p device_reference's postfix decrement operator.
     *
     *  \code
     *  #include <thrust/device_vector.h>
     *  #include <assert.h>
     *  ...
     *  thrust::device_vector<int> v(1,0);
     *  thrust::device_ptr<int> ptr = &v[0];
     *  thrust::device_reference<int> ref(ptr);
     *
     *  // ref equals 0
     *  assert(ref == 0);
     *
     *  // the object pointed to by ptr equals 0
     *  assert(*ptr == 0);
     *
     *  // v[0] equals 0
     *  assert(v[0] == 0);
     *
     *  // decrement ref
     *  int x = ref--;
     *
     *  // x equals 0
     *  assert(x == 0)
     *
     *  // ref equals -1
     *  assert(ref == -1);
     *
     *  // the object pointed to by ptr equals -1
     *  assert(*ptr == -1);
     *
     *  // v[0] equals -1
     *  assert(v[0] == -1);
     *  \endcode
     *
     *  \note The decrement executes as if it were executed on the host.
     *  This may change in a later version.
     */
    value_type operator--(int);

    /*! Subtraction assignment operator subtract-assigns the object referenced by this
     *  \p device_reference and returns this \p device_reference.
     *
     *  \param rhs The right hand side of the subtraction-assignment.
     *  \return <tt>*this</tt>.
     *
     *  The following code snippet demonstrates the semantics of
     *  \p device_reference's addition assignment operator.
     *
     *  \code
     *  #include <thrust/device_vector.h>
     *  #include <assert.h>
     *  ...
     *  thrust::device_vector<int> v(1,0);
     *  thrust::device_ptr<int> ptr = &v[0];
     *  thrust::device_reference<int> ref(ptr);
     *
     *  // ref equals 0
     *  assert(ref == 0);
     *
     *  // the object pointed to by ptr equals 0
     *  assert(*ptr == 0);
     *
     *  // v[0] equals 0
     *  assert(v[0] == 0);
     *
     *  // subtract-assign ref
     *  ref -= 5;
     *
     *  // ref equals -5
     *  assert(ref == -5);
     *
     *  // the object pointed to by ptr equals -5
     *  assert(*ptr == -5);
     *
     *  // v[0] equals -5
     *  assert(v[0] == -5);
     *  \endcode
     *
     *  \note The subtract-assignment executes as as if it were executed on the host.
     *  This may change in a later version.
     */
    device_reference &operator-=(const T &rhs);

    /*! Multiplication assignment operator multiply-assigns the object referenced by this
     *  \p device_reference and returns this \p device_reference.
     *
     *  \param rhs The right hand side of the multiply-assignment.
     *  \return <tt>*this</tt>.
     *
     *  The following code snippet demonstrates the semantics of
     *  \p device_reference's multiply assignment operator.
     *
     *  \code
     *  #include <thrust/device_vector.h>
     *  #include <assert.h>
     *  ...
     *  thrust::device_vector<int> v(1,1);
     *  thrust::device_ptr<int> ptr = &v[0];
     *  thrust::device_reference<int> ref(ptr);
     *
     *  // ref equals 1
     *  assert(ref == 1);
     *
     *  // the object pointed to by ptr equals 1
     *  assert(*ptr == 1);
     *
     *  // v[0] equals 1
     *  assert(v[0] == 1);
     *
     *  // multiply-assign ref
     *  ref *= 5;
     *
     *  // ref equals 5
     *  assert(ref == 5);
     *
     *  // the object pointed to by ptr equals 5
     *  assert(*ptr == 5);
     *
     *  // v[0] equals 5
     *  assert(v[0] == 5);
     *  \endcode
     *
     *  \note The multiply-assignment executes as as if it were executed on the host.
     *  This may change in a later version.
     */
    device_reference &operator*=(const T &rhs);

    /*! Division assignment operator divide-assigns the object referenced by this
     *  \p device_reference and returns this \p device_reference.
     *
     *  \param rhs The right hand side of the divide-assignment.
     *  \return <tt>*this</tt>.
     *
     *  The following code snippet demonstrates the semantics of
     *  \p device_reference's divide assignment operator.
     *
     *  \code
     *  #include <thrust/device_vector.h>
     *  #include <assert.h>
     *  ...
     *  thrust::device_vector<int> v(1,5);
     *  thrust::device_ptr<int> ptr = &v[0];
     *  thrust::device_reference<int> ref(ptr);
     *
     *  // ref equals 5
     *  assert(ref == 5);
     *
     *  // the object pointed to by ptr equals 5
     *  assert(*ptr == 5);
     *
     *  // v[0] equals 5
     *  assert(v[0] == 5);
     *
     *  // divide-assign ref
     *  ref /= 5;
     *
     *  // ref equals 1
     *  assert(ref == 1);
     *
     *  // the object pointed to by ptr equals 1
     *  assert(*ptr == 1);
     *
     *  // v[0] equals 1
     *  assert(v[0] == 1);
     *  \endcode
     *
     *  \note The divide-assignment executes as as if it were executed on the host.
     *  This may change in a later version.
     */
    device_reference &operator/=(const T &rhs);

    /*! Modulation assignment operator modulus-assigns the object referenced by this
     *  \p device_reference and returns this \p device_reference.
     *
     *  \param rhs The right hand side of the divide-assignment.
     *  \return <tt>*this</tt>.
     *
     *  The following code snippet demonstrates the semantics of
     *  \p device_reference's divide assignment operator.
     *
     *  \code
     *  #include <thrust/device_vector.h>
     *  #include <assert.h>
     *  ...
     *  thrust::device_vector<int> v(1,5);
     *  thrust::device_ptr<int> ptr = &v[0];
     *  thrust::device_reference<int> ref(ptr);
     *
     *  // ref equals 5
     *  assert(ref == 5);
     *
     *  // the object pointed to by ptr equals 5
     *  assert(*ptr == 5);
     *
     *  // v[0] equals 5
     *  assert(v[0] == 5);
     *
     *  // modulus-assign ref
     *  ref %= 5;
     *
     *  // ref equals 0
     *  assert(ref == 0);
     *
     *  // the object pointed to by ptr equals 0
     *  assert(*ptr == 0);
     *
     *  // v[0] equals 0
     *  assert(v[0] == 0);
     *  \endcode
     *
     *  \note The modulus-assignment executes as as if it were executed on the host.
     *  This may change in a later version.
     */
    device_reference &operator%=(const T &rhs);

    /*! Bitwise left shift assignment operator left shift-assigns the object referenced by this
     *  \p device_reference and returns this \p device_reference.
     *
     *  \param rhs The right hand side of the left shift-assignment.
     *  \return <tt>*this</tt>.
     *
     *  The following code snippet demonstrates the semantics of
     *  \p device_reference's left shift assignment operator.
     *
     *  \code
     *  #include <thrust/device_vector.h>
     *  #include <assert.h>
     *  ...
     *  thrust::device_vector<int> v(1,1);
     *  thrust::device_ptr<int> ptr = &v[0];
     *  thrust::device_reference<int> ref(ptr);
     *
     *  // ref equals 1
     *  assert(ref == 1);
     *
     *  // the object pointed to by ptr equals 1
     *  assert(*ptr == 1);
     *
     *  // v[0] equals 1
     *  assert(v[0] == 1);
     *
     *  // left shift-assign ref
     *  ref <<= 1;
     *
     *  // ref equals 2
     *  assert(ref == 2);
     *
     *  // the object pointed to by ptr equals 2
     *  assert(*ptr == 2);
     *
     *  // v[0] equals 2
     *  assert(v[0] == 2);
     *  \endcode
     *
     *  \note The left shift-assignment executes as as if it were executed on the host.
     *  This may change in a later version.
     */
    device_reference &operator<<=(const T &rhs);

    /*! Bitwise right shift assignment operator right shift-assigns the object referenced by this
     *  \p device_reference and returns this \p device_reference.
     *
     *  \param rhs The right hand side of the right shift-assignment.
     *  \return <tt>*this</tt>.
     *
     *  The following code snippet demonstrates the semantics of
     *  \p device_reference's right shift assignment operator.
     *
     *  \code
     *  #include <thrust/device_vector.h>
     *  #include <assert.h>
     *  ...
     *  thrust::device_vector<int> v(1,2);
     *  thrust::device_ptr<int> ptr = &v[0];
     *  thrust::device_reference<int> ref(ptr);
     *
     *  // ref equals 2
     *  assert(ref == 2);
     *
     *  // the object pointed to by ptr equals 2
     *  assert(*ptr == 2);
     *
     *  // v[0] equals 2
     *  assert(v[0] == 2);
     *
     *  // right shift-assign ref
     *  ref >>= 1;
     *
     *  // ref equals 1
     *  assert(ref == 1);
     *
     *  // the object pointed to by ptr equals 1
     *  assert(*ptr == 1);
     *
     *  // v[0] equals 1
     *  assert(v[0] == 1);
     *  \endcode
     *
     *  \note The right shift-assignment executes as as if it were executed on the host.
     *  This may change in a later version.
     */
    device_reference &operator>>=(const T &rhs);

    /*! Bitwise AND assignment operator AND-assigns the object referenced by this
     *  \p device_reference and returns this \p device_reference.
     *
     *  \param rhs The right hand side of the AND-assignment.
     *  \return <tt>*this</tt>.
     *
     *  The following code snippet demonstrates the semantics of
     *  \p device_reference's AND assignment operator.
     *
     *  \code
     *  #include <thrust/device_vector.h>
     *  #include <assert.h>
     *  ...
     *  thrust::device_vector<int> v(1,1);
     *  thrust::device_ptr<int> ptr = &v[0];
     *  thrust::device_reference<int> ref(ptr);
     *
     *  // ref equals 1
     *  assert(ref == 1);
     *
     *  // the object pointed to by ptr equals 1
     *  assert(*ptr == 1);
     *
     *  // v[0] equals 1
     *  assert(v[0] == 1);
     *
     *  // right AND-assign ref
     *  ref &= 0;
     *
     *  // ref equals 0
     *  assert(ref == 0);
     *
     *  // the object pointed to by ptr equals 0
     *  assert(*ptr == 0);
     *
     *  // v[0] equals 0
     *  assert(v[0] == 0);
     *  \endcode
     *
     *  \note The AND-assignment executes as as if it were executed on the host.
     *  This may change in a later version.
     */
    device_reference &operator&=(const T &rhs);

    /*! Bitwise OR assignment operator OR-assigns the object referenced by this
     *  \p device_reference and returns this \p device_reference.
     *
     *  \param rhs The right hand side of the OR-assignment.
     *  \return <tt>*this</tt>.
     *
     *  The following code snippet demonstrates the semantics of
     *  \p device_reference's OR assignment operator.
     *
     *  \code
     *  #include <thrust/device_vector.h>
     *  #include <assert.h>
     *  ...
     *  thrust::device_vector<int> v(1,0);
     *  thrust::device_ptr<int> ptr = &v[0];
     *  thrust::device_reference<int> ref(ptr);
     *
     *  // ref equals 0
     *  assert(ref == 0);
     *
     *  // the object pointed to by ptr equals 0
     *  assert(*ptr == 0);
     *
     *  // v[0] equals 0
     *  assert(v[0] == 0);
     *
     *  // right OR-assign ref
     *  ref |= 1;
     *
     *  // ref equals 1
     *  assert(ref == 1);
     *
     *  // the object pointed to by ptr equals 1
     *  assert(*ptr == 1);
     *
     *  // v[0] equals 1
     *  assert(v[0] == 1);
     *  \endcode
     *
     *  \note The OR-assignment executes as as if it were executed on the host.
     *  This may change in a later version.
     */
    device_reference &operator|=(const T &rhs);

    /*! Bitwise XOR assignment operator XOR-assigns the object referenced by this
     *  \p device_reference and returns this \p device_reference.
     *
     *  \param rhs The right hand side of the XOR-assignment.
     *  \return <tt>*this</tt>.
     *
     *  The following code snippet demonstrates the semantics of
     *  \p device_reference's XOR assignment operator.
     *
     *  \code
     *  #include <thrust/device_vector.h>
     *  #include <assert.h>
     *  ...
     *  thrust::device_vector<int> v(1,1);
     *  thrust::device_ptr<int> ptr = &v[0];
     *  thrust::device_reference<int> ref(ptr);
     *
     *  // ref equals 1
     *  assert(ref == 1);
     *
     *  // the object pointed to by ptr equals 1
     *  assert(*ptr == 1);
     *
     *  // v[0] equals 1
     *  assert(v[0] == 1);
     *
     *  // right XOR-assign ref
     *  ref ^= 1;
     *
     *  // ref equals 0
     *  assert(ref == 0);
     *
     *  // the object pointed to by ptr equals 0
     *  assert(*ptr == 0);
     *
     *  // v[0] equals 0
     *  assert(v[0] == 0);
     *  \endcode
     *
     *  \note The XOR-assignment executes as as if it were executed on the host.
     *  This may change in a later version.
     */
    device_reference &operator^=(const T &rhs);
#endif // end doxygen-only members
}; // end device_reference

/*! swaps the value of one \p device_reference with another.
 *  \p x The first \p device_reference of interest.
 *  \p y The second \p device_reference of interest.
 */
template<typename T>
__host__ __device__
void swap(device_reference<T>& x, device_reference<T>& y)
{
  x.swap(y);
}

// declare these methods for the purpose of Doxygenating them
// they actually are defined for a derived-from class
#if THRUST_DOXYGEN
/*! Writes to an output stream the value of a \p device_reference.
 *
 *  \param os The output stream.
 *  \param y The \p device_reference to output.
 *  \return os.
 */
template<typename T, typename charT, typename traits>
std::basic_ostream<charT, traits> &
operator<<(std::basic_ostream<charT, traits> &os, const device_reference<T> &y);
#endif

/*! \} // memory_management
 */

THRUST_NAMESPACE_END
