/*
 *  Copyright 2018-2021 NVIDIA Corporation
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
 *  \brief C++20's
 *  <a href="https://en.cppreference.com/w/cpp/types/remove_cvref">std::remove_cvref</a>.
 */

#pragma once

#include <thrust/detail/config.h>

#if  THRUST_CPP_DIALECT >= 2017
#if __has_include(<version>)
#  include <version>
#endif
#endif

#include <type_traits>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup utility
 *  \{
 */

/*! \addtogroup type_traits Type Traits
 *  \{
 */

/*! \brief <a href="https://en.cppreference.com/w/cpp/named_req/UnaryTypeTrait"><i>UnaryTypeTrait</i></a>
 *  that removes
 *  <a href="https://en.cppreference.com/w/cpp/language/cv">const-volatile qualifiers</a>
 *  and
 *  <a href="https://en.cppreference.com/w/cpp/language/reference">references</a>
 *  from \c T.
 *  Equivalent to \c remove_cv_t<remove_reference_t<T>>.
 *
 *  \see <a href="https://en.cppreference.com/w/cpp/types/remove_cvref">std::remove_cvref</a>
 *  \see <a href="https://en.cppreference.com/w/cpp/types/remove_cv">std::remove_cv</a>
 *  \see <a href="https://en.cppreference.com/w/cpp/types/remove_cv">std::remove_const</a>
 *  \see <a href="https://en.cppreference.com/w/cpp/types/remove_cv">std::remove_volatile</a>
 *  \see <a href="https://en.cppreference.com/w/cpp/types/remove_cv">std::remove_reference</a>
 */
#if defined(__cpp_lib_remove_cvref) && (__cpp_lib_remove_cvref >= 201711L)
using std::remove_cvref;
#else // Older than C++20.
template <typename T>
struct remove_cvref
{
  using type = typename std::remove_cv<
    typename std::remove_reference<T>::type
  >::type;
};
#endif

/*! \brief Type alias that removes
 *  <a href="https://en.cppreference.com/w/cpp/language/cv">const-volatile qualifiers</a>
 *  and
 *  <a href="https://en.cppreference.com/w/cpp/language/reference">references</a>
 *  from \c T.
 *  Equivalent to \c remove_cv_t<remove_reference_t<T>>.
 *
 *  \see <a href="https://en.cppreference.com/w/cpp/types/remove_cvref">std::remove_cvref</a>
 *  \see <a href="https://en.cppreference.com/w/cpp/types/remove_cv">std::remove_cv</a>
 *  \see <a href="https://en.cppreference.com/w/cpp/types/remove_cv">std::remove_const</a>
 *  \see <a href="https://en.cppreference.com/w/cpp/types/remove_cv">std::remove_volatile</a>
 *  \see <a href="https://en.cppreference.com/w/cpp/types/remove_cv">std::remove_reference</a>
 */
#if defined(__cpp_lib_remove_cvref) && (__cpp_lib_remove_cvref >= 201711L)
using std::remove_cvref_t;
#else // Older than C++20.
template <typename T>
using remove_cvref_t = typename remove_cvref<T>::type;
#endif

/*! \} // type traits
 */

/*! \} // utility
 */

THRUST_NAMESPACE_END

