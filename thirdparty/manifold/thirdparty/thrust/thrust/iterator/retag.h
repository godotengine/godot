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

/*! \file thrust/iterator/retag.h
 *  \brief Functionality for altering an iterator's associated system.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/iterator/detail/retag.h>

THRUST_NAMESPACE_BEGIN


/*! \ingroup iterator_tags
 *  \{
 */

#if 0
/*! \p reinterpret_tag returns a copy of an iterator and changes the type of the result's system tag.
 *  \tparam Tag Any system tag.
 *  \tparam Iterator Any iterator type.
 *  \param iter The iterator of interest.
 *  \return An iterator of unspecified type whose system tag is \p Tag and whose behavior is otherwise
 *          equivalent to \p iter.
 *  \note Unlike \p retag, \p reinterpret_tag does not enforce that the converted-to system tag be
 *        related to the converted-from system tag.
 *  \see retag
 */
template<typename Tag, typename Iterator>
__host__ __device__
unspecified_iterator_type reinterpret_tag(Iterator iter);

/*! \p retag returns a copy of an iterator and changes the type of the result's system tag.
 *  \tparam Tag \p Tag shall be convertible to <tt>thrust::iterator_system<Iterator>::type</tt>,
 *              or <tt>thrust::iterator_system<Iterator>::type</tt> is a base type of \p Tag.
 *  \tparam Iterator Any iterator type.
 *  \param iter The iterator of interest.
 *  \return An iterator of unspecified type whose system tag is \p Tag and whose behavior is
 *          otherwise equivalent to \p iter.
 *  \note Unlike \p reinterpret_tag, \p retag enforces that the converted-to system tag be
 *        related to the converted-from system tag.
 *  \see reinterpret_tag
 */
template<typename Tag, typename Iterator>
__host__ __device__
unspecified_iterator_type retag(Iterator iter);
#endif

/*! \} // iterator_tags
 */


THRUST_NAMESPACE_END

