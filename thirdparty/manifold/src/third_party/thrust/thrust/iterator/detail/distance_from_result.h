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

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/type_traits.h>

THRUST_NAMESPACE_BEGIN

namespace detail
{

// since both arguments are known to be specializations of iterator_facade,
// it's legal to access IteratorFacade2::difference_type
template<typename IteratorFacade1, typename IteratorFacade2>
  struct distance_from_result
    : eval_if<
        is_convertible<IteratorFacade2,IteratorFacade1>::value,
        identity_<typename IteratorFacade1::difference_type>,
        identity_<typename IteratorFacade2::difference_type>
      >
{};

} // end detail

THRUST_NAMESPACE_END

