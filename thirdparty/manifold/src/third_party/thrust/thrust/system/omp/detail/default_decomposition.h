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


/*! \file default_decomposition.h
 *  \brief Return a decomposition that is appropriate for the OpenMP backend.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/system/detail/internal/decompose.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace omp
{
namespace detail
{

template <typename IndexType>
thrust::system::detail::internal::uniform_decomposition<IndexType> default_decomposition(IndexType n);

} // end namespace detail
} // end namespace omp
} // end namespace system
THRUST_NAMESPACE_END

#include <thrust/system/omp/detail/default_decomposition.inl>

