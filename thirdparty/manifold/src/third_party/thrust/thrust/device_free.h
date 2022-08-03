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
 *  \brief Deallocates storage allocated by \p device_malloc.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/device_ptr.h>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup memory_management Memory Management
 *  \{
 */

/*! \p device_free deallocates memory allocated by the function \p device_malloc.
 *
 *  \param ptr A \p device_ptr pointing to memory to be deallocated.
 *
 *  The following code snippet demonstrates how to use \p device_free to
 *  deallocate memory allocated by \p device_malloc.
 *
 *  \code
 *  #include <thrust/device_malloc.h>
 *  #include <thrust/device_free.h>
 *  ...
 *  // allocate some integers with device_malloc
 *  const int N = 100;
 *  thrust::device_ptr<int> int_array = thrust::device_malloc<int>(N);
 *
 *  // manipulate integers
 *  ...
 *
 *  // deallocate with device_free
 *  thrust::device_free(int_array);
 *  \endcode
 *
 *  \see device_ptr
 *  \see device_malloc
 */
inline void device_free(thrust::device_ptr<void> ptr);

/*! \} // memory_management
 */

THRUST_NAMESPACE_END

#include <thrust/detail/device_free.inl>

