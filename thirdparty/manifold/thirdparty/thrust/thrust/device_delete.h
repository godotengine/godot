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
 *  \brief Deletes variables in device memory.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/device_ptr.h>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup memory_management Memory Management
 *  \{
 */

/*! \p device_delete deletes a \p device_ptr allocated with
 *  \p device_new.
 *
 *  \param ptr The \p device_ptr to delete, assumed to have
 *         been allocated with \p device_new.
 *  \param n The number of objects to destroy at \p ptr. Defaults to \c 1
 *         similar to \p device_new.
 *
 *  \see device_ptr
 *  \see device_new
 */
template<typename T>
  inline void device_delete(thrust::device_ptr<T> ptr,
                            const size_t n = 1);

/*! \} // memory_management
 */

THRUST_NAMESPACE_END

#include <thrust/detail/device_delete.inl>

