/*
 *  Copyright 2021 NVIDIA Corporation
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

/**
 * \file namespace.h
 * \brief Utilities that allow `thrust::` to be placed inside an
 * application-specific namespace.
 */

/**
 * \def THRUST_CUB_WRAPPED_NAMESPACE
 * If defined, this value will be used as the name of a namespace that wraps the
 * `thrust::` and `cub::` namespaces.
 * This macro should not be used with any other Thrust namespace macros.
 */
#ifdef THRUST_CUB_WRAPPED_NAMESPACE
#define THRUST_WRAPPED_NAMESPACE THRUST_CUB_WRAPPED_NAMESPACE
#endif

/**
 * \def THRUST_WRAPPED_NAMESPACE
 * If defined, this value will be used as the name of a namespace that wraps the
 * `thrust::` namespace.
 * If THRUST_CUB_WRAPPED_NAMESPACE is set, this will inherit that macro's value.
 * This macro should not be used with any other Thrust namespace macros.
 */
#ifdef THRUST_WRAPPED_NAMESPACE
#define THRUST_NS_PREFIX                                                       \
  namespace THRUST_WRAPPED_NAMESPACE                                           \
  {

#define THRUST_NS_POSTFIX }

#define THRUST_NS_QUALIFIER ::THRUST_WRAPPED_NAMESPACE::thrust
#endif

/**
 * \def THRUST_NS_PREFIX
 * This macro is inserted prior to all `namespace thrust { ... }` blocks. It is
 * derived from THRUST_WRAPPED_NAMESPACE, if set, and will be empty otherwise.
 * It may be defined by users, in which case THRUST_NS_PREFIX,
 * THRUST_NS_POSTFIX, and THRUST_NS_QUALIFIER must all be set consistently.
 */
#ifndef THRUST_NS_PREFIX
#define THRUST_NS_PREFIX
#endif

/**
 * \def THRUST_NS_POSTFIX
 * This macro is inserted following the closing braces of all
 * `namespace thrust { ... }` block. It is defined appropriately when
 * THRUST_WRAPPED_NAMESPACE is set, and will be empty otherwise. It may be
 * defined by users, in which case THRUST_NS_PREFIX, THRUST_NS_POSTFIX, and
 * THRUST_NS_QUALIFIER must all be set consistently.
 */
#ifndef THRUST_NS_POSTFIX
#define THRUST_NS_POSTFIX
#endif

/**
 * \def THRUST_NS_QUALIFIER
 * This macro is used to qualify members of thrust:: when accessing them from
 * outside of their namespace. By default, this is just `::thrust`, and will be
 * set appropriately when THRUST_WRAPPED_NAMESPACE is defined. This macro may be
 * defined by users, in which case THRUST_NS_PREFIX, THRUST_NS_POSTFIX, and
 * THRUST_NS_QUALIFIER must all be set consistently.
 */
#ifndef THRUST_NS_QUALIFIER
#define THRUST_NS_QUALIFIER ::thrust
#endif

/**
 * \def THRUST_NAMESPACE_BEGIN
 * This macro is used to open a `thrust::` namespace block, along with any
 * enclosing namespaces requested by THRUST_WRAPPED_NAMESPACE, etc.
 * This macro is defined by Thrust and may not be overridden.
 */
#define THRUST_NAMESPACE_BEGIN                                                 \
  THRUST_NS_PREFIX                                                             \
  namespace thrust                                                             \
  {

/**
 * \def THRUST_NAMESPACE_END
 * This macro is used to close a `thrust::` namespace block, along with any
 * enclosing namespaces requested by THRUST_WRAPPED_NAMESPACE, etc.
 * This macro is defined by Thrust and may not be overridden.
 */
#define THRUST_NAMESPACE_END                                                   \
  } /* end namespace thrust */                                                 \
  THRUST_NS_POSTFIX

// The following is just here to add docs for the thrust namespace:

THRUST_NS_PREFIX

/*! \namespace thrust
 *  \brief \p thrust is the top-level namespace which contains all Thrust
 *         functions and types.
 */
namespace thrust
{
}

THRUST_NS_POSTFIX
