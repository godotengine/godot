/*
 *  Copyright 2008-2022 NVIDIA Corporation
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

/*! \file version.h
 *  \brief Compile-time macros encoding Thrust release version
 *
 *         <thrust/version.h> is the only Thrust header that is guaranteed to
 *         change with every thrust release.
 *
 *         It is also the only header that does not cause THRUST_HOST_SYSTEM
 *         and THRUST_DEVICE_SYSTEM to be defined. This way, a user may include
 *         this header and inspect THRUST_VERSION before programatically defining
 *         either of these macros herself.
 */

#pragma once

//  This is the only Thrust header that is guaranteed to
//  change with every Thrust release.
//
//  THRUST_VERSION % 100 is the sub-minor version
//  THRUST_VERSION / 100 % 1000 is the minor version
//  THRUST_VERSION / 100000 is the major version
//
//  Because this header does not #include <thrust/detail/config.h>,
//  it is the only Thrust header that does not cause
//  THRUST_HOST_SYSTEM and THRUST_DEVICE_SYSTEM to be defined.

/*! \def THRUST_VERSION
 *  \brief The preprocessor macro \p THRUST_VERSION encodes the version
 *         number of the Thrust library.
 *
 *         <tt>THRUST_VERSION % 100</tt> is the sub-minor version.
 *         <tt>THRUST_VERSION / 100 % 1000</tt> is the minor version.
 *         <tt>THRUST_VERSION / 100000</tt> is the major version.
 */
#define THRUST_VERSION 200100

/*! \def THRUST_MAJOR_VERSION
 *  \brief The preprocessor macro \p THRUST_MAJOR_VERSION encodes the
 *         major version number of the Thrust library.
 */
#define THRUST_MAJOR_VERSION     (THRUST_VERSION / 100000)

/*! \def THRUST_MINOR_VERSION
 *  \brief The preprocessor macro \p THRUST_MINOR_VERSION encodes the
 *         minor version number of the Thrust library.
 */
#define THRUST_MINOR_VERSION     (THRUST_VERSION / 100 % 1000)

/*! \def THRUST_SUBMINOR_VERSION
 *  \brief The preprocessor macro \p THRUST_SUBMINOR_VERSION encodes the
 *         sub-minor version number of the Thrust library.
 */
#define THRUST_SUBMINOR_VERSION  (THRUST_VERSION % 100)

/*! \def THRUST_PATCH_NUMBER
 *  \brief The preprocessor macro \p THRUST_PATCH_NUMBER encodes the
 *         patch number of the Thrust library.
 *         Legacy; will be 0 for all future releases.
 */
#define THRUST_PATCH_NUMBER 0
