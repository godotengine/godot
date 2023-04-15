/*
    Copyright (c) 2005-2020 Intel Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#include "internal/_deprecated_header_message_guard.h"

#if !defined(__TBB_show_deprecation_message_runtime_loader_H) && defined(__TBB_show_deprecated_header_message)
#define  __TBB_show_deprecation_message_runtime_loader_H
#pragma message("TBB Warning: tbb/runtime_loader.h is deprecated. For details, please see Deprecated Features appendix in the TBB reference manual.")
#endif

#if defined(__TBB_show_deprecated_header_message)
#undef __TBB_show_deprecated_header_message
#endif

#ifndef __TBB_runtime_loader_H
#define __TBB_runtime_loader_H

#define __TBB_runtime_loader_H_include_area
#include "internal/_warning_suppress_enable_notice.h"

#if ! TBB_PREVIEW_RUNTIME_LOADER
    #error Set TBB_PREVIEW_RUNTIME_LOADER to include runtime_loader.h
#endif

#include "tbb_stddef.h"
#include <climits>

#if _MSC_VER
    #if ! __TBB_NO_IMPLICIT_LINKAGE
        #ifdef _DEBUG
            #pragma comment( linker, "/nodefaultlib:tbb_debug.lib" )
            #pragma comment( linker, "/defaultlib:tbbproxy_debug.lib" )
        #else
            #pragma comment( linker, "/nodefaultlib:tbb.lib" )
            #pragma comment( linker, "/defaultlib:tbbproxy.lib" )
        #endif
    #endif
#endif

namespace tbb {

namespace interface6 {

//! Load TBB at runtime.
/*!

\b Usage:

In source code:

\code
#include "tbb/runtime_loader.h"

char const * path[] = { "<install dir>/lib/ia32", NULL };
tbb::runtime_loader loader( path );

// Now use TBB.
\endcode

Link with \c tbbproxy.lib (or \c libtbbproxy.a) instead of \c tbb.lib (\c libtbb.dylib,
\c libtbb.so).

TBB library will be loaded at runtime from \c <install dir>/lib/ia32 directory.

\b Attention:

All \c runtime_loader objects (in the same module, i.e. exe or dll) share some global state.
The most noticeable piece of global state is loaded TBB library.
There are some implications:

    -   Only one TBB library can be loaded per module.

    -   If one object has already loaded TBB library, another object will not load TBB.
        If the loaded TBB library is suitable for the second object, both will use TBB
        cooperatively, otherwise the second object will report an error.

    -   \c runtime_loader objects will not work (correctly) in parallel due to absence of
        synchronization.

*/

class __TBB_DEPRECATED_IN_VERBOSE_MODE runtime_loader : tbb::internal::no_copy {

    public:

        //! Error mode constants.
        enum error_mode {
            em_status,     //!< Save status of operation and continue.
            em_throw,      //!< Throw an exception of tbb::runtime_loader::error_code type.
            em_abort       //!< Print message to \c stderr and call \c abort().
        }; // error_mode

        //! Error codes.
        enum error_code {
            ec_ok,         //!< No errors.
            ec_bad_call,   //!< Invalid function call (e. g. load() called when TBB is already loaded).
            ec_bad_arg,    //!< Invalid argument passed.
            ec_bad_lib,    //!< Invalid library found (e. g. \c TBB_runtime_version symbol not found).
            ec_bad_ver,    //!< TBB found but version is not suitable.
            ec_no_lib      //!< No suitable TBB library found.
        }; // error_code

        //! Initialize object but do not load TBB.
        runtime_loader( error_mode mode = em_abort );

        //! Initialize object and load TBB.
        /*!
            See load() for details.

            If error mode is \c em_status, call status() to check whether TBB was loaded or not.
        */
        runtime_loader(
            char const * path[],                           //!< List of directories to search TBB in.
            int          min_ver = TBB_INTERFACE_VERSION,  //!< Minimal suitable version of TBB.
            int          max_ver = INT_MAX,                //!< Maximal suitable version of TBB.
            error_mode   mode    = em_abort                //!< Error mode for this object.
        );

        //! Destroy object.
        ~runtime_loader();

        //! Load TBB.
        /*!
            The method searches the directories specified in \c path[] array for the TBB library.
            When the library is found, it is loaded and its version is checked. If the version is
            not suitable, the library is unloaded, and the search continues.

            \b Note:

            For security reasons, avoid using relative directory names. For example, never load
            TBB from current (\c "."), parent (\c "..") or any other relative directory (like
            \c "lib" ). Use only absolute directory names (e. g. "/usr/local/lib").

            For the same security reasons, avoid using system default directories (\c "") on
            Windows. (See http://www.microsoft.com/technet/security/advisory/2269637.mspx for
            details.)

            Neglecting these rules may cause your program to execute 3-rd party malicious code.

            \b Errors:
                -   \c ec_bad_call - TBB already loaded by this object.
                -   \c ec_bad_arg - \p min_ver and/or \p max_ver negative or zero,
                    or \p min_ver > \p max_ver.
                -   \c ec_bad_ver - TBB of unsuitable version already loaded by another object.
                -   \c ec_no_lib - No suitable library found.
        */
        error_code
        load(
            char const * path[],                           //!< List of directories to search TBB in.
            int          min_ver = TBB_INTERFACE_VERSION,  //!< Minimal suitable version of TBB.
            int          max_ver = INT_MAX                 //!< Maximal suitable version of TBB.

        );


        //! Report status.
        /*!
            If error mode is \c em_status, the function returns status of the last operation.
        */
        error_code status();

    private:

        error_mode const my_mode;
        error_code       my_status;
        bool             my_loaded;

}; // class runtime_loader

} // namespace interface6

using interface6::runtime_loader;

} // namespace tbb

#include "internal/_warning_suppress_disable_notice.h"
#undef __TBB_runtime_loader_H_include_area

#endif /* __TBB_runtime_loader_H */

