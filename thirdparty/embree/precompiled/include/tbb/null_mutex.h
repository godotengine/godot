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

#ifndef __TBB_null_mutex_H
#define __TBB_null_mutex_H

#include "tbb_stddef.h"

namespace tbb {

//! A mutex which does nothing
/** A null_mutex does no operation and simulates success.
    @ingroup synchronization */
class null_mutex : internal::mutex_copy_deprecated_and_disabled {
public:
    //! Represents acquisition of a mutex.
    class scoped_lock : internal::no_copy {
    public:
        scoped_lock() {}
        scoped_lock( null_mutex& ) {}
        ~scoped_lock() {}
        void acquire( null_mutex& ) {}
        bool try_acquire( null_mutex& ) { return true; }
        void release() {}
    };

    null_mutex() {}

    // Mutex traits
    static const bool is_rw_mutex = false;
    static const bool is_recursive_mutex = true;
    static const bool is_fair_mutex = true;
};

}

#endif /* __TBB_null_mutex_H */
