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

#if !defined(__TBB_show_deprecation_message_aligned_space_H) && defined(__TBB_show_deprecated_header_message)
#define  __TBB_show_deprecation_message_aligned_space_H
#pragma message("TBB Warning: tbb/aligned_space.h is deprecated. For details, please see Deprecated Features appendix in the TBB reference manual.")
#endif

#if defined(__TBB_show_deprecated_header_message)
#undef __TBB_show_deprecated_header_message
#endif

#ifndef __TBB_aligned_space_H
#define __TBB_aligned_space_H

#define __TBB_aligned_space_H_include_area
#include "internal/_warning_suppress_enable_notice.h"

#include "tbb_stddef.h"
#include "tbb_machine.h"

namespace tbb {

//! Block of space aligned sufficiently to construct an array T with N elements.
/** The elements are not constructed or destroyed by this class.
    @ingroup memory_allocation */
template<typename T,size_t N=1>
class __TBB_DEPRECATED_IN_VERBOSE_MODE_MSG("tbb::aligned_space is deprecated, use std::aligned_storage") aligned_space {
private:
    typedef __TBB_TypeWithAlignmentAtLeastAsStrict(T) element_type;
    element_type array[(sizeof(T)*N+sizeof(element_type)-1)/sizeof(element_type)];
public:
    //! Pointer to beginning of array
    T* begin() const {return internal::punned_cast<T*>(this);}

    //! Pointer to one past last element in array.
    T* end() const {return begin()+N;}
};

} // namespace tbb

#include "internal/_warning_suppress_disable_notice.h"
#undef __TBB_aligned_space_H_include_area

#endif /* __TBB_aligned_space_H */
