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

#include "../internal/_deprecated_header_message_guard.h"

#if !defined(__TBB_show_deprecation_message_ppl_H) && defined(__TBB_show_deprecated_header_message)
#define  __TBB_show_deprecation_message_ppl_H
#pragma message("TBB Warning: tbb/compat/ppl.h is deprecated. For details, please see Deprecated Features appendix in the TBB reference manual.")
#endif

#if defined(__TBB_show_deprecated_header_message)
#undef __TBB_show_deprecated_header_message
#endif

#ifndef __TBB_compat_ppl_H
#define __TBB_compat_ppl_H

#define __TBB_ppl_H_include_area
#include "../internal/_warning_suppress_enable_notice.h"

#include "../task_group.h"
#include "../parallel_invoke.h"
#include "../parallel_for_each.h"
#include "../parallel_for.h"
#include "../tbb_exception.h"
#include "../critical_section.h"
#include "../reader_writer_lock.h"
#include "../combinable.h"

namespace Concurrency {

#if __TBB_TASK_GROUP_CONTEXT
    using tbb::task_handle;
    using tbb::task_group_status;
    using tbb::task_group;
    using tbb::structured_task_group;
    using tbb::invalid_multiple_scheduling;
    using tbb::missing_wait;
    using tbb::make_task;

    using tbb::not_complete;
    using tbb::complete;
    using tbb::canceled;

    using tbb::is_current_task_group_canceling;
#endif /* __TBB_TASK_GROUP_CONTEXT */

    using tbb::parallel_invoke;
    using tbb::strict_ppl::parallel_for;
    using tbb::parallel_for_each;
    using tbb::critical_section;
    using tbb::reader_writer_lock;
    using tbb::combinable;

    using tbb::improper_lock;

} // namespace Concurrency

#include "../internal/_warning_suppress_disable_notice.h"
#undef __TBB_ppl_H_include_area

#endif /* __TBB_compat_ppl_H */
