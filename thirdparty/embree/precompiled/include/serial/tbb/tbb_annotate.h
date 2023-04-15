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

#ifndef __TBB_annotate_H
#define __TBB_annotate_H

// Macros used by the Intel(R) Parallel Advisor.
#ifdef __TBB_NORMAL_EXECUTION
    #define ANNOTATE_SITE_BEGIN( site )
    #define ANNOTATE_SITE_END( site )
    #define ANNOTATE_TASK_BEGIN( task )
    #define ANNOTATE_TASK_END( task )
    #define ANNOTATE_LOCK_ACQUIRE( lock )
    #define ANNOTATE_LOCK_RELEASE( lock )
#else
    #include <advisor-annotate.h>
#endif

#endif /* __TBB_annotate_H */
