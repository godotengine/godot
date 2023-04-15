// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(TASKING_INTERNAL)
#  include "taskschedulerinternal.h"
#elif defined(TASKING_TBB)
#  include "taskschedulertbb.h"
#elif defined(TASKING_PPL)
#  include "taskschedulerppl.h"
#else
#  error "no tasking system enabled"
#endif

