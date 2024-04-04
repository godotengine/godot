// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include "parallel_reduce.h"

namespace embree
{
  
  template<typename Index, class UnaryPredicate>
    __forceinline bool parallel_any_of (Index first, Index last, UnaryPredicate pred)
  {
    std::atomic_bool ret;
    ret = false;
    
#if defined(TASKING_TBB)
#if TBB_INTERFACE_VERSION >= 12002
    tbb::task_group_context context;
    tbb::parallel_for(tbb::blocked_range<size_t>{first, last}, [&ret,pred,&context](const tbb::blocked_range<size_t>& r) {
        if (context.is_group_execution_cancelled()) return;
        for (size_t i = r.begin(); i != r.end(); ++i) {
          if (pred(i)) {
            ret = true;
            context.cancel_group_execution();
          }
        }
      });
#else
    tbb::parallel_for(tbb::blocked_range<size_t>{first, last}, [&ret,pred](const tbb::blocked_range<size_t>& r) {
        if (tbb::task::self().is_cancelled()) return;
        for (size_t i = r.begin(); i != r.end(); ++i) {
          if (pred(i)) {
            ret = true;
            tbb::task::self().cancel_group_execution();
          }
        }
      });
#endif
#else
    ret = parallel_reduce (first, last, false, [pred](const range<size_t>& r)->bool {
        bool localret = false;
        for (auto i=r.begin(); i<r.end(); ++i) {
          localret |= pred(i);
        }
        return localret;
      },
      std::bit_or<bool>()
      );
#endif
    
    return ret;
  }
  
} // end namespace
