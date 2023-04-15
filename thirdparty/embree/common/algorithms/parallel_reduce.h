// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "parallel_for.h"

namespace embree
{
  template<typename Index, typename Value, typename Func, typename Reduction>
    __forceinline Value sequential_reduce( const Index first, const Index last, const Value& identity, const Func& func, const Reduction& reduction ) 
  {
    return func(range<Index>(first,last));
  }

  template<typename Index, typename Value, typename Func, typename Reduction>
    __forceinline Value sequential_reduce( const Index first, const Index last, const Index minStepSize, const Value& identity, const Func& func, const Reduction& reduction )
  {
    return func(range<Index>(first,last));
  }

  template<typename Index, typename Value, typename Func, typename Reduction>
    __noinline Value parallel_reduce_internal( Index taskCount, const Index first, const Index last, const Index minStepSize, const Value& identity, const Func& func, const Reduction& reduction )
  {
    const Index maxTasks = 512;
    const Index threadCount = (Index) TaskScheduler::threadCount();
    taskCount = min(taskCount,threadCount,maxTasks);

    /* parallel invocation of all tasks */
    dynamic_large_stack_array(Value,values,taskCount,8192); // consumes at most 8192 bytes on the stack
    parallel_for(taskCount, [&](const Index taskIndex) {
        const Index k0 = first+(taskIndex+0)*(last-first)/taskCount;
        const Index k1 = first+(taskIndex+1)*(last-first)/taskCount;
        values[taskIndex] = func(range<Index>(k0,k1));
      });

    /* perform reduction over all tasks */
    Value v = identity;
    for (Index i=0; i<taskCount; i++) v = reduction(v,values[i]);
    return v;
  }

  template<typename Index, typename Value, typename Func, typename Reduction>
    __forceinline Value parallel_reduce( const Index first, const Index last, const Index minStepSize, const Value& identity, const Func& func, const Reduction& reduction )
  {
#if defined(TASKING_INTERNAL)

    /* fast path for small number of iterations */
    Index taskCount = (last-first+minStepSize-1)/minStepSize;
    if (likely(taskCount == 1)) {
      return func(range<Index>(first,last));
    }
    return parallel_reduce_internal(taskCount,first,last,minStepSize,identity,func,reduction);

#elif defined(TASKING_TBB)
  #if TBB_INTERFACE_VERSION >= 12002
    tbb::task_group_context context;
    const Value v = tbb::parallel_reduce(tbb::blocked_range<Index>(first,last,minStepSize),identity,
      [&](const tbb::blocked_range<Index>& r, const Value& start) { return reduction(start,func(range<Index>(r.begin(),r.end()))); },
      reduction,context);
    // -- GODOT start --
    // if (context.is_group_execution_cancelled())
    //   throw std::runtime_error("task cancelled");
    // -- GODOT end --
    return v;
  #else
    const Value v = tbb::parallel_reduce(tbb::blocked_range<Index>(first,last,minStepSize),identity,
      [&](const tbb::blocked_range<Index>& r, const Value& start) { return reduction(start,func(range<Index>(r.begin(),r.end()))); },
      reduction);
    // -- GODOT start --
    // if (tbb::task::self().is_cancelled())
    //   throw std::runtime_error("task cancelled");
    // -- GODOT end --
    return v;
  #endif
#else // TASKING_PPL
    struct AlignedValue
    {
      char storage[__alignof(Value)+sizeof(Value)];
      static uintptr_t alignUp(uintptr_t p, size_t a) { return p + (~(p - 1) % a); };
      Value* getValuePtr() { return reinterpret_cast<Value*>(alignUp(uintptr_t(storage), __alignof(Value))); }
      const Value* getValuePtr() const { return reinterpret_cast<Value*>(alignUp(uintptr_t(storage), __alignof(Value))); }
      AlignedValue(const Value& v) { new(getValuePtr()) Value(v); }
      AlignedValue(const AlignedValue& v) { new(getValuePtr()) Value(*v.getValuePtr()); }
      AlignedValue(const AlignedValue&& v) { new(getValuePtr()) Value(*v.getValuePtr()); };
      AlignedValue& operator = (const AlignedValue& v) { *getValuePtr() = *v.getValuePtr(); return *this; };
      AlignedValue& operator = (const AlignedValue&& v) { *getValuePtr() = *v.getValuePtr(); return *this; };
      operator Value() const { return *getValuePtr(); }
    };
    
    struct Iterator_Index
    {
      Index v;
      typedef std::forward_iterator_tag iterator_category;
      typedef AlignedValue value_type;
      typedef Index difference_type;
      typedef Index distance_type;
      typedef AlignedValue* pointer;
      typedef AlignedValue& reference;
      __forceinline Iterator_Index() {}
      __forceinline Iterator_Index(Index v) : v(v) {}
      __forceinline bool operator== (Iterator_Index other) { return v == other.v; }
      __forceinline bool operator!= (Iterator_Index other) { return v != other.v; }
      __forceinline Iterator_Index operator++() { return Iterator_Index(++v); }
      __forceinline Iterator_Index operator++(int) { return Iterator_Index(v++); }
    };
    
    auto range_reduction = [&](Iterator_Index begin, Iterator_Index end, const AlignedValue& start) {
      assert(begin.v < end.v);
      return reduction(start, func(range<Index>(begin.v, end.v)));
    };
    const Value v = concurrency::parallel_reduce(Iterator_Index(first), Iterator_Index(last), AlignedValue(identity), range_reduction, reduction);
    return v;
#endif
  }

  template<typename Index, typename Value, typename Func, typename Reduction>
    __forceinline Value parallel_reduce( const Index first, const Index last, const Index minStepSize, const Index parallel_threshold, const Value& identity, const Func& func, const Reduction& reduction )
  {
    if (likely(last-first < parallel_threshold)) {
      return func(range<Index>(first,last)); 
    } else {
      return parallel_reduce(first,last,minStepSize,identity,func,reduction);
    }
  }

  template<typename Index, typename Value, typename Func, typename Reduction>
    __forceinline Value parallel_reduce( const range<Index> range, const Index minStepSize, const Index parallel_threshold, const Value& identity, const Func& func, const Reduction& reduction ) 
  {
    return parallel_reduce(range.begin(),range.end(),minStepSize,parallel_threshold,identity,func,reduction);
  }

  template<typename Index, typename Value, typename Func, typename Reduction>
    __forceinline Value parallel_reduce( const Index first, const Index last, const Value& identity, const Func& func, const Reduction& reduction )
  {
    auto funcr = [&] ( const range<Index> r ) {
      Value v = identity;
      for (Index i=r.begin(); i<r.end(); i++)
        v = reduction(v,func(i));
      return v;
    };
    return parallel_reduce(first,last,Index(1),identity,funcr,reduction);
  }

  template<typename Index, typename Value, typename Func, typename Reduction>
    __forceinline Value parallel_reduce( const range<Index> range, const Value& identity, const Func& func, const Reduction& reduction )
  {
    return parallel_reduce(range.begin(),range.end(),Index(1),identity,func,reduction);
  }
}
