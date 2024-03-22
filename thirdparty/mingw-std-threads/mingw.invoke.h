/// \file mingw.invoke.h
/// \brief Lightweight `invoke` implementation, for C++11 and C++14.
///
/// (c) 2018-2019 by Nathaniel J. McClatchey, San Jose, CA, United States
/// \author Nathaniel J. McClatchey, PhD
///
/// \copyright Simplified (2-clause) BSD License.
///
/// \note This file may become part of the mingw-w64 runtime package. If/when
/// this happens, the appropriate license will be added, i.e. this code will
/// become dual-licensed, and the current BSD 2-clause license will stay.

#ifndef MINGW_INVOKE_H_
#define MINGW_INVOKE_H_

#include <type_traits>  //  For std::result_of, etc.
#include <utility>      //  For std::forward
#include <functional>   //  For std::reference_wrapper

namespace mingw_stdthread
{
namespace detail
{
//  For compatibility, implement std::invoke for C++11 and C++14
#if __cplusplus < 201703L
  template<bool PMemFunc, bool PMemData>
  struct Invoker
  {
    template<class F, class... Args>
    inline static typename std::result_of<F(Args...)>::type invoke (F&& f, Args&&... args)
    {
      return std::forward<F>(f)(std::forward<Args>(args)...);
    }
  };
  template<bool>
  struct InvokerHelper;

  template<>
  struct InvokerHelper<false>
  {
    template<class T1>
    inline static auto get (T1&& t1) -> decltype(*std::forward<T1>(t1))
    {
      return *std::forward<T1>(t1);
    }

    template<class T1>
    inline static auto get (const std::reference_wrapper<T1>& t1) -> decltype(t1.get())
    {
      return t1.get();
    }
  };

  template<>
  struct InvokerHelper<true>
  {
    template<class T1>
    inline static auto get (T1&& t1) -> decltype(std::forward<T1>(t1))
    {
      return std::forward<T1>(t1);
    }
  };

  template<>
  struct Invoker<true, false>
  {
    template<class T, class F, class T1, class... Args>
    inline static auto invoke (F T::* f, T1&& t1, Args&&... args) ->\
      decltype((InvokerHelper<std::is_base_of<T,typename std::decay<T1>::type>::value>::get(std::forward<T1>(t1)).*f)(std::forward<Args>(args)...))
    {
      return (InvokerHelper<std::is_base_of<T,typename std::decay<T1>::type>::value>::get(std::forward<T1>(t1)).*f)(std::forward<Args>(args)...);
    }
  };

  template<>
  struct Invoker<false, true>
  {
    template<class T, class F, class T1, class... Args>
    inline static auto invoke (F T::* f, T1&& t1, Args&&... args) ->\
      decltype(InvokerHelper<std::is_base_of<T,typename std::decay<T1>::type>::value>::get(t1).*f)
    {
      return InvokerHelper<std::is_base_of<T,typename std::decay<T1>::type>::value>::get(t1).*f;
    }
  };

  template<class F, class... Args>
  struct InvokeResult
  {
    typedef Invoker<std::is_member_function_pointer<typename std::remove_reference<F>::type>::value,
                    std::is_member_object_pointer<typename std::remove_reference<F>::type>::value &&
                    (sizeof...(Args) == 1)> invoker;
    inline static auto invoke (F&& f, Args&&... args) -> decltype(invoker::invoke(std::forward<F>(f), std::forward<Args>(args)...))
    {
      return invoker::invoke(std::forward<F>(f), std::forward<Args>(args)...);
    }
  };

  template<class F, class...Args>
  auto invoke (F&& f, Args&&... args) -> decltype(InvokeResult<F, Args...>::invoke(std::forward<F>(f), std::forward<Args>(args)...))
  {
    return InvokeResult<F, Args...>::invoke(std::forward<F>(f), std::forward<Args>(args)...);
  }
#else
    using std::invoke;
#endif
} //  Namespace "detail"
} //  Namespace "mingw_stdthread"

#endif
