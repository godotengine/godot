/*
 *  Copyright 2008-2018 NVIDIA Corporation
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

/// \file thrust/detail/event_error.h
/// \brief \c thrust::future and thrust::future error handling types and codes.

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/cpp14_required.h>

#if THRUST_CPP_DIALECT >= 2014

#include <thrust/detail/type_traits.h>
#include <thrust/system/error_code.h>

#include <stdexcept>

THRUST_NAMESPACE_BEGIN

enum class event_errc
{
  unknown_event_error
, no_state
, no_content
, last_event_error
};

/// \return <tt>error_code(static_cast<int>(e), event_category())</tt>
inline error_code make_error_code(event_errc e);

/// \return <tt>error_condition(static_cast<int>(e), event_category())</tt>.
inline error_condition make_error_condition(event_errc e);

struct event_error_category : error_category
{
  event_error_category() = default;

  virtual char const* name() const
  {
    return "event";
  }

  virtual std::string message(int ev) const
  {
    switch (static_cast<event_errc>(ev))
    {
      case event_errc::no_state:
      {
        return "no_state: an operation that requires an event or future to have "
               "a stream or content has been performed on a event or future "
               "without either, e.g. a moved-from or default constructed event "
               "or future (an event or future may have been consumed more than "
               "once)";
      }
      case event_errc::no_content:
      {
        return "no_content: an operation that requires a future to have content "
               "has been performed on future without any, e.g. a moved-from, "
               "default constructed, or `thrust::new_stream` constructed future "
               "(a future may have been consumed more than once)";
      }
      default:
      {
        return "unknown_event_error: an unknown error with a future "
               "object has occurred";
      }
    };
  }

  virtual error_condition default_error_condition(int ev) const
  {
    if (
         event_errc::last_event_error
         >
         static_cast<event_errc>(ev)
       )
      return make_error_condition(static_cast<event_errc>(ev));

    return system_category().default_error_condition(ev);
  }
};

/// Obtains a reference to the static error category object for the errors
/// related to futures and promises. The object is required to override the
/// virtual function error_category::name() to return a pointer to the string
/// "event". It is used to identify error codes provided in the
/// exceptions of type event_error.
inline error_category const& event_category()
{
  static const event_error_category result;
  return result;
}

namespace system
{
/// Specialization of \p is_error_code_enum for \p event_errc.
template<> struct is_error_code_enum<event_errc> : true_type {};
} // end system

/// \return <tt>error_code(static_cast<int>(e), event_category())</tt>
inline error_code make_error_code(event_errc e)
{
  return error_code(static_cast<int>(e), event_category());
}

/// \return <tt>error_condition(static_cast<int>(e), event_category())</tt>.
inline error_condition make_error_condition(event_errc e)
{
  return error_condition(static_cast<int>(e), event_category());
}

struct event_error : std::logic_error
{
  __host__
  explicit event_error(error_code ec)
    : std::logic_error(ec.message()), ec_(ec)
  {}

  __host__
  explicit event_error(event_errc e)
    : event_error(make_error_code(e))
  {}

  __host__
  error_code const& code() const noexcept
  {
    return ec_;
  }

  __host__
  virtual ~event_error() noexcept {}

private:
  error_code ec_;
};

inline bool operator==(event_error const& lhs, event_error const& rhs) noexcept
{
  return lhs.code() == rhs.code();
}

inline bool operator<(event_error const& lhs, event_error const& rhs) noexcept
{
  return lhs.code() < rhs.code();
}

THRUST_NAMESPACE_END

#endif // C++14

