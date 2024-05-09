/*
 *  Copyright 2008-2013 NVIDIA Corporation
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


#pragma once

#include <thrust/detail/config.h>

#include <thrust/system/error_code.h>
#include <thrust/system/detail/errno.h>
#include <thrust/functional.h>
#include <cstring>

THRUST_NAMESPACE_BEGIN

namespace system
{

error_category
  ::~error_category(void)
{
  ;
} // end error_category::~error_category()


error_condition error_category
  ::default_error_condition(int ev) const
{
  return error_condition(ev, *this);
} // end error_category::default_error_condition()


bool error_category
  ::equivalent(int code, const error_condition &condition) const
{
  return default_error_condition(code) == condition;
} // end error_condition::equivalent()


bool error_category
  ::equivalent(const error_code &code, int condition) const
{
  bool result = (this->operator==(code.category())) && (code.value() == condition);
  return result;
} // end error_code::equivalent()


bool error_category
  ::operator==(const error_category &rhs) const
{
  return this == &rhs;
} // end error_category::operator==()


bool error_category
  ::operator!=(const error_category &rhs) const
{
  return !this->operator==(rhs);
} // end error_category::operator!=()


bool error_category
  ::operator<(const error_category &rhs) const
{
  return thrust::less<const error_category*>()(this,&rhs);
} // end error_category::operator<()


namespace detail
{


class generic_error_category
  : public error_category
{
  public:
    inline generic_error_category(void) {}

    inline virtual const char *name(void) const
    {
      return "generic";
    }

    inline virtual std::string message(int ev) const
    {
      static const std::string unknown_err("Unknown error");

      // XXX strerror is not thread-safe:
      //     prefer strerror_r (which is not provided on windows)
      THRUST_DISABLE_MSVC_WARNING_BEGIN(4996)
      const char *c_str = std::strerror(ev);
      THRUST_DISABLE_MSVC_WARNING_END(4996)
      return c_str ? std::string(c_str) : unknown_err;
    }
}; // end generic_category_result


class system_error_category
  : public error_category
{
  public:
    inline system_error_category(void) {}

    inline virtual const char *name(void) const
    {
      return "system";
    }

    inline virtual std::string message(int ev) const
    {
      return generic_category().message(ev);
    }

    inline virtual error_condition default_error_condition(int ev) const
    {
      using namespace errc;

      switch(ev)
      {
        case eafnosupport:    return make_error_condition(address_family_not_supported);
        case eaddrinuse:      return make_error_condition(address_in_use);
        case eaddrnotavail:   return make_error_condition(address_not_available);
        case eisconn:         return make_error_condition(already_connected);
        case e2big:           return make_error_condition(argument_list_too_long);
        case edom:            return make_error_condition(argument_out_of_domain);
        case efault:          return make_error_condition(bad_address);
        case ebadf:           return make_error_condition(bad_file_descriptor);
        case ebadmsg:         return make_error_condition(bad_message);
        case epipe:           return make_error_condition(broken_pipe);
        case econnaborted:    return make_error_condition(connection_aborted);
        case ealready:        return make_error_condition(connection_already_in_progress);
        case econnrefused:    return make_error_condition(connection_refused);
        case econnreset:      return make_error_condition(connection_reset);
        case exdev:           return make_error_condition(cross_device_link);
        case edestaddrreq:    return make_error_condition(destination_address_required);
        case ebusy:           return make_error_condition(device_or_resource_busy);
        case enotempty:       return make_error_condition(directory_not_empty);
        case enoexec:         return make_error_condition(executable_format_error);
        case eexist:          return make_error_condition(file_exists);
        case efbig:           return make_error_condition(file_too_large);
        case enametoolong:    return make_error_condition(filename_too_long);
        case enosys:          return make_error_condition(function_not_supported);
        case ehostunreach:    return make_error_condition(host_unreachable);
        case eidrm:           return make_error_condition(identifier_removed);
        case eilseq:          return make_error_condition(illegal_byte_sequence);
        case enotty:          return make_error_condition(inappropriate_io_control_operation);
        case eintr:           return make_error_condition(interrupted);
        case einval:          return make_error_condition(invalid_argument);
        case espipe:          return make_error_condition(invalid_seek);
        case eio:             return make_error_condition(io_error);
        case eisdir:          return make_error_condition(is_a_directory);
        case emsgsize:        return make_error_condition(message_size);
        case enetdown:        return make_error_condition(network_down);
        case enetreset:       return make_error_condition(network_reset);
        case enetunreach:     return make_error_condition(network_unreachable);
        case enobufs:         return make_error_condition(no_buffer_space);
        case echild:          return make_error_condition(no_child_process);
        case enolink:         return make_error_condition(no_link);
        case enolck:          return make_error_condition(no_lock_available);
        case enodata:         return make_error_condition(no_message_available);
        case enomsg:          return make_error_condition(no_message);
        case enoprotoopt:     return make_error_condition(no_protocol_option);
        case enospc:          return make_error_condition(no_space_on_device);
        case enosr:           return make_error_condition(no_stream_resources);
        case enxio:           return make_error_condition(no_such_device_or_address);
        case enodev:          return make_error_condition(no_such_device);
        case enoent:          return make_error_condition(no_such_file_or_directory);
        case esrch:           return make_error_condition(no_such_process);
        case enotdir:         return make_error_condition(not_a_directory);
        case enotsock:        return make_error_condition(not_a_socket);
        case enostr:          return make_error_condition(not_a_stream);
        case enotconn:        return make_error_condition(not_connected);
        case enomem:          return make_error_condition(not_enough_memory);
        case enotsup:         return make_error_condition(not_supported);
        case ecanceled:       return make_error_condition(operation_canceled);
        case einprogress:     return make_error_condition(operation_in_progress);
        case eperm:           return make_error_condition(operation_not_permitted);
        case eopnotsupp:      return make_error_condition(operation_not_supported);
        case ewouldblock:     return make_error_condition(operation_would_block);
        case eownerdead:      return make_error_condition(owner_dead);
        case eacces:          return make_error_condition(permission_denied);
        case eproto:          return make_error_condition(protocol_error);
        case eprotonosupport: return make_error_condition(protocol_not_supported);
        case erofs:           return make_error_condition(read_only_file_system);
        case edeadlk:         return make_error_condition(resource_deadlock_would_occur);
        case eagain:          return make_error_condition(resource_unavailable_try_again);
        case erange:          return make_error_condition(result_out_of_range);
        case enotrecoverable: return make_error_condition(state_not_recoverable);
        case etime:           return make_error_condition(stream_timeout);
        case etxtbsy:         return make_error_condition(text_file_busy);
        case etimedout:       return make_error_condition(timed_out);
        case enfile:          return make_error_condition(too_many_files_open_in_system);
        case emfile:          return make_error_condition(too_many_files_open);
        case emlink:          return make_error_condition(too_many_links);
        case eloop:           return make_error_condition(too_many_symbolic_link_levels);
        case eoverflow:       return make_error_condition(value_too_large);
        case eprototype:      return make_error_condition(wrong_protocol_type);
        default:              return error_condition(ev,system_category());
      }
    }
}; // end system_category_result


} // end detail


const error_category &generic_category(void)
{
  static const detail::generic_error_category result;
  return result;
}


const error_category &system_category(void)
{
  static const detail::system_error_category result;
  return result;
}


} // end system

THRUST_NAMESPACE_END

