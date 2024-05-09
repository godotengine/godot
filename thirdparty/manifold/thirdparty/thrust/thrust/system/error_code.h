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


/*! \file error_code.h
 *  \brief An object used to hold error values, such as those originating from the
 *         operating system or other low-level application program interfaces.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/type_traits.h>
#include <thrust/system/detail/errno.h>
#include <iostream>

THRUST_NAMESPACE_BEGIN

namespace system
{


/*! \addtogroup system_diagnostics
 *  \{
 */

class error_condition;
class error_code;

/*! A metafunction returning whether or not the parameter is an \p error_code enum.
 */
template<typename T> struct is_error_code_enum : public thrust::detail::false_type {};

/*! A metafunction returning whether or not the parameter is an \p error_condition enum.
 */
template<typename T> struct is_error_condition_enum : public thrust::detail::false_type {};


// XXX N3092 prefers enum class errc { ... }
namespace errc
{

/*! An enum containing common error codes.
 */
enum errc_t
{
  address_family_not_supported       = detail::eafnosupport,
  address_in_use                     = detail::eaddrinuse,
  address_not_available              = detail::eaddrnotavail,
  already_connected                  = detail::eisconn,
  argument_list_too_long             = detail::e2big,
  argument_out_of_domain             = detail::edom,
  bad_address                        = detail::efault,
  bad_file_descriptor                = detail::ebadf,
  bad_message                        = detail::ebadmsg,
  broken_pipe                        = detail::epipe,
  connection_aborted                 = detail::econnaborted,
  connection_already_in_progress     = detail::ealready,
  connection_refused                 = detail::econnrefused,
  connection_reset                   = detail::econnreset,
  cross_device_link                  = detail::exdev,
  destination_address_required       = detail::edestaddrreq,
  device_or_resource_busy            = detail::ebusy,
  directory_not_empty                = detail::enotempty,
  executable_format_error            = detail::enoexec,
  file_exists                        = detail::eexist,
  file_too_large                     = detail::efbig,
  filename_too_long                  = detail::enametoolong,
  function_not_supported             = detail::enosys,
  host_unreachable                   = detail::ehostunreach,
  identifier_removed                 = detail::eidrm,
  illegal_byte_sequence              = detail::eilseq,
  inappropriate_io_control_operation = detail::enotty,
  interrupted                        = detail::eintr,
  invalid_argument                   = detail::einval,
  invalid_seek                       = detail::espipe,
  io_error                           = detail::eio,
  is_a_directory                     = detail::eisdir,
  message_size                       = detail::emsgsize,
  network_down                       = detail::enetdown,
  network_reset                      = detail::enetreset,
  network_unreachable                = detail::enetunreach,
  no_buffer_space                    = detail::enobufs,
  no_child_process                   = detail::echild,
  no_link                            = detail::enolink,
  no_lock_available                  = detail::enolck,
  no_message_available               = detail::enodata,
  no_message                         = detail::enomsg,
  no_protocol_option                 = detail::enoprotoopt,
  no_space_on_device                 = detail::enospc,
  no_stream_resources                = detail::enosr,
  no_such_device_or_address          = detail::enxio,
  no_such_device                     = detail::enodev,
  no_such_file_or_directory          = detail::enoent,
  no_such_process                    = detail::esrch,
  not_a_directory                    = detail::enotdir,
  not_a_socket                       = detail::enotsock,
  not_a_stream                       = detail::enostr,
  not_connected                      = detail::enotconn,
  not_enough_memory                  = detail::enomem,
  not_supported                      = detail::enotsup,
  operation_canceled                 = detail::ecanceled,
  operation_in_progress              = detail::einprogress,
  operation_not_permitted            = detail::eperm,
  operation_not_supported            = detail::eopnotsupp,
  operation_would_block              = detail::ewouldblock,
  owner_dead                         = detail::eownerdead,
  permission_denied                  = detail::eacces,
  protocol_error                     = detail::eproto,
  protocol_not_supported             = detail::eprotonosupport,
  read_only_file_system              = detail::erofs,
  resource_deadlock_would_occur      = detail::edeadlk,
  resource_unavailable_try_again     = detail::eagain,
  result_out_of_range                = detail::erange,
  state_not_recoverable              = detail::enotrecoverable,
  stream_timeout                     = detail::etime,
  text_file_busy                     = detail::etxtbsy,
  timed_out                          = detail::etimedout,
  too_many_files_open_in_system      = detail::enfile,
  too_many_files_open                = detail::emfile,
  too_many_links                     = detail::emlink,
  too_many_symbolic_link_levels      = detail::eloop,
  value_too_large                    = detail::eoverflow,
  wrong_protocol_type                = detail::eprototype
}; // end errc_t

} // end namespace errc


/*! Specialization of \p is_error_condition_enum for \p errc::errc_t
 */
template<> struct is_error_condition_enum<errc::errc_t> : public thrust::detail::true_type {};


// [19.5.1.1] class error_category

/*! \brief The class \p error_category serves as a base class for types used to identify the
 *         source and encoding of a particular category of error code. Classes may be derived
 *         from \p error_category to support categories of errors in addition to those defined
 *         in the C++ International Standard.
 */
class error_category
{
  public:
    /*! Destructor does nothing.
     */
    inline virtual ~error_category(void);

    // XXX enable upon c++0x
    // error_category(const error_category &) = delete;
    // error_category &operator=(const error_category &) = delete;

    /*! \return A string naming the error category.
     */
    inline virtual const char *name(void) const = 0;

    /*! \return \p error_condition(ev, *this).
     */
    inline virtual error_condition default_error_condition(int ev) const;

    /*! \return <tt>default_error_condition(code) == condition</tt>
     */
    inline virtual bool equivalent(int code, const error_condition &condition) const;

    /*! \return <tt>*this == code.category() && code.value() == condition</tt>
     */
    inline virtual bool equivalent(const error_code &code, int condition) const;

    /*! \return A string that describes the error condition denoted by \p ev.
     */
    virtual std::string message(int ev) const = 0;

    /*! \return <tt>*this == &rhs</tt>
     */
    inline bool operator==(const error_category &rhs) const;

    /*! \return <tt>!(*this == rhs)</tt>
     */
    inline bool operator!=(const error_category &rhs) const;

    /*! \return <tt>less<const error_category*>()(this, &rhs)</tt>
     *  \note \c less provides a total ordering for pointers.
     */
    inline bool operator<(const error_category &rhs) const;
}; // end error_category


// [19.5.1.5] error_category objects


/*! \return A reference to an object of a type derived from class \p error_category.
 *  \note The object's \p default_error_condition and \p equivalent virtual functions
 *        shall behave as specified for the class \p error_category. The object's
 *        \p name virtual function shall return a pointer to the string <tt>"generic"</tt>.
 */
inline const error_category &generic_category(void);


/*! \return A reference to an object of a type derived from class \p error_category.
 *  \note The object's \p equivalent virtual functions shall behave as specified for
 *        class \p error_category. The object's \p name virtual function shall return
 *        a pointer to the string <tt>"system"</tt>. The object's \p default_error_condition
 *        virtual function shall behave as follows:
 *
 *        If the argument <tt>ev</tt> corresponds to a POSIX <tt>errno</tt> value
 *        \c posv, the function shall return <tt>error_condition(ev,generic_category())</tt>.
 *        Otherwise, the function shall return <tt>error_condition(ev,system_category())</tt>.
 *        What constitutes correspondence for any given operating system is unspecified.
 */
inline const error_category &system_category(void);


// [19.5.2] Class error_code


/*! \brief The class \p error_code describes an object used to hold error code values, such as
 *         those originating from the operating system or other low-level application program
 *         interfaces.
 */
class error_code
{
  public:
    // [19.5.2.2] constructors:

    /*! Effects: Constructs an object of type \p error_code.
     *  \post <tt>value() == 0</tt> and <tt>category() == &system_category()</tt>.
     */
    inline error_code(void);

    /*! Effects: Constructs an object of type \p error_code.
     *  \post <tt>value() == val</tt> and <tt>category() == &cat</tt>.
     */
    inline error_code(int val, const error_category &cat);

    /*! Effects: Constructs an object of type \p error_code.
     *  \post <tt>*this == make_error_code(e)</tt>.
     */
    template <typename ErrorCodeEnum>
      error_code(ErrorCodeEnum e
// XXX WAR msvc's problem with enable_if
#if THRUST_HOST_COMPILER != THRUST_HOST_COMPILER_MSVC
        , typename thrust::detail::enable_if<is_error_code_enum<ErrorCodeEnum>::value>::type * = 0
#endif // THRUST_HOST_COMPILER_MSVC
        );

    // [19.5.2.3] modifiers:

    /*! \post <tt>value() == val</tt> and <tt>category() == &cat</tt>.
     */
    inline void assign(int val, const error_category &cat);

    /*! \post <tt>*this == make_error_code(e)</tt>.
     */
    template <typename ErrorCodeEnum>
// XXX WAR msvc's problem with enable_if
#if THRUST_HOST_COMPILER != THRUST_HOST_COMPILER_MSVC
      typename thrust::detail::enable_if<is_error_code_enum<ErrorCodeEnum>::value, error_code>::type &
#else
      error_code &
#endif // THRUST_HOST_COMPILER_MSVC
        operator=(ErrorCodeEnum e);

    /*! \post <tt>value() == 0</tt> and <tt>category() == system_category()</tt>.
     */
    inline void clear(void);

    // [19.5.2.4] observers:

    /*! \return An integral value of this \p error_code object.
     */
    inline int value(void) const;

    /*! \return An \p error_category describing the category of this \p error_code object.
     */
    inline const error_category &category(void) const;

    /*! \return <tt>category().default_error_condition()</tt>.
     */
    inline error_condition default_error_condition(void) const;

    /*! \return <tt>category().message(value())</tt>.
     */
    inline std::string message(void) const;

    // XXX replace the below upon c++0x
    // inline explicit operator bool (void) const;

    /*! \return <tt>value() != 0</tt>.
     */
    inline operator bool (void) const;

    /*! \cond
     */
  private:
    int m_val;
    const error_category *m_cat;
    /*! \endcond
     */
}; // end error_code


// [19.5.2.5] Class error_code non-member functions


// XXX replace errc::errc_t with errc upon c++0x
/*! \return <tt>error_code(static_cast<int>(e), generic_category())</tt>
 */
inline error_code make_error_code(errc::errc_t e);


/*! \return <tt>lhs.category() < rhs.category() || lhs.category() == rhs.category() && lhs.value() < rhs.value()</tt>.
 */
inline bool operator<(const error_code &lhs, const error_code &rhs);


/*! Effects: <tt>os << ec.category().name() << ':' << ec.value()</tt>.
 */
template <typename charT, typename traits>
  std::basic_ostream<charT,traits>&
    operator<<(std::basic_ostream<charT,traits>& os, const error_code &ec);


// [19.5.3] class error_condition


/*! \brief The class \p error_condition describes an object used to hold values identifying
 *  error conditions.
 *
 *  \note \p error_condition values are portable abstractions, while \p error_code values
 *        are implementation specific.
 */
class error_condition
{
  public:
    // [19.5.3.2] constructors

    /*! Constructs an object of type \p error_condition.
     *  \post <tt>value() == 0</tt>.
     *  \post <tt>category() == generic_category()</tt>.
     */
    inline error_condition(void);

    /*! Constructs an object of type \p error_condition.
     *  \post <tt>value() == val</tt>.
     *  \post <tt>category() == cat</tt>.
     */
    inline error_condition(int val, const error_category &cat);

    /*! Constructs an object of type \p error_condition.
     *  \post <tt>*this == make_error_condition(e)</tt>.
     *  \note This constructor shall not participate in overload resolution unless
     *        <tt>is_error_condition_enum<ErrorConditionEnum>::value</tt> is <tt>true</tt>.
     */
    template<typename ErrorConditionEnum>
      error_condition(ErrorConditionEnum e
// XXX WAR msvc's problem with enable_if
#if THRUST_HOST_COMPILER != THRUST_HOST_COMPILER_MSVC
        , typename thrust::detail::enable_if<is_error_condition_enum<ErrorConditionEnum>::value>::type * = 0
#endif // THRUST_HOST_COMPILER != THRUST_HOST_COMPILER_MSVC
                     );

    // [19.5.3.3] modifiers

    /*! Assigns to this \p error_code object from an error value and an \p error_category.
     *  \param val The new value to return from <tt>value()</tt>.
     *  \param cat The new \p error_category to return from <tt>category()</tt>.
     *  \post <tt>value() == val</tt>.
     *  \post <tt>category() == cat</tt>.
     */
    inline void assign(int val, const error_category &cat);

    /*! Assigns to this \p error_code object from an error condition enumeration.
     *  \return *this
     *  \post <tt>*this == make_error_condition(e)</tt>.
     *  \note This operator shall not participate in overload resolution unless
     *        <tt>is_error_condition_enum<ErrorConditionEnum>::value</tt> is <tt>true</tt>.
     */
    template<typename ErrorConditionEnum>
// XXX WAR msvc's problem with enable_if
#if THRUST_HOST_COMPILER != THRUST_HOST_COMPILER_MSVC
      typename thrust::detail::enable_if<is_error_condition_enum<ErrorConditionEnum>::value, error_condition>::type &
#else
      error_condition &
#endif // THRUST_HOST_COMPILER != THRUST_HOST_COMPILER_MSVC
        operator=(ErrorConditionEnum e);

    /*! Clears this \p error_code object.
     *  \post <tt>value == 0</tt>
     *  \post <tt>category() == generic_category()</tt>.
     */
    inline void clear(void);

    // [19.5.3.4] observers

    /*! \return The value encoded by this \p error_condition.
     */
    inline int value(void) const;

    /*! \return A <tt>const</tt> reference to the \p error_category encoded by this \p error_condition.
     */
    inline const error_category &category(void) const;

    /*! \return <tt>category().message(value())</tt>.
     */
    inline std::string message(void) const;

    // XXX replace below with this upon c++0x
    //explicit operator bool (void) const;
    
    /*! \return <tt>value() != 0</tt>.
     */
    inline operator bool (void) const;

    /*! \cond
     */

  private:
    int m_val;
    const error_category *m_cat;

    /*! \endcond
     */
}; // end error_condition



// [19.5.3.5] Class error_condition non-member functions

// XXX replace errc::errc_t with errc upon c++0x
/*! \return <tt>error_condition(static_cast<int>(e), generic_category())</tt>.
 */
inline error_condition make_error_condition(errc::errc_t e);


/*! \return <tt>lhs.category() < rhs.category() || lhs.category() == rhs.category() && lhs.value() < rhs.value()</tt>.
 */
inline bool operator<(const error_condition &lhs, const error_condition &rhs);


// [19.5.4] Comparison operators


/*! \return <tt>lhs.category() == rhs.category() && lhs.value() == rhs.value()</tt>.
 */
inline bool operator==(const error_code &lhs, const error_code &rhs);


/*! \return <tt>lhs.category().equivalent(lhs.value(), rhs) || rhs.category().equivalent(lhs,rhs.value())</tt>.
 */
inline bool operator==(const error_code &lhs, const error_condition &rhs);


/*! \return <tt>rhs.category().equivalent(lhs.value(), lhs) || lhs.category().equivalent(rhs, lhs.value())</tt>.
 */
inline bool operator==(const error_condition &lhs, const error_code &rhs);


/*! \return <tt>lhs.category() == rhs.category() && lhs.value() == rhs.value()</tt>
 */
inline bool operator==(const error_condition &lhs, const error_condition &rhs);


/*! \return <tt>!(lhs == rhs)</tt>
 */
inline bool operator!=(const error_code &lhs, const error_code &rhs);


/*! \return <tt>!(lhs == rhs)</tt>
 */
inline bool operator!=(const error_code &lhs, const error_condition &rhs);


/*! \return <tt>!(lhs == rhs)</tt>
 */
inline bool operator!=(const error_condition &lhs, const error_code &rhs);


/*! \return <tt>!(lhs == rhs)</tt>
 */
inline bool operator!=(const error_condition &lhs, const error_condition &rhs);

/*! \} // end system_diagnostics
 */


} // end system


// import names into thrust::
using system::error_category;
using system::error_code;
using system::error_condition;
using system::is_error_code_enum;
using system::is_error_condition_enum;
using system::make_error_code;
using system::make_error_condition;

// XXX replace with using system::errc upon c++0x
namespace errc = system::errc;

using system::generic_category;
using system::system_category;

THRUST_NAMESPACE_END

#include <thrust/system/detail/error_category.inl>
#include <thrust/system/detail/error_code.inl>
#include <thrust/system/detail/error_condition.inl>

