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

THRUST_NAMESPACE_BEGIN

namespace system
{

error_code
  ::error_code(void)
    :m_val(0),m_cat(&system_category())
{
  ;
} // end error_code::error_code()


error_code
  ::error_code(int val, const error_category &cat)
    :m_val(val),m_cat(&cat)
{
  ;
} // end error_code::error_code()


template <typename ErrorCodeEnum>
  error_code
    ::error_code(ErrorCodeEnum e
// XXX WAR msvc's problem with enable_if
#if THRUST_HOST_COMPILER != THRUST_HOST_COMPILER_MSVC
                 , typename thrust::detail::enable_if<is_error_code_enum<ErrorCodeEnum>::value>::type *
#endif // THRUST_HOST_COMPILER_MSVC
                )
{
  *this = make_error_code(e);
} // end error_code::error_code()


void error_code
  ::assign(int val, const error_category &cat)
{
  m_val = val;
  m_cat = &cat;
} // end error_code::assign()


template <typename ErrorCodeEnum>
// XXX WAR msvc's problem with enable_if
#if THRUST_HOST_COMPILER != THRUST_HOST_COMPILER_MSVC
  typename thrust::detail::enable_if<is_error_code_enum<ErrorCodeEnum>::value, error_code>::type &
#else
  error_code &
#endif // THRUST_HOST_COMPILER_MSVC
    error_code
      ::operator=(ErrorCodeEnum e)
{
  *this = make_error_code(e);
  return *this;
} // end error_code::operator=()


void error_code
  ::clear(void)
{
  m_val = 0;
  m_cat = &system_category();
} // end error_code::clear()


int error_code
  ::value(void) const
{
  return m_val;
} // end error_code::value()


const error_category &error_code
  ::category(void) const
{
  return *m_cat;
} // end error_code::category()


error_condition error_code
  ::default_error_condition(void) const
{
  return category().default_error_condition(value());
} // end error_code::default_error_condition()


std::string error_code
  ::message(void) const
{
  return category().message(value());
} // end error_code::message()


error_code
  ::operator bool (void) const
{
  return value() != 0;
} // end error_code::operator bool ()


error_code make_error_code(errc::errc_t e)
{
  return error_code(static_cast<int>(e), generic_category());
} // end make_error_code()


bool operator<(const error_code &lhs, const error_code &rhs)
{
  bool result = lhs.category().operator<(rhs.category());
  result = result || lhs.category().operator==(rhs.category());
  result = result || lhs.value() < rhs.value();
  return result;
} // end operator==()


template<typename charT, typename traits>
  std::basic_ostream<charT,traits>&
    operator<<(std::basic_ostream<charT,traits> &os, const error_code &ec)
{
  return os << ec.category().name() << ':' << ec.value();
} // end operator<<()


bool operator==(const error_code &lhs, const error_code &rhs)
{
  return lhs.category().operator==(rhs.category()) && lhs.value() == rhs.value();
} // end operator==()


bool operator==(const error_code &lhs, const error_condition &rhs)
{
  return lhs.category().equivalent(lhs.value(), rhs) || rhs.category().equivalent(lhs,rhs.value());
} // end operator==()


bool operator==(const error_condition &lhs, const error_code &rhs)
{
  return rhs.category().equivalent(lhs.value(), lhs) || lhs.category().equivalent(rhs, lhs.value());
} // end operator==()


bool operator==(const error_condition &lhs, const error_condition &rhs)
{
  return lhs.category().operator==(rhs.category()) && lhs.value() == rhs.value();
} // end operator==()


bool operator!=(const error_code &lhs, const error_code &rhs)
{
  return !(lhs == rhs);
} // end operator!=()


bool operator!=(const error_code &lhs, const error_condition &rhs)
{
  return !(lhs == rhs);
} // end operator!=()


bool operator!=(const error_condition &lhs, const error_code &rhs)
{
  return !(lhs == rhs);
} // end operator!=()


bool operator!=(const error_condition &lhs, const error_condition &rhs)
{
  return !(lhs == rhs);
} // end operator!=()


} // end system

THRUST_NAMESPACE_END

