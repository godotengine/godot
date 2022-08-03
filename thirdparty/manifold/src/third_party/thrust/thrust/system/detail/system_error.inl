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

#include <thrust/system/system_error.h>

THRUST_NAMESPACE_BEGIN

namespace system
{


system_error
  ::system_error(error_code ec, const std::string &what_arg)
    : std::runtime_error(what_arg), m_error_code(ec)
{

} // end system_error::system_error()


system_error
  ::system_error(error_code ec, const char *what_arg)
    : std::runtime_error(what_arg), m_error_code(ec)
{
  ;
} // end system_error::system_error()


system_error
  ::system_error(error_code ec)
    : std::runtime_error(""), m_error_code(ec)
{
  ;
} // end system_error::system_error()


system_error
  ::system_error(int ev, const error_category &ecat, const std::string &what_arg)
    : std::runtime_error(what_arg), m_error_code(ev,ecat)
{
  ;
} // end system_error::system_error()


system_error
  ::system_error(int ev, const error_category &ecat, const char *what_arg)
    : std::runtime_error(what_arg), m_error_code(ev,ecat)
{
  ;
} // end system_error::system_error()


system_error
  ::system_error(int ev, const error_category &ecat)
    : std::runtime_error(""), m_error_code(ev,ecat)
{
  ;
} // end system_error::system_error()


const error_code &system_error
  ::code(void) const throw()
{
  return m_error_code;
} // end system_error::code()


const char *system_error
  ::what(void) const throw()
{
  if(m_what.empty())
  {
    try
    {
      m_what = this->std::runtime_error::what();
      if(m_error_code)
      {
        if(!m_what.empty()) m_what += ": ";
        m_what += m_error_code.message();
      }
    }
    catch(...)
    {
      return std::runtime_error::what();
    }
  }

  return m_what.c_str();
} // end system_error::what()


} // end system

THRUST_NAMESPACE_END

