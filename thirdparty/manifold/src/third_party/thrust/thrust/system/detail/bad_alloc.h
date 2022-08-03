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

#include <new>
#include <string>

#include <thrust/detail/config.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{

// define our own bad_alloc so we can set its .what()
class bad_alloc
  : public std::bad_alloc
{
  public:
    inline bad_alloc(const std::string &w)
      : std::bad_alloc(), m_what()
    {
      m_what = std::bad_alloc::what();
      m_what += ": ";
      m_what += w;
    } // end bad_alloc()

    inline virtual ~bad_alloc(void) throw () {};

    inline virtual const char *what(void) const throw()
    {
      return m_what.c_str();
    } // end what()

  private:
    std::string m_what;
}; // end bad_alloc
  
} // end detail
} // end system
THRUST_NAMESPACE_END

