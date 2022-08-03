/*
 *  Copyright 2008-2021 NVIDIA Corporation
 *  Copyright 2013 Filipe RNC Maia
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

#include <thrust/complex.h>

THRUST_NAMESPACE_BEGIN
template<typename ValueType,class charT, class traits>
std::basic_ostream<charT, traits>& operator<<(std::basic_ostream<charT, traits>& os, const complex<ValueType>& z)
{
  os << '(' << z.real() << ',' << z.imag() << ')';
  return os;
}

template<typename ValueType, typename charT, class traits>
std::basic_istream<charT, traits>&
operator>>(std::basic_istream<charT, traits>& is, complex<ValueType>& z)
{
  ValueType re, im;

  charT ch;
  is >> ch;

  if(ch == '(')
    {
      is >> re >> ch;
      if (ch == ',')
        {
          is >> im >> ch;
          if (ch == ')')
	    {
	      z = complex<ValueType>(re, im);
	    }
          else
	    {
	      is.setstate(std::ios_base::failbit);
	    }
        }
      else if (ch == ')')
        {
          z = re;
        }
      else
        {
          is.setstate(std::ios_base::failbit);
        }
    }
  else
    {
      is.putback(ch);
      is >> re;
      z = re;
    }
  return is;
}

THRUST_NAMESPACE_END
