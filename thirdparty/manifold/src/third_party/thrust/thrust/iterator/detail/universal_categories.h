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
#include <thrust/iterator/iterator_categories.h>

// XXX eliminate this file

THRUST_NAMESPACE_BEGIN

// define these types without inheritance to avoid ambiguous conversion to base classes

struct input_universal_iterator_tag
{
  operator input_host_iterator_tag () {return input_host_iterator_tag();}

  operator input_device_iterator_tag () {return input_device_iterator_tag();}
};

struct output_universal_iterator_tag
{
  operator output_host_iterator_tag () {return output_host_iterator_tag();}

  operator output_device_iterator_tag () {return output_device_iterator_tag();}
};

struct forward_universal_iterator_tag
  : input_universal_iterator_tag
{
  operator forward_host_iterator_tag () {return forward_host_iterator_tag();};

  operator forward_device_iterator_tag () {return forward_device_iterator_tag();};
};

struct bidirectional_universal_iterator_tag
  : forward_universal_iterator_tag
{
  operator bidirectional_host_iterator_tag () {return bidirectional_host_iterator_tag();};

  operator bidirectional_device_iterator_tag () {return bidirectional_device_iterator_tag();};
};


namespace detail
{

// create this struct to control conversion precedence in random_access_universal_iterator_tag
template<typename T>
struct one_degree_of_separation
  : T
{
};

} // end detail


struct random_access_universal_iterator_tag
{
  // these conversions are all P0
  operator random_access_host_iterator_tag () {return random_access_host_iterator_tag();};

  operator random_access_device_iterator_tag () {return random_access_device_iterator_tag();};

  // bidirectional_universal_iterator_tag is P1
  operator detail::one_degree_of_separation<bidirectional_universal_iterator_tag> () {return detail::one_degree_of_separation<bidirectional_universal_iterator_tag>();}

};


THRUST_NAMESPACE_END

