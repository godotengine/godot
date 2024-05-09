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


THRUST_NAMESPACE_BEGIN
namespace detail
{
namespace static_map_detail
{


template<unsigned int k, unsigned int v>
struct key_value
{
  static const unsigned int key = k;
  static const unsigned int value = v;
};


template<typename Head, typename Tail = void>
struct cons
{
  template<unsigned int key, unsigned int default_value>
  struct static_get
  {
    static const unsigned int value = (key == Head::key) ? (Head::value) : Tail::template static_get<key,default_value>::value;
  };


  template<unsigned int default_value>
  __host__ __device__
  static unsigned int get(unsigned int key)
  {
    return (key == Head::key) ? (Head::value) : Tail::template get<default_value>(key);
  }
};


template<typename Head>
struct cons<Head,void>
{
  template<unsigned int key, unsigned int default_value>
  struct static_get
  {
    static const unsigned int value = (key == Head::key) ? (Head::value) : default_value;
  };

  template<unsigned int default_value>
  __host__ __device__
  static unsigned int get(unsigned int key)
  {
    return (key == Head::key) ? (Head::value) : default_value;
  }
};


template<unsigned int default_value,
         unsigned int key0 = 0, unsigned int value0 = default_value,
         unsigned int key1 = 0, unsigned int value1 = default_value,
         unsigned int key2 = 0, unsigned int value2 = default_value,
         unsigned int key3 = 0, unsigned int value3 = default_value,
         unsigned int key4 = 0, unsigned int value4 = default_value,
         unsigned int key5 = 0, unsigned int value5 = default_value,
         unsigned int key6 = 0, unsigned int value6 = default_value,
         unsigned int key7 = 0, unsigned int value7 = default_value>
struct static_map
{
  typedef cons<
    key_value<key0,value0>,
    cons<
      key_value<key1,value1>,
      cons<
        key_value<key2,value2>,
        cons<
          key_value<key3,value3>,
          cons<
            key_value<key4,value4>,
            cons<
              key_value<key5,value5>,
              cons<
                key_value<key6,value6>,
                cons<
                  key_value<key7,value7>
                >
              >
            >
          >
        >
      >
    >
  > impl;

  template<unsigned int key>
  struct static_get
  {
    static const unsigned int value = impl::template static_get<key,default_value>::value;
  };

  __host__ __device__
  static unsigned int get(unsigned int key)
  {
    return impl::template get<default_value>(key);
  }
};


} // end namespace static_map_detail


template<unsigned int default_value,
         unsigned int key0 = 0, unsigned int value0 = default_value,
         unsigned int key1 = 0, unsigned int value1 = default_value,
         unsigned int key2 = 0, unsigned int value2 = default_value,
         unsigned int key3 = 0, unsigned int value3 = default_value,
         unsigned int key4 = 0, unsigned int value4 = default_value,
         unsigned int key5 = 0, unsigned int value5 = default_value,
         unsigned int key6 = 0, unsigned int value6 = default_value,
         unsigned int key7 = 0, unsigned int value7 = default_value>
struct static_map
  : static_map_detail::static_map<
      default_value,
      key0, value0,
      key1, value1,
      key2, value2,
      key3, value3,
      key4, value4,
      key5, value5,
      key6, value6,
      key7, value7
    >
{};


template<unsigned int key, typename StaticMap>
struct static_lookup
{
  static const unsigned int value = StaticMap::template static_get<key>::value;
};


template<typename StaticMap>
__host__ __device__
unsigned int lookup(unsigned int key)
{
  return StaticMap::get(key);
}


} // end namespace detail
THRUST_NAMESPACE_END

