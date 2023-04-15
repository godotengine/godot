/*
    Copyright (c) 2005-2020 Intel Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#ifndef __TBB_range_iterator_H
#define __TBB_range_iterator_H

#include "../tbb_stddef.h"

#if __TBB_CPP11_STD_BEGIN_END_PRESENT && __TBB_CPP11_AUTO_PRESENT && __TBB_CPP11_DECLTYPE_PRESENT
    #include <iterator>
#endif

namespace tbb {
    // iterators to first and last elements of container
    namespace internal {

#if __TBB_CPP11_STD_BEGIN_END_PRESENT && __TBB_CPP11_AUTO_PRESENT && __TBB_CPP11_DECLTYPE_PRESENT
        using std::begin;
        using std::end;
        template<typename Container>
        auto first(Container& c)-> decltype(begin(c))  {return begin(c);}

        template<typename Container>
        auto first(const Container& c)-> decltype(begin(c))  {return begin(c);}

        template<typename Container>
        auto last(Container& c)-> decltype(begin(c))  {return end(c);}

        template<typename Container>
        auto last(const Container& c)-> decltype(begin(c)) {return end(c);}
#else
        template<typename Container>
        typename Container::iterator first(Container& c) {return c.begin();}

        template<typename Container>
        typename Container::const_iterator first(const Container& c) {return c.begin();}

        template<typename Container>
        typename Container::iterator last(Container& c) {return c.end();}

        template<typename Container>
        typename Container::const_iterator last(const Container& c) {return c.end();}
#endif

        template<typename T, size_t size>
        T* first(T (&arr) [size]) {return arr;}

        template<typename T, size_t size>
        T* last(T (&arr) [size]) {return arr + size;}
    } //namespace internal
}  //namespace tbb

#endif // __TBB_range_iterator_H
