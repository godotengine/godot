// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../sys/platform.h"
#include "../math/emath.h"

namespace embree
{
  template<typename Ty>
    struct range 
    {
      __forceinline range() {}

      __forceinline range(const Ty& begin)
        : _begin(begin), _end(begin+1) {}
      
      __forceinline range(const Ty& begin, const Ty& end)
        : _begin(begin), _end(end) {}
 
      __forceinline range(const range& other)
        : _begin(other._begin), _end(other._end) {}

      template<typename T1>
      __forceinline range(const range<T1>& other)
        : _begin(Ty(other._begin)), _end(Ty(other._end)) {}

      template<typename T1>
      __forceinline range& operator =(const range<T1>& other) {
        _begin = other._begin;
        _end = other._end;
        return *this;
      }
      
      __forceinline Ty begin() const {
        return _begin;
      }
      
      __forceinline Ty end() const {
	return _end;
      }

      __forceinline range intersect(const range& r) const {
        return range (max(_begin,r._begin),min(_end,r._end));
      }

      __forceinline Ty size() const {
        return _end - _begin;
      }

      __forceinline bool empty() const { 
        return _end <= _begin; 
      }

      __forceinline Ty center() const {
        return (_begin + _end)/2;
      }

      __forceinline std::pair<range,range> split() const 
      {
        const Ty _center = center();
        return std::make_pair(range(_begin,_center),range(_center,_end));
      }

      __forceinline void split(range& left_o, range& right_o) const 
      {
        const Ty _center = center();
        left_o = range(_begin,_center);
        right_o = range(_center,_end);
      }

      __forceinline friend bool operator< (const range& r0, const range& r1) {
        return r0.size() < r1.size();
      }
	
      friend embree_ostream operator<<(embree_ostream cout, const range& r) {
        return cout << "range [" << r.begin() << ", " << r.end() << "]";
      }
      
      Ty _begin, _end;
    };

  template<typename Ty>
    range<Ty> make_range(const Ty& begin, const Ty& end) {
    return range<Ty>(begin,end);
  }

  template<typename Ty>
    struct extended_range : public range<Ty>
    {
      __forceinline extended_range () {}

      __forceinline extended_range (const Ty& begin)
        : range<Ty>(begin), _ext_end(begin+1) {}
      
      __forceinline extended_range (const Ty& begin, const Ty& end)
        : range<Ty>(begin,end), _ext_end(end) {}

      __forceinline extended_range (const Ty& begin, const Ty& end, const Ty& ext_end)
        : range<Ty>(begin,end), _ext_end(ext_end) {}
      
      __forceinline Ty ext_end() const {
	return _ext_end;
      }

      __forceinline Ty ext_size() const {
        return _ext_end - range<Ty>::_begin;
      }

      __forceinline Ty ext_range_size() const {
        return _ext_end - range<Ty>::_end;
      }

      __forceinline bool has_ext_range() const {
        assert(_ext_end >= range<Ty>::_end);
        return (_ext_end - range<Ty>::_end) > 0;
      }

      __forceinline void set_ext_range(const size_t ext_end){
        assert(ext_end >= range<Ty>::_end);
        _ext_end = ext_end;
      }

      __forceinline void move_right(const size_t plus){
        range<Ty>::_begin   += plus;
        range<Ty>::_end     += plus;
        _ext_end += plus;
      }

      friend embree_ostream operator<<(embree_ostream cout, const extended_range& r) {
        return cout << "extended_range [" << r.begin() << ", " << r.end() <<  " (" << r.ext_end() << ")]";
      }
      
      Ty _ext_end;
    };
}
