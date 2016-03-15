/* -----------------------------------------------------------------------------
 For without stl dependent under android platfrom
----------------------------------------------------------------------------- */
#ifndef __ALGORITHM_H__
#define __ALGORITHM_H__

#ifdef ANDROID

namespace std {

template<typename T>
const T& min(const T& a, const T&b) { return (a<b) ? a : b; }

template<typename T>
const T& max(const T& a, const T&b) { return (a>b) ? a : b; }

template<typename T>
void swap(T& a, T&b) { T tmp = a; a = b; b = a; }

};
#else

#include <algorithm>

#endif

#endif // __ALGORITHM_H__
