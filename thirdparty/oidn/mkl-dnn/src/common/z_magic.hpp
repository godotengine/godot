/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef Z_MAGIC_HPP
#define Z_MAGIC_HPP

#define CHAIn2(a,b) a b
#define CHAIN2(a,b) CHAIn2(a,b)

#define CONCAt2(a,b) a ## b
#define CONCAT2(a,b) CONCAt2(a,b)

#define STRINGIFy(s) #s
#define STRINGIFY(s) STRINGIFy(s)

#ifdef _MSC_VER
#   define PRAGMA_MACRo(x) __pragma(x)
#   define PRAGMA_MACRO(x) PRAGMA_MACRo(x)
#else
#   define PRAGMA_MACRo(x) _Pragma(#x)
#   define PRAGMA_MACRO(x) PRAGMA_MACRo(x)
#endif

#define UNUSED(x) ((void)x)
#define MAYBE_UNUSED(x) UNUSED(x)

#if defined(_WIN32) && !defined(__GNUC__)
#define __PRETTY_FUNCTION__ __FUNCSIG__
#endif

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
