//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define iswalpha sun_iswalpha
#define iswupper sun_iswupper
#define iswlower sun_iswlower
#define iswdigit sun_iswdigit
#define iswxdigit sun_iswxdigit
#define iswalnum sun_iswalnum
#define iswspace sun_iswspace
#define iswpunct sun_iswpunct
#define iswprint sun_iswprint
#define iswgraph sun_iswgraph
#define iswcntrl sun_iswcntrl
#define iswctype sun_iswctype
#define towlower sun_towlower
#define towupper sun_towupper
#define wcswcs sun_wcswcs
#define wcswidth sun_wcswidth
#define wcwidth sun_wcwidth
#define wctype sun_wctype
#define _WCHAR_T 1
#include_next "wchar.h"
#undef iswalpha 
#undef iswupper
#undef iswlower
#undef iswdigit
#undef iswxdigit
#undef iswalnum
#undef iswspace
#undef iswpunct
#undef iswprint
#undef iswgraph
#undef iswcntrl
#undef iswctype
#undef towlower
#undef towupper
#undef wcswcs
#undef wcswidth
#undef wcwidth
#undef wctype
