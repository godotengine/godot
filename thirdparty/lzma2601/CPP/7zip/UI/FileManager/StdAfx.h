// StdAfx.h

#ifndef ZIP7_INC_STDAFX_H
#define ZIP7_INC_STDAFX_H

#if defined(_MSC_VER) && _MSC_VER >= 1800
#pragma warning(disable : 4464) // relative include path contains '..'
#endif

#include "../../../Common/Common.h"

#endif

/*
WINVER and _WIN32_WINNT

MSVC6 / 2003sdk:
{
  <windows.h> doesn't set _WIN32_WINNT
  if WINVER is not set <windows.h> sets WINVER to value:
    0x0400 : MSVC6
    0x0501 : Windows Server 2003 PSDK / 2003 R2 PSDK
}

SDK for Win7 (and later)
{
  <windows.h> sets _WIN32_WINNT if it's not set.
  <windows.h> sets WINVER if it's not set.
<windows.h> includes <sdkddkver.h> that does:
#if !defined(_WIN32_WINNT) && !defined(_CHICAGO_)
  #define _WIN32_WINNT 0x0601  // in win7 sdk
  #define _WIN32_WINNT 0x0A00  // in win10 sdk
#endif
#ifndef WINVER
 #ifdef _WIN32_WINNT
  #define WINVER _WIN32_WINNT
 else
  #define WINVER 0x0601 // in win7 sdk
  #define WINVER 0x0A00 // in win10 sdk
 endif
#endif
}

Some GUI structures defined by windows will be larger,
If (_WIN32_WINNT) value is larger.

Also if we send sizeof(win_gui_struct) to some windows function,
and we compile that code with big (_WIN32_WINNT) value,
the window function in old Windows can fail, if that old Windows
doesn't understand new big version of (win_gui_struct) compiled
with big (_WIN32_WINNT) value.

So it's better to define smallest (_WIN32_WINNT) value here.
In 7-Zip FM we use some functions that require (_WIN32_WINNT == 0x0500).
So it's simpler to define (_WIN32_WINNT == 0x0500) here.
If we define (_WIN32_WINNT == 0x0400) here, we need some manual
declarations for functions and macros that require (0x0500) functions.
Also libs must contain these (0x0500+) functions.

Some code in 7-zip FM uses also CommCtrl.h structures
that depend from (_WIN32_IE) value. But default
(_WIN32_IE) value from <windows.h> probably is OK for us.
So we don't set _WIN32_IE here.
default _WIN32_IE value set by <windows.h>:
  0x501 2003sdk
  0xa00 win10 sdk
*/
