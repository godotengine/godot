//========================================================================
// GLFW 3.4 Win32 - www.glfw.org
//------------------------------------------------------------------------
// Copyright (c) 2002-2006 Marcus Geelnard
// Copyright (c) 2006-2017 Camilla Löwy <elmindreda@glfw.org>
//
// This software is provided 'as-is', without any express or implied
// warranty. In no event will the authors be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you must not
//    claim that you wrote the original software. If you use this software
//    in a product, an acknowledgment in the product documentation would
//    be appreciated but is not required.
//
// 2. Altered source versions must be plainly marked as such, and must not
//    be misrepresented as being the original software.
//
// 3. This notice may not be removed or altered from any source
//    distribution.
//
//========================================================================
// Please use C89 style variable declarations in this file because VS 2010
//========================================================================

#include "internal.h"


//////////////////////////////////////////////////////////////////////////
//////                       GLFW internal API                      //////
//////////////////////////////////////////////////////////////////////////

// Initialise timer
//
void _glfwInitTimerWin32(void)
{
    uint64_t frequency;

    if (QueryPerformanceFrequency((LARGE_INTEGER*) &frequency))
    {
        _glfw.timer.win32.hasPC = GLFW_TRUE;
        _glfw.timer.win32.frequency = frequency;
    }
    else
    {
        _glfw.timer.win32.hasPC = GLFW_FALSE;
        _glfw.timer.win32.frequency = 1000;
    }
}


//////////////////////////////////////////////////////////////////////////
//////                       GLFW platform API                      //////
//////////////////////////////////////////////////////////////////////////

uint64_t _glfwPlatformGetTimerValue(void)
{
    if (_glfw.timer.win32.hasPC)
    {
        uint64_t value;
        QueryPerformanceCounter((LARGE_INTEGER*) &value);
        return value;
    }
    else
        return (uint64_t) timeGetTime();
}

uint64_t _glfwPlatformGetTimerFrequency(void)
{
    return _glfw.timer.win32.frequency;
}

