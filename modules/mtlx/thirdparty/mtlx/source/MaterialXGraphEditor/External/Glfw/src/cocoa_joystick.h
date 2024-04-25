//========================================================================
// GLFW 3.4 Cocoa - www.glfw.org
//------------------------------------------------------------------------
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

#include <IOKit/IOKitLib.h>
#include <IOKit/IOCFPlugIn.h>
#include <IOKit/hid/IOHIDLib.h>
#include <IOKit/hid/IOHIDKeys.h>

#define _GLFW_PLATFORM_JOYSTICK_STATE         _GLFWjoystickNS ns
#define _GLFW_PLATFORM_LIBRARY_JOYSTICK_STATE struct { int dummyJoystick; }

#define _GLFW_PLATFORM_MAPPING_NAME "Mac OS X"

// Cocoa-specific per-joystick data
//
typedef struct _GLFWjoystickNS
{
    IOHIDDeviceRef      device;
    CFMutableArrayRef   axes;
    CFMutableArrayRef   buttons;
    CFMutableArrayRef   hats;
} _GLFWjoystickNS;


void _glfwInitJoysticksNS(void);
void _glfwTerminateJoysticksNS(void);

