//========================================================================
// GLFW 3.4 Cocoa - www.glfw.org
//------------------------------------------------------------------------
// Copyright (c) 2009-2019 Camilla Löwy <elmindreda@glfw.org>
// Copyright (c) 2012 Torsten Walluhn <tw@mad-cad.net>
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
// It is fine to use C99 in this file because it will not be built with VS
//========================================================================

#include "internal.h"

#include <unistd.h>
#include <ctype.h>
#include <string.h>

#include <mach/mach.h>
#include <mach/mach_error.h>

#include <CoreFoundation/CoreFoundation.h>
#include <Kernel/IOKit/hidsystem/IOHIDUsageTables.h>


// Joystick element information
//
typedef struct _GLFWjoyelementNS
{
    IOHIDElementRef native;
    uint32_t        usage;
    int             index;
    long            minimum;
    long            maximum;

} _GLFWjoyelementNS;


// Returns the value of the specified element of the specified joystick
//
static long getElementValue(_GLFWjoystick* js, _GLFWjoyelementNS* element)
{
    IOHIDValueRef valueRef;
    long value = 0;

    if (js->ns.device)
    {
        if (IOHIDDeviceGetValue(js->ns.device,
                                element->native,
                                &valueRef) == kIOReturnSuccess)
        {
            value = IOHIDValueGetIntegerValue(valueRef);
        }
    }

    return value;
}

// Comparison function for matching the SDL element order
//
static CFComparisonResult compareElements(const void* fp,
                                          const void* sp,
                                          void* user)
{
    const _GLFWjoyelementNS* fe = fp;
    const _GLFWjoyelementNS* se = sp;
    if (fe->usage < se->usage)
        return kCFCompareLessThan;
    if (fe->usage > se->usage)
        return kCFCompareGreaterThan;
    if (fe->index < se->index)
        return kCFCompareLessThan;
    if (fe->index > se->index)
        return kCFCompareGreaterThan;
    return kCFCompareEqualTo;
}

// Removes the specified joystick
//
static void closeJoystick(_GLFWjoystick* js)
{
    int i;

    if (!js->present)
        return;

    for (i = 0;  i < CFArrayGetCount(js->ns.axes);  i++)
        free((void*) CFArrayGetValueAtIndex(js->ns.axes, i));
    CFRelease(js->ns.axes);

    for (i = 0;  i < CFArrayGetCount(js->ns.buttons);  i++)
        free((void*) CFArrayGetValueAtIndex(js->ns.buttons, i));
    CFRelease(js->ns.buttons);

    for (i = 0;  i < CFArrayGetCount(js->ns.hats);  i++)
        free((void*) CFArrayGetValueAtIndex(js->ns.hats, i));
    CFRelease(js->ns.hats);

    _glfwFreeJoystick(js);
    _glfwInputJoystick(js, GLFW_DISCONNECTED);
}

// Callback for user-initiated joystick addition
//
static void matchCallback(void* context,
                          IOReturn result,
                          void* sender,
                          IOHIDDeviceRef device)
{
    int jid;
    char name[256];
    char guid[33];
    CFIndex i;
    CFTypeRef property;
    uint32_t vendor = 0, product = 0, version = 0;
    _GLFWjoystick* js;
    CFMutableArrayRef axes, buttons, hats;

    for (jid = 0;  jid <= GLFW_JOYSTICK_LAST;  jid++)
    {
        if (_glfw.joysticks[jid].ns.device == device)
            return;
    }

    axes    = CFArrayCreateMutable(NULL, 0, NULL);
    buttons = CFArrayCreateMutable(NULL, 0, NULL);
    hats    = CFArrayCreateMutable(NULL, 0, NULL);

    property = IOHIDDeviceGetProperty(device, CFSTR(kIOHIDProductKey));
    if (property)
    {
        CFStringGetCString(property,
                           name,
                           sizeof(name),
                           kCFStringEncodingUTF8);
    }
    else
        strncpy(name, "Unknown", sizeof(name));

    property = IOHIDDeviceGetProperty(device, CFSTR(kIOHIDVendorIDKey));
    if (property)
        CFNumberGetValue(property, kCFNumberSInt32Type, &vendor);

    property = IOHIDDeviceGetProperty(device, CFSTR(kIOHIDProductIDKey));
    if (property)
        CFNumberGetValue(property, kCFNumberSInt32Type, &product);

    property = IOHIDDeviceGetProperty(device, CFSTR(kIOHIDVersionNumberKey));
    if (property)
        CFNumberGetValue(property, kCFNumberSInt32Type, &version);

    // Generate a joystick GUID that matches the SDL 2.0.5+ one
    if (vendor && product)
    {
        sprintf(guid, "03000000%02x%02x0000%02x%02x0000%02x%02x0000",
                (uint8_t) vendor, (uint8_t) (vendor >> 8),
                (uint8_t) product, (uint8_t) (product >> 8),
                (uint8_t) version, (uint8_t) (version >> 8));
    }
    else
    {
        sprintf(guid, "05000000%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x00",
                name[0], name[1], name[2], name[3],
                name[4], name[5], name[6], name[7],
                name[8], name[9], name[10]);
    }

    CFArrayRef elements =
        IOHIDDeviceCopyMatchingElements(device, NULL, kIOHIDOptionsTypeNone);

    for (i = 0;  i < CFArrayGetCount(elements);  i++)
    {
        IOHIDElementRef native = (IOHIDElementRef)
            CFArrayGetValueAtIndex(elements, i);
        if (CFGetTypeID(native) != IOHIDElementGetTypeID())
            continue;

        const IOHIDElementType type = IOHIDElementGetType(native);
        if ((type != kIOHIDElementTypeInput_Axis) &&
            (type != kIOHIDElementTypeInput_Button) &&
            (type != kIOHIDElementTypeInput_Misc))
        {
            continue;
        }

        CFMutableArrayRef target = NULL;

        const uint32_t usage = IOHIDElementGetUsage(native);
        const uint32_t page = IOHIDElementGetUsagePage(native);
        if (page == kHIDPage_GenericDesktop)
        {
            switch (usage)
            {
                case kHIDUsage_GD_X:
                case kHIDUsage_GD_Y:
                case kHIDUsage_GD_Z:
                case kHIDUsage_GD_Rx:
                case kHIDUsage_GD_Ry:
                case kHIDUsage_GD_Rz:
                case kHIDUsage_GD_Slider:
                case kHIDUsage_GD_Dial:
                case kHIDUsage_GD_Wheel:
                    target = axes;
                    break;
                case kHIDUsage_GD_Hatswitch:
                    target = hats;
                    break;
                case kHIDUsage_GD_DPadUp:
                case kHIDUsage_GD_DPadRight:
                case kHIDUsage_GD_DPadDown:
                case kHIDUsage_GD_DPadLeft:
                case kHIDUsage_GD_SystemMainMenu:
                case kHIDUsage_GD_Select:
                case kHIDUsage_GD_Start:
                    target = buttons;
                    break;
            }
        }
        else if (page == kHIDPage_Simulation)
        {
            switch (usage)
            {
                case kHIDUsage_Sim_Accelerator:
                case kHIDUsage_Sim_Brake:
                case kHIDUsage_Sim_Throttle:
                case kHIDUsage_Sim_Rudder:
                case kHIDUsage_Sim_Steering:
                    target = axes;
                    break;
            }
        }
        else if (page == kHIDPage_Button || page == kHIDPage_Consumer)
            target = buttons;

        if (target)
        {
            _GLFWjoyelementNS* element = calloc(1, sizeof(_GLFWjoyelementNS));
            element->native  = native;
            element->usage   = usage;
            element->index   = (int) CFArrayGetCount(target);
            element->minimum = IOHIDElementGetLogicalMin(native);
            element->maximum = IOHIDElementGetLogicalMax(native);
            CFArrayAppendValue(target, element);
        }
    }

    CFRelease(elements);

    CFArraySortValues(axes, CFRangeMake(0, CFArrayGetCount(axes)),
                      compareElements, NULL);
    CFArraySortValues(buttons, CFRangeMake(0, CFArrayGetCount(buttons)),
                      compareElements, NULL);
    CFArraySortValues(hats, CFRangeMake(0, CFArrayGetCount(hats)),
                      compareElements, NULL);

    js = _glfwAllocJoystick(name, guid,
                            (int) CFArrayGetCount(axes),
                            (int) CFArrayGetCount(buttons),
                            (int) CFArrayGetCount(hats));

    js->ns.device  = device;
    js->ns.axes    = axes;
    js->ns.buttons = buttons;
    js->ns.hats    = hats;

    _glfwInputJoystick(js, GLFW_CONNECTED);
}

// Callback for user-initiated joystick removal
//
static void removeCallback(void* context,
                           IOReturn result,
                           void* sender,
                           IOHIDDeviceRef device)
{
    int jid;

    for (jid = 0;  jid <= GLFW_JOYSTICK_LAST;  jid++)
    {
        if (_glfw.joysticks[jid].ns.device == device)
        {
            closeJoystick(_glfw.joysticks + jid);
            break;
        }
    }
}


//////////////////////////////////////////////////////////////////////////
//////                       GLFW internal API                      //////
//////////////////////////////////////////////////////////////////////////

// Initialize joystick interface
//
void _glfwInitJoysticksNS(void)
{
    CFMutableArrayRef matching;
    const long usages[] =
    {
        kHIDUsage_GD_Joystick,
        kHIDUsage_GD_GamePad,
        kHIDUsage_GD_MultiAxisController
    };

    _glfw.ns.hidManager = IOHIDManagerCreate(kCFAllocatorDefault,
                                             kIOHIDOptionsTypeNone);

    matching = CFArrayCreateMutable(kCFAllocatorDefault,
                                    0,
                                    &kCFTypeArrayCallBacks);
    if (!matching)
    {
        _glfwInputError(GLFW_PLATFORM_ERROR, "Cocoa: Failed to create array");
        return;
    }

    for (size_t i = 0;  i < sizeof(usages) / sizeof(long);  i++)
    {
        const long page = kHIDPage_GenericDesktop;

        CFMutableDictionaryRef dict =
            CFDictionaryCreateMutable(kCFAllocatorDefault,
                                      0,
                                      &kCFTypeDictionaryKeyCallBacks,
                                      &kCFTypeDictionaryValueCallBacks);
        if (!dict)
            continue;

        CFNumberRef pageRef = CFNumberCreate(kCFAllocatorDefault,
                                             kCFNumberLongType,
                                             &page);
        CFNumberRef usageRef = CFNumberCreate(kCFAllocatorDefault,
                                              kCFNumberLongType,
                                              &usages[i]);
        if (pageRef && usageRef)
        {
            CFDictionarySetValue(dict,
                                 CFSTR(kIOHIDDeviceUsagePageKey),
                                 pageRef);
            CFDictionarySetValue(dict,
                                 CFSTR(kIOHIDDeviceUsageKey),
                                 usageRef);
            CFArrayAppendValue(matching, dict);
        }

        if (pageRef)
            CFRelease(pageRef);
        if (usageRef)
            CFRelease(usageRef);

        CFRelease(dict);
    }

    IOHIDManagerSetDeviceMatchingMultiple(_glfw.ns.hidManager, matching);
    CFRelease(matching);

    IOHIDManagerRegisterDeviceMatchingCallback(_glfw.ns.hidManager,
                                               &matchCallback, NULL);
    IOHIDManagerRegisterDeviceRemovalCallback(_glfw.ns.hidManager,
                                              &removeCallback, NULL);
    IOHIDManagerScheduleWithRunLoop(_glfw.ns.hidManager,
                                    CFRunLoopGetMain(),
                                    kCFRunLoopDefaultMode);
    IOHIDManagerOpen(_glfw.ns.hidManager, kIOHIDOptionsTypeNone);

    // Execute the run loop once in order to register any initially-attached
    // joysticks
    CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0, false);
}

// Close all opened joystick handles
//
void _glfwTerminateJoysticksNS(void)
{
    int jid;

    for (jid = 0;  jid <= GLFW_JOYSTICK_LAST;  jid++)
        closeJoystick(_glfw.joysticks + jid);

    CFRelease(_glfw.ns.hidManager);
    _glfw.ns.hidManager = NULL;
}


//////////////////////////////////////////////////////////////////////////
//////                       GLFW platform API                      //////
//////////////////////////////////////////////////////////////////////////

int _glfwPlatformPollJoystick(_GLFWjoystick* js, int mode)
{
    if (mode & _GLFW_POLL_AXES)
    {
        CFIndex i;

        for (i = 0;  i < CFArrayGetCount(js->ns.axes);  i++)
        {
            _GLFWjoyelementNS* axis = (_GLFWjoyelementNS*)
                CFArrayGetValueAtIndex(js->ns.axes, i);

            const long raw = getElementValue(js, axis);
            // Perform auto calibration
            if (raw < axis->minimum)
                axis->minimum = raw;
            if (raw > axis->maximum)
                axis->maximum = raw;

            const long size = axis->maximum - axis->minimum;
            if (size == 0)
                _glfwInputJoystickAxis(js, (int) i, 0.f);
            else
            {
                const float value = (2.f * (raw - axis->minimum) / size) - 1.f;
                _glfwInputJoystickAxis(js, (int) i, value);
            }
        }
    }

    if (mode & _GLFW_POLL_BUTTONS)
    {
        CFIndex i;

        for (i = 0;  i < CFArrayGetCount(js->ns.buttons);  i++)
        {
            _GLFWjoyelementNS* button = (_GLFWjoyelementNS*)
                CFArrayGetValueAtIndex(js->ns.buttons, i);
            const char value = getElementValue(js, button) - button->minimum;
            const int state = (value > 0) ? GLFW_PRESS : GLFW_RELEASE;
            _glfwInputJoystickButton(js, (int) i, state);
        }

        for (i = 0;  i < CFArrayGetCount(js->ns.hats);  i++)
        {
            const int states[9] =
            {
                GLFW_HAT_UP,
                GLFW_HAT_RIGHT_UP,
                GLFW_HAT_RIGHT,
                GLFW_HAT_RIGHT_DOWN,
                GLFW_HAT_DOWN,
                GLFW_HAT_LEFT_DOWN,
                GLFW_HAT_LEFT,
                GLFW_HAT_LEFT_UP,
                GLFW_HAT_CENTERED
            };

            _GLFWjoyelementNS* hat = (_GLFWjoyelementNS*)
                CFArrayGetValueAtIndex(js->ns.hats, i);
            long state = getElementValue(js, hat) - hat->minimum;
            if (state < 0 || state > 8)
                state = 8;

            _glfwInputJoystickHat(js, (int) i, states[state]);
        }
    }

    return js->present;
}

void _glfwPlatformUpdateGamepadGUID(char* guid)
{
    if ((strncmp(guid + 4, "000000000000", 12) == 0) &&
        (strncmp(guid + 20, "000000000000", 12) == 0))
    {
        char original[33];
        strncpy(original, guid, sizeof(original) - 1);
        sprintf(guid, "03000000%.4s0000%.4s000000000000",
                original, original + 16);
    }
}

