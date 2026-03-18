/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2025 Sam Lantinga <slouken@libsdl.org>

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/
#include "SDL_internal.h"

// This is the iOS implementation of the SDL joystick API
#include "../SDL_sysjoystick.h"
#include "../SDL_joystick_c.h"
#include "../hidapi/SDL_hidapijoystick_c.h"
#include "../usb_ids.h"
#include "../../events/SDL_events_c.h"

#include "SDL_mfijoystick_c.h"


#if defined(SDL_PLATFORM_IOS) && !defined(SDL_PLATFORM_TVOS)
#import <CoreMotion/CoreMotion.h>
#endif

#ifdef SDL_PLATFORM_MACOS
#include <IOKit/hid/IOHIDManager.h>
#include <AppKit/NSApplication.h>
#ifndef NSAppKitVersionNumber10_15
#define NSAppKitVersionNumber10_15 1894
#endif
#endif // SDL_PLATFORM_MACOS

#import <GameController/GameController.h>

#ifdef SDL_JOYSTICK_MFI
static id connectObserver = nil;
static id disconnectObserver = nil;

#include <objc/message.h>

// Fix build errors when using an older SDK by defining these selectors
@interface GCController (SDL)
#if !((__IPHONE_OS_VERSION_MAX_ALLOWED >= 140500) || (__APPLETV_OS_VERSION_MAX_ALLOWED >= 140500) || (__MAC_OS_X_VERSION_MAX_ALLOWED >= 110300))
@property(class, nonatomic, readwrite) BOOL shouldMonitorBackgroundEvents;
#endif
@end

#import <CoreHaptics/CoreHaptics.h>

#endif // SDL_JOYSTICK_MFI

static SDL_JoystickDeviceItem *deviceList = NULL;

static int numjoysticks = 0;
int SDL_AppleTVRemoteOpenedAsJoystick = 0;

static SDL_JoystickDeviceItem *GetDeviceForIndex(int device_index)
{
    SDL_JoystickDeviceItem *device = deviceList;
    int i = 0;

    while (i < device_index) {
        if (device == NULL) {
            return NULL;
        }
        device = device->next;
        i++;
    }

    return device;
}

#ifdef SDL_JOYSTICK_MFI
static bool IsControllerPS4(GCController *controller)
{
    if (@available(macOS 10.15, iOS 13.0, tvOS 13.0, *)) {
        if ([controller.productCategory isEqualToString:@"DualShock 4"]) {
            return true;
        }
    } else {
        if ([controller.vendorName containsString:@"DUALSHOCK"]) {
            return true;
        }
    }
    return false;
}
static bool IsControllerPS5(GCController *controller)
{
    if (@available(macOS 10.15, iOS 13.0, tvOS 13.0, *)) {
        if ([controller.productCategory isEqualToString:@"DualSense"]) {
            return true;
        }
    } else {
        if ([controller.vendorName containsString:@"DualSense"]) {
            return true;
        }
    }
    return false;
}
static bool IsControllerXbox(GCController *controller)
{
    if (@available(macOS 10.15, iOS 13.0, tvOS 13.0, *)) {
        if ([controller.productCategory isEqualToString:@"Xbox One"]) {
            return true;
        }
    } else {
        if ([controller.vendorName containsString:@"Xbox"]) {
            return true;
        }
    }
    return false;
}
static bool IsControllerSwitchPro(GCController *controller)
{
    if (@available(macOS 10.15, iOS 13.0, tvOS 13.0, *)) {
        if ([controller.productCategory isEqualToString:@"Switch Pro Controller"]) {
            return true;
        }
    }
    return false;
}
static bool IsControllerSwitchJoyConL(GCController *controller)
{
    if (@available(macOS 10.15, iOS 13.0, tvOS 13.0, *)) {
        if ([controller.productCategory isEqualToString:@"Nintendo Switch Joy-Con (L)"]) {
            return true;
        }
    }
    return false;
}
static bool IsControllerSwitchJoyConR(GCController *controller)
{
    if (@available(macOS 10.15, iOS 13.0, tvOS 13.0, *)) {
        if ([controller.productCategory isEqualToString:@"Nintendo Switch Joy-Con (R)"]) {
            return true;
        }
    }
    return false;
}
static bool IsControllerSwitchJoyConPair(GCController *controller)
{
    if (@available(macOS 10.15, iOS 13.0, tvOS 13.0, *)) {
        if ([controller.productCategory isEqualToString:@"Nintendo Switch Joy-Con (L/R)"]) {
            return true;
        }
    }
    return false;
}
static bool IsControllerStadia(GCController *controller)
{
    if ([controller.vendorName hasPrefix:@"Stadia"]) {
        return true;
    }
    return false;
}
static bool IsControllerBackboneOne(GCController *controller)
{
    if ([controller.vendorName hasPrefix:@"Backbone One"]) {
        return true;
    }
    return false;
}
static void CheckControllerSiriRemote(GCController *controller, int *is_siri_remote)
{
    if (@available(macOS 10.15, iOS 13.0, tvOS 13.0, *)) {
        if ([controller.productCategory hasPrefix:@"Siri Remote"]) {
            *is_siri_remote = 1;
            SDL_sscanf(controller.productCategory.UTF8String, "Siri Remote (%i%*s Generation)", is_siri_remote);
            return;
        }
    }
    *is_siri_remote = 0;
}

static bool ElementAlreadyHandled(SDL_JoystickDeviceItem *device, NSString *element, NSDictionary<NSString *, GCControllerElement *> *elements)
{
    if ([element isEqualToString:@"Left Thumbstick Left"] ||
        [element isEqualToString:@"Left Thumbstick Right"]) {
        if (elements[@"Left Thumbstick X Axis"]) {
            return true;
        }
    }
    if ([element isEqualToString:@"Left Thumbstick Up"] ||
        [element isEqualToString:@"Left Thumbstick Down"]) {
        if (elements[@"Left Thumbstick Y Axis"]) {
            return true;
        }
    }
    if ([element isEqualToString:@"Right Thumbstick Left"] ||
        [element isEqualToString:@"Right Thumbstick Right"]) {
        if (elements[@"Right Thumbstick X Axis"]) {
            return true;
        }
    }
    if ([element isEqualToString:@"Right Thumbstick Up"] ||
        [element isEqualToString:@"Right Thumbstick Down"]) {
        if (elements[@"Right Thumbstick Y Axis"]) {
            return true;
        }
    }
    if (device->is_siri_remote) {
        if ([element isEqualToString:@"Direction Pad Left"] ||
            [element isEqualToString:@"Direction Pad Right"]) {
            if (elements[@"Direction Pad X Axis"]) {
                return true;
            }
        }
        if ([element isEqualToString:@"Direction Pad Up"] ||
            [element isEqualToString:@"Direction Pad Down"]) {
            if (elements[@"Direction Pad Y Axis"]) {
                return true;
            }
        }
    } else {
        if ([element isEqualToString:@"Direction Pad X Axis"]) {
            if (elements[@"Direction Pad Left"] &&
                elements[@"Direction Pad Right"]) {
                return true;
            }
        }
        if ([element isEqualToString:@"Direction Pad Y Axis"]) {
            if (elements[@"Direction Pad Up"] &&
                elements[@"Direction Pad Down"]) {
                return true;
            }
        }
    }
    if ([element isEqualToString:@"Cardinal Direction Pad X Axis"]) {
        if (elements[@"Cardinal Direction Pad Left"] &&
            elements[@"Cardinal Direction Pad Right"]) {
            return true;
        }
    }
    if ([element isEqualToString:@"Cardinal Direction Pad Y Axis"]) {
        if (elements[@"Cardinal Direction Pad Up"] &&
            elements[@"Cardinal Direction Pad Down"]) {
            return true;
        }
    }
    if ([element isEqualToString:@"Touchpad 1 X Axis"] ||
        [element isEqualToString:@"Touchpad 1 Y Axis"] ||
        [element isEqualToString:@"Touchpad 1 Left"] ||
        [element isEqualToString:@"Touchpad 1 Right"] ||
        [element isEqualToString:@"Touchpad 1 Up"] ||
        [element isEqualToString:@"Touchpad 1 Down"] ||
        [element isEqualToString:@"Touchpad 2 X Axis"] ||
        [element isEqualToString:@"Touchpad 2 Y Axis"] ||
        [element isEqualToString:@"Touchpad 2 Left"] ||
        [element isEqualToString:@"Touchpad 2 Right"] ||
        [element isEqualToString:@"Touchpad 2 Up"] ||
        [element isEqualToString:@"Touchpad 2 Down"]) {
        // The touchpad is handled separately
        return true;
    }
    if ([element isEqualToString:@"Button Home"]) {
        if (device->is_switch_joycon_pair) {
            // The Nintendo Switch JoyCon home button doesn't ever show as being held down
            return true;
        }
#ifdef SDL_PLATFORM_TVOS
        // The OS uses the home button, it's not available to apps
        return true;
#endif
    }
    if ([element isEqualToString:@"Button Share"]) {
        if (device->is_backbone_one) {
            // The Backbone app uses share button
            return true;
        }
    }
    return false;
}

static bool IOS_AddMFIJoystickDevice(SDL_JoystickDeviceItem *device, GCController *controller)
{
    Uint16 vendor = 0;
    Uint16 product = 0;
    Uint8 subtype = 0;
    const char *name = NULL;

    if (@available(macOS 11.3, iOS 14.5, tvOS 14.5, *)) {
        if (!GCController.shouldMonitorBackgroundEvents) {
            GCController.shouldMonitorBackgroundEvents = YES;
        }
    }

    /* Explicitly retain the controller because SDL_JoystickDeviceItem is a
     * struct, and ARC doesn't work with structs. */
    device->controller = (__bridge GCController *)CFBridgingRetain(controller);

    if (controller.vendorName) {
        name = controller.vendorName.UTF8String;
    }

    if (!name) {
        name = "MFi Gamepad";
    }

    device->name = SDL_CreateJoystickName(0, 0, NULL, name);

#ifdef DEBUG_CONTROLLER_PROFILE
    NSLog(@"Product name: %@\n", controller.vendorName);
    NSLog(@"Product category: %@\n", controller.productCategory);
    NSLog(@"Elements available:\n");
    if (@available(macOS 10.16, iOS 14.0, tvOS 14.0, *)) {
        NSDictionary<NSString *, GCControllerElement *> *elements = controller.physicalInputProfile.elements;
        for (id key in controller.physicalInputProfile.buttons) {
            NSLog(@"\tButton: %@ (%s)\n", key, elements[key].analog ? "analog" : "digital");
        }
        for (id key in controller.physicalInputProfile.axes) {
            NSLog(@"\tAxis: %@\n", key);
        }
        for (id key in controller.physicalInputProfile.dpads) {
            NSLog(@"\tHat: %@\n", key);
        }
    }
#endif // DEBUG_CONTROLLER_PROFILE

    device->is_xbox = IsControllerXbox(controller);
    device->is_ps4 = IsControllerPS4(controller);
    device->is_ps5 = IsControllerPS5(controller);
    device->is_switch_pro = IsControllerSwitchPro(controller);
    device->is_switch_joycon_pair = IsControllerSwitchJoyConPair(controller);
    device->is_stadia = IsControllerStadia(controller);
    device->is_backbone_one = IsControllerBackboneOne(controller);
    device->is_switch_joyconL = IsControllerSwitchJoyConL(controller);
    device->is_switch_joyconR = IsControllerSwitchJoyConR(controller);
#ifdef SDL_JOYSTICK_HIDAPI
    if ((device->is_xbox && (HIDAPI_IsDeviceTypePresent(SDL_GAMEPAD_TYPE_XBOXONE) ||
                             HIDAPI_IsDeviceTypePresent(SDL_GAMEPAD_TYPE_XBOX360))) ||
        (device->is_ps4 && HIDAPI_IsDeviceTypePresent(SDL_GAMEPAD_TYPE_PS4)) ||
        (device->is_ps5 && HIDAPI_IsDeviceTypePresent(SDL_GAMEPAD_TYPE_PS5)) ||
        (device->is_switch_pro && HIDAPI_IsDeviceTypePresent(SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_PRO)) ||
        (device->is_switch_joycon_pair && HIDAPI_IsDevicePresent(USB_VENDOR_NINTENDO, USB_PRODUCT_NINTENDO_SWITCH_JOYCON_PAIR, 0, "")) ||
        (device->is_stadia && HIDAPI_IsDevicePresent(USB_VENDOR_GOOGLE, USB_PRODUCT_GOOGLE_STADIA_CONTROLLER, 0, "")) ||
        (device->is_switch_joyconL && HIDAPI_IsDevicePresent(USB_VENDOR_NINTENDO, USB_PRODUCT_NINTENDO_SWITCH_JOYCON_LEFT, 0, "")) ||
        (device->is_switch_joyconR && HIDAPI_IsDevicePresent(USB_VENDOR_NINTENDO, USB_PRODUCT_NINTENDO_SWITCH_JOYCON_RIGHT, 0, ""))) {
        // The HIDAPI driver is taking care of this device
        return false;
    }
#endif
    if (device->is_xbox && SDL_strncmp(name, "GamePad-", 8) == 0) {
        // This is a Steam Virtual Gamepad, which isn't supported by GCController
        return false;
    }
    CheckControllerSiriRemote(controller, &device->is_siri_remote);

    if (device->is_siri_remote && !SDL_GetHintBoolean(SDL_HINT_TV_REMOTE_AS_JOYSTICK, true)) {
        // Ignore remotes, they'll be handled as keyboard input
        return false;
    }

    if (@available(macOS 10.16, iOS 14.0, tvOS 14.0, *)) {
        if (controller.physicalInputProfile.buttons[GCInputDualShockTouchpadButton] != nil) {
            device->has_dualshock_touchpad = TRUE;
        }
        if (controller.physicalInputProfile.buttons[GCInputXboxPaddleOne] != nil) {
            device->has_xbox_paddles = TRUE;
        }
        if (controller.physicalInputProfile.buttons[@"Button Share"] != nil) {
            device->has_xbox_share_button = TRUE;
        }
    }

    if (device->is_backbone_one) {
        vendor = USB_VENDOR_BACKBONE;
        if (device->is_ps5) {
            product = USB_PRODUCT_BACKBONE_ONE_IOS_PS5;
        } else {
            product = USB_PRODUCT_BACKBONE_ONE_IOS;
        }
    } else if (device->is_xbox) {
        vendor = USB_VENDOR_MICROSOFT;
        if (device->has_xbox_paddles) {
            // Assume Xbox One Elite Series 2 Controller unless/until GCController flows VID/PID
            product = USB_PRODUCT_XBOX_ONE_ELITE_SERIES_2_BLUETOOTH;
        } else if (device->has_xbox_share_button) {
            // Assume Xbox Series X Controller unless/until GCController flows VID/PID
            product = USB_PRODUCT_XBOX_SERIES_X_BLE;
        } else {
            // Assume Xbox One S Bluetooth Controller unless/until GCController flows VID/PID
            product = USB_PRODUCT_XBOX_ONE_S_REV1_BLUETOOTH;
        }
    } else if (device->is_ps4) {
        // Assume DS4 Slim unless/until GCController flows VID/PID
        vendor = USB_VENDOR_SONY;
        product = USB_PRODUCT_SONY_DS4_SLIM;
        if (device->has_dualshock_touchpad) {
            subtype = 1;
        }
    } else if (device->is_ps5) {
        vendor = USB_VENDOR_SONY;
        product = USB_PRODUCT_SONY_DS5;
    } else if (device->is_switch_pro) {
        vendor = USB_VENDOR_NINTENDO;
        product = USB_PRODUCT_NINTENDO_SWITCH_PRO;
        device->has_nintendo_buttons = TRUE;
    } else if (device->is_switch_joycon_pair) {
        vendor = USB_VENDOR_NINTENDO;
        product = USB_PRODUCT_NINTENDO_SWITCH_JOYCON_PAIR;
        device->has_nintendo_buttons = TRUE;
    } else if (device->is_switch_joyconL) {
        vendor = USB_VENDOR_NINTENDO;
        product = USB_PRODUCT_NINTENDO_SWITCH_JOYCON_LEFT;
    } else if (device->is_switch_joyconR) {
        vendor = USB_VENDOR_NINTENDO;
        product = USB_PRODUCT_NINTENDO_SWITCH_JOYCON_RIGHT;
    } else if (@available(macOS 10.16, iOS 14.0, tvOS 14.0, *)) {
        vendor = USB_VENDOR_APPLE;
        product = 4;
        subtype = 4;
    } else if (controller.extendedGamepad) {
        vendor = USB_VENDOR_APPLE;
        product = 1;
        subtype = 1;
#ifdef SDL_PLATFORM_TVOS
    } else if (controller.microGamepad) {
        vendor = USB_VENDOR_APPLE;
        product = 3;
        subtype = 3;
#endif
    } else {
        // We don't know how to get input events from this device
        return false;
    }

    if (SDL_ShouldIgnoreJoystick(vendor, product, 0, name)) {
        return false;
    }

    if (@available(macOS 10.16, iOS 14.0, tvOS 14.0, *)) {
        NSDictionary<NSString *, GCControllerElement *> *elements = controller.physicalInputProfile.elements;

        // Provide both axes and analog buttons as SDL axes
        NSArray *axes = [[[elements allKeys] sortedArrayUsingSelector:@selector(caseInsensitiveCompare:)]
                                         filteredArrayUsingPredicate:[NSPredicate predicateWithBlock:^BOOL(id object, NSDictionary *bindings) {
            if (ElementAlreadyHandled(device, (NSString *)object, elements)) {
                return false;
            }

            GCControllerElement *element = elements[object];
            if (element.analog) {
                if ([element isKindOfClass:[GCControllerAxisInput class]] ||
                    [element isKindOfClass:[GCControllerButtonInput class]]) {
                    return true;
                }
            }
            return false;
        }]];
        NSArray *buttons = [[[elements allKeys] sortedArrayUsingSelector:@selector(caseInsensitiveCompare:)]
                                            filteredArrayUsingPredicate:[NSPredicate predicateWithBlock:^BOOL(id object, NSDictionary *bindings) {
            if (ElementAlreadyHandled(device, (NSString *)object, elements)) {
                return false;
            }

            GCControllerElement *element = elements[object];
            if ([element isKindOfClass:[GCControllerButtonInput class]]) {
                return true;
            }
            return false;
        }]];
        /* Explicitly retain the arrays because SDL_JoystickDeviceItem is a
         * struct, and ARC doesn't work with structs. */
        device->naxes = (int)axes.count;
        device->axes = (__bridge NSArray *)CFBridgingRetain(axes);
        device->nbuttons = (int)buttons.count;
        device->buttons = (__bridge NSArray *)CFBridgingRetain(buttons);
        subtype = 4;

#ifdef DEBUG_CONTROLLER_PROFILE
        NSLog(@"Elements used:\n", controller.vendorName);
        for (id key in device->buttons) {
            NSLog(@"\tButton: %@ (%s)\n", key, elements[key].analog ? "analog" : "digital");
        }
        for (id key in device->axes) {
            NSLog(@"\tAxis: %@\n", key);
        }
#endif // DEBUG_CONTROLLER_PROFILE

#ifdef SDL_PLATFORM_TVOS
        // tvOS turns the menu button into a system gesture, so we grab it here instead
        if (elements[GCInputButtonMenu] && !elements[@"Button Home"]) {
            device->pause_button_index = (int)[device->buttons indexOfObject:GCInputButtonMenu];
        }
#endif
    } else if (controller.extendedGamepad) {
        GCExtendedGamepad *gamepad = controller.extendedGamepad;
        int nbuttons = 0;
        BOOL has_direct_menu = FALSE;

        // These buttons are part of the original MFi spec
        device->button_mask |= (1 << SDL_GAMEPAD_BUTTON_SOUTH);
        device->button_mask |= (1 << SDL_GAMEPAD_BUTTON_EAST);
        device->button_mask |= (1 << SDL_GAMEPAD_BUTTON_WEST);
        device->button_mask |= (1 << SDL_GAMEPAD_BUTTON_NORTH);
        device->button_mask |= (1 << SDL_GAMEPAD_BUTTON_LEFT_SHOULDER);
        device->button_mask |= (1 << SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER);
        nbuttons += 6;

        // These buttons are available on some newer controllers
        if (@available(macOS 10.14.1, iOS 12.1, tvOS 12.1, *)) {
            if (gamepad.leftThumbstickButton) {
                device->button_mask |= (1 << SDL_GAMEPAD_BUTTON_LEFT_STICK);
                ++nbuttons;
            }
            if (gamepad.rightThumbstickButton) {
                device->button_mask |= (1 << SDL_GAMEPAD_BUTTON_RIGHT_STICK);
                ++nbuttons;
            }
        }
        if (@available(macOS 10.15, iOS 13.0, tvOS 13.0, *)) {
            if (gamepad.buttonOptions) {
                device->button_mask |= (1 << SDL_GAMEPAD_BUTTON_BACK);
                ++nbuttons;
            }
        }
        device->button_mask |= (1 << SDL_GAMEPAD_BUTTON_START);
        ++nbuttons;

        if (@available(macOS 10.15, iOS 13.0, tvOS 13.0, *)) {
            if (gamepad.buttonMenu) {
                has_direct_menu = TRUE;
            }
        }
#ifdef SDL_PLATFORM_TVOS
        // The single menu button isn't very reliable, at least as of tvOS 16.1
        if ((device->button_mask & (1 << SDL_GAMEPAD_BUTTON_BACK)) == 0) {
            has_direct_menu = FALSE;
        }
#endif
        if (!has_direct_menu) {
            device->pause_button_index = (nbuttons - 1);
        }

        device->naxes = 6; // 2 thumbsticks and 2 triggers
        device->nhats = 1; // d-pad
        device->nbuttons = nbuttons;
    }
#ifdef SDL_PLATFORM_TVOS
    else if (controller.microGamepad) {
        int nbuttons = 0;

        device->button_mask |= (1 << SDL_GAMEPAD_BUTTON_SOUTH);
        device->button_mask |= (1 << SDL_GAMEPAD_BUTTON_WEST); // Button X on microGamepad
        device->button_mask |= (1 << SDL_GAMEPAD_BUTTON_EAST);
        nbuttons += 3;
        device->pause_button_index = (nbuttons - 1);

        device->naxes = 2; // treat the touch surface as two axes
        device->nhats = 0; // apparently the touch surface-as-dpad is buggy
        device->nbuttons = nbuttons;

        controller.microGamepad.allowsRotation = SDL_GetHintBoolean(SDL_HINT_APPLE_TV_REMOTE_ALLOW_ROTATION, false);
    }
#endif
    else {
        // We don't know how to get input events from this device
        return false;
    }

    Uint16 signature;
    if (@available(macOS 10.16, iOS 14.0, tvOS 14.0, *)) {
        signature = 0;
        signature = SDL_crc16(signature, device->name, SDL_strlen(device->name));
        for (id key in device->axes) {
            const char *string = ((NSString *)key).UTF8String;
            signature = SDL_crc16(signature, string, SDL_strlen(string));
        }
        for (id key in device->buttons) {
            const char *string = ((NSString *)key).UTF8String;
            signature = SDL_crc16(signature, string, SDL_strlen(string));
        }
    } else {
        signature = device->button_mask;
    }
    device->guid = SDL_CreateJoystickGUID(SDL_HARDWARE_BUS_BLUETOOTH, vendor, product, signature, NULL, name, 'm', subtype);

    /* This will be set when the first button press of the controller is
     * detected. */
    controller.playerIndex = -1;
    return true;
}
#endif // SDL_JOYSTICK_MFI

#ifdef SDL_JOYSTICK_MFI
static void IOS_AddJoystickDevice(GCController *controller)
{
    SDL_JoystickDeviceItem *device = deviceList;

    while (device != NULL) {
        if (device->controller == controller) {
            return;
        }
        device = device->next;
    }

    device = (SDL_JoystickDeviceItem *)SDL_calloc(1, sizeof(SDL_JoystickDeviceItem));
    if (device == NULL) {
        return;
    }

    device->instance_id = SDL_GetNextObjectID();
    device->pause_button_index = -1;

    if (controller) {
#ifdef SDL_JOYSTICK_MFI
        if (!IOS_AddMFIJoystickDevice(device, controller)) {
            SDL_free(device->name);
            SDL_free(device);
            return;
        }
#else
        SDL_free(device);
        return;
#endif // SDL_JOYSTICK_MFI
    }

    if (deviceList == NULL) {
        deviceList = device;
    } else {
        SDL_JoystickDeviceItem *lastdevice = deviceList;
        while (lastdevice->next != NULL) {
            lastdevice = lastdevice->next;
        }
        lastdevice->next = device;
    }

    ++numjoysticks;

    SDL_PrivateJoystickAdded(device->instance_id);
}
#endif // SDL_JOYSTICK_MFI

static SDL_JoystickDeviceItem *IOS_RemoveJoystickDevice(SDL_JoystickDeviceItem *device)
{
    SDL_JoystickDeviceItem *prev = NULL;
    SDL_JoystickDeviceItem *next = NULL;
    SDL_JoystickDeviceItem *item = deviceList;

    if (device == NULL) {
        return NULL;
    }

    next = device->next;

    while (item != NULL) {
        if (item == device) {
            break;
        }
        prev = item;
        item = item->next;
    }

    // Unlink the device item from the device list.
    if (prev) {
        prev->next = device->next;
    } else if (device == deviceList) {
        deviceList = device->next;
    }

    if (device->joystick) {
        device->joystick->hwdata = NULL;
    }

#ifdef SDL_JOYSTICK_MFI
    @autoreleasepool {
        // These were explicitly retained in the struct, so they should be explicitly released before freeing the struct.
        if (device->controller) {
            GCController *controller = CFBridgingRelease((__bridge CFTypeRef)(device->controller));
            controller.controllerPausedHandler = nil;
            device->controller = nil;
        }
        if (device->axes) {
            CFRelease((__bridge CFTypeRef)device->axes);
            device->axes = nil;
        }
        if (device->buttons) {
            CFRelease((__bridge CFTypeRef)device->buttons);
            device->buttons = nil;
        }
    }
#endif // SDL_JOYSTICK_MFI

    --numjoysticks;

    SDL_PrivateJoystickRemoved(device->instance_id);

    SDL_free(device->name);
    SDL_free(device);

    return next;
}

#ifdef SDL_PLATFORM_TVOS
static void SDLCALL SDL_AppleTVRemoteRotationHintChanged(void *udata, const char *name, const char *oldValue, const char *newValue)
{
    BOOL allowRotation = newValue != NULL && *newValue != '0';

    @autoreleasepool {
        for (GCController *controller in [GCController controllers]) {
            if (controller.microGamepad) {
                controller.microGamepad.allowsRotation = allowRotation;
            }
        }
    }
}
#endif // SDL_PLATFORM_TVOS

static bool IOS_JoystickInit(void)
{
    if (!SDL_GetHintBoolean(SDL_HINT_JOYSTICK_MFI, true)) {
        return true;
    }

#ifdef SDL_PLATFORM_MACOS
    if (@available(macOS 10.16, *)) {
        // Continue with initialization on macOS 11+
    } else {
        return true;
    }
#endif

    @autoreleasepool {
#ifdef SDL_JOYSTICK_MFI
        NSNotificationCenter *center;
#endif

#ifdef SDL_JOYSTICK_MFI
        // GameController.framework was added in iOS 7.
        if (![GCController class]) {
            return true;
        }

        /* For whatever reason, this always returns an empty array on
         macOS 11.0.1 */
        for (GCController *controller in [GCController controllers]) {
            IOS_AddJoystickDevice(controller);
        }

#ifdef SDL_PLATFORM_TVOS
        SDL_AddHintCallback(SDL_HINT_APPLE_TV_REMOTE_ALLOW_ROTATION,
                            SDL_AppleTVRemoteRotationHintChanged, NULL);
#endif // SDL_PLATFORM_TVOS

        center = [NSNotificationCenter defaultCenter];

        connectObserver = [center addObserverForName:GCControllerDidConnectNotification
                                              object:nil
                                               queue:nil
                                          usingBlock:^(NSNotification *note) {
                                            GCController *controller = note.object;
                                            SDL_LockJoysticks();
                                            IOS_AddJoystickDevice(controller);
                                            SDL_UnlockJoysticks();
                                          }];

        disconnectObserver = [center addObserverForName:GCControllerDidDisconnectNotification
                                                 object:nil
                                                  queue:nil
                                             usingBlock:^(NSNotification *note) {
                                               GCController *controller = note.object;
                                               SDL_JoystickDeviceItem *device;
                                               SDL_LockJoysticks();
                                               for (device = deviceList; device != NULL; device = device->next) {
                                                   if (device->controller == controller) {
                                                       IOS_RemoveJoystickDevice(device);
                                                       break;
                                                   }
                                               }
                                               SDL_UnlockJoysticks();
                                             }];
#endif // SDL_JOYSTICK_MFI
    }

    return true;
}

static int IOS_JoystickGetCount(void)
{
    return numjoysticks;
}

static void IOS_JoystickDetect(void)
{
}

static bool IOS_JoystickIsDevicePresent(Uint16 vendor_id, Uint16 product_id, Uint16 version, const char *name)
{
    // We don't override any other drivers through this method
    return false;
}

static const char *IOS_JoystickGetDeviceName(int device_index)
{
    SDL_JoystickDeviceItem *device = GetDeviceForIndex(device_index);
    return device ? device->name : "Unknown";
}

static const char *IOS_JoystickGetDevicePath(int device_index)
{
    return NULL;
}

static int IOS_JoystickGetDeviceSteamVirtualGamepadSlot(int device_index)
{
    return -1;
}

static int IOS_JoystickGetDevicePlayerIndex(int device_index)
{
#ifdef SDL_JOYSTICK_MFI
    SDL_JoystickDeviceItem *device = GetDeviceForIndex(device_index);
    if (device && device->controller) {
        return (int)device->controller.playerIndex;
    }
#endif
    return -1;
}

static void IOS_JoystickSetDevicePlayerIndex(int device_index, int player_index)
{
#ifdef SDL_JOYSTICK_MFI
    SDL_JoystickDeviceItem *device = GetDeviceForIndex(device_index);
    if (device && device->controller) {
        device->controller.playerIndex = player_index;
    }
#endif
}

static SDL_GUID IOS_JoystickGetDeviceGUID(int device_index)
{
    SDL_JoystickDeviceItem *device = GetDeviceForIndex(device_index);
    SDL_GUID guid;
    if (device) {
        guid = device->guid;
    } else {
        SDL_zero(guid);
    }
    return guid;
}

static SDL_JoystickID IOS_JoystickGetDeviceInstanceID(int device_index)
{
    SDL_JoystickDeviceItem *device = GetDeviceForIndex(device_index);
    return device ? device->instance_id : 0;
}

static bool IOS_JoystickOpen(SDL_Joystick *joystick, int device_index)
{
    SDL_JoystickDeviceItem *device = GetDeviceForIndex(device_index);
    if (device == NULL) {
        return SDL_SetError("Could not open Joystick: no hardware device for the specified index");
    }

    joystick->hwdata = device;

    joystick->naxes = device->naxes;
    joystick->nhats = device->nhats;
    joystick->nbuttons = device->nbuttons;

    if (device->has_dualshock_touchpad) {
        SDL_PrivateJoystickAddTouchpad(joystick, 2);
    }

    device->joystick = joystick;

    @autoreleasepool {
#ifdef SDL_JOYSTICK_MFI
        if (device->pause_button_index >= 0) {
            GCController *controller = device->controller;
            controller.controllerPausedHandler = ^(GCController *c) {
              if (joystick->hwdata) {
                  joystick->hwdata->pause_button_pressed = SDL_GetTicks();
              }
            };
        }

        if (@available(macOS 10.16, iOS 14.0, tvOS 14.0, *)) {
            GCController *controller = joystick->hwdata->controller;
            GCMotion *motion = controller.motion;
            if (motion && motion.hasRotationRate) {
                SDL_PrivateJoystickAddSensor(joystick, SDL_SENSOR_GYRO, 0.0f);
            }
            if (motion && motion.hasGravityAndUserAcceleration) {
                SDL_PrivateJoystickAddSensor(joystick, SDL_SENSOR_ACCEL, 0.0f);
            }
        }

        if (@available(macOS 10.16, iOS 14.0, tvOS 14.0, *)) {
            GCController *controller = joystick->hwdata->controller;
            for (id key in controller.physicalInputProfile.buttons) {
                GCControllerButtonInput *button = controller.physicalInputProfile.buttons[key];
                if ([button isBoundToSystemGesture]) {
                    button.preferredSystemGestureState = GCSystemGestureStateDisabled;
                }
            }
        }

        if (@available(macOS 10.16, iOS 14.0, tvOS 14.0, *)) {
            GCController *controller = device->controller;
            if (controller.light) {
                SDL_SetBooleanProperty(SDL_GetJoystickProperties(joystick), SDL_PROP_JOYSTICK_CAP_RGB_LED_BOOLEAN, true);
            }

            if (controller.haptics) {
                for (GCHapticsLocality locality in controller.haptics.supportedLocalities) {
                    if ([locality isEqualToString:GCHapticsLocalityHandles]) {
                        SDL_SetBooleanProperty(SDL_GetJoystickProperties(joystick), SDL_PROP_JOYSTICK_CAP_RUMBLE_BOOLEAN, true);
                    } else if ([locality isEqualToString:GCHapticsLocalityTriggers]) {
                        SDL_SetBooleanProperty(SDL_GetJoystickProperties(joystick), SDL_PROP_JOYSTICK_CAP_TRIGGER_RUMBLE_BOOLEAN, true);
                    }
                }
            }
        }
#endif // SDL_JOYSTICK_MFI
    }
    if (device->is_siri_remote) {
        ++SDL_AppleTVRemoteOpenedAsJoystick;
    }

    return true;
}

#ifdef SDL_JOYSTICK_MFI
static Uint8 IOS_MFIJoystickHatStateForDPad(GCControllerDirectionPad *dpad)
{
    Uint8 hat = 0;

    if (dpad.up.isPressed) {
        hat |= SDL_HAT_UP;
    } else if (dpad.down.isPressed) {
        hat |= SDL_HAT_DOWN;
    }

    if (dpad.left.isPressed) {
        hat |= SDL_HAT_LEFT;
    } else if (dpad.right.isPressed) {
        hat |= SDL_HAT_RIGHT;
    }

    if (hat == 0) {
        return SDL_HAT_CENTERED;
    }

    return hat;
}
#endif

static void IOS_MFIJoystickUpdate(SDL_Joystick *joystick)
{
#ifdef SDL_JOYSTICK_MFI
    @autoreleasepool {
        SDL_JoystickDeviceItem *device = joystick->hwdata;
        GCController *controller = device->controller;
        Uint8 hatstate = SDL_HAT_CENTERED;
        int i;
        Uint64 timestamp = SDL_GetTicksNS();

#ifdef DEBUG_CONTROLLER_STATE
        if (@available(macOS 10.16, iOS 14.0, tvOS 14.0, *)) {
            if (controller.physicalInputProfile) {
                for (id key in controller.physicalInputProfile.buttons) {
                    GCControllerButtonInput *button = controller.physicalInputProfile.buttons[key];
                    if (button.isPressed)
                        NSLog(@"Button %@ = %s\n", key, button.isPressed ? "pressed" : "released");
                }
                for (id key in controller.physicalInputProfile.axes) {
                    GCControllerAxisInput *axis = controller.physicalInputProfile.axes[key];
                    if (axis.value != 0.0f)
                        NSLog(@"Axis %@ = %g\n", key, axis.value);
                }
                for (id key in controller.physicalInputProfile.dpads) {
                    GCControllerDirectionPad *dpad = controller.physicalInputProfile.dpads[key];
                    if (dpad.up.isPressed || dpad.down.isPressed || dpad.left.isPressed || dpad.right.isPressed) {
                        NSLog(@"Hat %@ =%s%s%s%s\n", key,
                            dpad.up.isPressed ? " UP" : "",
                            dpad.down.isPressed ? " DOWN" : "",
                            dpad.left.isPressed ? " LEFT" : "",
                            dpad.right.isPressed ? " RIGHT" : "");
                    }
                }
            }
        }
#endif // DEBUG_CONTROLLER_STATE

        if (@available(macOS 10.16, iOS 14.0, tvOS 14.0, *)) {
            NSDictionary<NSString *, GCControllerElement *> *elements = controller.physicalInputProfile.elements;
            NSDictionary<NSString *, GCControllerButtonInput *> *buttons = controller.physicalInputProfile.buttons;

            int axis = 0;
            for (id key in device->axes) {
                Sint16 value;
                GCControllerElement *element = elements[key];
                if ([element isKindOfClass:[GCControllerAxisInput class]]) {
                    value = (Sint16)([(GCControllerAxisInput *)element value] * 32767);
                } else {
                    value = (Sint16)([(GCControllerButtonInput *)element value] * 32767);
                }
                SDL_SendJoystickAxis(timestamp, joystick, axis++, value);
            }

            int button = 0;
            for (id key in device->buttons) {
                bool down;
                if (button == device->pause_button_index) {
                    down = (device->pause_button_pressed > 0);
                } else {
                    down = buttons[key].isPressed;
                }
                SDL_SendJoystickButton(timestamp, joystick, button++, down);
            }
        } else if (controller.extendedGamepad) {
            bool isstack;
            GCExtendedGamepad *gamepad = controller.extendedGamepad;

            // Axis order matches the XInput Windows mappings.
            Sint16 axes[] = {
                (Sint16)(gamepad.leftThumbstick.xAxis.value * 32767),
                (Sint16)(gamepad.leftThumbstick.yAxis.value * -32767),
                (Sint16)((gamepad.leftTrigger.value * 65535) - 32768),
                (Sint16)(gamepad.rightThumbstick.xAxis.value * 32767),
                (Sint16)(gamepad.rightThumbstick.yAxis.value * -32767),
                (Sint16)((gamepad.rightTrigger.value * 65535) - 32768),
            };

            // Button order matches the XInput Windows mappings.
            bool *buttons = SDL_small_alloc(bool, joystick->nbuttons, &isstack);
            int button_count = 0;

            if (buttons == NULL) {
                return;
            }

            // These buttons are part of the original MFi spec
            buttons[button_count++] = gamepad.buttonA.isPressed;
            buttons[button_count++] = gamepad.buttonB.isPressed;
            buttons[button_count++] = gamepad.buttonX.isPressed;
            buttons[button_count++] = gamepad.buttonY.isPressed;
            buttons[button_count++] = gamepad.leftShoulder.isPressed;
            buttons[button_count++] = gamepad.rightShoulder.isPressed;

            // These buttons are available on some newer controllers
            if (@available(macOS 10.14.1, iOS 12.1, tvOS 12.1, *)) {
                if (device->button_mask & (1 << SDL_GAMEPAD_BUTTON_LEFT_STICK)) {
                    buttons[button_count++] = gamepad.leftThumbstickButton.isPressed;
                }
                if (device->button_mask & (1 << SDL_GAMEPAD_BUTTON_RIGHT_STICK)) {
                    buttons[button_count++] = gamepad.rightThumbstickButton.isPressed;
                }
            }
            if (@available(macOS 10.15, iOS 13.0, tvOS 13.0, *)) {
                if (device->button_mask & (1 << SDL_GAMEPAD_BUTTON_BACK)) {
                    buttons[button_count++] = gamepad.buttonOptions.isPressed;
                }
            }
            if (device->button_mask & (1 << SDL_GAMEPAD_BUTTON_START)) {
                if (device->pause_button_index >= 0) {
                    // Guaranteed if buttonMenu is not supported on this OS
                    buttons[button_count++] = (device->pause_button_pressed > 0);
                } else {
                    if (@available(macOS 10.15, iOS 13.0, tvOS 13.0, *)) {
                        buttons[button_count++] = gamepad.buttonMenu.isPressed;
                    }
                }
            }

            hatstate = IOS_MFIJoystickHatStateForDPad(gamepad.dpad);

            for (i = 0; i < SDL_arraysize(axes); i++) {
                SDL_SendJoystickAxis(timestamp, joystick, i, axes[i]);
            }

            for (i = 0; i < button_count; i++) {
                SDL_SendJoystickButton(timestamp, joystick, i, buttons[i]);
            }

            SDL_small_free(buttons, isstack);
        }
#ifdef SDL_PLATFORM_TVOS
        else if (controller.microGamepad) {
            GCMicroGamepad *gamepad = controller.microGamepad;

            Sint16 axes[] = {
                (Sint16)(gamepad.dpad.xAxis.value * 32767),
                (Sint16)(gamepad.dpad.yAxis.value * -32767),
            };

            for (i = 0; i < SDL_arraysize(axes); i++) {
                SDL_SendJoystickAxis(timestamp, joystick, i, axes[i]);
            }

            bool buttons[joystick->nbuttons];
            int button_count = 0;
            buttons[button_count++] = gamepad.buttonA.isPressed;
            buttons[button_count++] = gamepad.buttonX.isPressed;
            buttons[button_count++] = (device->pause_button_pressed > 0);

            for (i = 0; i < button_count; i++) {
                SDL_SendJoystickButton(timestamp, joystick, i, buttons[i]);
            }
        }
#endif // SDL_PLATFORM_TVOS

        if (joystick->nhats > 0) {
            SDL_SendJoystickHat(timestamp, joystick, 0, hatstate);
        }

        if (device->pause_button_pressed) {
            // The pause callback is instantaneous, so we extend the duration to allow "holding down" by pressing it repeatedly
            const int PAUSE_BUTTON_PRESS_DURATION_MS = 250;
            if (SDL_GetTicks() >= device->pause_button_pressed + PAUSE_BUTTON_PRESS_DURATION_MS) {
                device->pause_button_pressed = 0;
            }
        }

        if (@available(macOS 10.16, iOS 14.0, tvOS 14.0, *)) {
            if (device->has_dualshock_touchpad) {
                GCControllerDirectionPad *dpad;

                dpad = controller.physicalInputProfile.dpads[GCInputDualShockTouchpadOne];
                if (dpad.xAxis.value != 0.f || dpad.yAxis.value != 0.f) {
                    SDL_SendJoystickTouchpad(timestamp, joystick, 0, 0, true, (1.0f + dpad.xAxis.value) * 0.5f, 1.0f - (1.0f + dpad.yAxis.value) * 0.5f, 1.0f);
                } else {
                    SDL_SendJoystickTouchpad(timestamp, joystick, 0, 0, false, 0.0f, 0.0f, 1.0f);
                }

                dpad = controller.physicalInputProfile.dpads[GCInputDualShockTouchpadTwo];
                if (dpad.xAxis.value != 0.f || dpad.yAxis.value != 0.f) {
                    SDL_SendJoystickTouchpad(timestamp, joystick, 0, 1, true, (1.0f + dpad.xAxis.value) * 0.5f, 1.0f - (1.0f + dpad.yAxis.value) * 0.5f, 1.0f);
                } else {
                    SDL_SendJoystickTouchpad(timestamp, joystick, 0, 1, false, 0.0f, 0.0f, 1.0f);
                }
            }
        }

        if (@available(macOS 10.16, iOS 14.0, tvOS 14.0, *)) {
            GCMotion *motion = controller.motion;
            if (motion && motion.sensorsActive) {
                float data[3];

                if (motion.hasRotationRate) {
                    GCRotationRate rate = motion.rotationRate;
                    data[0] = rate.x;
                    data[1] = rate.z;
                    data[2] = -rate.y;
                    SDL_SendJoystickSensor(timestamp, joystick, SDL_SENSOR_GYRO, timestamp, data, 3);
                }
                if (motion.hasGravityAndUserAcceleration) {
                    GCAcceleration accel = motion.acceleration;
                    data[0] = -accel.x * SDL_STANDARD_GRAVITY;
                    data[1] = -accel.y * SDL_STANDARD_GRAVITY;
                    data[2] = -accel.z * SDL_STANDARD_GRAVITY;
                    SDL_SendJoystickSensor(timestamp, joystick, SDL_SENSOR_ACCEL, timestamp, data, 3);
                }
            }
        }

        if (@available(macOS 10.16, iOS 14.0, tvOS 14.0, *)) {
            GCDeviceBattery *battery = controller.battery;
            if (battery) {
                SDL_PowerState state = SDL_POWERSTATE_UNKNOWN;
                int percent = (int)SDL_roundf(battery.batteryLevel * 100.0f);

                switch (battery.batteryState) {
                case GCDeviceBatteryStateDischarging:
                    state = SDL_POWERSTATE_ON_BATTERY;
                    break;
                case GCDeviceBatteryStateCharging:
                    state = SDL_POWERSTATE_CHARGING;
                    break;
                case GCDeviceBatteryStateFull:
                    state = SDL_POWERSTATE_CHARGED;
                    break;
                default:
                    break;
                }

                SDL_SendJoystickPowerInfo(joystick, state, percent);
            }
        }
    }
#endif // SDL_JOYSTICK_MFI
}

#ifdef SDL_JOYSTICK_MFI
@interface SDL3_RumbleMotor : NSObject
@property(nonatomic, strong) CHHapticEngine *engine API_AVAILABLE(macos(10.16), ios(13.0), tvos(14.0));
@property(nonatomic, strong) id<CHHapticPatternPlayer> player API_AVAILABLE(macos(10.16), ios(13.0), tvos(14.0));
@property bool active;
@end

@implementation SDL3_RumbleMotor
{
}

- (void)cleanup
{
    @autoreleasepool {
        if (@available(macOS 10.16, iOS 14.0, tvOS 14.0, *)) {
            if (self.player != nil) {
                [self.player cancelAndReturnError:nil];
                self.player = nil;
            }
            if (self.engine != nil) {
                [self.engine stopWithCompletionHandler:nil];
                self.engine = nil;
            }
        }
    }
}

- (bool)setIntensity:(float)intensity
{
    @autoreleasepool {
        if (@available(macOS 10.16, iOS 14.0, tvOS 14.0, *)) {
            NSError *error = nil;
            CHHapticDynamicParameter *param;

            if (self.engine == nil) {
                return SDL_SetError("Haptics engine was stopped");
            }

            if (intensity == 0.0f) {
                if (self.player && self.active) {
                    [self.player stopAtTime:0 error:&error];
                }
                self.active = false;
                return true;
            }

            if (self.player == nil) {
                CHHapticEventParameter *event_param = [[CHHapticEventParameter alloc] initWithParameterID:CHHapticEventParameterIDHapticIntensity value:1.0f];
                CHHapticEvent *event = [[CHHapticEvent alloc] initWithEventType:CHHapticEventTypeHapticContinuous parameters:[NSArray arrayWithObjects:event_param, nil] relativeTime:0 duration:GCHapticDurationInfinite];
                CHHapticPattern *pattern = [[CHHapticPattern alloc] initWithEvents:[NSArray arrayWithObject:event] parameters:[[NSArray alloc] init] error:&error];
                if (error != nil) {
                    return SDL_SetError("Couldn't create haptic pattern: %s", [error.localizedDescription UTF8String]);
                }

                self.player = [self.engine createPlayerWithPattern:pattern error:&error];
                if (error != nil) {
                    return SDL_SetError("Couldn't create haptic player: %s", [error.localizedDescription UTF8String]);
                }
                self.active = false;
            }

            param = [[CHHapticDynamicParameter alloc] initWithParameterID:CHHapticDynamicParameterIDHapticIntensityControl value:intensity relativeTime:0];
            [self.player sendParameters:[NSArray arrayWithObject:param] atTime:0 error:&error];
            if (error != nil) {
                return SDL_SetError("Couldn't update haptic player: %s", [error.localizedDescription UTF8String]);
            }

            if (!self.active) {
                [self.player startAtTime:0 error:&error];
                self.active = true;
            }
        }

        return true;
    }
}

- (id)initWithController:(GCController *)controller locality:(GCHapticsLocality)locality API_AVAILABLE(macos(10.16), ios(14.0), tvos(14.0))
{
    @autoreleasepool {
        NSError *error;
        __weak __typeof(self) weakSelf;
        self = [super init];
        weakSelf = self;

        self.engine = [controller.haptics createEngineWithLocality:locality];
        if (self.engine == nil) {
            SDL_SetError("Couldn't create haptics engine");
            return nil;
        }

        [self.engine startAndReturnError:&error];
        if (error != nil) {
            SDL_SetError("Couldn't start haptics engine");
            return nil;
        }

        self.engine.stoppedHandler = ^(CHHapticEngineStoppedReason stoppedReason) {
          SDL3_RumbleMotor *_this = weakSelf;
          if (_this == nil) {
              return;
          }

          _this.player = nil;
          _this.engine = nil;
        };
        self.engine.resetHandler = ^{
          SDL3_RumbleMotor *_this = weakSelf;
          if (_this == nil) {
              return;
          }

          _this.player = nil;
          [_this.engine startAndReturnError:nil];
        };

        return self;
    }
}

@end

@interface SDL3_RumbleContext : NSObject
@property(nonatomic, strong) SDL3_RumbleMotor *lowFrequencyMotor;
@property(nonatomic, strong) SDL3_RumbleMotor *highFrequencyMotor;
@property(nonatomic, strong) SDL3_RumbleMotor *leftTriggerMotor;
@property(nonatomic, strong) SDL3_RumbleMotor *rightTriggerMotor;
@end

@implementation SDL3_RumbleContext
{
}

- (id)initWithLowFrequencyMotor:(SDL3_RumbleMotor *)low_frequency_motor
             HighFrequencyMotor:(SDL3_RumbleMotor *)high_frequency_motor
               LeftTriggerMotor:(SDL3_RumbleMotor *)left_trigger_motor
              RightTriggerMotor:(SDL3_RumbleMotor *)right_trigger_motor
{
    self = [super init];
    self.lowFrequencyMotor = low_frequency_motor;
    self.highFrequencyMotor = high_frequency_motor;
    self.leftTriggerMotor = left_trigger_motor;
    self.rightTriggerMotor = right_trigger_motor;
    return self;
}

- (bool)rumbleWithLowFrequency:(Uint16)low_frequency_rumble andHighFrequency:(Uint16)high_frequency_rumble
{
    bool result = true;

    result &= [self.lowFrequencyMotor setIntensity:((float)low_frequency_rumble / 65535.0f)];
    result &= [self.highFrequencyMotor setIntensity:((float)high_frequency_rumble / 65535.0f)];
    return result;
}

- (bool)rumbleLeftTrigger:(Uint16)left_rumble andRightTrigger:(Uint16)right_rumble
{
    bool result = false;

    if (self.leftTriggerMotor && self.rightTriggerMotor) {
        result &= [self.leftTriggerMotor setIntensity:((float)left_rumble / 65535.0f)];
        result &= [self.rightTriggerMotor setIntensity:((float)right_rumble / 65535.0f)];
    } else {
        result = SDL_Unsupported();
    }
    return result;
}

- (void)cleanup
{
    [self.lowFrequencyMotor cleanup];
    [self.highFrequencyMotor cleanup];
}

@end

static SDL3_RumbleContext *IOS_JoystickInitRumble(GCController *controller)
{
    @autoreleasepool {
        if (@available(macOS 10.16, iOS 14.0, tvOS 14.0, *)) {
            SDL3_RumbleMotor *low_frequency_motor = [[SDL3_RumbleMotor alloc] initWithController:controller locality:GCHapticsLocalityLeftHandle];
            SDL3_RumbleMotor *high_frequency_motor = [[SDL3_RumbleMotor alloc] initWithController:controller locality:GCHapticsLocalityRightHandle];
            SDL3_RumbleMotor *left_trigger_motor = [[SDL3_RumbleMotor alloc] initWithController:controller locality:GCHapticsLocalityLeftTrigger];
            SDL3_RumbleMotor *right_trigger_motor = [[SDL3_RumbleMotor alloc] initWithController:controller locality:GCHapticsLocalityRightTrigger];
            if (low_frequency_motor && high_frequency_motor) {
                return [[SDL3_RumbleContext alloc] initWithLowFrequencyMotor:low_frequency_motor
                                                         HighFrequencyMotor:high_frequency_motor
                                                           LeftTriggerMotor:left_trigger_motor
                                                          RightTriggerMotor:right_trigger_motor];
            }
        }
    }
    return nil;
}

#endif // SDL_JOYSTICK_MFI

static bool IOS_JoystickRumble(SDL_Joystick *joystick, Uint16 low_frequency_rumble, Uint16 high_frequency_rumble)
{
#ifdef SDL_JOYSTICK_MFI
    SDL_JoystickDeviceItem *device = joystick->hwdata;

    if (device == NULL) {
        return SDL_SetError("Controller is no longer connected");
    }

    if (@available(macOS 10.16, iOS 14.0, tvOS 14.0, *)) {
        if (!device->rumble && device->controller && device->controller.haptics) {
            SDL3_RumbleContext *rumble = IOS_JoystickInitRumble(device->controller);
            if (rumble) {
                device->rumble = (void *)CFBridgingRetain(rumble);
            }
        }
    }

    if (device->rumble) {
        SDL3_RumbleContext *rumble = (__bridge SDL3_RumbleContext *)device->rumble;
        return [rumble rumbleWithLowFrequency:low_frequency_rumble andHighFrequency:high_frequency_rumble];
    }
#endif
    return SDL_Unsupported();
}

static bool IOS_JoystickRumbleTriggers(SDL_Joystick *joystick, Uint16 left_rumble, Uint16 right_rumble)
{
#ifdef SDL_JOYSTICK_MFI
    SDL_JoystickDeviceItem *device = joystick->hwdata;

    if (device == NULL) {
        return SDL_SetError("Controller is no longer connected");
    }

    if (@available(macOS 10.16, iOS 14.0, tvOS 14.0, *)) {
        if (!device->rumble && device->controller && device->controller.haptics) {
            SDL3_RumbleContext *rumble = IOS_JoystickInitRumble(device->controller);
            if (rumble) {
                device->rumble = (void *)CFBridgingRetain(rumble);
            }
        }
    }

    if (device->rumble) {
        SDL3_RumbleContext *rumble = (__bridge SDL3_RumbleContext *)device->rumble;
        return [rumble rumbleLeftTrigger:left_rumble andRightTrigger:right_rumble];
    }
#endif
    return SDL_Unsupported();
}

static bool IOS_JoystickSetLED(SDL_Joystick *joystick, Uint8 red, Uint8 green, Uint8 blue)
{
    @autoreleasepool {
        SDL_JoystickDeviceItem *device = joystick->hwdata;

        if (device == NULL) {
            return SDL_SetError("Controller is no longer connected");
        }

        if (@available(macOS 10.16, iOS 14.0, tvOS 14.0, *)) {
            GCController *controller = device->controller;
            GCDeviceLight *light = controller.light;
            if (light) {
                light.color = [[GCColor alloc] initWithRed:(float)red / 255.0f
                                                     green:(float)green / 255.0f
                                                      blue:(float)blue / 255.0f];
                return true;
            }
        }
    }
    return SDL_Unsupported();
}

static bool IOS_JoystickSendEffect(SDL_Joystick *joystick, const void *data, int size)
{
    return SDL_Unsupported();
}

static bool IOS_JoystickSetSensorsEnabled(SDL_Joystick *joystick, bool enabled)
{
    @autoreleasepool {
        SDL_JoystickDeviceItem *device = joystick->hwdata;

        if (device == NULL) {
            return SDL_SetError("Controller is no longer connected");
        }

        if (@available(macOS 10.16, iOS 14.0, tvOS 14.0, *)) {
            GCController *controller = device->controller;
            GCMotion *motion = controller.motion;
            if (motion) {
                motion.sensorsActive = enabled ? YES : NO;
                return true;
            }
        }
    }

    return SDL_Unsupported();
}

static void IOS_JoystickUpdate(SDL_Joystick *joystick)
{
    SDL_JoystickDeviceItem *device = joystick->hwdata;

    if (device == NULL) {
        return;
    }

    if (device->controller) {
        IOS_MFIJoystickUpdate(joystick);
    }
}

static void IOS_JoystickClose(SDL_Joystick *joystick)
{
    SDL_JoystickDeviceItem *device = joystick->hwdata;

    if (device == NULL) {
        return;
    }

    device->joystick = NULL;

#ifdef SDL_JOYSTICK_MFI
    @autoreleasepool {
        if (device->rumble) {
            SDL3_RumbleContext *rumble = (__bridge SDL3_RumbleContext *)device->rumble;

            [rumble cleanup];
            CFRelease(device->rumble);
            device->rumble = NULL;
        }

        if (device->controller) {
            GCController *controller = device->controller;
            controller.controllerPausedHandler = nil;
            controller.playerIndex = -1;

            if (@available(macOS 10.16, iOS 14.0, tvOS 14.0, *)) {
                for (id key in controller.physicalInputProfile.buttons) {
                    GCControllerButtonInput *button = controller.physicalInputProfile.buttons[key];
                    if ([button isBoundToSystemGesture]) {
                        button.preferredSystemGestureState = GCSystemGestureStateEnabled;
                    }
                }
            }
        }
    }
#endif // SDL_JOYSTICK_MFI

    if (device->is_siri_remote) {
        --SDL_AppleTVRemoteOpenedAsJoystick;
    }
}

static void IOS_JoystickQuit(void)
{
    @autoreleasepool {
#ifdef SDL_JOYSTICK_MFI
        NSNotificationCenter *center = [NSNotificationCenter defaultCenter];

        if (connectObserver) {
            [center removeObserver:connectObserver name:GCControllerDidConnectNotification object:nil];
            connectObserver = nil;
        }

        if (disconnectObserver) {
            [center removeObserver:disconnectObserver name:GCControllerDidDisconnectNotification object:nil];
            disconnectObserver = nil;
        }

#ifdef SDL_PLATFORM_TVOS
        SDL_RemoveHintCallback(SDL_HINT_APPLE_TV_REMOTE_ALLOW_ROTATION,
                            SDL_AppleTVRemoteRotationHintChanged, NULL);
#endif // SDL_PLATFORM_TVOS
#endif // SDL_JOYSTICK_MFI

        while (deviceList != NULL) {
            IOS_RemoveJoystickDevice(deviceList);
        }
    }

    numjoysticks = 0;
}

static bool IOS_JoystickGetGamepadMapping(int device_index, SDL_GamepadMapping *out)
{
    SDL_JoystickDeviceItem *device = GetDeviceForIndex(device_index);
    if (device == NULL) {
        return false;
    }

    if (@available(macOS 10.16, iOS 14.0, tvOS 14.0, *)) {
        int axis = 0;
        for (id key in device->axes) {
            if ([(NSString *)key isEqualToString:@"Left Thumbstick X Axis"] ||
                [(NSString *)key isEqualToString:@"Direction Pad X Axis"]) {
                out->leftx.kind = EMappingKind_Axis;
                out->leftx.target = axis;
            } else if ([(NSString *)key isEqualToString:@"Left Thumbstick Y Axis"] ||
                       [(NSString *)key isEqualToString:@"Direction Pad Y Axis"]) {
                out->lefty.kind = EMappingKind_Axis;
                out->lefty.target = axis;
                out->lefty.axis_reversed = true;
            } else if ([(NSString *)key isEqualToString:@"Right Thumbstick X Axis"]) {
                out->rightx.kind = EMappingKind_Axis;
                out->rightx.target = axis;
            } else if ([(NSString *)key isEqualToString:@"Right Thumbstick Y Axis"]) {
                out->righty.kind = EMappingKind_Axis;
                out->righty.target = axis;
                out->righty.axis_reversed = true;
            } else if ([(NSString *)key isEqualToString:GCInputLeftTrigger]) {
                out->lefttrigger.kind = EMappingKind_Axis;
                out->lefttrigger.target = axis;
                out->lefttrigger.half_axis_positive = true;
            } else if ([(NSString *)key isEqualToString:GCInputRightTrigger]) {
                out->righttrigger.kind = EMappingKind_Axis;
                out->righttrigger.target = axis;
                out->righttrigger.half_axis_positive = true;
            }
            ++axis;
        }

        int button = 0;
        for (id key in device->buttons) {
            SDL_InputMapping *mapping = NULL;

            if ([(NSString *)key isEqualToString:GCInputButtonA]) {
                if (device->is_siri_remote > 1) {
                    // GCInputButtonA is triggered for any D-Pad press, ignore it in favor of "Button Center"
                } else if (device->has_nintendo_buttons) {
                    mapping = &out->b;
                } else {
                    mapping = &out->a;
                }
            } else if ([(NSString *)key isEqualToString:GCInputButtonB]) {
                if (device->has_nintendo_buttons) {
                    mapping = &out->a;
                } else if (device->is_switch_joyconL || device->is_switch_joyconR) {
                    mapping = &out->x;
                } else {
                    mapping = &out->b;
                }
            } else if ([(NSString *)key isEqualToString:GCInputButtonX]) {
                if (device->has_nintendo_buttons) {
                    mapping = &out->y;
                } else if (device->is_switch_joyconL || device->is_switch_joyconR) {
                    mapping = &out->b;
                } else {
                    mapping = &out->x;
                }
            } else if ([(NSString *)key isEqualToString:GCInputButtonY]) {
                if (device->has_nintendo_buttons) {
                    mapping = &out->x;
                } else {
                    mapping = &out->y;
                }
            } else if ([(NSString *)key isEqualToString:@"Direction Pad Left"]) {
                mapping = &out->dpleft;
            } else if ([(NSString *)key isEqualToString:@"Direction Pad Right"]) {
                mapping = &out->dpright;
            } else if ([(NSString *)key isEqualToString:@"Direction Pad Up"]) {
                mapping = &out->dpup;
            } else if ([(NSString *)key isEqualToString:@"Direction Pad Down"]) {
                mapping = &out->dpdown;
            } else if ([(NSString *)key isEqualToString:@"Cardinal Direction Pad Left"]) {
                mapping = &out->dpleft;
            } else if ([(NSString *)key isEqualToString:@"Cardinal Direction Pad Right"]) {
                mapping = &out->dpright;
            } else if ([(NSString *)key isEqualToString:@"Cardinal Direction Pad Up"]) {
                mapping = &out->dpup;
            } else if ([(NSString *)key isEqualToString:@"Cardinal Direction Pad Down"]) {
                mapping = &out->dpdown;
            } else if ([(NSString *)key isEqualToString:GCInputLeftShoulder]) {
                mapping = &out->leftshoulder;
            } else if ([(NSString *)key isEqualToString:GCInputRightShoulder]) {
                mapping = &out->rightshoulder;
            } else if ([(NSString *)key isEqualToString:GCInputLeftThumbstickButton]) {
                mapping = &out->leftstick;
            } else if ([(NSString *)key isEqualToString:GCInputRightThumbstickButton]) {
                mapping = &out->rightstick;
            } else if ([(NSString *)key isEqualToString:@"Button Home"]) {
                mapping = &out->guide;
            } else if ([(NSString *)key isEqualToString:GCInputButtonMenu]) {
                if (device->is_siri_remote) {
                    mapping = &out->b;
                } else {
                    mapping = &out->start;
                }
            } else if ([(NSString *)key isEqualToString:GCInputButtonOptions]) {
                mapping = &out->back;
            } else if ([(NSString *)key isEqualToString:@"Button Share"]) {
                mapping = &out->misc1;
            } else if ([(NSString *)key isEqualToString:GCInputXboxPaddleOne]) {
                mapping = &out->right_paddle1;
            } else if ([(NSString *)key isEqualToString:GCInputXboxPaddleTwo]) {
                mapping = &out->right_paddle2;
            } else if ([(NSString *)key isEqualToString:GCInputXboxPaddleThree]) {
                mapping = &out->left_paddle1;
            } else if ([(NSString *)key isEqualToString:GCInputXboxPaddleFour]) {
                mapping = &out->left_paddle2;
            } else if ([(NSString *)key isEqualToString:GCInputLeftTrigger]) {
                mapping = &out->lefttrigger;
            } else if ([(NSString *)key isEqualToString:GCInputRightTrigger]) {
                mapping = &out->righttrigger;
            } else if ([(NSString *)key isEqualToString:GCInputDualShockTouchpadButton]) {
                mapping = &out->touchpad;
            } else if ([(NSString *)key isEqualToString:@"Button Center"]) {
                mapping = &out->a;
            }
            if (mapping && mapping->kind == EMappingKind_None) {
                mapping->kind = EMappingKind_Button;
                mapping->target = button;
            }
            ++button;
        }

        return true;
    }
    return false;
}

#if defined(SDL_JOYSTICK_MFI) && defined(SDL_PLATFORM_MACOS)
bool IOS_SupportedHIDDevice(IOHIDDeviceRef device)
{
    if (!SDL_GetHintBoolean(SDL_HINT_JOYSTICK_MFI, true)) {
        return false;
    }

    if (@available(macOS 10.16, *)) {
        const int MAX_ATTEMPTS = 3;
        for (int attempt = 0; attempt < MAX_ATTEMPTS; ++attempt) {
            if ([GCController supportsHIDDevice:device]) {
                return true;
            }

            // The framework may not have seen the device yet
            SDL_Delay(10);
        }
    }
    return false;
}
#endif

#ifdef SDL_JOYSTICK_MFI
/* NOLINTNEXTLINE(readability-non-const-parameter): getCString takes a non-const char* */
static void GetAppleSFSymbolsNameForElement(GCControllerElement *element, char *name)
{
    if (@available(macOS 10.16, iOS 14.0, tvOS 14.0, *)) {
        if (element) {
            [element.sfSymbolsName getCString:name maxLength:255 encoding:NSASCIIStringEncoding];
        }
    }
}

static GCControllerDirectionPad *GetDirectionalPadForController(GCController *controller)
{
    if (@available(macOS 10.16, iOS 14.0, tvOS 14.0, *)) {
        return controller.physicalInputProfile.dpads[GCInputDirectionPad];
    }

    if (controller.extendedGamepad) {
        return controller.extendedGamepad.dpad;
    }

    if (controller.microGamepad) {
        return controller.microGamepad.dpad;
    }

    return nil;
}
#endif // SDL_JOYSTICK_MFI

const char *IOS_GetAppleSFSymbolsNameForButton(SDL_Gamepad *gamepad, SDL_GamepadButton button)
{
    char elementName[256];
    elementName[0] = '\0';

#ifdef SDL_JOYSTICK_MFI
    if (gamepad && SDL_GetGamepadJoystick(gamepad)->driver == &SDL_IOS_JoystickDriver) {
        if (@available(macOS 10.16, iOS 14.0, tvOS 14.0, *)) {
            GCController *controller = SDL_GetGamepadJoystick(gamepad)->hwdata->controller;
            NSDictionary<NSString *, GCControllerElement *> *elements = controller.physicalInputProfile.elements;
            switch (button) {
            case SDL_GAMEPAD_BUTTON_SOUTH:
                GetAppleSFSymbolsNameForElement(elements[GCInputButtonA], elementName);
                break;
            case SDL_GAMEPAD_BUTTON_EAST:
                GetAppleSFSymbolsNameForElement(elements[GCInputButtonB], elementName);
                break;
            case SDL_GAMEPAD_BUTTON_WEST:
                GetAppleSFSymbolsNameForElement(elements[GCInputButtonX], elementName);
                break;
            case SDL_GAMEPAD_BUTTON_NORTH:
                GetAppleSFSymbolsNameForElement(elements[GCInputButtonY], elementName);
                break;
            case SDL_GAMEPAD_BUTTON_BACK:
                GetAppleSFSymbolsNameForElement(elements[GCInputButtonOptions], elementName);
                break;
            case SDL_GAMEPAD_BUTTON_GUIDE:
                GetAppleSFSymbolsNameForElement(elements[@"Button Home"], elementName);
                break;
            case SDL_GAMEPAD_BUTTON_START:
                GetAppleSFSymbolsNameForElement(elements[GCInputButtonMenu], elementName);
                break;
            case SDL_GAMEPAD_BUTTON_LEFT_STICK:
                GetAppleSFSymbolsNameForElement(elements[GCInputLeftThumbstickButton], elementName);
                break;
            case SDL_GAMEPAD_BUTTON_RIGHT_STICK:
                GetAppleSFSymbolsNameForElement(elements[GCInputRightThumbstickButton], elementName);
                break;
            case SDL_GAMEPAD_BUTTON_LEFT_SHOULDER:
                GetAppleSFSymbolsNameForElement(elements[GCInputLeftShoulder], elementName);
                break;
            case SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER:
                GetAppleSFSymbolsNameForElement(elements[GCInputRightShoulder], elementName);
                break;
            case SDL_GAMEPAD_BUTTON_DPAD_UP:
            {
                GCControllerDirectionPad *dpad = GetDirectionalPadForController(controller);
                if (dpad) {
                    GetAppleSFSymbolsNameForElement(dpad.up, elementName);
                    if (SDL_strlen(elementName) == 0) {
                        SDL_strlcpy(elementName, "dpad.up.fill", sizeof(elementName));
                    }
                }
                break;
            }
            case SDL_GAMEPAD_BUTTON_DPAD_DOWN:
            {
                GCControllerDirectionPad *dpad = GetDirectionalPadForController(controller);
                if (dpad) {
                    GetAppleSFSymbolsNameForElement(dpad.down, elementName);
                    if (SDL_strlen(elementName) == 0) {
                        SDL_strlcpy(elementName, "dpad.down.fill", sizeof(elementName));
                    }
                }
                break;
            }
            case SDL_GAMEPAD_BUTTON_DPAD_LEFT:
            {
                GCControllerDirectionPad *dpad = GetDirectionalPadForController(controller);
                if (dpad) {
                    GetAppleSFSymbolsNameForElement(dpad.left, elementName);
                    if (SDL_strlen(elementName) == 0) {
                        SDL_strlcpy(elementName, "dpad.left.fill", sizeof(elementName));
                    }
                }
                break;
            }
            case SDL_GAMEPAD_BUTTON_DPAD_RIGHT:
            {
                GCControllerDirectionPad *dpad = GetDirectionalPadForController(controller);
                if (dpad) {
                    GetAppleSFSymbolsNameForElement(dpad.right, elementName);
                    if (SDL_strlen(elementName) == 0) {
                        SDL_strlcpy(elementName, "dpad.right.fill", sizeof(elementName));
                    }
                }
                break;
            }
            case SDL_GAMEPAD_BUTTON_MISC1:
                GetAppleSFSymbolsNameForElement(elements[GCInputDualShockTouchpadButton], elementName);
                break;
            case SDL_GAMEPAD_BUTTON_RIGHT_PADDLE1:
                GetAppleSFSymbolsNameForElement(elements[GCInputXboxPaddleOne], elementName);
                break;
            case SDL_GAMEPAD_BUTTON_LEFT_PADDLE1:
                GetAppleSFSymbolsNameForElement(elements[GCInputXboxPaddleThree], elementName);
                break;
            case SDL_GAMEPAD_BUTTON_RIGHT_PADDLE2:
                GetAppleSFSymbolsNameForElement(elements[GCInputXboxPaddleTwo], elementName);
                break;
            case SDL_GAMEPAD_BUTTON_LEFT_PADDLE2:
                GetAppleSFSymbolsNameForElement(elements[GCInputXboxPaddleFour], elementName);
                break;
            case SDL_GAMEPAD_BUTTON_TOUCHPAD:
                GetAppleSFSymbolsNameForElement(elements[GCInputDualShockTouchpadButton], elementName);
                break;
            default:
                break;
            }
        }
    }
#endif // SDL_JOYSTICK_MFI

    return *elementName ? SDL_GetPersistentString(elementName) : NULL;
}

const char *IOS_GetAppleSFSymbolsNameForAxis(SDL_Gamepad *gamepad, SDL_GamepadAxis axis)
{
    char elementName[256];
    elementName[0] = '\0';

#ifdef SDL_JOYSTICK_MFI
    if (gamepad && SDL_GetGamepadJoystick(gamepad)->driver == &SDL_IOS_JoystickDriver) {
        if (@available(macOS 10.16, iOS 14.0, tvOS 14.0, *)) {
            GCController *controller = SDL_GetGamepadJoystick(gamepad)->hwdata->controller;
            NSDictionary<NSString *, GCControllerElement *> *elements = controller.physicalInputProfile.elements;
            switch (axis) {
            case SDL_GAMEPAD_AXIS_LEFTX:
                GetAppleSFSymbolsNameForElement(elements[GCInputLeftThumbstick], elementName);
                break;
            case SDL_GAMEPAD_AXIS_LEFTY:
                GetAppleSFSymbolsNameForElement(elements[GCInputLeftThumbstick], elementName);
                break;
            case SDL_GAMEPAD_AXIS_RIGHTX:
                GetAppleSFSymbolsNameForElement(elements[GCInputRightThumbstick], elementName);
                break;
            case SDL_GAMEPAD_AXIS_RIGHTY:
                GetAppleSFSymbolsNameForElement(elements[GCInputRightThumbstick], elementName);
                break;
            case SDL_GAMEPAD_AXIS_LEFT_TRIGGER:
                GetAppleSFSymbolsNameForElement(elements[GCInputLeftTrigger], elementName);
                break;
            case SDL_GAMEPAD_AXIS_RIGHT_TRIGGER:
                GetAppleSFSymbolsNameForElement(elements[GCInputRightTrigger], elementName);
                break;
            default:
                break;
            }
        }
    }
#endif // SDL_JOYSTICK_MFI

    return *elementName ? SDL_GetPersistentString(elementName) : NULL;
}

SDL_JoystickDriver SDL_IOS_JoystickDriver = {
    IOS_JoystickInit,
    IOS_JoystickGetCount,
    IOS_JoystickDetect,
    IOS_JoystickIsDevicePresent,
    IOS_JoystickGetDeviceName,
    IOS_JoystickGetDevicePath,
    IOS_JoystickGetDeviceSteamVirtualGamepadSlot,
    IOS_JoystickGetDevicePlayerIndex,
    IOS_JoystickSetDevicePlayerIndex,
    IOS_JoystickGetDeviceGUID,
    IOS_JoystickGetDeviceInstanceID,
    IOS_JoystickOpen,
    IOS_JoystickRumble,
    IOS_JoystickRumbleTriggers,
    IOS_JoystickSetLED,
    IOS_JoystickSendEffect,
    IOS_JoystickSetSensorsEnabled,
    IOS_JoystickUpdate,
    IOS_JoystickClose,
    IOS_JoystickQuit,
    IOS_JoystickGetGamepadMapping
};
