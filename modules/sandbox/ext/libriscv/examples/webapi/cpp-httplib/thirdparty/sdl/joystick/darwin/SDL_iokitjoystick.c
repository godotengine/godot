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

#ifdef SDL_JOYSTICK_IOKIT

#include "../SDL_sysjoystick.h"
#include "../SDL_joystick_c.h"
#include "SDL_iokitjoystick_c.h"
#include "../hidapi/SDL_hidapijoystick_c.h"
#include "../../haptic/darwin/SDL_syshaptic_c.h" // For haptic hot plugging

#define SDL_JOYSTICK_RUNLOOP_MODE CFSTR("SDLJoystick")

#define CONVERT_MAGNITUDE(x) (((x)*10000) / 0x7FFF)

// The base object of the HID Manager API
static IOHIDManagerRef hidman = NULL;

// Linked list of all available devices
static recDevice *gpDeviceList = NULL;

void FreeRumbleEffectData(FFEFFECT *effect)
{
    if (!effect) {
        return;
    }
    SDL_free(effect->rgdwAxes);
    SDL_free(effect->rglDirection);
    SDL_free(effect->lpvTypeSpecificParams);
    SDL_free(effect);
}

FFEFFECT *CreateRumbleEffectData(Sint16 magnitude)
{
    FFEFFECT *effect;
    FFPERIODIC *periodic;

    // Create the effect
    effect = (FFEFFECT *)SDL_calloc(1, sizeof(*effect));
    if (!effect) {
        return NULL;
    }
    effect->dwSize = sizeof(*effect);
    effect->dwGain = 10000;
    effect->dwFlags = FFEFF_OBJECTOFFSETS;
    effect->dwDuration = SDL_MAX_RUMBLE_DURATION_MS * 1000; // In microseconds.
    effect->dwTriggerButton = FFEB_NOTRIGGER;

    effect->cAxes = 2;
    effect->rgdwAxes = (DWORD *)SDL_calloc(effect->cAxes, sizeof(DWORD));
    if (!effect->rgdwAxes) {
        FreeRumbleEffectData(effect);
        return NULL;
    }

    effect->rglDirection = (LONG *)SDL_calloc(effect->cAxes, sizeof(LONG));
    if (!effect->rglDirection) {
        FreeRumbleEffectData(effect);
        return NULL;
    }
    effect->dwFlags |= FFEFF_CARTESIAN;

    periodic = (FFPERIODIC *)SDL_calloc(1, sizeof(*periodic));
    if (!periodic) {
        FreeRumbleEffectData(effect);
        return NULL;
    }
    periodic->dwMagnitude = CONVERT_MAGNITUDE(magnitude);
    periodic->dwPeriod = 1000000;

    effect->cbTypeSpecificParams = sizeof(*periodic);
    effect->lpvTypeSpecificParams = periodic;

    return effect;
}

static recDevice *GetDeviceForIndex(int device_index)
{
    recDevice *device = gpDeviceList;
    while (device) {
        if (!device->removed) {
            if (device_index == 0) {
                break;
            }

            --device_index;
        }
        device = device->pNext;
    }
    return device;
}

static void FreeElementList(recElement *pElement)
{
    while (pElement) {
        recElement *pElementNext = pElement->pNext;
        SDL_free(pElement);
        pElement = pElementNext;
    }
}

static recDevice *FreeDevice(recDevice *removeDevice)
{
    recDevice *pDeviceNext = NULL;
    if (removeDevice) {
        if (removeDevice->deviceRef) {
            if (removeDevice->runLoopAttached) {
                /* Calling IOHIDDeviceUnscheduleFromRunLoop without a prior,
                 * paired call to IOHIDDeviceScheduleWithRunLoop can lead
                 * to crashes in MacOS 10.14.x and earlier.  This doesn't
                 * appear to be a problem in MacOS 10.15.x, but we'll
                 * do it anyways.  (Part-of fix for Bug 5034)
                 */
                IOHIDDeviceUnscheduleFromRunLoop(removeDevice->deviceRef, CFRunLoopGetCurrent(), SDL_JOYSTICK_RUNLOOP_MODE);
            }
            CFRelease(removeDevice->deviceRef);
            removeDevice->deviceRef = NULL;
        }

        /* clear out any reference to removeDevice from an associated,
         * live instance of SDL_Joystick  (Part-of fix for Bug 5034)
         */
        SDL_LockJoysticks();
        if (removeDevice->joystick) {
            removeDevice->joystick->hwdata = NULL;
        }
        SDL_UnlockJoysticks();

        // save next device prior to disposing of this device
        pDeviceNext = removeDevice->pNext;

        if (gpDeviceList == removeDevice) {
            gpDeviceList = pDeviceNext;
        } else if (gpDeviceList) {
            recDevice *device;

            for (device = gpDeviceList; device; device = device->pNext) {
                if (device->pNext == removeDevice) {
                    device->pNext = pDeviceNext;
                    break;
                }
            }
        }
        removeDevice->pNext = NULL;

        // free element lists
        FreeElementList(removeDevice->firstAxis);
        FreeElementList(removeDevice->firstButton);
        FreeElementList(removeDevice->firstHat);

        SDL_free(removeDevice);
    }
    return pDeviceNext;
}

static bool GetHIDElementState(recDevice *pDevice, recElement *pElement, SInt32 *pValue)
{
    SInt32 value = 0;
    bool result = false;

    if (pDevice && pDevice->deviceRef && pElement) {
        IOHIDValueRef valueRef;
        if (IOHIDDeviceGetValue(pDevice->deviceRef, pElement->elementRef, &valueRef) == kIOReturnSuccess) {
            value = (SInt32)IOHIDValueGetIntegerValue(valueRef);

            // record min and max for auto calibration
            if (value < pElement->minReport) {
                pElement->minReport = value;
            }
            if (value > pElement->maxReport) {
                pElement->maxReport = value;
            }
            *pValue = value;

            result = true;
        }
    }
    return result;
}

static bool GetHIDScaledCalibratedState(recDevice *pDevice, recElement *pElement, SInt32 min, SInt32 max, SInt32 *pValue)
{
    const float deviceScale = max - min;
    const float readScale = pElement->maxReport - pElement->minReport;
    bool result = false;
    if (GetHIDElementState(pDevice, pElement, pValue)) {
        if (readScale == 0) {
            result = true; // no scaling at all
        } else {
            *pValue = (Sint32)(((*pValue - pElement->minReport) * deviceScale / readScale) + min);
            result = true;
        }
    }
    return result;
}

static void JoystickDeviceWasRemovedCallback(void *ctx, IOReturn result, void *sender)
{
    recDevice *device = (recDevice *)ctx;
    device->removed = true;
    if (device->deviceRef) {
        // deviceRef was invalidated due to the remove
        CFRelease(device->deviceRef);
        device->deviceRef = NULL;
    }
    if (device->ffeffect_ref) {
        FFDeviceReleaseEffect(device->ffdevice, device->ffeffect_ref);
        device->ffeffect_ref = NULL;
    }
    if (device->ffeffect) {
        FreeRumbleEffectData(device->ffeffect);
        device->ffeffect = NULL;
    }
    if (device->ffdevice) {
        FFReleaseDevice(device->ffdevice);
        device->ffdevice = NULL;
        device->ff_initialized = false;
    }
#ifdef SDL_HAPTIC_IOKIT
    MacHaptic_MaybeRemoveDevice(device->ffservice);
#endif

    SDL_PrivateJoystickRemoved(device->instance_id);
}

static void AddHIDElement(const void *value, void *parameter);

// Call AddHIDElement() on all elements in an array of IOHIDElementRefs
static void AddHIDElements(CFArrayRef array, recDevice *pDevice)
{
    const CFRange range = { 0, CFArrayGetCount(array) };
    CFArrayApplyFunction(array, range, AddHIDElement, pDevice);
}

static bool ElementAlreadyAdded(const IOHIDElementCookie cookie, const recElement *listitem)
{
    while (listitem) {
        if (listitem->cookie == cookie) {
            return true;
        }
        listitem = listitem->pNext;
    }
    return false;
}

// See if we care about this HID element, and if so, note it in our recDevice.
static void AddHIDElement(const void *value, void *parameter)
{
    recDevice *pDevice = (recDevice *)parameter;
    IOHIDElementRef refElement = (IOHIDElementRef)value;
    const CFTypeID elementTypeID = refElement ? CFGetTypeID(refElement) : 0;

    if (refElement && (elementTypeID == IOHIDElementGetTypeID())) {
        const IOHIDElementCookie cookie = IOHIDElementGetCookie(refElement);
        const uint32_t usagePage = IOHIDElementGetUsagePage(refElement);
        const uint32_t usage = IOHIDElementGetUsage(refElement);
        recElement *element = NULL;
        recElement **headElement = NULL;

        // look at types of interest
        switch (IOHIDElementGetType(refElement)) {
        case kIOHIDElementTypeInput_Misc:
        case kIOHIDElementTypeInput_Button:
        case kIOHIDElementTypeInput_Axis:
        {
            switch (usagePage) { // only interested in kHIDPage_GenericDesktop and kHIDPage_Button
            case kHIDPage_GenericDesktop:
                switch (usage) {
                case kHIDUsage_GD_X:
                case kHIDUsage_GD_Y:
                case kHIDUsage_GD_Z:
                case kHIDUsage_GD_Rx:
                case kHIDUsage_GD_Ry:
                case kHIDUsage_GD_Rz:
                case kHIDUsage_GD_Slider:
                case kHIDUsage_GD_Dial:
                case kHIDUsage_GD_Wheel:
                    if (!ElementAlreadyAdded(cookie, pDevice->firstAxis)) {
                        element = (recElement *)SDL_calloc(1, sizeof(recElement));
                        if (element) {
                            pDevice->axes++;
                            headElement = &(pDevice->firstAxis);
                        }
                    }
                    break;

                case kHIDUsage_GD_Hatswitch:
                    if (!ElementAlreadyAdded(cookie, pDevice->firstHat)) {
                        element = (recElement *)SDL_calloc(1, sizeof(recElement));
                        if (element) {
                            pDevice->hats++;
                            headElement = &(pDevice->firstHat);
                        }
                    }
                    break;
                case kHIDUsage_GD_DPadUp:
                case kHIDUsage_GD_DPadDown:
                case kHIDUsage_GD_DPadRight:
                case kHIDUsage_GD_DPadLeft:
                case kHIDUsage_GD_Start:
                case kHIDUsage_GD_Select:
                case kHIDUsage_GD_SystemMainMenu:
                    if (!ElementAlreadyAdded(cookie, pDevice->firstButton)) {
                        element = (recElement *)SDL_calloc(1, sizeof(recElement));
                        if (element) {
                            pDevice->buttons++;
                            headElement = &(pDevice->firstButton);
                        }
                    }
                    break;
                }
                break;

            case kHIDPage_Simulation:
                switch (usage) {
                case kHIDUsage_Sim_Rudder:
                case kHIDUsage_Sim_Throttle:
                case kHIDUsage_Sim_Accelerator:
                case kHIDUsage_Sim_Brake:
                    if (!ElementAlreadyAdded(cookie, pDevice->firstAxis)) {
                        element = (recElement *)SDL_calloc(1, sizeof(recElement));
                        if (element) {
                            pDevice->axes++;
                            headElement = &(pDevice->firstAxis);
                        }
                    }
                    break;

                default:
                    break;
                }
                break;

            case kHIDPage_Button:
            case kHIDPage_Consumer: // e.g. 'pause' button on Steelseries MFi gamepads.
                if (!ElementAlreadyAdded(cookie, pDevice->firstButton)) {
                    element = (recElement *)SDL_calloc(1, sizeof(recElement));
                    if (element) {
                        pDevice->buttons++;
                        headElement = &(pDevice->firstButton);
                    }
                }
                break;

            default:
                break;
            }
        } break;

        case kIOHIDElementTypeCollection:
        {
            CFArrayRef array = IOHIDElementGetChildren(refElement);
            if (array) {
                AddHIDElements(array, pDevice);
            }
        } break;

        default:
            break;
        }

        if (element && headElement) { // add to list
            recElement *elementPrevious = NULL;
            recElement *elementCurrent = *headElement;
            while (elementCurrent && usage >= elementCurrent->usage) {
                elementPrevious = elementCurrent;
                elementCurrent = elementCurrent->pNext;
            }
            if (elementPrevious) {
                elementPrevious->pNext = element;
            } else {
                *headElement = element;
            }

            element->elementRef = refElement;
            element->usagePage = usagePage;
            element->usage = usage;
            element->pNext = elementCurrent;

            element->minReport = element->min = (SInt32)IOHIDElementGetLogicalMin(refElement);
            element->maxReport = element->max = (SInt32)IOHIDElementGetLogicalMax(refElement);
            element->cookie = IOHIDElementGetCookie(refElement);

            pDevice->elements++;
        }
    }
}

static int GetSteamVirtualGamepadSlot(Uint16 vendor_id, Uint16 product_id, const char *product_string)
{
    int slot = -1;

    if (vendor_id == USB_VENDOR_MICROSOFT && product_id == USB_PRODUCT_XBOX360_WIRED_CONTROLLER) {
        // Gamepad name is "GamePad-N", where N is slot + 1
        if (SDL_sscanf(product_string, "GamePad-%d", &slot) == 1) {
            slot -= 1;
        }
    }
    return slot;
}

static bool GetDeviceInfo(IOHIDDeviceRef hidDevice, recDevice *pDevice)
{
    Sint32 vendor = 0;
    Sint32 product = 0;
    Sint32 version = 0;
    char *name;
    char manufacturer_string[256];
    char product_string[256];
    CFTypeRef refCF = NULL;
    CFArrayRef array = NULL;

    // get usage page and usage
    refCF = IOHIDDeviceGetProperty(hidDevice, CFSTR(kIOHIDPrimaryUsagePageKey));
    if (refCF) {
        CFNumberGetValue(refCF, kCFNumberSInt32Type, &pDevice->usagePage);
    }
    if (pDevice->usagePage != kHIDPage_GenericDesktop) {
        return false; // Filter device list to non-keyboard/mouse stuff
    }

    refCF = IOHIDDeviceGetProperty(hidDevice, CFSTR(kIOHIDPrimaryUsageKey));
    if (refCF) {
        CFNumberGetValue(refCF, kCFNumberSInt32Type, &pDevice->usage);
    }

    if ((pDevice->usage != kHIDUsage_GD_Joystick &&
         pDevice->usage != kHIDUsage_GD_GamePad &&
         pDevice->usage != kHIDUsage_GD_MultiAxisController)) {
        return false; // Filter device list to non-keyboard/mouse stuff
    }

    /* Make sure we retain the use of the IOKit-provided device-object,
       lest the device get disconnected and we try to use it.  (Fixes
       SDL-Bugzilla #4961, aka. https://bugzilla.libsdl.org/show_bug.cgi?id=4961 )
    */
    CFRetain(hidDevice);

    /* Now that we've CFRetain'ed the device-object (for our use), we'll
       save the reference to it.
    */
    pDevice->deviceRef = hidDevice;

    refCF = IOHIDDeviceGetProperty(hidDevice, CFSTR(kIOHIDVendorIDKey));
    if (refCF) {
        CFNumberGetValue(refCF, kCFNumberSInt32Type, &vendor);
    }

    refCF = IOHIDDeviceGetProperty(hidDevice, CFSTR(kIOHIDProductIDKey));
    if (refCF) {
        CFNumberGetValue(refCF, kCFNumberSInt32Type, &product);
    }

    refCF = IOHIDDeviceGetProperty(hidDevice, CFSTR(kIOHIDVersionNumberKey));
    if (refCF) {
        CFNumberGetValue(refCF, kCFNumberSInt32Type, &version);
    }

    if (SDL_IsJoystickXboxOne(vendor, product)) {
        // We can't actually use this API for Xbox controllers
        return false;
    }

    // get device name
    refCF = IOHIDDeviceGetProperty(hidDevice, CFSTR(kIOHIDManufacturerKey));
    if ((!refCF) || (!CFStringGetCString(refCF, manufacturer_string, sizeof(manufacturer_string), kCFStringEncodingUTF8))) {
        manufacturer_string[0] = '\0';
    }
    refCF = IOHIDDeviceGetProperty(hidDevice, CFSTR(kIOHIDProductKey));
    if ((!refCF) || (!CFStringGetCString(refCF, product_string, sizeof(product_string), kCFStringEncodingUTF8))) {
        product_string[0] = '\0';
    }
    name = SDL_CreateJoystickName(vendor, product, manufacturer_string, product_string);
    if (name) {
        SDL_strlcpy(pDevice->product, name, sizeof(pDevice->product));
        SDL_free(name);
    }

    if (SDL_ShouldIgnoreJoystick(vendor, product, version, pDevice->product)) {
        return false;
    }

    if (SDL_JoystickHandledByAnotherDriver(&SDL_DARWIN_JoystickDriver, vendor, product, version, pDevice->product)) {
        return false;
    }

    pDevice->guid = SDL_CreateJoystickGUID(SDL_HARDWARE_BUS_USB, (Uint16)vendor, (Uint16)product, (Uint16)version, manufacturer_string, product_string, 0, 0);
    pDevice->steam_virtual_gamepad_slot = GetSteamVirtualGamepadSlot((Uint16)vendor, (Uint16)product, product_string);

    array = IOHIDDeviceCopyMatchingElements(hidDevice, NULL, kIOHIDOptionsTypeNone);
    if (array) {
        AddHIDElements(array, pDevice);
        CFRelease(array);
    }

    return true;
}

static bool JoystickAlreadyKnown(IOHIDDeviceRef ioHIDDeviceObject)
{
    recDevice *i;

#ifdef SDL_JOYSTICK_MFI
    extern bool IOS_SupportedHIDDevice(IOHIDDeviceRef device);
    if (IOS_SupportedHIDDevice(ioHIDDeviceObject)) {
        return true;
    }
#endif

    for (i = gpDeviceList; i; i = i->pNext) {
        if (i->deviceRef == ioHIDDeviceObject) {
            return true;
        }
    }
    return false;
}

static void JoystickDeviceWasAddedCallback(void *ctx, IOReturn res, void *sender, IOHIDDeviceRef ioHIDDeviceObject)
{
    recDevice *device;
    io_service_t ioservice;

    if (res != kIOReturnSuccess) {
        return;
    }

    if (JoystickAlreadyKnown(ioHIDDeviceObject)) {
        return; // IOKit sent us a duplicate.
    }

    device = (recDevice *)SDL_calloc(1, sizeof(recDevice));
    if (!device) {
        return;
    }

    if (!GetDeviceInfo(ioHIDDeviceObject, device)) {
        FreeDevice(device);
        return; // not a device we care about, probably.
    }

    // Get notified when this device is disconnected.
    IOHIDDeviceRegisterRemovalCallback(ioHIDDeviceObject, JoystickDeviceWasRemovedCallback, device);
    IOHIDDeviceScheduleWithRunLoop(ioHIDDeviceObject, CFRunLoopGetCurrent(), SDL_JOYSTICK_RUNLOOP_MODE);
    device->runLoopAttached = true;

    // Allocate an instance ID for this device
    device->instance_id = SDL_GetNextObjectID();

    // We have to do some storage of the io_service_t for SDL_OpenHapticFromJoystick
    ioservice = IOHIDDeviceGetService(ioHIDDeviceObject);
    if ((ioservice) && (FFIsForceFeedback(ioservice) == FF_OK)) {
        device->ffservice = ioservice;
#ifdef SDL_HAPTIC_IOKIT
        MacHaptic_MaybeAddDevice(ioservice);
#endif
    }

    // Add device to the end of the list
    if (!gpDeviceList) {
        gpDeviceList = device;
    } else {
        recDevice *curdevice;

        curdevice = gpDeviceList;
        while (curdevice->pNext) {
            curdevice = curdevice->pNext;
        }
        curdevice->pNext = device;
    }

    SDL_PrivateJoystickAdded(device->instance_id);
}

static bool ConfigHIDManager(CFArrayRef matchingArray)
{
    CFRunLoopRef runloop = CFRunLoopGetCurrent();

    if (IOHIDManagerOpen(hidman, kIOHIDOptionsTypeNone) != kIOReturnSuccess) {
        return false;
    }

    IOHIDManagerSetDeviceMatchingMultiple(hidman, matchingArray);
    IOHIDManagerRegisterDeviceMatchingCallback(hidman, JoystickDeviceWasAddedCallback, NULL);
    IOHIDManagerScheduleWithRunLoop(hidman, runloop, SDL_JOYSTICK_RUNLOOP_MODE);

    while (CFRunLoopRunInMode(SDL_JOYSTICK_RUNLOOP_MODE, 0, TRUE) == kCFRunLoopRunHandledSource) {
        // no-op. Callback fires once per existing device.
    }

    // future hotplug events will come through SDL_JOYSTICK_RUNLOOP_MODE now.

    return true; // good to go.
}

static CFDictionaryRef CreateHIDDeviceMatchDictionary(const UInt32 page, const UInt32 usage, int *okay)
{
    CFDictionaryRef result = NULL;
    CFNumberRef pageNumRef = CFNumberCreate(kCFAllocatorDefault, kCFNumberIntType, &page);
    CFNumberRef usageNumRef = CFNumberCreate(kCFAllocatorDefault, kCFNumberIntType, &usage);
    const void *keys[2] = { (void *)CFSTR(kIOHIDDeviceUsagePageKey), (void *)CFSTR(kIOHIDDeviceUsageKey) };
    const void *vals[2] = { (void *)pageNumRef, (void *)usageNumRef };

    if (pageNumRef && usageNumRef) {
        result = CFDictionaryCreate(kCFAllocatorDefault, keys, vals, 2, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
    }

    if (pageNumRef) {
        CFRelease(pageNumRef);
    }
    if (usageNumRef) {
        CFRelease(usageNumRef);
    }

    if (!result) {
        *okay = 0;
    }

    return result;
}

static bool CreateHIDManager(void)
{
    bool result = false;
    int okay = 1;
    const void *vals[] = {
        (void *)CreateHIDDeviceMatchDictionary(kHIDPage_GenericDesktop, kHIDUsage_GD_Joystick, &okay),
        (void *)CreateHIDDeviceMatchDictionary(kHIDPage_GenericDesktop, kHIDUsage_GD_GamePad, &okay),
        (void *)CreateHIDDeviceMatchDictionary(kHIDPage_GenericDesktop, kHIDUsage_GD_MultiAxisController, &okay),
    };
    const size_t numElements = SDL_arraysize(vals);
    CFArrayRef array = okay ? CFArrayCreate(kCFAllocatorDefault, vals, numElements, &kCFTypeArrayCallBacks) : NULL;
    size_t i;

    for (i = 0; i < numElements; i++) {
        if (vals[i]) {
            CFRelease((CFTypeRef)vals[i]);
        }
    }

    if (array) {
        hidman = IOHIDManagerCreate(kCFAllocatorDefault, kIOHIDOptionsTypeNone);
        if (hidman != NULL) {
            result = ConfigHIDManager(array);
        }
        CFRelease(array);
    }

    return result;
}

static bool DARWIN_JoystickInit(void)
{
    if (!SDL_GetHintBoolean(SDL_HINT_JOYSTICK_IOKIT, true)) {
        return true;
    }

    if (!CreateHIDManager()) {
        return SDL_SetError("Joystick: Couldn't initialize HID Manager");
    }

    return true;
}

static int DARWIN_JoystickGetCount(void)
{
    recDevice *device = gpDeviceList;
    int nJoySticks = 0;

    while (device) {
        if (!device->removed) {
            nJoySticks++;
        }
        device = device->pNext;
    }

    return nJoySticks;
}

static void DARWIN_JoystickDetect(void)
{
    recDevice *device = gpDeviceList;
    while (device) {
        if (device->removed) {
            device = FreeDevice(device);
        } else {
            device = device->pNext;
        }
    }

    if (hidman) {
        /* run this after the checks above so we don't set device->removed and delete the device before
           DARWIN_JoystickUpdate can run to clean up the SDL_Joystick object that owns this device */
        while (CFRunLoopRunInMode(SDL_JOYSTICK_RUNLOOP_MODE, 0, TRUE) == kCFRunLoopRunHandledSource) {
            // no-op. Pending callbacks will fire in CFRunLoopRunInMode().
        }
    }
}

static bool DARWIN_JoystickIsDevicePresent(Uint16 vendor_id, Uint16 product_id, Uint16 version, const char *name)
{
    // We don't override any other drivers
    return false;
}

static const char *DARWIN_JoystickGetDeviceName(int device_index)
{
    recDevice *device = GetDeviceForIndex(device_index);
    return device ? device->product : "UNKNOWN";
}

static const char *DARWIN_JoystickGetDevicePath(int device_index)
{
    return NULL;
}

static int DARWIN_JoystickGetDeviceSteamVirtualGamepadSlot(int device_index)
{
    recDevice *device = GetDeviceForIndex(device_index);
    return device ? device->steam_virtual_gamepad_slot : -1;
}

static int DARWIN_JoystickGetDevicePlayerIndex(int device_index)
{
    return -1;
}

static void DARWIN_JoystickSetDevicePlayerIndex(int device_index, int player_index)
{
}

static SDL_GUID DARWIN_JoystickGetDeviceGUID(int device_index)
{
    recDevice *device = GetDeviceForIndex(device_index);
    SDL_GUID guid;
    if (device) {
        guid = device->guid;
    } else {
        SDL_zero(guid);
    }
    return guid;
}

static SDL_JoystickID DARWIN_JoystickGetDeviceInstanceID(int device_index)
{
    recDevice *device = GetDeviceForIndex(device_index);
    return device ? device->instance_id : 0;
}

static bool DARWIN_JoystickOpen(SDL_Joystick *joystick, int device_index)
{
    recDevice *device = GetDeviceForIndex(device_index);

    joystick->hwdata = device;
    device->joystick = joystick;
    joystick->name = device->product;

    joystick->naxes = device->axes;
    joystick->nhats = device->hats;
    joystick->nbuttons = device->buttons;

    if (device->ffservice) {
        SDL_SetBooleanProperty(SDL_GetJoystickProperties(joystick), SDL_PROP_JOYSTICK_CAP_RUMBLE_BOOLEAN, true);
    }

    return true;
}

/*
 * Like strerror but for force feedback errors.
 */
static const char *FFStrError(unsigned int err)
{
    switch (err) {
    case FFERR_DEVICEFULL:
        return "device full";
    // This should be valid, but for some reason isn't defined...
    /* case FFERR_DEVICENOTREG:
        return "device not registered"; */
    case FFERR_DEVICEPAUSED:
        return "device paused";
    case FFERR_DEVICERELEASED:
        return "device released";
    case FFERR_EFFECTPLAYING:
        return "effect playing";
    case FFERR_EFFECTTYPEMISMATCH:
        return "effect type mismatch";
    case FFERR_EFFECTTYPENOTSUPPORTED:
        return "effect type not supported";
    case FFERR_GENERIC:
        return "undetermined error";
    case FFERR_HASEFFECTS:
        return "device has effects";
    case FFERR_INCOMPLETEEFFECT:
        return "incomplete effect";
    case FFERR_INTERNAL:
        return "internal fault";
    case FFERR_INVALIDDOWNLOADID:
        return "invalid download id";
    case FFERR_INVALIDPARAM:
        return "invalid parameter";
    case FFERR_MOREDATA:
        return "more data";
    case FFERR_NOINTERFACE:
        return "interface not supported";
    case FFERR_NOTDOWNLOADED:
        return "effect is not downloaded";
    case FFERR_NOTINITIALIZED:
        return "object has not been initialized";
    case FFERR_OUTOFMEMORY:
        return "out of memory";
    case FFERR_UNPLUGGED:
        return "device is unplugged";
    case FFERR_UNSUPPORTED:
        return "function call unsupported";
    case FFERR_UNSUPPORTEDAXIS:
        return "axis unsupported";

    default:
        return "unknown error";
    }
}

static bool DARWIN_JoystickInitRumble(recDevice *device, Sint16 magnitude)
{
    HRESULT result;

    if (!device->ffdevice) {
        result = FFCreateDevice(device->ffservice, &device->ffdevice);
        if (result != FF_OK) {
            return SDL_SetError("Unable to create force feedback device from service: %s", FFStrError(result));
        }
    }

    // Reset and then enable actuators
    result = FFDeviceSendForceFeedbackCommand(device->ffdevice, FFSFFC_RESET);
    if (result != FF_OK) {
        return SDL_SetError("Unable to reset force feedback device: %s", FFStrError(result));
    }

    result = FFDeviceSendForceFeedbackCommand(device->ffdevice, FFSFFC_SETACTUATORSON);
    if (result != FF_OK) {
        return SDL_SetError("Unable to enable force feedback actuators: %s", FFStrError(result));
    }

    // Create the effect
    device->ffeffect = CreateRumbleEffectData(magnitude);
    if (!device->ffeffect) {
        return false;
    }

    result = FFDeviceCreateEffect(device->ffdevice, kFFEffectType_Sine_ID,
                                  device->ffeffect, &device->ffeffect_ref);
    if (result != FF_OK) {
        return SDL_SetError("Haptic: Unable to create effect: %s", FFStrError(result));
    }
    return true;
}

static bool DARWIN_JoystickRumble(SDL_Joystick *joystick, Uint16 low_frequency_rumble, Uint16 high_frequency_rumble)
{
    HRESULT result;
    recDevice *device = joystick->hwdata;

    // Scale and average the two rumble strengths
    Sint16 magnitude = (Sint16)(((low_frequency_rumble / 2) + (high_frequency_rumble / 2)) / 2);

    if (!device) {
        return SDL_SetError("Rumble failed, device disconnected");
    }

    if (!device->ffservice) {
        return SDL_Unsupported();
    }

    if (device->ff_initialized) {
        FFPERIODIC *periodic = ((FFPERIODIC *)device->ffeffect->lpvTypeSpecificParams);
        periodic->dwMagnitude = CONVERT_MAGNITUDE(magnitude);

        result = FFEffectSetParameters(device->ffeffect_ref, device->ffeffect,
                                       (FFEP_DURATION | FFEP_TYPESPECIFICPARAMS));
        if (result != FF_OK) {
            return SDL_SetError("Unable to update rumble effect: %s", FFStrError(result));
        }
    } else {
        if (!DARWIN_JoystickInitRumble(device, magnitude)) {
            return false;
        }
        device->ff_initialized = true;
    }

    result = FFEffectStart(device->ffeffect_ref, 1, 0);
    if (result != FF_OK) {
        return SDL_SetError("Unable to run the rumble effect: %s", FFStrError(result));
    }
    return true;
}

static bool DARWIN_JoystickRumbleTriggers(SDL_Joystick *joystick, Uint16 left_rumble, Uint16 right_rumble)
{
    return SDL_Unsupported();
}

static bool DARWIN_JoystickSetLED(SDL_Joystick *joystick, Uint8 red, Uint8 green, Uint8 blue)
{
    return SDL_Unsupported();
}

static bool DARWIN_JoystickSendEffect(SDL_Joystick *joystick, const void *data, int size)
{
    return SDL_Unsupported();
}

static bool DARWIN_JoystickSetSensorsEnabled(SDL_Joystick *joystick, bool enabled)
{
    return SDL_Unsupported();
}

static void DARWIN_JoystickUpdate(SDL_Joystick *joystick)
{
    recDevice *device = joystick->hwdata;
    recElement *element;
    SInt32 value, range;
    int i, goodRead = false;
    Uint64 timestamp = SDL_GetTicksNS();

    if (!device) {
        return;
    }

    if (device->removed) { // device was unplugged; ignore it.
        if (joystick->hwdata) {
            joystick->hwdata = NULL;
        }
        return;
    }

    element = device->firstAxis;
    i = 0;

    while (element) {
        goodRead = GetHIDScaledCalibratedState(device, element, -32768, 32767, &value);
        if (goodRead) {
            SDL_SendJoystickAxis(timestamp, joystick, i, value);
        }

        element = element->pNext;
        ++i;
    }

    element = device->firstButton;
    i = 0;
    while (element) {
        goodRead = GetHIDElementState(device, element, &value);
        if (goodRead) {
            SDL_SendJoystickButton(timestamp, joystick, i, (value != 0));
        }

        element = element->pNext;
        ++i;
    }

    element = device->firstHat;
    i = 0;

    while (element) {
        Uint8 pos = 0;

        range = (element->max - element->min + 1);
        goodRead = GetHIDElementState(device, element, &value);
        if (goodRead) {
            value -= element->min;
            if (range == 4) { // 4 position hatswitch - scale up value
                value *= 2;
            } else if (range != 8) { // Neither a 4 nor 8 positions - fall back to default position (centered)
                value = -1;
            }
            switch (value) {
            case 0:
                pos = SDL_HAT_UP;
                break;
            case 1:
                pos = SDL_HAT_RIGHTUP;
                break;
            case 2:
                pos = SDL_HAT_RIGHT;
                break;
            case 3:
                pos = SDL_HAT_RIGHTDOWN;
                break;
            case 4:
                pos = SDL_HAT_DOWN;
                break;
            case 5:
                pos = SDL_HAT_LEFTDOWN;
                break;
            case 6:
                pos = SDL_HAT_LEFT;
                break;
            case 7:
                pos = SDL_HAT_LEFTUP;
                break;
            default:
                /* Every other value is mapped to center. We do that because some
                 * joysticks use 8 and some 15 for this value, and apparently
                 * there are even more variants out there - so we try to be generous.
                 */
                pos = SDL_HAT_CENTERED;
                break;
            }

            SDL_SendJoystickHat(timestamp, joystick, i, pos);
        }

        element = element->pNext;
        ++i;
    }
}

static void DARWIN_JoystickClose(SDL_Joystick *joystick)
{
    recDevice *device = joystick->hwdata;
    if (device) {
        device->joystick = NULL;
    }
}

static void DARWIN_JoystickQuit(void)
{
    while (FreeDevice(gpDeviceList)) {
        // spin
    }

    if (hidman) {
        IOHIDManagerUnscheduleFromRunLoop(hidman, CFRunLoopGetCurrent(), SDL_JOYSTICK_RUNLOOP_MODE);
        IOHIDManagerClose(hidman, kIOHIDOptionsTypeNone);
        CFRelease(hidman);
        hidman = NULL;
    }
}

static bool DARWIN_JoystickGetGamepadMapping(int device_index, SDL_GamepadMapping *out)
{
    return false;
}

SDL_JoystickDriver SDL_DARWIN_JoystickDriver = {
    DARWIN_JoystickInit,
    DARWIN_JoystickGetCount,
    DARWIN_JoystickDetect,
    DARWIN_JoystickIsDevicePresent,
    DARWIN_JoystickGetDeviceName,
    DARWIN_JoystickGetDevicePath,
    DARWIN_JoystickGetDeviceSteamVirtualGamepadSlot,
    DARWIN_JoystickGetDevicePlayerIndex,
    DARWIN_JoystickSetDevicePlayerIndex,
    DARWIN_JoystickGetDeviceGUID,
    DARWIN_JoystickGetDeviceInstanceID,
    DARWIN_JoystickOpen,
    DARWIN_JoystickRumble,
    DARWIN_JoystickRumbleTriggers,
    DARWIN_JoystickSetLED,
    DARWIN_JoystickSendEffect,
    DARWIN_JoystickSetSensorsEnabled,
    DARWIN_JoystickUpdate,
    DARWIN_JoystickClose,
    DARWIN_JoystickQuit,
    DARWIN_JoystickGetGamepadMapping
};

#endif // SDL_JOYSTICK_IOKIT
