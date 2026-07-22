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

#ifndef SDL_JOYSTICK_HIDAPI_H
#define SDL_JOYSTICK_HIDAPI_H

#include "../usb_ids.h"

// This is the full set of HIDAPI drivers available
#define SDL_JOYSTICK_HIDAPI_GAMECUBE
#define SDL_JOYSTICK_HIDAPI_LUNA
#define SDL_JOYSTICK_HIDAPI_PS3
#define SDL_JOYSTICK_HIDAPI_PS4
#define SDL_JOYSTICK_HIDAPI_PS5
#define SDL_JOYSTICK_HIDAPI_STADIA
#define SDL_JOYSTICK_HIDAPI_STEAM
#define SDL_JOYSTICK_HIDAPI_STEAMDECK
#define SDL_JOYSTICK_HIDAPI_SWITCH
#define SDL_JOYSTICK_HIDAPI_WII
#define SDL_JOYSTICK_HIDAPI_XBOX360
#define SDL_JOYSTICK_HIDAPI_XBOXONE
#define SDL_JOYSTICK_HIDAPI_SHIELD
#define SDL_JOYSTICK_HIDAPI_STEAM_HORI

// Joystick capability definitions
#define SDL_JOYSTICK_CAP_MONO_LED       0x00000001
#define SDL_JOYSTICK_CAP_RGB_LED        0x00000002
#define SDL_JOYSTICK_CAP_PLAYER_LED     0x00000004
#define SDL_JOYSTICK_CAP_RUMBLE         0x00000010
#define SDL_JOYSTICK_CAP_TRIGGER_RUMBLE 0x00000020

// Whether HIDAPI is enabled by default
#if defined(SDL_PLATFORM_ANDROID) || \
    defined(SDL_PLATFORM_IOS) || \
    defined(SDL_PLATFORM_TVOS) || \
    defined(SDL_PLATFORM_VISIONOS)
// On Android, HIDAPI prompts for permissions and acquires exclusive access to the device, and on Apple mobile platforms it doesn't do anything except for handling Bluetooth Steam Controllers, so we'll leave it off by default.
#define SDL_HIDAPI_DEFAULT false
#else
#define SDL_HIDAPI_DEFAULT true
#endif

// The maximum size of a USB packet for HID devices
#define USB_PACKET_LENGTH 64

// Forward declaration
struct SDL_HIDAPI_DeviceDriver;

typedef struct SDL_HIDAPI_Device
{
    char *name;
    char *manufacturer_string;
    char *product_string;
    char *path;
    Uint16 vendor_id;
    Uint16 product_id;
    Uint16 version;
    char *serial;
    SDL_GUID guid;
    int interface_number; // Available on Windows and Linux
    int interface_class;
    int interface_subclass;
    int interface_protocol;
    Uint16 usage_page; // Available on Windows and macOS
    Uint16 usage;      // Available on Windows and macOS
    bool is_bluetooth;
    SDL_JoystickType joystick_type;
    SDL_GamepadType type;
    int steam_virtual_gamepad_slot;

    struct SDL_HIDAPI_DeviceDriver *driver;
    void *context;
    SDL_Mutex *dev_lock;
    SDL_hid_device *dev;
    SDL_AtomicInt rumble_pending;
    int num_joysticks;
    SDL_JoystickID *joysticks;

    // Used during scanning for device changes
    bool seen;

    // Used to flag that the device is being updated
    bool updating;

    // Used to flag devices that failed open
    // This can happen on Windows with Bluetooth devices that have turned off
    bool broken;

    struct SDL_HIDAPI_Device *parent;
    int num_children;
    struct SDL_HIDAPI_Device **children;

    struct SDL_HIDAPI_Device *next;
} SDL_HIDAPI_Device;

typedef struct SDL_HIDAPI_DeviceDriver
{
    const char *name;
    bool enabled;
    void (*RegisterHints)(SDL_HintCallback callback, void *userdata);
    void (*UnregisterHints)(SDL_HintCallback callback, void *userdata);
    bool (*IsEnabled)(void);
    bool (*IsSupportedDevice)(SDL_HIDAPI_Device *device, const char *name, SDL_GamepadType type, Uint16 vendor_id, Uint16 product_id, Uint16 version, int interface_number, int interface_class, int interface_subclass, int interface_protocol);
    bool (*InitDevice)(SDL_HIDAPI_Device *device);
    int (*GetDevicePlayerIndex)(SDL_HIDAPI_Device *device, SDL_JoystickID instance_id);
    void (*SetDevicePlayerIndex)(SDL_HIDAPI_Device *device, SDL_JoystickID instance_id, int player_index);
    bool (*UpdateDevice)(SDL_HIDAPI_Device *device);
    bool (*OpenJoystick)(SDL_HIDAPI_Device *device, SDL_Joystick *joystick);
    bool (*RumbleJoystick)(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint16 low_frequency_rumble, Uint16 high_frequency_rumble);
    bool (*RumbleJoystickTriggers)(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint16 left_rumble, Uint16 right_rumble);
    Uint32 (*GetJoystickCapabilities)(SDL_HIDAPI_Device *device, SDL_Joystick *joystick);
    bool (*SetJoystickLED)(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint8 red, Uint8 green, Uint8 blue);
    bool (*SendJoystickEffect)(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, const void *data, int size);
    bool (*SetJoystickSensorsEnabled)(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, bool enabled);
    void (*CloseJoystick)(SDL_HIDAPI_Device *device, SDL_Joystick *joystick);
    void (*FreeDevice)(SDL_HIDAPI_Device *device);

} SDL_HIDAPI_DeviceDriver;

// HIDAPI device support
extern SDL_HIDAPI_DeviceDriver SDL_HIDAPI_DriverCombined;
extern SDL_HIDAPI_DeviceDriver SDL_HIDAPI_DriverGameCube;
extern SDL_HIDAPI_DeviceDriver SDL_HIDAPI_DriverJoyCons;
extern SDL_HIDAPI_DeviceDriver SDL_HIDAPI_DriverLuna;
extern SDL_HIDAPI_DeviceDriver SDL_HIDAPI_DriverNintendoClassic;
extern SDL_HIDAPI_DeviceDriver SDL_HIDAPI_DriverPS3;
extern SDL_HIDAPI_DeviceDriver SDL_HIDAPI_DriverPS3ThirdParty;
extern SDL_HIDAPI_DeviceDriver SDL_HIDAPI_DriverPS3SonySixaxis;
extern SDL_HIDAPI_DeviceDriver SDL_HIDAPI_DriverPS4;
extern SDL_HIDAPI_DeviceDriver SDL_HIDAPI_DriverPS5;
extern SDL_HIDAPI_DeviceDriver SDL_HIDAPI_DriverShield;
extern SDL_HIDAPI_DeviceDriver SDL_HIDAPI_DriverStadia;
extern SDL_HIDAPI_DeviceDriver SDL_HIDAPI_DriverSteam;
extern SDL_HIDAPI_DeviceDriver SDL_HIDAPI_DriverSteamDeck;
extern SDL_HIDAPI_DeviceDriver SDL_HIDAPI_DriverSwitch;
extern SDL_HIDAPI_DeviceDriver SDL_HIDAPI_DriverWii;
extern SDL_HIDAPI_DeviceDriver SDL_HIDAPI_DriverXbox360;
extern SDL_HIDAPI_DeviceDriver SDL_HIDAPI_DriverXbox360W;
extern SDL_HIDAPI_DeviceDriver SDL_HIDAPI_DriverXboxOne;
extern SDL_HIDAPI_DeviceDriver SDL_HIDAPI_DriverSteamHori;

// Return true if a HID device is present and supported as a joystick of the given type
extern bool HIDAPI_IsDeviceTypePresent(SDL_GamepadType type);

// Return true if a HID device is present and supported as a joystick
extern bool HIDAPI_IsDevicePresent(Uint16 vendor_id, Uint16 product_id, Uint16 version, const char *name);

// Return the name of a connected device, which should be freed with SDL_free(), or NULL if it's not available
extern char *HIDAPI_GetDeviceProductName(Uint16 vendor_id, Uint16 product_id);

// Return the manufacturer of a connected device, which should be freed with SDL_free(), or NULL if it's not available
extern char *HIDAPI_GetDeviceManufacturerName(Uint16 vendor_id, Uint16 product_id);

// Return the type of a joystick if it's present and supported
extern SDL_JoystickType HIDAPI_GetJoystickTypeFromGUID(SDL_GUID guid);

// Return the type of a game controller if it's present and supported
extern SDL_GamepadType HIDAPI_GetGamepadTypeFromGUID(SDL_GUID guid);

extern void HIDAPI_UpdateDevices(void);
extern void HIDAPI_SetDeviceName(SDL_HIDAPI_Device *device, const char *name);
extern void HIDAPI_SetDeviceProduct(SDL_HIDAPI_Device *device, Uint16 vendor_id, Uint16 product_id);
extern void HIDAPI_SetDeviceSerial(SDL_HIDAPI_Device *device, const char *serial);
extern bool HIDAPI_HasConnectedUSBDevice(const char *serial);
extern void HIDAPI_DisconnectBluetoothDevice(const char *serial);
extern bool HIDAPI_JoystickConnected(SDL_HIDAPI_Device *device, SDL_JoystickID *pJoystickID);
extern void HIDAPI_JoystickDisconnected(SDL_HIDAPI_Device *device, SDL_JoystickID joystickID);
extern void HIDAPI_UpdateDeviceProperties(SDL_HIDAPI_Device *device);

extern void HIDAPI_DumpPacket(const char *prefix, const Uint8 *data, int size);

extern bool HIDAPI_SupportsPlaystationDetection(Uint16 vendor, Uint16 product);

extern float HIDAPI_RemapVal(float val, float val_min, float val_max, float output_min, float output_max);

#endif // SDL_JOYSTICK_HIDAPI_H
