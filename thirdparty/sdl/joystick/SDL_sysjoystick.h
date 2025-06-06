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

#ifndef SDL_sysjoystick_h_
#define SDL_sysjoystick_h_

// This is the system specific header for the SDL joystick API
#include "SDL_joystick_c.h"

// Set up for C function definitions, even when using C++
#ifdef __cplusplus
extern "C" {
#endif

// The SDL joystick structure

typedef struct SDL_JoystickAxisInfo
{
    Sint16 initial_value;           // Initial axis state
    Sint16 value;                   // Current axis state
    Sint16 zero;                    // Zero point on the axis (-32768 for triggers)
    bool has_initial_value;     // Whether we've seen a value on the axis yet
    bool has_second_value;      // Whether we've seen a second value on the axis yet
    bool sent_initial_value;    // Whether we've sent the initial axis value
    bool sending_initial_value; // Whether we are sending the initial axis value
} SDL_JoystickAxisInfo;

typedef struct SDL_JoystickBallData
{
    int dx;
    int dy;
} SDL_JoystickBallData;

typedef struct SDL_JoystickTouchpadFingerInfo
{
    bool down;
    float x;
    float y;
    float pressure;
} SDL_JoystickTouchpadFingerInfo;

typedef struct SDL_JoystickTouchpadInfo
{
    int nfingers;
    SDL_JoystickTouchpadFingerInfo *fingers;
} SDL_JoystickTouchpadInfo;

typedef struct SDL_JoystickSensorInfo
{
    SDL_SensorType type;
    bool enabled;
    float rate;
    float data[3]; // If this needs to expand, update SDL_GamepadSensorEvent
} SDL_JoystickSensorInfo;

#define _guarded SDL_GUARDED_BY(SDL_joystick_lock)

struct SDL_Joystick
{
    SDL_JoystickID instance_id _guarded; // Device instance, monotonically increasing from 0
    char *name _guarded;                 // Joystick name - system dependent
    char *path _guarded;                 // Joystick path - system dependent
    char *serial _guarded;               // Joystick serial
    SDL_GUID guid _guarded;      // Joystick guid
    Uint16 firmware_version _guarded;    // Firmware version, if available
    Uint64 steam_handle _guarded;        // Steam controller API handle
    bool swap_face_buttons _guarded;     // Whether we should swap face buttons
    bool is_virtual _guarded;            // Whether this is a virtual joystick

    int naxes _guarded; // Number of axis controls on the joystick
    SDL_JoystickAxisInfo *axes _guarded;

    int nballs _guarded; // Number of trackballs on the joystick
    SDL_JoystickBallData *balls _guarded; // Current ball motion deltas

    int nhats _guarded;   // Number of hats on the joystick
    Uint8 *hats _guarded; // Current hat states

    int nbuttons _guarded;   // Number of buttons on the joystick
    bool *buttons _guarded; // Current button states

    int ntouchpads _guarded;                      // Number of touchpads on the joystick
    SDL_JoystickTouchpadInfo *touchpads _guarded; // Current touchpad states

    int nsensors _guarded; // Number of sensors on the joystick
    int nsensors_enabled _guarded;
    SDL_JoystickSensorInfo *sensors _guarded;

    Uint16 low_frequency_rumble _guarded;
    Uint16 high_frequency_rumble _guarded;
    Uint64 rumble_expiration _guarded;
    Uint64 rumble_resend _guarded;

    Uint16 left_trigger_rumble _guarded;
    Uint16 right_trigger_rumble _guarded;
    Uint64 trigger_rumble_expiration _guarded;
    Uint64 trigger_rumble_resend _guarded;

    Uint8 led_red _guarded;
    Uint8 led_green _guarded;
    Uint8 led_blue _guarded;
    Uint64 led_expiration _guarded;

    bool attached _guarded;
    SDL_JoystickConnectionState connection_state _guarded;
    SDL_PowerState battery_state _guarded;
    int battery_percent _guarded;

    bool delayed_guide_button _guarded;      // true if this device has the guide button event delayed

    SDL_SensorID accel_sensor _guarded;
    SDL_Sensor *accel _guarded;
    SDL_SensorID gyro_sensor _guarded;
    SDL_Sensor *gyro _guarded;
    float sensor_transform[3][3] _guarded;

    Uint64 update_complete _guarded;

    struct SDL_JoystickDriver *driver _guarded;

    struct joystick_hwdata *hwdata _guarded; // Driver dependent information

    SDL_PropertiesID props _guarded;

    int ref_count _guarded; // Reference count for multiple opens

    struct SDL_Joystick *next _guarded; // pointer to next joystick we have allocated
};

#undef _guarded

// Device bus definitions
#define SDL_HARDWARE_BUS_UNKNOWN   0x00
#define SDL_HARDWARE_BUS_USB       0x03
#define SDL_HARDWARE_BUS_BLUETOOTH 0x05
#define SDL_HARDWARE_BUS_VIRTUAL   0xFF

// Macro to combine a USB vendor ID and product ID into a single Uint32 value
#define MAKE_VIDPID(VID, PID) (((Uint32)(VID)) << 16 | (PID))

typedef struct SDL_JoystickDriver
{
    /* Function to scan the system for joysticks.
     * Joystick 0 should be the system default joystick.
     * This function should return 0, or -1 on an unrecoverable error.
     */
    bool (*Init)(void);

    // Function to return the number of joystick devices plugged in right now
    int (*GetCount)(void);

    // Function to cause any queued joystick insertions to be processed
    void (*Detect)(void);

    // Function to determine whether a device is currently detected by this driver
    bool (*IsDevicePresent)(Uint16 vendor_id, Uint16 product_id, Uint16 version, const char *name);

    // Function to get the device-dependent name of a joystick
    const char *(*GetDeviceName)(int device_index);

    // Function to get the device-dependent path of a joystick
    const char *(*GetDevicePath)(int device_index);

    // Function to get the Steam virtual gamepad slot of a joystick
    int (*GetDeviceSteamVirtualGamepadSlot)(int device_index);

    // Function to get the player index of a joystick
    int (*GetDevicePlayerIndex)(int device_index);

    // Function to set the player index of a joystick
    void (*SetDevicePlayerIndex)(int device_index, int player_index);

    // Function to return the stable GUID for a plugged in device
    SDL_GUID (*GetDeviceGUID)(int device_index);

    // Function to get the current instance id of the joystick located at device_index
    SDL_JoystickID (*GetDeviceInstanceID)(int device_index);

    /* Function to open a joystick for use.
       The joystick to open is specified by the device index.
       This should fill the nbuttons and naxes fields of the joystick structure.
       It returns 0, or -1 if there is an error.
     */
    bool (*Open)(SDL_Joystick *joystick, int device_index);

    // Rumble functionality
    bool (*Rumble)(SDL_Joystick *joystick, Uint16 low_frequency_rumble, Uint16 high_frequency_rumble);
    bool (*RumbleTriggers)(SDL_Joystick *joystick, Uint16 left_rumble, Uint16 right_rumble);

    // LED functionality
    bool (*SetLED)(SDL_Joystick *joystick, Uint8 red, Uint8 green, Uint8 blue);

    // General effects
    bool (*SendEffect)(SDL_Joystick *joystick, const void *data, int size);

    // Sensor functionality
    bool (*SetSensorsEnabled)(SDL_Joystick *joystick, bool enabled);

    /* Function to update the state of a joystick - called as a device poll.
     * This function shouldn't update the joystick structure directly,
     * but instead should call SDL_PrivateJoystick*() to deliver events
     * and update joystick device state.
     */
    void (*Update)(SDL_Joystick *joystick);

    // Function to close a joystick after use
    void (*Close)(SDL_Joystick *joystick);

    // Function to perform any system-specific joystick related cleanup
    void (*Quit)(void);

    // Function to get the autodetected controller mapping; returns false if there isn't any.
    bool (*GetGamepadMapping)(int device_index, SDL_GamepadMapping *out);

} SDL_JoystickDriver;

// Windows and Mac OSX has a limit of MAX_DWORD / 1000, Linux kernel has a limit of 0xFFFF
#define SDL_MAX_RUMBLE_DURATION_MS 0xFFFF

/* Dualshock4 only rumbles for about 5 seconds max, resend rumble command every 2 seconds
 * to make long rumble work. */
#define SDL_RUMBLE_RESEND_MS 2000

#define SDL_LED_MIN_REPEAT_MS 5000

// The available joystick drivers
extern SDL_JoystickDriver SDL_PRIVATE_JoystickDriver;
extern SDL_JoystickDriver SDL_ANDROID_JoystickDriver;
extern SDL_JoystickDriver SDL_BSD_JoystickDriver;
extern SDL_JoystickDriver SDL_DARWIN_JoystickDriver;
extern SDL_JoystickDriver SDL_DUMMY_JoystickDriver;
extern SDL_JoystickDriver SDL_EMSCRIPTEN_JoystickDriver;
extern SDL_JoystickDriver SDL_HAIKU_JoystickDriver;
extern SDL_JoystickDriver SDL_HIDAPI_JoystickDriver;
extern SDL_JoystickDriver SDL_RAWINPUT_JoystickDriver;
extern SDL_JoystickDriver SDL_IOS_JoystickDriver;
extern SDL_JoystickDriver SDL_LINUX_JoystickDriver;
extern SDL_JoystickDriver SDL_VIRTUAL_JoystickDriver;
extern SDL_JoystickDriver SDL_WGI_JoystickDriver;
extern SDL_JoystickDriver SDL_WINDOWS_JoystickDriver;
extern SDL_JoystickDriver SDL_WINMM_JoystickDriver;
extern SDL_JoystickDriver SDL_PS2_JoystickDriver;
extern SDL_JoystickDriver SDL_PSP_JoystickDriver;
extern SDL_JoystickDriver SDL_VITA_JoystickDriver;
extern SDL_JoystickDriver SDL_N3DS_JoystickDriver;
extern SDL_JoystickDriver SDL_GAMEINPUT_JoystickDriver;

// Ends C function definitions when using C++
#ifdef __cplusplus
}
#endif

#endif // SDL_sysjoystick_h_
