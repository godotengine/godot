/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2026 Sam Lantinga <slouken@libsdl.org>

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

#ifdef SDL_JOYSTICK_HIDAPI

#include "../../SDL_hints_c.h"
#include "../../misc/SDL_libusb.h"
#include "../SDL_sysjoystick.h"
#include "SDL_hidapijoystick_c.h"
#include "SDL_hidapi_rumble.h"
#include "../../hidapi/SDL_hidapi_c.h"

#ifdef SDL_JOYSTICK_HIDAPI_GAMECUBE

// Define this if you want to log all packets from the controller
#if 0
#define DEBUG_GAMECUBE_PROTOCOL
#endif

#define MAX_CONTROLLERS 4
#define PC_RUMBLE_REFRESH 16

typedef struct
{
    bool pc_mode;
    SDL_JoystickID joysticks[MAX_CONTROLLERS];
    Uint8 wireless[MAX_CONTROLLERS];
    Uint8 min_axis[MAX_CONTROLLERS * SDL_GAMEPAD_AXIS_COUNT];
    Uint8 max_axis[MAX_CONTROLLERS * SDL_GAMEPAD_AXIS_COUNT];
    Uint8 rumbleAllowed[MAX_CONTROLLERS];
    Uint8 rumble[1 + MAX_CONTROLLERS];
    // Without this variable, hid_write starts to lag a TON
    bool rumbleUpdate;
    bool useRumbleBrake;
    bool rumbleActive;
    Uint64 pc_rumble_sent;
} SDL_DriverGameCube_Context;

static void HIDAPI_DriverGameCube_RegisterHints(SDL_HintCallback callback, void *userdata)
{
    SDL_AddHintCallback(SDL_HINT_JOYSTICK_HIDAPI_GAMECUBE, callback, userdata);
}

static void HIDAPI_DriverGameCube_UnregisterHints(SDL_HintCallback callback, void *userdata)
{
    SDL_RemoveHintCallback(SDL_HINT_JOYSTICK_HIDAPI_GAMECUBE, callback, userdata);
}

static bool HIDAPI_DriverGameCube_IsEnabled(void)
{
    return SDL_GetHintBoolean(SDL_HINT_JOYSTICK_HIDAPI_GAMECUBE,
                              SDL_GetHintBoolean(SDL_HINT_JOYSTICK_HIDAPI,
                                                 SDL_HIDAPI_DEFAULT));
}

static bool HIDAPI_DriverGameCube_IsSupportedDevice(SDL_HIDAPI_Device *device, const char *name, SDL_GamepadType type, Uint16 vendor_id, Uint16 product_id, Uint16 version, int interface_number, int interface_class, int interface_subclass, int interface_protocol)
{
    if (vendor_id == USB_VENDOR_NINTENDO && product_id == USB_PRODUCT_NINTENDO_GAMECUBE_ADAPTER) {
        // Nintendo Co., Ltd.  Wii U GameCube Controller Adapter
        return true;
    }
    if (vendor_id == USB_VENDOR_DRAGONRISE &&
        (product_id == USB_PRODUCT_EVORETRO_GAMECUBE_ADAPTER1 ||
         product_id == USB_PRODUCT_EVORETRO_GAMECUBE_ADAPTER2 ||
         product_id == USB_PRODUCT_EVORETRO_GAMECUBE_ADAPTER3)) {
        // EVORETRO GameCube Controller Adapter
        return true;
    }
    return false;
}

static void ResetAxisRange(SDL_DriverGameCube_Context *ctx, int joystick_index)
{
    SDL_memset(&ctx->min_axis[joystick_index * SDL_GAMEPAD_AXIS_COUNT], 128 - 88, SDL_GAMEPAD_AXIS_COUNT);
    SDL_memset(&ctx->max_axis[joystick_index * SDL_GAMEPAD_AXIS_COUNT], 128 + 88, SDL_GAMEPAD_AXIS_COUNT);

    // Trigger axes may have a higher resting value
    ctx->min_axis[joystick_index * SDL_GAMEPAD_AXIS_COUNT + SDL_GAMEPAD_AXIS_LEFT_TRIGGER] = 40;
    ctx->min_axis[joystick_index * SDL_GAMEPAD_AXIS_COUNT + SDL_GAMEPAD_AXIS_RIGHT_TRIGGER] = 40;
}

static void SDLCALL SDL_JoystickGameCubeRumbleBrakeHintChanged(void *userdata, const char *name, const char *oldValue, const char *hint)
{
    if (hint) {
        SDL_DriverGameCube_Context *ctx = (SDL_DriverGameCube_Context *)userdata;
        ctx->useRumbleBrake = SDL_GetStringBoolean(hint, false);
    }
}

static bool HIDAPI_DriverGameCube_EnableAdapter(SDL_HIDAPI_Device *device)
{
#ifdef HAVE_LIBUSB
    // Need to close the device while sending USB commands to it
    SDL_hid_close(device->dev);

    // This is needed to enable input for Nyko and EVORETRO GameCube adapters
    SDL_LibUSBContext *libusb_ctx;
    if (SDL_InitLibUSB(&libusb_ctx)) {
        libusb_context *context = NULL;
        libusb_device **devs = NULL;
        libusb_device_handle *handle = NULL;
        struct libusb_device_descriptor desc;
        ssize_t i, num_devs;
        bool kernel_detached = false;

        if (libusb_ctx->init(&context) == 0) {
            num_devs = libusb_ctx->get_device_list(context, &devs);
            for (i = 0; i < num_devs; ++i) {
                if (libusb_ctx->get_device_descriptor(devs[i], &desc) != 0) {
                    continue;
                }

                if (desc.idVendor != USB_VENDOR_NINTENDO ||
                    desc.idProduct != USB_PRODUCT_NINTENDO_GAMECUBE_ADAPTER) {
                    continue;
                }

                if (libusb_ctx->open(devs[i], &handle) != 0) {
                    continue;
                }

                if (libusb_ctx->kernel_driver_active(handle, 0)) {
                    if (libusb_ctx->detach_kernel_driver(handle, 0) == 0) {
                        kernel_detached = true;
                    }
                }

                if (libusb_ctx->claim_interface(handle, 0) == 0) {
                    libusb_ctx->control_transfer(handle, 0x21, 11, 0x0001, 0, NULL, 0, 1000);
                    libusb_ctx->release_interface(handle, 0);
                }

                if (kernel_detached) {
                    libusb_ctx->attach_kernel_driver(handle, 0);
                }

                libusb_ctx->close(handle);
            }

            libusb_ctx->free_device_list(devs, 1);

            libusb_ctx->exit(context);
        }
        SDL_QuitLibUSB();
    }

    // Reopen the device now that we're done
    device->dev = SDL_hid_open_path(device->path);
    if (!device->dev) {
        return false;
    }
#endif // HAVE_LIBUSB

    Uint8 initMagic = 0x13;
    if (SDL_hid_write(device->dev, &initMagic, sizeof(initMagic)) != sizeof(initMagic)) {
        SDL_LogDebug(SDL_LOG_CATEGORY_INPUT,
                     "HIDAPI_DriverGameCube_InitDevice(): Couldn't initialize WUP-028");
        return false;
    }

    // Wait for the adapter to initialize
    SDL_Delay(10);

    return true;
}

static bool HIDAPI_DriverGameCube_InitDevice(SDL_HIDAPI_Device *device)
{
    SDL_DriverGameCube_Context *ctx;
    Uint8 packet[37];
    Uint8 *curSlot;
    Uint8 i;
    int size;
    Uint8 rumbleMagic = 0x11;

    ctx = (SDL_DriverGameCube_Context *)SDL_calloc(1, sizeof(*ctx));
    if (!ctx) {
        return false;
    }
    device->context = ctx;

    ctx->rumble[0] = rumbleMagic;

    if (device->vendor_id != USB_VENDOR_NINTENDO) {
        ctx->pc_mode = true;
    }

    if (ctx->pc_mode) {
        // Check to see if this firmware supports rumble
        bool rumbleAllowed = false;
        if ((size = SDL_hid_read_timeout(device->dev, packet, sizeof(packet), 0)) > 0) {
#ifdef DEBUG_GAMECUBE_PROTOCOL
            HIDAPI_DumpPacket("Nintendo GameCube packet: size = %d", packet, size);
#endif
            if (size == 9) {
                // This is firmware version 0x7 or newer
                // Rumble is supported if the second USB cable is plugged in
                rumbleAllowed = true;
            }
        }
#ifdef SDL_PLATFORM_WIN32
        // We get a separate device for each slot
        ctx->rumbleAllowed[0] = rumbleAllowed;
        ResetAxisRange(ctx, 0);
        HIDAPI_JoystickConnected(device, &ctx->joysticks[0]);
#else
        for (i = 0; i < MAX_CONTROLLERS; ++i) {
            ctx->rumbleAllowed[i] = rumbleAllowed;
            ResetAxisRange(ctx, i);
            HIDAPI_JoystickConnected(device, &ctx->joysticks[i]);
        }
#endif
    } else {
        if (!HIDAPI_DriverGameCube_EnableAdapter(device)) {
            return false;
        }

        // Add all the applicable joysticks
        while ((size = SDL_hid_read_timeout(device->dev, packet, sizeof(packet), 0)) > 0) {
#ifdef DEBUG_GAMECUBE_PROTOCOL
            HIDAPI_DumpPacket("Nintendo GameCube packet: size = %d", packet, size);
#endif
            if (size < 37 || packet[0] != 0x21) {
                continue; // Nothing to do yet...?
            }

            // Go through all 4 slots
            curSlot = packet + 1;
            for (i = 0; i < MAX_CONTROLLERS; i += 1, curSlot += 9) {
                ctx->wireless[i] = (curSlot[0] & 0x20) != 0;

                // Only allow rumble if the adapter's second USB cable is connected
                ctx->rumbleAllowed[i] = (curSlot[0] & 0x04) && !ctx->wireless[i];

                if (curSlot[0] & 0x30) { // 0x10 - Wired, 0x20 - Wireless
                    if (ctx->joysticks[i] == 0) {
                        ResetAxisRange(ctx, i);
                        HIDAPI_JoystickConnected(device, &ctx->joysticks[i]);
                    }
                } else {
                    if (ctx->joysticks[i] != 0) {
                        HIDAPI_JoystickDisconnected(device, ctx->joysticks[i]);
                        ctx->joysticks[i] = 0;
                    }
                    continue;
                }
            }
        }
    }

    SDL_AddHintCallback(SDL_HINT_JOYSTICK_HIDAPI_GAMECUBE_RUMBLE_BRAKE,
                        SDL_JoystickGameCubeRumbleBrakeHintChanged, ctx);

    HIDAPI_SetDeviceName(device, "Nintendo GameCube Controller");

    return true;
}

static int HIDAPI_DriverGameCube_GetDevicePlayerIndex(SDL_HIDAPI_Device *device, SDL_JoystickID instance_id)
{
    SDL_DriverGameCube_Context *ctx = (SDL_DriverGameCube_Context *)device->context;
    Uint8 i;

    for (i = 0; i < 4; ++i) {
        if (instance_id == ctx->joysticks[i]) {
            return i;
        }
    }
    return -1;
}

static void HIDAPI_DriverGameCube_SetDevicePlayerIndex(SDL_HIDAPI_Device *device, SDL_JoystickID instance_id, int player_index)
{
}

static void HIDAPI_DriverGameCube_HandleJoystickPacket(SDL_HIDAPI_Device *device, SDL_DriverGameCube_Context *ctx, Uint8 slot, const Uint8 *packet, bool invert_c_stick)
{
    SDL_Joystick *joystick;
    Uint8 i;
    Uint8 v;
    Sint16 axis_value;
    Uint64 timestamp = SDL_GetTicksNS();

#ifdef SDL_PLATFORM_WIN32
    // We get a separate device for each slot
    i = 0;
#else
    i = slot;
    if (i >= MAX_CONTROLLERS) {
        // Invalid packet?
        return;
    }
#endif

    joystick = SDL_GetJoystickFromID(ctx->joysticks[i]);
    if (!joystick) {
        // Hasn't been opened yet, skip
        return;
    }

#define READ_BUTTON(off, flag, button) \
    SDL_SendJoystickButton(            \
        timestamp,                     \
        joystick,                      \
        button,                        \
        ((packet[off] & flag) != 0));
    READ_BUTTON(0, 0x02, 0) // A
    READ_BUTTON(0, 0x04, 1) // B
    READ_BUTTON(0, 0x08, 3) // Y
    READ_BUTTON(0, 0x01, 2) // X
    READ_BUTTON(1, 0x80, 4) // DPAD_LEFT
    READ_BUTTON(1, 0x20, 5) // DPAD_RIGHT
    READ_BUTTON(1, 0x40, 6) // DPAD_DOWN
    READ_BUTTON(1, 0x10, 7) // DPAD_UP
    READ_BUTTON(1, 0x02, 8) // START
    READ_BUTTON(0, 0x80, 9) // RIGHTSHOULDER
    /* These two buttons are for the bottoms of the analog triggers.
     * More than likely, you're going to want to read the axes instead!
     * -flibit
     */
    READ_BUTTON(0, 0x20, 10) // TRIGGERRIGHT
    READ_BUTTON(0, 0x10, 11) // TRIGGERLEFT
#undef READ_BUTTON

#define READ_AXIS(off, axis, invert)                                                                                                                                             \
    v = (invert) ? (0xff - packet[off]) : packet[off];                                                                                                                           \
    if (v < ctx->min_axis[i * SDL_GAMEPAD_AXIS_COUNT + axis])                                                                                                                    \
        ctx->min_axis[i * SDL_GAMEPAD_AXIS_COUNT + axis] = v;                                                                                                                    \
    if (v > ctx->max_axis[i * SDL_GAMEPAD_AXIS_COUNT + axis])                                                                                                                    \
        ctx->max_axis[i * SDL_GAMEPAD_AXIS_COUNT + axis] = v;                                                                                                                    \
    axis_value = (Sint16)HIDAPI_RemapVal(v, ctx->min_axis[i * SDL_GAMEPAD_AXIS_COUNT + axis], ctx->max_axis[i * SDL_GAMEPAD_AXIS_COUNT + axis], SDL_MIN_SINT16, SDL_MAX_SINT16); \
    SDL_SendJoystickAxis(                                                                                                                                                        \
        timestamp,                                                                                                                                                               \
        joystick,                                                                                                                                                                \
        axis, axis_value);
    READ_AXIS(2, SDL_GAMEPAD_AXIS_LEFTX, 0)
    READ_AXIS(3, SDL_GAMEPAD_AXIS_LEFTY, 1)
    READ_AXIS(5, SDL_GAMEPAD_AXIS_RIGHTX, invert_c_stick ? 1 : 0)
    READ_AXIS(4, SDL_GAMEPAD_AXIS_RIGHTY, invert_c_stick ? 0 : 1)
    READ_AXIS(6, SDL_GAMEPAD_AXIS_LEFT_TRIGGER, 0)
    READ_AXIS(7, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER, 0)
#undef READ_AXIS
}

static void HIDAPI_DriverGameCube_HandleNintendoPacket(SDL_HIDAPI_Device *device, SDL_DriverGameCube_Context *ctx, Uint8 *packet, int size)
{
    SDL_Joystick *joystick;
    Uint8 *curSlot;
    Uint8 i;
    Sint16 axis_value;
    Uint64 timestamp = SDL_GetTicksNS();

    if (size < 37 || packet[0] != 0x21) {
        return; // Nothing to do right now...?
    }

    // Go through all 4 slots
    curSlot = packet + 1;
    for (i = 0; i < MAX_CONTROLLERS; i += 1, curSlot += 9) {
        ctx->wireless[i] = (curSlot[0] & 0x20) != 0;

        // Only allow rumble if the adapter's second USB cable is connected
        ctx->rumbleAllowed[i] = (curSlot[0] & 0x04) && !ctx->wireless[i];

        if (curSlot[0] & 0x30) { // 0x10 - Wired, 0x20 - Wireless
            if (ctx->joysticks[i] == 0) {
                ResetAxisRange(ctx, i);
                HIDAPI_JoystickConnected(device, &ctx->joysticks[i]);
            }
            joystick = SDL_GetJoystickFromID(ctx->joysticks[i]);

            // Hasn't been opened yet, skip
            if (!joystick) {
                continue;
            }
        } else {
            if (ctx->joysticks[i] != 0) {
                HIDAPI_JoystickDisconnected(device, ctx->joysticks[i]);
                ctx->joysticks[i] = 0;
            }
            continue;
        }

#define READ_BUTTON(off, flag, button)  \
    SDL_SendJoystickButton(             \
        timestamp,                      \
        joystick,                       \
        button,                         \
        ((curSlot[off] & flag) != 0));
        READ_BUTTON(1, 0x01, 0) // A
        READ_BUTTON(1, 0x02, 1) // B
        READ_BUTTON(1, 0x04, 2) // X
        READ_BUTTON(1, 0x08, 3) // Y
        READ_BUTTON(1, 0x10, 4) // DPAD_LEFT
        READ_BUTTON(1, 0x20, 5) // DPAD_RIGHT
        READ_BUTTON(1, 0x40, 6) // DPAD_DOWN
        READ_BUTTON(1, 0x80, 7) // DPAD_UP
        READ_BUTTON(2, 0x01, 8) // START
        READ_BUTTON(2, 0x02, 9) // RIGHTSHOULDER
        /* These two buttons are for the bottoms of the analog triggers.
         * More than likely, you're going to want to read the axes instead!
         * -flibit
         */
        READ_BUTTON(2, 0x04, 10) // TRIGGERRIGHT
        READ_BUTTON(2, 0x08, 11) // TRIGGERLEFT
#undef READ_BUTTON

#define READ_AXIS(off, axis)                                                                \
    if (curSlot[off] < ctx->min_axis[i * SDL_GAMEPAD_AXIS_COUNT + axis])                   \
        ctx->min_axis[i * SDL_GAMEPAD_AXIS_COUNT + axis] = curSlot[off];                   \
    if (curSlot[off] > ctx->max_axis[i * SDL_GAMEPAD_AXIS_COUNT + axis])                   \
        ctx->max_axis[i * SDL_GAMEPAD_AXIS_COUNT + axis] = curSlot[off];                   \
    axis_value = (Sint16)HIDAPI_RemapVal(curSlot[off], ctx->min_axis[i * SDL_GAMEPAD_AXIS_COUNT + axis], ctx->max_axis[i * SDL_GAMEPAD_AXIS_COUNT + axis], SDL_MIN_SINT16, SDL_MAX_SINT16); \
    SDL_SendJoystickAxis(                                                                \
        timestamp,                                                                          \
        joystick,                                                                           \
        axis, axis_value);
        READ_AXIS(3, SDL_GAMEPAD_AXIS_LEFTX)
        READ_AXIS(4, SDL_GAMEPAD_AXIS_LEFTY)
        READ_AXIS(5, SDL_GAMEPAD_AXIS_RIGHTX)
        READ_AXIS(6, SDL_GAMEPAD_AXIS_RIGHTY)
        READ_AXIS(7, SDL_GAMEPAD_AXIS_LEFT_TRIGGER)
        READ_AXIS(8, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER)
#undef READ_AXIS
    }
}

static void HIDAPI_DriverGameCube_SendPCRumble(SDL_HIDAPI_Device *device, SDL_DriverGameCube_Context *ctx)
{
    Uint8 rumblepkt[3] = { 0x00, 0x00, 0x00 };
    rumblepkt[1] = rumblepkt[2] = ctx->rumbleActive ? 0xFF : 0x00;
    SDL_HIDAPI_SendRumble(device, rumblepkt, sizeof(rumblepkt));
    ctx->pc_rumble_sent = SDL_GetTicks();
}

static bool HIDAPI_DriverGameCube_UpdateDevice(SDL_HIDAPI_Device *device)
{
    SDL_DriverGameCube_Context *ctx = (SDL_DriverGameCube_Context *)device->context;
    Uint8 packet[USB_PACKET_LENGTH];
    int size;

    // Read input packet
    while ((size = SDL_hid_read_timeout(device->dev, packet, sizeof(packet), 0)) > 0) {
#ifdef DEBUG_GAMECUBE_PROTOCOL
        HIDAPI_DumpPacket("Nintendo GameCube packet: size = %d", packet, size);
#endif
        if (ctx->pc_mode) {
            if (size == 10) {
                // This is the older firmware
                // The first byte is the index of the connected controller
                // The C stick has an inverted value range compared to the left stick
                HIDAPI_DriverGameCube_HandleJoystickPacket(device, ctx, packet[0] - 1, &packet[1], true);
            } else if (size == 9) {
                // This is the newer firmware (version 0x7)
                // The C stick has the same value range compared to the left stick
                HIDAPI_DriverGameCube_HandleJoystickPacket(device, ctx, 0, packet, false);
            } else {
                // How do we handle this packet?
            }
        } else {
            HIDAPI_DriverGameCube_HandleNintendoPacket(device, ctx, packet, size);
        }
    }

    // PC_Mode rumble needs constant packets in order to keep rumble going
    if (ctx->pc_mode && ctx->rumbleActive && SDL_GetTicks() >= (ctx->pc_rumble_sent + PC_RUMBLE_REFRESH)) {
        HIDAPI_DriverGameCube_SendPCRumble(device, ctx);
    }

    // Write rumble packet
    if (ctx->rumbleUpdate) {
        SDL_HIDAPI_SendRumble(device, ctx->rumble, sizeof(ctx->rumble));
        ctx->rumbleUpdate = false;
    }

    // If we got here, nothing bad happened!
    return true;
}

static bool HIDAPI_DriverGameCube_OpenJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
    SDL_DriverGameCube_Context *ctx = (SDL_DriverGameCube_Context *)device->context;
    Uint8 i;

    SDL_AssertJoysticksLocked();

    for (i = 0; i < MAX_CONTROLLERS; i += 1) {
        if (joystick->instance_id == ctx->joysticks[i]) {
            joystick->nbuttons = 12;
            joystick->naxes = SDL_GAMEPAD_AXIS_COUNT;
            if (ctx->wireless[i]) {
                joystick->connection_state = SDL_JOYSTICK_CONNECTION_WIRELESS;
            } else {
                joystick->connection_state = SDL_JOYSTICK_CONNECTION_WIRED;
            }
            return true;
        }
    }
    return false; // Should never get here!
}

static bool HIDAPI_DriverGameCube_RumbleJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint16 low_frequency_rumble, Uint16 high_frequency_rumble)
{
    SDL_DriverGameCube_Context *ctx = (SDL_DriverGameCube_Context *)device->context;
    Uint8 i, val;

    SDL_AssertJoysticksLocked();

    if (ctx->pc_mode) {
        for (i = 0; i < MAX_CONTROLLERS; i += 1) {
            if (joystick->instance_id == ctx->joysticks[i]) {
                if (!ctx->rumbleAllowed[i]) {
                    return SDL_SetError("Rumble is disabled because the adapter is not at least of firmware 0x7");
                }
                bool shouldrumble = false, coast = false;
                if (ctx->useRumbleBrake) {
                    coast = (low_frequency_rumble == 0 && high_frequency_rumble > 0);
                }
                shouldrumble = (low_frequency_rumble > 0 || high_frequency_rumble > 0);
                if (coast) {
                    ctx->rumbleActive = false;
                } else if (shouldrumble != ctx->rumbleActive) {
                    ctx->rumbleActive = shouldrumble;
                    HIDAPI_DriverGameCube_SendPCRumble(device, ctx);
                }
                return true;
            }
        }
    } else {
        for (i = 0; i < MAX_CONTROLLERS; i += 1) {
            if (joystick->instance_id == ctx->joysticks[i]) {
                if (ctx->wireless[i]) {
                    return SDL_SetError("Nintendo GameCube WaveBird controllers do not support rumble");
                }
                if (!ctx->rumbleAllowed[i]) {
                    return SDL_SetError("Second USB cable for WUP-028 not connected");
                }
                if (ctx->useRumbleBrake) {
                    if (low_frequency_rumble == 0 && high_frequency_rumble > 0) {
                        val = 0; // if only low is 0 we want to do a regular stop
                    } else if (low_frequency_rumble == 0 && high_frequency_rumble == 0) {
                        val = 2; // if both frequencies are 0 we want to do a hard stop
                    } else {
                        val = 1; // normal rumble
                    }
                } else {
                    val = (low_frequency_rumble > 0 || high_frequency_rumble > 0);
                }
                if (val != ctx->rumble[i + 1]) {
                    ctx->rumble[i + 1] = val;
                    ctx->rumbleUpdate = true;
                }
                return true;
            }
        }
    }

    // Should never get here!
    return SDL_SetError("Couldn't find joystick");
}

static bool HIDAPI_DriverGameCube_RumbleJoystickTriggers(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint16 left_rumble, Uint16 right_rumble)
{
    return SDL_Unsupported();
}

static Uint32 HIDAPI_DriverGameCube_GetJoystickCapabilities(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
    SDL_DriverGameCube_Context *ctx = (SDL_DriverGameCube_Context *)device->context;
    Uint32 result = 0;

    SDL_AssertJoysticksLocked();
    Uint8 i;
    if (ctx->pc_mode) {
        for (i = 0; i < MAX_CONTROLLERS; i += 1) {
            if (joystick->instance_id == ctx->joysticks[i]) {
                if (ctx->rumbleAllowed[i]) {
                    result |= SDL_JOYSTICK_CAP_RUMBLE;
                    break;
                }
            }
        }
    } else {
        for (i = 0; i < MAX_CONTROLLERS; i += 1) {
            if (joystick->instance_id == ctx->joysticks[i]) {
                if (!ctx->wireless[i] && ctx->rumbleAllowed[i]) {
                    result |= SDL_JOYSTICK_CAP_RUMBLE;
                    break;
                }
            }
        }
    }

    return result;
}

static bool HIDAPI_DriverGameCube_SetJoystickLED(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint8 red, Uint8 green, Uint8 blue)
{
    return SDL_Unsupported();
}

static bool HIDAPI_DriverGameCube_SendJoystickEffect(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, const void *data, int size)
{
    return SDL_Unsupported();
}

static bool HIDAPI_DriverGameCube_SetJoystickSensorsEnabled(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, bool enabled)
{
    return SDL_Unsupported();
}

static void HIDAPI_DriverGameCube_CloseJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
    SDL_DriverGameCube_Context *ctx = (SDL_DriverGameCube_Context *)device->context;

    // Stop rumble activity
    if (ctx->rumbleUpdate) {
        SDL_HIDAPI_SendRumble(device, ctx->rumble, sizeof(ctx->rumble));
        ctx->rumbleUpdate = false;
    }

    if (ctx->pc_mode && ctx->rumbleActive) {
        ctx->rumbleActive = false;
        HIDAPI_DriverGameCube_SendPCRumble(device, ctx);
    }
}

static void HIDAPI_DriverGameCube_FreeDevice(SDL_HIDAPI_Device *device)
{
    SDL_DriverGameCube_Context *ctx = (SDL_DriverGameCube_Context *)device->context;

    SDL_RemoveHintCallback(SDL_HINT_JOYSTICK_HIDAPI_GAMECUBE_RUMBLE_BRAKE,
                        SDL_JoystickGameCubeRumbleBrakeHintChanged, ctx);
}

SDL_HIDAPI_DeviceDriver SDL_HIDAPI_DriverGameCube = {
    SDL_HINT_JOYSTICK_HIDAPI_GAMECUBE,
    true,
    HIDAPI_DriverGameCube_RegisterHints,
    HIDAPI_DriverGameCube_UnregisterHints,
    HIDAPI_DriverGameCube_IsEnabled,
    HIDAPI_DriverGameCube_IsSupportedDevice,
    HIDAPI_DriverGameCube_InitDevice,
    HIDAPI_DriverGameCube_GetDevicePlayerIndex,
    HIDAPI_DriverGameCube_SetDevicePlayerIndex,
    HIDAPI_DriverGameCube_UpdateDevice,
    HIDAPI_DriverGameCube_OpenJoystick,
    HIDAPI_DriverGameCube_RumbleJoystick,
    HIDAPI_DriverGameCube_RumbleJoystickTriggers,
    HIDAPI_DriverGameCube_GetJoystickCapabilities,
    HIDAPI_DriverGameCube_SetJoystickLED,
    HIDAPI_DriverGameCube_SendJoystickEffect,
    HIDAPI_DriverGameCube_SetJoystickSensorsEnabled,
    HIDAPI_DriverGameCube_CloseJoystick,
    HIDAPI_DriverGameCube_FreeDevice,
};

#endif // SDL_JOYSTICK_HIDAPI_GAMECUBE

#endif // SDL_JOYSTICK_HIDAPI
