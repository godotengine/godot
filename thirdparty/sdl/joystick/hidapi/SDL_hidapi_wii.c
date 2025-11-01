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

#ifdef SDL_JOYSTICK_HIDAPI

#include "../../SDL_hints_c.h"
#include "../SDL_sysjoystick.h"
#include "SDL_hidapijoystick_c.h"
#include "SDL_hidapi_rumble.h"
#include "SDL_hidapi_nintendo.h"

#ifdef SDL_JOYSTICK_HIDAPI_WII

// Define this if you want to log all packets from the controller
// #define DEBUG_WII_PROTOCOL

#define ENABLE_CONTINUOUS_REPORTING true

#define INPUT_WAIT_TIMEOUT_MS      (3 * 1000)
#define MOTION_PLUS_UPDATE_TIME_MS (8 * 1000)
#define STATUS_UPDATE_TIME_MS      (15 * 60 * 1000)

#define WII_EXTENSION_NONE            0x2E2E
#define WII_EXTENSION_UNINITIALIZED   0xFFFF
#define WII_EXTENSION_NUNCHUK         0x0000
#define WII_EXTENSION_GAMEPAD         0x0101
#define WII_EXTENSION_WIIUPRO         0x0120
#define WII_EXTENSION_MOTIONPLUS_MASK 0xF0FF
#define WII_EXTENSION_MOTIONPLUS_ID   0x0005

#define WII_MOTIONPLUS_MODE_NONE     0x00
#define WII_MOTIONPLUS_MODE_STANDARD 0x04
#define WII_MOTIONPLUS_MODE_NUNCHUK  0x05
#define WII_MOTIONPLUS_MODE_GAMEPAD  0x07

typedef enum
{
    k_eWiiInputReportIDs_Status = 0x20,
    k_eWiiInputReportIDs_ReadMemory = 0x21,
    k_eWiiInputReportIDs_Acknowledge = 0x22,
    k_eWiiInputReportIDs_ButtonData0 = 0x30,
    k_eWiiInputReportIDs_ButtonData1 = 0x31,
    k_eWiiInputReportIDs_ButtonData2 = 0x32,
    k_eWiiInputReportIDs_ButtonData3 = 0x33,
    k_eWiiInputReportIDs_ButtonData4 = 0x34,
    k_eWiiInputReportIDs_ButtonData5 = 0x35,
    k_eWiiInputReportIDs_ButtonData6 = 0x36,
    k_eWiiInputReportIDs_ButtonData7 = 0x37,
    k_eWiiInputReportIDs_ButtonDataD = 0x3D,
    k_eWiiInputReportIDs_ButtonDataE = 0x3E,
    k_eWiiInputReportIDs_ButtonDataF = 0x3F,
} EWiiInputReportIDs;

typedef enum
{
    k_eWiiOutputReportIDs_Rumble = 0x10,
    k_eWiiOutputReportIDs_LEDs = 0x11,
    k_eWiiOutputReportIDs_DataReportingMode = 0x12,
    k_eWiiOutputReportIDs_IRCameraEnable = 0x13,
    k_eWiiOutputReportIDs_SpeakerEnable = 0x14,
    k_eWiiOutputReportIDs_StatusRequest = 0x15,
    k_eWiiOutputReportIDs_WriteMemory = 0x16,
    k_eWiiOutputReportIDs_ReadMemory = 0x17,
    k_eWiiOutputReportIDs_SpeakerData = 0x18,
    k_eWiiOutputReportIDs_SpeakerMute = 0x19,
    k_eWiiOutputReportIDs_IRCameraEnable2 = 0x1a,
} EWiiOutputReportIDs;

typedef enum
{
    k_eWiiPlayerLEDs_P1 = 0x10,
    k_eWiiPlayerLEDs_P2 = 0x20,
    k_eWiiPlayerLEDs_P3 = 0x40,
    k_eWiiPlayerLEDs_P4 = 0x80,
} EWiiPlayerLEDs;

typedef enum
{
    k_eWiiCommunicationState_None,                  // No special communications happening
    k_eWiiCommunicationState_CheckMotionPlusStage1, // Sent standard extension identify request
    k_eWiiCommunicationState_CheckMotionPlusStage2, // Sent Motion Plus extension identify request
} EWiiCommunicationState;

typedef enum
{
    k_eWiiButtons_A = SDL_GAMEPAD_BUTTON_MISC1,
    k_eWiiButtons_B,
    k_eWiiButtons_One,
    k_eWiiButtons_Two,
    k_eWiiButtons_Plus,
    k_eWiiButtons_Minus,
    k_eWiiButtons_Home,
    k_eWiiButtons_DPad_Up,
    k_eWiiButtons_DPad_Down,
    k_eWiiButtons_DPad_Left,
    k_eWiiButtons_DPad_Right,
    k_eWiiButtons_Max
} EWiiButtons;

#define k_unWiiPacketDataLength 22

typedef struct
{
    Uint8 rgucBaseButtons[2];
    Uint8 rgucAccelerometer[3];
    Uint8 rgucExtension[21];
    bool hasBaseButtons;
    bool hasAccelerometer;
    Uint8 ucNExtensionBytes;
} WiiButtonData;

typedef struct
{
    Uint16 min;
    Uint16 max;
    Uint16 center;
    Uint16 deadzone;
} StickCalibrationData;

typedef struct
{
    SDL_HIDAPI_Device *device;
    SDL_Joystick *joystick;
    Uint64 timestamp;
    EWiiCommunicationState m_eCommState;
    EWiiExtensionControllerType m_eExtensionControllerType;
    bool m_bPlayerLights;
    int m_nPlayerIndex;
    bool m_bRumbleActive;
    bool m_bMotionPlusPresent;
    Uint8 m_ucMotionPlusMode;
    bool m_bReportSensors;
    Uint8 m_rgucReadBuffer[k_unWiiPacketDataLength];
    Uint64 m_ulLastInput;
    Uint64 m_ulLastStatus;
    Uint64 m_ulNextMotionPlusCheck;
    bool m_bDisconnected;

    StickCalibrationData m_StickCalibrationData[6];
} SDL_DriverWii_Context;

static void HIDAPI_DriverWii_RegisterHints(SDL_HintCallback callback, void *userdata)
{
    SDL_AddHintCallback(SDL_HINT_JOYSTICK_HIDAPI_WII, callback, userdata);
}

static void HIDAPI_DriverWii_UnregisterHints(SDL_HintCallback callback, void *userdata)
{
    SDL_RemoveHintCallback(SDL_HINT_JOYSTICK_HIDAPI_WII, callback, userdata);
}

static bool HIDAPI_DriverWii_IsEnabled(void)
{
#if 1 // This doesn't work with the dolphinbar, so don't enable by default right now
    return SDL_GetHintBoolean(SDL_HINT_JOYSTICK_HIDAPI_WII, false);
#else
    return SDL_GetHintBoolean(SDL_HINT_JOYSTICK_HIDAPI_WII,
                              SDL_GetHintBoolean(SDL_HINT_JOYSTICK_HIDAPI,
                                                 SDL_HIDAPI_DEFAULT));
#endif
}

static bool HIDAPI_DriverWii_IsSupportedDevice(SDL_HIDAPI_Device *device, const char *name, SDL_GamepadType type, Uint16 vendor_id, Uint16 product_id, Uint16 version, int interface_number, int interface_class, int interface_subclass, int interface_protocol)
{
    if (vendor_id == USB_VENDOR_NINTENDO &&
        (product_id == USB_PRODUCT_NINTENDO_WII_REMOTE ||
         product_id == USB_PRODUCT_NINTENDO_WII_REMOTE2)) {
        return true;
    }
    return false;
}

static int ReadInput(SDL_DriverWii_Context *ctx)
{
    int size;

    // Make sure we don't try to read at the same time a write is happening
    if (SDL_GetAtomicInt(&ctx->device->rumble_pending) > 0) {
        return 0;
    }

    size = SDL_hid_read_timeout(ctx->device->dev, ctx->m_rgucReadBuffer, sizeof(ctx->m_rgucReadBuffer), 0);
#ifdef DEBUG_WII_PROTOCOL
    if (size > 0) {
        HIDAPI_DumpPacket("Wii packet: size = %d", ctx->m_rgucReadBuffer, size);
    }
#endif
    return size;
}

static bool WriteOutput(SDL_DriverWii_Context *ctx, const Uint8 *data, int size, bool sync)
{
#ifdef DEBUG_WII_PROTOCOL
    if (size > 0) {
        HIDAPI_DumpPacket("Wii write packet: size = %d", data, size);
    }
#endif
    if (sync) {
        return SDL_hid_write(ctx->device->dev, data, size) >= 0;
    } else {
        // Use the rumble thread for general asynchronous writes
        if (!SDL_HIDAPI_LockRumble()) {
            return false;
        }
        return SDL_HIDAPI_SendRumbleAndUnlock(ctx->device, data, size) >= 0;
    }
}

static bool ReadInputSync(SDL_DriverWii_Context *ctx, EWiiInputReportIDs expectedID, bool (*isMine)(const Uint8 *))
{
    Uint64 endTicks = SDL_GetTicks() + 250; // Seeing successful reads after about 200 ms

    int nRead = 0;
    while ((nRead = ReadInput(ctx)) != -1) {
        if (nRead > 0) {
            if (ctx->m_rgucReadBuffer[0] == expectedID && (!isMine || isMine(ctx->m_rgucReadBuffer))) {
                return true;
            }
        } else {
            if (SDL_GetTicks() >= endTicks) {
                break;
            }
            SDL_Delay(1);
        }
    }
    SDL_SetError("Read timed out");
    return false;
}

static bool IsWriteMemoryResponse(const Uint8 *data)
{
    return data[3] == k_eWiiOutputReportIDs_WriteMemory;
}

static bool WriteRegister(SDL_DriverWii_Context *ctx, Uint32 address, const Uint8 *data, int size, bool sync)
{
    Uint8 writeRequest[k_unWiiPacketDataLength];

    SDL_zeroa(writeRequest);
    writeRequest[0] = k_eWiiOutputReportIDs_WriteMemory;
    writeRequest[1] = (Uint8)(0x04 | (Uint8)ctx->m_bRumbleActive);
    writeRequest[2] = (address >> 16) & 0xff;
    writeRequest[3] = (address >> 8) & 0xff;
    writeRequest[4] = address & 0xff;
    writeRequest[5] = (Uint8)size;
    SDL_assert(size > 0 && size <= 16);
    SDL_memcpy(writeRequest + 6, data, size);

    if (!WriteOutput(ctx, writeRequest, sizeof(writeRequest), sync)) {
        return false;
    }
    if (sync) {
        // Wait for response
        if (!ReadInputSync(ctx, k_eWiiInputReportIDs_Acknowledge, IsWriteMemoryResponse)) {
            return false;
        }
        if (ctx->m_rgucReadBuffer[4]) {
            SDL_SetError("Write memory failed: %u", ctx->m_rgucReadBuffer[4]);
            return false;
        }
    }
    return true;
}

static bool ReadRegister(SDL_DriverWii_Context *ctx, Uint32 address, int size, bool sync)
{
    Uint8 readRequest[7];

    readRequest[0] = k_eWiiOutputReportIDs_ReadMemory;
    readRequest[1] = (Uint8)(0x04 | (Uint8)ctx->m_bRumbleActive);
    readRequest[2] = (address >> 16) & 0xff;
    readRequest[3] = (address >> 8) & 0xff;
    readRequest[4] = address & 0xff;
    readRequest[5] = (size >> 8) & 0xff;
    readRequest[6] = size & 0xff;

    SDL_assert(size > 0 && size <= 0xffff);

    if (!WriteOutput(ctx, readRequest, sizeof(readRequest), sync)) {
        return false;
    }
    if (sync) {
        SDL_assert(size <= 16); // Only waiting for one packet is supported right now
        // Wait for response
        if (!ReadInputSync(ctx, k_eWiiInputReportIDs_ReadMemory, NULL)) {
            return false;
        }
    }
    return true;
}

static bool SendExtensionIdentify(SDL_DriverWii_Context *ctx, bool sync)
{
    return ReadRegister(ctx, 0xA400FE, 2, sync);
}

static bool ParseExtensionIdentifyResponse(SDL_DriverWii_Context *ctx, Uint16 *extension)
{
    int i;

    if (ctx->m_rgucReadBuffer[0] != k_eWiiInputReportIDs_ReadMemory) {
        SDL_SetError("Unexpected extension response type");
        return false;
    }

    if (ctx->m_rgucReadBuffer[4] != 0x00 || ctx->m_rgucReadBuffer[5] != 0xFE) {
        SDL_SetError("Unexpected extension response address");
        return false;
    }

    if (ctx->m_rgucReadBuffer[3] != 0x10) {
        Uint8 error = (ctx->m_rgucReadBuffer[3] & 0xF);

        if (error == 7) {
            // The extension memory isn't mapped
            *extension = WII_EXTENSION_NONE;
            return true;
        }

        if (error) {
            SDL_SetError("Failed to read extension type: %u", error);
        } else {
            SDL_SetError("Unexpected read length when reading extension type: %d", (ctx->m_rgucReadBuffer[3] >> 4) + 1);
        }
        return false;
    }

    *extension = 0;
    for (i = 6; i < 8; i++) {
        *extension = *extension << 8 | ctx->m_rgucReadBuffer[i];
    }
    return true;
}

static EWiiExtensionControllerType GetExtensionType(Uint16 extension_id)
{
    switch (extension_id) {
    case WII_EXTENSION_NONE:
        return k_eWiiExtensionControllerType_None;
    case WII_EXTENSION_NUNCHUK:
        return k_eWiiExtensionControllerType_Nunchuk;
    case WII_EXTENSION_GAMEPAD:
        return k_eWiiExtensionControllerType_Gamepad;
    case WII_EXTENSION_WIIUPRO:
        return k_eWiiExtensionControllerType_WiiUPro;
    default:
        return k_eWiiExtensionControllerType_Unknown;
    }
}

static bool SendExtensionReset(SDL_DriverWii_Context *ctx, bool sync)
{
    bool result = true;
    {
        Uint8 data = 0x55;
        result = result && WriteRegister(ctx, 0xA400F0, &data, sizeof(data), sync);
    }
    // This write will fail if there is no extension connected, that's fine
    {
        Uint8 data = 0x00;
        (void)WriteRegister(ctx, 0xA400FB, &data, sizeof(data), sync);
    }
    return result;
}

static bool GetMotionPlusState(SDL_DriverWii_Context *ctx, bool *connected, Uint8 *mode)
{
    Uint16 extension;

    if (connected) {
        *connected = false;
    }
    if (mode) {
        *mode = 0;
    }

    if (ctx->m_eExtensionControllerType == k_eWiiExtensionControllerType_WiiUPro) {
        // The Wii U Pro controller never has the Motion Plus extension
        return true;
    }

    if (SendExtensionIdentify(ctx, true) &&
        ParseExtensionIdentifyResponse(ctx, &extension)) {
        if ((extension & WII_EXTENSION_MOTIONPLUS_MASK) == WII_EXTENSION_MOTIONPLUS_ID) {
            // Motion Plus is currently active
            if (connected) {
                *connected = true;
            }
            if (mode) {
                *mode = (extension >> 8);
            }
            return true;
        }
    }

    if (ReadRegister(ctx, 0xA600FE, 2, true) &&
        ParseExtensionIdentifyResponse(ctx, &extension)) {
        if ((extension & WII_EXTENSION_MOTIONPLUS_MASK) == WII_EXTENSION_MOTIONPLUS_ID) {
            // Motion Plus is currently connected
            if (connected) {
                *connected = true;
            }
        }
        return true;
    }

    // Failed to read the register or parse the response
    return false;
}

static bool NeedsPeriodicMotionPlusCheck(SDL_DriverWii_Context *ctx, bool status_update)
{
    if (ctx->m_eExtensionControllerType == k_eWiiExtensionControllerType_WiiUPro) {
        // The Wii U Pro controller never has the Motion Plus extension
        return false;
    }

    if (ctx->m_ucMotionPlusMode != WII_MOTIONPLUS_MODE_NONE && !status_update) {
        // We'll get a status update when Motion Plus is disconnected
        return false;
    }

    return true;
}

static void SchedulePeriodicMotionPlusCheck(SDL_DriverWii_Context *ctx)
{
    ctx->m_ulNextMotionPlusCheck = SDL_GetTicks() + MOTION_PLUS_UPDATE_TIME_MS;
}

static void CheckMotionPlusConnection(SDL_DriverWii_Context *ctx)
{
    SendExtensionIdentify(ctx, false);

    ctx->m_eCommState = k_eWiiCommunicationState_CheckMotionPlusStage1;
}

static void ActivateMotionPlusWithMode(SDL_DriverWii_Context *ctx, Uint8 mode)
{
#ifdef SDL_PLATFORM_LINUX
    /* Linux drivers maintain a lot of state around the Motion Plus
     * extension, so don't mess with it here.
     */
#else
    WriteRegister(ctx, 0xA600FE, &mode, sizeof(mode), true);

    ctx->m_ucMotionPlusMode = mode;
#endif // LINUX
}

static void ActivateMotionPlus(SDL_DriverWii_Context *ctx)
{
    Uint8 mode = WII_MOTIONPLUS_MODE_STANDARD;

    // Pick the pass-through mode based on the connected controller
    if (ctx->m_eExtensionControllerType == k_eWiiExtensionControllerType_Nunchuk) {
        mode = WII_MOTIONPLUS_MODE_NUNCHUK;
    } else if (ctx->m_eExtensionControllerType == k_eWiiExtensionControllerType_Gamepad) {
        mode = WII_MOTIONPLUS_MODE_GAMEPAD;
    }
    ActivateMotionPlusWithMode(ctx, mode);
}

static void DeactivateMotionPlus(SDL_DriverWii_Context *ctx)
{
    Uint8 data = 0x55;
    WriteRegister(ctx, 0xA400F0, &data, sizeof(data), true);

    // Wait for the deactivation status message
    ReadInputSync(ctx, k_eWiiInputReportIDs_Status, NULL);

    ctx->m_ucMotionPlusMode = WII_MOTIONPLUS_MODE_NONE;
}

static void UpdatePowerLevelWii(SDL_Joystick *joystick, Uint8 batteryLevelByte)
{
    int percent;
    if (batteryLevelByte > 178) {
        percent = 100;
    } else if (batteryLevelByte > 51) {
        percent = 70;
    } else if (batteryLevelByte > 13) {
        percent = 20;
    } else {
        percent = 5;
    }
    SDL_SendJoystickPowerInfo(joystick, SDL_POWERSTATE_ON_BATTERY, percent);
}

static void UpdatePowerLevelWiiU(SDL_Joystick *joystick, Uint8 extensionBatteryByte)
{
    bool charging = !(extensionBatteryByte & 0x08);
    bool pluggedIn = !(extensionBatteryByte & 0x04);
    Uint8 batteryLevel = extensionBatteryByte >> 4;

    if (pluggedIn) {
        joystick->connection_state = SDL_JOYSTICK_CONNECTION_WIRED;
    } else {
        joystick->connection_state = SDL_JOYSTICK_CONNECTION_WIRELESS;
    }

    /* Not sure if all Wii U Pro controllers act like this, but on mine
     * 4, 3, and 2 are held for about 20 hours each
     * 1 is held for about 6 hours
     * 0 is held for about 2 hours
     * No value above 4 has been observed.
     */
    SDL_PowerState state;
    int percent;
    if (charging) {
        state = SDL_POWERSTATE_CHARGING;
    } else if (pluggedIn) {
        state = SDL_POWERSTATE_CHARGED;
    } else {
        state = SDL_POWERSTATE_ON_BATTERY;
    }
    if (batteryLevel >= 4) {
        percent = 100;
    } else if (batteryLevel == 3) {
        percent = 70;
    } else if (batteryLevel == 2) {
        percent = 40;
    } else if (batteryLevel == 1) {
        percent = 10;
    } else {
        percent = 3;
    }
    SDL_SendJoystickPowerInfo(joystick, state, percent);
}

static EWiiInputReportIDs GetButtonPacketType(SDL_DriverWii_Context *ctx)
{
    switch (ctx->m_eExtensionControllerType) {
    case k_eWiiExtensionControllerType_WiiUPro:
        return k_eWiiInputReportIDs_ButtonDataD;
    case k_eWiiExtensionControllerType_Nunchuk:
    case k_eWiiExtensionControllerType_Gamepad:
        if (ctx->m_bReportSensors) {
            return k_eWiiInputReportIDs_ButtonData5;
        } else {
            return k_eWiiInputReportIDs_ButtonData2;
        }
    default:
        if (ctx->m_bReportSensors) {
            return k_eWiiInputReportIDs_ButtonData5;
        } else {
            return k_eWiiInputReportIDs_ButtonData0;
        }
    }
}

static bool RequestButtonPacketType(SDL_DriverWii_Context *ctx, EWiiInputReportIDs type)
{
    Uint8 data[3];
    Uint8 tt = (Uint8)ctx->m_bRumbleActive;

    // Continuous reporting off, tt & 4 == 0
    if (ENABLE_CONTINUOUS_REPORTING) {
        tt |= 4;
    }

    data[0] = k_eWiiOutputReportIDs_DataReportingMode;
    data[1] = tt;
    data[2] = type;
    return WriteOutput(ctx, data, sizeof(data), false);
}

static void ResetButtonPacketType(SDL_DriverWii_Context *ctx)
{
    RequestButtonPacketType(ctx, GetButtonPacketType(ctx));
}

static void InitStickCalibrationData(SDL_DriverWii_Context *ctx)
{
    int i;
    switch (ctx->m_eExtensionControllerType) {
    case k_eWiiExtensionControllerType_WiiUPro:
        for (i = 0; i < 4; i++) {
            ctx->m_StickCalibrationData[i].min = 1000;
            ctx->m_StickCalibrationData[i].max = 3000;
            ctx->m_StickCalibrationData[i].center = 0;
            ctx->m_StickCalibrationData[i].deadzone = 100;
        }
        break;
    case k_eWiiExtensionControllerType_Gamepad:
        for (i = 0; i < 4; i++) {
            ctx->m_StickCalibrationData[i].min = i < 2 ? 9 : 5;
            ctx->m_StickCalibrationData[i].max = i < 2 ? 54 : 26;
            ctx->m_StickCalibrationData[i].center = 0;
            ctx->m_StickCalibrationData[i].deadzone = i < 2 ? 4 : 2;
        }
        break;
    case k_eWiiExtensionControllerType_Nunchuk:
        for (i = 0; i < 2; i++) {
            ctx->m_StickCalibrationData[i].min = 40;
            ctx->m_StickCalibrationData[i].max = 215;
            ctx->m_StickCalibrationData[i].center = 0;
            ctx->m_StickCalibrationData[i].deadzone = 10;
        }
        break;
    default:
        break;
    }
}

static void InitializeExtension(SDL_DriverWii_Context *ctx)
{
    SendExtensionReset(ctx, true);
    InitStickCalibrationData(ctx);
    ResetButtonPacketType(ctx);
}

static void UpdateSlotLED(SDL_DriverWii_Context *ctx)
{
    Uint8 leds;
    Uint8 data[2];

    // The lowest bit needs to have the rumble status
    leds = (Uint8)ctx->m_bRumbleActive;

    if (ctx->m_bPlayerLights) {
        // Use the same LED codes as Smash 8-player for 5-7
        if (ctx->m_nPlayerIndex == 0 || ctx->m_nPlayerIndex > 3) {
            leds |= k_eWiiPlayerLEDs_P1;
        }
        if (ctx->m_nPlayerIndex == 1 || ctx->m_nPlayerIndex == 4) {
            leds |= k_eWiiPlayerLEDs_P2;
        }
        if (ctx->m_nPlayerIndex == 2 || ctx->m_nPlayerIndex == 5) {
            leds |= k_eWiiPlayerLEDs_P3;
        }
        if (ctx->m_nPlayerIndex == 3 || ctx->m_nPlayerIndex == 6) {
            leds |= k_eWiiPlayerLEDs_P4;
        }
        // Turn on all lights for other player indexes
        if (ctx->m_nPlayerIndex < 0 || ctx->m_nPlayerIndex > 6) {
            leds |= k_eWiiPlayerLEDs_P1 | k_eWiiPlayerLEDs_P2 | k_eWiiPlayerLEDs_P3 | k_eWiiPlayerLEDs_P4;
        }
    }

    data[0] = k_eWiiOutputReportIDs_LEDs;
    data[1] = leds;
    WriteOutput(ctx, data, sizeof(data), false);
}

static void SDLCALL SDL_PlayerLEDHintChanged(void *userdata, const char *name, const char *oldValue, const char *hint)
{
    SDL_DriverWii_Context *ctx = (SDL_DriverWii_Context *)userdata;
    bool bPlayerLights = SDL_GetStringBoolean(hint, true);

    if (bPlayerLights != ctx->m_bPlayerLights) {
        ctx->m_bPlayerLights = bPlayerLights;

        UpdateSlotLED(ctx);
    }
}

static EWiiExtensionControllerType ReadExtensionControllerType(SDL_HIDAPI_Device *device)
{
    SDL_DriverWii_Context *ctx = (SDL_DriverWii_Context *)device->context;
    EWiiExtensionControllerType eExtensionControllerType = k_eWiiExtensionControllerType_Unknown;
    const int MAX_ATTEMPTS = 20;
    int attempts = 0;

    // Create enough of a context to read the controller type from the device
    for (attempts = 0; attempts < MAX_ATTEMPTS; ++attempts) {
        Uint16 extension;
        if (SendExtensionIdentify(ctx, true) &&
            ParseExtensionIdentifyResponse(ctx, &extension)) {
            Uint8 motion_plus_mode = 0;
            if ((extension & WII_EXTENSION_MOTIONPLUS_MASK) == WII_EXTENSION_MOTIONPLUS_ID) {
                motion_plus_mode = (Uint8)(extension >> 8);
            }
            if (motion_plus_mode || extension == WII_EXTENSION_UNINITIALIZED) {
                SendExtensionReset(ctx, true);
                if (SendExtensionIdentify(ctx, true)) {
                    ParseExtensionIdentifyResponse(ctx, &extension);
                }
            }

            eExtensionControllerType = GetExtensionType(extension);

            // Reset the Motion Plus controller if needed
            if (motion_plus_mode) {
                ActivateMotionPlusWithMode(ctx, motion_plus_mode);
            }
            break;
        }
    }
    return eExtensionControllerType;
}

static void UpdateDeviceIdentity(SDL_HIDAPI_Device *device)
{
    SDL_DriverWii_Context *ctx = (SDL_DriverWii_Context *)device->context;

    switch (ctx->m_eExtensionControllerType) {
    case k_eWiiExtensionControllerType_None:
        HIDAPI_SetDeviceName(device, "Nintendo Wii Remote");
        break;
    case k_eWiiExtensionControllerType_Nunchuk:
        HIDAPI_SetDeviceName(device, "Nintendo Wii Remote with Nunchuk");
        break;
    case k_eWiiExtensionControllerType_Gamepad:
        HIDAPI_SetDeviceName(device, "Nintendo Wii Remote with Classic Controller");
        break;
    case k_eWiiExtensionControllerType_WiiUPro:
        HIDAPI_SetDeviceName(device, "Nintendo Wii U Pro Controller");
        break;
    default:
        HIDAPI_SetDeviceName(device, "Nintendo Wii Remote with Unknown Extension");
        break;
    }
    device->guid.data[15] = ctx->m_eExtensionControllerType;
}

static bool HIDAPI_DriverWii_InitDevice(SDL_HIDAPI_Device *device)
{
    SDL_DriverWii_Context *ctx;

    ctx = (SDL_DriverWii_Context *)SDL_calloc(1, sizeof(*ctx));
    if (!ctx) {
        return false;
    }
    ctx->device = device;
    device->context = ctx;

    if (device->vendor_id == USB_VENDOR_NINTENDO) {
        ctx->m_eExtensionControllerType = ReadExtensionControllerType(device);

        UpdateDeviceIdentity(device);
    }
    return HIDAPI_JoystickConnected(device, NULL);
}

static int HIDAPI_DriverWii_GetDevicePlayerIndex(SDL_HIDAPI_Device *device, SDL_JoystickID instance_id)
{
    return -1;
}

static void HIDAPI_DriverWii_SetDevicePlayerIndex(SDL_HIDAPI_Device *device, SDL_JoystickID instance_id, int player_index)
{
    SDL_DriverWii_Context *ctx = (SDL_DriverWii_Context *)device->context;

    if (!ctx->joystick) {
        return;
    }

    ctx->m_nPlayerIndex = player_index;

    UpdateSlotLED(ctx);
}

static bool HIDAPI_DriverWii_OpenJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
    SDL_DriverWii_Context *ctx = (SDL_DriverWii_Context *)device->context;

    SDL_AssertJoysticksLocked();

    ctx->joystick = joystick;

    InitializeExtension(ctx);

    GetMotionPlusState(ctx, &ctx->m_bMotionPlusPresent, &ctx->m_ucMotionPlusMode);

    if (NeedsPeriodicMotionPlusCheck(ctx, false)) {
        SchedulePeriodicMotionPlusCheck(ctx);
    }

    if (ctx->m_eExtensionControllerType == k_eWiiExtensionControllerType_None ||
        ctx->m_eExtensionControllerType == k_eWiiExtensionControllerType_Nunchuk) {
        SDL_PrivateJoystickAddSensor(joystick, SDL_SENSOR_ACCEL, 100.0f);
        if (ctx->m_eExtensionControllerType == k_eWiiExtensionControllerType_Nunchuk) {
            SDL_PrivateJoystickAddSensor(joystick, SDL_SENSOR_ACCEL_L, 100.0f);
        }

        if (ctx->m_bMotionPlusPresent) {
            SDL_PrivateJoystickAddSensor(joystick, SDL_SENSOR_GYRO, 100.0f);
        }
    }

    // Initialize player index (needed for setting LEDs)
    ctx->m_nPlayerIndex = SDL_GetJoystickPlayerIndex(joystick);
    ctx->m_bPlayerLights = SDL_GetHintBoolean(SDL_HINT_JOYSTICK_HIDAPI_WII_PLAYER_LED, true);
    UpdateSlotLED(ctx);

    SDL_AddHintCallback(SDL_HINT_JOYSTICK_HIDAPI_WII_PLAYER_LED,
                        SDL_PlayerLEDHintChanged, ctx);

    // Initialize the joystick capabilities
    if (ctx->m_eExtensionControllerType == k_eWiiExtensionControllerType_WiiUPro) {
        joystick->nbuttons = 15;
    } else {
        // Maximum is Classic Controller + Wiimote
        joystick->nbuttons = k_eWiiButtons_Max;
    }
    joystick->naxes = SDL_GAMEPAD_AXIS_COUNT;

    ctx->m_ulLastInput = SDL_GetTicks();

    return true;
}

static bool HIDAPI_DriverWii_RumbleJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint16 low_frequency_rumble, Uint16 high_frequency_rumble)
{
    SDL_DriverWii_Context *ctx = (SDL_DriverWii_Context *)device->context;
    bool active = (low_frequency_rumble || high_frequency_rumble);

    if (active != ctx->m_bRumbleActive) {
        Uint8 data[2];

        data[0] = k_eWiiOutputReportIDs_Rumble;
        data[1] = (Uint8)active;
        WriteOutput(ctx, data, sizeof(data), false);

        ctx->m_bRumbleActive = active;
    }
    return true;
}

static bool HIDAPI_DriverWii_RumbleJoystickTriggers(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint16 left_rumble, Uint16 right_rumble)
{
    return SDL_Unsupported();
}

static Uint32 HIDAPI_DriverWii_GetJoystickCapabilities(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
    return SDL_JOYSTICK_CAP_RUMBLE;
}

static bool HIDAPI_DriverWii_SetJoystickLED(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint8 red, Uint8 green, Uint8 blue)
{
    return SDL_Unsupported();
}

static bool HIDAPI_DriverWii_SendJoystickEffect(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, const void *data, int size)
{
    return SDL_Unsupported();
}

static bool HIDAPI_DriverWii_SetJoystickSensorsEnabled(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, bool enabled)
{
    SDL_DriverWii_Context *ctx = (SDL_DriverWii_Context *)device->context;

    if (enabled != ctx->m_bReportSensors) {
        ctx->m_bReportSensors = enabled;

        if (ctx->m_bMotionPlusPresent) {
            if (enabled) {
                ActivateMotionPlus(ctx);
            } else {
                DeactivateMotionPlus(ctx);
            }
        }

        ResetButtonPacketType(ctx);
    }
    return true;
}

static void PostStickCalibrated(Uint64 timestamp, SDL_Joystick *joystick, StickCalibrationData *calibration, Uint8 axis, Uint16 data)
{
    Sint16 value = 0;
    if (!calibration->center) {
        // Center on first read
        calibration->center = data;
        return;
    }
    if (data < calibration->min) {
        calibration->min = data;
    }
    if (data > calibration->max) {
        calibration->max = data;
    }
    if (data < calibration->center - calibration->deadzone) {
        Uint16 zero = calibration->center - calibration->deadzone;
        Uint16 range = zero - calibration->min;
        Uint16 distance = zero - data;
        float fvalue = (float)distance / (float)range;
        value = (Sint16)(fvalue * SDL_JOYSTICK_AXIS_MIN);
    } else if (data > calibration->center + calibration->deadzone) {
        Uint16 zero = calibration->center + calibration->deadzone;
        Uint16 range = calibration->max - zero;
        Uint16 distance = data - zero;
        float fvalue = (float)distance / (float)range;
        value = (Sint16)(fvalue * SDL_JOYSTICK_AXIS_MAX);
    }
    if (axis == SDL_GAMEPAD_AXIS_LEFTY || axis == SDL_GAMEPAD_AXIS_RIGHTY) {
        if (value) {
            value = ~value;
        }
    }
    SDL_SendJoystickAxis(timestamp, joystick, axis, value);
}

/* Send button data to SDL
 *`defs` is a mapping for each bit to which button it represents.  0xFF indicates an unused bit
 *`data` is the button data from the controller
 *`size` is the number of bytes in `data` and the number of arrays of 8 mappings in `defs`
 *`on` is the joystick value to be sent if a bit is on
 *`off` is the joystick value to be sent if a bit is off
 */
static void PostPackedButtonData(Uint64 timestamp, SDL_Joystick *joystick, const Uint8 defs[][8], const Uint8 *data, int size, bool on, bool off)
{
    int i, j;

    for (i = 0; i < size; i++) {
        for (j = 0; j < 8; j++) {
            Uint8 button = defs[i][j];
            if (button != 0xFF) {
                bool down = (data[i] >> j) & 1 ? on : off;
                SDL_SendJoystickButton(timestamp, joystick, button, down);
            }
        }
    }
}

static const Uint8 GAMEPAD_BUTTON_DEFS[3][8] = {
    {
        0xFF /* Unused */,
        SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER,
        SDL_GAMEPAD_BUTTON_START,
        SDL_GAMEPAD_BUTTON_GUIDE,
        SDL_GAMEPAD_BUTTON_BACK,
        SDL_GAMEPAD_BUTTON_LEFT_SHOULDER,
        SDL_GAMEPAD_BUTTON_DPAD_DOWN,
        SDL_GAMEPAD_BUTTON_DPAD_RIGHT,
    },
    {
        SDL_GAMEPAD_BUTTON_DPAD_UP,
        SDL_GAMEPAD_BUTTON_DPAD_LEFT,
        0xFF /* ZR */,
        SDL_GAMEPAD_BUTTON_NORTH,
        SDL_GAMEPAD_BUTTON_EAST,
        SDL_GAMEPAD_BUTTON_WEST,
        SDL_GAMEPAD_BUTTON_SOUTH,
        0xFF /*ZL*/,
    },
    {
        SDL_GAMEPAD_BUTTON_RIGHT_STICK,
        SDL_GAMEPAD_BUTTON_LEFT_STICK,
        0xFF /* Charging */,
        0xFF /* Plugged In */,
        0xFF /* Unused */,
        0xFF /* Unused */,
        0xFF /* Unused */,
        0xFF /* Unused */,
    }
};

static const Uint8 MP_GAMEPAD_BUTTON_DEFS[3][8] = {
    {
        0xFF /* Unused */,
        SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER,
        SDL_GAMEPAD_BUTTON_START,
        SDL_GAMEPAD_BUTTON_GUIDE,
        SDL_GAMEPAD_BUTTON_BACK,
        SDL_GAMEPAD_BUTTON_LEFT_SHOULDER,
        SDL_GAMEPAD_BUTTON_DPAD_DOWN,
        SDL_GAMEPAD_BUTTON_DPAD_RIGHT,
    },
    {
        0xFF /* Motion Plus data */,
        0xFF /* Motion Plus data */,
        0xFF /* ZR */,
        SDL_GAMEPAD_BUTTON_NORTH,
        SDL_GAMEPAD_BUTTON_EAST,
        SDL_GAMEPAD_BUTTON_WEST,
        SDL_GAMEPAD_BUTTON_SOUTH,
        0xFF /*ZL*/,
    },
    {
        SDL_GAMEPAD_BUTTON_RIGHT_STICK,
        SDL_GAMEPAD_BUTTON_LEFT_STICK,
        0xFF /* Charging */,
        0xFF /* Plugged In */,
        0xFF /* Unused */,
        0xFF /* Unused */,
        0xFF /* Unused */,
        0xFF /* Unused */,
    }
};

static const Uint8 MP_FIXUP_DPAD_BUTTON_DEFS[2][8] = {
    {
        SDL_GAMEPAD_BUTTON_DPAD_UP,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
    },
    {
        SDL_GAMEPAD_BUTTON_DPAD_LEFT,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
    }
};

static void HandleWiiUProButtonData(SDL_DriverWii_Context *ctx, SDL_Joystick *joystick, const WiiButtonData *data)
{
    static const Uint8 axes[] = { SDL_GAMEPAD_AXIS_LEFTX, SDL_GAMEPAD_AXIS_RIGHTX, SDL_GAMEPAD_AXIS_LEFTY, SDL_GAMEPAD_AXIS_RIGHTY };
    const Uint8(*buttons)[8] = GAMEPAD_BUTTON_DEFS;
    Uint8 zl, zr;
    int i;

    if (data->ucNExtensionBytes < 11) {
        return;
    }

    // Buttons
    PostPackedButtonData(ctx->timestamp, joystick, buttons, data->rgucExtension + 8, 3, false, true);

    // Triggers
    zl = data->rgucExtension[9] & 0x80;
    zr = data->rgucExtension[9] & 0x04;
    SDL_SendJoystickAxis(ctx->timestamp, joystick, SDL_GAMEPAD_AXIS_LEFT_TRIGGER, zl ? SDL_JOYSTICK_AXIS_MIN : SDL_JOYSTICK_AXIS_MAX);
    SDL_SendJoystickAxis(ctx->timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER, zr ? SDL_JOYSTICK_AXIS_MIN : SDL_JOYSTICK_AXIS_MAX);

    // Sticks
    for (i = 0; i < 4; i++) {
        Uint16 value = data->rgucExtension[i * 2] | (data->rgucExtension[i * 2 + 1] << 8);
        PostStickCalibrated(ctx->timestamp, joystick, &ctx->m_StickCalibrationData[i], axes[i], value);
    }

    // Power
    UpdatePowerLevelWiiU(joystick, data->rgucExtension[10]);
}

static void HandleGamepadControllerButtonData(SDL_DriverWii_Context *ctx, SDL_Joystick *joystick, const WiiButtonData *data)
{
    const Uint8(*buttons)[8] = (ctx->m_ucMotionPlusMode == WII_MOTIONPLUS_MODE_GAMEPAD) ? MP_GAMEPAD_BUTTON_DEFS : GAMEPAD_BUTTON_DEFS;
    Uint8 lx, ly, rx, ry, zl, zr;

    if (data->ucNExtensionBytes < 6) {
        return;
    }

    // Buttons
    PostPackedButtonData(ctx->timestamp, joystick, buttons, data->rgucExtension + 4, 2, false, true);
    if (ctx->m_ucMotionPlusMode == WII_MOTIONPLUS_MODE_GAMEPAD) {
        PostPackedButtonData(ctx->timestamp, joystick, MP_FIXUP_DPAD_BUTTON_DEFS, data->rgucExtension, 2, false, true);
    }

    // Triggers
    zl = data->rgucExtension[5] & 0x80;
    zr = data->rgucExtension[5] & 0x04;
    SDL_SendJoystickAxis(ctx->timestamp, joystick, SDL_GAMEPAD_AXIS_LEFT_TRIGGER, zl ? SDL_JOYSTICK_AXIS_MIN : SDL_JOYSTICK_AXIS_MAX);
    SDL_SendJoystickAxis(ctx->timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER, zr ? SDL_JOYSTICK_AXIS_MIN : SDL_JOYSTICK_AXIS_MAX);

    // Sticks
    if (ctx->m_ucMotionPlusMode == WII_MOTIONPLUS_MODE_GAMEPAD) {
        lx = data->rgucExtension[0] & 0x3E;
        ly = data->rgucExtension[1] & 0x3E;
    } else {
        lx = data->rgucExtension[0] & 0x3F;
        ly = data->rgucExtension[1] & 0x3F;
    }
    rx = (data->rgucExtension[2] >> 7) | ((data->rgucExtension[1] >> 5) & 0x06) | ((data->rgucExtension[0] >> 3) & 0x18);
    ry = data->rgucExtension[2] & 0x1F;
    PostStickCalibrated(ctx->timestamp, joystick, &ctx->m_StickCalibrationData[0], SDL_GAMEPAD_AXIS_LEFTX, lx);
    PostStickCalibrated(ctx->timestamp, joystick, &ctx->m_StickCalibrationData[1], SDL_GAMEPAD_AXIS_LEFTY, ly);
    PostStickCalibrated(ctx->timestamp, joystick, &ctx->m_StickCalibrationData[2], SDL_GAMEPAD_AXIS_RIGHTX, rx);
    PostStickCalibrated(ctx->timestamp, joystick, &ctx->m_StickCalibrationData[3], SDL_GAMEPAD_AXIS_RIGHTY, ry);
}

static void HandleWiiRemoteButtonData(SDL_DriverWii_Context *ctx, SDL_Joystick *joystick, const WiiButtonData *data)
{
    static const Uint8 buttons[2][8] = {
        {
            k_eWiiButtons_DPad_Left,
            k_eWiiButtons_DPad_Right,
            k_eWiiButtons_DPad_Down,
            k_eWiiButtons_DPad_Up,
            k_eWiiButtons_Plus,
            0xFF /* Unused */,
            0xFF /* Unused */,
            0xFF /* Unused */,
        },
        {
            k_eWiiButtons_Two,
            k_eWiiButtons_One,
            k_eWiiButtons_B,
            k_eWiiButtons_A,
            k_eWiiButtons_Minus,
            0xFF /* Unused */,
            0xFF /* Unused */,
            k_eWiiButtons_Home,
        }
    };
    if (data->hasBaseButtons) {
        PostPackedButtonData(ctx->timestamp, joystick, buttons, data->rgucBaseButtons, 2, true, false);
    }
}

static void HandleWiiRemoteButtonDataAsMainController(SDL_DriverWii_Context *ctx, SDL_Joystick *joystick, const WiiButtonData *data)
{
    /* Wii remote maps really badly to a normal controller
     * Mapped 1 and 2 as X and Y
     * Not going to attempt positional mapping
     */
    static const Uint8 buttons[2][8] = {
        {
            SDL_GAMEPAD_BUTTON_DPAD_LEFT,
            SDL_GAMEPAD_BUTTON_DPAD_RIGHT,
            SDL_GAMEPAD_BUTTON_DPAD_DOWN,
            SDL_GAMEPAD_BUTTON_DPAD_UP,
            SDL_GAMEPAD_BUTTON_START,
            0xFF /* Unused */,
            0xFF /* Unused */,
            0xFF /* Unused */,
        },
        {
            SDL_GAMEPAD_BUTTON_NORTH,
            SDL_GAMEPAD_BUTTON_WEST,
            SDL_GAMEPAD_BUTTON_SOUTH,
            SDL_GAMEPAD_BUTTON_EAST,
            SDL_GAMEPAD_BUTTON_BACK,
            0xFF /* Unused */,
            0xFF /* Unused */,
            SDL_GAMEPAD_BUTTON_GUIDE,
        }
    };
    if (data->hasBaseButtons) {
        PostPackedButtonData(ctx->timestamp, joystick, buttons, data->rgucBaseButtons, 2, true, false);
    }
}

static void HandleNunchuckButtonData(SDL_DriverWii_Context *ctx, SDL_Joystick *joystick, const WiiButtonData *data)
{
    bool c_button, z_button;

    if (data->ucNExtensionBytes < 6) {
        return;
    }

    if (ctx->m_ucMotionPlusMode == WII_MOTIONPLUS_MODE_NUNCHUK) {
        c_button = (data->rgucExtension[5] & 0x08) ? false : true;
        z_button = (data->rgucExtension[5] & 0x04) ? false : true;
    } else {
        c_button = (data->rgucExtension[5] & 0x02) ? false : true;
        z_button = (data->rgucExtension[5] & 0x01) ? false : true;
    }
    SDL_SendJoystickButton(ctx->timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_SHOULDER, c_button);
    SDL_SendJoystickAxis(ctx->timestamp, joystick, SDL_GAMEPAD_AXIS_LEFT_TRIGGER, z_button ? SDL_JOYSTICK_AXIS_MAX : SDL_JOYSTICK_AXIS_MIN);
    PostStickCalibrated(ctx->timestamp, joystick, &ctx->m_StickCalibrationData[0], SDL_GAMEPAD_AXIS_LEFTX, data->rgucExtension[0]);
    PostStickCalibrated(ctx->timestamp, joystick, &ctx->m_StickCalibrationData[1], SDL_GAMEPAD_AXIS_LEFTY, data->rgucExtension[1]);

    if (ctx->m_bReportSensors) {
        const float ACCEL_RES_PER_G = 200.0f;
        Sint16 x, y, z;
        float values[3];

        x = (data->rgucExtension[2] << 2);
        y = (data->rgucExtension[3] << 2);
        z = (data->rgucExtension[4] << 2);

        if (ctx->m_ucMotionPlusMode == WII_MOTIONPLUS_MODE_NUNCHUK) {
            x |= ((data->rgucExtension[5] >> 3) & 0x02);
            y |= ((data->rgucExtension[5] >> 4) & 0x02);
            z &= ~0x04;
            z |= ((data->rgucExtension[5] >> 5) & 0x06);
        } else {
            x |= ((data->rgucExtension[5] >> 2) & 0x03);
            y |= ((data->rgucExtension[5] >> 4) & 0x03);
            z |= ((data->rgucExtension[5] >> 6) & 0x03);
        }

        x -= 0x200;
        y -= 0x200;
        z -= 0x200;

        values[0] = -((float)x / ACCEL_RES_PER_G) * SDL_STANDARD_GRAVITY;
        values[1] = ((float)z / ACCEL_RES_PER_G) * SDL_STANDARD_GRAVITY;
        values[2] = ((float)y / ACCEL_RES_PER_G) * SDL_STANDARD_GRAVITY;
        SDL_SendJoystickSensor(ctx->timestamp, joystick, SDL_SENSOR_ACCEL_L, ctx->timestamp, values, 3);
    }
}

static void HandleMotionPlusData(SDL_DriverWii_Context *ctx, SDL_Joystick *joystick, const WiiButtonData *data)
{
    if (ctx->m_bReportSensors) {
        const float GYRO_RES_PER_DEGREE = 8192.0f;
        int x, y, z;
        float values[3];

        x = (data->rgucExtension[0] | ((data->rgucExtension[3] << 6) & 0xFF00)) - 8192;
        y = (data->rgucExtension[1] | ((data->rgucExtension[4] << 6) & 0xFF00)) - 8192;
        z = (data->rgucExtension[2] | ((data->rgucExtension[5] << 6) & 0xFF00)) - 8192;

        if (data->rgucExtension[3] & 0x02) {
            // Slow rotation rate: 8192/440 units per deg/s
            x *= 440;
        } else {
            // Fast rotation rate: 8192/2000 units per deg/s
            x *= 2000;
        }
        if (data->rgucExtension[4] & 0x02) {
            // Slow rotation rate: 8192/440 units per deg/s
            y *= 440;
        } else {
            // Fast rotation rate: 8192/2000 units per deg/s
            y *= 2000;
        }
        if (data->rgucExtension[3] & 0x01) {
            // Slow rotation rate: 8192/440 units per deg/s
            z *= 440;
        } else {
            // Fast rotation rate: 8192/2000 units per deg/s
            z *= 2000;
        }

        values[0] = -((float)z / GYRO_RES_PER_DEGREE) * SDL_PI_F / 180.0f;
        values[1] = ((float)x / GYRO_RES_PER_DEGREE) * SDL_PI_F / 180.0f;
        values[2] = ((float)y / GYRO_RES_PER_DEGREE) * SDL_PI_F / 180.0f;
        SDL_SendJoystickSensor(ctx->timestamp, joystick, SDL_SENSOR_GYRO, ctx->timestamp, values, 3);
    }
}

static void HandleWiiRemoteAccelData(SDL_DriverWii_Context *ctx, SDL_Joystick *joystick, const WiiButtonData *data)
{
    const float ACCEL_RES_PER_G = 100.0f;
    Sint16 x, y, z;
    float values[3];

    if (!ctx->m_bReportSensors) {
        return;
    }

    x = ((data->rgucAccelerometer[0] << 2) | ((data->rgucBaseButtons[0] >> 5) & 0x03)) - 0x200;
    y = ((data->rgucAccelerometer[1] << 2) | ((data->rgucBaseButtons[1] >> 4) & 0x02)) - 0x200;
    z = ((data->rgucAccelerometer[2] << 2) | ((data->rgucBaseButtons[1] >> 5) & 0x02)) - 0x200;

    values[0] = -((float)x / ACCEL_RES_PER_G) * SDL_STANDARD_GRAVITY;
    values[1] = ((float)z / ACCEL_RES_PER_G) * SDL_STANDARD_GRAVITY;
    values[2] = ((float)y / ACCEL_RES_PER_G) * SDL_STANDARD_GRAVITY;
    SDL_SendJoystickSensor(ctx->timestamp, joystick, SDL_SENSOR_ACCEL, ctx->timestamp, values, 3);
}

static void HandleButtonData(SDL_DriverWii_Context *ctx, SDL_Joystick *joystick, WiiButtonData *data)
{
    if (ctx->m_eExtensionControllerType == k_eWiiExtensionControllerType_WiiUPro) {
        HandleWiiUProButtonData(ctx, joystick, data);
        return;
    }

    if (ctx->m_ucMotionPlusMode != WII_MOTIONPLUS_MODE_NONE &&
        data->ucNExtensionBytes > 5) {
        if (data->rgucExtension[5] & 0x01) {
            // The data is invalid, possibly during a hotplug
            return;
        }

        if (data->rgucExtension[4] & 0x01) {
            if (ctx->m_eExtensionControllerType == k_eWiiExtensionControllerType_None) {
                // Something was plugged into the extension port, reinitialize to get new state
                ctx->m_bDisconnected = true;
            }
        } else {
            if (ctx->m_eExtensionControllerType != k_eWiiExtensionControllerType_None) {
                // Something was removed from the extension port, reinitialize to get new state
                ctx->m_bDisconnected = true;
            }
        }

        if (data->rgucExtension[5] & 0x02) {
            HandleMotionPlusData(ctx, joystick, data);

            // The extension data is consumed
            data->ucNExtensionBytes = 0;
        }
    }

    HandleWiiRemoteButtonData(ctx, joystick, data);
    switch (ctx->m_eExtensionControllerType) {
    case k_eWiiExtensionControllerType_Nunchuk:
        HandleNunchuckButtonData(ctx, joystick, data);
        SDL_FALLTHROUGH;
    case k_eWiiExtensionControllerType_None:
        HandleWiiRemoteButtonDataAsMainController(ctx, joystick, data);
        break;
    case k_eWiiExtensionControllerType_Gamepad:
        HandleGamepadControllerButtonData(ctx, joystick, data);
        break;
    default:
        break;
    }
    HandleWiiRemoteAccelData(ctx, joystick, data);
}

static void GetBaseButtons(WiiButtonData *dst, const Uint8 *src)
{
    SDL_memcpy(dst->rgucBaseButtons, src, 2);
    dst->hasBaseButtons = true;
}

static void GetAccelerometer(WiiButtonData *dst, const Uint8 *src)
{
    SDL_memcpy(dst->rgucAccelerometer, src, 3);
    dst->hasAccelerometer = true;
}

static void GetExtensionData(WiiButtonData *dst, const Uint8 *src, int size)
{
    bool valid_data = false;
    int i;

    if (size > sizeof(dst->rgucExtension)) {
        size = sizeof(dst->rgucExtension);
    }

    for (i = 0; i < size; ++i) {
        if (src[i] != 0xFF) {
            valid_data = true;
            break;
        }
    }
    if (valid_data) {
        SDL_memcpy(dst->rgucExtension, src, size);
        dst->ucNExtensionBytes = (Uint8)size;
    }
}

static void HandleStatus(SDL_DriverWii_Context *ctx, SDL_Joystick *joystick)
{
    bool hadExtension = ctx->m_eExtensionControllerType != k_eWiiExtensionControllerType_None;
    bool hasExtension = (ctx->m_rgucReadBuffer[3] & 2) ? true : false;
    WiiButtonData data;
    SDL_zero(data);
    GetBaseButtons(&data, ctx->m_rgucReadBuffer + 1);
    HandleButtonData(ctx, joystick, &data);

    if (ctx->m_eExtensionControllerType != k_eWiiExtensionControllerType_WiiUPro) {
        // Wii U has separate battery level tracking
        UpdatePowerLevelWii(joystick, ctx->m_rgucReadBuffer[6]);
    }

    // The report data format has been reset, need to update it
    ResetButtonPacketType(ctx);

    SDL_LogDebug(SDL_LOG_CATEGORY_INPUT, "HIDAPI Wii: Status update, extension %s", hasExtension ? "CONNECTED" : "DISCONNECTED");

    /* When Motion Plus is active, we get extension connect/disconnect status
     * through the Motion Plus packets. Otherwise we can use the status here.
     */
    if (ctx->m_ucMotionPlusMode != WII_MOTIONPLUS_MODE_NONE) {
        /* Check to make sure the Motion Plus extension state hasn't changed,
         * otherwise we'll get extension connect/disconnect status through
         * Motion Plus packets.
         */
        if (NeedsPeriodicMotionPlusCheck(ctx, true)) {
            ctx->m_ulNextMotionPlusCheck = SDL_GetTicks();
        }

    } else if (hadExtension != hasExtension) {
        // Reinitialize to get new state
        ctx->m_bDisconnected = true;
    }
}

static void HandleResponse(SDL_DriverWii_Context *ctx, SDL_Joystick *joystick)
{
    EWiiInputReportIDs type = (EWiiInputReportIDs)ctx->m_rgucReadBuffer[0];
    WiiButtonData data;
    SDL_assert(type == k_eWiiInputReportIDs_Acknowledge || type == k_eWiiInputReportIDs_ReadMemory);
    SDL_zero(data);
    GetBaseButtons(&data, ctx->m_rgucReadBuffer + 1);
    HandleButtonData(ctx, joystick, &data);

    switch (ctx->m_eCommState) {
    case k_eWiiCommunicationState_None:
        break;

    case k_eWiiCommunicationState_CheckMotionPlusStage1:
    case k_eWiiCommunicationState_CheckMotionPlusStage2:
    {
        Uint16 extension = 0;
        if (ParseExtensionIdentifyResponse(ctx, &extension)) {
            if ((extension & WII_EXTENSION_MOTIONPLUS_MASK) == WII_EXTENSION_MOTIONPLUS_ID) {
                // Motion Plus is currently active
                SDL_LogDebug(SDL_LOG_CATEGORY_INPUT, "HIDAPI Wii: Motion Plus CONNECTED (stage %d)", ctx->m_eCommState == k_eWiiCommunicationState_CheckMotionPlusStage1 ? 1 : 2);

                if (!ctx->m_bMotionPlusPresent) {
                    // Reinitialize to get new sensor availability
                    ctx->m_bDisconnected = true;
                }
                ctx->m_eCommState = k_eWiiCommunicationState_None;

            } else if (ctx->m_eCommState == k_eWiiCommunicationState_CheckMotionPlusStage1) {
                // Check to see if Motion Plus is present
                ReadRegister(ctx, 0xA600FE, 2, false);

                ctx->m_eCommState = k_eWiiCommunicationState_CheckMotionPlusStage2;

            } else {
                // Motion Plus is not present
                SDL_LogDebug(SDL_LOG_CATEGORY_INPUT, "HIDAPI Wii: Motion Plus DISCONNECTED (stage %d)", ctx->m_eCommState == k_eWiiCommunicationState_CheckMotionPlusStage1 ? 1 : 2);

                if (ctx->m_bMotionPlusPresent) {
                    // Reinitialize to get new sensor availability
                    ctx->m_bDisconnected = true;
                }
                ctx->m_eCommState = k_eWiiCommunicationState_None;
            }
        }
    } break;
    default:
        // Should never happen
        break;
    }
}

static void HandleButtonPacket(SDL_DriverWii_Context *ctx, SDL_Joystick *joystick)
{
    EWiiInputReportIDs eExpectedReport = GetButtonPacketType(ctx);
    WiiButtonData data;

    // FIXME: This should see if the data format is compatible rather than equal
    if (eExpectedReport != ctx->m_rgucReadBuffer[0]) {
        SDL_LogDebug(SDL_LOG_CATEGORY_INPUT, "HIDAPI Wii: Resetting report mode to %d", eExpectedReport);
        RequestButtonPacketType(ctx, eExpectedReport);
    }

    // IR camera data is not supported
    SDL_zero(data);
    switch (ctx->m_rgucReadBuffer[0]) {
    case k_eWiiInputReportIDs_ButtonData0: // 30 BB BB
        GetBaseButtons(&data, ctx->m_rgucReadBuffer + 1);
        break;
    case k_eWiiInputReportIDs_ButtonData1: // 31 BB BB AA AA AA
    case k_eWiiInputReportIDs_ButtonData3: // 33 BB BB AA AA AA II II II II II II II II II II II II
        GetBaseButtons(&data, ctx->m_rgucReadBuffer + 1);
        GetAccelerometer(&data, ctx->m_rgucReadBuffer + 3);
        break;
    case k_eWiiInputReportIDs_ButtonData2: // 32 BB BB EE EE EE EE EE EE EE EE
        GetBaseButtons(&data, ctx->m_rgucReadBuffer + 1);
        GetExtensionData(&data, ctx->m_rgucReadBuffer + 3, 8);
        break;
    case k_eWiiInputReportIDs_ButtonData4: // 34 BB BB EE EE EE EE EE EE EE EE EE EE EE EE EE EE EE EE EE EE EE
        GetBaseButtons(&data, ctx->m_rgucReadBuffer + 1);
        GetExtensionData(&data, ctx->m_rgucReadBuffer + 3, 19);
        break;
    case k_eWiiInputReportIDs_ButtonData5: // 35 BB BB AA AA AA EE EE EE EE EE EE EE EE EE EE EE EE EE EE EE EE
        GetBaseButtons(&data, ctx->m_rgucReadBuffer + 1);
        GetAccelerometer(&data, ctx->m_rgucReadBuffer + 3);
        GetExtensionData(&data, ctx->m_rgucReadBuffer + 6, 16);
        break;
    case k_eWiiInputReportIDs_ButtonData6: // 36 BB BB II II II II II II II II II II EE EE EE EE EE EE EE EE EE
        GetBaseButtons(&data, ctx->m_rgucReadBuffer + 1);
        GetExtensionData(&data, ctx->m_rgucReadBuffer + 13, 9);
        break;
    case k_eWiiInputReportIDs_ButtonData7: // 37 BB BB AA AA AA II II II II II II II II II II EE EE EE EE EE EE
        GetBaseButtons(&data, ctx->m_rgucReadBuffer + 1);
        GetExtensionData(&data, ctx->m_rgucReadBuffer + 16, 6);
        break;
    case k_eWiiInputReportIDs_ButtonDataD: // 3d EE EE EE EE EE EE EE EE EE EE EE EE EE EE EE EE EE EE EE EE EE
        GetExtensionData(&data, ctx->m_rgucReadBuffer + 1, 21);
        break;
    case k_eWiiInputReportIDs_ButtonDataE:
    case k_eWiiInputReportIDs_ButtonDataF:
    default:
        SDL_LogDebug(SDL_LOG_CATEGORY_INPUT, "HIDAPI Wii: Unsupported button data type %02x", ctx->m_rgucReadBuffer[0]);
        return;
    }
    HandleButtonData(ctx, joystick, &data);
}

static void HandleInput(SDL_DriverWii_Context *ctx, SDL_Joystick *joystick)
{
    EWiiInputReportIDs type = (EWiiInputReportIDs)ctx->m_rgucReadBuffer[0];

    // Set up for handling input
    ctx->timestamp = SDL_GetTicksNS();

    if (type == k_eWiiInputReportIDs_Status) {
        HandleStatus(ctx, joystick);
    } else if (type == k_eWiiInputReportIDs_Acknowledge || type == k_eWiiInputReportIDs_ReadMemory) {
        HandleResponse(ctx, joystick);
    } else if (type >= k_eWiiInputReportIDs_ButtonData0 && type <= k_eWiiInputReportIDs_ButtonDataF) {
        HandleButtonPacket(ctx, joystick);
    } else {
        SDL_LogDebug(SDL_LOG_CATEGORY_INPUT, "HIDAPI Wii: Unexpected input packet of type %x", type);
    }
}

static bool HIDAPI_DriverWii_UpdateDevice(SDL_HIDAPI_Device *device)
{
    SDL_DriverWii_Context *ctx = (SDL_DriverWii_Context *)device->context;
    SDL_Joystick *joystick = NULL;
    int size;
    Uint64 now;

    if (device->num_joysticks > 0) {
        joystick = SDL_GetJoystickFromID(device->joysticks[0]);
    } else {
        return false;
    }

    now = SDL_GetTicks();

    while ((size = ReadInput(ctx)) > 0) {
        if (joystick) {
            HandleInput(ctx, joystick);
        }
        ctx->m_ulLastInput = now;
    }

    /* Check to see if we've lost connection to the controller.
     * We have continuous reporting enabled, so this should be reliable now.
     */
    {
        SDL_COMPILE_TIME_ASSERT(ENABLE_CONTINUOUS_REPORTING, ENABLE_CONTINUOUS_REPORTING);
    }
    if (now >= (ctx->m_ulLastInput + INPUT_WAIT_TIMEOUT_MS)) {
        // Bluetooth may have disconnected, try reopening the controller
        size = -1;
    }

    if (joystick) {
        // These checks aren't needed on the Wii U Pro Controller
        if (ctx->m_eExtensionControllerType != k_eWiiExtensionControllerType_WiiUPro) {

            // Check to see if the Motion Plus extension status has changed
            if (ctx->m_ulNextMotionPlusCheck && now >= ctx->m_ulNextMotionPlusCheck) {
                CheckMotionPlusConnection(ctx);
                if (NeedsPeriodicMotionPlusCheck(ctx, false)) {
                    SchedulePeriodicMotionPlusCheck(ctx);
                } else {
                    ctx->m_ulNextMotionPlusCheck = 0;
                }
            }

            // Request a status update periodically to make sure our battery value is up to date
            if (!ctx->m_ulLastStatus || now >= (ctx->m_ulLastStatus + STATUS_UPDATE_TIME_MS)) {
                Uint8 data[2];

                data[0] = k_eWiiOutputReportIDs_StatusRequest;
                data[1] = (Uint8)ctx->m_bRumbleActive;
                WriteOutput(ctx, data, sizeof(data), false);

                ctx->m_ulLastStatus = now;
            }
        }
    }

    if (size < 0 || ctx->m_bDisconnected) {
        // Read error, device is disconnected
        HIDAPI_JoystickDisconnected(device, device->joysticks[0]);
    }
    return (size >= 0);
}

static void HIDAPI_DriverWii_CloseJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
    SDL_DriverWii_Context *ctx = (SDL_DriverWii_Context *)device->context;

    SDL_RemoveHintCallback(SDL_HINT_JOYSTICK_HIDAPI_WII_PLAYER_LED,
                        SDL_PlayerLEDHintChanged, ctx);

    ctx->joystick = NULL;
}

static void HIDAPI_DriverWii_FreeDevice(SDL_HIDAPI_Device *device)
{
}

SDL_HIDAPI_DeviceDriver SDL_HIDAPI_DriverWii = {
    SDL_HINT_JOYSTICK_HIDAPI_WII,
    true,
    HIDAPI_DriverWii_RegisterHints,
    HIDAPI_DriverWii_UnregisterHints,
    HIDAPI_DriverWii_IsEnabled,
    HIDAPI_DriverWii_IsSupportedDevice,
    HIDAPI_DriverWii_InitDevice,
    HIDAPI_DriverWii_GetDevicePlayerIndex,
    HIDAPI_DriverWii_SetDevicePlayerIndex,
    HIDAPI_DriverWii_UpdateDevice,
    HIDAPI_DriverWii_OpenJoystick,
    HIDAPI_DriverWii_RumbleJoystick,
    HIDAPI_DriverWii_RumbleJoystickTriggers,
    HIDAPI_DriverWii_GetJoystickCapabilities,
    HIDAPI_DriverWii_SetJoystickLED,
    HIDAPI_DriverWii_SendJoystickEffect,
    HIDAPI_DriverWii_SetJoystickSensorsEnabled,
    HIDAPI_DriverWii_CloseJoystick,
    HIDAPI_DriverWii_FreeDevice,
};

#endif // SDL_JOYSTICK_HIDAPI_WII

#endif // SDL_JOYSTICK_HIDAPI
