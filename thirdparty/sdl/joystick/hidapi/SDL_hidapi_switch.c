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
/* This driver supports the Nintendo Switch Pro controller.
   Code and logic contributed by Valve Corporation under the SDL zlib license.
*/
#include "SDL_internal.h"

#ifdef SDL_JOYSTICK_HIDAPI

#include "../../SDL_hints_c.h"
#include "../SDL_sysjoystick.h"
#include "SDL_hidapijoystick_c.h"
#include "SDL_hidapi_rumble.h"
#include "SDL_hidapi_nintendo.h"

#ifdef SDL_JOYSTICK_HIDAPI_SWITCH

// Define this if you want to log all packets from the controller
// #define DEBUG_SWITCH_PROTOCOL

// Define this to get log output for rumble logic
// #define DEBUG_RUMBLE

/* The initialization sequence doesn't appear to work correctly on Windows unless
   the reads and writes are on the same thread.

   ... and now I can't reproduce this, so I'm leaving it in, but disabled for now.
 */
// #define SWITCH_SYNCHRONOUS_WRITES

/* How often you can write rumble commands to the controller.
   If you send commands more frequently than this, you can turn off the controller
   in Bluetooth mode, or the motors can miss the command in USB mode.
 */
#define RUMBLE_WRITE_FREQUENCY_MS 30

// How often you have to refresh a long duration rumble to keep the motors running
#define RUMBLE_REFRESH_FREQUENCY_MS 50

#define SWITCH_GYRO_SCALE  14.2842f
#define SWITCH_ACCEL_SCALE 4096.f

#define SWITCH_GYRO_SCALE_MULT    936.0f
#define SWITCH_ACCEL_SCALE_MULT   4.0f

enum
{
    SDL_GAMEPAD_BUTTON_SWITCH_SHARE = 11,
    SDL_GAMEPAD_BUTTON_SWITCH_RIGHT_PADDLE1,
    SDL_GAMEPAD_BUTTON_SWITCH_LEFT_PADDLE1,
    SDL_GAMEPAD_BUTTON_SWITCH_RIGHT_PADDLE2,
    SDL_GAMEPAD_BUTTON_SWITCH_LEFT_PADDLE2,
    SDL_GAMEPAD_NUM_SWITCH_BUTTONS,
};

typedef enum
{
    k_eSwitchInputReportIDs_SubcommandReply = 0x21,
    k_eSwitchInputReportIDs_FullControllerState = 0x30,
    k_eSwitchInputReportIDs_FullControllerAndMcuState = 0x31,
    k_eSwitchInputReportIDs_SimpleControllerState = 0x3F,
    k_eSwitchInputReportIDs_CommandAck = 0x81,
} ESwitchInputReportIDs;

typedef enum
{
    k_eSwitchOutputReportIDs_RumbleAndSubcommand = 0x01,
    k_eSwitchOutputReportIDs_Rumble = 0x10,
    k_eSwitchOutputReportIDs_Proprietary = 0x80,
} ESwitchOutputReportIDs;

typedef enum
{
    k_eSwitchSubcommandIDs_BluetoothManualPair = 0x01,
    k_eSwitchSubcommandIDs_RequestDeviceInfo = 0x02,
    k_eSwitchSubcommandIDs_SetInputReportMode = 0x03,
    k_eSwitchSubcommandIDs_SetHCIState = 0x06,
    k_eSwitchSubcommandIDs_SPIFlashRead = 0x10,
    k_eSwitchSubcommandIDs_SetPlayerLights = 0x30,
    k_eSwitchSubcommandIDs_SetHomeLight = 0x38,
    k_eSwitchSubcommandIDs_EnableIMU = 0x40,
    k_eSwitchSubcommandIDs_SetIMUSensitivity = 0x41,
    k_eSwitchSubcommandIDs_EnableVibration = 0x48,
} ESwitchSubcommandIDs;

typedef enum
{
    k_eSwitchProprietaryCommandIDs_Status = 0x01,
    k_eSwitchProprietaryCommandIDs_Handshake = 0x02,
    k_eSwitchProprietaryCommandIDs_HighSpeed = 0x03,
    k_eSwitchProprietaryCommandIDs_ForceUSB = 0x04,
    k_eSwitchProprietaryCommandIDs_ClearUSB = 0x05,
    k_eSwitchProprietaryCommandIDs_ResetMCU = 0x06,
} ESwitchProprietaryCommandIDs;

#define k_unSwitchOutputPacketDataLength 49
#define k_unSwitchMaxOutputPacketLength  64
#define k_unSwitchBluetoothPacketLength  k_unSwitchOutputPacketDataLength
#define k_unSwitchUSBPacketLength        k_unSwitchMaxOutputPacketLength

#define k_unSPIStickFactoryCalibrationStartOffset 0x603D
#define k_unSPIStickFactoryCalibrationEndOffset   0x604E
#define k_unSPIStickFactoryCalibrationLength      (k_unSPIStickFactoryCalibrationEndOffset - k_unSPIStickFactoryCalibrationStartOffset + 1)

#define k_unSPIStickUserCalibrationStartOffset 0x8010
#define k_unSPIStickUserCalibrationEndOffset   0x8025
#define k_unSPIStickUserCalibrationLength      (k_unSPIStickUserCalibrationEndOffset - k_unSPIStickUserCalibrationStartOffset + 1)

#define k_unSPIIMUScaleStartOffset 0x6020
#define k_unSPIIMUScaleEndOffset   0x6037
#define k_unSPIIMUScaleLength      (k_unSPIIMUScaleEndOffset - k_unSPIIMUScaleStartOffset + 1)

#define k_unSPIIMUUserScaleStartOffset 0x8026
#define k_unSPIIMUUserScaleEndOffset   0x8039
#define k_unSPIIMUUserScaleLength      (k_unSPIIMUUserScaleEndOffset - k_unSPIIMUUserScaleStartOffset + 1)

#pragma pack(1)
typedef struct
{
    Uint8 rgucButtons[2];
    Uint8 ucStickHat;
    Uint8 rgucJoystickLeft[2];
    Uint8 rgucJoystickRight[2];
} SwitchInputOnlyControllerStatePacket_t;

typedef struct
{
    Uint8 rgucButtons[2];
    Uint8 ucStickHat;
    Sint16 sJoystickLeft[2];
    Sint16 sJoystickRight[2];
} SwitchSimpleStatePacket_t;

typedef struct
{
    Uint8 ucCounter;
    Uint8 ucBatteryAndConnection;
    Uint8 rgucButtons[3];
    Uint8 rgucJoystickLeft[3];
    Uint8 rgucJoystickRight[3];
    Uint8 ucVibrationCode;
} SwitchControllerStatePacket_t;

typedef struct
{
    Sint16 sAccelX;
    Sint16 sAccelY;
    Sint16 sAccelZ;

    Sint16 sGyroX;
    Sint16 sGyroY;
    Sint16 sGyroZ;
} SwitchControllerIMUState_t;

typedef struct
{
    SwitchControllerStatePacket_t controllerState;
    SwitchControllerIMUState_t imuState[3];
} SwitchStatePacket_t;

typedef struct
{
    Uint32 unAddress;
    Uint8 ucLength;
} SwitchSPIOpData_t;

typedef struct
{
    SwitchControllerStatePacket_t m_controllerState;

    Uint8 ucSubcommandAck;
    Uint8 ucSubcommandID;

#define k_unSubcommandDataBytes 35
    union
    {
        Uint8 rgucSubcommandData[k_unSubcommandDataBytes];

        struct
        {
            SwitchSPIOpData_t opData;
            Uint8 rgucReadData[k_unSubcommandDataBytes - sizeof(SwitchSPIOpData_t)];
        } spiReadData;

        struct
        {
            Uint8 rgucFirmwareVersion[2];
            Uint8 ucDeviceType;
            Uint8 ucFiller1;
            Uint8 rgucMACAddress[6];
            Uint8 ucFiller2;
            Uint8 ucColorLocation;
        } deviceInfo;

        struct
        {
            SwitchSPIOpData_t opData;
            Uint8 rgucLeftCalibration[9];
            Uint8 rgucRightCalibration[9];
        } stickFactoryCalibration;

        struct
        {
            SwitchSPIOpData_t opData;
            Uint8 rgucLeftMagic[2];
            Uint8 rgucLeftCalibration[9];
            Uint8 rgucRightMagic[2];
            Uint8 rgucRightCalibration[9];
        } stickUserCalibration;
    };
} SwitchSubcommandInputPacket_t;

typedef struct
{
    Uint8 ucPacketType;
    Uint8 ucCommandID;
    Uint8 ucFiller;

    Uint8 ucDeviceType;
    Uint8 rgucMACAddress[6];
} SwitchProprietaryStatusPacket_t;

typedef struct
{
    Uint8 rgucData[4];
} SwitchRumbleData_t;

typedef struct
{
    Uint8 ucPacketType;
    Uint8 ucPacketNumber;
    SwitchRumbleData_t rumbleData[2];
} SwitchCommonOutputPacket_t;

typedef struct
{
    SwitchCommonOutputPacket_t commonData;

    Uint8 ucSubcommandID;
    Uint8 rgucSubcommandData[k_unSwitchOutputPacketDataLength - sizeof(SwitchCommonOutputPacket_t) - 1];
} SwitchSubcommandOutputPacket_t;

typedef struct
{
    Uint8 ucPacketType;
    Uint8 ucProprietaryID;

    Uint8 rgucProprietaryData[k_unSwitchOutputPacketDataLength - 1 - 1];
} SwitchProprietaryOutputPacket_t;
#pragma pack()

/* Enhanced report hint mode:
 * "0": enhanced features are never used
 * "1": enhanced features are always used
 * "auto": enhanced features are advertised to the application, but SDL doesn't touch the controller state unless the application explicitly requests it.
 */
typedef enum
{
    SWITCH_ENHANCED_REPORT_HINT_OFF,
    SWITCH_ENHANCED_REPORT_HINT_ON,
    SWITCH_ENHANCED_REPORT_HINT_AUTO
} HIDAPI_Switch_EnhancedReportHint;

typedef struct
{
    SDL_HIDAPI_Device *device;
    SDL_Joystick *joystick;
    bool m_bInputOnly;
    bool m_bUseButtonLabels;
    bool m_bPlayerLights;
    int m_nPlayerIndex;
    bool m_bSyncWrite;
    int m_nMaxWriteAttempts;
    ESwitchDeviceInfoControllerType m_eControllerType;
    Uint8 m_nInitialInputMode;
    Uint8 m_nCurrentInputMode;
    Uint8 m_rgucMACAddress[6];
    Uint8 m_nCommandNumber;
    HIDAPI_Switch_EnhancedReportHint m_eEnhancedReportHint;
    bool m_bEnhancedMode;
    bool m_bEnhancedModeAvailable;
    SwitchCommonOutputPacket_t m_RumblePacket;
    Uint8 m_rgucReadBuffer[k_unSwitchMaxOutputPacketLength];
    bool m_bRumbleActive;
    Uint64 m_ulRumbleSent;
    bool m_bRumblePending;
    bool m_bRumbleZeroPending;
    Uint32 m_unRumblePending;
    bool m_bSensorsSupported;
    bool m_bReportSensors;
    bool m_bHasSensorData;
    Uint64 m_ulLastInput;
    Uint64 m_ulLastIMUReset;
    Uint64 m_ulIMUSampleTimestampNS;
    Uint32 m_unIMUSamples;
    Uint64 m_ulIMUUpdateIntervalNS;
    Uint64 m_ulTimestampNS;
    bool m_bVerticalMode;

    SwitchInputOnlyControllerStatePacket_t m_lastInputOnlyState;
    SwitchSimpleStatePacket_t m_lastSimpleState;
    SwitchStatePacket_t m_lastFullState;

    struct StickCalibrationData
    {
        struct
        {
            Sint16 sCenter;
            Sint16 sMin;
            Sint16 sMax;
        } axis[2];
    } m_StickCalData[2];

    struct StickExtents
    {
        struct
        {
            Sint16 sMin;
            Sint16 sMax;
        } axis[2];
    } m_StickExtents[2], m_SimpleStickExtents[2];

    struct IMUScaleData
    {
        float fAccelScaleX;
        float fAccelScaleY;
        float fAccelScaleZ;

        float fGyroScaleX;
        float fGyroScaleY;
        float fGyroScaleZ;
    } m_IMUScaleData;
} SDL_DriverSwitch_Context;

static int ReadInput(SDL_DriverSwitch_Context *ctx)
{
    int result;

    // Make sure we don't try to read at the same time a write is happening
    if (SDL_GetAtomicInt(&ctx->device->rumble_pending) > 0) {
        return 0;
    }

    result = SDL_hid_read_timeout(ctx->device->dev, ctx->m_rgucReadBuffer, sizeof(ctx->m_rgucReadBuffer), 0);

    // See if we can guess the initial input mode
    if (result > 0 && !ctx->m_bInputOnly && !ctx->m_nInitialInputMode) {
        switch (ctx->m_rgucReadBuffer[0]) {
        case k_eSwitchInputReportIDs_FullControllerState:
        case k_eSwitchInputReportIDs_FullControllerAndMcuState:
        case k_eSwitchInputReportIDs_SimpleControllerState:
            ctx->m_nInitialInputMode = ctx->m_rgucReadBuffer[0];
            break;
        default:
            break;
        }
    }
    return result;
}

static int WriteOutput(SDL_DriverSwitch_Context *ctx, const Uint8 *data, int size)
{
#ifdef SWITCH_SYNCHRONOUS_WRITES
    return SDL_hid_write(ctx->device->dev, data, size);
#else
    // Use the rumble thread for general asynchronous writes
    if (!SDL_HIDAPI_LockRumble()) {
        return -1;
    }
    return SDL_HIDAPI_SendRumbleAndUnlock(ctx->device, data, size);
#endif // SWITCH_SYNCHRONOUS_WRITES
}

static SwitchSubcommandInputPacket_t *ReadSubcommandReply(SDL_DriverSwitch_Context *ctx, ESwitchSubcommandIDs expectedID, const Uint8 *pBuf, Uint8 ucLen)
{
    // Average response time for messages is ~30ms
    Uint64 endTicks = SDL_GetTicks() + 100;

    int nRead = 0;
    while ((nRead = ReadInput(ctx)) != -1) {
        if (nRead > 0) {
            if (ctx->m_rgucReadBuffer[0] == k_eSwitchInputReportIDs_SubcommandReply) {
                SwitchSubcommandInputPacket_t *reply = (SwitchSubcommandInputPacket_t *)&ctx->m_rgucReadBuffer[1];
                if (reply->ucSubcommandID != expectedID || !(reply->ucSubcommandAck & 0x80)) {
                    continue;
                }
                if (reply->ucSubcommandID == k_eSwitchSubcommandIDs_SPIFlashRead) {
                    SDL_assert(ucLen == sizeof(reply->spiReadData.opData));
                    if (SDL_memcmp(&reply->spiReadData.opData, pBuf, ucLen) != 0) {
                        // This was a reply for another SPI read command
                        continue;
                    }
                }
                return reply;
            }
        } else {
            SDL_Delay(1);
        }

        if (SDL_GetTicks() >= endTicks) {
            break;
        }
    }
    return NULL;
}

static bool ReadProprietaryReply(SDL_DriverSwitch_Context *ctx, ESwitchProprietaryCommandIDs expectedID)
{
    // Average response time for messages is ~30ms
    Uint64 endTicks = SDL_GetTicks() + 100;

    int nRead = 0;
    while ((nRead = ReadInput(ctx)) != -1) {
        if (nRead > 0) {
            if (ctx->m_rgucReadBuffer[0] == k_eSwitchInputReportIDs_CommandAck && ctx->m_rgucReadBuffer[1] == expectedID) {
                return true;
            }
        } else {
            SDL_Delay(1);
        }

        if (SDL_GetTicks() >= endTicks) {
            break;
        }
    }
    return false;
}

static void ConstructSubcommand(SDL_DriverSwitch_Context *ctx, ESwitchSubcommandIDs ucCommandID, const Uint8 *pBuf, Uint8 ucLen, SwitchSubcommandOutputPacket_t *outPacket)
{
    SDL_memset(outPacket, 0, sizeof(*outPacket));

    outPacket->commonData.ucPacketType = k_eSwitchOutputReportIDs_RumbleAndSubcommand;
    outPacket->commonData.ucPacketNumber = ctx->m_nCommandNumber;

    SDL_memcpy(outPacket->commonData.rumbleData, ctx->m_RumblePacket.rumbleData, sizeof(ctx->m_RumblePacket.rumbleData));

    outPacket->ucSubcommandID = ucCommandID;
    if (pBuf) {
        SDL_memcpy(outPacket->rgucSubcommandData, pBuf, ucLen);
    }

    ctx->m_nCommandNumber = (ctx->m_nCommandNumber + 1) & 0xF;
}

static bool WritePacket(SDL_DriverSwitch_Context *ctx, void *pBuf, Uint8 ucLen)
{
    Uint8 rgucBuf[k_unSwitchMaxOutputPacketLength];
    const size_t unWriteSize = ctx->device->is_bluetooth ? k_unSwitchBluetoothPacketLength : k_unSwitchUSBPacketLength;

    if (ucLen > k_unSwitchOutputPacketDataLength) {
        return false;
    }

    if (ucLen < unWriteSize) {
        SDL_memcpy(rgucBuf, pBuf, ucLen);
        SDL_memset(rgucBuf + ucLen, 0, unWriteSize - ucLen);
        pBuf = rgucBuf;
        ucLen = (Uint8)unWriteSize;
    }
    if (ctx->m_bSyncWrite) {
        return SDL_hid_write(ctx->device->dev, (Uint8 *)pBuf, ucLen) >= 0;
    } else {
        return WriteOutput(ctx, (Uint8 *)pBuf, ucLen) >= 0;
    }
}

static bool WriteSubcommand(SDL_DriverSwitch_Context *ctx, ESwitchSubcommandIDs ucCommandID, const Uint8 *pBuf, Uint8 ucLen, SwitchSubcommandInputPacket_t **ppReply)
{
    SwitchSubcommandInputPacket_t *reply = NULL;
    int nTries;

    for (nTries = 1; !reply && nTries <= ctx->m_nMaxWriteAttempts; ++nTries) {
        SwitchSubcommandOutputPacket_t commandPacket;
        ConstructSubcommand(ctx, ucCommandID, pBuf, ucLen, &commandPacket);

        if (!WritePacket(ctx, &commandPacket, sizeof(commandPacket))) {
            continue;
        }

        reply = ReadSubcommandReply(ctx, ucCommandID, pBuf, ucLen);
    }

    if (ppReply) {
        *ppReply = reply;
    }
    return reply != NULL;
}

static bool WriteProprietary(SDL_DriverSwitch_Context *ctx, ESwitchProprietaryCommandIDs ucCommand, Uint8 *pBuf, Uint8 ucLen, bool waitForReply)
{
    int nTries;

    for (nTries = 1; nTries <= ctx->m_nMaxWriteAttempts; ++nTries) {
        SwitchProprietaryOutputPacket_t packet;

        if ((!pBuf && ucLen > 0) || ucLen > sizeof(packet.rgucProprietaryData)) {
            return false;
        }

        SDL_zero(packet);
        packet.ucPacketType = k_eSwitchOutputReportIDs_Proprietary;
        packet.ucProprietaryID = ucCommand;
        if (pBuf) {
            SDL_memcpy(packet.rgucProprietaryData, pBuf, ucLen);
        }

        if (!WritePacket(ctx, &packet, sizeof(packet))) {
            continue;
        }

        if (!waitForReply || ReadProprietaryReply(ctx, ucCommand)) {
            // SDL_Log("Succeeded%s after %d tries", ctx->m_bSyncWrite ? " (sync)" : "", nTries);
            return true;
        }
    }
    // SDL_Log("Failed%s after %d tries", ctx->m_bSyncWrite ? " (sync)" : "", nTries);
    return false;
}

static Uint8 EncodeRumbleHighAmplitude(Uint16 amplitude)
{
    /* More information about these values can be found here:
     * https://github.com/dekuNukem/Nintendo_Switch_Reverse_Engineering/blob/master/rumble_data_table.md
     */
    Uint16 hfa[101][2] = { { 0, 0x0 }, { 514, 0x2 }, { 775, 0x4 }, { 921, 0x6 }, { 1096, 0x8 }, { 1303, 0x0a }, { 1550, 0x0c }, { 1843, 0x0e }, { 2192, 0x10 }, { 2606, 0x12 }, { 3100, 0x14 }, { 3686, 0x16 }, { 4383, 0x18 }, { 5213, 0x1a }, { 6199, 0x1c }, { 7372, 0x1e }, { 7698, 0x20 }, { 8039, 0x22 }, { 8395, 0x24 }, { 8767, 0x26 }, { 9155, 0x28 }, { 9560, 0x2a }, { 9984, 0x2c }, { 10426, 0x2e }, { 10887, 0x30 }, { 11369, 0x32 }, { 11873, 0x34 }, { 12398, 0x36 }, { 12947, 0x38 }, { 13520, 0x3a }, { 14119, 0x3c }, { 14744, 0x3e }, { 15067, 0x40 }, { 15397, 0x42 }, { 15734, 0x44 }, { 16079, 0x46 }, { 16431, 0x48 }, { 16790, 0x4a }, { 17158, 0x4c }, { 17534, 0x4e }, { 17918, 0x50 }, { 18310, 0x52 }, { 18711, 0x54 }, { 19121, 0x56 }, { 19540, 0x58 }, { 19967, 0x5a }, { 20405, 0x5c }, { 20851, 0x5e }, { 21308, 0x60 }, { 21775, 0x62 }, { 22251, 0x64 }, { 22739, 0x66 }, { 23236, 0x68 }, { 23745, 0x6a }, { 24265, 0x6c }, { 24797, 0x6e }, { 25340, 0x70 }, { 25894, 0x72 }, { 26462, 0x74 }, { 27041, 0x76 }, { 27633, 0x78 }, { 28238, 0x7a }, { 28856, 0x7c }, { 29488, 0x7e }, { 30134, 0x80 }, { 30794, 0x82 }, { 31468, 0x84 }, { 32157, 0x86 }, { 32861, 0x88 }, { 33581, 0x8a }, { 34316, 0x8c }, { 35068, 0x8e }, { 35836, 0x90 }, { 36620, 0x92 }, { 37422, 0x94 }, { 38242, 0x96 }, { 39079, 0x98 }, { 39935, 0x9a }, { 40809, 0x9c }, { 41703, 0x9e }, { 42616, 0xa0 }, { 43549, 0xa2 }, { 44503, 0xa4 }, { 45477, 0xa6 }, { 46473, 0xa8 }, { 47491, 0xaa }, { 48531, 0xac }, { 49593, 0xae }, { 50679, 0xb0 }, { 51789, 0xb2 }, { 52923, 0xb4 }, { 54082, 0xb6 }, { 55266, 0xb8 }, { 56476, 0xba }, { 57713, 0xbc }, { 58977, 0xbe }, { 60268, 0xc0 }, { 61588, 0xc2 }, { 62936, 0xc4 }, { 64315, 0xc6 }, { 65535, 0xc8 } };
    int index = 0;
    for (; index < 101; index++) {
        if (amplitude <= hfa[index][0]) {
            return (Uint8)hfa[index][1];
        }
    }
    return (Uint8)hfa[100][1];
}

static Uint16 EncodeRumbleLowAmplitude(Uint16 amplitude)
{
    /* More information about these values can be found here:
     * https://github.com/dekuNukem/Nintendo_Switch_Reverse_Engineering/blob/master/rumble_data_table.md
     */
    Uint16 lfa[101][2] = { { 0, 0x0040 }, { 514, 0x8040 }, { 775, 0x0041 }, { 921, 0x8041 }, { 1096, 0x0042 }, { 1303, 0x8042 }, { 1550, 0x0043 }, { 1843, 0x8043 }, { 2192, 0x0044 }, { 2606, 0x8044 }, { 3100, 0x0045 }, { 3686, 0x8045 }, { 4383, 0x0046 }, { 5213, 0x8046 }, { 6199, 0x0047 }, { 7372, 0x8047 }, { 7698, 0x0048 }, { 8039, 0x8048 }, { 8395, 0x0049 }, { 8767, 0x8049 }, { 9155, 0x004a }, { 9560, 0x804a }, { 9984, 0x004b }, { 10426, 0x804b }, { 10887, 0x004c }, { 11369, 0x804c }, { 11873, 0x004d }, { 12398, 0x804d }, { 12947, 0x004e }, { 13520, 0x804e }, { 14119, 0x004f }, { 14744, 0x804f }, { 15067, 0x0050 }, { 15397, 0x8050 }, { 15734, 0x0051 }, { 16079, 0x8051 }, { 16431, 0x0052 }, { 16790, 0x8052 }, { 17158, 0x0053 }, { 17534, 0x8053 }, { 17918, 0x0054 }, { 18310, 0x8054 }, { 18711, 0x0055 }, { 19121, 0x8055 }, { 19540, 0x0056 }, { 19967, 0x8056 }, { 20405, 0x0057 }, { 20851, 0x8057 }, { 21308, 0x0058 }, { 21775, 0x8058 }, { 22251, 0x0059 }, { 22739, 0x8059 }, { 23236, 0x005a }, { 23745, 0x805a }, { 24265, 0x005b }, { 24797, 0x805b }, { 25340, 0x005c }, { 25894, 0x805c }, { 26462, 0x005d }, { 27041, 0x805d }, { 27633, 0x005e }, { 28238, 0x805e }, { 28856, 0x005f }, { 29488, 0x805f }, { 30134, 0x0060 }, { 30794, 0x8060 }, { 31468, 0x0061 }, { 32157, 0x8061 }, { 32861, 0x0062 }, { 33581, 0x8062 }, { 34316, 0x0063 }, { 35068, 0x8063 }, { 35836, 0x0064 }, { 36620, 0x8064 }, { 37422, 0x0065 }, { 38242, 0x8065 }, { 39079, 0x0066 }, { 39935, 0x8066 }, { 40809, 0x0067 }, { 41703, 0x8067 }, { 42616, 0x0068 }, { 43549, 0x8068 }, { 44503, 0x0069 }, { 45477, 0x8069 }, { 46473, 0x006a }, { 47491, 0x806a }, { 48531, 0x006b }, { 49593, 0x806b }, { 50679, 0x006c }, { 51789, 0x806c }, { 52923, 0x006d }, { 54082, 0x806d }, { 55266, 0x006e }, { 56476, 0x806e }, { 57713, 0x006f }, { 58977, 0x806f }, { 60268, 0x0070 }, { 61588, 0x8070 }, { 62936, 0x0071 }, { 64315, 0x8071 }, { 65535, 0x0072 } };
    int index = 0;
    for (; index < 101; index++) {
        if (amplitude <= lfa[index][0]) {
            return lfa[index][1];
        }
    }
    return lfa[100][1];
}

static void SetNeutralRumble(SwitchRumbleData_t *pRumble)
{
    pRumble->rgucData[0] = 0x00;
    pRumble->rgucData[1] = 0x01;
    pRumble->rgucData[2] = 0x40;
    pRumble->rgucData[3] = 0x40;
}

static void EncodeRumble(SwitchRumbleData_t *pRumble, Uint16 usHighFreq, Uint8 ucHighFreqAmp, Uint8 ucLowFreq, Uint16 usLowFreqAmp)
{
    if (ucHighFreqAmp > 0 || usLowFreqAmp > 0) {
        // High-band frequency and low-band amplitude are actually nine-bits each so they
        // take a bit from the high-band amplitude and low-band frequency bytes respectively
        pRumble->rgucData[0] = usHighFreq & 0xFF;
        pRumble->rgucData[1] = ucHighFreqAmp | ((usHighFreq >> 8) & 0x01);

        pRumble->rgucData[2] = ucLowFreq | ((usLowFreqAmp >> 8) & 0x80);
        pRumble->rgucData[3] = usLowFreqAmp & 0xFF;

#ifdef DEBUG_RUMBLE
        SDL_Log("Freq: %.2X %.2X  %.2X, Amp: %.2X  %.2X %.2X",
                usHighFreq & 0xFF, ((usHighFreq >> 8) & 0x01), ucLowFreq,
                ucHighFreqAmp, ((usLowFreqAmp >> 8) & 0x80), usLowFreqAmp & 0xFF);
#endif
    } else {
        SetNeutralRumble(pRumble);
    }
}

static bool WriteRumble(SDL_DriverSwitch_Context *ctx)
{
    /* Write into m_RumblePacket rather than a temporary buffer to allow the current rumble state
     * to be retained for subsequent rumble or subcommand packets sent to the controller
     */
    ctx->m_RumblePacket.ucPacketType = k_eSwitchOutputReportIDs_Rumble;
    ctx->m_RumblePacket.ucPacketNumber = ctx->m_nCommandNumber;
    ctx->m_nCommandNumber = (ctx->m_nCommandNumber + 1) & 0xF;

    // Refresh the rumble state periodically
    ctx->m_ulRumbleSent = SDL_GetTicks();

    return WritePacket(ctx, (Uint8 *)&ctx->m_RumblePacket, sizeof(ctx->m_RumblePacket));
}

static ESwitchDeviceInfoControllerType CalculateControllerType(SDL_DriverSwitch_Context *ctx, ESwitchDeviceInfoControllerType eControllerType)
{
    SDL_HIDAPI_Device *device = ctx->device;

    // The N64 controller reports as a Pro controller over USB
    if (eControllerType == k_eSwitchDeviceInfoControllerType_ProController &&
        device->product_id == USB_PRODUCT_NINTENDO_N64_CONTROLLER) {
        eControllerType = k_eSwitchDeviceInfoControllerType_N64;
    }

    if (eControllerType == k_eSwitchDeviceInfoControllerType_Unknown) {
        // This might be a Joy-Con that's missing from a charging grip slot
        if (device->product_id == USB_PRODUCT_NINTENDO_SWITCH_JOYCON_GRIP) {
            if (device->interface_number == 1) {
                eControllerType = k_eSwitchDeviceInfoControllerType_JoyConLeft;
            } else {
                eControllerType = k_eSwitchDeviceInfoControllerType_JoyConRight;
            }
        }
    }
    return eControllerType;
}

static bool BReadDeviceInfo(SDL_DriverSwitch_Context *ctx)
{
    SwitchSubcommandInputPacket_t *reply = NULL;

    if (ctx->device->is_bluetooth) {
        if (WriteSubcommand(ctx, k_eSwitchSubcommandIDs_RequestDeviceInfo, NULL, 0, &reply)) {
            // Byte 2: Controller ID (1=LJC, 2=RJC, 3=Pro)
            ctx->m_eControllerType = CalculateControllerType(ctx, (ESwitchDeviceInfoControllerType)reply->deviceInfo.ucDeviceType);

            // Bytes 4-9: MAC address (big-endian)
            SDL_memcpy(ctx->m_rgucMACAddress, reply->deviceInfo.rgucMACAddress, sizeof(ctx->m_rgucMACAddress));

            return true;
        }
    } else {
        if (WriteProprietary(ctx, k_eSwitchProprietaryCommandIDs_Status, NULL, 0, true)) {
            SwitchProprietaryStatusPacket_t *status = (SwitchProprietaryStatusPacket_t *)&ctx->m_rgucReadBuffer[0];
            size_t i;

            ctx->m_eControllerType = CalculateControllerType(ctx, (ESwitchDeviceInfoControllerType)status->ucDeviceType);

            for (i = 0; i < sizeof(ctx->m_rgucMACAddress); ++i) {
                ctx->m_rgucMACAddress[i] = status->rgucMACAddress[sizeof(ctx->m_rgucMACAddress) - i - 1];
            }

            return true;
        }
    }
    return false;
}

static bool BTrySetupUSB(SDL_DriverSwitch_Context *ctx)
{
    /* We have to send a connection handshake to the controller when communicating over USB
     * before we're able to send it other commands. Luckily this command is not supported
     * over Bluetooth, so we can use the controller's lack of response as a way to
     * determine if the connection is over USB or Bluetooth
     */
    if (!WriteProprietary(ctx, k_eSwitchProprietaryCommandIDs_Handshake, NULL, 0, true)) {
        return false;
    }
    if (!WriteProprietary(ctx, k_eSwitchProprietaryCommandIDs_HighSpeed, NULL, 0, true)) {
        // The 8BitDo M30 and SF30 Pro don't respond to this command, but otherwise work correctly
        // return false;
    }
    if (!WriteProprietary(ctx, k_eSwitchProprietaryCommandIDs_Handshake, NULL, 0, true)) {
        // This fails on the right Joy-Con when plugged into the charging grip
        // return false;
    }
    if (!WriteProprietary(ctx, k_eSwitchProprietaryCommandIDs_ForceUSB, NULL, 0, false)) {
        return false;
    }
    return true;
}

static bool SetVibrationEnabled(SDL_DriverSwitch_Context *ctx, Uint8 enabled)
{
    return WriteSubcommand(ctx, k_eSwitchSubcommandIDs_EnableVibration, &enabled, sizeof(enabled), NULL);
}
static bool SetInputMode(SDL_DriverSwitch_Context *ctx, Uint8 input_mode)
{
#ifdef FORCE_SIMPLE_REPORTS
    input_mode = k_eSwitchInputReportIDs_SimpleControllerState;
#endif
#ifdef FORCE_FULL_REPORTS
    input_mode = k_eSwitchInputReportIDs_FullControllerState;
#endif

    if (input_mode == ctx->m_nCurrentInputMode) {
        return true;
    } else {
        ctx->m_nCurrentInputMode = input_mode;

        return WriteSubcommand(ctx, k_eSwitchSubcommandIDs_SetInputReportMode, &input_mode, sizeof(input_mode), NULL);
    }
}

static bool SetHomeLED(SDL_DriverSwitch_Context *ctx, Uint8 brightness)
{
    Uint8 ucLedIntensity = 0;
    Uint8 rgucBuffer[4];

    if (brightness > 0) {
        if (brightness < 65) {
            ucLedIntensity = (brightness + 5) / 10;
        } else {
            ucLedIntensity = (Uint8)SDL_ceilf(0xF * SDL_powf((float)brightness / 100.f, 2.13f));
        }
    }

    rgucBuffer[0] = (0x0 << 4) | 0x1;                    // 0 mini cycles (besides first), cycle duration 8ms
    rgucBuffer[1] = ((ucLedIntensity & 0xF) << 4) | 0x0; // LED start intensity (0x0-0xF), 0 cycles (LED stays on at start intensity after first cycle)
    rgucBuffer[2] = ((ucLedIntensity & 0xF) << 4) | 0x0; // First cycle LED intensity, 0x0 intensity for second cycle
    rgucBuffer[3] = (0x0 << 4) | 0x0;                    // 8ms fade transition to first cycle, 8ms first cycle LED duration

    return WriteSubcommand(ctx, k_eSwitchSubcommandIDs_SetHomeLight, rgucBuffer, sizeof(rgucBuffer), NULL);
}

static void SDLCALL SDL_HomeLEDHintChanged(void *userdata, const char *name, const char *oldValue, const char *hint)
{
    SDL_DriverSwitch_Context *ctx = (SDL_DriverSwitch_Context *)userdata;

    if (hint && *hint) {
        int value;

        if (SDL_strchr(hint, '.') != NULL) {
            value = (int)(100.0f * SDL_atof(hint));
            if (value > 255) {
                value = 255;
            }
        } else if (SDL_GetStringBoolean(hint, true)) {
            value = 100;
        } else {
            value = 0;
        }
        SetHomeLED(ctx, (Uint8)value);
    }
}

static void UpdateSlotLED(SDL_DriverSwitch_Context *ctx)
{
    if (!ctx->m_bInputOnly) {
        Uint8 led_data = 0;

        if (ctx->m_bPlayerLights && ctx->m_nPlayerIndex >= 0) {
            led_data = (1 << (ctx->m_nPlayerIndex % 4));
        }
        WriteSubcommand(ctx, k_eSwitchSubcommandIDs_SetPlayerLights, &led_data, sizeof(led_data), NULL);
    }
}

static void SDLCALL SDL_PlayerLEDHintChanged(void *userdata, const char *name, const char *oldValue, const char *hint)
{
    SDL_DriverSwitch_Context *ctx = (SDL_DriverSwitch_Context *)userdata;
    bool bPlayerLights = SDL_GetStringBoolean(hint, true);

    if (bPlayerLights != ctx->m_bPlayerLights) {
        ctx->m_bPlayerLights = bPlayerLights;

        UpdateSlotLED(ctx);
        HIDAPI_UpdateDeviceProperties(ctx->device);
    }
}

static void GetInitialInputMode(SDL_DriverSwitch_Context *ctx)
{
    if (!ctx->m_nInitialInputMode) {
        // This will set the initial input mode if it can
        ReadInput(ctx);
    }
}

static Uint8 GetDefaultInputMode(SDL_DriverSwitch_Context *ctx)
{
    Uint8 input_mode;

    // Determine the desired input mode
    if (ctx->m_nInitialInputMode) {
        input_mode = ctx->m_nInitialInputMode;
    } else {
        if (ctx->device->is_bluetooth) {
            input_mode = k_eSwitchInputReportIDs_SimpleControllerState;
        } else {
            input_mode = k_eSwitchInputReportIDs_FullControllerState;
        }
    }

    switch (ctx->m_eEnhancedReportHint) {
    case SWITCH_ENHANCED_REPORT_HINT_OFF:
        input_mode = k_eSwitchInputReportIDs_SimpleControllerState;
        break;
    case SWITCH_ENHANCED_REPORT_HINT_ON:
        if (input_mode == k_eSwitchInputReportIDs_SimpleControllerState) {
            input_mode = k_eSwitchInputReportIDs_FullControllerState;
        }
        break;
    case SWITCH_ENHANCED_REPORT_HINT_AUTO:
        /* Joy-Con controllers switch their thumbsticks into D-pad mode in simple mode,
         * so let's enable full controller state for them.
         */
        if (ctx->device->vendor_id == USB_VENDOR_NINTENDO &&
            (ctx->device->product_id == USB_PRODUCT_NINTENDO_SWITCH_JOYCON_LEFT ||
             ctx->device->product_id == USB_PRODUCT_NINTENDO_SWITCH_JOYCON_RIGHT)) {
            input_mode = k_eSwitchInputReportIDs_FullControllerState;
        }
        break;
    }

    // Wired controllers break if they are put into simple controller state
    if (input_mode == k_eSwitchInputReportIDs_SimpleControllerState &&
        !ctx->device->is_bluetooth) {
        input_mode = k_eSwitchInputReportIDs_FullControllerState;
    }
    return input_mode;
}

static Uint8 GetSensorInputMode(SDL_DriverSwitch_Context *ctx)
{
    Uint8 input_mode;

    // Determine the desired input mode
    if (!ctx->m_nInitialInputMode ||
        ctx->m_nInitialInputMode == k_eSwitchInputReportIDs_SimpleControllerState) {
        input_mode = k_eSwitchInputReportIDs_FullControllerState;
    } else {
        input_mode = ctx->m_nInitialInputMode;
    }
    return input_mode;
}

static void UpdateInputMode(SDL_DriverSwitch_Context *ctx)
{
    Uint8 input_mode;

    if (ctx->m_bReportSensors) {
        input_mode = GetSensorInputMode(ctx);
    } else {
        input_mode = GetDefaultInputMode(ctx);
    }
    SetInputMode(ctx, input_mode);
}

static void SetEnhancedModeAvailable(SDL_DriverSwitch_Context *ctx)
{
    if (ctx->m_bEnhancedModeAvailable) {
        return;
    }
    ctx->m_bEnhancedModeAvailable = true;

    if (ctx->m_bSensorsSupported) {
        // Use the right sensor in the combined Joy-Con pair
        if (!ctx->device->parent ||
            ctx->m_eControllerType == k_eSwitchDeviceInfoControllerType_JoyConRight) {
            SDL_PrivateJoystickAddSensor(ctx->joystick, SDL_SENSOR_GYRO, 200.0f);
            SDL_PrivateJoystickAddSensor(ctx->joystick, SDL_SENSOR_ACCEL, 200.0f);
        }
        if (ctx->device->parent &&
            ctx->m_eControllerType == k_eSwitchDeviceInfoControllerType_JoyConLeft) {
            SDL_PrivateJoystickAddSensor(ctx->joystick, SDL_SENSOR_GYRO_L, 200.0f);
            SDL_PrivateJoystickAddSensor(ctx->joystick, SDL_SENSOR_ACCEL_L, 200.0f);
        }
        if (ctx->device->parent &&
            ctx->m_eControllerType == k_eSwitchDeviceInfoControllerType_JoyConRight) {
            SDL_PrivateJoystickAddSensor(ctx->joystick, SDL_SENSOR_GYRO_R, 200.0f);
            SDL_PrivateJoystickAddSensor(ctx->joystick, SDL_SENSOR_ACCEL_R, 200.0f);
        }
    }
}

static void SetEnhancedReportHint(SDL_DriverSwitch_Context *ctx, HIDAPI_Switch_EnhancedReportHint eEnhancedReportHint)
{
    ctx->m_eEnhancedReportHint = eEnhancedReportHint;

    switch (eEnhancedReportHint) {
    case SWITCH_ENHANCED_REPORT_HINT_OFF:
        ctx->m_bEnhancedMode = false;
        break;
    case SWITCH_ENHANCED_REPORT_HINT_ON:
        SetEnhancedModeAvailable(ctx);
        ctx->m_bEnhancedMode = true;
        break;
    case SWITCH_ENHANCED_REPORT_HINT_AUTO:
        SetEnhancedModeAvailable(ctx);
        break;
    }

    UpdateInputMode(ctx);
}

static void UpdateEnhancedModeOnEnhancedReport(SDL_DriverSwitch_Context *ctx)
{
    if (ctx->m_eEnhancedReportHint == SWITCH_ENHANCED_REPORT_HINT_AUTO) {
        SetEnhancedReportHint(ctx, SWITCH_ENHANCED_REPORT_HINT_ON);
    }
}

static void UpdateEnhancedModeOnApplicationUsage(SDL_DriverSwitch_Context *ctx)
{
    if (ctx->m_eEnhancedReportHint == SWITCH_ENHANCED_REPORT_HINT_AUTO) {
        SetEnhancedReportHint(ctx, SWITCH_ENHANCED_REPORT_HINT_ON);
    }
}

static void SDLCALL SDL_EnhancedReportsChanged(void *userdata, const char *name, const char *oldValue, const char *hint)
{
    SDL_DriverSwitch_Context *ctx = (SDL_DriverSwitch_Context *)userdata;

    if (hint && SDL_strcasecmp(hint, "auto") == 0) {
        SetEnhancedReportHint(ctx, SWITCH_ENHANCED_REPORT_HINT_AUTO);
    } else if (SDL_GetStringBoolean(hint, true)) {
        SetEnhancedReportHint(ctx, SWITCH_ENHANCED_REPORT_HINT_ON);
    } else {
        SetEnhancedReportHint(ctx, SWITCH_ENHANCED_REPORT_HINT_OFF);
    }
}

static bool SetIMUEnabled(SDL_DriverSwitch_Context *ctx, bool enabled)
{
    Uint8 imu_data = enabled ? 1 : 0;
    return WriteSubcommand(ctx, k_eSwitchSubcommandIDs_EnableIMU, &imu_data, sizeof(imu_data), NULL);
}

static bool LoadStickCalibration(SDL_DriverSwitch_Context *ctx)
{
    Uint8 *pLeftStickCal = NULL;
    Uint8 *pRightStickCal = NULL;
    size_t stick, axis;
    SwitchSubcommandInputPacket_t *user_reply = NULL;
    SwitchSubcommandInputPacket_t *factory_reply = NULL;
    SwitchSPIOpData_t readUserParams;
    SwitchSPIOpData_t readFactoryParams;
    Uint8 userParamsReadSuccessCount = 0;

    // Read User Calibration Info
    readUserParams.unAddress = k_unSPIStickUserCalibrationStartOffset;
    readUserParams.ucLength = k_unSPIStickUserCalibrationLength;

    // This isn't readable on all controllers, so ignore failure
    WriteSubcommand(ctx, k_eSwitchSubcommandIDs_SPIFlashRead, (uint8_t *)&readUserParams, sizeof(readUserParams), &user_reply);

    // Read Factory Calibration Info
    readFactoryParams.unAddress = k_unSPIStickFactoryCalibrationStartOffset;
    readFactoryParams.ucLength = k_unSPIStickFactoryCalibrationLength;

    // Automatically select the user calibration if magic bytes are set
    if (user_reply && user_reply->stickUserCalibration.rgucLeftMagic[0] == 0xB2 && user_reply->stickUserCalibration.rgucLeftMagic[1] == 0xA1) {
        userParamsReadSuccessCount += 1;
        pLeftStickCal = user_reply->stickUserCalibration.rgucLeftCalibration;
    }

    if (user_reply && user_reply->stickUserCalibration.rgucRightMagic[0] == 0xB2 && user_reply->stickUserCalibration.rgucRightMagic[1] == 0xA1) {
        userParamsReadSuccessCount += 1;
        pRightStickCal = user_reply->stickUserCalibration.rgucRightCalibration;
    } 

    // Only read the factory calibration info if we failed to receive the correct magic bytes
    if (userParamsReadSuccessCount < 2) {
        // Read Factory Calibration Info
        readFactoryParams.unAddress = k_unSPIStickFactoryCalibrationStartOffset;
        readFactoryParams.ucLength = k_unSPIStickFactoryCalibrationLength;

        const int MAX_ATTEMPTS = 3;
        for (int attempt = 0;; ++attempt) {
            if (!WriteSubcommand(ctx, k_eSwitchSubcommandIDs_SPIFlashRead, (uint8_t *)&readFactoryParams, sizeof(readFactoryParams), &factory_reply)) {
                return false;
            }

            if (factory_reply->stickFactoryCalibration.opData.unAddress == k_unSPIStickFactoryCalibrationStartOffset) {
                // We successfully read the calibration data
                pLeftStickCal = factory_reply->stickFactoryCalibration.rgucLeftCalibration;
                pRightStickCal = factory_reply->stickFactoryCalibration.rgucRightCalibration;
                break;
            }

            if (attempt == MAX_ATTEMPTS) {
                return false;
            }
        }
    }

    // If we still don't have calibration data, return false
    if (pLeftStickCal == NULL || pRightStickCal == NULL)
    {
        return false;
    }

    /* Stick calibration values are 12-bits each and are packed by bit
     * For whatever reason the fields are in a different order for each stick
     * Left:  X-Max, Y-Max, X-Center, Y-Center, X-Min, Y-Min
     * Right: X-Center, Y-Center, X-Min, Y-Min, X-Max, Y-Max
     */

    // Left stick
    ctx->m_StickCalData[0].axis[0].sMax = ((pLeftStickCal[1] << 8) & 0xF00) | pLeftStickCal[0];    // X Axis max above center
    ctx->m_StickCalData[0].axis[1].sMax = (pLeftStickCal[2] << 4) | (pLeftStickCal[1] >> 4);       // Y Axis max above center
    ctx->m_StickCalData[0].axis[0].sCenter = ((pLeftStickCal[4] << 8) & 0xF00) | pLeftStickCal[3]; // X Axis center
    ctx->m_StickCalData[0].axis[1].sCenter = (pLeftStickCal[5] << 4) | (pLeftStickCal[4] >> 4);    // Y Axis center
    ctx->m_StickCalData[0].axis[0].sMin = ((pLeftStickCal[7] << 8) & 0xF00) | pLeftStickCal[6];    // X Axis min below center
    ctx->m_StickCalData[0].axis[1].sMin = (pLeftStickCal[8] << 4) | (pLeftStickCal[7] >> 4);       // Y Axis min below center

    // Right stick
    ctx->m_StickCalData[1].axis[0].sCenter = ((pRightStickCal[1] << 8) & 0xF00) | pRightStickCal[0]; // X Axis center
    ctx->m_StickCalData[1].axis[1].sCenter = (pRightStickCal[2] << 4) | (pRightStickCal[1] >> 4);    // Y Axis center
    ctx->m_StickCalData[1].axis[0].sMin = ((pRightStickCal[4] << 8) & 0xF00) | pRightStickCal[3];    // X Axis min below center
    ctx->m_StickCalData[1].axis[1].sMin = (pRightStickCal[5] << 4) | (pRightStickCal[4] >> 4);       // Y Axis min below center
    ctx->m_StickCalData[1].axis[0].sMax = ((pRightStickCal[7] << 8) & 0xF00) | pRightStickCal[6];    // X Axis max above center
    ctx->m_StickCalData[1].axis[1].sMax = (pRightStickCal[8] << 4) | (pRightStickCal[7] >> 4);       // Y Axis max above center

    // Filter out any values that were uninitialized (0xFFF) in the SPI read
    for (stick = 0; stick < 2; ++stick) {
        for (axis = 0; axis < 2; ++axis) {
            if (ctx->m_StickCalData[stick].axis[axis].sCenter == 0xFFF) {
                ctx->m_StickCalData[stick].axis[axis].sCenter = 2048;
            }
            if (ctx->m_StickCalData[stick].axis[axis].sMax == 0xFFF) {
                ctx->m_StickCalData[stick].axis[axis].sMax = (Sint16)(ctx->m_StickCalData[stick].axis[axis].sCenter * 0.7f);
            }
            if (ctx->m_StickCalData[stick].axis[axis].sMin == 0xFFF) {
                ctx->m_StickCalData[stick].axis[axis].sMin = (Sint16)(ctx->m_StickCalData[stick].axis[axis].sCenter * 0.7f);
            }
        }
    }

    for (stick = 0; stick < 2; ++stick) {
        for (axis = 0; axis < 2; ++axis) {
            ctx->m_StickExtents[stick].axis[axis].sMin = -(Sint16)(ctx->m_StickCalData[stick].axis[axis].sMin * 0.7f);
            ctx->m_StickExtents[stick].axis[axis].sMax = (Sint16)(ctx->m_StickCalData[stick].axis[axis].sMax * 0.7f);
        }
    }

    for (stick = 0; stick < 2; ++stick) {
        for (axis = 0; axis < 2; ++axis) {
            ctx->m_SimpleStickExtents[stick].axis[axis].sMin = (Sint16)(SDL_MIN_SINT16 * 0.5f);
            ctx->m_SimpleStickExtents[stick].axis[axis].sMax = (Sint16)(SDL_MAX_SINT16 * 0.5f);
        }
    }

    return true;
}

static bool LoadIMUCalibration(SDL_DriverSwitch_Context *ctx)
{
    SwitchSubcommandInputPacket_t *reply = NULL;

    // Read Calibration Info
    SwitchSPIOpData_t readParams;
    readParams.unAddress = k_unSPIIMUScaleStartOffset;
    readParams.ucLength = k_unSPIIMUScaleLength;

    if (WriteSubcommand(ctx, k_eSwitchSubcommandIDs_SPIFlashRead, (uint8_t *)&readParams, sizeof(readParams), &reply)) {
        Uint8 *pIMUScale;
        Sint16 sAccelRawX, sAccelRawY, sAccelRawZ, sGyroRawX, sGyroRawY, sGyroRawZ;
        Sint16 sAccelSensCoeffX, sAccelSensCoeffY, sAccelSensCoeffZ;
        Sint16 sGyroSensCoeffX, sGyroSensCoeffY, sGyroSensCoeffZ;

        // IMU scale gives us multipliers for converting raw values to real world values
        pIMUScale = reply->spiReadData.rgucReadData;

        sAccelRawX = (pIMUScale[1] << 8) | pIMUScale[0];
        sAccelRawY = (pIMUScale[3] << 8) | pIMUScale[2];
        sAccelRawZ = (pIMUScale[5] << 8) | pIMUScale[4];

        sAccelSensCoeffX = (pIMUScale[7] << 8) | pIMUScale[6];
        sAccelSensCoeffY = (pIMUScale[9] << 8) | pIMUScale[8];
        sAccelSensCoeffZ = (pIMUScale[11] << 8) | pIMUScale[10];

        sGyroRawX = (pIMUScale[13] << 8) | pIMUScale[12];
        sGyroRawY = (pIMUScale[15] << 8) | pIMUScale[14];
        sGyroRawZ = (pIMUScale[17] << 8) | pIMUScale[16];

        sGyroSensCoeffX = (pIMUScale[19] << 8) | pIMUScale[18];
        sGyroSensCoeffY = (pIMUScale[21] << 8) | pIMUScale[20];
        sGyroSensCoeffZ = (pIMUScale[23] << 8) | pIMUScale[22];

        // Check for user calibration data. If it's present and set, it'll override the factory settings
        readParams.unAddress = k_unSPIIMUUserScaleStartOffset;
        readParams.ucLength = k_unSPIIMUUserScaleLength;
        if (WriteSubcommand(ctx, k_eSwitchSubcommandIDs_SPIFlashRead, (uint8_t *)&readParams, sizeof(readParams), &reply) && (pIMUScale[0] | pIMUScale[1] << 8) == 0xA1B2) {
            pIMUScale = reply->spiReadData.rgucReadData;

            sAccelRawX = (pIMUScale[3] << 8) | pIMUScale[2];
            sAccelRawY = (pIMUScale[5] << 8) | pIMUScale[4];
            sAccelRawZ = (pIMUScale[7] << 8) | pIMUScale[6];

            sGyroRawX = (pIMUScale[15] << 8) | pIMUScale[14];
            sGyroRawY = (pIMUScale[17] << 8) | pIMUScale[16];
            sGyroRawZ = (pIMUScale[19] << 8) | pIMUScale[18];
        }

        // Accelerometer scale
        ctx->m_IMUScaleData.fAccelScaleX = SWITCH_ACCEL_SCALE_MULT / ((float)sAccelSensCoeffX - (float)sAccelRawX) * SDL_STANDARD_GRAVITY;
        ctx->m_IMUScaleData.fAccelScaleY = SWITCH_ACCEL_SCALE_MULT / ((float)sAccelSensCoeffY - (float)sAccelRawY) * SDL_STANDARD_GRAVITY;
        ctx->m_IMUScaleData.fAccelScaleZ = SWITCH_ACCEL_SCALE_MULT / ((float)sAccelSensCoeffZ - (float)sAccelRawZ) * SDL_STANDARD_GRAVITY;

        // Gyro scale
        ctx->m_IMUScaleData.fGyroScaleX = SWITCH_GYRO_SCALE_MULT / ((float)sGyroSensCoeffX - (float)sGyroRawX) * SDL_PI_F / 180.0f;
        ctx->m_IMUScaleData.fGyroScaleY = SWITCH_GYRO_SCALE_MULT / ((float)sGyroSensCoeffY - (float)sGyroRawY) * SDL_PI_F / 180.0f;
        ctx->m_IMUScaleData.fGyroScaleZ = SWITCH_GYRO_SCALE_MULT / ((float)sGyroSensCoeffZ - (float)sGyroRawZ) * SDL_PI_F / 180.0f;

    } else {
        // Use default values
        const float accelScale = SDL_STANDARD_GRAVITY / SWITCH_ACCEL_SCALE;
        const float gyroScale = SDL_PI_F / 180.0f / SWITCH_GYRO_SCALE;

        ctx->m_IMUScaleData.fAccelScaleX = accelScale;
        ctx->m_IMUScaleData.fAccelScaleY = accelScale;
        ctx->m_IMUScaleData.fAccelScaleZ = accelScale;

        ctx->m_IMUScaleData.fGyroScaleX = gyroScale;
        ctx->m_IMUScaleData.fGyroScaleY = gyroScale;
        ctx->m_IMUScaleData.fGyroScaleZ = gyroScale;
    }
    return true;
}

static Sint16 ApplyStickCalibration(SDL_DriverSwitch_Context *ctx, int nStick, int nAxis, Sint16 sRawValue)
{
    sRawValue -= ctx->m_StickCalData[nStick].axis[nAxis].sCenter;

    if (sRawValue >= 0) {
        if (sRawValue > ctx->m_StickExtents[nStick].axis[nAxis].sMax) {
            ctx->m_StickExtents[nStick].axis[nAxis].sMax = sRawValue;
        }
        return (Sint16)HIDAPI_RemapVal(sRawValue, 0, ctx->m_StickExtents[nStick].axis[nAxis].sMax, 0, SDL_MAX_SINT16);
    } else {
        if (sRawValue < ctx->m_StickExtents[nStick].axis[nAxis].sMin) {
            ctx->m_StickExtents[nStick].axis[nAxis].sMin = sRawValue;
        }
        return (Sint16)HIDAPI_RemapVal(sRawValue, ctx->m_StickExtents[nStick].axis[nAxis].sMin, 0, SDL_MIN_SINT16, 0);
    }
}

static Sint16 ApplySimpleStickCalibration(SDL_DriverSwitch_Context *ctx, int nStick, int nAxis, Sint16 sRawValue)
{
    // 0x8000 is the neutral value for all joystick axes
    const Uint16 usJoystickCenter = 0x8000;

    sRawValue -= usJoystickCenter;

    if (sRawValue >= 0) {
        if (sRawValue > ctx->m_SimpleStickExtents[nStick].axis[nAxis].sMax) {
            ctx->m_SimpleStickExtents[nStick].axis[nAxis].sMax = sRawValue;
        }
        return (Sint16)HIDAPI_RemapVal(sRawValue, 0, ctx->m_SimpleStickExtents[nStick].axis[nAxis].sMax, 0, SDL_MAX_SINT16);
    } else {
        if (sRawValue < ctx->m_SimpleStickExtents[nStick].axis[nAxis].sMin) {
            ctx->m_SimpleStickExtents[nStick].axis[nAxis].sMin = sRawValue;
        }
        return (Sint16)HIDAPI_RemapVal(sRawValue, ctx->m_SimpleStickExtents[nStick].axis[nAxis].sMin, 0, SDL_MIN_SINT16, 0);
    }
}

static Uint8 RemapButton(SDL_DriverSwitch_Context *ctx, Uint8 button)
{
    if (ctx->m_bUseButtonLabels) {
        // Use button labels instead of positions, e.g. Nintendo Online Classic controllers
        switch (button) {
        case SDL_GAMEPAD_BUTTON_SOUTH:
            return SDL_GAMEPAD_BUTTON_EAST;
        case SDL_GAMEPAD_BUTTON_EAST:
            return SDL_GAMEPAD_BUTTON_SOUTH;
        case SDL_GAMEPAD_BUTTON_WEST:
            return SDL_GAMEPAD_BUTTON_NORTH;
        case SDL_GAMEPAD_BUTTON_NORTH:
            return SDL_GAMEPAD_BUTTON_WEST;
        default:
            break;
        }
    }
    return button;
}

static int GetMaxWriteAttempts(SDL_HIDAPI_Device *device)
{
    if (device->vendor_id == USB_VENDOR_NINTENDO &&
        device->product_id == USB_PRODUCT_NINTENDO_SWITCH_JOYCON_GRIP) {
        // This device is a little slow and we know we're always on USB
        return 20;
    } else {
        return 5;
    }
}

static ESwitchDeviceInfoControllerType ReadJoyConControllerType(SDL_HIDAPI_Device *device)
{
    ESwitchDeviceInfoControllerType eControllerType = k_eSwitchDeviceInfoControllerType_Unknown;
    const int MAX_ATTEMPTS = 1; // Don't try too long, in case this is a zombie Bluetooth controller
    int attempts = 0;

    // Create enough of a context to read the controller type from the device
    SDL_DriverSwitch_Context *ctx = (SDL_DriverSwitch_Context *)SDL_calloc(1, sizeof(*ctx));
    if (ctx) {
        ctx->device = device;
        ctx->m_bSyncWrite = true;
        ctx->m_nMaxWriteAttempts = GetMaxWriteAttempts(device);

        for ( ; ; ) {
            ++attempts;
            if (device->is_bluetooth) {
                SwitchSubcommandInputPacket_t *reply = NULL;

                if (WriteSubcommand(ctx, k_eSwitchSubcommandIDs_RequestDeviceInfo, NULL, 0, &reply)) {
                    eControllerType = CalculateControllerType(ctx, (ESwitchDeviceInfoControllerType)reply->deviceInfo.ucDeviceType);
                }
            } else {
                if (WriteProprietary(ctx, k_eSwitchProprietaryCommandIDs_Status, NULL, 0, true)) {
                    SwitchProprietaryStatusPacket_t *status = (SwitchProprietaryStatusPacket_t *)&ctx->m_rgucReadBuffer[0];

                    eControllerType = CalculateControllerType(ctx, (ESwitchDeviceInfoControllerType)status->ucDeviceType);
                }
            }
            if (eControllerType == k_eSwitchDeviceInfoControllerType_Unknown && attempts < MAX_ATTEMPTS) {
                // Wait a bit and try again
                SDL_Delay(100);
                continue;
            }
            break;
        }
        SDL_free(ctx);
    }
    return eControllerType;
}

static bool HasHomeLED(SDL_DriverSwitch_Context *ctx)
{
    Uint16 vendor_id = ctx->device->vendor_id;
    Uint16 product_id = ctx->device->product_id;

    // The Power A Nintendo Switch Pro controllers don't have a Home LED
    if (vendor_id == 0 && product_id == 0) {
        return false;
    }

    // HORI Wireless Switch Pad
    if (vendor_id == 0x0f0d && product_id == 0x00f6) {
        return false;
    }

    // Third party controllers don't have a home LED and will shut off if we try to set it
    if (ctx->m_eControllerType == k_eSwitchDeviceInfoControllerType_Unknown ||
        ctx->m_eControllerType == k_eSwitchDeviceInfoControllerType_LicProController) {
        return false;
    }

    // The Nintendo Online classic controllers don't have a Home LED
    if (vendor_id == USB_VENDOR_NINTENDO &&
        ctx->m_eControllerType > k_eSwitchDeviceInfoControllerType_ProController) {
        return false;
    }

    return true;
}

static bool AlwaysUsesLabels(Uint16 vendor_id, Uint16 product_id, ESwitchDeviceInfoControllerType eControllerType)
{
    // Some controllers don't have a diamond button configuration, so should always use labels
    if (SDL_IsJoystickGameCube(vendor_id, product_id)) {
        return true;
    }
    switch (eControllerType) {
    case k_eSwitchDeviceInfoControllerType_HVCLeft:
    case k_eSwitchDeviceInfoControllerType_HVCRight:
    case k_eSwitchDeviceInfoControllerType_NESLeft:
    case k_eSwitchDeviceInfoControllerType_NESRight:
    case k_eSwitchDeviceInfoControllerType_N64:
    case k_eSwitchDeviceInfoControllerType_SEGA_Genesis:
        return true;
    default:
        return false;
    }
}

static void HIDAPI_DriverNintendoClassic_RegisterHints(SDL_HintCallback callback, void *userdata)
{
    SDL_AddHintCallback(SDL_HINT_JOYSTICK_HIDAPI_NINTENDO_CLASSIC, callback, userdata);
}

static void HIDAPI_DriverNintendoClassic_UnregisterHints(SDL_HintCallback callback, void *userdata)
{
    SDL_RemoveHintCallback(SDL_HINT_JOYSTICK_HIDAPI_NINTENDO_CLASSIC, callback, userdata);
}

static bool HIDAPI_DriverNintendoClassic_IsEnabled(void)
{
    return SDL_GetHintBoolean(SDL_HINT_JOYSTICK_HIDAPI_NINTENDO_CLASSIC, SDL_GetHintBoolean(SDL_HINT_JOYSTICK_HIDAPI, SDL_HIDAPI_DEFAULT));
}

static bool HIDAPI_DriverNintendoClassic_IsSupportedDevice(SDL_HIDAPI_Device *device, const char *name, SDL_GamepadType type, Uint16 vendor_id, Uint16 product_id, Uint16 version, int interface_number, int interface_class, int interface_subclass, int interface_protocol)
{
    if (vendor_id == USB_VENDOR_NINTENDO) {
        if (product_id == USB_PRODUCT_NINTENDO_SWITCH_JOYCON_RIGHT) {
            if (SDL_strncmp(name, "NES Controller", 14) == 0 ||
                SDL_strncmp(name, "HVC Controller", 14) == 0) {
                return true;
            }
        }

        if (product_id == USB_PRODUCT_NINTENDO_N64_CONTROLLER) {
            return true;
        }

        if (product_id == USB_PRODUCT_NINTENDO_SEGA_GENESIS_CONTROLLER) {
            return true;
        }

        if (product_id == USB_PRODUCT_NINTENDO_SNES_CONTROLLER) {
            return true;
        }
    }

    return false;
}

static void HIDAPI_DriverJoyCons_RegisterHints(SDL_HintCallback callback, void *userdata)
{
    SDL_AddHintCallback(SDL_HINT_JOYSTICK_HIDAPI_JOY_CONS, callback, userdata);
}

static void HIDAPI_DriverJoyCons_UnregisterHints(SDL_HintCallback callback, void *userdata)
{
    SDL_RemoveHintCallback(SDL_HINT_JOYSTICK_HIDAPI_JOY_CONS, callback, userdata);
}

static bool HIDAPI_DriverJoyCons_IsEnabled(void)
{
    return SDL_GetHintBoolean(SDL_HINT_JOYSTICK_HIDAPI_JOY_CONS, SDL_GetHintBoolean(SDL_HINT_JOYSTICK_HIDAPI, SDL_HIDAPI_DEFAULT));
}

static bool HIDAPI_DriverJoyCons_IsSupportedDevice(SDL_HIDAPI_Device *device, const char *name, SDL_GamepadType type, Uint16 vendor_id, Uint16 product_id, Uint16 version, int interface_number, int interface_class, int interface_subclass, int interface_protocol)
{
    if (vendor_id == USB_VENDOR_NINTENDO) {
        if (product_id == USB_PRODUCT_NINTENDO_SWITCH_PRO && device && device->dev) {
            // This might be a Kinvoca Joy-Con that reports VID/PID as a Switch Pro controller
            ESwitchDeviceInfoControllerType eControllerType = ReadJoyConControllerType(device);
            if (eControllerType == k_eSwitchDeviceInfoControllerType_JoyConLeft ||
                eControllerType == k_eSwitchDeviceInfoControllerType_JoyConRight) {
                return true;
            }
        }

        if (product_id == USB_PRODUCT_NINTENDO_SWITCH_JOYCON_LEFT ||
            product_id == USB_PRODUCT_NINTENDO_SWITCH_JOYCON_RIGHT ||
            product_id == USB_PRODUCT_NINTENDO_SWITCH_JOYCON_GRIP) {
            return true;
        }
    }
    return false;
}

static void HIDAPI_DriverSwitch_RegisterHints(SDL_HintCallback callback, void *userdata)
{
    SDL_AddHintCallback(SDL_HINT_JOYSTICK_HIDAPI_SWITCH, callback, userdata);
}

static void HIDAPI_DriverSwitch_UnregisterHints(SDL_HintCallback callback, void *userdata)
{
    SDL_RemoveHintCallback(SDL_HINT_JOYSTICK_HIDAPI_SWITCH, callback, userdata);
}

static bool HIDAPI_DriverSwitch_IsEnabled(void)
{
    return SDL_GetHintBoolean(SDL_HINT_JOYSTICK_HIDAPI_SWITCH, SDL_GetHintBoolean(SDL_HINT_JOYSTICK_HIDAPI, SDL_HIDAPI_DEFAULT));
}

static bool HIDAPI_DriverSwitch_IsSupportedDevice(SDL_HIDAPI_Device *device, const char *name, SDL_GamepadType type, Uint16 vendor_id, Uint16 product_id, Uint16 version, int interface_number, int interface_class, int interface_subclass, int interface_protocol)
{
    /* The HORI Wireless Switch Pad enumerates as a HID device when connected via USB
       with the same VID/PID as when connected over Bluetooth but doesn't actually
       support communication over USB. The most reliable way to block this without allowing the
       controller to continually attempt to reconnect is to filter it out by manufacturer/product string.
       Note that the controller does have a different product string when connected over Bluetooth.
     */
    if (SDL_strcmp(name, "HORI Wireless Switch Pad") == 0) {
        return false;
    }

    // If it's handled by another driver, it's not handled here
    if (HIDAPI_DriverNintendoClassic_IsSupportedDevice(device, name, type, vendor_id, product_id, version, interface_number, interface_class, interface_subclass, interface_protocol) ||
        HIDAPI_DriverJoyCons_IsSupportedDevice(device, name, type, vendor_id, product_id, version, interface_number, interface_class, interface_subclass, interface_protocol)) {
        return false;
    }

    return (type == SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_PRO);
}

static void UpdateDeviceIdentity(SDL_HIDAPI_Device *device)
{
    SDL_DriverSwitch_Context *ctx = (SDL_DriverSwitch_Context *)device->context;

    if (ctx->m_bInputOnly) {
        if (SDL_IsJoystickGameCube(device->vendor_id, device->product_id)) {
            device->type = SDL_GAMEPAD_TYPE_STANDARD;
        }
    } else {
        char serial[18];

        switch (ctx->m_eControllerType) {
        case k_eSwitchDeviceInfoControllerType_JoyConLeft:
            HIDAPI_SetDeviceName(device, "Nintendo Switch Joy-Con (L)");
            HIDAPI_SetDeviceProduct(device, USB_VENDOR_NINTENDO, USB_PRODUCT_NINTENDO_SWITCH_JOYCON_LEFT);
            device->type = SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_LEFT;
            break;
        case k_eSwitchDeviceInfoControllerType_JoyConRight:
            HIDAPI_SetDeviceName(device, "Nintendo Switch Joy-Con (R)");
            HIDAPI_SetDeviceProduct(device, USB_VENDOR_NINTENDO, USB_PRODUCT_NINTENDO_SWITCH_JOYCON_RIGHT);
            device->type = SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_RIGHT;
            break;
        case k_eSwitchDeviceInfoControllerType_ProController:
        case k_eSwitchDeviceInfoControllerType_LicProController:
            HIDAPI_SetDeviceName(device, "Nintendo Switch Pro Controller");
            HIDAPI_SetDeviceProduct(device, USB_VENDOR_NINTENDO, USB_PRODUCT_NINTENDO_SWITCH_PRO);
            device->type = SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_PRO;
            break;
        case k_eSwitchDeviceInfoControllerType_HVCLeft:
            HIDAPI_SetDeviceName(device, "Nintendo HVC Controller (1)");
            device->type = SDL_GAMEPAD_TYPE_STANDARD;
            break;
        case k_eSwitchDeviceInfoControllerType_HVCRight:
            HIDAPI_SetDeviceName(device, "Nintendo HVC Controller (2)");
            device->type = SDL_GAMEPAD_TYPE_STANDARD;
            break;
        case k_eSwitchDeviceInfoControllerType_NESLeft:
            HIDAPI_SetDeviceName(device, "Nintendo NES Controller (L)");
            device->type = SDL_GAMEPAD_TYPE_STANDARD;
            break;
        case k_eSwitchDeviceInfoControllerType_NESRight:
            HIDAPI_SetDeviceName(device, "Nintendo NES Controller (R)");
            device->type = SDL_GAMEPAD_TYPE_STANDARD;
            break;
        case k_eSwitchDeviceInfoControllerType_SNES:
            HIDAPI_SetDeviceName(device, "Nintendo SNES Controller");
            HIDAPI_SetDeviceProduct(device, USB_VENDOR_NINTENDO, USB_PRODUCT_NINTENDO_SNES_CONTROLLER);
            device->type = SDL_GAMEPAD_TYPE_STANDARD;
            break;
        case k_eSwitchDeviceInfoControllerType_N64:
            HIDAPI_SetDeviceName(device, "Nintendo N64 Controller");
            HIDAPI_SetDeviceProduct(device, USB_VENDOR_NINTENDO, USB_PRODUCT_NINTENDO_N64_CONTROLLER);
            device->type = SDL_GAMEPAD_TYPE_STANDARD;
            break;
        case k_eSwitchDeviceInfoControllerType_SEGA_Genesis:
            HIDAPI_SetDeviceName(device, "Nintendo SEGA Genesis Controller");
            HIDAPI_SetDeviceProduct(device, USB_VENDOR_NINTENDO, USB_PRODUCT_NINTENDO_SEGA_GENESIS_CONTROLLER);
            device->type = SDL_GAMEPAD_TYPE_STANDARD;
            break;
        case k_eSwitchDeviceInfoControllerType_Unknown:
            // We couldn't read the device info for this controller, might not be fully compliant
            if (device->vendor_id == USB_VENDOR_NINTENDO) {
                switch (device->product_id) {
                case USB_PRODUCT_NINTENDO_SWITCH_JOYCON_LEFT:
                    ctx->m_eControllerType = k_eSwitchDeviceInfoControllerType_JoyConLeft;
                    HIDAPI_SetDeviceName(device, "Nintendo Switch Joy-Con (L)");
                    device->type = SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_LEFT;
                    break;
                case USB_PRODUCT_NINTENDO_SWITCH_JOYCON_RIGHT:
                    ctx->m_eControllerType = k_eSwitchDeviceInfoControllerType_JoyConRight;
                    HIDAPI_SetDeviceName(device, "Nintendo Switch Joy-Con (R)");
                    device->type = SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_RIGHT;
                    break;
                case USB_PRODUCT_NINTENDO_SWITCH_PRO:
                    ctx->m_eControllerType = k_eSwitchDeviceInfoControllerType_ProController;
                    HIDAPI_SetDeviceName(device, "Nintendo Switch Pro Controller");
                    device->type = SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_PRO;
                    break;
                default:
                    break;
                }
            }
            return;
        default:
            device->type = SDL_GAMEPAD_TYPE_STANDARD;
            break;
        }
        device->guid.data[15] = ctx->m_eControllerType;

        (void)SDL_snprintf(serial, sizeof(serial), "%.2x-%.2x-%.2x-%.2x-%.2x-%.2x",
                           ctx->m_rgucMACAddress[0],
                           ctx->m_rgucMACAddress[1],
                           ctx->m_rgucMACAddress[2],
                           ctx->m_rgucMACAddress[3],
                           ctx->m_rgucMACAddress[4],
                           ctx->m_rgucMACAddress[5]);
        HIDAPI_SetDeviceSerial(device, serial);
    }
}

static bool HIDAPI_DriverSwitch_InitDevice(SDL_HIDAPI_Device *device)
{
    SDL_DriverSwitch_Context *ctx;

    ctx = (SDL_DriverSwitch_Context *)SDL_calloc(1, sizeof(*ctx));
    if (!ctx) {
        return false;
    }
    ctx->device = device;
    device->context = ctx;

    ctx->m_nMaxWriteAttempts = GetMaxWriteAttempts(device);
    ctx->m_bSyncWrite = true;

    // Find out whether or not we can send output reports
    ctx->m_bInputOnly = SDL_IsJoystickNintendoSwitchProInputOnly(device->vendor_id, device->product_id);
    if (!ctx->m_bInputOnly) {
        // Initialize rumble data, important for reading device info on the MOBAPAD M073
        SetNeutralRumble(&ctx->m_RumblePacket.rumbleData[0]);
        SetNeutralRumble(&ctx->m_RumblePacket.rumbleData[1]);

        BReadDeviceInfo(ctx);
    }
    UpdateDeviceIdentity(device);

    // Prefer the USB device over the Bluetooth device
    if (device->is_bluetooth) {
        if (HIDAPI_HasConnectedUSBDevice(device->serial)) {
            return true;
        }
    } else {
        HIDAPI_DisconnectBluetoothDevice(device->serial);
    }
    return HIDAPI_JoystickConnected(device, NULL);
}

static int HIDAPI_DriverSwitch_GetDevicePlayerIndex(SDL_HIDAPI_Device *device, SDL_JoystickID instance_id)
{
    return -1;
}

static void HIDAPI_DriverSwitch_SetDevicePlayerIndex(SDL_HIDAPI_Device *device, SDL_JoystickID instance_id, int player_index)
{
    SDL_DriverSwitch_Context *ctx = (SDL_DriverSwitch_Context *)device->context;

    if (!ctx->joystick) {
        return;
    }

    ctx->m_nPlayerIndex = player_index;

    UpdateSlotLED(ctx);
}

static bool HIDAPI_DriverSwitch_OpenJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
    SDL_DriverSwitch_Context *ctx = (SDL_DriverSwitch_Context *)device->context;

    SDL_AssertJoysticksLocked();

    ctx->joystick = joystick;

    ctx->m_bSyncWrite = true;

    if (!ctx->m_bInputOnly) {
#ifdef SDL_PLATFORM_MACOS
        // Wait for the OS to finish its handshake with the controller
        SDL_Delay(250);
#endif
        GetInitialInputMode(ctx);
        ctx->m_nCurrentInputMode = ctx->m_nInitialInputMode;

        // Initialize rumble data
        SetNeutralRumble(&ctx->m_RumblePacket.rumbleData[0]);
        SetNeutralRumble(&ctx->m_RumblePacket.rumbleData[1]);

        if (!device->is_bluetooth) {
            if (!BTrySetupUSB(ctx)) {
                SDL_SetError("Couldn't setup USB mode");
                return false;
            }
        }

        if (!LoadStickCalibration(ctx)) {
            SDL_SetError("Couldn't load stick calibration");
            return false;
        }

        if (ctx->m_eControllerType != k_eSwitchDeviceInfoControllerType_HVCLeft &&
            ctx->m_eControllerType != k_eSwitchDeviceInfoControllerType_HVCRight &&
            ctx->m_eControllerType != k_eSwitchDeviceInfoControllerType_NESLeft &&
            ctx->m_eControllerType != k_eSwitchDeviceInfoControllerType_NESRight &&
            ctx->m_eControllerType != k_eSwitchDeviceInfoControllerType_SNES &&
            ctx->m_eControllerType != k_eSwitchDeviceInfoControllerType_N64 &&
            ctx->m_eControllerType != k_eSwitchDeviceInfoControllerType_SEGA_Genesis) {
            if (LoadIMUCalibration(ctx)) {
                ctx->m_bSensorsSupported = true;
            }
        }

        // Enable vibration
        SetVibrationEnabled(ctx, 1);

        // Set desired input mode
        SDL_AddHintCallback(SDL_HINT_JOYSTICK_ENHANCED_REPORTS,
                            SDL_EnhancedReportsChanged, ctx);

        // Start sending USB reports
        if (!device->is_bluetooth) {
            // ForceUSB doesn't generate an ACK, so don't wait for a reply
            if (!WriteProprietary(ctx, k_eSwitchProprietaryCommandIDs_ForceUSB, NULL, 0, false)) {
                SDL_SetError("Couldn't start USB reports");
                return false;
            }
        }

        // Set the LED state
        if (HasHomeLED(ctx)) {
            if (ctx->m_eControllerType == k_eSwitchDeviceInfoControllerType_JoyConLeft ||
                ctx->m_eControllerType == k_eSwitchDeviceInfoControllerType_JoyConRight) {
                SDL_AddHintCallback(SDL_HINT_JOYSTICK_HIDAPI_JOYCON_HOME_LED,
                                    SDL_HomeLEDHintChanged, ctx);
            } else {
                SDL_AddHintCallback(SDL_HINT_JOYSTICK_HIDAPI_SWITCH_HOME_LED,
                                    SDL_HomeLEDHintChanged, ctx);
            }
        }
    }

    if (AlwaysUsesLabels(device->vendor_id, device->product_id, ctx->m_eControllerType)) {
        ctx->m_bUseButtonLabels = true;
    }

    // Initialize player index (needed for setting LEDs)
    ctx->m_nPlayerIndex = SDL_GetJoystickPlayerIndex(joystick);
    ctx->m_bPlayerLights = SDL_GetHintBoolean(SDL_HINT_JOYSTICK_HIDAPI_SWITCH_PLAYER_LED, true);
    UpdateSlotLED(ctx);

    SDL_AddHintCallback(SDL_HINT_JOYSTICK_HIDAPI_SWITCH_PLAYER_LED,
                        SDL_PlayerLEDHintChanged, ctx);

    // Initialize the joystick capabilities
    joystick->nbuttons = SDL_GAMEPAD_NUM_SWITCH_BUTTONS;
    joystick->naxes = SDL_GAMEPAD_AXIS_COUNT;
    joystick->nhats = 1;

    // Set up for input
    ctx->m_bSyncWrite = false;
    ctx->m_ulLastIMUReset = ctx->m_ulLastInput = SDL_GetTicks();
    ctx->m_ulIMUUpdateIntervalNS = SDL_MS_TO_NS(5); // Start off at 5 ms update rate

    // Set up for vertical mode
    ctx->m_bVerticalMode = SDL_GetHintBoolean(SDL_HINT_JOYSTICK_HIDAPI_VERTICAL_JOY_CONS, false);

    return true;
}

static bool HIDAPI_DriverSwitch_ActuallyRumbleJoystick(SDL_DriverSwitch_Context *ctx, Uint16 low_frequency_rumble, Uint16 high_frequency_rumble)
{
    /* Experimentally determined rumble values. These will only matter on some controllers as tested ones
     * seem to disregard these and just use any non-zero rumble values as a binary flag for constant rumble
     *
     * More information about these values can be found here:
     * https://github.com/dekuNukem/Nintendo_Switch_Reverse_Engineering/blob/master/rumble_data_table.md
     */
    const Uint16 k_usHighFreq = 0x0074;
    const Uint8 k_ucHighFreqAmp = EncodeRumbleHighAmplitude(high_frequency_rumble);
    const Uint8 k_ucLowFreq = 0x3D;
    const Uint16 k_usLowFreqAmp = EncodeRumbleLowAmplitude(low_frequency_rumble);

    if (low_frequency_rumble || high_frequency_rumble) {
        EncodeRumble(&ctx->m_RumblePacket.rumbleData[0], k_usHighFreq, k_ucHighFreqAmp, k_ucLowFreq, k_usLowFreqAmp);
        EncodeRumble(&ctx->m_RumblePacket.rumbleData[1], k_usHighFreq, k_ucHighFreqAmp, k_ucLowFreq, k_usLowFreqAmp);
    } else {
        SetNeutralRumble(&ctx->m_RumblePacket.rumbleData[0]);
        SetNeutralRumble(&ctx->m_RumblePacket.rumbleData[1]);
    }

    ctx->m_bRumbleActive = (low_frequency_rumble || high_frequency_rumble);

    if (!WriteRumble(ctx)) {
        return SDL_SetError("Couldn't send rumble packet");
    }
    return true;
}

static bool HIDAPI_DriverSwitch_SendPendingRumble(SDL_DriverSwitch_Context *ctx)
{
    if (SDL_GetTicks() < (ctx->m_ulRumbleSent + RUMBLE_WRITE_FREQUENCY_MS)) {
        return true;
    }

    if (ctx->m_bRumblePending) {
        Uint16 low_frequency_rumble = (Uint16)(ctx->m_unRumblePending >> 16);
        Uint16 high_frequency_rumble = (Uint16)ctx->m_unRumblePending;

#ifdef DEBUG_RUMBLE
        SDL_Log("Sent pending rumble %d/%d, %d ms after previous rumble", low_frequency_rumble, high_frequency_rumble, SDL_GetTicks() - ctx->m_ulRumbleSent);
#endif
        ctx->m_bRumblePending = false;
        ctx->m_unRumblePending = 0;

        return HIDAPI_DriverSwitch_ActuallyRumbleJoystick(ctx, low_frequency_rumble, high_frequency_rumble);
    }

    if (ctx->m_bRumbleZeroPending) {
        ctx->m_bRumbleZeroPending = false;

#ifdef DEBUG_RUMBLE
        SDL_Log("Sent pending zero rumble, %d ms after previous rumble", SDL_GetTicks() - ctx->m_ulRumbleSent);
#endif
        return HIDAPI_DriverSwitch_ActuallyRumbleJoystick(ctx, 0, 0);
    }

    return true;
}

static bool HIDAPI_DriverSwitch_RumbleJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint16 low_frequency_rumble, Uint16 high_frequency_rumble)
{
    SDL_DriverSwitch_Context *ctx = (SDL_DriverSwitch_Context *)device->context;

    if (ctx->m_bInputOnly) {
        return SDL_Unsupported();
    }

    if (device->parent) {
        if (ctx->m_eControllerType == k_eSwitchDeviceInfoControllerType_JoyConLeft) {
            // Just handle low frequency rumble
            high_frequency_rumble = 0;
        } else if (ctx->m_eControllerType == k_eSwitchDeviceInfoControllerType_JoyConRight) {
            // Just handle high frequency rumble
            low_frequency_rumble = 0;
        }
    }

    if (ctx->m_bRumblePending) {
        if (!HIDAPI_DriverSwitch_SendPendingRumble(ctx)) {
            return false;
        }
    }

    if (SDL_GetTicks() < (ctx->m_ulRumbleSent + RUMBLE_WRITE_FREQUENCY_MS)) {
        if (low_frequency_rumble || high_frequency_rumble) {
            Uint32 unRumblePending = ((Uint32)low_frequency_rumble << 16) | high_frequency_rumble;

            // Keep the highest rumble intensity in the given interval
            if (unRumblePending > ctx->m_unRumblePending) {
                ctx->m_unRumblePending = unRumblePending;
            }
            ctx->m_bRumblePending = true;
            ctx->m_bRumbleZeroPending = false;
        } else {
            // When rumble is complete, turn it off
            ctx->m_bRumbleZeroPending = true;
        }
        return true;
    }

#ifdef DEBUG_RUMBLE
    SDL_Log("Sent rumble %d/%d", low_frequency_rumble, high_frequency_rumble);
#endif

    return HIDAPI_DriverSwitch_ActuallyRumbleJoystick(ctx, low_frequency_rumble, high_frequency_rumble);
}

static bool HIDAPI_DriverSwitch_RumbleJoystickTriggers(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint16 left_rumble, Uint16 right_rumble)
{
    return SDL_Unsupported();
}

static Uint32 HIDAPI_DriverSwitch_GetJoystickCapabilities(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
    SDL_DriverSwitch_Context *ctx = (SDL_DriverSwitch_Context *)device->context;
    Uint32 result = 0;

    if (ctx->m_bPlayerLights && !ctx->m_bInputOnly) {
        result |= SDL_JOYSTICK_CAP_PLAYER_LED;
    }

    if (ctx->m_eControllerType == k_eSwitchDeviceInfoControllerType_ProController && !ctx->m_bInputOnly) {
        // Doesn't have an RGB LED, so don't return SDL_JOYSTICK_CAP_RGB_LED here
        result |= SDL_JOYSTICK_CAP_RUMBLE;
    } else if (ctx->m_eControllerType == k_eSwitchDeviceInfoControllerType_JoyConLeft ||
               ctx->m_eControllerType == k_eSwitchDeviceInfoControllerType_JoyConRight) {
        result |= SDL_JOYSTICK_CAP_RUMBLE;
    }
    return result;
}

static bool HIDAPI_DriverSwitch_SetJoystickLED(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint8 red, Uint8 green, Uint8 blue)
{
    return SDL_Unsupported();
}

static bool HIDAPI_DriverSwitch_SendJoystickEffect(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, const void *data, int size)
{
    SDL_DriverSwitch_Context *ctx = (SDL_DriverSwitch_Context *)device->context;

    if (size == sizeof(SwitchCommonOutputPacket_t)) {
        const SwitchCommonOutputPacket_t *packet = (SwitchCommonOutputPacket_t *)data;

        if (packet->ucPacketType != k_eSwitchOutputReportIDs_Rumble) {
            return SDL_SetError("Unknown Nintendo Switch Pro effect type");
        }

        SDL_copyp(&ctx->m_RumblePacket.rumbleData[0], &packet->rumbleData[0]);
        SDL_copyp(&ctx->m_RumblePacket.rumbleData[1], &packet->rumbleData[1]);
        if (!WriteRumble(ctx)) {
            return false;
        }

        // This overwrites any internal rumble
        ctx->m_bRumblePending = false;
        ctx->m_bRumbleZeroPending = false;
        return true;
    } else if (size >= 2 && size <= 256) {
        const Uint8 *payload = (const Uint8 *)data;
        ESwitchSubcommandIDs cmd = (ESwitchSubcommandIDs)payload[0];

        if (cmd == k_eSwitchSubcommandIDs_SetInputReportMode && !device->is_bluetooth) {
            // Going into simple mode over USB disables input reports, so don't do that
            return true;
        }
        if (cmd == k_eSwitchSubcommandIDs_SetHomeLight && !HasHomeLED(ctx)) {
            // Setting the home LED when it's not supported can cause the controller to reset
            return true;
        }

        if (!WriteSubcommand(ctx, cmd, &payload[1], (Uint8)(size - 1), NULL)) {
            return false;
        }
        return true;
    }
    return SDL_Unsupported();
}

static bool HIDAPI_DriverSwitch_SetJoystickSensorsEnabled(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, bool enabled)
{
    SDL_DriverSwitch_Context *ctx = (SDL_DriverSwitch_Context *)device->context;

    UpdateEnhancedModeOnApplicationUsage(ctx);

    if (!ctx->m_bSensorsSupported || (enabled && !ctx->m_bEnhancedMode)) {
        return SDL_Unsupported();
    }

    ctx->m_bReportSensors = enabled;
    ctx->m_unIMUSamples = 0;
    ctx->m_ulIMUSampleTimestampNS = SDL_GetTicksNS();

    UpdateInputMode(ctx);
    SetIMUEnabled(ctx, enabled);

    return true;
}

static void HandleInputOnlyControllerState(SDL_Joystick *joystick, SDL_DriverSwitch_Context *ctx, SwitchInputOnlyControllerStatePacket_t *packet)
{
    Sint16 axis;
    Uint64 timestamp = SDL_GetTicksNS();

    if (packet->rgucButtons[0] != ctx->m_lastInputOnlyState.rgucButtons[0]) {
        Uint8 data = packet->rgucButtons[0];
        SDL_SendJoystickButton(timestamp, joystick, RemapButton(ctx, SDL_GAMEPAD_BUTTON_SOUTH), ((data & 0x02) != 0));
        SDL_SendJoystickButton(timestamp, joystick, RemapButton(ctx, SDL_GAMEPAD_BUTTON_EAST), ((data & 0x04) != 0));
        SDL_SendJoystickButton(timestamp, joystick, RemapButton(ctx, SDL_GAMEPAD_BUTTON_WEST), ((data & 0x01) != 0));
        SDL_SendJoystickButton(timestamp, joystick, RemapButton(ctx, SDL_GAMEPAD_BUTTON_NORTH), ((data & 0x08) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_SHOULDER, ((data & 0x10) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER, ((data & 0x20) != 0));
    }

    if (packet->rgucButtons[1] != ctx->m_lastInputOnlyState.rgucButtons[1]) {
        Uint8 data = packet->rgucButtons[1];
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_BACK, ((data & 0x01) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_START, ((data & 0x02) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_STICK, ((data & 0x04) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_RIGHT_STICK, ((data & 0x08) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_GUIDE, ((data & 0x10) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SWITCH_SHARE, ((data & 0x20) != 0));
    }

    if (packet->ucStickHat != ctx->m_lastInputOnlyState.ucStickHat) {
        Uint8 hat;

        switch (packet->ucStickHat) {
        case 0:
            hat = SDL_HAT_UP;
            break;
        case 1:
            hat = SDL_HAT_RIGHTUP;
            break;
        case 2:
            hat = SDL_HAT_RIGHT;
            break;
        case 3:
            hat = SDL_HAT_RIGHTDOWN;
            break;
        case 4:
            hat = SDL_HAT_DOWN;
            break;
        case 5:
            hat = SDL_HAT_LEFTDOWN;
            break;
        case 6:
            hat = SDL_HAT_LEFT;
            break;
        case 7:
            hat = SDL_HAT_LEFTUP;
            break;
        default:
            hat = SDL_HAT_CENTERED;
            break;
        }
        SDL_SendJoystickHat(timestamp, joystick, 0, hat);
    }

    axis = (packet->rgucButtons[0] & 0x40) ? 32767 : -32768;
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFT_TRIGGER, axis);

    axis = (packet->rgucButtons[0] & 0x80) ? 32767 : -32768;
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER, axis);

    if (packet->rgucJoystickLeft[0] != ctx->m_lastInputOnlyState.rgucJoystickLeft[0]) {
        axis = (Sint16)HIDAPI_RemapVal(packet->rgucJoystickLeft[0], SDL_MIN_UINT8, SDL_MAX_UINT8, SDL_MIN_SINT16, SDL_MAX_SINT16);
        SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTX, axis);
    }

    if (packet->rgucJoystickLeft[1] != ctx->m_lastInputOnlyState.rgucJoystickLeft[1]) {
        axis = (Sint16)HIDAPI_RemapVal(packet->rgucJoystickLeft[1], SDL_MIN_UINT8, SDL_MAX_UINT8, SDL_MIN_SINT16, SDL_MAX_SINT16);
        SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTY, axis);
    }

    if (packet->rgucJoystickRight[0] != ctx->m_lastInputOnlyState.rgucJoystickRight[0]) {
        axis = (Sint16)HIDAPI_RemapVal(packet->rgucJoystickRight[0], SDL_MIN_UINT8, SDL_MAX_UINT8, SDL_MIN_SINT16, SDL_MAX_SINT16);
        SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTX, axis);
    }

    if (packet->rgucJoystickRight[1] != ctx->m_lastInputOnlyState.rgucJoystickRight[1]) {
        axis = (Sint16)HIDAPI_RemapVal(packet->rgucJoystickRight[1], SDL_MIN_UINT8, SDL_MAX_UINT8, SDL_MIN_SINT16, SDL_MAX_SINT16);
        SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTY, axis);
    }

    ctx->m_lastInputOnlyState = *packet;
}

static void HandleCombinedSimpleControllerStateL(Uint64 timestamp, SDL_Joystick *joystick, SDL_DriverSwitch_Context *ctx, SwitchSimpleStatePacket_t *packet)
{
    if (packet->rgucButtons[0] != ctx->m_lastSimpleState.rgucButtons[0]) {
        Uint8 data = packet->rgucButtons[0];
        Uint8 hat = 0;

        if (data & 0x01) {
            hat |= SDL_HAT_LEFT;
        }
        if (data & 0x02) {
            hat |= SDL_HAT_DOWN;
        }
        if (data & 0x04) {
            hat |= SDL_HAT_UP;
        }
        if (data & 0x08) {
            hat |= SDL_HAT_RIGHT;
        }
        SDL_SendJoystickHat(timestamp, joystick, 0, hat);

        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SWITCH_LEFT_PADDLE1, ((data & 0x10) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SWITCH_LEFT_PADDLE2, ((data & 0x20) != 0));
    }

    if (packet->rgucButtons[1] != ctx->m_lastSimpleState.rgucButtons[1]) {
        Uint8 data = packet->rgucButtons[1];
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_BACK, ((data & 0x01) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_STICK, ((data & 0x04) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SWITCH_SHARE, ((data & 0x20) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_SHOULDER, ((data & 0x40) != 0));
    }

    Sint16 axis = (packet->rgucButtons[1] & 0x80) ? 32767 : -32768;
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFT_TRIGGER, axis);

    if (packet->ucStickHat != ctx->m_lastSimpleState.ucStickHat) {
        switch (packet->ucStickHat) {
        case 0:
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTX, SDL_JOYSTICK_AXIS_MAX);
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTY, 0);
            break;
        case 1:
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTX, SDL_JOYSTICK_AXIS_MAX);
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTY, SDL_JOYSTICK_AXIS_MAX);
            break;
        case 2:
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTX, 0);
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTY, SDL_JOYSTICK_AXIS_MAX);
            break;
        case 3:
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTX, SDL_JOYSTICK_AXIS_MIN);
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTY, SDL_JOYSTICK_AXIS_MAX);
            break;
        case 4:
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTX, SDL_JOYSTICK_AXIS_MIN);
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTY, 0);
            break;
        case 5:
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTX, SDL_JOYSTICK_AXIS_MIN);
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTY, SDL_JOYSTICK_AXIS_MIN);
            break;
        case 6:
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTX, 0);
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTY, SDL_JOYSTICK_AXIS_MIN);
            break;
        case 7:
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTX, SDL_JOYSTICK_AXIS_MAX);
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTY, SDL_JOYSTICK_AXIS_MIN);
            break;
        default:
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTX, 0);
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTY, 0);
            break;
        }
    }
}

static void HandleCombinedSimpleControllerStateR(Uint64 timestamp, SDL_Joystick *joystick, SDL_DriverSwitch_Context *ctx, SwitchSimpleStatePacket_t *packet)
{
    if (packet->rgucButtons[0] != ctx->m_lastSimpleState.rgucButtons[0]) {
        Uint8 data = packet->rgucButtons[0];
        SDL_SendJoystickButton(timestamp, joystick, RemapButton(ctx, SDL_GAMEPAD_BUTTON_EAST), ((data & 0x01) != 0));
        SDL_SendJoystickButton(timestamp, joystick, RemapButton(ctx, SDL_GAMEPAD_BUTTON_NORTH), ((data & 0x02) != 0));
        SDL_SendJoystickButton(timestamp, joystick, RemapButton(ctx, SDL_GAMEPAD_BUTTON_SOUTH), ((data & 0x04) != 0));
        SDL_SendJoystickButton(timestamp, joystick, RemapButton(ctx, SDL_GAMEPAD_BUTTON_WEST), ((data & 0x08) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SWITCH_RIGHT_PADDLE2, ((data & 0x10) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SWITCH_RIGHT_PADDLE1, ((data & 0x20) != 0));
    }

    if (packet->rgucButtons[1] != ctx->m_lastSimpleState.rgucButtons[1]) {
        Uint8 data = packet->rgucButtons[1];
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_START, ((data & 0x02) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_RIGHT_STICK, ((data & 0x08) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_GUIDE, ((data & 0x10) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER, ((data & 0x40) != 0));
    }

    Sint16 axis = (packet->rgucButtons[1] & 0x80) ? 32767 : -32768;
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER, axis);

    if (packet->ucStickHat != ctx->m_lastSimpleState.ucStickHat) {
        switch (packet->ucStickHat) {
        case 0:
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTX, SDL_JOYSTICK_AXIS_MIN);
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTY, 0);
            break;
        case 1:
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTX, SDL_JOYSTICK_AXIS_MIN);
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTY, SDL_JOYSTICK_AXIS_MIN);
            break;
        case 2:
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTX, 0);
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTY, SDL_JOYSTICK_AXIS_MIN);
            break;
        case 3:
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTX, SDL_JOYSTICK_AXIS_MAX);
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTY, SDL_JOYSTICK_AXIS_MIN);
            break;
        case 4:
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTX, SDL_JOYSTICK_AXIS_MAX);
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTY, 0);
            break;
        case 5:
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTX, SDL_JOYSTICK_AXIS_MAX);
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTY, SDL_JOYSTICK_AXIS_MAX);
            break;
        case 6:
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTX, 0);
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTY, SDL_JOYSTICK_AXIS_MAX);
            break;
        case 7:
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTX, SDL_JOYSTICK_AXIS_MIN);
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTY, SDL_JOYSTICK_AXIS_MAX);
            break;
        default:
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTX, 0);
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTY, 0);
            break;
        }
    }
}

static void HandleMiniSimpleControllerStateL(Uint64 timestamp, SDL_Joystick *joystick, SDL_DriverSwitch_Context *ctx, SwitchSimpleStatePacket_t *packet)
{
    if (packet->rgucButtons[0] != ctx->m_lastSimpleState.rgucButtons[0]) {
        Uint8 data = packet->rgucButtons[0];
        SDL_SendJoystickButton(timestamp, joystick, RemapButton(ctx, SDL_GAMEPAD_BUTTON_SOUTH), ((data & 0x01) != 0));
        SDL_SendJoystickButton(timestamp, joystick, RemapButton(ctx, SDL_GAMEPAD_BUTTON_EAST), ((data & 0x02) != 0));
        SDL_SendJoystickButton(timestamp, joystick, RemapButton(ctx, SDL_GAMEPAD_BUTTON_WEST), ((data & 0x04) != 0));
        SDL_SendJoystickButton(timestamp, joystick, RemapButton(ctx, SDL_GAMEPAD_BUTTON_NORTH), ((data & 0x08) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_SHOULDER, ((data & 0x10) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER, ((data & 0x20) != 0));
    }

    if (packet->rgucButtons[1] != ctx->m_lastSimpleState.rgucButtons[1]) {
        Uint8 data = packet->rgucButtons[1];
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_START, ((data & 0x01) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_STICK, ((data & 0x04) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_GUIDE, ((data & 0x20) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SWITCH_LEFT_PADDLE1, ((data & 0x40) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SWITCH_LEFT_PADDLE2, ((data & 0x80) != 0));
    }

    if (packet->ucStickHat != ctx->m_lastSimpleState.ucStickHat) {
        switch (packet->ucStickHat) {
        case 0:
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTX, 0);
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTY, SDL_JOYSTICK_AXIS_MIN);
            break;
        case 1:
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTX, SDL_JOYSTICK_AXIS_MAX);
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTY, SDL_JOYSTICK_AXIS_MIN);
            break;
        case 2:
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTX, SDL_JOYSTICK_AXIS_MAX);
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTY, 0);
            break;
        case 3:
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTX, SDL_JOYSTICK_AXIS_MAX);
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTY, SDL_JOYSTICK_AXIS_MAX);
            break;
        case 4:
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTX, 0);
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTY, SDL_JOYSTICK_AXIS_MAX);
            break;
        case 5:
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTX, SDL_JOYSTICK_AXIS_MIN);
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTY, SDL_JOYSTICK_AXIS_MAX);
            break;
        case 6:
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTX, SDL_JOYSTICK_AXIS_MIN);
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTY, 0);
            break;
        case 7:
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTX, SDL_JOYSTICK_AXIS_MIN);
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTY, SDL_JOYSTICK_AXIS_MIN);
            break;
        default:
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTX, 0);
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTY, 0);
            break;
        }
    }
}

static void HandleMiniSimpleControllerStateR(Uint64 timestamp, SDL_Joystick *joystick, SDL_DriverSwitch_Context *ctx, SwitchSimpleStatePacket_t *packet)
{
    if (packet->rgucButtons[0] != ctx->m_lastSimpleState.rgucButtons[0]) {
        Uint8 data = packet->rgucButtons[0];
        SDL_SendJoystickButton(timestamp, joystick, RemapButton(ctx, SDL_GAMEPAD_BUTTON_SOUTH), ((data & 0x01) != 0));
        SDL_SendJoystickButton(timestamp, joystick, RemapButton(ctx, SDL_GAMEPAD_BUTTON_EAST), ((data & 0x02) != 0));
        SDL_SendJoystickButton(timestamp, joystick, RemapButton(ctx, SDL_GAMEPAD_BUTTON_WEST), ((data & 0x04) != 0));
        SDL_SendJoystickButton(timestamp, joystick, RemapButton(ctx, SDL_GAMEPAD_BUTTON_NORTH), ((data & 0x08) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_SHOULDER, ((data & 0x10) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER, ((data & 0x20) != 0));
    }

    if (packet->rgucButtons[1] != ctx->m_lastSimpleState.rgucButtons[1]) {
        Uint8 data = packet->rgucButtons[1];
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_START, ((data & 0x02) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_STICK, ((data & 0x08) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_GUIDE, ((data & 0x10) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SWITCH_SHARE, ((data & 0x20) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SWITCH_RIGHT_PADDLE1, ((data & 0x40) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SWITCH_RIGHT_PADDLE2, ((data & 0x80) != 0));
    }

    if (packet->ucStickHat != ctx->m_lastSimpleState.ucStickHat) {
        switch (packet->ucStickHat) {
        case 0:
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTX, 0);
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTY, SDL_JOYSTICK_AXIS_MIN);
            break;
        case 1:
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTX, SDL_JOYSTICK_AXIS_MAX);
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTY, SDL_JOYSTICK_AXIS_MIN);
            break;
        case 2:
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTX, SDL_JOYSTICK_AXIS_MAX);
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTY, 0);
            break;
        case 3:
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTX, SDL_JOYSTICK_AXIS_MAX);
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTY, SDL_JOYSTICK_AXIS_MAX);
            break;
        case 4:
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTX, 0);
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTY, SDL_JOYSTICK_AXIS_MAX);
            break;
        case 5:
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTX, SDL_JOYSTICK_AXIS_MIN);
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTY, SDL_JOYSTICK_AXIS_MAX);
            break;
        case 6:
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTX, SDL_JOYSTICK_AXIS_MIN);
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTY, 0);
            break;
        case 7:
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTX, SDL_JOYSTICK_AXIS_MIN);
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTY, SDL_JOYSTICK_AXIS_MIN);
            break;
        default:
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTX, 0);
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTY, 0);
            break;
        }
    }
}

static void HandleSimpleControllerState(SDL_Joystick *joystick, SDL_DriverSwitch_Context *ctx, SwitchSimpleStatePacket_t *packet)
{
    Uint64 timestamp = SDL_GetTicksNS();

    if (ctx->m_eControllerType == k_eSwitchDeviceInfoControllerType_JoyConLeft) {
        if (ctx->device->parent || ctx->m_bVerticalMode) {
            HandleCombinedSimpleControllerStateL(timestamp, joystick, ctx, packet);
        } else {
            HandleMiniSimpleControllerStateL(timestamp, joystick, ctx, packet);
        }
    } else if (ctx->m_eControllerType == k_eSwitchDeviceInfoControllerType_JoyConRight) {
        if (ctx->device->parent || ctx->m_bVerticalMode) {
            HandleCombinedSimpleControllerStateR(timestamp, joystick, ctx, packet);
        } else {
            HandleMiniSimpleControllerStateR(timestamp, joystick, ctx, packet);
        }
    } else {
        Sint16 axis;

        if (packet->rgucButtons[0] != ctx->m_lastSimpleState.rgucButtons[0]) {
            Uint8 data = packet->rgucButtons[0];
            SDL_SendJoystickButton(timestamp, joystick, RemapButton(ctx, SDL_GAMEPAD_BUTTON_SOUTH), ((data & 0x01) != 0));
            SDL_SendJoystickButton(timestamp, joystick, RemapButton(ctx, SDL_GAMEPAD_BUTTON_EAST), ((data & 0x02) != 0));
            SDL_SendJoystickButton(timestamp, joystick, RemapButton(ctx, SDL_GAMEPAD_BUTTON_WEST), ((data & 0x04) != 0));
            SDL_SendJoystickButton(timestamp, joystick, RemapButton(ctx, SDL_GAMEPAD_BUTTON_NORTH), ((data & 0x08) != 0));
            SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_SHOULDER, ((data & 0x10) != 0));
            SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER, ((data & 0x20) != 0));
        }

        if (packet->rgucButtons[1] != ctx->m_lastSimpleState.rgucButtons[1]) {
            Uint8 data = packet->rgucButtons[1];
            SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_BACK, ((data & 0x01) != 0));
            SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_START, ((data & 0x02) != 0));
            SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_STICK, ((data & 0x04) != 0));
            SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_RIGHT_STICK, ((data & 0x08) != 0));
            SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_GUIDE, ((data & 0x10) != 0));
            SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SWITCH_SHARE, ((data & 0x20) != 0));
        }

        if (packet->ucStickHat != ctx->m_lastSimpleState.ucStickHat) {
            Uint8 hat;

            switch (packet->ucStickHat) {
            case 0:
                hat = SDL_HAT_UP;
                break;
            case 1:
                hat = SDL_HAT_RIGHTUP;
                break;
            case 2:
                hat = SDL_HAT_RIGHT;
                break;
            case 3:
                hat = SDL_HAT_RIGHTDOWN;
                break;
            case 4:
                hat = SDL_HAT_DOWN;
                break;
            case 5:
                hat = SDL_HAT_LEFTDOWN;
                break;
            case 6:
                hat = SDL_HAT_LEFT;
                break;
            case 7:
                hat = SDL_HAT_LEFTUP;
                break;
            default:
                hat = SDL_HAT_CENTERED;
                break;
            }
            SDL_SendJoystickHat(timestamp, joystick, 0, hat);
        }

        axis = (packet->rgucButtons[0] & 0x40) ? 32767 : -32768;
        SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFT_TRIGGER, axis);

        axis = ((packet->rgucButtons[0] & 0x80) || (packet->rgucButtons[1] & 0x80)) ? 32767 : -32768;
        SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER, axis);

        axis = ApplySimpleStickCalibration(ctx, 0, 0, packet->sJoystickLeft[0]);
        SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTX, axis);

        axis = ApplySimpleStickCalibration(ctx, 0, 1, packet->sJoystickLeft[1]);
        SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTY, axis);

        axis = ApplySimpleStickCalibration(ctx, 1, 0, packet->sJoystickRight[0]);
        SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTX, axis);

        axis = ApplySimpleStickCalibration(ctx, 1, 1, packet->sJoystickRight[1]);
        SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTY, axis);
    }

    ctx->m_lastSimpleState = *packet;
}

static void SendSensorUpdate(Uint64 timestamp, SDL_Joystick *joystick, SDL_DriverSwitch_Context *ctx, SDL_SensorType type, Uint64 sensor_timestamp, const Sint16 *values)
{
    float data[3];

    /* Note the order of components has been shuffled to match PlayStation controllers,
     * since that's our de facto standard from already supporting those controllers, and
     * users will want consistent axis mappings across devices.
     */
    if (type == SDL_SENSOR_GYRO || type == SDL_SENSOR_GYRO_L || type == SDL_SENSOR_GYRO_R) {
        data[0] = -(ctx->m_IMUScaleData.fGyroScaleY * (float)values[1]);
        data[1] = ctx->m_IMUScaleData.fGyroScaleZ * (float)values[2];
        data[2] = -(ctx->m_IMUScaleData.fGyroScaleX * (float)values[0]);
    } else {
        data[0] = -(ctx->m_IMUScaleData.fAccelScaleY * (float)values[1]);
        data[1] = ctx->m_IMUScaleData.fAccelScaleZ * (float)values[2];
        data[2] = -(ctx->m_IMUScaleData.fAccelScaleX * (float)values[0]);
    }

    // Right Joy-Con flips some axes, so let's flip them back for consistency
    if (ctx->m_eControllerType == k_eSwitchDeviceInfoControllerType_JoyConRight) {
        data[0] = -data[0];
        data[1] = -data[1];
    }

    if (ctx->m_eControllerType == k_eSwitchDeviceInfoControllerType_JoyConLeft &&
        !ctx->device->parent && !ctx->m_bVerticalMode) {
        // Mini-gamepad mode, swap some axes around
        float tmp = data[2];
        data[2] = -data[0];
        data[0] = tmp;
    }

    if (ctx->m_eControllerType == k_eSwitchDeviceInfoControllerType_JoyConRight &&
        !ctx->device->parent && !ctx->m_bVerticalMode) {
        // Mini-gamepad mode, swap some axes around
        float tmp = data[2];
        data[2] = data[0];
        data[0] = -tmp;
    }

    SDL_SendJoystickSensor(timestamp, joystick, type, sensor_timestamp, data, 3);
}

static void HandleCombinedControllerStateL(Uint64 timestamp, SDL_Joystick *joystick, SDL_DriverSwitch_Context *ctx, SwitchStatePacket_t *packet)
{
    Sint16 axis;

    if (packet->controllerState.rgucButtons[1] != ctx->m_lastFullState.controllerState.rgucButtons[1]) {
        Uint8 data = packet->controllerState.rgucButtons[1];
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_BACK, ((data & 0x01) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_STICK, ((data & 0x08) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SWITCH_SHARE, ((data & 0x20) != 0));
    }

    if (packet->controllerState.rgucButtons[2] != ctx->m_lastFullState.controllerState.rgucButtons[2]) {
        Uint8 data = packet->controllerState.rgucButtons[2];
        Uint8 hat = 0;

        if (data & 0x01) {
            hat |= SDL_HAT_DOWN;
        }
        if (data & 0x02) {
            hat |= SDL_HAT_UP;
        }
        if (data & 0x04) {
            hat |= SDL_HAT_RIGHT;
        }
        if (data & 0x08) {
            hat |= SDL_HAT_LEFT;
        }
        SDL_SendJoystickHat(timestamp, joystick, 0, hat);

        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SWITCH_LEFT_PADDLE2, ((data & 0x10) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SWITCH_LEFT_PADDLE1, ((data & 0x20) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_SHOULDER, ((data & 0x40) != 0));
        axis = (data & 0x80) ? 32767 : -32768;
        SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFT_TRIGGER, axis);
    }

    axis = packet->controllerState.rgucJoystickLeft[0] | ((packet->controllerState.rgucJoystickLeft[1] & 0xF) << 8);
    axis = ApplyStickCalibration(ctx, 0, 0, axis);
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTX, axis);

    axis = ((packet->controllerState.rgucJoystickLeft[1] & 0xF0) >> 4) | (packet->controllerState.rgucJoystickLeft[2] << 4);
    axis = ApplyStickCalibration(ctx, 0, 1, axis);
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTY, ~axis);
}

static void HandleCombinedControllerStateR(Uint64 timestamp, SDL_Joystick *joystick, SDL_DriverSwitch_Context *ctx, SwitchStatePacket_t *packet)
{
    Sint16 axis;

    if (packet->controllerState.rgucButtons[0] != ctx->m_lastFullState.controllerState.rgucButtons[0]) {
        Uint8 data = packet->controllerState.rgucButtons[0];
        SDL_SendJoystickButton(timestamp, joystick, RemapButton(ctx, SDL_GAMEPAD_BUTTON_SOUTH), ((data & 0x04) != 0));
        SDL_SendJoystickButton(timestamp, joystick, RemapButton(ctx, SDL_GAMEPAD_BUTTON_EAST), ((data & 0x08) != 0));
        SDL_SendJoystickButton(timestamp, joystick, RemapButton(ctx, SDL_GAMEPAD_BUTTON_WEST), ((data & 0x01) != 0));
        SDL_SendJoystickButton(timestamp, joystick, RemapButton(ctx, SDL_GAMEPAD_BUTTON_NORTH), ((data & 0x02) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SWITCH_RIGHT_PADDLE1, ((data & 0x10) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SWITCH_RIGHT_PADDLE2, ((data & 0x20) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER, ((data & 0x40) != 0));
        axis = (data & 0x80) ? 32767 : -32768;
        SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER, axis);
    }

    if (packet->controllerState.rgucButtons[1] != ctx->m_lastFullState.controllerState.rgucButtons[1]) {
        Uint8 data = packet->controllerState.rgucButtons[1];
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_START, ((data & 0x02) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_RIGHT_STICK, ((data & 0x04) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_GUIDE, ((data & 0x10) != 0));
    }

    axis = packet->controllerState.rgucJoystickRight[0] | ((packet->controllerState.rgucJoystickRight[1] & 0xF) << 8);
    axis = ApplyStickCalibration(ctx, 1, 0, axis);
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTX, axis);

    axis = ((packet->controllerState.rgucJoystickRight[1] & 0xF0) >> 4) | (packet->controllerState.rgucJoystickRight[2] << 4);
    axis = ApplyStickCalibration(ctx, 1, 1, axis);
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTY, ~axis);
}

static void HandleMiniControllerStateL(Uint64 timestamp, SDL_Joystick *joystick, SDL_DriverSwitch_Context *ctx, SwitchStatePacket_t *packet)
{
    Sint16 axis;

    if (packet->controllerState.rgucButtons[1] != ctx->m_lastFullState.controllerState.rgucButtons[1]) {
        Uint8 data = packet->controllerState.rgucButtons[1];
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_START, ((data & 0x01) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_STICK, ((data & 0x08) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_GUIDE, ((data & 0x20) != 0));
    }

    if (packet->controllerState.rgucButtons[2] != ctx->m_lastFullState.controllerState.rgucButtons[2]) {
        Uint8 data = packet->controllerState.rgucButtons[2];
        SDL_SendJoystickButton(timestamp, joystick, RemapButton(ctx, SDL_GAMEPAD_BUTTON_SOUTH), ((data & 0x08) != 0));
        SDL_SendJoystickButton(timestamp, joystick, RemapButton(ctx, SDL_GAMEPAD_BUTTON_EAST), ((data & 0x01) != 0));
        SDL_SendJoystickButton(timestamp, joystick, RemapButton(ctx, SDL_GAMEPAD_BUTTON_WEST), ((data & 0x02) != 0));
        SDL_SendJoystickButton(timestamp, joystick, RemapButton(ctx, SDL_GAMEPAD_BUTTON_NORTH), ((data & 0x04) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER, ((data & 0x10) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_SHOULDER, ((data & 0x20) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SWITCH_LEFT_PADDLE1, ((data & 0x40) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SWITCH_LEFT_PADDLE2, ((data & 0x80) != 0));
    }

    axis = packet->controllerState.rgucJoystickLeft[0] | ((packet->controllerState.rgucJoystickLeft[1] & 0xF) << 8);
    axis = ApplyStickCalibration(ctx, 0, 0, axis);
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTY, ~axis);

    axis = ((packet->controllerState.rgucJoystickLeft[1] & 0xF0) >> 4) | (packet->controllerState.rgucJoystickLeft[2] << 4);
    axis = ApplyStickCalibration(ctx, 0, 1, axis);
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTX, ~axis);
}

static void HandleMiniControllerStateR(Uint64 timestamp, SDL_Joystick *joystick, SDL_DriverSwitch_Context *ctx, SwitchStatePacket_t *packet)
{
    Sint16 axis;

    if (packet->controllerState.rgucButtons[0] != ctx->m_lastFullState.controllerState.rgucButtons[0]) {
        Uint8 data = packet->controllerState.rgucButtons[0];
        SDL_SendJoystickButton(timestamp, joystick, RemapButton(ctx, SDL_GAMEPAD_BUTTON_SOUTH), ((data & 0x08) != 0));
        SDL_SendJoystickButton(timestamp, joystick, RemapButton(ctx, SDL_GAMEPAD_BUTTON_EAST), ((data & 0x02) != 0));
        SDL_SendJoystickButton(timestamp, joystick, RemapButton(ctx, SDL_GAMEPAD_BUTTON_WEST), ((data & 0x04) != 0));
        SDL_SendJoystickButton(timestamp, joystick, RemapButton(ctx, SDL_GAMEPAD_BUTTON_NORTH), ((data & 0x01) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER, ((data & 0x10) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_SHOULDER, ((data & 0x20) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SWITCH_RIGHT_PADDLE1, ((data & 0x40) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SWITCH_RIGHT_PADDLE2, ((data & 0x80) != 0));
    }

    if (packet->controllerState.rgucButtons[1] != ctx->m_lastFullState.controllerState.rgucButtons[1]) {
        Uint8 data = packet->controllerState.rgucButtons[1];
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_START, ((data & 0x02) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_STICK, ((data & 0x04) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_GUIDE, ((data & 0x10) != 0));
    }

    axis = packet->controllerState.rgucJoystickRight[0] | ((packet->controllerState.rgucJoystickRight[1] & 0xF) << 8);
    axis = ApplyStickCalibration(ctx, 1, 0, axis);
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTY, axis);

    axis = ((packet->controllerState.rgucJoystickRight[1] & 0xF0) >> 4) | (packet->controllerState.rgucJoystickRight[2] << 4);
    axis = ApplyStickCalibration(ctx, 1, 1, axis);
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTX, axis);
}

static void HandleFullControllerState(SDL_Joystick *joystick, SDL_DriverSwitch_Context *ctx, SwitchStatePacket_t *packet) SDL_NO_THREAD_SAFETY_ANALYSIS // We unlock and lock the device lock to be able to change IMU state
{
    Uint64 timestamp = SDL_GetTicksNS();

    if (ctx->m_eControllerType == k_eSwitchDeviceInfoControllerType_JoyConLeft) {
        if (ctx->device->parent || ctx->m_bVerticalMode) {
            HandleCombinedControllerStateL(timestamp, joystick, ctx, packet);
        } else {
            HandleMiniControllerStateL(timestamp, joystick, ctx, packet);
        }
    } else if (ctx->m_eControllerType == k_eSwitchDeviceInfoControllerType_JoyConRight) {
        if (ctx->device->parent || ctx->m_bVerticalMode) {
            HandleCombinedControllerStateR(timestamp, joystick, ctx, packet);
        } else {
            HandleMiniControllerStateR(timestamp, joystick, ctx, packet);
        }
    } else {
        Sint16 axis;

        if (packet->controllerState.rgucButtons[0] != ctx->m_lastFullState.controllerState.rgucButtons[0]) {
            Uint8 data = packet->controllerState.rgucButtons[0];
            SDL_SendJoystickButton(timestamp, joystick, RemapButton(ctx, SDL_GAMEPAD_BUTTON_SOUTH), ((data & 0x04) != 0));
            SDL_SendJoystickButton(timestamp, joystick, RemapButton(ctx, SDL_GAMEPAD_BUTTON_EAST), ((data & 0x08) != 0));
            SDL_SendJoystickButton(timestamp, joystick, RemapButton(ctx, SDL_GAMEPAD_BUTTON_WEST), ((data & 0x01) != 0));
            SDL_SendJoystickButton(timestamp, joystick, RemapButton(ctx, SDL_GAMEPAD_BUTTON_NORTH), ((data & 0x02) != 0));
            SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER, ((data & 0x40) != 0));
        }

        if (packet->controllerState.rgucButtons[1] != ctx->m_lastFullState.controllerState.rgucButtons[1]) {
            Uint8 data = packet->controllerState.rgucButtons[1];
            SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_BACK, ((data & 0x01) != 0));
            SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_START, ((data & 0x02) != 0));
            SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_RIGHT_STICK, ((data & 0x04) != 0));
            SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_STICK, ((data & 0x08) != 0));

            SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_GUIDE, ((data & 0x10) != 0));
            SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SWITCH_SHARE, ((data & 0x20) != 0));
        }

        if (packet->controllerState.rgucButtons[2] != ctx->m_lastFullState.controllerState.rgucButtons[2]) {
            Uint8 data = packet->controllerState.rgucButtons[2];
            Uint8 hat = 0;

            if (data & 0x01) {
                hat |= SDL_HAT_DOWN;
            }
            if (data & 0x02) {
                hat |= SDL_HAT_UP;
            }
            if (data & 0x04) {
                hat |= SDL_HAT_RIGHT;
            }
            if (data & 0x08) {
                hat |= SDL_HAT_LEFT;
            }
            SDL_SendJoystickHat(timestamp, joystick, 0, hat);

            SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_SHOULDER, ((data & 0x40) != 0));
        }

        axis = (packet->controllerState.rgucButtons[0] & 0x80) ? 32767 : -32768;
        SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER, axis);

        axis = (packet->controllerState.rgucButtons[2] & 0x80) ? 32767 : -32768;
        SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFT_TRIGGER, axis);

        axis = packet->controllerState.rgucJoystickLeft[0] | ((packet->controllerState.rgucJoystickLeft[1] & 0xF) << 8);
        axis = ApplyStickCalibration(ctx, 0, 0, axis);
        SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTX, axis);

        axis = ((packet->controllerState.rgucJoystickLeft[1] & 0xF0) >> 4) | (packet->controllerState.rgucJoystickLeft[2] << 4);
        axis = ApplyStickCalibration(ctx, 0, 1, axis);
        SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTY, ~axis);

        axis = packet->controllerState.rgucJoystickRight[0] | ((packet->controllerState.rgucJoystickRight[1] & 0xF) << 8);
        axis = ApplyStickCalibration(ctx, 1, 0, axis);
        SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTX, axis);

        axis = ((packet->controllerState.rgucJoystickRight[1] & 0xF0) >> 4) | (packet->controllerState.rgucJoystickRight[2] << 4);
        axis = ApplyStickCalibration(ctx, 1, 1, axis);
        SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTY, ~axis);
    }

    /* High nibble of battery/connection byte is battery level, low nibble is connection status (always 0 on 8BitDo Pro 2)
     * LSB of connection nibble is USB/Switch connection status
     * LSB of the battery nibble is used to report charging.
     * The battery level is reported from 0(empty)-8(full)
     */
    SDL_PowerState state;
    int charging = (packet->controllerState.ucBatteryAndConnection & 0x10);
    int level = (packet->controllerState.ucBatteryAndConnection & 0xE0) >> 4;
    int percent = (int)SDL_roundf((level / 8.0f) * 100.0f);

    if (charging) {
        if (level == 8) {
            state = SDL_POWERSTATE_CHARGED;
        } else {
            state = SDL_POWERSTATE_CHARGING;
        }
    } else {
        state = SDL_POWERSTATE_ON_BATTERY;
    }
    SDL_SendJoystickPowerInfo(joystick, state, percent);

    if (ctx->m_bReportSensors) {
        // Need to copy the imuState to an aligned variable
        SwitchControllerIMUState_t imuState[3];
        SDL_assert(sizeof(imuState) == sizeof(packet->imuState));
        SDL_memcpy(imuState, packet->imuState, sizeof(imuState));

        bool bHasSensorData = (imuState[0].sAccelZ != 0 ||
                               imuState[0].sAccelY != 0 ||
                               imuState[0].sAccelX != 0);
        if (bHasSensorData) {
            const Uint32 IMU_UPDATE_RATE_SAMPLE_FREQUENCY = 1000;
            Uint64 sensor_timestamp[3];

            ctx->m_bHasSensorData = true;

            // We got three IMU samples, calculate the IMU update rate and timestamps
            ctx->m_unIMUSamples += 3;
            if (ctx->m_unIMUSamples >= IMU_UPDATE_RATE_SAMPLE_FREQUENCY) {
                Uint64 now = SDL_GetTicksNS();
                Uint64 elapsed = (now - ctx->m_ulIMUSampleTimestampNS);

                if (elapsed > 0) {
                    ctx->m_ulIMUUpdateIntervalNS = elapsed / ctx->m_unIMUSamples;
                }
                ctx->m_unIMUSamples = 0;
                ctx->m_ulIMUSampleTimestampNS = now;
            }

            ctx->m_ulTimestampNS += ctx->m_ulIMUUpdateIntervalNS;
            sensor_timestamp[0] = ctx->m_ulTimestampNS;
            ctx->m_ulTimestampNS += ctx->m_ulIMUUpdateIntervalNS;
            sensor_timestamp[1] = ctx->m_ulTimestampNS;
            ctx->m_ulTimestampNS += ctx->m_ulIMUUpdateIntervalNS;
            sensor_timestamp[2] = ctx->m_ulTimestampNS;

            if (!ctx->device->parent ||
                ctx->m_eControllerType == k_eSwitchDeviceInfoControllerType_JoyConRight) {
                SendSensorUpdate(timestamp, joystick, ctx, SDL_SENSOR_GYRO, sensor_timestamp[0], &imuState[2].sGyroX);
                SendSensorUpdate(timestamp, joystick, ctx, SDL_SENSOR_ACCEL, sensor_timestamp[0], &imuState[2].sAccelX);

                SendSensorUpdate(timestamp, joystick, ctx, SDL_SENSOR_GYRO, sensor_timestamp[1], &imuState[1].sGyroX);
                SendSensorUpdate(timestamp, joystick, ctx, SDL_SENSOR_ACCEL, sensor_timestamp[1], &imuState[1].sAccelX);

                SendSensorUpdate(timestamp, joystick, ctx, SDL_SENSOR_GYRO, sensor_timestamp[2], &imuState[0].sGyroX);
                SendSensorUpdate(timestamp, joystick, ctx, SDL_SENSOR_ACCEL, sensor_timestamp[2], &imuState[0].sAccelX);
            }

            if (ctx->device->parent &&
                ctx->m_eControllerType == k_eSwitchDeviceInfoControllerType_JoyConLeft) {
                SendSensorUpdate(timestamp, joystick, ctx, SDL_SENSOR_GYRO_L, sensor_timestamp[0], &imuState[2].sGyroX);
                SendSensorUpdate(timestamp, joystick, ctx, SDL_SENSOR_ACCEL_L, sensor_timestamp[0], &imuState[2].sAccelX);

                SendSensorUpdate(timestamp, joystick, ctx, SDL_SENSOR_GYRO_L, sensor_timestamp[1], &imuState[1].sGyroX);
                SendSensorUpdate(timestamp, joystick, ctx, SDL_SENSOR_ACCEL_L, sensor_timestamp[1], &imuState[1].sAccelX);

                SendSensorUpdate(timestamp, joystick, ctx, SDL_SENSOR_GYRO_L, sensor_timestamp[2], &imuState[0].sGyroX);
                SendSensorUpdate(timestamp, joystick, ctx, SDL_SENSOR_ACCEL_L, sensor_timestamp[2], &imuState[0].sAccelX);
            }
            if (ctx->device->parent &&
                ctx->m_eControllerType == k_eSwitchDeviceInfoControllerType_JoyConRight) {
                SendSensorUpdate(timestamp, joystick, ctx, SDL_SENSOR_GYRO_R, sensor_timestamp[0], &imuState[2].sGyroX);
                SendSensorUpdate(timestamp, joystick, ctx, SDL_SENSOR_ACCEL_R, sensor_timestamp[0], &imuState[2].sAccelX);

                SendSensorUpdate(timestamp, joystick, ctx, SDL_SENSOR_GYRO_R, sensor_timestamp[1], &imuState[1].sGyroX);
                SendSensorUpdate(timestamp, joystick, ctx, SDL_SENSOR_ACCEL_R, sensor_timestamp[1], &imuState[1].sAccelX);

                SendSensorUpdate(timestamp, joystick, ctx, SDL_SENSOR_GYRO_R, sensor_timestamp[2], &imuState[0].sGyroX);
                SendSensorUpdate(timestamp, joystick, ctx, SDL_SENSOR_ACCEL_R, sensor_timestamp[2], &imuState[0].sAccelX);
            }

        } else if (ctx->m_bHasSensorData) {
            // Uh oh, someone turned off the IMU?
            const int IMU_RESET_DELAY_MS = 3000;
            Uint64 now = SDL_GetTicks();

            if (now >= (ctx->m_ulLastIMUReset + IMU_RESET_DELAY_MS)) {
                SDL_HIDAPI_Device *device = ctx->device;

                if (device->updating) {
                    SDL_UnlockMutex(device->dev_lock);
                }

                SetIMUEnabled(ctx, true);

                if (device->updating) {
                    SDL_LockMutex(device->dev_lock);
                }
                ctx->m_ulLastIMUReset = now;
            }

        } else {
            // We have never gotten IMU data, probably not supported on this device
        }
    }

    ctx->m_lastFullState = *packet;
}

static bool HIDAPI_DriverSwitch_UpdateDevice(SDL_HIDAPI_Device *device)
{
    SDL_DriverSwitch_Context *ctx = (SDL_DriverSwitch_Context *)device->context;
    SDL_Joystick *joystick = NULL;
    int size;
    int packet_count = 0;
    Uint64 now = SDL_GetTicks();

    if (device->num_joysticks > 0) {
        joystick = SDL_GetJoystickFromID(device->joysticks[0]);
    }

    while ((size = ReadInput(ctx)) > 0) {
#ifdef DEBUG_SWITCH_PROTOCOL
        HIDAPI_DumpPacket("Nintendo Switch packet: size = %d", ctx->m_rgucReadBuffer, size);
#endif
        ++packet_count;
        ctx->m_ulLastInput = now;

        if (!joystick) {
            continue;
        }

        if (ctx->m_bInputOnly) {
            HandleInputOnlyControllerState(joystick, ctx, (SwitchInputOnlyControllerStatePacket_t *)&ctx->m_rgucReadBuffer[0]);
        } else {
            if (ctx->m_rgucReadBuffer[0] == k_eSwitchInputReportIDs_SubcommandReply) {
                continue;
            }

            ctx->m_nCurrentInputMode = ctx->m_rgucReadBuffer[0];

            switch (ctx->m_rgucReadBuffer[0]) {
            case k_eSwitchInputReportIDs_SimpleControllerState:
                HandleSimpleControllerState(joystick, ctx, (SwitchSimpleStatePacket_t *)&ctx->m_rgucReadBuffer[1]);
                break;
            case k_eSwitchInputReportIDs_FullControllerState:
            case k_eSwitchInputReportIDs_FullControllerAndMcuState:
                // This is the extended report, we can enable sensors now in auto mode
                UpdateEnhancedModeOnEnhancedReport(ctx);

                HandleFullControllerState(joystick, ctx, (SwitchStatePacket_t *)&ctx->m_rgucReadBuffer[1]);
                break;
            default:
                break;
            }
        }
    }

    if (joystick) {
        if (packet_count == 0) {
            if (!ctx->m_bInputOnly && !device->is_bluetooth &&
                ctx->device->product_id != USB_PRODUCT_NINTENDO_SWITCH_JOYCON_GRIP) {
                const int INPUT_WAIT_TIMEOUT_MS = 100;
                if (now >= (ctx->m_ulLastInput + INPUT_WAIT_TIMEOUT_MS)) {
                    // Steam may have put the controller back into non-reporting mode
                    bool wasSyncWrite = ctx->m_bSyncWrite;

                    ctx->m_bSyncWrite = true;
                    WriteProprietary(ctx, k_eSwitchProprietaryCommandIDs_ForceUSB, NULL, 0, false);
                    ctx->m_bSyncWrite = wasSyncWrite;
                }
            } else if (device->is_bluetooth &&
                       ctx->m_nCurrentInputMode != k_eSwitchInputReportIDs_SimpleControllerState) {
                const int INPUT_WAIT_TIMEOUT_MS = 3000;
                if (now >= (ctx->m_ulLastInput + INPUT_WAIT_TIMEOUT_MS)) {
                    // Bluetooth may have disconnected, try reopening the controller
                    size = -1;
                }
            }
        }

        if (ctx->m_bRumblePending || ctx->m_bRumbleZeroPending) {
            HIDAPI_DriverSwitch_SendPendingRumble(ctx);
        } else if (ctx->m_bRumbleActive &&
                   now >= (ctx->m_ulRumbleSent + RUMBLE_REFRESH_FREQUENCY_MS)) {
#ifdef DEBUG_RUMBLE
            SDL_Log("Sent continuing rumble, %d ms after previous rumble", now - ctx->m_ulRumbleSent);
#endif
            WriteRumble(ctx);
        }
    }

    // Reconnect the Bluetooth device once the USB device is gone
    if (device->num_joysticks == 0 && device->is_bluetooth && packet_count > 0 &&
        !device->parent &&
        !HIDAPI_HasConnectedUSBDevice(device->serial)) {
        HIDAPI_JoystickConnected(device, NULL);
    }

    if (size < 0 && device->num_joysticks > 0) {
        // Read error, device is disconnected
        HIDAPI_JoystickDisconnected(device, device->joysticks[0]);
    }
    return (size >= 0);
}

static void HIDAPI_DriverSwitch_CloseJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
    SDL_DriverSwitch_Context *ctx = (SDL_DriverSwitch_Context *)device->context;

    if (!ctx->m_bInputOnly) {
        // Restore simple input mode for other applications
        if (!ctx->m_nInitialInputMode ||
            ctx->m_nInitialInputMode == k_eSwitchInputReportIDs_SimpleControllerState) {
            SetInputMode(ctx, k_eSwitchInputReportIDs_SimpleControllerState);
        }
    }

    SDL_RemoveHintCallback(SDL_HINT_JOYSTICK_ENHANCED_REPORTS,
                        SDL_EnhancedReportsChanged, ctx);

    if (ctx->m_eControllerType == k_eSwitchDeviceInfoControllerType_JoyConLeft ||
        ctx->m_eControllerType == k_eSwitchDeviceInfoControllerType_JoyConRight) {
        SDL_RemoveHintCallback(SDL_HINT_JOYSTICK_HIDAPI_JOYCON_HOME_LED,
                            SDL_HomeLEDHintChanged, ctx);
    } else {
        SDL_RemoveHintCallback(SDL_HINT_JOYSTICK_HIDAPI_SWITCH_HOME_LED,
                            SDL_HomeLEDHintChanged, ctx);
    }

    SDL_RemoveHintCallback(SDL_HINT_JOYSTICK_HIDAPI_SWITCH_PLAYER_LED,
                        SDL_PlayerLEDHintChanged, ctx);

    ctx->joystick = NULL;

    ctx->m_bReportSensors = false;
    ctx->m_bEnhancedMode = false;
    ctx->m_bEnhancedModeAvailable = false;
}

static void HIDAPI_DriverSwitch_FreeDevice(SDL_HIDAPI_Device *device)
{
}

SDL_HIDAPI_DeviceDriver SDL_HIDAPI_DriverNintendoClassic = {
    SDL_HINT_JOYSTICK_HIDAPI_NINTENDO_CLASSIC,
    true,
    HIDAPI_DriverNintendoClassic_RegisterHints,
    HIDAPI_DriverNintendoClassic_UnregisterHints,
    HIDAPI_DriverNintendoClassic_IsEnabled,
    HIDAPI_DriverNintendoClassic_IsSupportedDevice,
    HIDAPI_DriverSwitch_InitDevice,
    HIDAPI_DriverSwitch_GetDevicePlayerIndex,
    HIDAPI_DriverSwitch_SetDevicePlayerIndex,
    HIDAPI_DriverSwitch_UpdateDevice,
    HIDAPI_DriverSwitch_OpenJoystick,
    HIDAPI_DriverSwitch_RumbleJoystick,
    HIDAPI_DriverSwitch_RumbleJoystickTriggers,
    HIDAPI_DriverSwitch_GetJoystickCapabilities,
    HIDAPI_DriverSwitch_SetJoystickLED,
    HIDAPI_DriverSwitch_SendJoystickEffect,
    HIDAPI_DriverSwitch_SetJoystickSensorsEnabled,
    HIDAPI_DriverSwitch_CloseJoystick,
    HIDAPI_DriverSwitch_FreeDevice,
};

SDL_HIDAPI_DeviceDriver SDL_HIDAPI_DriverJoyCons = {
    SDL_HINT_JOYSTICK_HIDAPI_JOY_CONS,
    true,
    HIDAPI_DriverJoyCons_RegisterHints,
    HIDAPI_DriverJoyCons_UnregisterHints,
    HIDAPI_DriverJoyCons_IsEnabled,
    HIDAPI_DriverJoyCons_IsSupportedDevice,
    HIDAPI_DriverSwitch_InitDevice,
    HIDAPI_DriverSwitch_GetDevicePlayerIndex,
    HIDAPI_DriverSwitch_SetDevicePlayerIndex,
    HIDAPI_DriverSwitch_UpdateDevice,
    HIDAPI_DriverSwitch_OpenJoystick,
    HIDAPI_DriverSwitch_RumbleJoystick,
    HIDAPI_DriverSwitch_RumbleJoystickTriggers,
    HIDAPI_DriverSwitch_GetJoystickCapabilities,
    HIDAPI_DriverSwitch_SetJoystickLED,
    HIDAPI_DriverSwitch_SendJoystickEffect,
    HIDAPI_DriverSwitch_SetJoystickSensorsEnabled,
    HIDAPI_DriverSwitch_CloseJoystick,
    HIDAPI_DriverSwitch_FreeDevice,
};

SDL_HIDAPI_DeviceDriver SDL_HIDAPI_DriverSwitch = {
    SDL_HINT_JOYSTICK_HIDAPI_SWITCH,
    true,
    HIDAPI_DriverSwitch_RegisterHints,
    HIDAPI_DriverSwitch_UnregisterHints,
    HIDAPI_DriverSwitch_IsEnabled,
    HIDAPI_DriverSwitch_IsSupportedDevice,
    HIDAPI_DriverSwitch_InitDevice,
    HIDAPI_DriverSwitch_GetDevicePlayerIndex,
    HIDAPI_DriverSwitch_SetDevicePlayerIndex,
    HIDAPI_DriverSwitch_UpdateDevice,
    HIDAPI_DriverSwitch_OpenJoystick,
    HIDAPI_DriverSwitch_RumbleJoystick,
    HIDAPI_DriverSwitch_RumbleJoystickTriggers,
    HIDAPI_DriverSwitch_GetJoystickCapabilities,
    HIDAPI_DriverSwitch_SetJoystickLED,
    HIDAPI_DriverSwitch_SendJoystickEffect,
    HIDAPI_DriverSwitch_SetJoystickSensorsEnabled,
    HIDAPI_DriverSwitch_CloseJoystick,
    HIDAPI_DriverSwitch_FreeDevice,
};

#endif // SDL_JOYSTICK_HIDAPI_SWITCH

#endif // SDL_JOYSTICK_HIDAPI
