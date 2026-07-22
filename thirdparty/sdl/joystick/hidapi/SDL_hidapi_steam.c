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

#ifdef SDL_JOYSTICK_HIDAPI_STEAM

// Define this if you want to log all packets from the controller
// #define DEBUG_STEAM_PROTOCOL

#define SDL_HINT_JOYSTICK_HIDAPI_STEAM_PAIRING_ENABLED    "SDL_JOYSTICK_HIDAPI_STEAM_PAIRING_ENABLED"

#if defined(SDL_PLATFORM_ANDROID) || defined(SDL_PLATFORM_IOS) || defined(SDL_PLATFORM_TVOS)
// This requires prompting for Bluetooth permissions, so make sure the application really wants it
#define SDL_HINT_JOYSTICK_HIDAPI_STEAM_DEFAULT  false
#else
#define SDL_HINT_JOYSTICK_HIDAPI_STEAM_DEFAULT  SDL_GetHintBoolean(SDL_HINT_JOYSTICK_HIDAPI, SDL_HIDAPI_DEFAULT)
#endif

#define PAIRING_STATE_DURATION_SECONDS  60


/*****************************************************************************************************/

#include "steam/controller_constants.h"
#include "steam/controller_structs.h"

enum
{
    SDL_GAMEPAD_BUTTON_STEAM_RIGHT_PADDLE = 11,
    SDL_GAMEPAD_BUTTON_STEAM_LEFT_PADDLE,
    SDL_GAMEPAD_NUM_STEAM_BUTTONS,
};

typedef struct SteamControllerStateInternal_t
{
    // Controller Type for this Controller State
    Uint32 eControllerType;

    // If packet num matches that on your prior call, then the controller state hasn't been changed since
    // your last call and there is no need to process it
    Uint32 unPacketNum;

    // bit flags for each of the buttons
    Uint64 ulButtons;

    // Left pad coordinates
    short sLeftPadX;
    short sLeftPadY;

    // Right pad coordinates
    short sRightPadX;
    short sRightPadY;

    // Center pad coordinates
    short sCenterPadX;
    short sCenterPadY;

    // Left analog stick coordinates
    short sLeftStickX;
    short sLeftStickY;

    // Right analog stick coordinates
    short sRightStickX;
    short sRightStickY;

    unsigned short sTriggerL;
    unsigned short sTriggerR;

    short sAccelX;
    short sAccelY;
    short sAccelZ;

    short sGyroX;
    short sGyroY;
    short sGyroZ;

    float sGyroQuatW;
    float sGyroQuatX;
    float sGyroQuatY;
    float sGyroQuatZ;

    short sGyroSteeringAngle;

    unsigned short sBatteryLevel;

    // Pressure sensor data.
    unsigned short sPressurePadLeft;
    unsigned short sPressurePadRight;

    unsigned short sPressureBumperLeft;
    unsigned short sPressureBumperRight;

    // Internal state data
    short sPrevLeftPad[2];
    short sPrevLeftStick[2];
} SteamControllerStateInternal_t;

// Defines for ulButtons in SteamControllerStateInternal_t
#define STEAM_RIGHT_TRIGGER_MASK           0x00000001
#define STEAM_LEFT_TRIGGER_MASK            0x00000002
#define STEAM_RIGHT_BUMPER_MASK            0x00000004
#define STEAM_LEFT_BUMPER_MASK             0x00000008
#define STEAM_BUTTON_NORTH_MASK            0x00000010 // Y
#define STEAM_BUTTON_EAST_MASK             0x00000020 // B
#define STEAM_BUTTON_WEST_MASK             0x00000040 // X
#define STEAM_BUTTON_SOUTH_MASK            0x00000080 // A
#define STEAM_DPAD_UP_MASK                 0x00000100 // DPAD UP
#define STEAM_DPAD_RIGHT_MASK              0x00000200 // DPAD RIGHT
#define STEAM_DPAD_LEFT_MASK               0x00000400 // DPAD LEFT
#define STEAM_DPAD_DOWN_MASK               0x00000800 // DPAD DOWN
#define STEAM_BUTTON_MENU_MASK             0x00001000 // SELECT
#define STEAM_BUTTON_STEAM_MASK            0x00002000 // GUIDE
#define STEAM_BUTTON_ESCAPE_MASK           0x00004000 // START
#define STEAM_BUTTON_BACK_LEFT_MASK        0x00008000
#define STEAM_BUTTON_BACK_RIGHT_MASK       0x00010000
#define STEAM_BUTTON_LEFTPAD_CLICKED_MASK  0x00020000
#define STEAM_BUTTON_RIGHTPAD_CLICKED_MASK 0x00040000
#define STEAM_LEFTPAD_FINGERDOWN_MASK      0x00080000
#define STEAM_RIGHTPAD_FINGERDOWN_MASK     0x00100000
#define STEAM_JOYSTICK_BUTTON_MASK         0x00400000
#define STEAM_LEFTPAD_AND_JOYSTICK_MASK    0x00800000

// Look for report version 0x0001, type WIRELESS (3), length >= 1 byte
#define D0G_IS_VALID_WIRELESS_EVENT(data, len) ((len) >= 5 && (data)[0] == 1 && (data)[1] == 0 && (data)[2] == 3 && (data)[3] >= 1)
#define D0G_GET_WIRELESS_EVENT_TYPE(data)      ((data)[4])
#define D0G_WIRELESS_DISCONNECTED              1
#define D0G_WIRELESS_ESTABLISHED               2
#define D0G_WIRELESS_NEWLYPAIRED               3

#define D0G_IS_WIRELESS_DISCONNECT(data, len) (D0G_IS_VALID_WIRELESS_EVENT(data, len) && D0G_GET_WIRELESS_EVENT_TYPE(data) == D0G_WIRELESS_DISCONNECTED)
#define D0G_IS_WIRELESS_CONNECT(data, len)    (D0G_IS_VALID_WIRELESS_EVENT(data, len) && D0G_GET_WIRELESS_EVENT_TYPE(data) != D0G_WIRELESS_DISCONNECTED)


#define MAX_REPORT_SEGMENT_PAYLOAD_SIZE 18
/*
 * SteamControllerPacketAssembler has to be used when reading output repots from controllers.
 */
typedef struct
{
    uint8_t uBuffer[MAX_REPORT_SEGMENT_PAYLOAD_SIZE * 8 + 1];
    int nExpectedSegmentNumber;
    bool bIsBle;
} SteamControllerPacketAssembler;

#undef clamp
#define clamp(val, min, max) (((val) > (max)) ? (max) : (((val) < (min)) ? (min) : (val)))

#undef offsetof
#define offsetof(s, m) (size_t) & (((s *)0)->m)

#ifdef DEBUG_STEAM_CONTROLLER
#define DPRINTF(format, ...) printf(format, ##__VA_ARGS__)
#define HEXDUMP(ptr, len)    hexdump(ptr, len)
#else
#define DPRINTF(format, ...)
#define HEXDUMP(ptr, len)
#endif
#define printf SDL_Log

#define MAX_REPORT_SEGMENT_SIZE        (MAX_REPORT_SEGMENT_PAYLOAD_SIZE + 2)
#define CALC_REPORT_SEGMENT_NUM(index) ((index / MAX_REPORT_SEGMENT_PAYLOAD_SIZE) & 0x07)
#define REPORT_SEGMENT_DATA_FLAG       0x80
#define REPORT_SEGMENT_LAST_FLAG       0x40
#define BLE_REPORT_NUMBER              0x03

#define STEAMCONTROLLER_TRIGGER_MAX_ANALOG 26000

// Enable mouse mode when using the Steam Controller locally
#undef ENABLE_MOUSE_MODE

// Wireless firmware quirk: the firmware intentionally signals "failure" when performing
// SET_FEATURE / GET_FEATURE when it actually means "pending radio roundtrip". The only
// way to make SET_FEATURE / GET_FEATURE work is to loop several times with a sleep. If
// it takes more than 50ms to get the response for SET_FEATURE / GET_FEATURE, we assume
// that the controller has failed.
#define RADIO_WORKAROUND_SLEEP_ATTEMPTS    50
#define RADIO_WORKAROUND_SLEEP_DURATION_US 500

// This was defined by experimentation. 2000 seemed to work but to give that extra bit of margin, set to 3ms.
#define CONTROLLER_CONFIGURATION_DELAY_US 3000

static uint8_t GetSegmentHeader(int nSegmentNumber, bool bLastPacket)
{
    uint8_t header = REPORT_SEGMENT_DATA_FLAG;
    header |= nSegmentNumber;
    if (bLastPacket) {
        header |= REPORT_SEGMENT_LAST_FLAG;
    }

    return header;
}

static void hexdump(const uint8_t *ptr, int len)
{
    HIDAPI_DumpPacket("Data", ptr, len);
}

static void ResetSteamControllerPacketAssembler(SteamControllerPacketAssembler *pAssembler)
{
    SDL_memset(pAssembler->uBuffer, 0, sizeof(pAssembler->uBuffer));
    pAssembler->nExpectedSegmentNumber = 0;
}

static void InitializeSteamControllerPacketAssembler(SteamControllerPacketAssembler *pAssembler, bool bIsBle)
{
    pAssembler->bIsBle = bIsBle;
    ResetSteamControllerPacketAssembler(pAssembler);
}

// Returns:
//     <0 on error
//     0 on not ready
//     Complete packet size on completion
static int WriteSegmentToSteamControllerPacketAssembler(SteamControllerPacketAssembler *pAssembler, const uint8_t *pSegment, int nSegmentLength)
{
    if (pAssembler->bIsBle) {
        uint8_t uSegmentHeader = pSegment[1];
        int nSegmentNumber = uSegmentHeader & 0x07;

        HEXDUMP(pSegment, nSegmentLength);

        if (pSegment[0] != BLE_REPORT_NUMBER) {
            // We may get keyboard/mouse input events until controller stops sending them
            return 0;
        }

        if (nSegmentLength != MAX_REPORT_SEGMENT_SIZE) {
            printf("Bad segment size! %d\n", nSegmentLength);
            hexdump(pSegment, nSegmentLength);
            ResetSteamControllerPacketAssembler(pAssembler);
            return -1;
        }

        DPRINTF("GOT PACKET HEADER = 0x%x\n", uSegmentHeader);

        if (!(uSegmentHeader & REPORT_SEGMENT_DATA_FLAG)) {
            // We get empty segments, just ignore them
            return 0;
        }

        if (nSegmentNumber != pAssembler->nExpectedSegmentNumber) {
            ResetSteamControllerPacketAssembler(pAssembler);

            if (nSegmentNumber) {
                // This happens occasionally
                DPRINTF("Bad segment number, got %d, expected %d\n",
                        nSegmentNumber, pAssembler->nExpectedSegmentNumber);
                return -1;
            }
        }

        SDL_memcpy(pAssembler->uBuffer + nSegmentNumber * MAX_REPORT_SEGMENT_PAYLOAD_SIZE,
                   pSegment + 2, // ignore header and report number
                   MAX_REPORT_SEGMENT_PAYLOAD_SIZE);

        if (uSegmentHeader & REPORT_SEGMENT_LAST_FLAG) {
            pAssembler->nExpectedSegmentNumber = 0;
            return (nSegmentNumber + 1) * MAX_REPORT_SEGMENT_PAYLOAD_SIZE;
        }

        pAssembler->nExpectedSegmentNumber++;
    } else {
        // Just pass through
        SDL_memcpy(pAssembler->uBuffer,
                   pSegment,
                   nSegmentLength);
        return nSegmentLength;
    }

    return 0;
}

#define BLE_MAX_READ_RETRIES 8

static int SetFeatureReport(SDL_HIDAPI_Device *dev, const unsigned char uBuffer[65], int nActualDataLen)
{
    int nRet = -1;

    DPRINTF("SetFeatureReport %p %p %d\n", dev, uBuffer, nActualDataLen);

    if (dev->is_bluetooth) {
        int nSegmentNumber = 0;
        uint8_t uPacketBuffer[MAX_REPORT_SEGMENT_SIZE];
        const unsigned char *pBufferPtr = uBuffer + 1;

        if (nActualDataLen < 1) {
            return -1;
        }

        // Skip report number in data
        nActualDataLen--;

        while (nActualDataLen > 0) {
            int nBytesInPacket = nActualDataLen > MAX_REPORT_SEGMENT_PAYLOAD_SIZE ? MAX_REPORT_SEGMENT_PAYLOAD_SIZE : nActualDataLen;

            nActualDataLen -= nBytesInPacket;

            // Construct packet
            SDL_memset(uPacketBuffer, 0, sizeof(uPacketBuffer));
            uPacketBuffer[0] = BLE_REPORT_NUMBER;
            uPacketBuffer[1] = GetSegmentHeader(nSegmentNumber, nActualDataLen == 0);
            SDL_memcpy(&uPacketBuffer[2], pBufferPtr, nBytesInPacket);

            pBufferPtr += nBytesInPacket;
            nSegmentNumber++;

            nRet = SDL_hid_send_feature_report(dev->dev, uPacketBuffer, sizeof(uPacketBuffer));
        }
    } else {
        for (int nRetries = 0; nRetries < RADIO_WORKAROUND_SLEEP_ATTEMPTS; nRetries++) {
            nRet = SDL_hid_send_feature_report(dev->dev, uBuffer, 65);
            if (nRet >= 0) {
                break;
            }

            SDL_DelayNS(RADIO_WORKAROUND_SLEEP_DURATION_US * 1000);
        }
    }

    DPRINTF("SetFeatureReport() ret = %d\n", nRet);

    return nRet;
}

static int GetFeatureReport(SDL_HIDAPI_Device *dev, unsigned char uBuffer[65])
{
    int nRet = -1;

    DPRINTF("GetFeatureReport( %p %p )\n", dev, uBuffer);

    if (dev->is_bluetooth) {
        int nRetries = 0;
        uint8_t uSegmentBuffer[MAX_REPORT_SEGMENT_SIZE + 1];
        uint8_t ucBytesToRead = MAX_REPORT_SEGMENT_SIZE;
        uint8_t ucDataStartOffset = 0;

        SteamControllerPacketAssembler assembler;
        InitializeSteamControllerPacketAssembler(&assembler, dev->is_bluetooth);

        // On Windows and macOS, BLE devices get 2 copies of the feature report ID, one that is removed by ReadFeatureReport,
        // and one that's included in the buffer we receive. We pad the bytes to read and skip over the report ID
        // if necessary.
#if defined(SDL_PLATFORM_WIN32) || defined(SDL_PLATFORM_MACOS)
        ++ucBytesToRead;
        ++ucDataStartOffset;
#endif

        while (nRetries < BLE_MAX_READ_RETRIES) {
            SDL_memset(uSegmentBuffer, 0, sizeof(uSegmentBuffer));
            uSegmentBuffer[0] = BLE_REPORT_NUMBER;
            nRet = SDL_hid_get_feature_report(dev->dev, uSegmentBuffer, ucBytesToRead);

            DPRINTF("GetFeatureReport ble ret=%d\n", nRet);
            HEXDUMP(uSegmentBuffer, nRet);

            // Zero retry counter if we got data
            if (nRet > 2 && (uSegmentBuffer[ucDataStartOffset + 1] & REPORT_SEGMENT_DATA_FLAG)) {
                nRetries = 0;
            } else {
                nRetries++;
            }

            if (nRet > 0) {
                int nPacketLength = WriteSegmentToSteamControllerPacketAssembler(&assembler,
                                                                                 uSegmentBuffer + ucDataStartOffset,
                                                                                 nRet - ucDataStartOffset);

                if (nPacketLength > 0 && nPacketLength < 65) {
                    // Leave space for "report number"
                    uBuffer[0] = 0;
                    SDL_memcpy(uBuffer + 1, assembler.uBuffer, nPacketLength);
                    return nPacketLength;
                }
            }
        }
        printf("Could not get a full ble packet after %d retries\n", nRetries);
        return -1;
    } else {
        SDL_memset(uBuffer, 0, 65);

        for (int nRetries = 0; nRetries < RADIO_WORKAROUND_SLEEP_ATTEMPTS; nRetries++) {
            nRet = SDL_hid_get_feature_report(dev->dev, uBuffer, 65);
            if (nRet >= 0) {
                break;
            }

            SDL_DelayNS(RADIO_WORKAROUND_SLEEP_DURATION_US * 1000);
        }

        DPRINTF("GetFeatureReport USB ret=%d\n", nRet);
        HEXDUMP(uBuffer, nRet);
    }

    return nRet;
}

static int ReadResponse(SDL_HIDAPI_Device *dev, uint8_t uBuffer[65], int nExpectedResponse)
{
    for (int nRetries = 0; nRetries < 10; nRetries++) {
        int nRet = GetFeatureReport(dev, uBuffer);

        DPRINTF("ReadResponse( %p %p 0x%x )\n", dev, uBuffer, nExpectedResponse);

        if (nRet < 0) {
            continue;
        }

        DPRINTF("ReadResponse got %d bytes of data: ", nRet);
        HEXDUMP(uBuffer, nRet);

        if (uBuffer[1] != nExpectedResponse) {
            continue;
        }

        return nRet;
    }
    return -1;
}

//---------------------------------------------------------------------------
// Reset steam controller (unmap buttons and pads) and re-fetch capability bits
//---------------------------------------------------------------------------
static bool ResetSteamController(SDL_HIDAPI_Device *dev, bool bSuppressErrorSpew, uint32_t *punUpdateRateUS)
{
    // Firmware quirk: Set Feature and Get Feature requests always require a 65-byte buffer.
    unsigned char buf[65];
    unsigned int i;
    int res = -1;
    int nSettings = 0;
    int nAttributesLength;
    FeatureReportMsg *msg;
    uint32_t unUpdateRateUS = 9000; // Good default rate

    DPRINTF("ResetSteamController hid=%p\n", dev);

    buf[0] = 0;
    buf[1] = ID_GET_ATTRIBUTES_VALUES;
    res = SetFeatureReport(dev, buf, 2);
    if (res < 0) {
        if (!bSuppressErrorSpew) {
            printf("GET_ATTRIBUTES_VALUES failed for controller %p\n", dev);
        }
        return false;
    }

    // Retrieve GET_ATTRIBUTES_VALUES result
    // Wireless controller endpoints without a connected controller will return nAttrs == 0
    res = ReadResponse(dev, buf, ID_GET_ATTRIBUTES_VALUES);
    if (res < 0 || buf[1] != ID_GET_ATTRIBUTES_VALUES) {
        HEXDUMP(buf, res);
        if (!bSuppressErrorSpew) {
            printf("Bad GET_ATTRIBUTES_VALUES response for controller %p\n", dev);
        }
        return false;
    }

    nAttributesLength = buf[2];
    if (nAttributesLength > res) {
        if (!bSuppressErrorSpew) {
            printf("Bad GET_ATTRIBUTES_VALUES response for controller %p\n", dev);
        }
        return false;
    }

    msg = (FeatureReportMsg *)&buf[1];
    for (i = 0; i < (int)msg->header.length / sizeof(ControllerAttribute); ++i) {
        uint8_t unAttribute = msg->payload.getAttributes.attributes[i].attributeTag;
        uint32_t unValue = msg->payload.getAttributes.attributes[i].attributeValue;

        switch (unAttribute) {
        case ATTRIB_UNIQUE_ID:
            break;
        case ATTRIB_PRODUCT_ID:
            break;
        case ATTRIB_CAPABILITIES:
            break;
        case ATTRIB_CONNECTION_INTERVAL_IN_US:
            unUpdateRateUS = unValue;
            break;
        default:
            break;
        }
    }
    if (punUpdateRateUS) {
        *punUpdateRateUS = unUpdateRateUS;
    }

    // Clear digital button mappings
    buf[0] = 0;
    buf[1] = ID_CLEAR_DIGITAL_MAPPINGS;
    res = SetFeatureReport(dev, buf, 2);
    if (res < 0) {
        if (!bSuppressErrorSpew) {
            printf("CLEAR_DIGITAL_MAPPINGS failed for controller %p\n", dev);
        }
        return false;
    }

    // Reset the default settings
    SDL_memset(buf, 0, 65);
    buf[1] = ID_LOAD_DEFAULT_SETTINGS;
    buf[2] = 0;
    res = SetFeatureReport(dev, buf, 3);
    if (res < 0) {
        if (!bSuppressErrorSpew) {
            printf("LOAD_DEFAULT_SETTINGS failed for controller %p\n", dev);
        }
        return false;
    }

    // Apply custom settings - clear trackpad modes (cancel mouse emulation), etc
#define ADD_SETTING(SETTING, VALUE)                        \
    buf[3 + nSettings * 3] = SETTING;                      \
    buf[3 + nSettings * 3 + 1] = ((uint16_t)VALUE) & 0xFF; \
    buf[3 + nSettings * 3 + 2] = ((uint16_t)VALUE) >> 8;   \
    ++nSettings;

    SDL_memset(buf, 0, 65);
    buf[1] = ID_SET_SETTINGS_VALUES;
    ADD_SETTING(SETTING_WIRELESS_PACKET_VERSION, 2);
    ADD_SETTING(SETTING_LEFT_TRACKPAD_MODE, TRACKPAD_NONE);
#ifdef ENABLE_MOUSE_MODE
    ADD_SETTING(SETTING_RIGHT_TRACKPAD_MODE, TRACKPAD_ABSOLUTE_MOUSE);
    ADD_SETTING(SETTING_SMOOTH_ABSOLUTE_MOUSE, 1);
    ADD_SETTING(SETTING_MOMENTUM_MAXIMUM_VELOCITY, 20000); // [0-20000] default 8000
    ADD_SETTING(SETTING_MOMENTUM_DECAY_AMOUNT, 50);       // [0-50] default 5
#else
    ADD_SETTING(SETTING_RIGHT_TRACKPAD_MODE, TRACKPAD_NONE);
    ADD_SETTING(SETTING_SMOOTH_ABSOLUTE_MOUSE, 0);
#endif
    buf[2] = (unsigned char)(nSettings * 3);

    res = SetFeatureReport(dev, buf, 3 + nSettings * 3);
    if (res < 0) {
        if (!bSuppressErrorSpew) {
            printf("SET_SETTINGS failed for controller %p\n", dev);
        }
        return false;
    }

#ifdef ENABLE_MOUSE_MODE
    // Wait for ID_CLEAR_DIGITAL_MAPPINGS to be processed on the controller
    bool bMappingsCleared = false;
    int iRetry;
    for (iRetry = 0; iRetry < 2; ++iRetry) {
        SDL_memset(buf, 0, 65);
        buf[1] = ID_GET_DIGITAL_MAPPINGS;
        buf[2] = 1; // one byte - requesting from index 0
        buf[3] = 0;
        res = SetFeatureReport(dev, buf, 4);
        if (res < 0) {
            printf("GET_DIGITAL_MAPPINGS failed for controller %p\n", dev);
            return false;
        }

        res = ReadResponse(dev, buf, ID_GET_DIGITAL_MAPPINGS);
        if (res < 0 || buf[1] != ID_GET_DIGITAL_MAPPINGS) {
            printf("Bad GET_DIGITAL_MAPPINGS response for controller %p\n", dev);
            return false;
        }

        // If the length of the digital mappings result is not 1 (index byte, no mappings) then clearing hasn't executed
        if (buf[2] == 1 && buf[3] == 0xFF) {
            bMappingsCleared = true;
            break;
        }
        usleep(CONTROLLER_CONFIGURATION_DELAY_US);
    }

    if (!bMappingsCleared && !bSuppressErrorSpew) {
        printf("Warning: CLEAR_DIGITAL_MAPPINGS never completed for controller %p\n", dev);
    }

    // Set our new mappings
    SDL_memset(buf, 0, 65);
    buf[1] = ID_SET_DIGITAL_MAPPINGS;
    buf[2] = 6; // 2 settings x 3 bytes
    buf[3] = IO_DIGITAL_BUTTON_RIGHT_TRIGGER;
    buf[4] = DEVICE_MOUSE;
    buf[5] = MOUSE_BTN_LEFT;
    buf[6] = IO_DIGITAL_BUTTON_LEFT_TRIGGER;
    buf[7] = DEVICE_MOUSE;
    buf[8] = MOUSE_BTN_RIGHT;

    res = SetFeatureReport(dev, buf, 9);
    if (res < 0) {
        if (!bSuppressErrorSpew) {
            printf("SET_DIGITAL_MAPPINGS failed for controller %p\n", dev);
        }
        return false;
    }
#endif // ENABLE_MOUSE_MODE

    return true;
}

//---------------------------------------------------------------------------
// Read from a Steam Controller
//---------------------------------------------------------------------------
static int ReadSteamController(SDL_hid_device *dev, uint8_t *pData, int nDataSize)
{
    SDL_memset(pData, 0, nDataSize);
    pData[0] = BLE_REPORT_NUMBER; // hid_read will also overwrite this with the same value, 0x03
    return SDL_hid_read(dev, pData, nDataSize);
}

//---------------------------------------------------------------------------
// Set Steam Controller pairing state
//---------------------------------------------------------------------------
static void SetPairingState(SDL_HIDAPI_Device *dev, bool bEnablePairing)
{
    unsigned char buf[65];
    SDL_memset(buf, 0, 65);
    buf[1] = ID_ENABLE_PAIRING;
    buf[2] = 2; // 2 payload bytes: bool + timeout
    buf[3] = bEnablePairing ? 1 : 0;
    buf[4] = bEnablePairing ? PAIRING_STATE_DURATION_SECONDS : 0;
    SetFeatureReport(dev, buf, 5);
}

//---------------------------------------------------------------------------
// Commit Steam Controller pairing
//---------------------------------------------------------------------------
static void CommitPairing(SDL_HIDAPI_Device *dev)
{
    unsigned char buf[65];
    SDL_memset(buf, 0, 65);
    buf[1] = ID_DONGLE_COMMIT_DEVICE;
    SetFeatureReport(dev, buf, 2);
}

//---------------------------------------------------------------------------
// Close a Steam Controller
//---------------------------------------------------------------------------
static void CloseSteamController(SDL_HIDAPI_Device *dev)
{
    // Switch the Steam Controller back to lizard mode so it works with the OS
    unsigned char buf[65];
    int nSettings = 0;

    // Reset digital button mappings
    SDL_memset(buf, 0, 65);
    buf[1] = ID_SET_DEFAULT_DIGITAL_MAPPINGS;
    SetFeatureReport(dev, buf, 2);

    // Reset the default settings
    SDL_memset(buf, 0, 65);
    buf[1] = ID_LOAD_DEFAULT_SETTINGS;
    buf[2] = 0;
    SetFeatureReport(dev, buf, 3);

    // Reset mouse mode for lizard mode
    SDL_memset(buf, 0, 65);
    buf[1] = ID_SET_SETTINGS_VALUES;
    ADD_SETTING(SETTING_RIGHT_TRACKPAD_MODE, TRACKPAD_ABSOLUTE_MOUSE);
    buf[2] = (unsigned char)(nSettings * 3);
    SetFeatureReport(dev, buf, 3 + nSettings * 3);
}

//---------------------------------------------------------------------------
// Scale and clamp values to a range
//---------------------------------------------------------------------------
static float RemapValClamped(float val, float A, float B, float C, float D)
{
    if (A == B) {
        return (val - B) >= 0.0f ? D : C;
    } else {
        float cVal = (val - A) / (B - A);
        cVal = clamp(cVal, 0.0f, 1.0f);

        return C + (D - C) * cVal;
    }
}

//---------------------------------------------------------------------------
// Rotate the pad coordinates
//---------------------------------------------------------------------------
static void RotatePad(int *pX, int *pY, float flAngleInRad)
{
    int origX = *pX, origY = *pY;

    *pX = (int)(SDL_cosf(flAngleInRad) * origX - SDL_sinf(flAngleInRad) * origY);
    *pY = (int)(SDL_sinf(flAngleInRad) * origX + SDL_cosf(flAngleInRad) * origY);
}

//---------------------------------------------------------------------------
// Format the first part of the state packet
//---------------------------------------------------------------------------
static void FormatStatePacketUntilGyro(SteamControllerStateInternal_t *pState, ValveControllerStatePacket_t *pStatePacket)
{
    int nLeftPadX;
    int nLeftPadY;
    int nRightPadX;
    int nRightPadY;
    int nPadOffset;

    // 15 degrees in rad
    const float flRotationAngle = 0.261799f;

    SDL_memset(pState, 0, offsetof(SteamControllerStateInternal_t, sBatteryLevel));

    // pState->eControllerType = m_eControllerType;
    pState->eControllerType = 2; // k_eControllerType_SteamController;
    pState->unPacketNum = pStatePacket->unPacketNum;

    // We have a chunk of trigger data in the packet format here, so zero it out afterwards
    SDL_memcpy(&pState->ulButtons, &pStatePacket->ButtonTriggerData.ulButtons, 8);
    pState->ulButtons &= ~0xFFFF000000LL;

    // The firmware uses this bit to tell us what kind of data is packed into the left two axes
    if (pStatePacket->ButtonTriggerData.ulButtons & STEAM_LEFTPAD_FINGERDOWN_MASK) {
        // Finger-down bit not set; "left pad" is actually trackpad
        pState->sLeftPadX = pState->sPrevLeftPad[0] = pStatePacket->sLeftPadX;
        pState->sLeftPadY = pState->sPrevLeftPad[1] = pStatePacket->sLeftPadY;

        if (pStatePacket->ButtonTriggerData.ulButtons & STEAM_LEFTPAD_AND_JOYSTICK_MASK) {
            // The controller is interleaving both stick and pad data, both are active
            pState->sLeftStickX = pState->sPrevLeftStick[0];
            pState->sLeftStickY = pState->sPrevLeftStick[1];
        } else {
            // The stick is not active
            pState->sPrevLeftStick[0] = 0;
            pState->sPrevLeftStick[1] = 0;
        }
    } else {
        // Finger-down bit not set; "left pad" is actually joystick

        // XXX there's a firmware bug where sometimes padX is 0 and padY is a large number (actually the battery voltage)
        // If that happens skip this packet and report last frames stick
        /*
                if ( m_eControllerType == k_eControllerType_SteamControllerV2 && pStatePacket->sLeftPadY > 900 ) {
                    pState->sLeftStickX = pState->sPrevLeftStick[0];
                    pState->sLeftStickY = pState->sPrevLeftStick[1];
                } else
        */
        {
            pState->sPrevLeftStick[0] = pState->sLeftStickX = pStatePacket->sLeftPadX;
            pState->sPrevLeftStick[1] = pState->sLeftStickY = pStatePacket->sLeftPadY;
        }
        /*
                if (m_eControllerType == k_eControllerType_SteamControllerV2) {
                    UpdateV2JoystickCap(&state);
                }
        */

        if (pStatePacket->ButtonTriggerData.ulButtons & STEAM_LEFTPAD_AND_JOYSTICK_MASK) {
            // The controller is interleaving both stick and pad data, both are active
            pState->sLeftPadX = pState->sPrevLeftPad[0];
            pState->sLeftPadY = pState->sPrevLeftPad[1];
        } else {
            // The trackpad is not active
            pState->sPrevLeftPad[0] = 0;
            pState->sPrevLeftPad[1] = 0;

            // Old controllers send trackpad click for joystick button when trackpad is not active
            if (pState->ulButtons & STEAM_BUTTON_LEFTPAD_CLICKED_MASK) {
                pState->ulButtons &= ~STEAM_BUTTON_LEFTPAD_CLICKED_MASK;
                pState->ulButtons |= STEAM_JOYSTICK_BUTTON_MASK;
            }
        }
    }

    // Fingerdown bit indicates if the packed left axis data was joystick or pad,
    // but if we are interleaving both, the left finger is definitely on the pad.
    if (pStatePacket->ButtonTriggerData.ulButtons & STEAM_LEFTPAD_AND_JOYSTICK_MASK) {
        pState->ulButtons |= STEAM_LEFTPAD_FINGERDOWN_MASK;
    }

    pState->sRightPadX = pStatePacket->sRightPadX;
    pState->sRightPadY = pStatePacket->sRightPadY;

    nLeftPadX = pState->sLeftPadX;
    nLeftPadY = pState->sLeftPadY;
    nRightPadX = pState->sRightPadX;
    nRightPadY = pState->sRightPadY;

    RotatePad(&nLeftPadX, &nLeftPadY, -flRotationAngle);
    RotatePad(&nRightPadX, &nRightPadY, flRotationAngle);

    if (pState->ulButtons & STEAM_LEFTPAD_FINGERDOWN_MASK) {
        nPadOffset = 1000;
    } else {
        nPadOffset = 0;
    }

    pState->sLeftPadX = (short)clamp(nLeftPadX + nPadOffset, SDL_MIN_SINT16, SDL_MAX_SINT16);
    pState->sLeftPadY = (short)clamp(nLeftPadY + nPadOffset, SDL_MIN_SINT16, SDL_MAX_SINT16);

    nPadOffset = 0;
    if (pState->ulButtons & STEAM_RIGHTPAD_FINGERDOWN_MASK) {
        nPadOffset = 1000;
    } else {
        nPadOffset = 0;
    }

    pState->sRightPadX = (short)clamp(nRightPadX + nPadOffset, SDL_MIN_SINT16, SDL_MAX_SINT16);
    pState->sRightPadY = (short)clamp(nRightPadY + nPadOffset, SDL_MIN_SINT16, SDL_MAX_SINT16);

    pState->sTriggerL = (unsigned short)RemapValClamped((float)((pStatePacket->ButtonTriggerData.Triggers.nLeft << 7) | pStatePacket->ButtonTriggerData.Triggers.nLeft), 0, STEAMCONTROLLER_TRIGGER_MAX_ANALOG, 0, SDL_MAX_SINT16);
    pState->sTriggerR = (unsigned short)RemapValClamped((float)((pStatePacket->ButtonTriggerData.Triggers.nRight << 7) | pStatePacket->ButtonTriggerData.Triggers.nRight), 0, STEAMCONTROLLER_TRIGGER_MAX_ANALOG, 0, SDL_MAX_SINT16);
}

//---------------------------------------------------------------------------
// Update Steam Controller state from a BLE data packet, returns true if it parsed data
//---------------------------------------------------------------------------
static bool UpdateBLESteamControllerState(const uint8_t *pData, int nDataSize, SteamControllerStateInternal_t *pState)
{
    int nLeftPadX;
    int nLeftPadY;
    int nRightPadX;
    int nRightPadY;
    int nPadOffset;
    uint32_t ucOptionDataMask;

    // 15 degrees in rad
    const float flRotationAngle = 0.261799f;

    pState->unPacketNum++;
    ucOptionDataMask = (*pData++ & 0xF0);
    ucOptionDataMask |= (uint32_t)(*pData++) << 8;
    if (ucOptionDataMask & k_EBLEButtonChunk1) {
        SDL_memcpy(&pState->ulButtons, pData, 3);
        pData += 3;
    }
    if (ucOptionDataMask & k_EBLEButtonChunk2) {
        // The middle 2 bytes of the button bits over the wire are triggers when over the wire and non-SC buttons in the internal controller state packet
        pState->sTriggerL = (unsigned short)RemapValClamped((float)((pData[0] << 7) | pData[0]), 0, STEAMCONTROLLER_TRIGGER_MAX_ANALOG, 0, SDL_MAX_SINT16);
        pState->sTriggerR = (unsigned short)RemapValClamped((float)((pData[1] << 7) | pData[1]), 0, STEAMCONTROLLER_TRIGGER_MAX_ANALOG, 0, SDL_MAX_SINT16);
        pData += 2;
    }
    if (ucOptionDataMask & k_EBLEButtonChunk3) {
        uint8_t *pButtonByte = (uint8_t *)&pState->ulButtons;
        pButtonByte[5] = *pData++;
        pButtonByte[6] = *pData++;
        pButtonByte[7] = *pData++;
    }
    if (ucOptionDataMask & k_EBLELeftJoystickChunk) {
        // This doesn't handle any of the special headcrab stuff for raw joystick which is OK for now since that FW doesn't support
        // this protocol yet either
        int nLength = sizeof(pState->sLeftStickX) + sizeof(pState->sLeftStickY);
        SDL_memcpy(&pState->sLeftStickX, pData, nLength);
        pData += nLength;
    }
    if (ucOptionDataMask & k_EBLELeftTrackpadChunk) {
        int nLength = sizeof(pState->sLeftPadX) + sizeof(pState->sLeftPadY);
        SDL_memcpy(&pState->sLeftPadX, pData, nLength);
        if (pState->ulButtons & STEAM_LEFTPAD_FINGERDOWN_MASK) {
            nPadOffset = 1000;
        } else {
            nPadOffset = 0;
        }

        nLeftPadX = pState->sLeftPadX;
        nLeftPadY = pState->sLeftPadY;
        RotatePad(&nLeftPadX, &nLeftPadY, -flRotationAngle);
        pState->sLeftPadX = (short)clamp(nLeftPadX + nPadOffset, SDL_MIN_SINT16, SDL_MAX_SINT16);
        pState->sLeftPadY = (short)clamp(nLeftPadY + nPadOffset, SDL_MIN_SINT16, SDL_MAX_SINT16);
        pData += nLength;
    }
    if (ucOptionDataMask & k_EBLERightTrackpadChunk) {
        int nLength = sizeof(pState->sRightPadX) + sizeof(pState->sRightPadY);

        SDL_memcpy(&pState->sRightPadX, pData, nLength);

        if (pState->ulButtons & STEAM_RIGHTPAD_FINGERDOWN_MASK) {
            nPadOffset = 1000;
        } else {
            nPadOffset = 0;
        }

        nRightPadX = pState->sRightPadX;
        nRightPadY = pState->sRightPadY;
        RotatePad(&nRightPadX, &nRightPadY, flRotationAngle);
        pState->sRightPadX = (short)clamp(nRightPadX + nPadOffset, SDL_MIN_SINT16, SDL_MAX_SINT16);
        pState->sRightPadY = (short)clamp(nRightPadY + nPadOffset, SDL_MIN_SINT16, SDL_MAX_SINT16);
        pData += nLength;
    }
    if (ucOptionDataMask & k_EBLEIMUAccelChunk) {
        int nLength = sizeof(pState->sAccelX) + sizeof(pState->sAccelY) + sizeof(pState->sAccelZ);
        SDL_memcpy(&pState->sAccelX, pData, nLength);
        pData += nLength;
    }
    if (ucOptionDataMask & k_EBLEIMUGyroChunk) {
        int nLength = sizeof(pState->sAccelX) + sizeof(pState->sAccelY) + sizeof(pState->sAccelZ);
        SDL_memcpy(&pState->sGyroX, pData, nLength);
        pData += nLength;
    }
    if (ucOptionDataMask & k_EBLEIMUQuatChunk) {
        int nLength = sizeof(pState->sGyroQuatW) + sizeof(pState->sGyroQuatX) + sizeof(pState->sGyroQuatY) + sizeof(pState->sGyroQuatZ);
        SDL_memcpy(&pState->sGyroQuatW, pData, nLength);
        pData += nLength;
    }
    return true;
}

//---------------------------------------------------------------------------
// Update Steam Controller state from a data packet, returns true if it parsed data
//---------------------------------------------------------------------------
static bool UpdateSteamControllerState(const uint8_t *pData, int nDataSize, SteamControllerStateInternal_t *pState)
{
    ValveInReport_t *pInReport = (ValveInReport_t *)pData;

    if (pInReport->header.unReportVersion != k_ValveInReportMsgVersion) {
        if ((pData[0] & 0x0F) == k_EBLEReportState) {
            return UpdateBLESteamControllerState(pData, nDataSize, pState);
        }
        return false;
    }

    if ((pInReport->header.ucType != ID_CONTROLLER_STATE) &&
        (pInReport->header.ucType != ID_CONTROLLER_BLE_STATE)) {
        return false;
    }

    if (pInReport->header.ucType == ID_CONTROLLER_STATE) {
        ValveControllerStatePacket_t *pStatePacket = &pInReport->payload.controllerState;

        // No new data to process; indicate that we received a state packet, but otherwise do nothing.
        if (pState->unPacketNum == pStatePacket->unPacketNum) {
            return true;
        }

        FormatStatePacketUntilGyro(pState, pStatePacket);

        pState->sAccelX = pStatePacket->sAccelX;
        pState->sAccelY = pStatePacket->sAccelY;
        pState->sAccelZ = pStatePacket->sAccelZ;

        pState->sGyroQuatW = pStatePacket->sGyroQuatW;
        pState->sGyroQuatX = pStatePacket->sGyroQuatX;
        pState->sGyroQuatY = pStatePacket->sGyroQuatY;
        pState->sGyroQuatZ = pStatePacket->sGyroQuatZ;

        pState->sGyroX = pStatePacket->sGyroX;
        pState->sGyroY = pStatePacket->sGyroY;
        pState->sGyroZ = pStatePacket->sGyroZ;

    } else if (pInReport->header.ucType == ID_CONTROLLER_BLE_STATE) {
        ValveControllerBLEStatePacket_t *pBLEStatePacket = &pInReport->payload.controllerBLEState;
        ValveControllerStatePacket_t *pStatePacket = &pInReport->payload.controllerState;

        // No new data to process; indicate that we received a state packet, but otherwise do nothing.
        if (pState->unPacketNum == pStatePacket->unPacketNum) {
            return true;
        }

        FormatStatePacketUntilGyro(pState, pStatePacket);

        switch (pBLEStatePacket->ucGyroDataType) {
        case 1:
            pState->sGyroQuatW = ((float)pBLEStatePacket->sGyro[0]);
            pState->sGyroQuatX = ((float)pBLEStatePacket->sGyro[1]);
            pState->sGyroQuatY = ((float)pBLEStatePacket->sGyro[2]);
            pState->sGyroQuatZ = ((float)pBLEStatePacket->sGyro[3]);
            break;

        case 2:
            pState->sAccelX = pBLEStatePacket->sGyro[0];
            pState->sAccelY = pBLEStatePacket->sGyro[1];
            pState->sAccelZ = pBLEStatePacket->sGyro[2];
            break;

        case 3:
            pState->sGyroX = pBLEStatePacket->sGyro[0];
            pState->sGyroY = pBLEStatePacket->sGyro[1];
            pState->sGyroZ = pBLEStatePacket->sGyro[2];
            break;

        default:
            break;
        }
    }

    return true;
}

/*****************************************************************************************************/

typedef struct
{
    SDL_HIDAPI_Device *device;
    bool connected;
    bool report_sensors;
    uint32_t update_rate_in_us;
    Uint64 sensor_timestamp;
    Uint64 pairing_time;

    SteamControllerPacketAssembler m_assembler;
    SteamControllerStateInternal_t m_state;
    SteamControllerStateInternal_t m_last_state;
} SDL_DriverSteam_Context;

static bool IsDongle(Uint16 product_id)
{
    return (product_id == USB_PRODUCT_VALVE_STEAM_CONTROLLER_DONGLE);
}

static void HIDAPI_DriverSteam_RegisterHints(SDL_HintCallback callback, void *userdata)
{
    SDL_AddHintCallback(SDL_HINT_JOYSTICK_HIDAPI_STEAM, callback, userdata);
}

static void HIDAPI_DriverSteam_UnregisterHints(SDL_HintCallback callback, void *userdata)
{
    SDL_RemoveHintCallback(SDL_HINT_JOYSTICK_HIDAPI_STEAM, callback, userdata);
}

static bool HIDAPI_DriverSteam_IsEnabled(void)
{
    return SDL_GetHintBoolean(SDL_HINT_JOYSTICK_HIDAPI_STEAM, SDL_HINT_JOYSTICK_HIDAPI_STEAM_DEFAULT);
}

static bool HIDAPI_DriverSteam_IsSupportedDevice(SDL_HIDAPI_Device *device, const char *name, SDL_GamepadType type, Uint16 vendor_id, Uint16 product_id, Uint16 version, int interface_number, int interface_class, int interface_subclass, int interface_protocol)
{
    if (!SDL_IsJoystickSteamController(vendor_id, product_id)) {
        return false;
    }

    if (!device) {
        // Might be supported by this driver, enumerate and find out
        return true;
    }

    if (device->is_bluetooth) {
        return true;
    }

    if (IsDongle(product_id)) {
        if (interface_number >= 1 && interface_number <= 4) {
            // This is one of the wireless controller interfaces
            return true;
        }
    } else {
        if (interface_number == 2) {
            // This is the controller interface (not mouse or keyboard)
            return true;
        }
    }
    return false;
}

static void HIDAPI_DriverSteam_SetPairingState(SDL_DriverSteam_Context *ctx, bool enabled)
{
    // Only have one dongle in pairing mode at a time
    static SDL_DriverSteam_Context *s_PairingContext = NULL;

    if (enabled && s_PairingContext != NULL) {
        return;
    }

    if (!enabled && s_PairingContext != ctx) {
        return;
    }

    if (ctx->connected) {
        return;
    }

    SetPairingState(ctx->device, enabled);

    if (enabled) {
        ctx->pairing_time = SDL_GetTicks();
        s_PairingContext = ctx;
    } else {
        ctx->pairing_time = 0;
        s_PairingContext = NULL;
    }
}

static void HIDAPI_DriverSteam_RenewPairingState(SDL_DriverSteam_Context *ctx)
{
    Uint64 now = SDL_GetTicks();

    if (now >= ctx->pairing_time + PAIRING_STATE_DURATION_SECONDS * 1000) {
        SetPairingState(ctx->device, true);
        ctx->pairing_time = now;
    }
}

static void HIDAPI_DriverSteam_CommitPairing(SDL_DriverSteam_Context *ctx)
{
    CommitPairing(ctx->device);
}

static void SDLCALL SDL_PairingEnabledHintChanged(void *userdata, const char *name, const char *oldValue, const char *hint)
{
    SDL_DriverSteam_Context *ctx = (SDL_DriverSteam_Context *)userdata;
    bool enabled = SDL_GetStringBoolean(hint, false);

    HIDAPI_DriverSteam_SetPairingState(ctx, enabled);
}

static bool HIDAPI_DriverSteam_InitDevice(SDL_HIDAPI_Device *device)
{
    SDL_DriverSteam_Context *ctx;

    ctx = (SDL_DriverSteam_Context *)SDL_calloc(1, sizeof(*ctx));
    if (!ctx) {
        return false;
    }
    ctx->device = device;
    device->context = ctx;

#ifdef SDL_PLATFORM_WIN32
    if (device->serial) {
        // We get a garbage serial number on Windows
        SDL_free(device->serial);
        device->serial = NULL;
    }
#endif // SDL_PLATFORM_WIN32

    HIDAPI_SetDeviceName(device, "Steam Controller");

    // If this is a wireless dongle, request a wireless state update
    if (IsDongle(device->product_id)) {
        unsigned char buf[65];
        int res;

        buf[0] = 0;
        buf[1] = ID_DONGLE_GET_WIRELESS_STATE;
        res = SetFeatureReport(device, buf, 2);
        if (res < 0) {
            return SDL_SetError("Failed to send ID_DONGLE_GET_WIRELESS_STATE request");
        }

        for (int attempt = 0; attempt < 5; ++attempt) {
            uint8_t data[128];

            res = ReadSteamController(device->dev, data, sizeof(data));
            if (res == 0) {
                SDL_Delay(1);
                continue;
            }
            if (res < 0) {
                break;
            }

#ifdef DEBUG_STEAM_PROTOCOL
            HIDAPI_DumpPacket("Initial dongle packet: size = %d", data, res);
#endif

            if (D0G_IS_WIRELESS_CONNECT(data, res)) {
                ctx->connected = true;
                break;
            } else if (D0G_IS_WIRELESS_DISCONNECT(data, res)) {
                ctx->connected = false;
                break;
            }
        }

        SDL_AddHintCallback(SDL_HINT_JOYSTICK_HIDAPI_STEAM_PAIRING_ENABLED,
                            SDL_PairingEnabledHintChanged, ctx);
    } else {
        // Wired and BLE controllers are always connected if HIDAPI can see them
        ctx->connected = true;
    }

    if (ctx->connected) {
        return HIDAPI_JoystickConnected(device, NULL);
    } else {
        // We will enumerate any attached controllers in UpdateDevice()
        return true;
    }
}

static int HIDAPI_DriverSteam_GetDevicePlayerIndex(SDL_HIDAPI_Device *device, SDL_JoystickID instance_id)
{
    return -1;
}

static void HIDAPI_DriverSteam_SetDevicePlayerIndex(SDL_HIDAPI_Device *device, SDL_JoystickID instance_id, int player_index)
{
}

static bool SetHomeLED(SDL_HIDAPI_Device *device, Uint8 value)
{
    unsigned char buf[65];
    int nSettings = 0;

    SDL_memset(buf, 0, 65);
    buf[1] = ID_SET_SETTINGS_VALUES;
    ADD_SETTING(SETTING_LED_USER_BRIGHTNESS, value);
    buf[2] = (unsigned char)(nSettings * 3);
    if (SetFeatureReport(device, buf, 3 + nSettings * 3) < 0) {
        return SDL_SetError("Couldn't write feature report");
    }
    return true;
}

static void SDLCALL SDL_HomeLEDHintChanged(void *userdata, const char *name, const char *oldValue, const char *hint)
{
    SDL_DriverSteam_Context *ctx = (SDL_DriverSteam_Context *)userdata;

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
        SetHomeLED(ctx->device, (Uint8)value);
    }
}

static bool HIDAPI_DriverSteam_OpenJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
    SDL_DriverSteam_Context *ctx = (SDL_DriverSteam_Context *)device->context;
    float update_rate_in_hz = 0.0f;

    SDL_AssertJoysticksLocked();

    ctx->report_sensors = false;
    SDL_zero(ctx->m_assembler);
    SDL_zero(ctx->m_state);
    SDL_zero(ctx->m_last_state);

    if (!ResetSteamController(device, false, &ctx->update_rate_in_us)) {
        SDL_SetError("Couldn't reset controller");
        return false;
    }
    if (ctx->update_rate_in_us > 0) {
        update_rate_in_hz = 1000000.0f / ctx->update_rate_in_us;
    }

    InitializeSteamControllerPacketAssembler(&ctx->m_assembler, device->is_bluetooth);

    // Initialize the joystick capabilities
    joystick->nbuttons = SDL_GAMEPAD_NUM_STEAM_BUTTONS;
    joystick->naxes = SDL_GAMEPAD_AXIS_COUNT;
    joystick->nhats = 1;

    if (IsDongle(device->product_id)) {
        joystick->connection_state = SDL_JOYSTICK_CONNECTION_WIRELESS;
    }

    SDL_PrivateJoystickAddSensor(joystick, SDL_SENSOR_GYRO, update_rate_in_hz);
    SDL_PrivateJoystickAddSensor(joystick, SDL_SENSOR_ACCEL, update_rate_in_hz);

    SDL_AddHintCallback(SDL_HINT_JOYSTICK_HIDAPI_STEAM_HOME_LED,
                        SDL_HomeLEDHintChanged, ctx);

    return true;
}

static bool HIDAPI_DriverSteam_RumbleJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint16 low_frequency_rumble, Uint16 high_frequency_rumble)
{
    // You should use the full Steam Input API for rumble support
    return SDL_Unsupported();
}

static bool HIDAPI_DriverSteam_RumbleJoystickTriggers(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint16 left_rumble, Uint16 right_rumble)
{
    return SDL_Unsupported();
}

static Uint32 HIDAPI_DriverSteam_GetJoystickCapabilities(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
    // You should use the full Steam Input API for extended capabilities
    return 0;
}

static bool HIDAPI_DriverSteam_SetJoystickLED(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint8 red, Uint8 green, Uint8 blue)
{
    // You should use the full Steam Input API for LED support
    return SDL_Unsupported();
}

static bool HIDAPI_DriverSteam_SendJoystickEffect(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, const void *data, int size)
{
    if (size == 65) {
        if (SetFeatureReport(device, data, size) < 0) {
            return SDL_SetError("Couldn't write feature report");
        }
        return true;
    }
    return SDL_Unsupported();
}

static bool HIDAPI_DriverSteam_SetSensorsEnabled(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, bool enabled)
{
    SDL_DriverSteam_Context *ctx = (SDL_DriverSteam_Context *)device->context;
    unsigned char buf[65];
    int nSettings = 0;

    SDL_memset(buf, 0, 65);
    buf[1] = ID_SET_SETTINGS_VALUES;
    if (enabled) {
        ADD_SETTING(SETTING_IMU_MODE, SETTING_GYRO_MODE_SEND_RAW_ACCEL | SETTING_GYRO_MODE_SEND_RAW_GYRO);
    } else {
        ADD_SETTING(SETTING_IMU_MODE, SETTING_GYRO_MODE_OFF);
    }
    buf[2] = (unsigned char)(nSettings * 3);
    if (SetFeatureReport(device, buf, 3 + nSettings * 3) < 0) {
        return SDL_SetError("Couldn't write feature report");
    }

    ctx->report_sensors = enabled;

    return true;
}

static bool ControllerConnected(SDL_HIDAPI_Device *device, SDL_Joystick **joystick)
{
    SDL_DriverSteam_Context *ctx = (SDL_DriverSteam_Context *)device->context;

    if (!HIDAPI_JoystickConnected(device, NULL)) {
        return false;
    }

    // We'll automatically accept this controller if we're in pairing mode
    HIDAPI_DriverSteam_CommitPairing(ctx);

    *joystick = SDL_GetJoystickFromID(device->joysticks[0]);
    ctx->connected = true;
    return true;
}

static void ControllerDisconnected(SDL_HIDAPI_Device *device, SDL_Joystick **joystick)
{
    SDL_DriverSteam_Context *ctx = (SDL_DriverSteam_Context *)device->context;

    if (device->joysticks) {
        HIDAPI_JoystickDisconnected(device, device->joysticks[0]);
    }
    ctx->connected = false;
    *joystick = NULL;
}

static bool HIDAPI_DriverSteam_UpdateDevice(SDL_HIDAPI_Device *device)
{
    SDL_DriverSteam_Context *ctx = (SDL_DriverSteam_Context *)device->context;
    SDL_Joystick *joystick = NULL;

    if (device->num_joysticks > 0) {
        joystick = SDL_GetJoystickFromID(device->joysticks[0]);
    }

    if (ctx->pairing_time) {
        HIDAPI_DriverSteam_RenewPairingState(ctx);
    }

    for (;;) {
        uint8_t data[128];
        int r, nPacketLength;
        const Uint8 *pPacket;

        r = ReadSteamController(device->dev, data, sizeof(data));
        if (r == 0) {
            break;
        }
        if (r < 0) {
            // Failed to read from controller
            ControllerDisconnected(device, &joystick);
            return false;
        }

#ifdef DEBUG_STEAM_PROTOCOL
        HIDAPI_DumpPacket("Steam Controller packet: size = %d", data, r);
#endif

        nPacketLength = WriteSegmentToSteamControllerPacketAssembler(&ctx->m_assembler, data, r);
        pPacket = ctx->m_assembler.uBuffer;

        if (nPacketLength > 0 && UpdateSteamControllerState(pPacket, nPacketLength, &ctx->m_state)) {
            Uint64 timestamp = SDL_GetTicksNS();

            if (!ctx->connected) {
                // Maybe we missed a wireless status packet?
                ControllerConnected(device, &joystick);
            }

            if (!joystick) {
                continue;
            }

            if (ctx->m_state.ulButtons != ctx->m_last_state.ulButtons) {
                Uint8 hat = 0;

                SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SOUTH,
                                          ((ctx->m_state.ulButtons & STEAM_BUTTON_SOUTH_MASK) != 0));

                SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_EAST,
                                          ((ctx->m_state.ulButtons & STEAM_BUTTON_EAST_MASK) != 0));

                SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_WEST,
                                          ((ctx->m_state.ulButtons & STEAM_BUTTON_WEST_MASK) != 0));

                SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_NORTH,
                                          ((ctx->m_state.ulButtons & STEAM_BUTTON_NORTH_MASK) != 0));

                SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_SHOULDER,
                                          ((ctx->m_state.ulButtons & STEAM_LEFT_BUMPER_MASK) != 0));

                SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER,
                                          ((ctx->m_state.ulButtons & STEAM_RIGHT_BUMPER_MASK) != 0));

                SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_BACK,
                                          ((ctx->m_state.ulButtons & STEAM_BUTTON_MENU_MASK) != 0));

                SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_START,
                                          ((ctx->m_state.ulButtons & STEAM_BUTTON_ESCAPE_MASK) != 0));

                SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_GUIDE,
                                          ((ctx->m_state.ulButtons & STEAM_BUTTON_STEAM_MASK) != 0));

                SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_STICK,
                                          ((ctx->m_state.ulButtons & STEAM_JOYSTICK_BUTTON_MASK) != 0));
                SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_STEAM_LEFT_PADDLE,
                                          ((ctx->m_state.ulButtons & STEAM_BUTTON_BACK_LEFT_MASK) != 0));
                SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_STEAM_RIGHT_PADDLE,
                                          ((ctx->m_state.ulButtons & STEAM_BUTTON_BACK_RIGHT_MASK) != 0));

                SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_RIGHT_STICK,
                                          ((ctx->m_state.ulButtons & STEAM_BUTTON_RIGHTPAD_CLICKED_MASK) != 0));

                if (ctx->m_state.ulButtons & STEAM_DPAD_UP_MASK) {
                    hat |= SDL_HAT_UP;
                }
                if (ctx->m_state.ulButtons & STEAM_DPAD_DOWN_MASK) {
                    hat |= SDL_HAT_DOWN;
                }
                if (ctx->m_state.ulButtons & STEAM_DPAD_LEFT_MASK) {
                    hat |= SDL_HAT_LEFT;
                }
                if (ctx->m_state.ulButtons & STEAM_DPAD_RIGHT_MASK) {
                    hat |= SDL_HAT_RIGHT;
                }
                SDL_SendJoystickHat(timestamp, joystick, 0, hat);
            }

            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFT_TRIGGER, (int)ctx->m_state.sTriggerL * 2 - 32768);
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER, (int)ctx->m_state.sTriggerR * 2 - 32768);

            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTX, ctx->m_state.sLeftStickX);
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTY, ~ctx->m_state.sLeftStickY);
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTX, ctx->m_state.sRightPadX);
            SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTY, ~ctx->m_state.sRightPadY);

            if (ctx->report_sensors) {
                float values[3];

                ctx->sensor_timestamp += SDL_US_TO_NS(ctx->update_rate_in_us);

                values[0] = (ctx->m_state.sGyroX / 32768.0f) * (2000.0f * (SDL_PI_F / 180.0f));
                values[1] = (ctx->m_state.sGyroZ / 32768.0f) * (2000.0f * (SDL_PI_F / 180.0f));
                values[2] = (ctx->m_state.sGyroY / 32768.0f) * (2000.0f * (SDL_PI_F / 180.0f));
                SDL_SendJoystickSensor(timestamp, joystick, SDL_SENSOR_GYRO, ctx->sensor_timestamp, values, 3);

                values[0] = (ctx->m_state.sAccelX / 32768.0f) * 2.0f * SDL_STANDARD_GRAVITY;
                values[1] = (ctx->m_state.sAccelZ / 32768.0f) * 2.0f * SDL_STANDARD_GRAVITY;
                values[2] = (-ctx->m_state.sAccelY / 32768.0f) * 2.0f * SDL_STANDARD_GRAVITY;
                SDL_SendJoystickSensor(timestamp, joystick, SDL_SENSOR_ACCEL, ctx->sensor_timestamp, values, 3);
            }

            ctx->m_last_state = ctx->m_state;
        } else if (!ctx->connected && D0G_IS_WIRELESS_CONNECT(pPacket, nPacketLength)) {
            ControllerConnected(device, &joystick);
        } else if (ctx->connected && D0G_IS_WIRELESS_DISCONNECT(pPacket, nPacketLength)) {
            ControllerDisconnected(device, &joystick);
        }
    }
    return true;
}

static void HIDAPI_DriverSteam_CloseJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
    SDL_DriverSteam_Context *ctx = (SDL_DriverSteam_Context *)device->context;

    SDL_RemoveHintCallback(SDL_HINT_JOYSTICK_HIDAPI_STEAM_HOME_LED,
                           SDL_HomeLEDHintChanged, ctx);

    CloseSteamController(device);
}

static void HIDAPI_DriverSteam_FreeDevice(SDL_HIDAPI_Device *device)
{
    SDL_DriverSteam_Context *ctx = (SDL_DriverSteam_Context *)device->context;

    if (IsDongle(device->product_id)) {
        SDL_RemoveHintCallback(SDL_HINT_JOYSTICK_HIDAPI_STEAM_PAIRING_ENABLED,
                               SDL_PairingEnabledHintChanged, ctx);

        HIDAPI_DriverSteam_SetPairingState(ctx, false);
    }
}

SDL_HIDAPI_DeviceDriver SDL_HIDAPI_DriverSteam = {
    SDL_HINT_JOYSTICK_HIDAPI_STEAM,
    true,
    HIDAPI_DriverSteam_RegisterHints,
    HIDAPI_DriverSteam_UnregisterHints,
    HIDAPI_DriverSteam_IsEnabled,
    HIDAPI_DriverSteam_IsSupportedDevice,
    HIDAPI_DriverSteam_InitDevice,
    HIDAPI_DriverSteam_GetDevicePlayerIndex,
    HIDAPI_DriverSteam_SetDevicePlayerIndex,
    HIDAPI_DriverSteam_UpdateDevice,
    HIDAPI_DriverSteam_OpenJoystick,
    HIDAPI_DriverSteam_RumbleJoystick,
    HIDAPI_DriverSteam_RumbleJoystickTriggers,
    HIDAPI_DriverSteam_GetJoystickCapabilities,
    HIDAPI_DriverSteam_SetJoystickLED,
    HIDAPI_DriverSteam_SendJoystickEffect,
    HIDAPI_DriverSteam_SetSensorsEnabled,
    HIDAPI_DriverSteam_CloseJoystick,
    HIDAPI_DriverSteam_FreeDevice,
};

#endif // SDL_JOYSTICK_HIDAPI_STEAM

#endif // SDL_JOYSTICK_HIDAPI
