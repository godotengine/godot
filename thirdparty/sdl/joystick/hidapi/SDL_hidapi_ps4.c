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
/* This driver supports both simplified reports and the extended input reports enabled by Steam.
   Code and logic contributed by Valve Corporation under the SDL zlib license.
*/
#include "SDL_internal.h"

#ifdef SDL_JOYSTICK_HIDAPI

#include "../../SDL_hints_c.h"
#include "../SDL_sysjoystick.h"
#include "SDL_hidapijoystick_c.h"
#include "SDL_hidapi_rumble.h"

#ifdef SDL_JOYSTICK_HIDAPI_PS4

// Define this if you want to log all packets from the controller
#if 0
#define DEBUG_PS4_PROTOCOL
#endif

// Define this if you want to log calibration data
#if 0
#define DEBUG_PS4_CALIBRATION
#endif

#define BLUETOOTH_DISCONNECT_TIMEOUT_MS 500

#define LOAD16(A, B)       (Sint16)((Uint16)(A) | (((Uint16)(B)) << 8))
#define LOAD32(A, B, C, D) ((((Uint32)(A)) << 0) |  \
                            (((Uint32)(B)) << 8) |  \
                            (((Uint32)(C)) << 16) | \
                            (((Uint32)(D)) << 24))

enum
{
    SDL_GAMEPAD_BUTTON_PS4_TOUCHPAD = 11
};

typedef enum
{
    k_EPS4ReportIdUsbState = 1,
    k_EPS4ReportIdUsbEffects = 5,
    k_EPS4ReportIdBluetoothState1 = 17,
    k_EPS4ReportIdBluetoothState2 = 18,
    k_EPS4ReportIdBluetoothState3 = 19,
    k_EPS4ReportIdBluetoothState4 = 20,
    k_EPS4ReportIdBluetoothState5 = 21,
    k_EPS4ReportIdBluetoothState6 = 22,
    k_EPS4ReportIdBluetoothState7 = 23,
    k_EPS4ReportIdBluetoothState8 = 24,
    k_EPS4ReportIdBluetoothState9 = 25,
    k_EPS4ReportIdBluetoothEffects = 17,
    k_EPS4ReportIdDisconnectMessage = 226,
} EPS4ReportId;

typedef enum
{
    k_ePS4FeatureReportIdGyroCalibration_USB = 0x02,
    k_ePS4FeatureReportIdCapabilities = 0x03,
    k_ePS4FeatureReportIdGyroCalibration_BT = 0x05,
    k_ePS4FeatureReportIdSerialNumber = 0x12,
} EPS4FeatureReportID;

typedef struct
{
    Uint8 ucLeftJoystickX;
    Uint8 ucLeftJoystickY;
    Uint8 ucRightJoystickX;
    Uint8 ucRightJoystickY;
    Uint8 rgucButtonsHatAndCounter[3];
    Uint8 ucTriggerLeft;
    Uint8 ucTriggerRight;
    Uint8 rgucTimestamp[2];
    Uint8 _rgucPad0[1];
    Uint8 rgucGyroX[2];
    Uint8 rgucGyroY[2];
    Uint8 rgucGyroZ[2];
    Uint8 rgucAccelX[2];
    Uint8 rgucAccelY[2];
    Uint8 rgucAccelZ[2];
    Uint8 _rgucPad1[5];
    Uint8 ucBatteryLevel;
    Uint8 _rgucPad2[4];
    Uint8 ucTouchpadCounter1;
    Uint8 rgucTouchpadData1[3];
    Uint8 ucTouchpadCounter2;
    Uint8 rgucTouchpadData2[3];
} PS4StatePacket_t;

typedef struct
{
    Uint8 ucRumbleRight;
    Uint8 ucRumbleLeft;
    Uint8 ucLedRed;
    Uint8 ucLedGreen;
    Uint8 ucLedBlue;
    Uint8 ucLedDelayOn;
    Uint8 ucLedDelayOff;
    Uint8 _rgucPad0[8];
    Uint8 ucVolumeLeft;
    Uint8 ucVolumeRight;
    Uint8 ucVolumeMic;
    Uint8 ucVolumeSpeaker;
} DS4EffectsState_t;

typedef struct
{
    Sint16 bias;
    float scale;
} IMUCalibrationData;

/* Rumble hint mode:
 * "0": enhanced features are never used
 * "1": enhanced features are always used
 * "auto": enhanced features are advertised to the application, but SDL doesn't touch the controller state unless the application explicitly requests it.
 */
typedef enum
{
    PS4_ENHANCED_REPORT_HINT_OFF,
    PS4_ENHANCED_REPORT_HINT_ON,
    PS4_ENHANCED_REPORT_HINT_AUTO
} HIDAPI_PS4_EnhancedReportHint;

typedef struct
{
    SDL_HIDAPI_Device *device;
    SDL_Joystick *joystick;
    bool is_dongle;
    bool is_nacon_dongle;
    bool official_controller;
    bool sensors_supported;
    bool lightbar_supported;
    bool vibration_supported;
    bool touchpad_supported;
    bool effects_supported;
    HIDAPI_PS4_EnhancedReportHint enhanced_report_hint;
    bool enhanced_reports;
    bool enhanced_mode;
    bool enhanced_mode_available;
    Uint8 report_interval;
    bool report_sensors;
    bool report_touchpad;
    bool report_battery;
    bool hardware_calibration;
    IMUCalibrationData calibration[6];
    Uint64 last_packet;
    int player_index;
    Uint8 rumble_left;
    Uint8 rumble_right;
    bool color_set;
    Uint8 led_red;
    Uint8 led_green;
    Uint8 led_blue;
    Uint16 gyro_numerator;
    Uint16 gyro_denominator;
    Uint16 accel_numerator;
    Uint16 accel_denominator;
    Uint64 sensor_ticks;
    Uint16 last_tick;
    Uint16 valid_crc_packets; // wrapping counter
    PS4StatePacket_t last_state;
} SDL_DriverPS4_Context;

static bool HIDAPI_DriverPS4_InternalSendJoystickEffect(SDL_DriverPS4_Context *ctx, const void *effect, int size, bool application_usage);

static void HIDAPI_DriverPS4_RegisterHints(SDL_HintCallback callback, void *userdata)
{
    SDL_AddHintCallback(SDL_HINT_JOYSTICK_HIDAPI_PS4, callback, userdata);
}

static void HIDAPI_DriverPS4_UnregisterHints(SDL_HintCallback callback, void *userdata)
{
    SDL_RemoveHintCallback(SDL_HINT_JOYSTICK_HIDAPI_PS4, callback, userdata);
}

static bool HIDAPI_DriverPS4_IsEnabled(void)
{
    return SDL_GetHintBoolean(SDL_HINT_JOYSTICK_HIDAPI_PS4, SDL_GetHintBoolean(SDL_HINT_JOYSTICK_HIDAPI, SDL_HIDAPI_DEFAULT));
}

static int ReadFeatureReport(SDL_hid_device *dev, Uint8 report_id, Uint8 *report, size_t length)
{
    SDL_memset(report, 0, length);
    report[0] = report_id;
    return SDL_hid_get_feature_report(dev, report, length);
}

static bool HIDAPI_DriverPS4_IsSupportedDevice(SDL_HIDAPI_Device *device, const char *name, SDL_GamepadType type, Uint16 vendor_id, Uint16 product_id, Uint16 version, int interface_number, int interface_class, int interface_subclass, int interface_protocol)
{
    Uint8 data[USB_PACKET_LENGTH];
    int size;

    if (type == SDL_GAMEPAD_TYPE_PS4) {
        return true;
    }

    if (HIDAPI_SupportsPlaystationDetection(vendor_id, product_id)) {
        if (device && device->dev) {
            size = ReadFeatureReport(device->dev, k_ePS4FeatureReportIdCapabilities, data, sizeof(data));
            if (size == 48 && data[2] == 0x27) {
                // Supported third party controller
                return true;
            } else {
                return false;
            }
        } else {
            // Might be supported by this driver, enumerate and find out
            return true;
        }
    }

    return false;
}

static void SetLedsForPlayerIndex(DS4EffectsState_t *effects, int player_index)
{
    /* This list is the same as what hid-sony.c uses in the Linux kernel.
       The first 4 values correspond to what the PS4 assigns.
    */
    static const Uint8 colors[7][3] = {
        { 0x00, 0x00, 0x40 }, // Blue
        { 0x40, 0x00, 0x00 }, // Red
        { 0x00, 0x40, 0x00 }, // Green
        { 0x20, 0x00, 0x20 }, // Pink
        { 0x02, 0x01, 0x00 }, // Orange
        { 0x00, 0x01, 0x01 }, // Teal
        { 0x01, 0x01, 0x01 }  // White
    };

    if (player_index >= 0) {
        player_index %= SDL_arraysize(colors);
    } else {
        player_index = 0;
    }

    effects->ucLedRed = colors[player_index][0];
    effects->ucLedGreen = colors[player_index][1];
    effects->ucLedBlue = colors[player_index][2];
}

static bool ReadWiredSerial(SDL_HIDAPI_Device *device, char *serial, size_t serial_size)
{
    Uint8 data[USB_PACKET_LENGTH];
    int size;

    size = ReadFeatureReport(device->dev, k_ePS4FeatureReportIdSerialNumber, data, sizeof(data));
    if (size >= 7 && (data[1] || data[2] || data[3] || data[4] || data[5] || data[6])) {
        (void)SDL_snprintf(serial, serial_size, "%.2x-%.2x-%.2x-%.2x-%.2x-%.2x",
                           data[6], data[5], data[4], data[3], data[2], data[1]);
        return true;
    }
    return false;
}

static bool HIDAPI_DriverPS4_InitDevice(SDL_HIDAPI_Device *device)
{
    SDL_DriverPS4_Context *ctx;
    Uint8 data[USB_PACKET_LENGTH];
    int size;
    char serial[18];
    SDL_JoystickType joystick_type = SDL_JOYSTICK_TYPE_GAMEPAD;

    ctx = (SDL_DriverPS4_Context *)SDL_calloc(1, sizeof(*ctx));
    if (!ctx) {
        return false;
    }
    ctx->device = device;

    ctx->gyro_numerator = 1;
    ctx->gyro_denominator = 16;
    ctx->accel_numerator = 1;
    ctx->accel_denominator = 8192;

    device->context = ctx;

    if (device->serial && SDL_strlen(device->serial) == 12) {
        int i, j;

        j = -1;
        for (i = 0; i < 12; i += 2) {
            j += 1;
            SDL_memmove(&serial[j], &device->serial[i], 2);
            j += 2;
            serial[j] = '-';
        }
        serial[j] = '\0';
    } else {
        serial[0] = '\0';
    }

    // Check for type of connection
    ctx->is_dongle = (device->vendor_id == USB_VENDOR_SONY && device->product_id == USB_PRODUCT_SONY_DS4_DONGLE);
    if (ctx->is_dongle) {
        ReadWiredSerial(device, serial, sizeof(serial));
        ctx->enhanced_reports = true;
    } else if (device->vendor_id == USB_VENDOR_SONY && device->product_id == USB_PRODUCT_SONY_DS4_STRIKEPAD) {
        ctx->enhanced_reports = true;

    } else if (device->vendor_id == USB_VENDOR_SONY) {
        if (device->is_bluetooth) {
            // Read a report to see if we're in enhanced mode
            size = SDL_hid_read_timeout(device->dev, data, sizeof(data), 16);
#ifdef DEBUG_PS4_PROTOCOL
            if (size > 0) {
                HIDAPI_DumpPacket("PS4 first packet: size = %d", data, size);
            } else {
                SDL_Log("PS4 first packet: size = %d", size);
            }
#endif
            if (size > 0 &&
                data[0] >= k_EPS4ReportIdBluetoothState1 &&
                data[0] <= k_EPS4ReportIdBluetoothState9) {
                ctx->enhanced_reports = true;
            }
        } else {
            ReadWiredSerial(device, serial, sizeof(serial));
            ctx->enhanced_reports = true;
        }
    } else {
        // Third party controllers appear to all be wired
        ctx->enhanced_reports = true;
    }

    if (device->vendor_id == USB_VENDOR_SONY) {
        ctx->official_controller = true;
        ctx->sensors_supported = true;
        ctx->lightbar_supported = true;
        ctx->vibration_supported = true;
        ctx->touchpad_supported = true;
    } else {
        // Third party controller capability request
        size = ReadFeatureReport(device->dev, k_ePS4FeatureReportIdCapabilities, data, sizeof(data));
        // Get the device capabilities
        if (size == 48 && data[2] == 0x27) {
            Uint8 capabilities = data[4];
            Uint8 device_type = data[5];
            Uint16 gyro_numerator = LOAD16(data[10], data[11]);
            Uint16 gyro_denominator = LOAD16(data[12], data[13]);
            Uint16 accel_numerator = LOAD16(data[14], data[15]);
            Uint16 accel_denominator = LOAD16(data[16], data[17]);

#ifdef DEBUG_PS4_PROTOCOL
            HIDAPI_DumpPacket("PS4 capabilities: size = %d", data, size);
#endif
            if (capabilities & 0x02) {
                ctx->sensors_supported = true;
            }
            if (capabilities & 0x04) {
                ctx->lightbar_supported = true;
            }
            if (capabilities & 0x08) {
                ctx->vibration_supported = true;
            }
            if (capabilities & 0x40) {
                ctx->touchpad_supported = true;
            }

            switch (device_type) {
            case 0x00:
                joystick_type = SDL_JOYSTICK_TYPE_GAMEPAD;
                break;
            case 0x01:
                joystick_type = SDL_JOYSTICK_TYPE_GUITAR;
                break;
            case 0x02:
                joystick_type = SDL_JOYSTICK_TYPE_DRUM_KIT;
                break;
            case 0x04:
                joystick_type = SDL_JOYSTICK_TYPE_DANCE_PAD;
                break;
            case 0x06:
                joystick_type = SDL_JOYSTICK_TYPE_WHEEL;
                break;
            case 0x07:
                joystick_type = SDL_JOYSTICK_TYPE_ARCADE_STICK;
                break;
            case 0x08:
                joystick_type = SDL_JOYSTICK_TYPE_FLIGHT_STICK;
                break;
            default:
                joystick_type = SDL_JOYSTICK_TYPE_UNKNOWN;
                break;
            }

            if (gyro_numerator && gyro_denominator) {
                ctx->gyro_numerator = gyro_numerator;
                ctx->gyro_denominator = gyro_denominator;
            }
            if (accel_numerator && accel_denominator) {
                ctx->accel_numerator = accel_numerator;
                ctx->accel_denominator = accel_denominator;
            }
        } else if (device->vendor_id == USB_VENDOR_RAZER) {
            // The Razer Raiju doesn't respond to the detection protocol, but has a touchpad and vibration
            ctx->vibration_supported = true;
            ctx->touchpad_supported = true;
        }
    }
    ctx->effects_supported = (ctx->lightbar_supported || ctx->vibration_supported);

    if (device->vendor_id == USB_VENDOR_NACON_ALT &&
        device->product_id == USB_PRODUCT_NACON_REVOLUTION_5_PRO_PS4_WIRELESS) {
        ctx->is_nacon_dongle = true;
    }

    if (device->vendor_id == USB_VENDOR_PDP &&
        (device->product_id == USB_PRODUCT_VICTRIX_FS_PRO ||
         device->product_id == USB_PRODUCT_VICTRIX_FS_PRO_V2)) {
        /* The Victrix FS Pro V2 reports that it has lightbar support,
         * but it doesn't respond to the effects packet, and will hang
         * on reboot if we send it.
         */
        ctx->effects_supported = false;
    }

    device->joystick_type = joystick_type;
    device->type = SDL_GAMEPAD_TYPE_PS4;
    if (ctx->official_controller) {
        HIDAPI_SetDeviceName(device, "PS4 Controller");
    }
    HIDAPI_SetDeviceSerial(device, serial);

    // Prefer the USB device over the Bluetooth device
    if (device->is_bluetooth) {
        if (HIDAPI_HasConnectedUSBDevice(device->serial)) {
            return true;
        }
    } else {
        HIDAPI_DisconnectBluetoothDevice(device->serial);
    }
    if ((ctx->is_dongle || ctx->is_nacon_dongle) && serial[0] == '\0') {
        // Not yet connected
        return true;
    }
    return HIDAPI_JoystickConnected(device, NULL);
}

static int HIDAPI_DriverPS4_GetDevicePlayerIndex(SDL_HIDAPI_Device *device, SDL_JoystickID instance_id)
{
    return -1;
}

static bool HIDAPI_DriverPS4_LoadOfficialCalibrationData(SDL_HIDAPI_Device *device)
{
    SDL_DriverPS4_Context *ctx = (SDL_DriverPS4_Context *)device->context;
    int i, tries, size;
    bool have_data = false;
    Uint8 data[USB_PACKET_LENGTH];

    if (!ctx->official_controller) {
#ifdef DEBUG_PS4_CALIBRATION
        SDL_Log("Not an official controller, ignoring calibration");
#endif
        return false;
    }

    for (tries = 0; tries < 5; ++tries) {
        // For Bluetooth controllers, this report switches them into advanced report mode
        size = ReadFeatureReport(device->dev, k_ePS4FeatureReportIdGyroCalibration_USB, data, sizeof(data));
        if (size < 35) {
#ifdef DEBUG_PS4_CALIBRATION
            SDL_Log("Short read of calibration data: %d, ignoring calibration", size);
#endif
            return false;
        }

        if (device->is_bluetooth) {
            size = ReadFeatureReport(device->dev, k_ePS4FeatureReportIdGyroCalibration_BT, data, sizeof(data));
            if (size < 35) {
#ifdef DEBUG_PS4_CALIBRATION
                SDL_Log("Short read of calibration data: %d, ignoring calibration", size);
#endif
                return false;
            }
        }

        // In some cases this report returns all zeros. Usually immediately after connection with the PS4 Dongle
        for (i = 0; i < size; ++i) {
            if (data[i]) {
                have_data = true;
                break;
            }
        }
        if (have_data) {
            break;
        }

        SDL_Delay(2);
    }

    if (have_data) {
        Sint16 sGyroPitchBias, sGyroYawBias, sGyroRollBias;
        Sint16 sGyroPitchPlus, sGyroPitchMinus;
        Sint16 sGyroYawPlus, sGyroYawMinus;
        Sint16 sGyroRollPlus, sGyroRollMinus;
        Sint16 sGyroSpeedPlus, sGyroSpeedMinus;

        Sint16 sAccXPlus, sAccXMinus;
        Sint16 sAccYPlus, sAccYMinus;
        Sint16 sAccZPlus, sAccZMinus;

        float flNumerator;
        float flDenominator;
        Sint16 sRange2g;

#ifdef DEBUG_PS4_CALIBRATION
        HIDAPI_DumpPacket("PS4 calibration packet: size = %d", data, size);
#endif

        sGyroPitchBias = LOAD16(data[1], data[2]);
        sGyroYawBias = LOAD16(data[3], data[4]);
        sGyroRollBias = LOAD16(data[5], data[6]);

        if (device->is_bluetooth || ctx->is_dongle) {
            sGyroPitchPlus = LOAD16(data[7], data[8]);
            sGyroYawPlus = LOAD16(data[9], data[10]);
            sGyroRollPlus = LOAD16(data[11], data[12]);
            sGyroPitchMinus = LOAD16(data[13], data[14]);
            sGyroYawMinus = LOAD16(data[15], data[16]);
            sGyroRollMinus = LOAD16(data[17], data[18]);
        } else {
            sGyroPitchPlus = LOAD16(data[7], data[8]);
            sGyroPitchMinus = LOAD16(data[9], data[10]);
            sGyroYawPlus = LOAD16(data[11], data[12]);
            sGyroYawMinus = LOAD16(data[13], data[14]);
            sGyroRollPlus = LOAD16(data[15], data[16]);
            sGyroRollMinus = LOAD16(data[17], data[18]);
        }

        sGyroSpeedPlus = LOAD16(data[19], data[20]);
        sGyroSpeedMinus = LOAD16(data[21], data[22]);

        sAccXPlus = LOAD16(data[23], data[24]);
        sAccXMinus = LOAD16(data[25], data[26]);
        sAccYPlus = LOAD16(data[27], data[28]);
        sAccYMinus = LOAD16(data[29], data[30]);
        sAccZPlus = LOAD16(data[31], data[32]);
        sAccZMinus = LOAD16(data[33], data[34]);

        flNumerator = (float)(sGyroSpeedPlus + sGyroSpeedMinus) * ctx->gyro_denominator / ctx->gyro_numerator;
        flDenominator = (float)(SDL_abs(sGyroPitchPlus - sGyroPitchBias) + SDL_abs(sGyroPitchMinus - sGyroPitchBias));
        if (flDenominator != 0.0f) {
            ctx->calibration[0].bias = sGyroPitchBias;
            ctx->calibration[0].scale = flNumerator / flDenominator;
        }

        flDenominator = (float)(SDL_abs(sGyroYawPlus - sGyroYawBias) + SDL_abs(sGyroYawMinus - sGyroYawBias));
        if (flDenominator != 0.0f) {
            ctx->calibration[1].bias = sGyroYawBias;
            ctx->calibration[1].scale = flNumerator / flDenominator;
        }

        flDenominator = (float)(SDL_abs(sGyroRollPlus - sGyroRollBias) + SDL_abs(sGyroRollMinus - sGyroRollBias));
        if (flDenominator != 0.0f) {
            ctx->calibration[2].bias = sGyroRollBias;
            ctx->calibration[2].scale = flNumerator / flDenominator;
        }

        sRange2g = sAccXPlus - sAccXMinus;
        ctx->calibration[3].bias = sAccXPlus - sRange2g / 2;
        ctx->calibration[3].scale = (2.0f * ctx->accel_denominator  / ctx->accel_numerator) / sRange2g;

        sRange2g = sAccYPlus - sAccYMinus;
        ctx->calibration[4].bias = sAccYPlus - sRange2g / 2;
        ctx->calibration[4].scale = (2.0f * ctx->accel_denominator / ctx->accel_numerator) / sRange2g;

        sRange2g = sAccZPlus - sAccZMinus;
        ctx->calibration[5].bias = sAccZPlus - sRange2g / 2;
        ctx->calibration[5].scale = (2.0f * ctx->accel_denominator / ctx->accel_numerator) / sRange2g;

        ctx->hardware_calibration = true;
        for (i = 0; i < 6; ++i) {
#ifdef DEBUG_PS4_CALIBRATION
            SDL_Log("calibration[%d] bias = %d, sensitivity = %f", i, ctx->calibration[i].bias, ctx->calibration[i].scale);
#endif
            // Some controllers have a bad calibration
            if (SDL_abs(ctx->calibration[i].bias) > 1024 || SDL_fabsf(1.0f - ctx->calibration[i].scale) > 0.5f) {
#ifdef DEBUG_PS4_CALIBRATION
                SDL_Log("invalid calibration, ignoring");
#endif
                ctx->hardware_calibration = false;
            }
        }
    } else {
#ifdef DEBUG_PS4_CALIBRATION
        SDL_Log("Calibration data not available");
#endif
    }
    return ctx->hardware_calibration;
}

static void HIDAPI_DriverPS4_LoadCalibrationData(SDL_HIDAPI_Device *device)
{
    SDL_DriverPS4_Context *ctx = (SDL_DriverPS4_Context *)device->context;
    int i;

    if (!HIDAPI_DriverPS4_LoadOfficialCalibrationData(device)) {
        for (i = 0; i < SDL_arraysize(ctx->calibration); ++i) {
            ctx->calibration[i].bias = 0;
            ctx->calibration[i].scale = 1.0f;
        }
    }

    // Scale the raw data to the units expected by SDL
    for (i = 0; i < SDL_arraysize(ctx->calibration); ++i) {
        double scale = ctx->calibration[i].scale;

        if (i < 3) {
            scale *= ((double)ctx->gyro_numerator / ctx->gyro_denominator) * SDL_PI_D / 180.0;

             if (device->vendor_id == USB_VENDOR_SONY &&
                 device->product_id == USB_PRODUCT_SONY_DS4_STRIKEPAD) {
                 // The Armor-X Pro seems to only deliver half the rotation it should
                 scale *= 2.0;
             }
        } else {
            scale *= ((double)ctx->accel_numerator / ctx->accel_denominator) * SDL_STANDARD_GRAVITY;

            if (device->vendor_id == USB_VENDOR_SONY &&
                device->product_id == USB_PRODUCT_SONY_DS4_STRIKEPAD) {
                /* The Armor-X Pro seems to only deliver half the acceleration it should,
                 * and in the opposite direction on all axes */
                scale *= -2.0;
            }
        }
        ctx->calibration[i].scale = (float)scale;
    }
}

static float HIDAPI_DriverPS4_ApplyCalibrationData(SDL_DriverPS4_Context *ctx, int index, Sint16 value)
{
    IMUCalibrationData *calibration = &ctx->calibration[index];

    return ((float)value - calibration->bias) * calibration->scale;
}

static bool HIDAPI_DriverPS4_UpdateEffects(SDL_DriverPS4_Context *ctx, bool application_usage)
{
    DS4EffectsState_t effects;

    SDL_zero(effects);

    if (ctx->vibration_supported) {
        effects.ucRumbleLeft = ctx->rumble_left;
        effects.ucRumbleRight = ctx->rumble_right;
    }

    if (ctx->lightbar_supported) {
        // Populate the LED state with the appropriate color from our lookup table
        if (ctx->color_set) {
            effects.ucLedRed = ctx->led_red;
            effects.ucLedGreen = ctx->led_green;
            effects.ucLedBlue = ctx->led_blue;
        } else {
            SetLedsForPlayerIndex(&effects, ctx->player_index);
        }
    }
    return HIDAPI_DriverPS4_InternalSendJoystickEffect(ctx, &effects, sizeof(effects), application_usage);
}

static void HIDAPI_DriverPS4_TickleBluetooth(SDL_HIDAPI_Device *device)
{
    SDL_DriverPS4_Context *ctx = (SDL_DriverPS4_Context *)device->context;

    if (ctx->enhanced_reports) {
        // This is just a dummy packet that should have no effect, since we don't set the CRC
        Uint8 data[78];

        SDL_zeroa(data);

        data[0] = k_EPS4ReportIdBluetoothEffects;
        data[1] = 0xC0; // Magic value HID + CRC

        if (SDL_HIDAPI_LockRumble()) {
            SDL_HIDAPI_SendRumbleAndUnlock(device, data, sizeof(data));
        }
    } else {
#if 0 /* The 8BitDo Zero 2 has perfect emulation of a PS4 controller, except it
       * only sends reports when the state changes, so we can't disconnect here.
       */
        // We can't even send an invalid effects packet, or it will put the controller in enhanced mode
        if (device->num_joysticks > 0) {
            HIDAPI_JoystickDisconnected(device, device->joysticks[0]);
        }
#endif
    }
}

static void HIDAPI_DriverPS4_SetEnhancedModeAvailable(SDL_DriverPS4_Context *ctx)
{
    if (ctx->enhanced_mode_available) {
        return;
    }
    ctx->enhanced_mode_available = true;

    if (ctx->touchpad_supported) {
        SDL_PrivateJoystickAddTouchpad(ctx->joystick, 2);
        ctx->report_touchpad = true;
    }

    if (ctx->sensors_supported) {
        SDL_PrivateJoystickAddSensor(ctx->joystick, SDL_SENSOR_GYRO, (float)(1000 / ctx->report_interval));
        SDL_PrivateJoystickAddSensor(ctx->joystick, SDL_SENSOR_ACCEL, (float)(1000 / ctx->report_interval));
    }

    if (ctx->official_controller) {
        ctx->report_battery = true;
    }

    HIDAPI_UpdateDeviceProperties(ctx->device);
}

static void HIDAPI_DriverPS4_SetEnhancedMode(SDL_DriverPS4_Context *ctx)
{
    HIDAPI_DriverPS4_SetEnhancedModeAvailable(ctx);

    if (!ctx->enhanced_mode) {
        ctx->enhanced_mode = true;

        // Switch into enhanced report mode
        HIDAPI_DriverPS4_UpdateEffects(ctx, false);
    }
}

static void HIDAPI_DriverPS4_SetEnhancedReportHint(SDL_DriverPS4_Context *ctx, HIDAPI_PS4_EnhancedReportHint enhanced_report_hint)
{
    switch (enhanced_report_hint) {
    case PS4_ENHANCED_REPORT_HINT_OFF:
        // Nothing to do, enhanced mode is a one-way ticket
        break;
    case PS4_ENHANCED_REPORT_HINT_ON:
        HIDAPI_DriverPS4_SetEnhancedMode(ctx);
        break;
    case PS4_ENHANCED_REPORT_HINT_AUTO:
        HIDAPI_DriverPS4_SetEnhancedModeAvailable(ctx);
        break;
    }
    ctx->enhanced_report_hint = enhanced_report_hint;
}

static void HIDAPI_DriverPS4_UpdateEnhancedModeOnEnhancedReport(SDL_DriverPS4_Context *ctx)
{
    ctx->enhanced_reports = true;

    if (ctx->enhanced_report_hint == PS4_ENHANCED_REPORT_HINT_AUTO) {
        HIDAPI_DriverPS4_SetEnhancedReportHint(ctx, PS4_ENHANCED_REPORT_HINT_ON);
    }
}

static void HIDAPI_DriverPS4_UpdateEnhancedModeOnApplicationUsage(SDL_DriverPS4_Context *ctx)
{
    if (ctx->enhanced_report_hint == PS4_ENHANCED_REPORT_HINT_AUTO) {
        HIDAPI_DriverPS4_SetEnhancedReportHint(ctx, PS4_ENHANCED_REPORT_HINT_ON);
    }
}

static void SDLCALL SDL_PS4EnhancedReportsChanged(void *userdata, const char *name, const char *oldValue, const char *hint)
{
    SDL_DriverPS4_Context *ctx = (SDL_DriverPS4_Context *)userdata;

    if (ctx->device->is_bluetooth) {
        if (hint && SDL_strcasecmp(hint, "auto") == 0) {
            HIDAPI_DriverPS4_SetEnhancedReportHint(ctx, PS4_ENHANCED_REPORT_HINT_AUTO);
        } else if (SDL_GetStringBoolean(hint, true)) {
            HIDAPI_DriverPS4_SetEnhancedReportHint(ctx, PS4_ENHANCED_REPORT_HINT_ON);
        } else {
            HIDAPI_DriverPS4_SetEnhancedReportHint(ctx, PS4_ENHANCED_REPORT_HINT_OFF);
        }
    } else {
        HIDAPI_DriverPS4_SetEnhancedReportHint(ctx, PS4_ENHANCED_REPORT_HINT_ON);
    }
}

static void SDLCALL SDL_PS4ReportIntervalHintChanged(void *userdata, const char *name, const char *oldValue, const char *hint)
{
    const int DEFAULT_REPORT_INTERVAL = 4;
    SDL_DriverPS4_Context *ctx = (SDL_DriverPS4_Context *)userdata;
    int new_report_interval = DEFAULT_REPORT_INTERVAL;

    if (hint) {
        int report_interval = SDL_atoi(hint);
        switch (report_interval) {
        case 1:
        case 2:
        case 4:
            // Valid values
            new_report_interval = report_interval;
            break;
        default:
            break;
        }
    }

    if (new_report_interval != ctx->report_interval) {
        ctx->report_interval = (Uint8)new_report_interval;

        HIDAPI_DriverPS4_UpdateEffects(ctx, false);
        SDL_LockJoysticks();
        SDL_PrivateJoystickSensorRate(ctx->joystick, SDL_SENSOR_GYRO, (float)(1000 / ctx->report_interval));
        SDL_PrivateJoystickSensorRate(ctx->joystick, SDL_SENSOR_ACCEL, (float)(1000 / ctx->report_interval));
        SDL_UnlockJoysticks();
    }
}

static void HIDAPI_DriverPS4_SetDevicePlayerIndex(SDL_HIDAPI_Device *device, SDL_JoystickID instance_id, int player_index)
{
    SDL_DriverPS4_Context *ctx = (SDL_DriverPS4_Context *)device->context;

    if (!ctx->joystick) {
        return;
    }

    ctx->player_index = player_index;

    // This will set the new LED state based on the new player index
    // SDL automatically calls this, so it doesn't count as an application action to enable enhanced mode
    HIDAPI_DriverPS4_UpdateEffects(ctx, false);
}

static bool HIDAPI_DriverPS4_OpenJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
    SDL_DriverPS4_Context *ctx = (SDL_DriverPS4_Context *)device->context;

    SDL_AssertJoysticksLocked();

    ctx->joystick = joystick;
    ctx->last_packet = SDL_GetTicks();
    ctx->report_sensors = false;
    ctx->report_touchpad = false;
    ctx->rumble_left = 0;
    ctx->rumble_right = 0;
    ctx->color_set = false;
    SDL_zero(ctx->last_state);

    // Initialize player index (needed for setting LEDs)
    ctx->player_index = SDL_GetJoystickPlayerIndex(joystick);

    // Initialize the joystick capabilities
    joystick->nbuttons = 11;
    if (ctx->touchpad_supported) {
        joystick->nbuttons += 1;
    }
    joystick->naxes = SDL_GAMEPAD_AXIS_COUNT;
    joystick->nhats = 1;

    SDL_AddHintCallback(SDL_HINT_JOYSTICK_HIDAPI_PS4_REPORT_INTERVAL,
                        SDL_PS4ReportIntervalHintChanged, ctx);
    SDL_AddHintCallback(SDL_HINT_JOYSTICK_ENHANCED_REPORTS,
                        SDL_PS4EnhancedReportsChanged, ctx);
    return true;
}

static bool HIDAPI_DriverPS4_RumbleJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint16 low_frequency_rumble, Uint16 high_frequency_rumble)
{
    SDL_DriverPS4_Context *ctx = (SDL_DriverPS4_Context *)device->context;

    if (!ctx->vibration_supported) {
        return SDL_Unsupported();
    }

    ctx->rumble_left = (low_frequency_rumble >> 8);
    ctx->rumble_right = (high_frequency_rumble >> 8);

    return HIDAPI_DriverPS4_UpdateEffects(ctx, true);
}

static bool HIDAPI_DriverPS4_RumbleJoystickTriggers(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint16 left_rumble, Uint16 right_rumble)
{
    return SDL_Unsupported();
}

static Uint32 HIDAPI_DriverPS4_GetJoystickCapabilities(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
    SDL_DriverPS4_Context *ctx = (SDL_DriverPS4_Context *)device->context;
    Uint32 result = 0;

    if (ctx->enhanced_mode_available) {
        if (ctx->lightbar_supported) {
            result |= SDL_JOYSTICK_CAP_RGB_LED;
        }
        if (ctx->vibration_supported) {
            result |= SDL_JOYSTICK_CAP_RUMBLE;
        }
    }

    return result;
}

static bool HIDAPI_DriverPS4_SetJoystickLED(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint8 red, Uint8 green, Uint8 blue)
{
    SDL_DriverPS4_Context *ctx = (SDL_DriverPS4_Context *)device->context;

    if (!ctx->lightbar_supported) {
        return SDL_Unsupported();
    }

    ctx->color_set = true;
    ctx->led_red = red;
    ctx->led_green = green;
    ctx->led_blue = blue;

    return HIDAPI_DriverPS4_UpdateEffects(ctx, true);
}

static bool HIDAPI_DriverPS4_InternalSendJoystickEffect(SDL_DriverPS4_Context *ctx, const void *effect, int size, bool application_usage)
{
    Uint8 data[78];
    int report_size, offset;

    if (!ctx->effects_supported) {
        // We shouldn't be sending packets to this controller
        return SDL_Unsupported();
    }

    if (!ctx->enhanced_mode) {
        if (application_usage) {
            HIDAPI_DriverPS4_UpdateEnhancedModeOnApplicationUsage(ctx);
        }

        if (!ctx->enhanced_mode) {
            // We're not in enhanced mode, effects aren't allowed
            return SDL_Unsupported();
        }
    }

    SDL_zeroa(data);

    if (ctx->device->is_bluetooth && ctx->official_controller) {
        data[0] = k_EPS4ReportIdBluetoothEffects;
        data[1] = 0xC0 | ctx->report_interval; // Magic value HID + CRC, also sets update interval
        data[3] = 0x03;        // 0x1 is rumble, 0x2 is lightbar, 0x4 is the blink interval

        report_size = 78;
        offset = 6;
    } else {
        data[0] = k_EPS4ReportIdUsbEffects;
        data[1] = 0x07; // Magic value

        report_size = 32;
        offset = 4;
    }

    SDL_memcpy(&data[offset], effect, SDL_min((sizeof(data) - offset), (size_t)size));

    if (ctx->device->is_bluetooth) {
        // Bluetooth reports need a CRC at the end of the packet (at least on Linux)
        Uint8 ubHdr = 0xA2; // hidp header is part of the CRC calculation
        Uint32 unCRC;
        unCRC = SDL_crc32(0, &ubHdr, 1);
        unCRC = SDL_crc32(unCRC, data, (size_t)(report_size - sizeof(unCRC)));
        SDL_memcpy(&data[report_size - sizeof(unCRC)], &unCRC, sizeof(unCRC));
    }

    if (SDL_HIDAPI_SendRumble(ctx->device, data, report_size) != report_size) {
        return SDL_SetError("Couldn't send rumble packet");
    }
    return true;
}

static bool HIDAPI_DriverPS4_SendJoystickEffect(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, const void *effect, int size)
{
    SDL_DriverPS4_Context *ctx = (SDL_DriverPS4_Context *)device->context;

    return HIDAPI_DriverPS4_InternalSendJoystickEffect(ctx, effect, size, true);
}

static bool HIDAPI_DriverPS4_SetJoystickSensorsEnabled(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, bool enabled)
{
    SDL_DriverPS4_Context *ctx = (SDL_DriverPS4_Context *)device->context;

    HIDAPI_DriverPS4_UpdateEnhancedModeOnApplicationUsage(ctx);

    if (!ctx->sensors_supported || (enabled && !ctx->enhanced_mode)) {
        return SDL_Unsupported();
    }

    if (enabled) {
        HIDAPI_DriverPS4_LoadCalibrationData(device);
    }
    ctx->report_sensors = enabled;

    return true;
}

static void HIDAPI_DriverPS4_HandleStatePacket(SDL_Joystick *joystick, SDL_hid_device *dev, SDL_DriverPS4_Context *ctx, PS4StatePacket_t *packet, int size)
{
    static const float TOUCHPAD_SCALEX = 1.0f / 1920;
    static const float TOUCHPAD_SCALEY = 1.0f / 920; // This is noted as being 944 resolution, but 920 feels better
    Sint16 axis;
    bool touchpad_down;
    int touchpad_x, touchpad_y;
    Uint64 timestamp = SDL_GetTicksNS();

    if (size > 9 && ctx->report_touchpad && ctx->enhanced_reports) {
        touchpad_down = ((packet->ucTouchpadCounter1 & 0x80) == 0);
        touchpad_x = packet->rgucTouchpadData1[0] | (((int)packet->rgucTouchpadData1[1] & 0x0F) << 8);
        touchpad_y = (packet->rgucTouchpadData1[1] >> 4) | ((int)packet->rgucTouchpadData1[2] << 4);
        SDL_SendJoystickTouchpad(timestamp, joystick, 0, 0, touchpad_down, touchpad_x * TOUCHPAD_SCALEX, touchpad_y * TOUCHPAD_SCALEY, touchpad_down ? 1.0f : 0.0f);

        touchpad_down = ((packet->ucTouchpadCounter2 & 0x80) == 0);
        touchpad_x = packet->rgucTouchpadData2[0] | (((int)packet->rgucTouchpadData2[1] & 0x0F) << 8);
        touchpad_y = (packet->rgucTouchpadData2[1] >> 4) | ((int)packet->rgucTouchpadData2[2] << 4);
        SDL_SendJoystickTouchpad(timestamp, joystick, 0, 1, touchpad_down, touchpad_x * TOUCHPAD_SCALEX, touchpad_y * TOUCHPAD_SCALEY, touchpad_down ? 1.0f : 0.0f);
    }

    if (ctx->last_state.rgucButtonsHatAndCounter[0] != packet->rgucButtonsHatAndCounter[0]) {
        {
            Uint8 data = (packet->rgucButtonsHatAndCounter[0] >> 4);

            SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_WEST, ((data & 0x01) != 0));
            SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SOUTH, ((data & 0x02) != 0));
            SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_EAST, ((data & 0x04) != 0));
            SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_NORTH, ((data & 0x08) != 0));
        }
        {
            Uint8 hat;
            Uint8 data = (packet->rgucButtonsHatAndCounter[0] & 0x0F);

            switch (data) {
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
    }

    if (ctx->last_state.rgucButtonsHatAndCounter[1] != packet->rgucButtonsHatAndCounter[1]) {
        Uint8 data = packet->rgucButtonsHatAndCounter[1];

        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_SHOULDER, ((data & 0x01) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER, ((data & 0x02) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_BACK, ((data & 0x10) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_START, ((data & 0x20) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_STICK, ((data & 0x40) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_RIGHT_STICK, ((data & 0x80) != 0));
    }

    /* Some fightsticks, ex: Victrix FS Pro will only this these digital trigger bits and not the analog values so this needs to run whenever the
       trigger is evaluated
    */
    if (packet->rgucButtonsHatAndCounter[1] & 0x0C) {
        Uint8 data = packet->rgucButtonsHatAndCounter[1];
        packet->ucTriggerLeft = (data & 0x04) && packet->ucTriggerLeft == 0 ? 255 : packet->ucTriggerLeft;
        packet->ucTriggerRight = (data & 0x08) && packet->ucTriggerRight == 0 ? 255 : packet->ucTriggerRight;
    }

    if (ctx->last_state.rgucButtonsHatAndCounter[2] != packet->rgucButtonsHatAndCounter[2]) {
        Uint8 data = (packet->rgucButtonsHatAndCounter[2] & 0x03);

        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_GUIDE, ((data & 0x01) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_PS4_TOUCHPAD, ((data & 0x02) != 0));
    }

    axis = ((int)packet->ucTriggerLeft * 257) - 32768;
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFT_TRIGGER, axis);
    axis = ((int)packet->ucTriggerRight * 257) - 32768;
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER, axis);
    axis = ((int)packet->ucLeftJoystickX * 257) - 32768;
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTX, axis);
    axis = ((int)packet->ucLeftJoystickY * 257) - 32768;
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTY, axis);
    axis = ((int)packet->ucRightJoystickX * 257) - 32768;
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTX, axis);
    axis = ((int)packet->ucRightJoystickY * 257) - 32768;
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTY, axis);

    if (size > 9 && ctx->report_battery && ctx->enhanced_reports) {
        SDL_PowerState state;
        int percent;
        Uint8 level = (packet->ucBatteryLevel & 0x0F);

        if (packet->ucBatteryLevel & 0x10) {
            if (level <= 10) {
                state = SDL_POWERSTATE_CHARGING;
                percent = SDL_min(level * 10 + 5, 100);
            } else if (level == 11) {
                state = SDL_POWERSTATE_CHARGED;
                percent = 100;
            } else {
                state = SDL_POWERSTATE_UNKNOWN;
                percent = 0;
            }
        } else {
            state = SDL_POWERSTATE_ON_BATTERY;
            percent = SDL_min(level * 10 + 5, 100);
        }
        SDL_SendJoystickPowerInfo(joystick, state, percent);
    }

    if (size > 9 && ctx->report_sensors) {
        Uint16 tick;
        Uint16 delta;
        Uint64 sensor_timestamp;
        float data[3];

        tick = LOAD16(packet->rgucTimestamp[0], packet->rgucTimestamp[1]);
        if (ctx->last_tick < tick) {
            delta = (tick - ctx->last_tick);
        } else {
            delta = (SDL_MAX_UINT16 - ctx->last_tick + tick + 1);
        }
        ctx->sensor_ticks += delta;
        ctx->last_tick = tick;

        // Sensor timestamp is in 5.33us units
        sensor_timestamp = (ctx->sensor_ticks * SDL_NS_PER_US * 16) / 3;

        data[0] = HIDAPI_DriverPS4_ApplyCalibrationData(ctx, 0, LOAD16(packet->rgucGyroX[0], packet->rgucGyroX[1]));
        data[1] = HIDAPI_DriverPS4_ApplyCalibrationData(ctx, 1, LOAD16(packet->rgucGyroY[0], packet->rgucGyroY[1]));
        data[2] = HIDAPI_DriverPS4_ApplyCalibrationData(ctx, 2, LOAD16(packet->rgucGyroZ[0], packet->rgucGyroZ[1]));
        SDL_SendJoystickSensor(timestamp, joystick, SDL_SENSOR_GYRO, sensor_timestamp, data, 3);

        data[0] = HIDAPI_DriverPS4_ApplyCalibrationData(ctx, 3, LOAD16(packet->rgucAccelX[0], packet->rgucAccelX[1]));
        data[1] = HIDAPI_DriverPS4_ApplyCalibrationData(ctx, 4, LOAD16(packet->rgucAccelY[0], packet->rgucAccelY[1]));
        data[2] = HIDAPI_DriverPS4_ApplyCalibrationData(ctx, 5, LOAD16(packet->rgucAccelZ[0], packet->rgucAccelZ[1]));
        SDL_SendJoystickSensor(timestamp, joystick, SDL_SENSOR_ACCEL, sensor_timestamp, data, 3);
    }

    SDL_memcpy(&ctx->last_state, packet, sizeof(ctx->last_state));
}

static bool VerifyCRC(Uint8 *data, int size)
{
    Uint8 ubHdr = 0xA1; // hidp header is part of the CRC calculation
    Uint32 unCRC, unPacketCRC;
    Uint8 *packetCRC = data + size - sizeof(unPacketCRC);
    unCRC = SDL_crc32(0, &ubHdr, 1);
    unCRC = SDL_crc32(unCRC, data, (size_t)(size - sizeof(unCRC)));

    unPacketCRC = LOAD32(packetCRC[0],
                         packetCRC[1],
                         packetCRC[2],
                         packetCRC[3]);
    return (unCRC == unPacketCRC);
}

static bool HIDAPI_DriverPS4_IsPacketValid(SDL_DriverPS4_Context *ctx, Uint8 *data, int size)
{
    switch (data[0]) {
    case k_EPS4ReportIdUsbState:
        if (size == 10) {
            // This is non-enhanced mode, this packet is fine
            return true;
        }

        if (ctx->is_nacon_dongle && size >= (1 + sizeof(PS4StatePacket_t))) {
            // The report timestamp doesn't change when the controller isn't connected
            PS4StatePacket_t *packet = (PS4StatePacket_t *)&data[1];
            if (SDL_memcmp(packet->rgucTimestamp, ctx->last_state.rgucTimestamp, sizeof(packet->rgucTimestamp)) == 0) {
                return false;
            }
            if (ctx->last_state.rgucAccelX[0] == 0 && ctx->last_state.rgucAccelX[1] == 0 &&
                ctx->last_state.rgucAccelY[0] == 0 && ctx->last_state.rgucAccelY[1] == 0 &&
                ctx->last_state.rgucAccelZ[0] == 0 && ctx->last_state.rgucAccelZ[1] == 0) {
                // We don't have any state to compare yet, go ahead and copy it
                SDL_memcpy(&ctx->last_state, &data[1], sizeof(PS4StatePacket_t));
                return false;
            }
        }

        /* In the case of a DS4 USB dongle, bit[2] of byte 31 indicates if a DS4 is actually connected (indicated by '0').
         * For non-dongle, this bit is always 0 (connected).
         * This is usually the ID over USB, but the DS4v2 that started shipping with the PS4 Slim will also send this
         * packet over BT with a size of 128
         */
        if (size >= 64 && !(data[31] & 0x04)) {
            return true;
        }
        break;
    case k_EPS4ReportIdBluetoothState1:
    case k_EPS4ReportIdBluetoothState2:
    case k_EPS4ReportIdBluetoothState3:
    case k_EPS4ReportIdBluetoothState4:
    case k_EPS4ReportIdBluetoothState5:
    case k_EPS4ReportIdBluetoothState6:
    case k_EPS4ReportIdBluetoothState7:
    case k_EPS4ReportIdBluetoothState8:
    case k_EPS4ReportIdBluetoothState9:
        // Bluetooth state packets have two additional bytes at the beginning, the first notes if HID data is present
        if (size >= 78 && (data[1] & 0x80)) {
            if (VerifyCRC(data, 78)) {
                ++ctx->valid_crc_packets;
            } else {
                if (ctx->valid_crc_packets > 0) {
                    --ctx->valid_crc_packets;
                }
                if (ctx->valid_crc_packets >= 3) {
                    // We're generally getting valid CRC, but failed one
                    return false;
                }
            }
            return true;
        }
        break;
    default:
        break;
    }
    return false;
}

static bool HIDAPI_DriverPS4_UpdateDevice(SDL_HIDAPI_Device *device)
{
    SDL_DriverPS4_Context *ctx = (SDL_DriverPS4_Context *)device->context;
    SDL_Joystick *joystick = NULL;
    Uint8 data[USB_PACKET_LENGTH * 2];
    int size;
    int packet_count = 0;
    Uint64 now = SDL_GetTicks();

    if (device->num_joysticks > 0) {
        joystick = SDL_GetJoystickFromID(device->joysticks[0]);
    }

    while ((size = SDL_hid_read_timeout(device->dev, data, sizeof(data), 0)) > 0) {
#ifdef DEBUG_PS4_PROTOCOL
        HIDAPI_DumpPacket("PS4 packet: size = %d", data, size);
#endif
        if (!HIDAPI_DriverPS4_IsPacketValid(ctx, data, size)) {
            continue;
        }

        ++packet_count;
        ctx->last_packet = now;

        if (!joystick) {
            continue;
        }

        switch (data[0]) {
        case k_EPS4ReportIdUsbState:
            HIDAPI_DriverPS4_HandleStatePacket(joystick, device->dev, ctx, (PS4StatePacket_t *)&data[1], size - 1);
            break;
        case k_EPS4ReportIdBluetoothState1:
        case k_EPS4ReportIdBluetoothState2:
        case k_EPS4ReportIdBluetoothState3:
        case k_EPS4ReportIdBluetoothState4:
        case k_EPS4ReportIdBluetoothState5:
        case k_EPS4ReportIdBluetoothState6:
        case k_EPS4ReportIdBluetoothState7:
        case k_EPS4ReportIdBluetoothState8:
        case k_EPS4ReportIdBluetoothState9:
            // This is the extended report, we can enable effects now in auto mode
            HIDAPI_DriverPS4_UpdateEnhancedModeOnEnhancedReport(ctx);

            // Bluetooth state packets have two additional bytes at the beginning, the first notes if HID is present
            HIDAPI_DriverPS4_HandleStatePacket(joystick, device->dev, ctx, (PS4StatePacket_t *)&data[3], size - 3);
            break;
        default:
#ifdef DEBUG_JOYSTICK
            SDL_Log("Unknown PS4 packet: 0x%.2x", data[0]);
#endif
            break;
        }
    }

    if (device->is_bluetooth) {
        if (packet_count == 0) {
            // Check to see if it looks like the device disconnected
            if (now >= (ctx->last_packet + BLUETOOTH_DISCONNECT_TIMEOUT_MS)) {
                // Send an empty output report to tickle the Bluetooth stack
                HIDAPI_DriverPS4_TickleBluetooth(device);
                ctx->last_packet = now;
            }
        } else {
            // Reconnect the Bluetooth device once the USB device is gone
            if (device->num_joysticks == 0 &&
                !HIDAPI_HasConnectedUSBDevice(device->serial)) {
                HIDAPI_JoystickConnected(device, NULL);
            }
        }
    }

    if (ctx->is_dongle || ctx->is_nacon_dongle) {
        if (packet_count == 0) {
            if (device->num_joysticks > 0) {
                // Check to see if it looks like the device disconnected
                if (now >= (ctx->last_packet + BLUETOOTH_DISCONNECT_TIMEOUT_MS)) {
                    HIDAPI_JoystickDisconnected(device, device->joysticks[0]);
                }
            }
        } else {
            if (device->num_joysticks == 0) {
                char serial[18];
                size = ReadFeatureReport(device->dev, k_ePS4FeatureReportIdSerialNumber, data, sizeof(data));
                if (size >= 7 && (data[1] || data[2] || data[3] || data[4] || data[5] || data[6])) {
                    (void)SDL_snprintf(serial, sizeof(serial), "%.2x-%.2x-%.2x-%.2x-%.2x-%.2x",
                                       data[6], data[5], data[4], data[3], data[2], data[1]);
                    HIDAPI_SetDeviceSerial(device, serial);
                }
                HIDAPI_JoystickConnected(device, NULL);
            }
        }
    }

    if (packet_count == 0 && size < 0 && device->num_joysticks > 0) {
        // Read error, device is disconnected
        HIDAPI_JoystickDisconnected(device, device->joysticks[0]);
    }
    return (size >= 0);
}

static void HIDAPI_DriverPS4_CloseJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
    SDL_DriverPS4_Context *ctx = (SDL_DriverPS4_Context *)device->context;

    SDL_RemoveHintCallback(SDL_HINT_JOYSTICK_HIDAPI_PS4_REPORT_INTERVAL,
                        SDL_PS4ReportIntervalHintChanged, ctx);
    SDL_RemoveHintCallback(SDL_HINT_JOYSTICK_ENHANCED_REPORTS,
                        SDL_PS4EnhancedReportsChanged, ctx);

    ctx->joystick = NULL;

    ctx->report_sensors = false;
    ctx->enhanced_mode = false;
    ctx->enhanced_mode_available = false;
}

static void HIDAPI_DriverPS4_FreeDevice(SDL_HIDAPI_Device *device)
{
}

SDL_HIDAPI_DeviceDriver SDL_HIDAPI_DriverPS4 = {
    SDL_HINT_JOYSTICK_HIDAPI_PS4,
    true,
    HIDAPI_DriverPS4_RegisterHints,
    HIDAPI_DriverPS4_UnregisterHints,
    HIDAPI_DriverPS4_IsEnabled,
    HIDAPI_DriverPS4_IsSupportedDevice,
    HIDAPI_DriverPS4_InitDevice,
    HIDAPI_DriverPS4_GetDevicePlayerIndex,
    HIDAPI_DriverPS4_SetDevicePlayerIndex,
    HIDAPI_DriverPS4_UpdateDevice,
    HIDAPI_DriverPS4_OpenJoystick,
    HIDAPI_DriverPS4_RumbleJoystick,
    HIDAPI_DriverPS4_RumbleJoystickTriggers,
    HIDAPI_DriverPS4_GetJoystickCapabilities,
    HIDAPI_DriverPS4_SetJoystickLED,
    HIDAPI_DriverPS4_SendJoystickEffect,
    HIDAPI_DriverPS4_SetJoystickSensorsEnabled,
    HIDAPI_DriverPS4_CloseJoystick,
    HIDAPI_DriverPS4_FreeDevice,
};

#endif // SDL_JOYSTICK_HIDAPI_PS4

#endif // SDL_JOYSTICK_HIDAPI
