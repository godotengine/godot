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

// #include "../../events/SDL_keyboard_c.h"
#include "../SDL_sysjoystick.h"
#include "SDL_hidapijoystick_c.h"
#include "SDL_hidapi_rumble.h"

#ifdef SDL_JOYSTICK_HIDAPI_GIP

// This driver is based on the Microsoft GIP spec at:
// https://aka.ms/gipdocs
// https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-gipusb/e7c90904-5e21-426e-b9ad-d82adeee0dbc

// Define this if you want to log all packets from the controller
#if 0
#define DEBUG_XBOX_PROTOCOL
#endif

#define MAX_MESSAGE_LENGTH 0x4000
#define MAX_ATTACHMENTS 8

#define GIP_DATA_CLASS_COMMAND (0u << 5)
#define GIP_DATA_CLASS_LOW_LATENCY (1u << 5)
#define GIP_DATA_CLASS_STANDARD_LATENCY (2u << 5)
#define GIP_DATA_CLASS_AUDIO (3u << 5)

#define GIP_DATA_CLASS_SHIFT 5
#define GIP_DATA_CLASS_MASK (7u << 5)

/* System messages */
#define GIP_CMD_PROTO_CONTROL 0x01
#define GIP_CMD_HELLO_DEVICE 0x02
#define GIP_CMD_STATUS_DEVICE 0x03
#define GIP_CMD_METADATA 0x04
#define GIP_CMD_SET_DEVICE_STATE 0x05
#define GIP_CMD_SECURITY 0x06
#define GIP_CMD_GUIDE_BUTTON 0x07
#define GIP_CMD_AUDIO_CONTROL 0x08
#define GIP_CMD_LED 0x0a
#define GIP_CMD_HID_REPORT 0x0b
#define GIP_CMD_FIRMWARE 0x0c
#define GIP_CMD_EXTENDED 0x1e
#define GIP_CMD_DEBUG 0x1f
#define GIP_AUDIO_DATA 0x60

/* Navigation vendor messages */
#define GIP_CMD_DIRECT_MOTOR 0x09
#define GIP_LL_INPUT_REPORT 0x20
#define GIP_LL_OVERFLOW_INPUT_REPORT 0x26

/* Wheel and ArcadeStick vendor messages */
#define GIP_CMD_INITIAL_REPORTS_REQUEST 0x0a
#define GIP_LL_STATIC_CONFIGURATION 0x21
#define GIP_LL_BUTTON_INFO_REPORT 0x22

/* Wheel vendor messages */
#define GIP_CMD_SET_APPLICATION_MEMORY 0x0b
#define GIP_CMD_SET_EQUATIONS_STATES 0x0c
#define GIP_CMD_SET_EQUATION 0x0d

/* FlightStick vendor messages */
#define GIP_CMD_DEVICE_CAPABILITIES 0x00
#define GIP_CMD_LED_CAPABILITIES 0x01
#define GIP_CMD_SET_LED_STATE 0x02

/* Undocumented Elite 2 vendor messages */
#define GIP_CMD_RAW_REPORT 0x0c
#define GIP_CMD_GUIDE_COLOR 0x0e
#define GIP_SL_ELITE_CONFIG 0x4d

#define GIP_BTN_OFFSET_XBE1 28
#define GIP_BTN_OFFSET_XBE2 14

#define GIP_FLAG_FRAGMENT (1u << 7)
#define GIP_FLAG_INIT_FRAG (1u << 6)
#define GIP_FLAG_SYSTEM (1u << 5)
#define GIP_FLAG_ACME (1u << 4)
#define GIP_FLAG_ATTACHMENT_MASK 0x7

#define GIP_AUDIO_FORMAT_NULL 0
#define GIP_AUDIO_FORMAT_8000HZ_1CH 1
#define GIP_AUDIO_FORMAT_8000HZ_2CH 2
#define GIP_AUDIO_FORMAT_12000HZ_1CH 3
#define GIP_AUDIO_FORMAT_12000HZ_2CH 4
#define GIP_AUDIO_FORMAT_16000HZ_1CH 5
#define GIP_AUDIO_FORMAT_16000HZ_2CH 6
#define GIP_AUDIO_FORMAT_20000HZ_1CH 7
#define GIP_AUDIO_FORMAT_20000HZ_2CH 8
#define GIP_AUDIO_FORMAT_24000HZ_1CH 9
#define GIP_AUDIO_FORMAT_24000HZ_2CH 10
#define GIP_AUDIO_FORMAT_32000HZ_1CH 11
#define GIP_AUDIO_FORMAT_32000HZ_2CH 12
#define GIP_AUDIO_FORMAT_40000HZ_1CH 13
#define GIP_AUDIO_FORMAT_40000HZ_2CH 14
#define GIP_AUDIO_FORMAT_48000HZ_1CH 15
#define GIP_AUDIO_FORMAT_48000HZ_2CH 16
#define GIP_AUDIO_FORMAT_48000HZ_6CH 32
#define GIP_AUDIO_FORMAT_48000HZ_8CH 33

/* Protocol Control constants */
#define GIP_CONTROL_CODE_ACK 0
#define GIP_CONTROL_CODE_NACK 1  /* obsolete */
#define GIP_CONTROL_CODE_UNK 2  /* obsolete */
#define GIP_CONTROL_CODE_AB 3  /* obsolete */
#define GIP_CONTROL_CODE_MPER 4  /* obsolete */
#define GIP_CONTROL_CODE_STOP 5  /* obsolete */
#define GIP_CONTROL_CODE_START 6  /* obsolete */
#define GIP_CONTROL_CODE_ERR 7  /* obsolete */

/* Status Device constants */
#define GIP_POWER_LEVEL_OFF 0
#define GIP_POWER_LEVEL_STANDBY 1  /* obsolete */
#define GIP_POWER_LEVEL_FULL 2

#define GIP_NOT_CHARGING 0
#define GIP_CHARGING 1
#define GIP_CHARGE_ERROR 2

#define GIP_BATTERY_ABSENT 0
#define GIP_BATTERY_STANDARD 1
#define GIP_BATTERY_RECHARGEABLE 2

#define GIP_BATTERY_CRITICAL 0
#define GIP_BATTERY_LOW 1
#define GIP_BATTERY_MEDIUM 2
#define GIP_BATTERY_FULL 3

#define GIP_EVENT_FAULT 0x0002

#define GIP_FAULT_UNKNOWN 0
#define GIP_FAULT_HARD 1
#define GIP_FAULT_NMI 2
#define GIP_FAULT_SVC 3
#define GIP_FAULT_PEND_SV 4
#define GIP_FAULT_SMART_PTR 5
#define GIP_FAULT_MCU 6
#define GIP_FAULT_BUS 7
#define GIP_FAULT_USAGE 8
#define GIP_FAULT_RADIO_HANG 9
#define GIP_FAULT_WATCHDOG 10
#define GIP_FAULT_LINK_STALL 11
#define GIP_FAULT_ASSERTION 12

/* Metadata constants */
#define GIP_MESSAGE_FLAG_BIG_ENDIAN (1u << 0)
#define GIP_MESSAGE_FLAG_RELIABLE (1u << 1)
#define GIP_MESSAGE_FLAG_SEQUENCED (1u << 2)
#define GIP_MESSAGE_FLAG_DOWNSTREAM (1u << 3)
#define GIP_MESSAGE_FLAG_UPSTREAM (1u << 4)
#define GIP_MESSAGE_FLAG_DS_REQUEST_RESPONSE (1u << 5)

#define GIP_DATA_TYPE_CUSTOM 1
#define GIP_DATA_TYPE_AUDIO 2
#define GIP_DATA_TYPE_SECURITY 3
#define GIP_DATA_TYPE_GIP 4

/* Set Device State constants */
#define GIP_STATE_START 0
#define GIP_STATE_STOP 1
#define GIP_STATE_STANDBY 2  /* obsolete */
#define GIP_STATE_FULL_POWER 3
#define GIP_STATE_OFF 4
#define GIP_STATE_QUIESCE 5
#define GIP_STATE_UNK6 6
#define GIP_STATE_RESET 7

/* Guide Button Status constants */
#define GIP_LED_GUIDE 0
#define GIP_LID_IR 1  /* deprecated */

#define GIP_LED_GUIDE_OFF 0
#define GIP_LED_GUIDE_ON 1
#define GIP_LED_GUIDE_FAST_BLINK 2
#define GIP_LED_GUIDE_SLOW_BLINK 3
#define GIP_LED_GUIDE_CHARGING_BLINK 4
#define GIP_LED_GUIDE_RAMP_TO_LEVEL 0xd

#define GIP_LED_IR_OFF 0
#define GIP_LED_IR_ON_100MS 1
#define GIP_LED_IR_PATTERN 4

/* Direct Motor Command constants */
#define GIP_MOTOR_RIGHT_VIBRATION (1u << 0)
#define GIP_MOTOR_LEFT_VIBRATION (1u << 1)
#define GIP_MOTOR_RIGHT_IMPULSE (1u << 2)
#define GIP_MOTOR_LEFT_IMPULSE (1u << 3)
#define GIP_MOTOR_ALL 0xF

/* Extended Command constants */
#define GIP_EXTCMD_GET_CAPABILITIES 0x00
#define GIP_EXTCMD_GET_TELEMETRY_DATA 0x01
#define GIP_EXTCMD_GET_SERIAL_NUMBER 0x04

#define GIP_EXTENDED_STATUS_OK 0
#define GIP_EXTENDED_STATUS_NOT_SUPPORTED 1
#define GIP_EXTENDED_STATUS_NOT_READY 2
#define GIP_EXTENDED_STATUS_ACCESS_DENIED 3
#define GIP_EXTENDED_STATUS_FAILED 4

/* Internal constants, not part of protocol */
#define GIP_HELLO_TIMEOUT 2000
#define GIP_ACME_TIMEOUT 10

#define GIP_DEFAULT_IN_SYSTEM_MESSAGES 0x5e
#define GIP_DEFAULT_OUT_SYSTEM_MESSAGES 0x472

#define GIP_FEATURE_CONSOLE_FUNCTION_MAP (1u << 0)
#define GIP_FEATURE_CONSOLE_FUNCTION_MAP_OVERFLOW (1u << 1)
#define GIP_FEATURE_ELITE_BUTTONS (1u << 2)
#define GIP_FEATURE_DYNAMIC_LATENCY_INPUT (1u << 3)
#define GIP_FEATURE_SECURITY_OPT_OUT (1u << 4)
#define GIP_FEATURE_MOTOR_CONTROL (1u << 5)
#define GIP_FEATURE_GUIDE_COLOR (1u << 6)
#define GIP_FEATURE_EXTENDED_SET_DEVICE_STATE (1u << 7)

#define GIP_QUIRK_NO_HELLO (1u << 0)
#define GIP_QUIRK_BROKEN_METADATA (1u << 1)
#define GIP_QUIRK_NO_IMPULSE_VIBRATION (1u << 2)

typedef enum
{
    GIP_METADATA_NONE = 0,
    GIP_METADATA_GOT = 1,
    GIP_METADATA_FAKED = 2,
    GIP_METADATA_PENDING = 3,
} GIP_MetadataStatus;

#ifndef VK_LWIN
#define VK_LWIN 0x5b
#endif

typedef enum
{
    GIP_TYPE_UNKNOWN = -1,
    GIP_TYPE_GAMEPAD = 0,
    GIP_TYPE_ARCADE_STICK = 1,
    GIP_TYPE_WHEEL = 2,
    GIP_TYPE_FLIGHT_STICK = 3,
    GIP_TYPE_NAVIGATION_CONTROLLER = 4,
    GIP_TYPE_CHATPAD = 5,
    GIP_TYPE_HEADSET = 6,
} GIP_AttachmentType;

typedef enum
{
    GIP_RUMBLE_STATE_IDLE,
    GIP_RUMBLE_STATE_QUEUED,
    GIP_RUMBLE_STATE_BUSY,
} GIP_RumbleState;

typedef enum
{
    GIP_BTN_FMT_UNKNOWN,
    GIP_BTN_FMT_XBE1,
    GIP_BTN_FMT_XBE2_RAW,
    GIP_BTN_FMT_XBE2_4,
    GIP_BTN_FMT_XBE2_5,
} GIP_EliteButtonFormat;

/* These come across the wire as little-endian, so let's store them in-memory as such so we can memcmp */
#define MAKE_GUID(NAME, A, B, C, D0, D1, D2, D3, D4, D5, D6, D7) \
    static const GUID NAME = { SDL_Swap32LE(A), SDL_Swap16LE(B), SDL_Swap16LE(C), { D0, D1, D2, D3, D4, D5, D6, D7 } }

typedef struct GUID
{
    Uint32 a;
    Uint16 b;
    Uint16 c;
    Uint8 d[8];
} GUID;
SDL_COMPILE_TIME_ASSERT(GUID, sizeof(GUID) == 16);

MAKE_GUID(GUID_ArcadeStick, 0x332054cc, 0xa34b, 0x41d5, 0xa3, 0x4a, 0xa6, 0xa6, 0x71, 0x1e, 0xc4, 0xb3);
MAKE_GUID(GUID_DynamicLatencyInput, 0x87f2e56b, 0xc3bb, 0x49b1, 0x82, 0x65, 0xff, 0xff, 0xf3, 0x77, 0x99, 0xee);
MAKE_GUID(GUID_FlightStick, 0x03f1a011, 0xefe9, 0x4cc1, 0x96, 0x9c, 0x38, 0xdc, 0x55, 0xf4, 0x04, 0xd0);
MAKE_GUID(GUID_IHeadset, 0xbc25d1a3, 0xc24e, 0x4992, 0x9d, 0xda, 0xef, 0x4f, 0x12, 0x3e, 0xf5, 0xdc);
MAKE_GUID(GUID_IConsoleFunctionMap_InputReport, 0xecddd2fe, 0xd387, 0x4294, 0xbd, 0x96, 0x1a, 0x71, 0x2e, 0x3d, 0xc7, 0x7d);
MAKE_GUID(GUID_IConsoleFunctionMap_OverflowInputReport, 0x137d4bd0, 0x9347, 0x4472, 0xaa, 0x26, 0x8c, 0x34, 0xa0, 0x8f, 0xf9, 0xbd);
MAKE_GUID(GUID_IController, 0x9776ff56, 0x9bfd, 0x4581, 0xad, 0x45, 0xb6, 0x45, 0xbb, 0xa5, 0x26, 0xd6);
MAKE_GUID(GUID_IDevAuthPCOptOut, 0x7a34ce77, 0x7de2, 0x45c6, 0x8c, 0xa4, 0x00, 0x42, 0xc0, 0x8b, 0xd9, 0x4a);
MAKE_GUID(GUID_IEliteButtons, 0x37d19ff7, 0xb5c6, 0x49d1, 0xa7, 0x5e, 0x03, 0xb2, 0x4b, 0xef, 0x8c, 0x89);
MAKE_GUID(GUID_IGamepad, 0x082e402c, 0x07df, 0x45e1, 0xa5, 0xab, 0xa3, 0x12, 0x7a, 0xf1, 0x97, 0xb5);
MAKE_GUID(GUID_NavigationController, 0xb8f31fe7, 0x7386, 0x40e9, 0xa9, 0xf8, 0x2f, 0x21, 0x26, 0x3a, 0xcf, 0xb7);
MAKE_GUID(GUID_Wheel, 0x646979cf, 0x6b71, 0x4e96, 0x8d, 0xf9, 0x59, 0xe3, 0x98, 0xd7, 0x42, 0x0c);

/*
 * The following GUIDs are observed, but the exact meanings aren't known, so
 * for now we document them but don't use them anywhere.
 *
 * MAKE_GUID(GUID_GamepadEmu, 0xe2e5f1bc, 0xa6e6, 0x41a2, 0x8f, 0x43, 0x33, 0xcf, 0xa2, 0x51, 0x09, 0x81);
 * MAKE_GUID(GUID_IAudioOnly, 0x92844cd1, 0xf7c8, 0x49ef, 0x97, 0x77, 0x46, 0x7d, 0xa7, 0x08, 0xad, 0x10);
 * MAKE_GUID(GUID_IControllerProfileModeState, 0xf758dc66, 0x022c, 0x48b8, 0xa4, 0xf6, 0x45, 0x7b, 0xa8, 0x0e, 0x2a, 0x5b);
 * MAKE_GUID(GUID_ICustomAudio, 0x63fd9cc9, 0x94ee, 0x4b5d, 0x9c, 0x4d, 0x8b, 0x86, 0x4c, 0x14, 0x9c, 0xac);
 * MAKE_GUID(GUID_IExtendedDeviceFlags, 0x34ad9b1e, 0x36ad, 0x4fb5, 0x8a, 0xc7, 0x17, 0x23, 0x4c, 0x9f, 0x54, 0x6f);
 * MAKE_GUID(GUID_IProgrammableGamepad, 0x31c1034d, 0xb5b7, 0x4551, 0x98, 0x13, 0x87, 0x69, 0xd4, 0xa0, 0xe4, 0xf9);
 * MAKE_GUID(GUID_IVirtualDevice, 0xdfd26825, 0x110a, 0x4e94, 0xb9, 0x37, 0xb2, 0x7c, 0xe4, 0x7b, 0x25, 0x40);
 * MAKE_GUID(GUID_OnlineDevAuth, 0x632b1fd1, 0xa3e9, 0x44f9, 0x84, 0x20, 0x5c, 0xe3, 0x44, 0xa0, 0x64, 0x04);
 *
 * Seen on Elite Controller, Adaptive Controller: 9ebd00a3-b5e6-4c08-a33b-673126459ec4
 * Seen on Adaptive Controller: ce1e58c5-221c-4bdb-9c24-bf3941601320
 * Seen on Elite 2 Controller: f758dc66-022c-48b8-a4f6-457ba80e2a5b (IControllerProfileModeState)
 * Seen on Elite 2 Controller: 31c1034d-b5b7-4551-9813-8769d4a0e4f9 (IProgrammableGamepad)
 * Seen on Elite 2 Controller: 34ad9b1e-36ad-4fb5-8ac7-17234c9f546f (IExtendedDeviceFlags)
 * Seen on Elite 2 Controller: 88e0b694-6bd9-4416-a560-e7fafdfa528f
 * Seen on Elite 2 Controller: ea96c8c0-b216-448b-be80-7e5deb0698e2
 */

static const int GIP_DataClassMtu[8] = { 64, 64, 64, 2048, 0, 0, 0, 0 };

typedef struct GIP_Quirks
{
    Uint16 vendor_id;
    Uint16 product_id;
    Uint8 attachment_index;
    Uint32 added_features;
    Uint32 filtered_features;
    Uint32 quirks;
    Uint32 extra_in_system[8];
    Uint32 extra_out_system[8];
    GIP_AttachmentType device_type;
    Uint8 extra_buttons;
    Uint8 extra_axes;
} GIP_Quirks;

static const GIP_Quirks quirks[] = {
    { USB_VENDOR_MICROSOFT, USB_PRODUCT_XBOX_ONE_ELITE_SERIES_1, 0,
      .added_features = GIP_FEATURE_ELITE_BUTTONS,
      .filtered_features = GIP_FEATURE_CONSOLE_FUNCTION_MAP },

    { USB_VENDOR_MICROSOFT, USB_PRODUCT_XBOX_ONE_ELITE_SERIES_2, 0,
      .added_features = GIP_FEATURE_ELITE_BUTTONS | GIP_FEATURE_DYNAMIC_LATENCY_INPUT | GIP_FEATURE_CONSOLE_FUNCTION_MAP | GIP_FEATURE_GUIDE_COLOR | GIP_FEATURE_EXTENDED_SET_DEVICE_STATE,
      .extra_in_system = { 1 << GIP_CMD_FIRMWARE },
      .extra_out_system = { 1 << GIP_CMD_FIRMWARE } },

    { USB_VENDOR_MICROSOFT, USB_PRODUCT_XBOX_SERIES_X, 0,
      .added_features = GIP_FEATURE_DYNAMIC_LATENCY_INPUT },

    { USB_VENDOR_PDP, USB_PRODUCT_PDP_ROCK_CANDY, 0,
      .quirks = GIP_QUIRK_NO_HELLO },

    { USB_VENDOR_POWERA, USB_PRODUCT_BDA_XB1_FIGHTPAD, 0,
      .filtered_features = GIP_FEATURE_MOTOR_CONTROL },

    { USB_VENDOR_POWERA, USB_PRODUCT_BDA_XB1_CLASSIC, 0,
      .quirks = GIP_QUIRK_NO_IMPULSE_VIBRATION },

    { USB_VENDOR_POWERA, USB_PRODUCT_BDA_XB1_SPECTRA_PRO, 0,
      .quirks = GIP_QUIRK_NO_IMPULSE_VIBRATION },

    { USB_VENDOR_RAZER, USB_PRODUCT_RAZER_ATROX, 0,
      .filtered_features = GIP_FEATURE_MOTOR_CONTROL,
      .device_type = GIP_TYPE_ARCADE_STICK },

    { USB_VENDOR_THRUSTMASTER, USB_PRODUCT_THRUSTMASTER_T_FLIGHT_HOTAS_ONE, 0,
      .filtered_features = GIP_FEATURE_MOTOR_CONTROL,
      .device_type = GIP_TYPE_FLIGHT_STICK,
      .extra_buttons = 5,
      .extra_axes = 3 },

    {0},
};

typedef struct GIP_Header
{
    Uint8 message_type;
    Uint8 flags;
    Uint8 sequence_id;
    Uint64 length;
} GIP_Header;

typedef struct GIP_AudioFormat
{
    Uint8 inbound;
    Uint8 outbound;
} GIP_AudioFormat;

typedef struct GIP_DeviceMetadata
{
    Uint8 num_audio_formats;
    Uint8 num_preferred_types;
    Uint8 num_supported_interfaces;
    Uint8 hid_descriptor_size;

    Uint32 in_system_messages[8];
    Uint32 out_system_messages[8];

    GIP_AudioFormat *audio_formats;
    char **preferred_types;
    GUID *supported_interfaces;
    Uint8 *hid_descriptor;

    GIP_AttachmentType device_type;
} GIP_DeviceMetadata;

typedef struct GIP_MessageMetadata
{
    Uint8 type;
    Uint16 length;
    Uint16 data_type;
    Uint32 flags;
    Uint16 period;
    Uint16 persistence_timeout;
} GIP_MessageMetadata;

typedef struct GIP_Metadata
{
    Uint16 version_major;
    Uint16 version_minor;

    GIP_DeviceMetadata device;

    Uint8 num_messages;
    GIP_MessageMetadata *message_metadata;
} GIP_Metadata;

struct GIP_Device;
typedef struct GIP_Attachment
{
    struct GIP_Device *device;
    Uint8 attachment_index;
    SDL_JoystickID joystick;
    SDL_KeyboardID keyboard;

    Uint8 fragment_message;
    Uint16 total_length;
    Uint8 *fragment_data;
    Uint32 fragment_offset;
    Uint64 fragment_timer;
    int fragment_retries;

    Uint16 firmware_major_version;
    Uint16 firmware_minor_version;

    GIP_MetadataStatus got_metadata;
    Uint64 metadata_next;
    int metadata_retries;
    GIP_Metadata metadata;

    Uint8 seq_system;
    Uint8 seq_security;
    Uint8 seq_extended;
    Uint8 seq_audio;
    Uint8 seq_vendor;

    int device_state;

    GIP_RumbleState rumble_state;
    Uint64 rumble_time;
    bool rumble_pending;
    Uint8 left_impulse_level;
    Uint8 right_impulse_level;
    Uint8 left_vibration_level;
    Uint8 right_vibration_level;

    Uint8 last_input[64];

    Uint8 last_modifiers;
    bool capslock;
    SDL_Keycode last_key;
    Uint32 altcode;
    int altcode_digit;

    GIP_AttachmentType attachment_type;
    GIP_EliteButtonFormat xbe_format;
    Uint32 features;
    Uint32 quirks;
    Uint8 share_button_idx;
    Uint8 paddle_idx;

    Uint8 extra_button_idx;
    int extra_buttons;
    int extra_axes;
} GIP_Attachment;

typedef struct GIP_Device
{
    SDL_HIDAPI_Device *device;

    Uint64 hello_deadline;
    bool got_hello;
    bool reset_for_metadata;
    int timeout;

    GIP_Attachment *attachments[MAX_ATTACHMENTS];
} GIP_Device;

typedef struct GIP_HelloDevice
{
    Uint64 device_id;
    Uint16 vendor_id;
    Uint16 product_id;
    Uint16 firmware_major_version;
    Uint16 firmware_minor_version;
    Uint16 firmware_build_version;
    Uint16 firmware_revision;
    Uint8 hardware_major_version;
    Uint8 hardware_minor_version;
    Uint8 rf_proto_major_version;
    Uint8 rf_proto_minor_version;
    Uint8 security_major_version;
    Uint8 security_minor_version;
    Uint8 gip_major_version;
    Uint8 gip_minor_version;
} GIP_HelloDevice;

typedef struct GIP_Status
{
    int power_level;
    int charge;
    int battery_type;
    int battery_level;
} GIP_Status;

typedef struct GIP_StatusEvent
{
    Uint16 event_type;
    Uint32 fault_tag;
    Uint32 fault_address;
} GIP_StatusEvent;

typedef struct GIP_ExtendedStatus
{
    GIP_Status base;
    bool device_active;

    int num_events;
    GIP_StatusEvent events[5];
} GIP_ExtendedStatus;

typedef struct GIP_DirectMotor
{
    Uint8 motor_bitmap;
    Uint8 left_impulse_level;
    Uint8 right_impulse_level;
    Uint8 left_vibration_level;
    Uint8 right_vibration_level;
    Uint8 duration;
    Uint8 delay;
    Uint8 repeat;
} GIP_DirectMotor;

typedef struct GIP_InitialReportsRequest
{
    Uint8 type;
    Uint8 data[2];
} GIP_InitialReportsRequest;

static bool GIP_SetMetadataDefaults(GIP_Attachment *attachment);

static int GIP_DecodeLength(Uint64 *length, const Uint8 *bytes, int num_bytes)
{
    *length = 0;
    int offset;

    for (offset = 0; offset < num_bytes; offset++) {
        Uint8 byte = bytes[offset];
        *length |= (byte & 0x7full) << (offset * 7);
        if (!(byte & 0x80)) {
            offset++;
            break;
        }
    }
    return offset;
}

static int GIP_EncodeLength(Uint64 length, Uint8 *bytes, int num_bytes)
{
    int offset;

    for (offset = 0; offset < num_bytes; offset++) {
        Uint8 byte = length & 0x7f;
        length >>= 7;
        if (length) {
            byte |= 0x80;
        }
        bytes[offset] = byte;
        if (!length) {
            offset++;
            break;
        }
    }
    return offset;
}

static bool GIP_SupportsSystemMessage(GIP_Attachment *attachment, Uint8 command, bool upstream)
{
    if (upstream) {
        return attachment->metadata.device.in_system_messages[command >> 5] & (1u << command);
    } else {
        return attachment->metadata.device.out_system_messages[command >> 5] & (1u << command);
    }
}

static bool GIP_SupportsVendorMessage(GIP_Attachment *attachment, Uint8 command, bool upstream)
{
    size_t i;
    for (i = 0; i < attachment->metadata.num_messages; i++) {
        GIP_MessageMetadata *metadata = &attachment->metadata.message_metadata[i];
        if (metadata->type != command) {
            continue;
        }
        if (metadata->flags & GIP_MESSAGE_FLAG_DS_REQUEST_RESPONSE) {
            return true;
        }
        if (upstream) {
            return metadata->flags & GIP_MESSAGE_FLAG_UPSTREAM;
        } else {
            return metadata->flags & GIP_MESSAGE_FLAG_DOWNSTREAM;
        }
    }
    return false;
}

static Uint8 GIP_SequenceNext(GIP_Attachment *attachment, Uint8 command, bool system)
{
    Uint8 seq;

    if (system) {
        switch (command) {
        case GIP_CMD_SECURITY:
            seq = attachment->seq_security++;
            if (!seq) {
                seq = attachment->seq_security++;
            }
            break;
        case GIP_CMD_EXTENDED:
            seq = attachment->seq_extended++;
            if (!seq) {
                seq = attachment->seq_extended++;
            }
            break;
        case GIP_AUDIO_DATA:
            seq = attachment->seq_audio++;
            if (!seq) {
                seq = attachment->seq_audio++;
            }
            break;
        default:
            seq = attachment->seq_system++;
            if (!seq) {
                seq = attachment->seq_system++;
            }
            break;
        }
    } else {
        if (command == GIP_CMD_DIRECT_MOTOR) {
            // The motor sequence number is optional and always works with 0
            return 0;
        }

        seq = attachment->seq_vendor++;
        if (!seq) {
            seq = attachment->seq_vendor++;
        }
    }
    return seq;
}

static void GIP_HandleQuirks(GIP_Attachment *attachment)
{
    size_t i, j;
    for (i = 0; quirks[i].vendor_id; i++) {
        if (quirks[i].vendor_id != attachment->device->device->vendor_id) {
            continue;
        }
        if (quirks[i].product_id != attachment->device->device->product_id) {
            continue;
        }
        if (quirks[i].attachment_index != attachment->attachment_index) {
            continue;
        }
        attachment->features |= quirks[i].added_features;
        attachment->features &= ~quirks[i].filtered_features;
        attachment->quirks = quirks[i].quirks;
        attachment->attachment_type = quirks[i].device_type;

        for (j = 0; j < 8; ++j) {
            attachment->metadata.device.in_system_messages[j] |= quirks[i].extra_in_system[j];
            attachment->metadata.device.out_system_messages[j] |= quirks[i].extra_out_system[j];
        }

        attachment->extra_buttons = quirks[i].extra_buttons;
        attachment->extra_axes = quirks[i].extra_axes;
        break;
    }
}

static bool GIP_SendRawMessage(
    GIP_Device *device,
    Uint8 message_type,
    Uint8 flags,
    Uint8 seq,
    const Uint8 *bytes,
    int num_bytes,
    bool async,
    SDL_HIDAPI_RumbleSentCallback callback,
    void *userdata)
{
    Uint8 buffer[2054] = { message_type, flags, seq };
    int offset = 3;

    if (num_bytes < 0) {
        SDL_LogWarn(SDL_LOG_CATEGORY_INPUT, "GIP: Invalid message length %d", num_bytes);
        return false;
    }

    if (num_bytes > GIP_DataClassMtu[message_type >> GIP_DATA_CLASS_SHIFT]) {
        SDL_LogError(SDL_LOG_CATEGORY_INPUT,
            "Attempted to send a message that requires fragmenting, which is not yet supported.");
        return false;
    }

    offset += GIP_EncodeLength(num_bytes, &buffer[offset], sizeof(buffer) - offset);

    if (num_bytes > 0) {
        SDL_memcpy(&buffer[offset], bytes, num_bytes);
    }
    num_bytes += offset;
#ifdef DEBUG_XBOX_PROTOCOL
    HIDAPI_DumpPacket("GIP sending message: size = %d", buffer, num_bytes);
#endif

    if (async) {
        if (!SDL_HIDAPI_LockRumble()) {
            return false;
        }

        return SDL_HIDAPI_SendRumbleWithCallbackAndUnlock(device->device, buffer, num_bytes, callback, userdata) == num_bytes;
    } else {
        return SDL_hid_write(device->device->dev, buffer, num_bytes) == num_bytes;
    }
}

static bool GIP_SendSystemMessage(
    GIP_Attachment *attachment,
    Uint8 message_type,
    Uint8 flags,
    const Uint8 *bytes,
    int num_bytes)
{
    return GIP_SendRawMessage(attachment->device,
        message_type,
        GIP_FLAG_SYSTEM | attachment->attachment_index | flags,
        GIP_SequenceNext(attachment, message_type, true),
        bytes,
        num_bytes,
        false,
        NULL,
        NULL);
}

static bool GIP_SendVendorMessage(
    GIP_Attachment *attachment,
    Uint8 message_type,
    Uint8 flags,
    const Uint8 *bytes,
    int num_bytes)
{
    return GIP_SendRawMessage(attachment->device,
        message_type,
        flags,
        GIP_SequenceNext(attachment, message_type, false),
        bytes,
        num_bytes,
        true,
        NULL,
        NULL);
}

static bool GIP_AttachmentIsController(GIP_Attachment *attachment)
{
    return attachment->attachment_type != GIP_TYPE_CHATPAD &&
        attachment->attachment_type != GIP_TYPE_HEADSET;
}

static void GIP_MetadataFree(GIP_Metadata *metadata)
{
    SDL_free(metadata->device.audio_formats);
    if (metadata->device.preferred_types) {
        int i;
        for (i = 0; i < metadata->device.num_preferred_types; i++) {
            SDL_free(metadata->device.preferred_types[i]);
        }
        SDL_free(metadata->device.preferred_types);
    }
    SDL_free(metadata->device.supported_interfaces);
    SDL_free(metadata->device.hid_descriptor);

    SDL_free(metadata->message_metadata);
    SDL_zerop(metadata);
}

static bool GIP_ParseDeviceMetadata(GIP_Metadata *metadata, const Uint8 *bytes, int num_bytes, int *offset)
{
    GIP_DeviceMetadata *device = &metadata->device;
    int buffer_offset;
    int count;
    int length;
    int i;

    bytes = &bytes[*offset];
    num_bytes -= *offset;
    if (num_bytes < 16) {
        return false;
    }

    length = bytes[0];
    length |= bytes[1] << 8;
    if (num_bytes < length) {
        return false;
    }

    /* Skip supported firmware versions for now */

    buffer_offset = bytes[4];
    buffer_offset |= bytes[5] << 8;
    if (buffer_offset >= length) {
        return false;
    }
    if (buffer_offset > 0) {
        device->num_audio_formats = bytes[buffer_offset];
        if (buffer_offset + device->num_audio_formats + 1 > length) {
            return false;
        }
        device->audio_formats = SDL_malloc(device->num_audio_formats);
        SDL_memcpy(device->audio_formats, &bytes[buffer_offset + 1], device->num_audio_formats);
    }

    buffer_offset = bytes[6];
    buffer_offset |= bytes[7] << 8;
    if (buffer_offset >= length) {
        return false;
    }
    if (buffer_offset > 0) {
        count = bytes[buffer_offset];
        if (buffer_offset + count + 1 > length) {
            return false;
        }

        for (i = 0; i < count; i++) {
            Uint8 message = bytes[buffer_offset + 1 + i];
#ifdef DEBUG_XBOX_PROTOCOL
            SDL_LogDebug(SDL_LOG_CATEGORY_INPUT,
                "GIP: Supported upstream system message %02x",
                message);
#endif
            device->in_system_messages[message >> 5] |= 1u << (message & 0x1F);
        }
    }

    buffer_offset = bytes[8];
    buffer_offset |= bytes[9] << 8;
    if (buffer_offset >= length) {
        return false;
    }
    if (buffer_offset > 0) {
        count = bytes[buffer_offset];
        if (buffer_offset + count + 1 > length) {
            return false;
        }

        for (i = 0; i < count; i++) {
            Uint8 message = bytes[buffer_offset + 1 + i];
#ifdef DEBUG_XBOX_PROTOCOL
            SDL_LogDebug(SDL_LOG_CATEGORY_INPUT,
                "GIP: Supported downstream system message %02x",
                message);
#endif
            device->out_system_messages[message >> 5] |= 1u << (message & 0x1F);
        }
    }

    buffer_offset = bytes[10];
    buffer_offset |= bytes[11] << 8;
    if (buffer_offset >= length) {
        return false;
    }
    if (buffer_offset > 0) {
        device->num_preferred_types = bytes[buffer_offset];
        device->preferred_types = SDL_calloc(device->num_preferred_types, sizeof(char *));
        buffer_offset++;
        for (i = 0; i < device->num_preferred_types; i++) {
            if (buffer_offset + 2 >= length) {
                return false;
            }

            count = bytes[buffer_offset];
            count |= bytes[buffer_offset];
            buffer_offset += 2;
            if (buffer_offset + count > length) {
                return false;
            }

            device->preferred_types[i] = SDL_calloc(count + 1, sizeof(char));
            SDL_memcpy(device->preferred_types[i], &bytes[buffer_offset], count);
            buffer_offset += count;
        }
    }

    buffer_offset = bytes[12];
    buffer_offset |= bytes[13] << 8;
    if (buffer_offset >= length) {
        return false;
    }
    if (buffer_offset > 0) {
        device->num_supported_interfaces = bytes[buffer_offset];
        if (buffer_offset + 1 + (Sint32) (device->num_supported_interfaces * sizeof(GUID)) > length) {
            return false;
        }
        device->supported_interfaces = SDL_calloc(device->num_supported_interfaces, sizeof(GUID));
        SDL_memcpy(device->supported_interfaces,
            &bytes[buffer_offset + 1],
            sizeof(GUID) * device->num_supported_interfaces);
    }

    if (metadata->version_major > 1 || metadata->version_minor >= 1) {
        /* HID descriptor support added in metadata version 1.1 */
        buffer_offset = bytes[14];
        buffer_offset |= bytes[15] << 8;
        if (buffer_offset >= length) {
            return false;
        }
        if (buffer_offset > 0) {
            device->hid_descriptor_size = bytes[buffer_offset];
            if (buffer_offset + 1 + device->hid_descriptor_size > length) {
                return false;
            }
            device->hid_descriptor = SDL_malloc(device->hid_descriptor_size);
            SDL_memcpy(device->hid_descriptor, &bytes[buffer_offset + 1], device->hid_descriptor_size);
#ifdef DEBUG_XBOX_PROTOCOL
            HIDAPI_DumpPacket("GIP received HID descriptor: size = %d", device->hid_descriptor, device->hid_descriptor_size);
#endif
        }
    }

    *offset += length;
    return true;
}

static bool GIP_ParseMessageMetadata(GIP_MessageMetadata *metadata, const Uint8 *bytes, int num_bytes, int *offset)
{
    Uint16 length;

    bytes = &bytes[*offset];
    num_bytes -= *offset;

    if (num_bytes < 2) {
        return false;
    }
    length = bytes[0];
    length |= bytes[1] << 8;
    if (num_bytes < length) {
        return false;
    }

    if (length < 15) {
        return false;
    }

    metadata->type = bytes[2];
    metadata->length = bytes[3];
    metadata->length |= bytes[4] << 8;
    metadata->data_type = bytes[5];
    metadata->data_type |= bytes[6] << 8;
    metadata->flags = bytes[7];
    metadata->flags |= bytes[8] << 8;
    metadata->flags |= bytes[9] << 16;
    metadata->flags |= bytes[10] << 24;
    metadata->period = bytes[11];
    metadata->period |= bytes[12] << 8;
    metadata->persistence_timeout = bytes[13];
    metadata->persistence_timeout |= bytes[14] << 8;

#ifdef DEBUG_XBOX_PROTOCOL
    SDL_LogDebug(SDL_LOG_CATEGORY_INPUT,
        "GIP: Supported vendor message type %02x of length %d, %s, %s, %s",
        metadata->type,
        metadata->length,
        metadata->flags & GIP_MESSAGE_FLAG_UPSTREAM ?
            (metadata->flags & GIP_MESSAGE_FLAG_DOWNSTREAM ? "bidirectional" : "upstream") :
            metadata->flags & GIP_MESSAGE_FLAG_DOWNSTREAM ? "downstream" :
            metadata->flags & GIP_MESSAGE_FLAG_DS_REQUEST_RESPONSE ? "downstream request response" :
            "unknown direction",
        metadata->flags & GIP_MESSAGE_FLAG_SEQUENCED ? "sequenced" : "not sequenced",
        metadata->flags & GIP_MESSAGE_FLAG_RELIABLE ? "reliable" : "unreliable");
#endif

    *offset += length;
    return true;
}

static bool GIP_ParseMetadata(GIP_Metadata *metadata, const Uint8 *bytes, int num_bytes)
{
    int header_size;
    int metadata_size;
    int offset = 0;
    int i;

    if (num_bytes < 16) {
        return false;
    }

#ifdef DEBUG_XBOX_PROTOCOL
    HIDAPI_DumpPacket("GIP received metadata: size = %d", bytes, num_bytes);
#endif

    header_size = bytes[0];
    header_size |= bytes[1] << 8;
    if (num_bytes < header_size || header_size < 16) {
        return false;
    }
    metadata->version_major = bytes[2];
    metadata->version_major |= bytes[3] << 8;
    metadata->version_minor = bytes[4];
    metadata->version_minor |= bytes[5] << 8;
    /* Middle bytes are reserved */
    metadata_size = bytes[14];
    metadata_size |= bytes[15] << 8;

    if (num_bytes < metadata_size || metadata_size < header_size) {
        return false;
    }
    offset = header_size;

    if (!GIP_ParseDeviceMetadata(metadata, bytes, num_bytes, &offset)) {
        goto err;
    }

    if (offset >= num_bytes) {
        goto err;
    }
    metadata->num_messages = bytes[offset];
    offset++;
    if (metadata->num_messages > 0) {
        metadata->message_metadata = SDL_calloc(metadata->num_messages, sizeof(*metadata->message_metadata));
        for (i = 0; i < metadata->num_messages; i++) {
            if (!GIP_ParseMessageMetadata(&metadata->message_metadata[i], bytes, num_bytes, &offset)) {
                goto err;
            }
        }
    }

    return true;

err:
    GIP_MetadataFree(metadata);
    return false;
}

static bool GIP_Acknowledge(
    GIP_Device *device,
    const GIP_Header *header,
    Uint32 fragment_offset,
    Uint16 bytes_remaining)
{
    Uint8 buffer[] = {
        GIP_CONTROL_CODE_ACK,
        header->message_type,
        header->flags & GIP_FLAG_SYSTEM,
        (Uint8) fragment_offset,
        (Uint8) (fragment_offset >> 8),
        (Uint8) (fragment_offset >> 16),
        fragment_offset >> 24,
        (Uint8) bytes_remaining,
        bytes_remaining >> 8,
    };

    return GIP_SendRawMessage(device,
        GIP_CMD_PROTO_CONTROL,
        GIP_FLAG_SYSTEM | (header->flags & GIP_FLAG_ATTACHMENT_MASK),
        header->sequence_id,
        buffer,
        sizeof(buffer),
        false,
        NULL,
        NULL);
}

static bool GIP_FragmentFailed(GIP_Attachment *attachment, const GIP_Header *header)
{
    attachment->fragment_retries++;
    if (attachment->fragment_retries > 8) {
        if (attachment->fragment_data) {
            SDL_free(attachment->fragment_data);
            attachment->fragment_data = NULL;
        }
        attachment->fragment_message = 0;
    }
    return GIP_Acknowledge(attachment->device,
        header,
        attachment->fragment_offset,
        (Uint16) (attachment->total_length - attachment->fragment_offset));
}

static bool GIP_EnableEliteButtons(GIP_Attachment *attachment) {
    if (attachment->device->device->vendor_id == USB_VENDOR_MICROSOFT) {
        if (attachment->device->device->product_id == USB_PRODUCT_XBOX_ONE_ELITE_SERIES_1) {
            attachment->xbe_format = GIP_BTN_FMT_XBE1;
        } else if (attachment->device->device->product_id == USB_PRODUCT_XBOX_ONE_ELITE_SERIES_2) {
            if (attachment->firmware_major_version == 4) {
                attachment->xbe_format = GIP_BTN_FMT_XBE2_4;
            } else if (attachment->firmware_major_version == 5) {
                /*
                 * The exact range for this being necessary is unknown, but it
                 * starts at 5.11 and at either 5.16 or 5.17. This approach
                 * still works on 5.21, even if it's not necessary, so having
                 * a loose upper limit is fine.
                 */
                if (attachment->firmware_minor_version >= 11 &&
                    attachment->firmware_minor_version < 17)
                {
                    attachment->xbe_format = GIP_BTN_FMT_XBE2_RAW;
                } else {
                    attachment->xbe_format = GIP_BTN_FMT_XBE2_5;
                }
            }
        }
    }
    if (attachment->xbe_format == GIP_BTN_FMT_XBE2_RAW) {
        /*
         * The meaning of this packet is unknown and not documented, but it's
         * needed for the Elite 2 controller to send raw reports
         */
        static const Uint8 enable_raw_report[] = { 7, 0 };

        return GIP_SendVendorMessage(attachment,
            GIP_SL_ELITE_CONFIG,
            0,
            enable_raw_report,
            sizeof(enable_raw_report));
    }

    return true;
}

static bool GIP_SendGuideButtonLED(GIP_Attachment *attachment, Uint8 pattern, Uint8 intensity)
{
    Uint8 buffer[] = {
        GIP_LED_GUIDE,
        pattern,
        intensity,
    };

    if (!GIP_SupportsSystemMessage(attachment, GIP_CMD_LED, false)) {
        return true;
    }
    return GIP_SendSystemMessage(attachment, GIP_CMD_LED, 0, buffer, sizeof(buffer));
}

static bool GIP_SendQueryFirmware(GIP_Attachment *attachment, Uint8 slot)
{
    /* The "slot" variable might not be correct; the packet format is still unclear */
    Uint8 buffer[] = { 0x1, slot, 0, 0, 0 };

    return GIP_SendSystemMessage(attachment, GIP_CMD_FIRMWARE, 0, buffer, sizeof(buffer));
}

static bool GIP_SendSetDeviceState(GIP_Attachment *attachment, Uint8 state)
{
    Uint8 buffer[] = { state };
    return GIP_SendSystemMessage(attachment,
        GIP_CMD_SET_DEVICE_STATE,
        attachment->attachment_index,
        buffer,
        sizeof(buffer));
}

static bool GIP_SendInitSequence(GIP_Attachment *attachment)
{
    if (attachment->features & GIP_FEATURE_EXTENDED_SET_DEVICE_STATE) {
        /*
         * The meaning of this packet is unknown and not documented, but it's
         * needed for the Elite 2 controller to start up on older firmwares
         */
        static const Uint8 set_device_state[] = { GIP_STATE_UNK6, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x55, 0x53, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 };

        if (!GIP_SendSystemMessage(attachment,
            GIP_CMD_SET_DEVICE_STATE,
            0,
            set_device_state,
            sizeof(set_device_state)))
        {
            return false;
        }
    }
    if (!GIP_EnableEliteButtons(attachment)) {
        return false;
    }
    if (!GIP_SendSetDeviceState(attachment, GIP_STATE_START)) {
        return false;
    }
    attachment->device_state = GIP_STATE_START;

    if (!GIP_SendGuideButtonLED(attachment, GIP_LED_GUIDE_ON, 20)) {
        return false;
    }

    if (GIP_SupportsSystemMessage(attachment, GIP_CMD_SECURITY, false) &&
        !(attachment->features & GIP_FEATURE_SECURITY_OPT_OUT))
    {
        /* TODO: Implement Security command property */
        Uint8 buffer[] = { 0x1, 0x0 };
        GIP_SendSystemMessage(attachment, GIP_CMD_SECURITY, 0, buffer, sizeof(buffer));
    }

    if (GIP_SupportsVendorMessage(attachment, GIP_CMD_INITIAL_REPORTS_REQUEST, false)) {
        GIP_InitialReportsRequest request = { 0 };
        GIP_SendVendorMessage(attachment, GIP_CMD_INITIAL_REPORTS_REQUEST, 0, (const Uint8 *)&request, sizeof(request));
    }

    if (GIP_SupportsVendorMessage(attachment, GIP_CMD_DEVICE_CAPABILITIES, false)) {
        GIP_SendVendorMessage(attachment, GIP_CMD_DEVICE_CAPABILITIES, 0, NULL, 0);
    }

    if ((!attachment->attachment_index || GIP_AttachmentIsController(attachment)) && !attachment->joystick) {
        return HIDAPI_JoystickConnected(attachment->device->device, &attachment->joystick);
    }
    if (attachment->attachment_type == GIP_TYPE_CHATPAD && !attachment->keyboard) {
        attachment->keyboard = (SDL_KeyboardID)(uintptr_t) attachment;
        // SDL_AddKeyboard(attachment->keyboard, "Xbox One Chatpad");
    }
    return true;
}

static bool GIP_EnsureMetadata(GIP_Attachment *attachment)
{
    switch (attachment->got_metadata) {
    case GIP_METADATA_GOT:
    case GIP_METADATA_FAKED:
        return true;
    case GIP_METADATA_NONE:
        if (attachment->device->got_hello) {
            attachment->device->timeout = GIP_ACME_TIMEOUT;
            attachment->got_metadata = GIP_METADATA_PENDING;
            attachment->metadata_next = SDL_GetTicks() + 500;
            attachment->metadata_retries = 0;
            return GIP_SendSystemMessage(attachment, GIP_CMD_METADATA, 0, NULL, 0);
        } else {
            return GIP_SetMetadataDefaults(attachment);
        }
    default:
        return true;
    }
}

static bool GIP_SetMetadataDefaults(GIP_Attachment *attachment)
{
    if (attachment->attachment_index == 0) {
        /* Some decent default settings */
        attachment->features |= GIP_FEATURE_MOTOR_CONTROL;
        attachment->attachment_type = GIP_TYPE_GAMEPAD;
        attachment->metadata.device.in_system_messages[0] |= (1u << GIP_CMD_GUIDE_BUTTON);

        if (SDL_IsJoystickXboxSeriesX(attachment->device->device->vendor_id, attachment->device->device->product_id)) {
            attachment->features |= GIP_FEATURE_CONSOLE_FUNCTION_MAP;
        }
    }

    GIP_HandleQuirks(attachment);

    if (GIP_SupportsSystemMessage(attachment, GIP_CMD_FIRMWARE, false)) {
        GIP_SendQueryFirmware(attachment, 2);
    }

    attachment->got_metadata = GIP_METADATA_FAKED;
    attachment->device->hello_deadline = 0;
    if (!attachment->joystick) {
        return HIDAPI_JoystickConnected(attachment->device->device, &attachment->joystick);
    }
    return true;
}

static bool GIP_HandleCommandProtocolControl(
    GIP_Attachment *attachment,
    const GIP_Header *header,
    const Uint8 *bytes,
    int num_bytes)
{
    // TODO
    SDL_LogDebug(SDL_LOG_CATEGORY_INPUT, "GIP: Unimplemented Protocol Control message");
    return false;
}

static bool GIP_HandleCommandHelloDevice(
    GIP_Attachment *attachment,
    const GIP_Header *header,
    const Uint8 *bytes,
    int num_bytes)
{
    GIP_HelloDevice message = {0};

    if (num_bytes != 28) {
        return false;
    }

    message.device_id = (Uint64) bytes[0];
    message.device_id |= (Uint64) bytes[1] << 8;
    message.device_id |= (Uint64) bytes[2] << 16;
    message.device_id |= (Uint64) bytes[3] << 24;
    message.device_id |= (Uint64) bytes[4] << 32;
    message.device_id |= (Uint64) bytes[5] << 40;
    message.device_id |= (Uint64) bytes[6] << 48;
    message.device_id |= (Uint64) bytes[7] << 56;

    message.vendor_id = bytes[8];
    message.vendor_id |= bytes[9] << 8;

    message.product_id = bytes[10];
    message.product_id |= bytes[11] << 8;

    message.firmware_major_version = bytes[12];
    message.firmware_major_version |= bytes[13] << 8;

    message.firmware_minor_version = bytes[14];
    message.firmware_minor_version |= bytes[15] << 8;

    message.firmware_build_version = bytes[16];
    message.firmware_build_version |= bytes[17] << 8;

    message.firmware_revision = bytes[18];
    message.firmware_revision |= bytes[19] << 8;

    message.hardware_major_version = bytes[20];
    message.hardware_minor_version = bytes[21];

    message.rf_proto_major_version = bytes[22];
    message.rf_proto_minor_version = bytes[23];

    message.security_major_version = bytes[24];
    message.security_minor_version = bytes[25];

    message.gip_major_version = bytes[26];
    message.gip_minor_version = bytes[27];

    SDL_LogInfo(SDL_LOG_CATEGORY_INPUT,
        "GIP: Device hello from %" SDL_PRIx64 " (%04x:%04x)",
        message.device_id, message.vendor_id, message.product_id);
    SDL_LogInfo(SDL_LOG_CATEGORY_INPUT,
        "GIP: Firmware version %d.%d.%d rev %d",
        message.firmware_major_version,
        message.firmware_minor_version,
        message.firmware_build_version,
        message.firmware_revision);

    /*
     * The GIP spec specifies that the host should reject the device if any of these are wrong.
     * I don't know if Windows or an Xbox do, however, so let's just log warnings instead.
     */
    if (message.rf_proto_major_version != 1 && message.rf_proto_minor_version != 0) {
        SDL_LogWarn(SDL_LOG_CATEGORY_INPUT,
            "GIP: Invalid RF protocol version %d.%d, expected 1.0",
            message.rf_proto_major_version, message.rf_proto_minor_version);
    }

    if (message.security_major_version != 1 && message.security_minor_version != 0) {
        SDL_LogWarn(SDL_LOG_CATEGORY_INPUT,
            "GIP: Invalid security protocol version %d.%d, expected 1.0",
            message.security_major_version, message.security_minor_version);
    }

    if (message.gip_major_version != 1 && message.gip_minor_version != 0) {
        SDL_LogWarn(SDL_LOG_CATEGORY_INPUT,
            "GIP: Invalid GIP version %d.%d, expected 1.0",
            message.gip_major_version, message.gip_minor_version);
    }

    if (header->flags & GIP_FLAG_ATTACHMENT_MASK) {
        return GIP_SendSystemMessage(attachment, GIP_CMD_METADATA, 0, NULL, 0);
    } else {
        attachment->firmware_major_version = message.firmware_major_version;
        attachment->firmware_minor_version = message.firmware_minor_version;

        if (attachment->attachment_index == 0) {
            attachment->device->hello_deadline = 0;
            attachment->device->got_hello = true;
        }
        if (attachment->got_metadata == GIP_METADATA_FAKED) {
            attachment->got_metadata = GIP_METADATA_NONE;
        }
        GIP_EnsureMetadata(attachment);
    }
    return true;
}

static bool GIP_HandleCommandStatusDevice(
    GIP_Attachment *attachment,
    const GIP_Header *header,
    const Uint8 *bytes,
    int num_bytes)
{
    GIP_ExtendedStatus status;
    SDL_Joystick *joystick = NULL;
    SDL_PowerState power_state;
    int power_percent = 0;
    int i;

    if (num_bytes < 1) {
        return false;
    }
    SDL_zero(status);
    status.base.battery_level = bytes[0] & 3;
    status.base.battery_type = (bytes[0] >> 2) & 3;
    status.base.charge = (bytes[0] >> 4) & 3;
    status.base.power_level = (bytes[0] >> 6) & 3;

    if (attachment->joystick) {
        joystick = SDL_GetJoystickFromID(attachment->joystick);
    }
    if (joystick) {
        switch (status.base.battery_level) {
        case GIP_BATTERY_CRITICAL:
            power_percent = 1;
            break;
        case GIP_BATTERY_LOW:
            power_percent = 25;
            break;
        case GIP_BATTERY_MEDIUM:
            power_percent = 50;
            break;
        case GIP_BATTERY_FULL:
            power_percent = 100;
            break;
        }
        switch (status.base.charge) {
        case GIP_CHARGING:
            if (status.base.battery_level == GIP_BATTERY_FULL) {
                power_state = SDL_POWERSTATE_CHARGED;
            } else {
                power_state = SDL_POWERSTATE_CHARGING;
            }
            break;
        case GIP_NOT_CHARGING:
            power_state = SDL_POWERSTATE_ON_BATTERY;
            break;
        case GIP_CHARGE_ERROR:
        default:
            power_state = SDL_POWERSTATE_UNKNOWN;
            break;
        }

        switch (status.base.battery_type) {
        case GIP_BATTERY_ABSENT:
            power_state = SDL_POWERSTATE_NO_BATTERY;
            break;
        case GIP_BATTERY_STANDARD:
        case GIP_BATTERY_RECHARGEABLE:
            break;
        default:
            power_state = SDL_POWERSTATE_UNKNOWN;
            break;
        }

        SDL_SendJoystickPowerInfo(joystick, power_state, power_percent);
    }

    if (num_bytes >= 4) {
        status.device_active = bytes[1] & 1;
        if (bytes[1] & 2) {
            /* Events present */
            if (num_bytes < 5) {
                return false;
            }
            status.num_events = bytes[4];
            if (status.num_events > 5) {
                SDL_LogWarn(SDL_LOG_CATEGORY_INPUT,
                    "GIP: Device reported too many events, %d > 5",
                    status.num_events);
                return false;
            }
            if (5 + status.num_events * 10 > num_bytes) {
                return false;
            }
            for (i = 0; i < status.num_events; i++) {
                status.events[i].event_type = bytes[i * 10 + 5];
                status.events[i].event_type |= bytes[i * 10 + 6] << 8;
                status.events[i].fault_tag = bytes[i * 10 + 7];
                status.events[i].fault_tag |= bytes[i * 10 + 8] << 8;
                status.events[i].fault_tag |= bytes[i * 10 + 9] << 16;
                status.events[i].fault_tag |= bytes[i * 10 + 10] << 24;
                status.events[i].fault_tag = bytes[i * 10 + 11];
                status.events[i].fault_tag |= bytes[i * 10 + 12] << 8;
                status.events[i].fault_tag |= bytes[i * 10 + 13] << 16;
                status.events[i].fault_tag |= bytes[i * 10 + 14] << 24;
            }
        }
    }

    GIP_EnsureMetadata(attachment);
    return true;
}

static bool GIP_HandleCommandMetadataRespose(
    GIP_Attachment *attachment,
    const GIP_Header *header,
    const Uint8 *bytes,
    int num_bytes)
{
    GIP_Metadata metadata = {0};
    const GUID *expected_guid = NULL;
    bool found_expected_guid;
    bool found_controller_guid = false;
    int i;

    if (!GIP_ParseMetadata(&metadata, bytes, num_bytes)) {
        return false;
    }

    if (attachment->got_metadata == GIP_METADATA_GOT) {
        GIP_MetadataFree(&attachment->metadata);
    }
    attachment->metadata = metadata;
    attachment->got_metadata = GIP_METADATA_GOT;
    attachment->features = 0;

    attachment->attachment_type = GIP_TYPE_UNKNOWN;
#ifdef DEBUG_XBOX_PROTOCOL
    for (i = 0; i < metadata.device.num_preferred_types; i++) {
        const char *type = metadata.device.preferred_types[i];
        SDL_LogDebug(SDL_LOG_CATEGORY_INPUT, "GIP: Device preferred type: %s", type);
    }
#endif
    for (i = 0; i < metadata.device.num_preferred_types; i++) {
        const char *type = metadata.device.preferred_types[i];
        if (SDL_strcmp(type, "Windows.Xbox.Input.Gamepad") == 0) {
            attachment->attachment_type = GIP_TYPE_GAMEPAD;
            expected_guid = &GUID_IGamepad;
            break;
        }
        if (SDL_strcmp(type, "Microsoft.Xbox.Input.ArcadeStick") == 0) {
            attachment->attachment_type = GIP_TYPE_ARCADE_STICK;
            expected_guid = &GUID_ArcadeStick;
            break;
        }
        if (SDL_strcmp(type, "Windows.Xbox.Input.ArcadeStick") == 0) {
            attachment->attachment_type = GIP_TYPE_ARCADE_STICK;
            expected_guid = &GUID_ArcadeStick;
            break;
        }
        if (SDL_strcmp(type, "Microsoft.Xbox.Input.FlightStick") == 0) {
            attachment->attachment_type = GIP_TYPE_FLIGHT_STICK;
            expected_guid = &GUID_FlightStick;
            break;
        }
        if (SDL_strcmp(type, "Windows.Xbox.Input.FlightStick") == 0) {
            attachment->attachment_type = GIP_TYPE_FLIGHT_STICK;
            expected_guid = &GUID_FlightStick;
            break;
        }
        if (SDL_strcmp(type, "Microsoft.Xbox.Input.Wheel") == 0) {
            attachment->attachment_type = GIP_TYPE_WHEEL;
            expected_guid = &GUID_Wheel;
            break;
        }
        if (SDL_strcmp(type, "Windows.Xbox.Input.Wheel") == 0) {
            attachment->attachment_type = GIP_TYPE_WHEEL;
            expected_guid = &GUID_Wheel;
            break;
        }
        if (SDL_strcmp(type, "Windows.Xbox.Input.NavigationController") == 0) {
            attachment->attachment_type = GIP_TYPE_NAVIGATION_CONTROLLER;
            expected_guid = &GUID_NavigationController;
            break;
        }
        if (SDL_strcmp(type, "Windows.Xbox.Input.Chatpad") == 0) {
            attachment->attachment_type = GIP_TYPE_CHATPAD;
            break;
        }
        if (SDL_strcmp(type, "Windows.Xbox.Input.Headset") == 0) {
            attachment->attachment_type = GIP_TYPE_HEADSET;
            expected_guid = &GUID_IHeadset;
            break;
        }
    }

    found_expected_guid = !expected_guid;
    for (i = 0; i < metadata.device.num_supported_interfaces; i++) {
        const GUID* guid = &metadata.device.supported_interfaces[i];
#ifdef DEBUG_XBOX_PROTOCOL
        SDL_LogDebug(SDL_LOG_CATEGORY_INPUT,
            "GIP: Supported interface: %08x-%04x-%04x-%02x%02x-%02x%02x%02x%02x%02x%02x",
            guid->a, guid->b, guid->c, guid->d[0], guid->d[1],
            guid->d[2], guid->d[3], guid->d[4], guid->d[5], guid->d[6], guid->d[7]);
#endif
        if (expected_guid && SDL_memcmp(expected_guid, guid, sizeof(GUID)) == 0) {
            found_expected_guid = true;
        }
        if (SDL_memcmp(&GUID_IController, guid, sizeof(GUID)) == 0) {
            found_controller_guid = true;
            continue;
        }
        if (SDL_memcmp(&GUID_IDevAuthPCOptOut, guid, sizeof(GUID)) == 0) {
            attachment->features |= GIP_FEATURE_SECURITY_OPT_OUT;
            continue;
        }
        if (SDL_memcmp(&GUID_IConsoleFunctionMap_InputReport, guid, sizeof(GUID)) == 0) {
            attachment->features |= GIP_FEATURE_CONSOLE_FUNCTION_MAP;
            continue;
        }
        if (SDL_memcmp(&GUID_IConsoleFunctionMap_OverflowInputReport, guid, sizeof(GUID)) == 0) {
            attachment->features |= GIP_FEATURE_CONSOLE_FUNCTION_MAP_OVERFLOW;
            continue;
        }
        if (SDL_memcmp(&GUID_IEliteButtons, guid, sizeof(GUID)) == 0) {
            attachment->features |= GIP_FEATURE_ELITE_BUTTONS;
            continue;
        }
        if (SDL_memcmp(&GUID_DynamicLatencyInput, guid, sizeof(GUID)) == 0) {
            attachment->features |= GIP_FEATURE_DYNAMIC_LATENCY_INPUT;
            continue;
        }
    }

    for (i = 0; i < metadata.num_messages; i++) {
        GIP_MessageMetadata *message = &metadata.message_metadata[i];
        if (message->type == GIP_CMD_DIRECT_MOTOR && message->length >= 9 &&
            (message->flags & GIP_MESSAGE_FLAG_DOWNSTREAM)) {
            attachment->features |= GIP_FEATURE_MOTOR_CONTROL;
        }
    }

    if (!found_expected_guid || (GIP_AttachmentIsController(attachment) && !found_controller_guid)) {
        SDL_LogDebug(SDL_LOG_CATEGORY_INPUT,
            "GIP: Controller was missing expected GUID. This controller probably won't work on an actual Xbox.");
    }

    if ((attachment->features & GIP_FEATURE_GUIDE_COLOR) &&
        !GIP_SupportsVendorMessage(attachment, GIP_CMD_GUIDE_COLOR, false))
    {
        attachment->features &= ~GIP_FEATURE_GUIDE_COLOR;
    }

    GIP_HandleQuirks(attachment);

    return GIP_SendInitSequence(attachment);
}

static bool GIP_HandleCommandSecurity(
    GIP_Attachment *attachment,
    const GIP_Header *header,
    const Uint8 *bytes,
    int num_bytes)
{
    // TODO
    SDL_LogDebug(SDL_LOG_CATEGORY_INPUT, "GIP: Unimplemented Security message");
    return false;
}

static bool GIP_HandleCommandGuideButtonStatus(
    GIP_Attachment *attachment,
    const GIP_Header *header,
    const Uint8 *bytes,
    int num_bytes)
{
    Uint64 timestamp = SDL_GetTicksNS();
    SDL_Joystick *joystick = NULL;

    if (attachment->device->device->num_joysticks < 1) {
        return true;
    }

    joystick = SDL_GetJoystickFromID(attachment->joystick);
    if (!joystick) {
        return false;
    }
    if (bytes[1] == VK_LWIN) {
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_GUIDE, (bytes[0] & 0x03) != 0);
    }

    return true;
}

static bool GIP_HandleCommandAudioControl(
    GIP_Attachment *attachment,
    const GIP_Header *header,
    const Uint8 *bytes,
    int num_bytes)
{
    // TODO
    SDL_LogDebug(SDL_LOG_CATEGORY_INPUT, "GIP: Unimplemented Audio Control message");
    return false;
}

static bool GIP_HandleCommandFirmware(
    GIP_Attachment *attachment,
    const GIP_Header *header,
    const Uint8 *bytes,
    int num_bytes)
{
    if (num_bytes < 1) {
        return false;
    }
    if (bytes[0] == 1) {
        Uint16 major, minor, build, rev;

        if (num_bytes < 14) {
            SDL_LogDebug(SDL_LOG_CATEGORY_INPUT, "GIP: Discarding too-short firmware message");

            return false;
        }
        major = bytes[6];
        major |= bytes[7] << 8;
        minor = bytes[8];
        minor |= bytes[9] << 8;
        build = bytes[10];
        build |= bytes[11] << 8;
        rev = bytes[12];
        rev |= bytes[13] << 8;

        SDL_LogDebug(SDL_LOG_CATEGORY_INPUT, "GIP: Firmware version: %d.%d.%d rev %d", major, minor, build, rev);

        attachment->firmware_major_version = major;
        attachment->firmware_minor_version = minor;

        if (attachment->device->device->vendor_id == USB_VENDOR_MICROSOFT &&
            attachment->device->device->product_id == USB_PRODUCT_XBOX_ONE_ELITE_SERIES_2)
        {
            return GIP_EnableEliteButtons(attachment);
        }
        return true;
    } else {
        SDL_LogDebug(SDL_LOG_CATEGORY_INPUT, "GIP: Unimplemented Firmware message");

        return false;
    }
}

static bool GIP_HandleCommandRawReport(
    GIP_Attachment *attachment,
    const GIP_Header *header,
    const Uint8 *bytes,
    int num_bytes)
{
    Uint64 timestamp = SDL_GetTicksNS();
    SDL_Joystick *joystick = NULL;

    if (attachment->device->device->num_joysticks < 1) {
        return true;
    }

    joystick = SDL_GetJoystickFromID(attachment->joystick);
    if (!joystick) {
        return true;
    }

    if (num_bytes < 17) {
        SDL_LogDebug(SDL_LOG_CATEGORY_INPUT, "GIP: Discarding too-short raw report");
        return false;
    }

    if ((attachment->features & GIP_FEATURE_ELITE_BUTTONS) && attachment->xbe_format == GIP_BTN_FMT_XBE2_RAW) {
        if (bytes[15] & 3) {
            SDL_SendJoystickButton(timestamp,
                joystick,
                attachment->paddle_idx,
                0);
            SDL_SendJoystickButton(timestamp,
                joystick,
                attachment->paddle_idx + 1,
                0);
            SDL_SendJoystickButton(timestamp,
                joystick,
                attachment->paddle_idx + 2,
                0);
            SDL_SendJoystickButton(timestamp,
                joystick,
                attachment->paddle_idx + 3,
                0);
        } else {
            SDL_SendJoystickButton(timestamp,
                joystick,
                attachment->paddle_idx,
                (bytes[GIP_BTN_OFFSET_XBE2] & 0x01) != 0);
            SDL_SendJoystickButton(timestamp,
                joystick,
                attachment->paddle_idx + 1,
                (bytes[GIP_BTN_OFFSET_XBE2] & 0x02) != 0);
            SDL_SendJoystickButton(timestamp,
                joystick,
                attachment->paddle_idx + 2,
                (bytes[GIP_BTN_OFFSET_XBE2] & 0x04) != 0);
            SDL_SendJoystickButton(timestamp,
                joystick,
                attachment->paddle_idx + 3,
                (bytes[GIP_BTN_OFFSET_XBE2] & 0x08) != 0);
        }
    }
    return true;
}

static bool GIP_HandleCommandHidReport(
    GIP_Attachment *attachment,
    const GIP_Header *header,
    const Uint8 *bytes,
    int num_bytes)
{
    Uint64 timestamp = SDL_GetTicksNS();
    // SDL doesn't have HID descriptor parsing, so we have to hardcode for the Chatpad descriptor instead.
    // I don't know of any other devices that emit HID reports, so this should be safe.
    if (attachment->attachment_type != GIP_TYPE_CHATPAD || !attachment->keyboard || num_bytes != 8) {
        SDL_LogDebug(SDL_LOG_CATEGORY_INPUT, "GIP: Unimplemented HID Report message");
        return false;
    }

    Uint8 modifiers = bytes[0];
    Uint8 changed_modifiers = modifiers ^ attachment->last_modifiers;
    if (changed_modifiers & 0x02) {
        if (modifiers & 0x02) {
            // SDL_SendKeyboardKey(timestamp, attachment->keyboard, 0, SDL_SCANCODE_LSHIFT, true);
        } else {
            // SDL_SendKeyboardKey(timestamp, attachment->keyboard, 0, SDL_SCANCODE_LSHIFT, false);
        }
    }
    // The chatpad has several non-ASCII characters that it sends as Alt codes
    if (changed_modifiers & 0x04) {
        if (modifiers & 0x04) {
            attachment->altcode_digit = 0;
            attachment->altcode = 0;
        } else {
            if (attachment->altcode_digit == 4) {
                char utf8[4] = {0};
                // Some Alt codes don't match their Unicode codepoint for some reason
                switch (attachment->altcode) {
                case 128:
                    SDL_UCS4ToUTF8(0x20AC, utf8);
                    break;
                case 138:
                    SDL_UCS4ToUTF8(0x0160, utf8);
                    break;
                case 140:
                    SDL_UCS4ToUTF8(0x0152, utf8);
                    break;
                case 154:
                    SDL_UCS4ToUTF8(0x0161, utf8);
                    break;
                case 156:
                    SDL_UCS4ToUTF8(0x0153, utf8);
                    break;
                default:
                    SDL_UCS4ToUTF8(attachment->altcode, utf8);
                    break;
                }
                // SDL_SendKeyboardText(utf8);
            }
            attachment->altcode_digit = -1;
            // SDL_SendKeyboardKey(timestamp, attachment->keyboard, 0, SDL_SCANCODE_NUMLOCKCLEAR, true);
            // SDL_SendKeyboardKey(timestamp, attachment->keyboard, 0, SDL_SCANCODE_NUMLOCKCLEAR, false);
        }
    }

    if (!bytes[2] && attachment->last_key) {
        if (attachment->last_key == SDL_SCANCODE_CAPSLOCK) {
            attachment->capslock = !attachment->capslock;
        }
        attachment->last_key = 0;
    } else {
        // SDL_SendKeyboardKey(timestamp, attachment->keyboard, 0, bytes[2], true);
        attachment->last_key = bytes[2];

        if ((modifiers & 0x04) && attachment->altcode_digit >= 0) {
            int digit = bytes[2] - SDL_SCANCODE_KP_1 + 1;
            if (digit < 1 || digit > 10) {
                attachment->altcode_digit = -1;
            } else {
                attachment->altcode_digit++;
                attachment->altcode *= 10;
                if (digit < 10) {
                    attachment->altcode += digit;
                }
            }
        }
    }

    attachment->last_modifiers = modifiers;
    return true;
}

static bool GIP_HandleCommandExtended(
    GIP_Attachment *attachment,
    const GIP_Header *header,
    const Uint8 *bytes,
    int num_bytes)
{
    char serial[33] = {0};

    if (num_bytes < 2) {
        return false;
    }

    switch (bytes[0]) {
    case GIP_EXTCMD_GET_SERIAL_NUMBER:
        if (bytes[1] != GIP_EXTENDED_STATUS_OK) {
            return true;
        }
        if (header->flags & GIP_FLAG_ATTACHMENT_MASK) {
            return true;
        }
        SDL_memcpy(serial, &bytes[2], SDL_min(sizeof(serial) - 1, num_bytes - 2));
        HIDAPI_SetDeviceSerial(attachment->device->device, serial);
        break;
    default:
        // TODO
        SDL_LogDebug(SDL_LOG_CATEGORY_INPUT, "GIP: Extended message type %02x", bytes[0]);
        return false;
    }

    return true;
}

static void GIP_HandleNavigationReport(
    GIP_Attachment *attachment,
    SDL_Joystick *joystick,
    Uint64 timestamp,
    const Uint8 *bytes,
    int num_bytes)
{
    if (attachment->last_input[0] != bytes[0]) {
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_START, ((bytes[0] & 0x04) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_BACK, ((bytes[0] & 0x08) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_SOUTH, ((bytes[0] & 0x10) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_EAST, ((bytes[0] & 0x20) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_WEST, ((bytes[0] & 0x40) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_NORTH, ((bytes[0] & 0x80) != 0));
    }

    if (attachment->last_input[1] != bytes[1]) {
        Uint8 hat = 0;

        if (bytes[1] & 0x01) {
            hat |= SDL_HAT_UP;
        }
        if (bytes[1] & 0x02) {
            hat |= SDL_HAT_DOWN;
        }
        if (bytes[1] & 0x04) {
            hat |= SDL_HAT_LEFT;
        }
        if (bytes[1] & 0x08) {
            hat |= SDL_HAT_RIGHT;
        }
        SDL_SendJoystickHat(timestamp, joystick, 0, hat);

        if (attachment->attachment_type == GIP_TYPE_ARCADE_STICK) {
            /* Previous */
            SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER, ((bytes[1] & 0x10) != 0));
            /* Next */
            SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_SHOULDER, ((bytes[1] & 0x20) != 0));
        } else {
            SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_SHOULDER, ((bytes[1] & 0x10) != 0));
            SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER, ((bytes[1] & 0x20) != 0));
        }
    }
}

static void GIP_HandleGamepadReport(
    GIP_Attachment *attachment,
    SDL_Joystick *joystick,
    Uint64 timestamp,
    const Uint8 *bytes,
    int num_bytes)
{
    Sint16 axis;

    SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_STICK, ((bytes[1] & 0x40) != 0));
    SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_RIGHT_STICK, ((bytes[1] & 0x80) != 0));

    axis = bytes[2];
    axis |= bytes[3] << 8;
    axis = SDL_clamp(axis, 0, 1023);
    axis = (axis - 512) * 64;
    if (axis == 32704) {
        axis = 32767;
    }
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFT_TRIGGER, axis);

    axis = bytes[4];
    axis |= bytes[5] << 8;
    axis = SDL_clamp(axis, 0, 1023);
    axis = (axis - 512) * 64;
    if (axis == 32704) {
        axis = 32767;
    }
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER, axis);

    axis = bytes[6];
    axis |= bytes[7] << 8;
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTX, axis);
    axis = bytes[8];
    axis |= bytes[9] << 8;
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTY, ~axis);
    axis = bytes[10];
    axis |= bytes[11] << 8;
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTX, axis);
    axis = bytes[12];
    axis |= bytes[13] << 8;
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTY, ~axis);
}

static void GIP_HandleArcadeStickReport(
    GIP_Attachment *attachment,
    SDL_Joystick *joystick,
    Uint64 timestamp,
    const Uint8 *bytes,
    int num_bytes)
{
    Sint16 axis;
    axis = bytes[2];
    axis |= bytes[3] << 8;
    axis = SDL_clamp(axis, 0, 1023);
    axis = (axis - 512) * 64;
    if (axis == 32704) {
        axis = 32767;
    }
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFT_TRIGGER, axis);

    axis = bytes[4];
    axis |= bytes[5] << 8;
    axis = SDL_clamp(axis, 0, 1023);
    axis = (axis - 512) * 64;
    if (axis == 32704) {
        axis = 32767;
    }
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER, axis);

    if (num_bytes >= 19) {
        /* Extra button 6 */
        SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER, (bytes[18] & 0x40) ? 32767 : -32768);
        /* Extra button 7 */
        SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFT_TRIGGER, (bytes[18] & 0x80) ? 32767 : -32768);
    }
}

static void GIP_HandleFlightStickReport(
    GIP_Attachment *attachment,
    SDL_Joystick *joystick,
    Uint64 timestamp,
    const Uint8 *bytes,
    int num_bytes)
{
    Sint16 axis;
    int i;

    if (num_bytes < 19) {
        return;
    }

    if (attachment->last_input[2] != bytes[2]) {
        /* Fire 1 and 2 */
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_LEFT_STICK, ((bytes[2] & 0x01) != 0));
        SDL_SendJoystickButton(timestamp, joystick, SDL_GAMEPAD_BUTTON_RIGHT_STICK, ((bytes[2] & 0x02) != 0));
    }
    for (i = 0; i < attachment->extra_buttons;) {
        if (attachment->last_input[i / 8 + 3] != bytes[i / 8 + 3]) {
            for (; i < attachment->extra_buttons; i++) {
                SDL_SendJoystickButton(timestamp,
                    joystick,
                    (Uint8) (attachment->extra_button_idx + i),
                    ((bytes[i / 8 + 3] & (1u << i)) != 0));
            }
        } else {
            i += 8;
        }
    }

    /* Roll, pitch and yaw are signed. Throttle and any extra axes are unsigned. All values are full-range. */
    axis = bytes[11];
    axis |= bytes[12] << 8;
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTX, axis);

    axis = bytes[13];
    axis |= bytes[14] << 8;
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFTY, axis);

    axis = bytes[15];
    axis |= bytes[16] << 8;
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_RIGHTX, axis);

    /* There are no more signed values, so skip RIGHTY */

    axis = (bytes[18] << 8) - 0x8000;
    axis |= bytes[17];
    SDL_SendJoystickAxis(timestamp, joystick, SDL_GAMEPAD_AXIS_LEFT_TRIGGER, axis);

    for (i = 0; i < attachment->extra_axes; i++) {
        if (20 + i * 2 >= num_bytes) {
            return;
        }
        axis = (bytes[20 + i * 2] << 8) - 0x8000;
        axis |= bytes[19 + i * 2];
        SDL_SendJoystickAxis(timestamp, joystick, (Uint8) (SDL_GAMEPAD_AXIS_RIGHT_TRIGGER + i), axis);
    }
}

static bool GIP_HandleLLInputReport(
    GIP_Attachment *attachment,
    const GIP_Header *header,
    const Uint8 *bytes,
    int num_bytes)
{
    Uint64 timestamp = SDL_GetTicksNS();
    SDL_Joystick *joystick = NULL;

    if (attachment->device->device->num_joysticks < 1) {
        GIP_EnsureMetadata(attachment);
        if (attachment->got_metadata != GIP_METADATA_GOT && attachment->got_metadata != GIP_METADATA_FAKED) {
            return true;
        }
    }

    joystick = SDL_GetJoystickFromID(attachment->joystick);
    if (!joystick) {
        return false;
    }

    if (attachment->device_state != GIP_STATE_START) {
        SDL_LogDebug(SDL_LOG_CATEGORY_INPUT, "GIP: Discarding early input report");
        attachment->device_state = GIP_STATE_START;
        return true;
    }

    if (num_bytes < 14) {
        SDL_LogDebug(SDL_LOG_CATEGORY_INPUT, "GIP: Discarding too-short input report");
        return false;
    }

    GIP_HandleNavigationReport(attachment, joystick, timestamp, bytes, num_bytes);

    switch (attachment->attachment_type) {
    case GIP_TYPE_GAMEPAD:
    default:
        GIP_HandleGamepadReport(attachment, joystick, timestamp, bytes, num_bytes);
        break;
    case GIP_TYPE_ARCADE_STICK:
        GIP_HandleArcadeStickReport(attachment, joystick, timestamp, bytes, num_bytes);
        break;
    case GIP_TYPE_FLIGHT_STICK:
        GIP_HandleFlightStickReport(attachment, joystick, timestamp, bytes, num_bytes);
        break;
    }

    if (attachment->features & GIP_FEATURE_ELITE_BUTTONS) {
        bool clear = false;
        if (attachment->xbe_format == GIP_BTN_FMT_XBE1 &&
            num_bytes > GIP_BTN_OFFSET_XBE1 &&
            attachment->last_input[GIP_BTN_OFFSET_XBE1] != bytes[GIP_BTN_OFFSET_XBE1] &&
            (bytes[GIP_BTN_OFFSET_XBE1] & 0x10))
        {
            SDL_SendJoystickButton(timestamp,
                joystick,
                attachment->paddle_idx,
                (bytes[GIP_BTN_OFFSET_XBE1] & 0x02) != 0);
            SDL_SendJoystickButton(timestamp,
                joystick,
                attachment->paddle_idx + 1,
                (bytes[GIP_BTN_OFFSET_XBE1] & 0x08) != 0);
            SDL_SendJoystickButton(timestamp,
                joystick,
                attachment->paddle_idx + 2,
                (bytes[GIP_BTN_OFFSET_XBE1] & 0x01) != 0);
            SDL_SendJoystickButton(timestamp,
                joystick,
                attachment->paddle_idx + 3,
                (bytes[GIP_BTN_OFFSET_XBE1] & 0x04) != 0);
        } else if ((attachment->xbe_format == GIP_BTN_FMT_XBE2_4 ||
            attachment->xbe_format == GIP_BTN_FMT_XBE2_5) &&
            num_bytes > GIP_BTN_OFFSET_XBE2)
        {
            int profile_offset = attachment->xbe_format == GIP_BTN_FMT_XBE2_4 ? 15 : 20;
            if (attachment->last_input[GIP_BTN_OFFSET_XBE2] != bytes[GIP_BTN_OFFSET_XBE2] ||
                attachment->last_input[profile_offset] != bytes[profile_offset])
            {
                if (bytes[profile_offset] & 3) {
                    clear = true;
                } else {
                    SDL_SendJoystickButton(timestamp,
                        joystick,
                        attachment->paddle_idx,
                        (bytes[GIP_BTN_OFFSET_XBE2] & 0x01) != 0);
                    SDL_SendJoystickButton(timestamp,
                        joystick,
                        attachment->paddle_idx + 1,
                        (bytes[GIP_BTN_OFFSET_XBE2] & 0x02) != 0);
                    SDL_SendJoystickButton(timestamp,
                        joystick,
                        attachment->paddle_idx + 2,
                        (bytes[GIP_BTN_OFFSET_XBE2] & 0x04) != 0);
                    SDL_SendJoystickButton(timestamp,
                        joystick,
                        attachment->paddle_idx + 3,
                        (bytes[GIP_BTN_OFFSET_XBE2] & 0x08) != 0);
                }
            }
        } else {
            clear = true;
        }
        if (clear) {
            SDL_SendJoystickButton(timestamp,
                joystick,
                attachment->paddle_idx,
                0);
            SDL_SendJoystickButton(timestamp,
                joystick,
                attachment->paddle_idx + 1,
                0);
            SDL_SendJoystickButton(timestamp,
                joystick,
                attachment->paddle_idx + 2,
                0);
            SDL_SendJoystickButton(timestamp,
                joystick,
                attachment->paddle_idx + 3,
                0);
        }
    }

    if ((attachment->features & GIP_FEATURE_CONSOLE_FUNCTION_MAP) && num_bytes >= 32) {
        int function_map_offset = -1;
        if (attachment->features & GIP_FEATURE_DYNAMIC_LATENCY_INPUT) {
            /* The dynamic latency input bytes are after the console function map */
            if (num_bytes >= 40) {
                function_map_offset = num_bytes - 26;
            }
        } else {
            function_map_offset = num_bytes - 18;
        }
        if (function_map_offset >= 14) {
            if (attachment->last_input[function_map_offset] != bytes[function_map_offset]) {
                SDL_SendJoystickButton(timestamp,
                    joystick,
                    attachment->share_button_idx,
                    (bytes[function_map_offset] & 0x01) != 0);
            }
        }
    }

    SDL_memcpy(attachment->last_input, bytes, SDL_min(num_bytes, sizeof(attachment->last_input)));

    return true;
}

static bool GIP_HandleLLStaticConfiguration(
    GIP_Attachment *attachment,
    const GIP_Header *header,
    const Uint8 *bytes,
    int num_bytes)
{
    // TODO
    SDL_LogDebug(SDL_LOG_CATEGORY_INPUT, "GIP: Unimplemented Static Configuration message");
    return false;
}

static bool GIP_HandleLLButtonInfoReport(
    GIP_Attachment *attachment,
    const GIP_Header *header,
    const Uint8 *bytes,
    int num_bytes)
{
    // TODO
    SDL_LogDebug(SDL_LOG_CATEGORY_INPUT, "GIP: Unimplemented Button Info Report message");
    return false;
}

static bool GIP_HandleLLOverflowInputReport(
    GIP_Attachment *attachment,
    const GIP_Header *header,
    const Uint8 *bytes,
    int num_bytes)
{
    // TODO
    SDL_LogDebug(SDL_LOG_CATEGORY_INPUT, "GIP: Unimplemented Overflow Input Report message");
    return false;
}

static bool GIP_HandleAudioData(
    GIP_Attachment *attachment,
    const GIP_Header *header,
    const Uint8 *bytes,
    int num_bytes)
{
    // TODO
    SDL_LogDebug(SDL_LOG_CATEGORY_INPUT, "GIP: Unimplemented Audio Data message");
    return false;
}

static bool GIP_HandleSystemMessage(
    GIP_Attachment *attachment,
    const GIP_Header *header,
    const Uint8 *bytes,
    int num_bytes)
{
    if (attachment->attachment_index > 0 && attachment->attachment_type == GIP_TYPE_UNKNOWN) {
        // XXX If we reattach to a controller after it's been initialized, it might have
        // attachments we don't know about. Try to figure out what this one is.
        if (header->message_type == GIP_CMD_HID_REPORT && num_bytes == 8) {
            if (!attachment->keyboard) {
                attachment->keyboard = (SDL_KeyboardID)(uintptr_t) attachment;
                // SDL_AddKeyboard(attachment->keyboard, "Xbox One Chatpad");
            }
            attachment->attachment_type = GIP_TYPE_CHATPAD;
            attachment->metadata.device.in_system_messages[0] |= (1u << GIP_CMD_HID_REPORT);
        }
    }
    if (!GIP_SupportsSystemMessage(attachment, header->message_type, true)) {
        SDL_LogWarn(SDL_LOG_CATEGORY_INPUT,
            "GIP: Received claimed-unsupported system message type %02x",
            header->message_type);
        return false;
    }
    switch (header->message_type) {
    case GIP_CMD_PROTO_CONTROL:
        return GIP_HandleCommandProtocolControl(attachment, header, bytes, num_bytes);
    case GIP_CMD_HELLO_DEVICE:
        return GIP_HandleCommandHelloDevice(attachment, header, bytes, num_bytes);
    case GIP_CMD_STATUS_DEVICE:
        return GIP_HandleCommandStatusDevice(attachment, header, bytes, num_bytes);
    case GIP_CMD_METADATA:
        return GIP_HandleCommandMetadataRespose(attachment, header, bytes, num_bytes);
    case GIP_CMD_SECURITY:
        return GIP_HandleCommandSecurity(attachment, header, bytes, num_bytes);
    case GIP_CMD_GUIDE_BUTTON:
        return GIP_HandleCommandGuideButtonStatus(attachment, header, bytes, num_bytes);
    case GIP_CMD_AUDIO_CONTROL:
        return GIP_HandleCommandAudioControl(attachment, header, bytes, num_bytes);
    case GIP_CMD_FIRMWARE:
        return GIP_HandleCommandFirmware(attachment, header, bytes, num_bytes);
    case GIP_CMD_HID_REPORT:
        return GIP_HandleCommandHidReport(attachment, header, bytes, num_bytes);
    case GIP_CMD_EXTENDED:
        return GIP_HandleCommandExtended(attachment, header, bytes, num_bytes);
    case GIP_AUDIO_DATA:
        return GIP_HandleAudioData(attachment, header, bytes, num_bytes);
    default:
        SDL_LogWarn(SDL_LOG_CATEGORY_INPUT,
            "GIP: Received unknown system message type %02x",
            header->message_type);
        return false;
    }
}

static GIP_Attachment *GIP_EnsureAttachment(GIP_Device *device, Uint8 attachment_index)
{
    GIP_Attachment *attachment = device->attachments[attachment_index];
    if (!attachment) {
        attachment = SDL_calloc(1, sizeof(*attachment));
        attachment->attachment_index = attachment_index;
        if (attachment_index > 0) {
            attachment->attachment_type = GIP_TYPE_UNKNOWN;
        }
        attachment->device = device;
        attachment->metadata.device.in_system_messages[0] = GIP_DEFAULT_IN_SYSTEM_MESSAGES;
        attachment->metadata.device.out_system_messages[0] = GIP_DEFAULT_OUT_SYSTEM_MESSAGES;
        device->attachments[attachment_index] = attachment;
    }
    return attachment;
}

static bool GIP_HandleMessage(
    GIP_Attachment *attachment,
    const GIP_Header *header,
    const Uint8 *bytes,
    int num_bytes)
{
    if (header->flags & GIP_FLAG_SYSTEM) {
        return GIP_HandleSystemMessage(attachment, header, bytes, num_bytes);
    } else {
        switch (header->message_type) {
        case GIP_CMD_RAW_REPORT:
            if (attachment->features & GIP_FEATURE_ELITE_BUTTONS) {
                return GIP_HandleCommandRawReport(attachment, header, bytes, num_bytes);
            }
            break;
        case GIP_LL_INPUT_REPORT:
            return GIP_HandleLLInputReport(attachment, header, bytes, num_bytes);
        case GIP_LL_STATIC_CONFIGURATION:
            return GIP_HandleLLStaticConfiguration(attachment, header, bytes, num_bytes);
        case GIP_LL_BUTTON_INFO_REPORT:
            return GIP_HandleLLButtonInfoReport(attachment, header, bytes, num_bytes);
        case GIP_LL_OVERFLOW_INPUT_REPORT:
            return GIP_HandleLLOverflowInputReport(attachment, header, bytes, num_bytes);
        }
    }
    SDL_LogWarn(SDL_LOG_CATEGORY_INPUT,
        "GIP: Received unknown vendor message type %02x",
        header->message_type);
    return false;
}

static void GIP_ReceivePacket(GIP_Device *device, const Uint8 *bytes, int num_bytes)
{
    GIP_Header header;
    int offset = 3;
    bool ok = true;
    Uint64 fragment_offset = 0;
    Uint16 bytes_remaining = 0;
    bool is_fragment;
    Uint8 attachment_index;
    GIP_Attachment *attachment;

    if (num_bytes < 5) {
        return;
    }

    header.message_type = bytes[0];
    header.flags = bytes[1];
    header.sequence_id = bytes[2];
    offset += GIP_DecodeLength(&header.length, &bytes[offset], num_bytes - offset);

    is_fragment = header.flags & GIP_FLAG_FRAGMENT;
    attachment_index = header.flags & GIP_FLAG_ATTACHMENT_MASK;
    attachment = GIP_EnsureAttachment(device, attachment_index);

#ifdef DEBUG_XBOX_PROTOCOL
    HIDAPI_DumpPacket("GIP received message: size = %d", bytes, num_bytes);
#endif

    /* Handle coalescing fragmented messages */
    if (is_fragment) {
        if (header.flags & GIP_FLAG_INIT_FRAG) {
            Uint64 total_length;
            if (attachment->fragment_message) {
                /*
                 * Reset fragment buffer if we get a new initial
                 * fragment before finishing the last message.
                 * TODO: Is this the correct behavior?
                 */
                if (attachment->fragment_data) {
                    SDL_free(attachment->fragment_data);
                    attachment->fragment_data = NULL;
                }
            }
            offset += GIP_DecodeLength(&total_length, &bytes[offset], num_bytes - offset);
            if (total_length > MAX_MESSAGE_LENGTH) {
                return;
            }
            attachment->total_length = (Uint16) total_length;
            attachment->fragment_message = header.message_type;
            if (header.length > num_bytes - offset) {
                SDL_LogWarn(SDL_LOG_CATEGORY_INPUT,
                    "GIP: Received fragment that claims to be %" SDL_PRIu64 " bytes, expected %i",
                    header.length, num_bytes - offset);
                return;
            }
            if (header.length > total_length) {
                SDL_LogWarn(SDL_LOG_CATEGORY_INPUT,
                    "GIP: Received too long fragment, %" SDL_PRIu64 " bytes, exceeds %d",
                    header.length, attachment->total_length);
                return;
            }
            attachment->fragment_data = SDL_malloc(attachment->total_length);
            SDL_memcpy(attachment->fragment_data, &bytes[offset], (size_t) header.length);
            fragment_offset = header.length;
            attachment->fragment_offset = (Uint32) fragment_offset;
            bytes_remaining = (Uint16) (attachment->total_length - fragment_offset);
        } else {
            if (header.message_type != attachment->fragment_message) {
                SDL_LogWarn(SDL_LOG_CATEGORY_INPUT,
                    "GIP: Received out of sequence message type %02x, expected %02x",
                    header.message_type, attachment->fragment_message);
                GIP_FragmentFailed(attachment, &header);
                return;
            }

            offset += GIP_DecodeLength(&fragment_offset, &bytes[offset], num_bytes - offset);
            if (fragment_offset != attachment->fragment_offset) {
                SDL_LogWarn(SDL_LOG_CATEGORY_INPUT,
                    "GIP: Received out of sequence fragment, (claimed %" SDL_PRIu64 ", expected %d)",
                    fragment_offset, attachment->fragment_offset);
                GIP_Acknowledge(device,
                    &header,
                    attachment->fragment_offset,
                    (Uint16) (attachment->total_length - attachment->fragment_offset));
                return;
            } else if (fragment_offset + header.length > attachment->total_length) {
                SDL_LogWarn(SDL_LOG_CATEGORY_INPUT,
                    "GIP: Received too long fragment, %" SDL_PRIu64 " exceeds %d",
                    fragment_offset + header.length, attachment->total_length);
                GIP_FragmentFailed(attachment, &header);
                return;
            }

            bytes_remaining = attachment->total_length - (Uint16) (fragment_offset + header.length);
            if (header.length != 0) {
                SDL_memcpy(&attachment->fragment_data[fragment_offset], &bytes[offset], (size_t) header.length);
            } else {
                ok = GIP_HandleMessage(attachment, &header, attachment->fragment_data, attachment->total_length);
                if (attachment->fragment_data) {
                    SDL_free(attachment->fragment_data);
                    attachment->fragment_data = NULL;
                }
                attachment->fragment_message = 0;
            }
            fragment_offset += header.length;
            attachment->fragment_offset = (Uint16) fragment_offset;
        }
        attachment->fragment_timer = SDL_GetTicks();
    } else if (header.length + offset > num_bytes) {
        SDL_LogWarn(SDL_LOG_CATEGORY_INPUT,
            "GIP: Received message with erroneous length (claimed %" SDL_PRIu64 ", actual %d), discarding",
            header.length + offset, num_bytes);
        return;
    } else {
        num_bytes -= offset;
        bytes += offset;
        fragment_offset = header.length;
        ok = GIP_HandleMessage(attachment, &header, bytes, num_bytes);
    }

    if (ok && (header.flags & GIP_FLAG_ACME)) {
        GIP_Acknowledge(device, &header, (Uint32) fragment_offset, bytes_remaining);
    }
}

static void HIDAPI_DriverGIP_RumbleSent(void *userdata)
{
    GIP_Attachment *ctx = (GIP_Attachment *)userdata;
    ctx->rumble_time = SDL_GetTicks();
}

static bool HIDAPI_DriverGIP_UpdateRumble(GIP_Attachment *attachment)
{
    GIP_DirectMotor motor;

    if (!(attachment->features & GIP_FEATURE_MOTOR_CONTROL)) {
        return true;
    }

    if (attachment->rumble_state == GIP_RUMBLE_STATE_QUEUED && attachment->rumble_time) {
        attachment->rumble_state = GIP_RUMBLE_STATE_BUSY;
    }

    if (attachment->rumble_state == GIP_RUMBLE_STATE_BUSY) {
        const int RUMBLE_BUSY_TIME_MS = 10;
        if (SDL_GetTicks() >= (attachment->rumble_time + RUMBLE_BUSY_TIME_MS)) {
            attachment->rumble_time = 0;
            attachment->rumble_state = GIP_RUMBLE_STATE_IDLE;
        }
    }

    if (!attachment->rumble_pending) {
        return true;
    }

    if (attachment->rumble_state != GIP_RUMBLE_STATE_IDLE) {
        return true;
    }

    // We're no longer pending, even if we fail to send the rumble below
    attachment->rumble_pending = false;

    motor.motor_bitmap = GIP_MOTOR_ALL;
    motor.left_impulse_level = attachment->left_impulse_level;
    motor.right_impulse_level = attachment->right_impulse_level;
    motor.left_vibration_level = attachment->left_vibration_level;
    motor.right_vibration_level = attachment->right_vibration_level;
    motor.duration = SDL_RUMBLE_RESEND_MS / 10 + 5; // Add a 50ms leniency, just in case
    motor.delay = 0;
    motor.repeat = 0;

    Uint8 message[9] = {0};
    SDL_memcpy(&message[1], &motor, sizeof(motor));
    if (!GIP_SendRawMessage(attachment->device,
        GIP_CMD_DIRECT_MOTOR,
        attachment->attachment_index,
        GIP_SequenceNext(attachment, GIP_CMD_DIRECT_MOTOR, false),
        message,
        sizeof(message),
        true,
        HIDAPI_DriverGIP_RumbleSent,
        attachment))
    {
        return SDL_SetError("Couldn't send rumble packet");
    }

    attachment->rumble_state = GIP_RUMBLE_STATE_QUEUED;

    return true;
}

static void HIDAPI_DriverGIP_RegisterHints(SDL_HintCallback callback, void *userdata)
{
    SDL_AddHintCallback(SDL_HINT_JOYSTICK_HIDAPI_GIP, callback, userdata);
    SDL_AddHintCallback(SDL_HINT_JOYSTICK_HIDAPI_GIP_RESET_FOR_METADATA, callback, userdata);
}

static void HIDAPI_DriverGIP_UnregisterHints(SDL_HintCallback callback, void *userdata)
{
    SDL_RemoveHintCallback(SDL_HINT_JOYSTICK_HIDAPI_GIP, callback, userdata);
    SDL_RemoveHintCallback(SDL_HINT_JOYSTICK_HIDAPI_GIP_RESET_FOR_METADATA, callback, userdata);
}

static bool HIDAPI_DriverGIP_IsEnabled(void)
{
    return SDL_GetHintBoolean(SDL_HINT_JOYSTICK_HIDAPI_GIP,
        SDL_GetHintBoolean(SDL_HINT_JOYSTICK_HIDAPI_XBOX_ONE,
            SDL_GetHintBoolean(SDL_HINT_JOYSTICK_HIDAPI_XBOX,
                SDL_GetHintBoolean(SDL_HINT_JOYSTICK_HIDAPI, SDL_HIDAPI_DEFAULT))));
}

static bool HIDAPI_DriverGIP_IsSupportedDevice(SDL_HIDAPI_Device *device, const char *name, SDL_GamepadType type, Uint16 vendor_id, Uint16 product_id, Uint16 version, int interface_number, int interface_class, int interface_subclass, int interface_protocol)
{
    // Xbox One controllers speak HID over bluetooth instead of GIP
    if (device && device->is_bluetooth) {
        return false;
    }
#if defined(SDL_PLATFORM_MACOS) && defined(SDL_JOYSTICK_MFI)
    if (!SDL_IsJoystickBluetoothXboxOne(vendor_id, product_id)) {
        // On macOS we get a shortened version of the real report and
        // you can't write output reports for wired controllers, so
        // we'll just use the GCController support instead.
        return false;
    }
#endif
    return (type == SDL_GAMEPAD_TYPE_XBOXONE);
}

static bool HIDAPI_DriverGIP_InitDevice(SDL_HIDAPI_Device *device)
{
    GIP_Device *ctx;
    GIP_Attachment *attachment;

    ctx = (GIP_Device *)SDL_calloc(1, sizeof(*ctx));
    if (!ctx) {
        return false;
    }
    ctx->device = device;
    ctx->reset_for_metadata = SDL_GetHintBoolean(SDL_HINT_JOYSTICK_HIDAPI_GIP_RESET_FOR_METADATA, false);

    attachment = GIP_EnsureAttachment(ctx, 0);
    GIP_HandleQuirks(attachment);

    if (attachment->quirks & GIP_QUIRK_NO_HELLO) {
        ctx->got_hello = true;
        GIP_EnsureMetadata(attachment);
    } else {
        ctx->hello_deadline = SDL_GetTicks() + GIP_HELLO_TIMEOUT;
    }

    device->context = ctx;
    device->type = SDL_GAMEPAD_TYPE_XBOXONE;

    return true;
}

static int HIDAPI_DriverGIP_GetDevicePlayerIndex(SDL_HIDAPI_Device *device, SDL_JoystickID instance_id)
{
    return -1;
}

static void HIDAPI_DriverGIP_SetDevicePlayerIndex(SDL_HIDAPI_Device *device, SDL_JoystickID instance_id, int player_index)
{
}

static GIP_Attachment * HIDAPI_DriverGIP_FindAttachment(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
    GIP_Device *ctx = (GIP_Device *)device->context;
    int i;

    SDL_AssertJoysticksLocked();

    for (i = 0; i < MAX_ATTACHMENTS; i++) {
        if (ctx->attachments[i] && ctx->attachments[i]->joystick == joystick->instance_id) {
            return ctx->attachments[i];
        }
    }
    return NULL;
}

static bool HIDAPI_DriverGIP_OpenJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
    GIP_Attachment *attachment = HIDAPI_DriverGIP_FindAttachment(device, joystick);
    if (!attachment) {
        return SDL_SetError("Invalid joystick");
    }

    SDL_AssertJoysticksLocked();

    attachment->left_impulse_level = 0;
    attachment->right_impulse_level = 0;
    attachment->left_vibration_level = 0;
    attachment->right_vibration_level = 0;
    attachment->rumble_state = GIP_RUMBLE_STATE_IDLE;
    attachment->rumble_time = 0;
    attachment->rumble_pending = false;
    SDL_zeroa(attachment->last_input);

    // Initialize the joystick capabilities
    joystick->nbuttons = 11;
    GIP_EnableEliteButtons(attachment);
    if (attachment->xbe_format != GIP_BTN_FMT_UNKNOWN ||
        (device->vendor_id == USB_VENDOR_MICROSOFT &&
        device->product_id == USB_PRODUCT_XBOX_ONE_ELITE_SERIES_2))
    {
        attachment->paddle_idx = (Uint8) joystick->nbuttons;
        joystick->nbuttons += 4;
    }
    if (attachment->features & GIP_FEATURE_CONSOLE_FUNCTION_MAP) {
        attachment->share_button_idx = (Uint8) joystick->nbuttons;
        joystick->nbuttons++;
    }
    if (attachment->extra_buttons > 0) {
        attachment->extra_button_idx = (Uint8) joystick->nbuttons;
        joystick->nbuttons += attachment->extra_buttons;
    }

    joystick->naxes = SDL_GAMEPAD_AXIS_COUNT;
    if (attachment->attachment_type == GIP_TYPE_FLIGHT_STICK) {
        /* Flight sticks have at least 4 axes, but only 3 are signed values, so we leave RIGHTY unused */
        joystick->naxes += attachment->extra_axes - 1;
    }

    joystick->nhats = 1;

    return true;
}

static bool HIDAPI_DriverGIP_RumbleJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint16 low_frequency_rumble, Uint16 high_frequency_rumble)
{
    GIP_Attachment *attachment = HIDAPI_DriverGIP_FindAttachment(device, joystick);
    if (!attachment) {
        return SDL_SetError("Invalid joystick");
    }

    if (!(attachment->features & GIP_FEATURE_MOTOR_CONTROL)) {
        return SDL_Unsupported();
    }

    // Magnitude is 1..100 so scale the 16-bit input here
    attachment->left_vibration_level = (Uint8)(low_frequency_rumble / 655);
    attachment->right_vibration_level = (Uint8)(high_frequency_rumble / 655);
    attachment->rumble_pending = true;

    return HIDAPI_DriverGIP_UpdateRumble(attachment);
}

static bool HIDAPI_DriverGIP_RumbleJoystickTriggers(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint16 left_rumble, Uint16 right_rumble)
{
    GIP_Attachment *attachment = HIDAPI_DriverGIP_FindAttachment(device, joystick);
    if (!attachment) {
        return SDL_SetError("Invalid joystick");
    }

    if (!(attachment->features & GIP_FEATURE_MOTOR_CONTROL) || (attachment->quirks & GIP_QUIRK_NO_IMPULSE_VIBRATION)) {
        return SDL_Unsupported();
    }

    // Magnitude is 1..100 so scale the 16-bit input here
    attachment->left_impulse_level = (Uint8)(left_rumble / 655);
    attachment->right_impulse_level = (Uint8)(right_rumble / 655);
    attachment->rumble_pending = true;

    return HIDAPI_DriverGIP_UpdateRumble(attachment);
}

static Uint32 HIDAPI_DriverGIP_GetJoystickCapabilities(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
    GIP_Attachment *attachment = HIDAPI_DriverGIP_FindAttachment(device, joystick);
    Uint32 result = 0;
    if (!attachment) {
        return 0;
    }

    if (attachment->features & GIP_FEATURE_MOTOR_CONTROL) {
        result |= SDL_JOYSTICK_CAP_RUMBLE;
        if (!(attachment->quirks & GIP_QUIRK_NO_IMPULSE_VIBRATION)) {
            result |= SDL_JOYSTICK_CAP_TRIGGER_RUMBLE;
        }
    }

    if (attachment->features & GIP_FEATURE_GUIDE_COLOR) {
        result |= SDL_JOYSTICK_CAP_RGB_LED;
    }

    return result;
}

static bool HIDAPI_DriverGIP_SetJoystickLED(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, Uint8 red, Uint8 green, Uint8 blue)
{
    GIP_Attachment *attachment = HIDAPI_DriverGIP_FindAttachment(device, joystick);
    Uint8 buffer[] = { 0x00, 0x00, 0x00, 0x00, 0x00 };

    if (!attachment) {
        return SDL_SetError("Invalid joystick");
    }

    if (!(attachment->features & GIP_FEATURE_GUIDE_COLOR)) {
        return SDL_Unsupported();
    }

    buffer[1] = 0x00; // Whiteness? Sets white intensity when RGB is 0, seems additive
    buffer[2] = red;
    buffer[3] = green;
    buffer[4] = blue;

    if (!GIP_SendVendorMessage(attachment, GIP_CMD_GUIDE_COLOR, 0, buffer, sizeof(buffer))) {
        return SDL_SetError("Couldn't send LED packet");
    }
    return true;
}

static bool HIDAPI_DriverGIP_SendJoystickEffect(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, const void *data, int size)
{
    return SDL_Unsupported();
}


static bool HIDAPI_DriverGIP_SetJoystickSensorsEnabled(SDL_HIDAPI_Device *device, SDL_Joystick *joystick, bool enabled)
{
    return SDL_Unsupported();
}

static bool HIDAPI_DriverGIP_UpdateDevice(SDL_HIDAPI_Device *device)
{
    GIP_Device *ctx = (GIP_Device *)device->context;
    Uint8 bytes[USB_PACKET_LENGTH];
    int i;
    int num_bytes;
    bool perform_reset = false;
    Uint64 timestamp;

    while ((num_bytes = SDL_hid_read_timeout(device->dev, bytes, sizeof(bytes), ctx->timeout)) > 0) {
        ctx->timeout = 0;
        GIP_ReceivePacket(ctx, bytes, num_bytes);
    }

    timestamp = SDL_GetTicks();
    if (ctx->hello_deadline && timestamp >= ctx->hello_deadline) {
        ctx->hello_deadline = 0;
        perform_reset = true;
    }
    for (i = 0; i < MAX_ATTACHMENTS; i++) {
        GIP_Attachment *attachment = ctx->attachments[i];
        if (!attachment) {
            continue;
        }
        if (attachment->fragment_message && timestamp >= attachment->fragment_timer + 1000) {
            SDL_LogWarn(SDL_LOG_CATEGORY_INPUT, "GIP: Reliable message transfer failed");
            attachment->fragment_message = 0;
        }
        if (!perform_reset &&
            attachment->got_metadata == GIP_METADATA_PENDING &&
            timestamp >= attachment->metadata_next &&
            attachment->fragment_message != GIP_CMD_METADATA)
        {
            if (attachment->metadata_retries < 3) {
                SDL_LogWarn(SDL_LOG_CATEGORY_INPUT, "GIP: Retrying metadata request");
                attachment->metadata_retries++;
                attachment->metadata_next = timestamp + 500;
                GIP_SendSystemMessage(attachment, GIP_CMD_METADATA, 0, NULL, 0);
            } else {
                perform_reset = true;
            }
        }
        if (perform_reset) {
            if (ctx->reset_for_metadata) {
                GIP_SendSetDeviceState(attachment, GIP_STATE_RESET);
            } else {
                GIP_SetMetadataDefaults(attachment);
                GIP_SendInitSequence(attachment);
            }
            perform_reset = false;
        }
        HIDAPI_DriverGIP_UpdateRumble(attachment);
    }

    if (num_bytes < 0 && device->num_joysticks > 0) {
        // Read error, device is disconnected
        for (i = 0; i < MAX_ATTACHMENTS; i++) {
            GIP_Attachment *attachment = ctx->attachments[i];
            if (attachment) {
                HIDAPI_JoystickDisconnected(device, attachment->joystick);
            }
        }
    }
    return (num_bytes >= 0);
}
static void HIDAPI_DriverGIP_CloseJoystick(SDL_HIDAPI_Device *device, SDL_Joystick *joystick)
{
}

static void HIDAPI_DriverGIP_FreeDevice(SDL_HIDAPI_Device *device)
{
    GIP_Device *context = (GIP_Device *)device->context;
    int i;

    for (i = 0; i < MAX_ATTACHMENTS; i++) {
        GIP_Attachment *attachment = context->attachments[i];
        if (!attachment) {
            continue;
        }
        if (attachment->fragment_data) {
            SDL_free(attachment->fragment_data);
            attachment->fragment_data = NULL;
        }
        if (attachment->keyboard) {
            // SDL_RemoveKeyboard(attachment->keyboard);
        }
        GIP_MetadataFree(&attachment->metadata);
        SDL_free(attachment);
        context->attachments[i] = NULL;
    }
}

SDL_HIDAPI_DeviceDriver SDL_HIDAPI_DriverGIP = {
    SDL_HINT_JOYSTICK_HIDAPI_GIP,
    true,
    HIDAPI_DriverGIP_RegisterHints,
    HIDAPI_DriverGIP_UnregisterHints,
    HIDAPI_DriverGIP_IsEnabled,
    HIDAPI_DriverGIP_IsSupportedDevice,
    HIDAPI_DriverGIP_InitDevice,
    HIDAPI_DriverGIP_GetDevicePlayerIndex,
    HIDAPI_DriverGIP_SetDevicePlayerIndex,
    HIDAPI_DriverGIP_UpdateDevice,
    HIDAPI_DriverGIP_OpenJoystick,
    HIDAPI_DriverGIP_RumbleJoystick,
    HIDAPI_DriverGIP_RumbleJoystickTriggers,
    HIDAPI_DriverGIP_GetJoystickCapabilities,
    HIDAPI_DriverGIP_SetJoystickLED,
    HIDAPI_DriverGIP_SendJoystickEffect,
    HIDAPI_DriverGIP_SetJoystickSensorsEnabled,
    HIDAPI_DriverGIP_CloseJoystick,
    HIDAPI_DriverGIP_FreeDevice,
};

#endif // SDL_JOYSTICK_HIDAPI_GIP

#endif // SDL_JOYSTICK_HIDAPI
