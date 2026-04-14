/*
  Simple DirectMedia Layer
  Copyright (C) 2020 Valve Corporation

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
#ifndef _CONTROLLER_STRUCTS_
#define _CONTROLLER_STRUCTS_

#pragma pack(1)

#define HID_FEATURE_REPORT_BYTES 64

// Header for all host <==> target messages
typedef struct
{
	unsigned char type;
	unsigned char length;
} FeatureReportHeader;

// Generic controller settings structure
typedef struct
{
	unsigned char settingNum;
	unsigned short settingValue;
} ControllerSetting;

// Generic controller attribute structure
typedef struct
{
	unsigned char attributeTag;
	uint32_t attributeValue;
} ControllerAttribute;

// Generic controller settings structure
typedef struct
{
	ControllerSetting settings[ ( HID_FEATURE_REPORT_BYTES - sizeof( FeatureReportHeader ) ) / sizeof( ControllerSetting ) ];
} MsgSetSettingsValues, MsgGetSettingsValues, MsgGetSettingsDefaults, MsgGetSettingsMaxs;

// Generic controller settings structure
typedef struct
{
	ControllerAttribute attributes[ ( HID_FEATURE_REPORT_BYTES - sizeof( FeatureReportHeader ) ) / sizeof( ControllerAttribute ) ];
} MsgGetAttributes;

typedef struct
{
	unsigned char attributeTag;
	char attributeValue[20];
} MsgGetStringAttribute;

typedef struct
{
	unsigned char mode;
} MsgSetControllerMode;

// Trigger a haptic pulse
typedef struct {
	unsigned char which_pad;
	unsigned short pulse_duration;
	unsigned short pulse_interval;
	unsigned short pulse_count;
	short dBgain;
	unsigned char priority;
} MsgFireHapticPulse;

typedef struct {
	uint8_t mode;
} MsgHapticSetMode;

typedef enum {
	HAPTIC_TYPE_OFF,
	HAPTIC_TYPE_TICK,
	HAPTIC_TYPE_CLICK,
	HAPTIC_TYPE_TONE,
	HAPTIC_TYPE_RUMBLE,
	HAPTIC_TYPE_NOISE,
	HAPTIC_TYPE_SCRIPT,
	HAPTIC_TYPE_LOG_SWEEP,
} haptic_type_t;

typedef enum {
	HAPTIC_INTENSITY_SYSTEM,
	HAPTIC_INTENSITY_SHORT,
	HAPTIC_INTENSITY_MEDIUM,
	HAPTIC_INTENSITY_LONG,
	HAPTIC_INTENSITY_INSANE,
} haptic_intensity_t;

typedef struct {
	uint8_t side; 				// 0x01 = L, 0x02 = R, 0x03 = Both
	uint8_t cmd; 				// 0 = Off, 1 = tick, 2 = click, 3 = tone, 4 = rumble, 5 =
								// rumble_noise, 6 = script, 7 = sweep,
	uint8_t ui_intensity; 		// 0-4 (0 = default)
	int8_t dBgain; 				// dB Can be positive (reasonable clipping / limiting will apply)
	uint16_t freq; 				// Frequency of tone (if applicable)
	int16_t dur_ms; 			// Duration of tone / rumble (if applicable) (neg = infinite)

	uint16_t noise_intensity;
	uint16_t lfo_freq; 			// Drives both tone and rumble geneators
	uint8_t lfo_depth; 			// percentage, typically 100
	uint8_t rand_tone_gain; 	// Randomize each LFO cycle's gain
	uint8_t script_id; 			// Used w/ dBgain for scripted haptics

	uint16_t lss_start_freq;	// Used w/ Log Sine Sweep
	uint16_t lss_end_freq;		// Ditto
} MsgTriggerHaptic;

typedef struct {
	uint8_t unRumbleType;
	uint16_t unIntensity;
	uint16_t unLeftMotorSpeed;
	uint16_t unRightMotorSpeed;
	int8_t nLeftGain;
	int8_t nRightGain;
} MsgSimpleRumbleCmd;

// This is the only message struct that application code should use to interact with feature request messages. Any new
// messages should be added to the union. The structures defined here should correspond to the ones defined in
// ValveDeviceCore.cpp.
//
typedef struct
{
	FeatureReportHeader header;
	union
	{
		MsgSetSettingsValues setSettingsValues;
		MsgGetSettingsValues getSettingsValues;
		MsgGetSettingsMaxs getSettingsMaxs;
		MsgGetSettingsDefaults getSettingsDefaults;
		MsgGetAttributes getAttributes;
		MsgSetControllerMode controllerMode;
		MsgFireHapticPulse fireHapticPulse;
		MsgGetStringAttribute getStringAttribute;
		MsgHapticSetMode hapticMode;
		MsgTriggerHaptic triggerHaptic;
		MsgSimpleRumbleCmd simpleRumble;
	} payload;

} FeatureReportMsg;

// Roll this version forward anytime that you are breaking compatibility of existing
// message types within ValveInReport_t or the header itself.  Hopefully this should
// be super rare and instead you should just add new message payloads to the union,
// or just add fields to the end of existing payload structs which is expected to be 
// safe in all code consuming these as they should just consume/copy up to the prior size 
// they were aware of when processing.
#define k_ValveInReportMsgVersion 0x01

typedef enum
{
	ID_CONTROLLER_STATE = 1,
	ID_CONTROLLER_DEBUG = 2,
	ID_CONTROLLER_WIRELESS = 3,
	ID_CONTROLLER_STATUS = 4,
	ID_CONTROLLER_DEBUG2 = 5,
	ID_CONTROLLER_SECONDARY_STATE = 6,
	ID_CONTROLLER_BLE_STATE = 7,
	ID_CONTROLLER_DECK_STATE = 9,
	ID_CONTROLLER_MSG_COUNT
} ValveInReportMessageIDs; 

typedef struct 
{
	unsigned short unReportVersion;
	
	unsigned char ucType;
	unsigned char ucLength;
	
} ValveInReportHeader_t;

// State payload
typedef struct 
{
	// If packet num matches that on your prior call, then the controller state hasn't been changed since 
	// your last call and there is no need to process it
	Uint32 unPacketNum;
	
	// Button bitmask and trigger data.
	union
	{
		Uint64 ulButtons;
		struct
		{
			unsigned char _pad0[3];
			unsigned char nLeft;
			unsigned char nRight;
			unsigned char _pad1[3];
		} Triggers;
	} ButtonTriggerData;
	
	// Left pad coordinates
	short sLeftPadX;
	short sLeftPadY;
	
	// Right pad coordinates
	short sRightPadX;
	short sRightPadY;
	
	// This is redundant, packed above, but still sent over wired
	unsigned short sTriggerL;
	unsigned short sTriggerR;

	// FIXME figure out a way to grab this stuff over wireless
	short sAccelX;
	short sAccelY;
	short sAccelZ;
	
	short sGyroX;
	short sGyroY;
	short sGyroZ;
	
	short sGyroQuatW;
	short sGyroQuatX;
	short sGyroQuatY;
	short sGyroQuatZ;

} ValveControllerStatePacket_t;

// BLE State payload this has to be re-formatted from the normal state because BLE controller shows up as 
//a HID device and we don't want to send all the optional parts of the message. Keep in sync with struct above.
typedef struct
{
	// If packet num matches that on your prior call, then the controller state hasn't been changed since 
	// your last call and there is no need to process it
	Uint32 unPacketNum;

	// Button bitmask and trigger data.
	union
	{
		Uint64 ulButtons;
		struct
		{
			unsigned char _pad0[3];
			unsigned char nLeft;
			unsigned char nRight;
			unsigned char _pad1[3];
		} Triggers;
	} ButtonTriggerData;

	// Left pad coordinates
	short sLeftPadX;
	short sLeftPadY;

	// Right pad coordinates
	short sRightPadX;
	short sRightPadY;

	//This mimcs how the dongle reconstitutes HID packets, there will be 0-4 shorts depending on gyro mode
	unsigned char ucGyroDataType; //TODO could maybe find some unused bits in the button field for this info (is only 2bits)
	short sGyro[4];

} ValveControllerBLEStatePacket_t;

// Define a payload for reporting debug information
typedef struct
{
	// Left pad coordinates
	short sLeftPadX;
	short sLeftPadY;

	// Right pad coordinates
	short sRightPadX;
	short sRightPadY;

	// Left mouse deltas
	short sLeftPadMouseDX;
	short sLeftPadMouseDY;

	// Right mouse deltas
	short sRightPadMouseDX;
	short sRightPadMouseDY;
	
	// Left mouse filtered deltas
	short sLeftPadMouseFilteredDX;
	short sLeftPadMouseFilteredDY;

	// Right mouse filtered deltas
	short sRightPadMouseFilteredDX;
	short sRightPadMouseFilteredDY;
	
	// Pad Z values
	unsigned char ucLeftZ;
	unsigned char ucRightZ;
	
	// FingerPresent
	unsigned char ucLeftFingerPresent;
	unsigned char ucRightFingerPresent;
	
	// Timestamps
	unsigned char ucLeftTimestamp;
	unsigned char ucRightTimestamp;
	
	// Double tap state
	unsigned char ucLeftTapState;
	unsigned char ucRightTapState;
	
	unsigned int unDigitalIOStates0;
	unsigned int unDigitalIOStates1;
	
} ValveControllerDebugPacket_t;

typedef struct
{
	unsigned char ucPadNum;
	unsigned char ucPad[3]; // need Data to be word aligned
	short Data[20];
	unsigned short unNoise;
} ValveControllerTrackpadImage_t;

typedef struct
{
	unsigned char ucPadNum;
	unsigned char ucOffset;
	unsigned char ucPad[2]; // need Data to be word aligned
	short rgData[28];
} ValveControllerRawTrackpadImage_t;

// Payload for wireless metadata
typedef struct 
{
	unsigned char ucEventType;
} SteamControllerWirelessEvent_t;

typedef struct 
{
	// Current packet number.
    unsigned int unPacketNum;
	
	// Event codes and state information.
    unsigned short sEventCode;
    unsigned short unStateFlags;

    // Current battery voltage (mV).
    unsigned short sBatteryVoltage;
	
	// Current battery level (0-100).
	unsigned char ucBatteryLevel;
} SteamControllerStatusEvent_t;

// Deck State payload
typedef struct
{
	// If packet num matches that on your prior call, then the controller
	// state hasn't been changed since your last call and there is no need to
	// process it
	Uint32 unPacketNum;

	// Button bitmask and trigger data.
	union
	{
		Uint64 ulButtons;
		struct
		{
			Uint32 ulButtonsL;
			Uint32 ulButtonsH;
		};
	};

	// Left pad coordinates
	short sLeftPadX;
	short sLeftPadY;

	// Right pad coordinates
	short sRightPadX;
	short sRightPadY;

	// Accelerometer values
	short sAccelX;
	short sAccelY;
	short sAccelZ;

	// Gyroscope values
	short sGyroX;
	short sGyroY;
	short sGyroZ;

	// Gyro quaternions
	short sGyroQuatW;
	short sGyroQuatX;
	short sGyroQuatY;
	short sGyroQuatZ;

	// Uncalibrated trigger values
	unsigned short sTriggerRawL;
	unsigned short sTriggerRawR;

	// Left stick values
	short sLeftStickX;
	short sLeftStickY;

	// Right stick values
	short sRightStickX;
	short sRightStickY;

	// Touchpad pressures
	unsigned short sPressurePadLeft;
	unsigned short sPressurePadRight;
} SteamDeckStatePacket_t;

typedef struct
{
	ValveInReportHeader_t header;
	
	union
	{
		ValveControllerStatePacket_t controllerState;
		ValveControllerBLEStatePacket_t controllerBLEState;
		ValveControllerDebugPacket_t debugState;
		ValveControllerTrackpadImage_t padImage;
		ValveControllerRawTrackpadImage_t rawPadImage;
		SteamControllerWirelessEvent_t wirelessEvent;
		SteamControllerStatusEvent_t statusEvent;
		SteamDeckStatePacket_t deckState;
	} payload;
	
} ValveInReport_t;


// Enumeration for BLE packet protocol
enum EBLEPacketReportNums
{
	// Skipping past 2-3 because they are escape characters in Uart protocol
	k_EBLEReportState = 4,
	k_EBLEReportStatus = 5,
};


// Enumeration of data chunks in BLE state packets
enum EBLEOptionDataChunksBitmask
{
	// First byte upper nibble
	k_EBLEButtonChunk1 = 0x10,
	k_EBLEButtonChunk2 = 0x20,
	k_EBLEButtonChunk3 = 0x40,
	k_EBLELeftJoystickChunk = 0x80,

	// Second full byte
	k_EBLELeftTrackpadChunk = 0x100,
	k_EBLERightTrackpadChunk = 0x200,
	k_EBLEIMUAccelChunk = 0x400,
	k_EBLEIMUGyroChunk = 0x800,
	k_EBLEIMUQuatChunk = 0x1000,
};

#pragma pack()

#endif // _CONTROLLER_STRUCTS
