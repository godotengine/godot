//====== Copyright 1996-2018, Valve Corporation, All rights reserved. =======
//
// Purpose: Steam Input is a flexible input API that supports over three hundred devices including all 
//          common variants of Xbox, Playstation, Nintendo Switch Pro, and Steam Controllers.
//			For more info including a getting started guide for developers 
//			please visit: https://partner.steamgames.com/doc/features/steam_controller
//
//=============================================================================

#ifndef ISTEAMINPUT_H
#define ISTEAMINPUT_H
#ifdef _WIN32
#pragma once	
#endif

#include "steam_api_common.h"

#define STEAM_INPUT_MAX_COUNT 16

#define STEAM_INPUT_MAX_ANALOG_ACTIONS 16

#define STEAM_INPUT_MAX_DIGITAL_ACTIONS 128

#define STEAM_INPUT_MAX_ORIGINS 8

#define STEAM_INPUT_MAX_ACTIVE_LAYERS 16

// When sending an option to a specific controller handle, you can send to all devices via this command
#define STEAM_INPUT_HANDLE_ALL_CONTROLLERS UINT64_MAX

#define STEAM_INPUT_MIN_ANALOG_ACTION_DATA -1.0f
#define STEAM_INPUT_MAX_ANALOG_ACTION_DATA 1.0f

enum EInputSourceMode
{
	k_EInputSourceMode_None,
	k_EInputSourceMode_Dpad,
	k_EInputSourceMode_Buttons,
	k_EInputSourceMode_FourButtons,
	k_EInputSourceMode_AbsoluteMouse,
	k_EInputSourceMode_RelativeMouse,
	k_EInputSourceMode_JoystickMove,
	k_EInputSourceMode_JoystickMouse,
	k_EInputSourceMode_JoystickCamera,
	k_EInputSourceMode_ScrollWheel,
	k_EInputSourceMode_Trigger,
	k_EInputSourceMode_TouchMenu,
	k_EInputSourceMode_MouseJoystick,
	k_EInputSourceMode_MouseRegion,
	k_EInputSourceMode_RadialMenu,
	k_EInputSourceMode_SingleButton,
	k_EInputSourceMode_Switches
};

// Note: Please do not use action origins as a way to identify controller types. There is no
// guarantee that they will be added in a contiguous manner - use GetInputTypeForHandle instead.
// Versions of Steam that add new controller types in the future will extend this enum so if you're
// using a lookup table please check the bounds of any origins returned by Steam.
enum EInputActionOrigin
{
	// Steam Controller
	k_EInputActionOrigin_None,
	k_EInputActionOrigin_SteamController_A,
	k_EInputActionOrigin_SteamController_B,
	k_EInputActionOrigin_SteamController_X,
	k_EInputActionOrigin_SteamController_Y,
	k_EInputActionOrigin_SteamController_LeftBumper,
	k_EInputActionOrigin_SteamController_RightBumper,
	k_EInputActionOrigin_SteamController_LeftGrip,
	k_EInputActionOrigin_SteamController_RightGrip,
	k_EInputActionOrigin_SteamController_Start,
	k_EInputActionOrigin_SteamController_Back,
	k_EInputActionOrigin_SteamController_LeftPad_Touch,
	k_EInputActionOrigin_SteamController_LeftPad_Swipe,
	k_EInputActionOrigin_SteamController_LeftPad_Click,
	k_EInputActionOrigin_SteamController_LeftPad_DPadNorth,
	k_EInputActionOrigin_SteamController_LeftPad_DPadSouth,
	k_EInputActionOrigin_SteamController_LeftPad_DPadWest,
	k_EInputActionOrigin_SteamController_LeftPad_DPadEast,
	k_EInputActionOrigin_SteamController_RightPad_Touch,
	k_EInputActionOrigin_SteamController_RightPad_Swipe,
	k_EInputActionOrigin_SteamController_RightPad_Click,
	k_EInputActionOrigin_SteamController_RightPad_DPadNorth,
	k_EInputActionOrigin_SteamController_RightPad_DPadSouth,
	k_EInputActionOrigin_SteamController_RightPad_DPadWest,
	k_EInputActionOrigin_SteamController_RightPad_DPadEast,
	k_EInputActionOrigin_SteamController_LeftTrigger_Pull,
	k_EInputActionOrigin_SteamController_LeftTrigger_Click,
	k_EInputActionOrigin_SteamController_RightTrigger_Pull,
	k_EInputActionOrigin_SteamController_RightTrigger_Click,
	k_EInputActionOrigin_SteamController_LeftStick_Move,
	k_EInputActionOrigin_SteamController_LeftStick_Click,
	k_EInputActionOrigin_SteamController_LeftStick_DPadNorth,
	k_EInputActionOrigin_SteamController_LeftStick_DPadSouth,
	k_EInputActionOrigin_SteamController_LeftStick_DPadWest,
	k_EInputActionOrigin_SteamController_LeftStick_DPadEast,
	k_EInputActionOrigin_SteamController_Gyro_Move,
	k_EInputActionOrigin_SteamController_Gyro_Pitch,
	k_EInputActionOrigin_SteamController_Gyro_Yaw,
	k_EInputActionOrigin_SteamController_Gyro_Roll,
	k_EInputActionOrigin_SteamController_Reserved0,
	k_EInputActionOrigin_SteamController_Reserved1,
	k_EInputActionOrigin_SteamController_Reserved2,
	k_EInputActionOrigin_SteamController_Reserved3,
	k_EInputActionOrigin_SteamController_Reserved4,
	k_EInputActionOrigin_SteamController_Reserved5,
	k_EInputActionOrigin_SteamController_Reserved6,
	k_EInputActionOrigin_SteamController_Reserved7,
	k_EInputActionOrigin_SteamController_Reserved8,
	k_EInputActionOrigin_SteamController_Reserved9,
	k_EInputActionOrigin_SteamController_Reserved10,
	
	// PS4 Dual Shock
	k_EInputActionOrigin_PS4_X,
	k_EInputActionOrigin_PS4_Circle,
	k_EInputActionOrigin_PS4_Triangle,
	k_EInputActionOrigin_PS4_Square,
	k_EInputActionOrigin_PS4_LeftBumper,
	k_EInputActionOrigin_PS4_RightBumper,
	k_EInputActionOrigin_PS4_Options,	//Start
	k_EInputActionOrigin_PS4_Share,		//Back
	k_EInputActionOrigin_PS4_LeftPad_Touch,
	k_EInputActionOrigin_PS4_LeftPad_Swipe,
	k_EInputActionOrigin_PS4_LeftPad_Click,
	k_EInputActionOrigin_PS4_LeftPad_DPadNorth,
	k_EInputActionOrigin_PS4_LeftPad_DPadSouth,
	k_EInputActionOrigin_PS4_LeftPad_DPadWest,
	k_EInputActionOrigin_PS4_LeftPad_DPadEast,
	k_EInputActionOrigin_PS4_RightPad_Touch,
	k_EInputActionOrigin_PS4_RightPad_Swipe,
	k_EInputActionOrigin_PS4_RightPad_Click,
	k_EInputActionOrigin_PS4_RightPad_DPadNorth,
	k_EInputActionOrigin_PS4_RightPad_DPadSouth,
	k_EInputActionOrigin_PS4_RightPad_DPadWest,
	k_EInputActionOrigin_PS4_RightPad_DPadEast,
	k_EInputActionOrigin_PS4_CenterPad_Touch,
	k_EInputActionOrigin_PS4_CenterPad_Swipe,
	k_EInputActionOrigin_PS4_CenterPad_Click,
	k_EInputActionOrigin_PS4_CenterPad_DPadNorth,
	k_EInputActionOrigin_PS4_CenterPad_DPadSouth,
	k_EInputActionOrigin_PS4_CenterPad_DPadWest,
	k_EInputActionOrigin_PS4_CenterPad_DPadEast,
	k_EInputActionOrigin_PS4_LeftTrigger_Pull,
	k_EInputActionOrigin_PS4_LeftTrigger_Click,
	k_EInputActionOrigin_PS4_RightTrigger_Pull,
	k_EInputActionOrigin_PS4_RightTrigger_Click,
	k_EInputActionOrigin_PS4_LeftStick_Move,
	k_EInputActionOrigin_PS4_LeftStick_Click,
	k_EInputActionOrigin_PS4_LeftStick_DPadNorth,
	k_EInputActionOrigin_PS4_LeftStick_DPadSouth,
	k_EInputActionOrigin_PS4_LeftStick_DPadWest,
	k_EInputActionOrigin_PS4_LeftStick_DPadEast,
	k_EInputActionOrigin_PS4_RightStick_Move,
	k_EInputActionOrigin_PS4_RightStick_Click,
	k_EInputActionOrigin_PS4_RightStick_DPadNorth,
	k_EInputActionOrigin_PS4_RightStick_DPadSouth,
	k_EInputActionOrigin_PS4_RightStick_DPadWest,
	k_EInputActionOrigin_PS4_RightStick_DPadEast,
	k_EInputActionOrigin_PS4_DPad_North,
	k_EInputActionOrigin_PS4_DPad_South,
	k_EInputActionOrigin_PS4_DPad_West,
	k_EInputActionOrigin_PS4_DPad_East,
	k_EInputActionOrigin_PS4_Gyro_Move,
	k_EInputActionOrigin_PS4_Gyro_Pitch,
	k_EInputActionOrigin_PS4_Gyro_Yaw,
	k_EInputActionOrigin_PS4_Gyro_Roll,
	k_EInputActionOrigin_PS4_DPad_Move,
	k_EInputActionOrigin_PS4_Reserved1,
	k_EInputActionOrigin_PS4_Reserved2,
	k_EInputActionOrigin_PS4_Reserved3,
	k_EInputActionOrigin_PS4_Reserved4,
	k_EInputActionOrigin_PS4_Reserved5,
	k_EInputActionOrigin_PS4_Reserved6,
	k_EInputActionOrigin_PS4_Reserved7,
	k_EInputActionOrigin_PS4_Reserved8,
	k_EInputActionOrigin_PS4_Reserved9,
	k_EInputActionOrigin_PS4_Reserved10,

	// XBox One
	k_EInputActionOrigin_XBoxOne_A,
	k_EInputActionOrigin_XBoxOne_B,
	k_EInputActionOrigin_XBoxOne_X,
	k_EInputActionOrigin_XBoxOne_Y,
	k_EInputActionOrigin_XBoxOne_LeftBumper,
	k_EInputActionOrigin_XBoxOne_RightBumper,
	k_EInputActionOrigin_XBoxOne_Menu,  //Start
	k_EInputActionOrigin_XBoxOne_View,  //Back
	k_EInputActionOrigin_XBoxOne_LeftTrigger_Pull,
	k_EInputActionOrigin_XBoxOne_LeftTrigger_Click,
	k_EInputActionOrigin_XBoxOne_RightTrigger_Pull,
	k_EInputActionOrigin_XBoxOne_RightTrigger_Click,
	k_EInputActionOrigin_XBoxOne_LeftStick_Move,
	k_EInputActionOrigin_XBoxOne_LeftStick_Click,
	k_EInputActionOrigin_XBoxOne_LeftStick_DPadNorth,
	k_EInputActionOrigin_XBoxOne_LeftStick_DPadSouth,
	k_EInputActionOrigin_XBoxOne_LeftStick_DPadWest,
	k_EInputActionOrigin_XBoxOne_LeftStick_DPadEast,
	k_EInputActionOrigin_XBoxOne_RightStick_Move,
	k_EInputActionOrigin_XBoxOne_RightStick_Click,
	k_EInputActionOrigin_XBoxOne_RightStick_DPadNorth,
	k_EInputActionOrigin_XBoxOne_RightStick_DPadSouth,
	k_EInputActionOrigin_XBoxOne_RightStick_DPadWest,
	k_EInputActionOrigin_XBoxOne_RightStick_DPadEast,
	k_EInputActionOrigin_XBoxOne_DPad_North,
	k_EInputActionOrigin_XBoxOne_DPad_South,
	k_EInputActionOrigin_XBoxOne_DPad_West,
	k_EInputActionOrigin_XBoxOne_DPad_East,
	k_EInputActionOrigin_XBoxOne_DPad_Move,
	k_EInputActionOrigin_XBoxOne_LeftGrip_Lower,
	k_EInputActionOrigin_XBoxOne_LeftGrip_Upper,
	k_EInputActionOrigin_XBoxOne_RightGrip_Lower,
	k_EInputActionOrigin_XBoxOne_RightGrip_Upper,
	k_EInputActionOrigin_XBoxOne_Share, // Xbox Series X controllers only
	k_EInputActionOrigin_XBoxOne_Reserved6,
	k_EInputActionOrigin_XBoxOne_Reserved7,
	k_EInputActionOrigin_XBoxOne_Reserved8,
	k_EInputActionOrigin_XBoxOne_Reserved9,
	k_EInputActionOrigin_XBoxOne_Reserved10,

	// XBox 360
	k_EInputActionOrigin_XBox360_A,
	k_EInputActionOrigin_XBox360_B,
	k_EInputActionOrigin_XBox360_X,
	k_EInputActionOrigin_XBox360_Y,
	k_EInputActionOrigin_XBox360_LeftBumper,
	k_EInputActionOrigin_XBox360_RightBumper,
	k_EInputActionOrigin_XBox360_Start,		//Start
	k_EInputActionOrigin_XBox360_Back,		//Back
	k_EInputActionOrigin_XBox360_LeftTrigger_Pull,
	k_EInputActionOrigin_XBox360_LeftTrigger_Click,
	k_EInputActionOrigin_XBox360_RightTrigger_Pull,
	k_EInputActionOrigin_XBox360_RightTrigger_Click,
	k_EInputActionOrigin_XBox360_LeftStick_Move,
	k_EInputActionOrigin_XBox360_LeftStick_Click,
	k_EInputActionOrigin_XBox360_LeftStick_DPadNorth,
	k_EInputActionOrigin_XBox360_LeftStick_DPadSouth,
	k_EInputActionOrigin_XBox360_LeftStick_DPadWest,
	k_EInputActionOrigin_XBox360_LeftStick_DPadEast,
	k_EInputActionOrigin_XBox360_RightStick_Move,
	k_EInputActionOrigin_XBox360_RightStick_Click,
	k_EInputActionOrigin_XBox360_RightStick_DPadNorth,
	k_EInputActionOrigin_XBox360_RightStick_DPadSouth,
	k_EInputActionOrigin_XBox360_RightStick_DPadWest,
	k_EInputActionOrigin_XBox360_RightStick_DPadEast,
	k_EInputActionOrigin_XBox360_DPad_North,
	k_EInputActionOrigin_XBox360_DPad_South,
	k_EInputActionOrigin_XBox360_DPad_West,
	k_EInputActionOrigin_XBox360_DPad_East,	
	k_EInputActionOrigin_XBox360_DPad_Move,
	k_EInputActionOrigin_XBox360_Reserved1,
	k_EInputActionOrigin_XBox360_Reserved2,
	k_EInputActionOrigin_XBox360_Reserved3,
	k_EInputActionOrigin_XBox360_Reserved4,
	k_EInputActionOrigin_XBox360_Reserved5,
	k_EInputActionOrigin_XBox360_Reserved6,
	k_EInputActionOrigin_XBox360_Reserved7,
	k_EInputActionOrigin_XBox360_Reserved8,
	k_EInputActionOrigin_XBox360_Reserved9,
	k_EInputActionOrigin_XBox360_Reserved10,


	// Switch - Pro or Joycons used as a single input device.
	// This does not apply to a single joycon
	k_EInputActionOrigin_Switch_A,
	k_EInputActionOrigin_Switch_B,
	k_EInputActionOrigin_Switch_X,
	k_EInputActionOrigin_Switch_Y,
	k_EInputActionOrigin_Switch_LeftBumper,
	k_EInputActionOrigin_Switch_RightBumper,
	k_EInputActionOrigin_Switch_Plus,	//Start
	k_EInputActionOrigin_Switch_Minus,	//Back
	k_EInputActionOrigin_Switch_Capture,
	k_EInputActionOrigin_Switch_LeftTrigger_Pull,
	k_EInputActionOrigin_Switch_LeftTrigger_Click,
	k_EInputActionOrigin_Switch_RightTrigger_Pull,
	k_EInputActionOrigin_Switch_RightTrigger_Click,
	k_EInputActionOrigin_Switch_LeftStick_Move,
	k_EInputActionOrigin_Switch_LeftStick_Click,
	k_EInputActionOrigin_Switch_LeftStick_DPadNorth,
	k_EInputActionOrigin_Switch_LeftStick_DPadSouth,
	k_EInputActionOrigin_Switch_LeftStick_DPadWest,
	k_EInputActionOrigin_Switch_LeftStick_DPadEast,
	k_EInputActionOrigin_Switch_RightStick_Move,
	k_EInputActionOrigin_Switch_RightStick_Click,
	k_EInputActionOrigin_Switch_RightStick_DPadNorth,
	k_EInputActionOrigin_Switch_RightStick_DPadSouth,
	k_EInputActionOrigin_Switch_RightStick_DPadWest,
	k_EInputActionOrigin_Switch_RightStick_DPadEast,
	k_EInputActionOrigin_Switch_DPad_North,
	k_EInputActionOrigin_Switch_DPad_South,
	k_EInputActionOrigin_Switch_DPad_West,
	k_EInputActionOrigin_Switch_DPad_East,
	k_EInputActionOrigin_Switch_ProGyro_Move,  // Primary Gyro in Pro Controller, or Right JoyCon
	k_EInputActionOrigin_Switch_ProGyro_Pitch,  // Primary Gyro in Pro Controller, or Right JoyCon
	k_EInputActionOrigin_Switch_ProGyro_Yaw,  // Primary Gyro in Pro Controller, or Right JoyCon
	k_EInputActionOrigin_Switch_ProGyro_Roll,  // Primary Gyro in Pro Controller, or Right JoyCon
	k_EInputActionOrigin_Switch_DPad_Move,
	k_EInputActionOrigin_Switch_Reserved1,
	k_EInputActionOrigin_Switch_Reserved2,
	k_EInputActionOrigin_Switch_Reserved3,
	k_EInputActionOrigin_Switch_Reserved4,
	k_EInputActionOrigin_Switch_Reserved5,
	k_EInputActionOrigin_Switch_Reserved6,
	k_EInputActionOrigin_Switch_Reserved7,
	k_EInputActionOrigin_Switch_Reserved8,
	k_EInputActionOrigin_Switch_Reserved9,
	k_EInputActionOrigin_Switch_Reserved10,

	// Switch JoyCon Specific
	k_EInputActionOrigin_Switch_RightGyro_Move,  // Right JoyCon Gyro generally should correspond to Pro's single gyro
	k_EInputActionOrigin_Switch_RightGyro_Pitch,  // Right JoyCon Gyro generally should correspond to Pro's single gyro
	k_EInputActionOrigin_Switch_RightGyro_Yaw,  // Right JoyCon Gyro generally should correspond to Pro's single gyro
	k_EInputActionOrigin_Switch_RightGyro_Roll,  // Right JoyCon Gyro generally should correspond to Pro's single gyro
	k_EInputActionOrigin_Switch_LeftGyro_Move,
	k_EInputActionOrigin_Switch_LeftGyro_Pitch,
	k_EInputActionOrigin_Switch_LeftGyro_Yaw,
	k_EInputActionOrigin_Switch_LeftGyro_Roll,
	k_EInputActionOrigin_Switch_LeftGrip_Lower, // Left JoyCon SR Button
	k_EInputActionOrigin_Switch_LeftGrip_Upper, // Left JoyCon SL Button
	k_EInputActionOrigin_Switch_RightGrip_Lower,  // Right JoyCon SL Button
	k_EInputActionOrigin_Switch_RightGrip_Upper,  // Right JoyCon SR Button
	k_EInputActionOrigin_Switch_Reserved11,
	k_EInputActionOrigin_Switch_Reserved12,
	k_EInputActionOrigin_Switch_Reserved13,
	k_EInputActionOrigin_Switch_Reserved14,
	k_EInputActionOrigin_Switch_Reserved15,
	k_EInputActionOrigin_Switch_Reserved16,
	k_EInputActionOrigin_Switch_Reserved17,
	k_EInputActionOrigin_Switch_Reserved18,
	k_EInputActionOrigin_Switch_Reserved19,
	k_EInputActionOrigin_Switch_Reserved20,
	
	// Added in SDK 1.51
	k_EInputActionOrigin_PS5_X,
	k_EInputActionOrigin_PS5_Circle,
	k_EInputActionOrigin_PS5_Triangle,
	k_EInputActionOrigin_PS5_Square,
	k_EInputActionOrigin_PS5_LeftBumper,
	k_EInputActionOrigin_PS5_RightBumper,
	k_EInputActionOrigin_PS5_Option,	//Start
	k_EInputActionOrigin_PS5_Create,		//Back
	k_EInputActionOrigin_PS5_Mute,
	k_EInputActionOrigin_PS5_LeftPad_Touch,
	k_EInputActionOrigin_PS5_LeftPad_Swipe,
	k_EInputActionOrigin_PS5_LeftPad_Click,
	k_EInputActionOrigin_PS5_LeftPad_DPadNorth,
	k_EInputActionOrigin_PS5_LeftPad_DPadSouth,
	k_EInputActionOrigin_PS5_LeftPad_DPadWest,
	k_EInputActionOrigin_PS5_LeftPad_DPadEast,
	k_EInputActionOrigin_PS5_RightPad_Touch,
	k_EInputActionOrigin_PS5_RightPad_Swipe,
	k_EInputActionOrigin_PS5_RightPad_Click,
	k_EInputActionOrigin_PS5_RightPad_DPadNorth,
	k_EInputActionOrigin_PS5_RightPad_DPadSouth,
	k_EInputActionOrigin_PS5_RightPad_DPadWest,
	k_EInputActionOrigin_PS5_RightPad_DPadEast,
	k_EInputActionOrigin_PS5_CenterPad_Touch,
	k_EInputActionOrigin_PS5_CenterPad_Swipe,
	k_EInputActionOrigin_PS5_CenterPad_Click,
	k_EInputActionOrigin_PS5_CenterPad_DPadNorth,
	k_EInputActionOrigin_PS5_CenterPad_DPadSouth,
	k_EInputActionOrigin_PS5_CenterPad_DPadWest,
	k_EInputActionOrigin_PS5_CenterPad_DPadEast,
	k_EInputActionOrigin_PS5_LeftTrigger_Pull,
	k_EInputActionOrigin_PS5_LeftTrigger_Click,
	k_EInputActionOrigin_PS5_RightTrigger_Pull,
	k_EInputActionOrigin_PS5_RightTrigger_Click,
	k_EInputActionOrigin_PS5_LeftStick_Move,
	k_EInputActionOrigin_PS5_LeftStick_Click,
	k_EInputActionOrigin_PS5_LeftStick_DPadNorth,
	k_EInputActionOrigin_PS5_LeftStick_DPadSouth,
	k_EInputActionOrigin_PS5_LeftStick_DPadWest,
	k_EInputActionOrigin_PS5_LeftStick_DPadEast,
	k_EInputActionOrigin_PS5_RightStick_Move,
	k_EInputActionOrigin_PS5_RightStick_Click,
	k_EInputActionOrigin_PS5_RightStick_DPadNorth,
	k_EInputActionOrigin_PS5_RightStick_DPadSouth,
	k_EInputActionOrigin_PS5_RightStick_DPadWest,
	k_EInputActionOrigin_PS5_RightStick_DPadEast,
	k_EInputActionOrigin_PS5_DPad_North,
	k_EInputActionOrigin_PS5_DPad_South,
	k_EInputActionOrigin_PS5_DPad_West,
	k_EInputActionOrigin_PS5_DPad_East,
	k_EInputActionOrigin_PS5_Gyro_Move,
	k_EInputActionOrigin_PS5_Gyro_Pitch,
	k_EInputActionOrigin_PS5_Gyro_Yaw,
	k_EInputActionOrigin_PS5_Gyro_Roll,
	k_EInputActionOrigin_PS5_DPad_Move,
	k_EInputActionOrigin_PS5_Reserved1,
	k_EInputActionOrigin_PS5_Reserved2,
	k_EInputActionOrigin_PS5_Reserved3,
	k_EInputActionOrigin_PS5_Reserved4,
	k_EInputActionOrigin_PS5_Reserved5,
	k_EInputActionOrigin_PS5_Reserved6,
	k_EInputActionOrigin_PS5_Reserved7,
	k_EInputActionOrigin_PS5_Reserved8,
	k_EInputActionOrigin_PS5_Reserved9,
	k_EInputActionOrigin_PS5_Reserved10,
	k_EInputActionOrigin_PS5_Reserved11,
	k_EInputActionOrigin_PS5_Reserved12,
	k_EInputActionOrigin_PS5_Reserved13,
	k_EInputActionOrigin_PS5_Reserved14,
	k_EInputActionOrigin_PS5_Reserved15,
	k_EInputActionOrigin_PS5_Reserved16,
	k_EInputActionOrigin_PS5_Reserved17,
	k_EInputActionOrigin_PS5_Reserved18,
	k_EInputActionOrigin_PS5_Reserved19,
	k_EInputActionOrigin_PS5_Reserved20,

	// Added in SDK 1.53
	k_EInputActionOrigin_SteamDeck_A,
	k_EInputActionOrigin_SteamDeck_B,
	k_EInputActionOrigin_SteamDeck_X,
	k_EInputActionOrigin_SteamDeck_Y,
	k_EInputActionOrigin_SteamDeck_L1,
	k_EInputActionOrigin_SteamDeck_R1,
	k_EInputActionOrigin_SteamDeck_Menu,
	k_EInputActionOrigin_SteamDeck_View,
	k_EInputActionOrigin_SteamDeck_LeftPad_Touch,
	k_EInputActionOrigin_SteamDeck_LeftPad_Swipe,
	k_EInputActionOrigin_SteamDeck_LeftPad_Click,
	k_EInputActionOrigin_SteamDeck_LeftPad_DPadNorth,
	k_EInputActionOrigin_SteamDeck_LeftPad_DPadSouth,
	k_EInputActionOrigin_SteamDeck_LeftPad_DPadWest,
	k_EInputActionOrigin_SteamDeck_LeftPad_DPadEast,
	k_EInputActionOrigin_SteamDeck_RightPad_Touch,
	k_EInputActionOrigin_SteamDeck_RightPad_Swipe,
	k_EInputActionOrigin_SteamDeck_RightPad_Click,
	k_EInputActionOrigin_SteamDeck_RightPad_DPadNorth,
	k_EInputActionOrigin_SteamDeck_RightPad_DPadSouth,
	k_EInputActionOrigin_SteamDeck_RightPad_DPadWest,
	k_EInputActionOrigin_SteamDeck_RightPad_DPadEast,
	k_EInputActionOrigin_SteamDeck_L2_SoftPull,
	k_EInputActionOrigin_SteamDeck_L2,
	k_EInputActionOrigin_SteamDeck_R2_SoftPull,
	k_EInputActionOrigin_SteamDeck_R2,
	k_EInputActionOrigin_SteamDeck_LeftStick_Move,
	k_EInputActionOrigin_SteamDeck_L3,
	k_EInputActionOrigin_SteamDeck_LeftStick_DPadNorth,
	k_EInputActionOrigin_SteamDeck_LeftStick_DPadSouth,
	k_EInputActionOrigin_SteamDeck_LeftStick_DPadWest,
	k_EInputActionOrigin_SteamDeck_LeftStick_DPadEast,
	k_EInputActionOrigin_SteamDeck_LeftStick_Touch,
	k_EInputActionOrigin_SteamDeck_RightStick_Move,
	k_EInputActionOrigin_SteamDeck_R3,
	k_EInputActionOrigin_SteamDeck_RightStick_DPadNorth,
	k_EInputActionOrigin_SteamDeck_RightStick_DPadSouth,
	k_EInputActionOrigin_SteamDeck_RightStick_DPadWest,
	k_EInputActionOrigin_SteamDeck_RightStick_DPadEast,
	k_EInputActionOrigin_SteamDeck_RightStick_Touch,
	k_EInputActionOrigin_SteamDeck_L4,
	k_EInputActionOrigin_SteamDeck_R4,
	k_EInputActionOrigin_SteamDeck_L5,
	k_EInputActionOrigin_SteamDeck_R5,
	k_EInputActionOrigin_SteamDeck_DPad_Move,
	k_EInputActionOrigin_SteamDeck_DPad_North,
	k_EInputActionOrigin_SteamDeck_DPad_South,
	k_EInputActionOrigin_SteamDeck_DPad_West,
	k_EInputActionOrigin_SteamDeck_DPad_East,
	k_EInputActionOrigin_SteamDeck_Gyro_Move,
	k_EInputActionOrigin_SteamDeck_Gyro_Pitch,
	k_EInputActionOrigin_SteamDeck_Gyro_Yaw,
	k_EInputActionOrigin_SteamDeck_Gyro_Roll,
	k_EInputActionOrigin_SteamDeck_Reserved1,
	k_EInputActionOrigin_SteamDeck_Reserved2,
	k_EInputActionOrigin_SteamDeck_Reserved3,
	k_EInputActionOrigin_SteamDeck_Reserved4,
	k_EInputActionOrigin_SteamDeck_Reserved5,
	k_EInputActionOrigin_SteamDeck_Reserved6,
	k_EInputActionOrigin_SteamDeck_Reserved7,
	k_EInputActionOrigin_SteamDeck_Reserved8,
	k_EInputActionOrigin_SteamDeck_Reserved9,
	k_EInputActionOrigin_SteamDeck_Reserved10,
	k_EInputActionOrigin_SteamDeck_Reserved11,
	k_EInputActionOrigin_SteamDeck_Reserved12,
	k_EInputActionOrigin_SteamDeck_Reserved13,
	k_EInputActionOrigin_SteamDeck_Reserved14,
	k_EInputActionOrigin_SteamDeck_Reserved15,
	k_EInputActionOrigin_SteamDeck_Reserved16,
	k_EInputActionOrigin_SteamDeck_Reserved17,
	k_EInputActionOrigin_SteamDeck_Reserved18,
	k_EInputActionOrigin_SteamDeck_Reserved19,
	k_EInputActionOrigin_SteamDeck_Reserved20,

	k_EInputActionOrigin_Count, // If Steam has added support for new controllers origins will go here.
	k_EInputActionOrigin_MaximumPossibleValue = 32767, // Origins are currently a maximum of 16 bits.
};

enum EXboxOrigin
{
	k_EXboxOrigin_A,
	k_EXboxOrigin_B,
	k_EXboxOrigin_X,
	k_EXboxOrigin_Y,
	k_EXboxOrigin_LeftBumper,
	k_EXboxOrigin_RightBumper,
	k_EXboxOrigin_Menu,  //Start
	k_EXboxOrigin_View,  //Back
	k_EXboxOrigin_LeftTrigger_Pull,
	k_EXboxOrigin_LeftTrigger_Click,
	k_EXboxOrigin_RightTrigger_Pull,
	k_EXboxOrigin_RightTrigger_Click,
	k_EXboxOrigin_LeftStick_Move,
	k_EXboxOrigin_LeftStick_Click,
	k_EXboxOrigin_LeftStick_DPadNorth,
	k_EXboxOrigin_LeftStick_DPadSouth,
	k_EXboxOrigin_LeftStick_DPadWest,
	k_EXboxOrigin_LeftStick_DPadEast,
	k_EXboxOrigin_RightStick_Move,
	k_EXboxOrigin_RightStick_Click,
	k_EXboxOrigin_RightStick_DPadNorth,
	k_EXboxOrigin_RightStick_DPadSouth,
	k_EXboxOrigin_RightStick_DPadWest,
	k_EXboxOrigin_RightStick_DPadEast,
	k_EXboxOrigin_DPad_North,
	k_EXboxOrigin_DPad_South,
	k_EXboxOrigin_DPad_West,
	k_EXboxOrigin_DPad_East,
	k_EXboxOrigin_Count,
};

enum ESteamControllerPad
{
	k_ESteamControllerPad_Left,
	k_ESteamControllerPad_Right
};

enum EControllerHapticLocation
{
	k_EControllerHapticLocation_Left = ( 1 << k_ESteamControllerPad_Left ),
	k_EControllerHapticLocation_Right = ( 1 << k_ESteamControllerPad_Right ),
	k_EControllerHapticLocation_Both = ( 1 << k_ESteamControllerPad_Left | 1 << k_ESteamControllerPad_Right ),
};

enum EControllerHapticType
{
	k_EControllerHapticType_Off,
	k_EControllerHapticType_Tick,
	k_EControllerHapticType_Click,
};

enum ESteamInputType
{
	k_ESteamInputType_Unknown,
	k_ESteamInputType_SteamController,
	k_ESteamInputType_XBox360Controller,
	k_ESteamInputType_XBoxOneController,
	k_ESteamInputType_GenericGamepad,		// DirectInput controllers
	k_ESteamInputType_PS4Controller,
	k_ESteamInputType_AppleMFiController,	// Unused
	k_ESteamInputType_AndroidController,	// Unused
	k_ESteamInputType_SwitchJoyConPair,		// Unused
	k_ESteamInputType_SwitchJoyConSingle,	// Unused
	k_ESteamInputType_SwitchProController,
	k_ESteamInputType_MobileTouch,			// Steam Link App On-screen Virtual Controller
	k_ESteamInputType_PS3Controller,		// Currently uses PS4 Origins
	k_ESteamInputType_PS5Controller,		// Added in SDK 151
	k_ESteamInputType_SteamDeckController,	// Added in SDK 153
	k_ESteamInputType_Count,
	k_ESteamInputType_MaximumPossibleValue = 255,
};

// Individual values are used by the GetSessionInputConfigurationSettings bitmask
enum ESteamInputConfigurationEnableType
{
	k_ESteamInputConfigurationEnableType_None			= 0x0000,
	k_ESteamInputConfigurationEnableType_Playstation	= 0x0001,
	k_ESteamInputConfigurationEnableType_Xbox			= 0x0002,
	k_ESteamInputConfigurationEnableType_Generic		= 0x0004,
	k_ESteamInputConfigurationEnableType_Switch			= 0x0008,
};

// These values are passed into SetLEDColor
enum ESteamInputLEDFlag
{
	k_ESteamInputLEDFlag_SetColor,
	// Restore the LED color to the user's preference setting as set in the controller personalization menu.
	// This also happens automatically on exit of your game.  
	k_ESteamInputLEDFlag_RestoreUserDefault 
};

// These values are passed into GetGlyphPNGForActionOrigin
enum ESteamInputGlyphSize
{
	k_ESteamInputGlyphSize_Small,	// 32x32 pixels
	k_ESteamInputGlyphSize_Medium,	// 128x128 pixels
	k_ESteamInputGlyphSize_Large,	// 256x256 pixels
	k_ESteamInputGlyphSize_Count,
};

enum ESteamInputGlyphStyle
{
	// Base-styles - cannot mix
	ESteamInputGlyphStyle_Knockout 	= 0x0, // Face buttons will have colored labels/outlines on a knocked out background
										   // Rest of inputs will have white detail/borders on a knocked out background
	ESteamInputGlyphStyle_Light		= 0x1, // Black detail/borders on a white background
	ESteamInputGlyphStyle_Dark 		= 0x2, // White detail/borders on a black background

	// Modifiers
	// Default ABXY/PS equivalent glyphs have a solid fill w/ color matching the physical buttons on the device
	ESteamInputGlyphStyle_NeutralColorABXY 	= 0x10, // ABXY Buttons will match the base style color instead of their normal associated color
	ESteamInputGlyphStyle_SolidABXY 		= 0x20,	// ABXY Buttons will have a solid fill
};

enum ESteamInputActionEventType
{
	ESteamInputActionEventType_DigitalAction,
	ESteamInputActionEventType_AnalogAction,
};

// InputHandle_t is used to refer to a specific controller.
// This handle will consistently identify a controller, even if it is disconnected and re-connected
typedef uint64 InputHandle_t;

// These handles are used to refer to a specific in-game action or action set
// All action handles should be queried during initialization for performance reasons
typedef uint64 InputActionSetHandle_t;
typedef uint64 InputDigitalActionHandle_t;
typedef uint64 InputAnalogActionHandle_t;

#pragma pack( push, 1 )

struct InputAnalogActionData_t
{
	// Type of data coming from this action, this will match what got specified in the action set
	EInputSourceMode eMode;
	
	// The current state of this action; will be delta updates for mouse actions
	float x, y;
	
	// Whether or not this action is currently available to be bound in the active action set
	bool bActive;
};

struct InputDigitalActionData_t
{
	// The current state of this action; will be true if currently pressed
	bool bState;
	
	// Whether or not this action is currently available to be bound in the active action set
	bool bActive;
};

struct InputMotionData_t
{
	// Sensor-fused absolute rotation; will drift in heading
	float rotQuatX;
	float rotQuatY;
	float rotQuatZ;
	float rotQuatW;
	
	// Positional acceleration
	float posAccelX;
	float posAccelY;
	float posAccelZ;

	// Angular velocity
	float rotVelX;
	float rotVelY;
	float rotVelZ;
};

//-----------------------------------------------------------------------------
// Purpose: when callbacks are enabled this fires each time a controller action
// state changes
//-----------------------------------------------------------------------------
struct SteamInputActionEvent_t
{
	InputHandle_t controllerHandle;
	ESteamInputActionEventType eEventType;
	struct AnalogAction_t {
		InputAnalogActionHandle_t actionHandle;
		InputAnalogActionData_t analogActionData;
	};
	struct DigitalAction_t {
		InputDigitalActionHandle_t actionHandle;
		InputDigitalActionData_t digitalActionData;
	};
	union {
		AnalogAction_t analogAction;
		DigitalAction_t digitalAction;
	};
};

#pragma pack( pop )

typedef void ( *SteamInputActionEventCallbackPointer )( SteamInputActionEvent_t * );

//-----------------------------------------------------------------------------
// Purpose: Steam Input API
//-----------------------------------------------------------------------------
class ISteamInput
{
public:
	
	// Init and Shutdown must be called when starting/ending use of this interface.
	// if bExplicitlyCallRunFrame is called then you will need to manually call RunFrame
	// each frame, otherwise Steam Input will updated when SteamAPI_RunCallbacks() is called
	virtual bool Init( bool bExplicitlyCallRunFrame ) = 0;
	virtual bool Shutdown() = 0;
	
	// Set the absolute path to the Input Action Manifest file containing the in-game actions
	// and file paths to the official configurations. Used in games that bundle Steam Input
	// configurations inside of the game depot instead of using the Steam Workshop
	virtual bool SetInputActionManifestFilePath( const char *pchInputActionManifestAbsolutePath ) = 0;

	// Synchronize API state with the latest Steam Input action data available. This
	// is performed automatically by SteamAPI_RunCallbacks, but for the absolute lowest
	// possible latency, you call this directly before reading controller state. 
	// Note: This must be called from somewhere before GetConnectedControllers will
	// return any handles
	virtual void RunFrame( bool bReservedValue = true ) = 0;

	// Waits on an IPC event from Steam sent when there is new data to be fetched from
	// the data drop. Returns true when data was recievied before the timeout expires.
	// Useful for games with a dedicated input thread
	virtual bool BWaitForData( bool bWaitForever, uint32 unTimeout ) = 0;

	// Returns true if new data has been received since the last time action data was accessed
	// via GetDigitalActionData or GetAnalogActionData. The game will still need to call
	// SteamInput()->RunFrame() or SteamAPI_RunCallbacks() before this to update the data stream
	virtual bool BNewDataAvailable() = 0;

	// Enumerate currently connected Steam Input enabled devices - developers can opt in controller by type (ex: Xbox/Playstation/etc) via
	// the Steam Input settings in the Steamworks site or users can opt-in in their controller settings in Steam.
	// handlesOut should point to a STEAM_INPUT_MAX_COUNT sized array of InputHandle_t handles
	// Returns the number of handles written to handlesOut
	virtual int GetConnectedControllers( STEAM_OUT_ARRAY_COUNT( STEAM_INPUT_MAX_COUNT, Receives list of connected controllers ) InputHandle_t *handlesOut ) = 0;
	
	//-----------------------------------------------------------------------------
	// CALLBACKS
	//-----------------------------------------------------------------------------
	
	// Controller configuration loaded - these callbacks will always fire if you have
	// a handler. Note: this is called within either SteamInput()->RunFrame or by SteamAPI_RunCallbacks
	STEAM_CALL_BACK( SteamInputConfigurationLoaded_t )

	// Enable SteamInputDeviceConnected_t and SteamInputDeviceDisconnected_t callbacks.
	// Each controller that is already connected will generate a device connected
	// callback when you enable them
	virtual void EnableDeviceCallbacks() = 0;

	// Controller Connected - provides info about a single newly connected controller
	// Note: this is called within either SteamInput()->RunFrame or by SteamAPI_RunCallbacks
	STEAM_CALL_BACK( SteamInputDeviceConnected_t )

	// Controller Disconnected - provides info about a single disconnected controller
	// Note: this is called within either SteamInput()->RunFrame or by SteamAPI_RunCallbacks
	STEAM_CALL_BACK( SteamInputDeviceDisconnected_t )

	// Enable SteamInputActionEvent_t callbacks. Directly calls your callback function
	// for lower latency than standard Steam callbacks. Supports one callback at a time.
	// Note: this is called within either SteamInput()->RunFrame or by SteamAPI_RunCallbacks
	virtual void EnableActionEventCallbacks( SteamInputActionEventCallbackPointer pCallback ) = 0;

	//-----------------------------------------------------------------------------
	// ACTION SETS
	//-----------------------------------------------------------------------------

	// Lookup the handle for an Action Set. Best to do this once on startup, and store the handles for all future API calls.
	virtual InputActionSetHandle_t GetActionSetHandle( const char *pszActionSetName ) = 0;
	
	// Reconfigure the controller to use the specified action set (ie 'Menu', 'Walk' or 'Drive')
	// This is cheap, and can be safely called repeatedly. It's often easier to repeatedly call it in
	// your state loops, instead of trying to place it in all of your state transitions.
	virtual void ActivateActionSet( InputHandle_t inputHandle, InputActionSetHandle_t actionSetHandle ) = 0;
	virtual InputActionSetHandle_t GetCurrentActionSet( InputHandle_t inputHandle ) = 0;

	// ACTION SET LAYERS
	virtual void ActivateActionSetLayer( InputHandle_t inputHandle, InputActionSetHandle_t actionSetLayerHandle ) = 0;
	virtual void DeactivateActionSetLayer( InputHandle_t inputHandle, InputActionSetHandle_t actionSetLayerHandle ) = 0;
	virtual void DeactivateAllActionSetLayers( InputHandle_t inputHandle ) = 0;

	// Enumerate currently active layers.
	// handlesOut should point to a STEAM_INPUT_MAX_ACTIVE_LAYERS sized array of InputActionSetHandle_t handles
	// Returns the number of handles written to handlesOut
	virtual int GetActiveActionSetLayers( InputHandle_t inputHandle, STEAM_OUT_ARRAY_COUNT( STEAM_INPUT_MAX_ACTIVE_LAYERS, Receives list of active layers ) InputActionSetHandle_t *handlesOut ) = 0;

	//-----------------------------------------------------------------------------
	// ACTIONS
	//-----------------------------------------------------------------------------

	// Lookup the handle for a digital action. Best to do this once on startup, and store the handles for all future API calls.
	virtual InputDigitalActionHandle_t GetDigitalActionHandle( const char *pszActionName ) = 0;
	
	// Returns the current state of the supplied digital game action
	virtual InputDigitalActionData_t GetDigitalActionData( InputHandle_t inputHandle, InputDigitalActionHandle_t digitalActionHandle ) = 0;
	
	// Get the origin(s) for a digital action within an action set. Returns the number of origins supplied in originsOut. Use this to display the appropriate on-screen prompt for the action.
	// originsOut should point to a STEAM_INPUT_MAX_ORIGINS sized array of EInputActionOrigin handles. The EInputActionOrigin enum will get extended as support for new controller controllers gets added to
	// the Steam client and will exceed the values from this header, please check bounds if you are using a look up table.
	virtual int GetDigitalActionOrigins( InputHandle_t inputHandle, InputActionSetHandle_t actionSetHandle, InputDigitalActionHandle_t digitalActionHandle, STEAM_OUT_ARRAY_COUNT( STEAM_INPUT_MAX_ORIGINS, Receives list of action origins ) EInputActionOrigin *originsOut ) = 0;
	
	// Returns a localized string (from Steam's language setting) for the user-facing action name corresponding to the specified handle
	virtual const char *GetStringForDigitalActionName( InputDigitalActionHandle_t eActionHandle ) = 0;

	// Lookup the handle for an analog action. Best to do this once on startup, and store the handles for all future API calls.
	virtual InputAnalogActionHandle_t GetAnalogActionHandle( const char *pszActionName ) = 0;
	
	// Returns the current state of these supplied analog game action
	virtual InputAnalogActionData_t GetAnalogActionData( InputHandle_t inputHandle, InputAnalogActionHandle_t analogActionHandle ) = 0;

	// Get the origin(s) for an analog action within an action set. Returns the number of origins supplied in originsOut. Use this to display the appropriate on-screen prompt for the action.
	// originsOut should point to a STEAM_INPUT_MAX_ORIGINS sized array of EInputActionOrigin handles. The EInputActionOrigin enum will get extended as support for new controller controllers gets added to
	// the Steam client and will exceed the values from this header, please check bounds if you are using a look up table.
	virtual int GetAnalogActionOrigins( InputHandle_t inputHandle, InputActionSetHandle_t actionSetHandle, InputAnalogActionHandle_t analogActionHandle, STEAM_OUT_ARRAY_COUNT( STEAM_INPUT_MAX_ORIGINS, Receives list of action origins ) EInputActionOrigin *originsOut ) = 0;

	// Get a local path to a PNG file for the provided origin's glyph. 
	virtual const char *GetGlyphPNGForActionOrigin( EInputActionOrigin eOrigin, ESteamInputGlyphSize eSize, uint32 unFlags ) = 0;

	// Get a local path to a SVG file for the provided origin's glyph. 
	virtual const char *GetGlyphSVGForActionOrigin( EInputActionOrigin eOrigin, uint32 unFlags ) = 0;

	// Get a local path to an older, Big Picture Mode-style PNG file for a particular origin
	virtual const char *GetGlyphForActionOrigin_Legacy( EInputActionOrigin eOrigin ) = 0;
	
	// Returns a localized string (from Steam's language setting) for the specified origin.
	virtual const char *GetStringForActionOrigin( EInputActionOrigin eOrigin ) = 0;

	// Returns a localized string (from Steam's language setting) for the user-facing action name corresponding to the specified handle
	virtual const char *GetStringForAnalogActionName( InputAnalogActionHandle_t eActionHandle ) = 0;

	// Stop analog momentum for the action if it is a mouse action in trackball mode
	virtual void StopAnalogActionMomentum( InputHandle_t inputHandle, InputAnalogActionHandle_t eAction ) = 0;

	// Returns raw motion data from the specified device
	virtual InputMotionData_t GetMotionData( InputHandle_t inputHandle ) = 0;

	//-----------------------------------------------------------------------------
	// OUTPUTS
	//-----------------------------------------------------------------------------

	// Trigger a vibration event on supported controllers - Steam will translate these commands into haptic pulses for Steam Controllers
	virtual void TriggerVibration( InputHandle_t inputHandle, unsigned short usLeftSpeed, unsigned short usRightSpeed ) = 0;

	// Trigger a vibration event on supported controllers including Xbox trigger impulse rumble - Steam will translate these commands into haptic pulses for Steam Controllers
	virtual void TriggerVibrationExtended( InputHandle_t inputHandle, unsigned short usLeftSpeed, unsigned short usRightSpeed, unsigned short usLeftTriggerSpeed, unsigned short usRightTriggerSpeed ) = 0;

	// Send a haptic pulse, works on Steam Deck and Steam Controller devices
	virtual void TriggerSimpleHapticEvent( InputHandle_t inputHandle, EControllerHapticLocation eHapticLocation, uint8 nIntensity, char nGainDB, uint8 nOtherIntensity, char nOtherGainDB ) = 0;

	// Set the controller LED color on supported controllers. nFlags is a bitmask of values from ESteamInputLEDFlag - 0 will default to setting a color. Steam will handle
	// the behavior on exit of your program so you don't need to try restore the default as you are shutting down
	virtual void SetLEDColor( InputHandle_t inputHandle, uint8 nColorR, uint8 nColorG, uint8 nColorB, unsigned int nFlags ) = 0;

	// Trigger a haptic pulse on a Steam Controller - if you are approximating rumble you may want to use TriggerVibration instead.
	// Good uses for Haptic pulses include chimes, noises, or directional gameplay feedback (taking damage, footstep locations, etc).
	virtual void Legacy_TriggerHapticPulse( InputHandle_t inputHandle, ESteamControllerPad eTargetPad, unsigned short usDurationMicroSec ) = 0;

	// Trigger a haptic pulse with a duty cycle of usDurationMicroSec / usOffMicroSec, unRepeat times. If you are approximating rumble you may want to use TriggerVibration instead.
	// nFlags is currently unused and reserved for future use.
	virtual void Legacy_TriggerRepeatedHapticPulse( InputHandle_t inputHandle, ESteamControllerPad eTargetPad, unsigned short usDurationMicroSec, unsigned short usOffMicroSec, unsigned short unRepeat, unsigned int nFlags ) = 0;

	//-----------------------------------------------------------------------------
	// Utility functions available without using the rest of Steam Input API
	//-----------------------------------------------------------------------------

	// Invokes the Steam overlay and brings up the binding screen if the user is using Big Picture Mode
	// If the user is not in Big Picture Mode it will open up the binding in a new window
	virtual bool ShowBindingPanel( InputHandle_t inputHandle ) = 0;

	// Returns the input type for a particular handle - unlike EInputActionOrigin which update with Steam and may return unrecognized values
	// ESteamInputType will remain static and only return valid values from your SDK version 
	virtual ESteamInputType GetInputTypeForHandle( InputHandle_t inputHandle ) = 0;

	// Returns the associated controller handle for the specified emulated gamepad - can be used with the above 2 functions
	// to identify controllers presented to your game over Xinput. Returns 0 if the Xinput index isn't associated with Steam Input
	virtual InputHandle_t GetControllerForGamepadIndex( int nIndex ) = 0;

	// Returns the associated gamepad index for the specified controller, if emulating a gamepad or -1 if not associated with an Xinput index
	virtual int GetGamepadIndexForController( InputHandle_t ulinputHandle ) = 0;
	
	// Returns a localized string (from Steam's language setting) for the specified Xbox controller origin.
	virtual const char *GetStringForXboxOrigin( EXboxOrigin eOrigin ) = 0;

	// Get a local path to art for on-screen glyph for a particular Xbox controller origin
	virtual const char *GetGlyphForXboxOrigin( EXboxOrigin eOrigin ) = 0;

	// Get the equivalent ActionOrigin for a given Xbox controller origin this can be chained with GetGlyphForActionOrigin to provide future proof glyphs for
	// non-Steam Input API action games. Note - this only translates the buttons directly and doesn't take into account any remapping a user has made in their configuration
	virtual EInputActionOrigin GetActionOriginFromXboxOrigin( InputHandle_t inputHandle, EXboxOrigin eOrigin ) = 0;

	// Convert an origin to another controller type - for inputs not present on the other controller type this will return k_EInputActionOrigin_None
	// When a new input type is added you will be able to pass in k_ESteamInputType_Unknown and the closest origin that your version of the SDK recognized will be returned
	// ex: if a Playstation 5 controller was released this function would return Playstation 4 origins.
	virtual EInputActionOrigin TranslateActionOrigin( ESteamInputType eDestinationInputType, EInputActionOrigin eSourceOrigin ) = 0;

	// Get the binding revision for a given device. Returns false if the handle was not valid or if a mapping is not yet loaded for the device
	virtual bool GetDeviceBindingRevision( InputHandle_t inputHandle, int *pMajor, int *pMinor ) = 0;

	// Get the Steam Remote Play session ID associated with a device, or 0 if there is no session associated with it
	// See isteamremoteplay.h for more information on Steam Remote Play sessions
	virtual uint32 GetRemotePlaySessionID( InputHandle_t inputHandle ) = 0;

	// Get a bitmask of the Steam Input Configuration types opted in for the current session. Returns ESteamInputConfigurationEnableType values.?	
	// Note: user can override the settings from the Steamworks Partner site so the returned values may not exactly match your default configuration
	virtual uint16 GetSessionInputConfigurationSettings() = 0;
};

#define STEAMINPUT_INTERFACE_VERSION "SteamInput006"

// Global interface accessor
inline ISteamInput *SteamInput();
STEAM_DEFINE_USER_INTERFACE_ACCESSOR( ISteamInput *, SteamInput, STEAMINPUT_INTERFACE_VERSION );

#if defined( VALVE_CALLBACK_PACK_SMALL )
#pragma pack( push, 4 )
#elif defined( VALVE_CALLBACK_PACK_LARGE )
#pragma pack( push, 8 )
#else
#error steam_api_common.h should define VALVE_CALLBACK_PACK_xxx
#endif 

//-----------------------------------------------------------------------------
// Purpose: called when a new controller has been connected, will fire once
// per controller if multiple new controllers connect in the same frame
//-----------------------------------------------------------------------------
struct SteamInputDeviceConnected_t
{
	enum { k_iCallback = k_iSteamControllerCallbacks + 1 };
	InputHandle_t		m_ulConnectedDeviceHandle;	// Handle for device
};

//-----------------------------------------------------------------------------
// Purpose: called when a new controller has been connected, will fire once
// per controller if multiple new controllers connect in the same frame
//-----------------------------------------------------------------------------
struct SteamInputDeviceDisconnected_t
{
	enum { k_iCallback = k_iSteamControllerCallbacks + 2 };
	InputHandle_t		m_ulDisconnectedDeviceHandle;	// Handle for device
};

//-----------------------------------------------------------------------------
// Purpose: called when a controller configuration has been loaded, will fire once
// per controller per focus change for Steam Input enabled controllers
//-----------------------------------------------------------------------------
struct SteamInputConfigurationLoaded_t
{
	enum { k_iCallback = k_iSteamControllerCallbacks + 3 };
	AppId_t			m_unAppID;
	InputHandle_t	m_ulDeviceHandle;		// Handle for device
	CSteamID		m_ulMappingCreator;		// May differ from local user when using
											// an unmodified community or official config
	uint32			m_unMajorRevision;		// Binding revision from In-game Action File. 
											// Same value as queried by GetDeviceBindingRevision
	uint32			m_unMinorRevision;
	bool			m_bUsesSteamInputAPI;	// Does the configuration contain any Analog/Digital actions?
	bool			m_bUsesGamepadAPI;		// Does the configuration contain any Xinput bindings?
};

#pragma pack( pop )

#endif // ISTEAMINPUT_H