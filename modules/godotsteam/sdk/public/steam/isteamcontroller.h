//====== Copyright 1996-2018, Valve Corporation, All rights reserved. =======
//    Note: The older ISteamController interface has been deprecated in favor of ISteamInput - this interface
//			was updated in this SDK but will be removed from future SDK's. The Steam Client will retain
//			compatibility with the older interfaces so your any existing integrations should be unaffected.
//
// Purpose: Steam Input is a flexible input API that supports over three hundred devices including all 
//          common variants of Xbox, Playstation, Nintendo Switch Pro, and Steam Controllers.
//			For more info including a getting started guide for developers 
//			please visit: https://partner.steamgames.com/doc/features/steam_controller
//
//=============================================================================

#ifndef ISTEAMCONTROLLER_H
#define ISTEAMCONTROLLER_H
#ifdef _WIN32
#pragma once
#endif

#include "steam_api_common.h"
#include "isteaminput.h"

#define STEAM_CONTROLLER_MAX_COUNT 16

#define STEAM_CONTROLLER_MAX_ANALOG_ACTIONS 16

#define STEAM_CONTROLLER_MAX_DIGITAL_ACTIONS 128

#define STEAM_CONTROLLER_MAX_ORIGINS 8

#define STEAM_CONTROLLER_MAX_ACTIVE_LAYERS 16

// When sending an option to a specific controller handle, you can send to all controllers via this command
#define STEAM_CONTROLLER_HANDLE_ALL_CONTROLLERS UINT64_MAX

#define STEAM_CONTROLLER_MIN_ANALOG_ACTION_DATA -1.0f
#define STEAM_CONTROLLER_MAX_ANALOG_ACTION_DATA 1.0f

#ifndef ISTEAMINPUT_H
enum ESteamControllerPad
{
	k_ESteamControllerPad_Left,
	k_ESteamControllerPad_Right
};
#endif

// Note: Please do not use action origins as a way to identify controller types. There is no
// guarantee that they will be added in a contiguous manner - use GetInputTypeForHandle instead
// Versions of Steam that add new controller types in the future will extend this enum if you're
// using a lookup table please check the bounds of any origins returned by Steam.
enum EControllerActionOrigin
{
	// Steam Controller
	k_EControllerActionOrigin_None,
	k_EControllerActionOrigin_A,
	k_EControllerActionOrigin_B,
	k_EControllerActionOrigin_X,
	k_EControllerActionOrigin_Y,
	k_EControllerActionOrigin_LeftBumper,
	k_EControllerActionOrigin_RightBumper,
	k_EControllerActionOrigin_LeftGrip,
	k_EControllerActionOrigin_RightGrip,
	k_EControllerActionOrigin_Start,
	k_EControllerActionOrigin_Back,
	k_EControllerActionOrigin_LeftPad_Touch,
	k_EControllerActionOrigin_LeftPad_Swipe,
	k_EControllerActionOrigin_LeftPad_Click,
	k_EControllerActionOrigin_LeftPad_DPadNorth,
	k_EControllerActionOrigin_LeftPad_DPadSouth,
	k_EControllerActionOrigin_LeftPad_DPadWest,
	k_EControllerActionOrigin_LeftPad_DPadEast,
	k_EControllerActionOrigin_RightPad_Touch,
	k_EControllerActionOrigin_RightPad_Swipe,
	k_EControllerActionOrigin_RightPad_Click,
	k_EControllerActionOrigin_RightPad_DPadNorth,
	k_EControllerActionOrigin_RightPad_DPadSouth,
	k_EControllerActionOrigin_RightPad_DPadWest,
	k_EControllerActionOrigin_RightPad_DPadEast,
	k_EControllerActionOrigin_LeftTrigger_Pull,
	k_EControllerActionOrigin_LeftTrigger_Click,
	k_EControllerActionOrigin_RightTrigger_Pull,
	k_EControllerActionOrigin_RightTrigger_Click,
	k_EControllerActionOrigin_LeftStick_Move,
	k_EControllerActionOrigin_LeftStick_Click,
	k_EControllerActionOrigin_LeftStick_DPadNorth,
	k_EControllerActionOrigin_LeftStick_DPadSouth,
	k_EControllerActionOrigin_LeftStick_DPadWest,
	k_EControllerActionOrigin_LeftStick_DPadEast,
	k_EControllerActionOrigin_Gyro_Move,
	k_EControllerActionOrigin_Gyro_Pitch,
	k_EControllerActionOrigin_Gyro_Yaw,
	k_EControllerActionOrigin_Gyro_Roll,
	
	// PS4 Dual Shock
	k_EControllerActionOrigin_PS4_X,
	k_EControllerActionOrigin_PS4_Circle,
	k_EControllerActionOrigin_PS4_Triangle,
	k_EControllerActionOrigin_PS4_Square,
	k_EControllerActionOrigin_PS4_LeftBumper,
	k_EControllerActionOrigin_PS4_RightBumper,
	k_EControllerActionOrigin_PS4_Options,  //Start
	k_EControllerActionOrigin_PS4_Share,	//Back
	k_EControllerActionOrigin_PS4_LeftPad_Touch,
	k_EControllerActionOrigin_PS4_LeftPad_Swipe,
	k_EControllerActionOrigin_PS4_LeftPad_Click,
	k_EControllerActionOrigin_PS4_LeftPad_DPadNorth,
	k_EControllerActionOrigin_PS4_LeftPad_DPadSouth,
	k_EControllerActionOrigin_PS4_LeftPad_DPadWest,
	k_EControllerActionOrigin_PS4_LeftPad_DPadEast,
	k_EControllerActionOrigin_PS4_RightPad_Touch,
	k_EControllerActionOrigin_PS4_RightPad_Swipe,
	k_EControllerActionOrigin_PS4_RightPad_Click,
	k_EControllerActionOrigin_PS4_RightPad_DPadNorth,
	k_EControllerActionOrigin_PS4_RightPad_DPadSouth,
	k_EControllerActionOrigin_PS4_RightPad_DPadWest,
	k_EControllerActionOrigin_PS4_RightPad_DPadEast,
	k_EControllerActionOrigin_PS4_CenterPad_Touch,
	k_EControllerActionOrigin_PS4_CenterPad_Swipe,
	k_EControllerActionOrigin_PS4_CenterPad_Click,
	k_EControllerActionOrigin_PS4_CenterPad_DPadNorth,
	k_EControllerActionOrigin_PS4_CenterPad_DPadSouth,
	k_EControllerActionOrigin_PS4_CenterPad_DPadWest,
	k_EControllerActionOrigin_PS4_CenterPad_DPadEast,
	k_EControllerActionOrigin_PS4_LeftTrigger_Pull,
	k_EControllerActionOrigin_PS4_LeftTrigger_Click,
	k_EControllerActionOrigin_PS4_RightTrigger_Pull,
	k_EControllerActionOrigin_PS4_RightTrigger_Click,
	k_EControllerActionOrigin_PS4_LeftStick_Move,
	k_EControllerActionOrigin_PS4_LeftStick_Click,
	k_EControllerActionOrigin_PS4_LeftStick_DPadNorth,
	k_EControllerActionOrigin_PS4_LeftStick_DPadSouth,
	k_EControllerActionOrigin_PS4_LeftStick_DPadWest,
	k_EControllerActionOrigin_PS4_LeftStick_DPadEast,
	k_EControllerActionOrigin_PS4_RightStick_Move,
	k_EControllerActionOrigin_PS4_RightStick_Click,
	k_EControllerActionOrigin_PS4_RightStick_DPadNorth,
	k_EControllerActionOrigin_PS4_RightStick_DPadSouth,
	k_EControllerActionOrigin_PS4_RightStick_DPadWest,
	k_EControllerActionOrigin_PS4_RightStick_DPadEast,
	k_EControllerActionOrigin_PS4_DPad_North,
	k_EControllerActionOrigin_PS4_DPad_South,
	k_EControllerActionOrigin_PS4_DPad_West,
	k_EControllerActionOrigin_PS4_DPad_East,
	k_EControllerActionOrigin_PS4_Gyro_Move,
	k_EControllerActionOrigin_PS4_Gyro_Pitch,
	k_EControllerActionOrigin_PS4_Gyro_Yaw,
	k_EControllerActionOrigin_PS4_Gyro_Roll,

	// XBox One
	k_EControllerActionOrigin_XBoxOne_A,
	k_EControllerActionOrigin_XBoxOne_B,
	k_EControllerActionOrigin_XBoxOne_X,
	k_EControllerActionOrigin_XBoxOne_Y,
	k_EControllerActionOrigin_XBoxOne_LeftBumper,
	k_EControllerActionOrigin_XBoxOne_RightBumper,
	k_EControllerActionOrigin_XBoxOne_Menu,  //Start
	k_EControllerActionOrigin_XBoxOne_View,  //Back
	k_EControllerActionOrigin_XBoxOne_LeftTrigger_Pull,
	k_EControllerActionOrigin_XBoxOne_LeftTrigger_Click,
	k_EControllerActionOrigin_XBoxOne_RightTrigger_Pull,
	k_EControllerActionOrigin_XBoxOne_RightTrigger_Click,
	k_EControllerActionOrigin_XBoxOne_LeftStick_Move,
	k_EControllerActionOrigin_XBoxOne_LeftStick_Click,
	k_EControllerActionOrigin_XBoxOne_LeftStick_DPadNorth,
	k_EControllerActionOrigin_XBoxOne_LeftStick_DPadSouth,
	k_EControllerActionOrigin_XBoxOne_LeftStick_DPadWest,
	k_EControllerActionOrigin_XBoxOne_LeftStick_DPadEast,
	k_EControllerActionOrigin_XBoxOne_RightStick_Move,
	k_EControllerActionOrigin_XBoxOne_RightStick_Click,
	k_EControllerActionOrigin_XBoxOne_RightStick_DPadNorth,
	k_EControllerActionOrigin_XBoxOne_RightStick_DPadSouth,
	k_EControllerActionOrigin_XBoxOne_RightStick_DPadWest,
	k_EControllerActionOrigin_XBoxOne_RightStick_DPadEast,
	k_EControllerActionOrigin_XBoxOne_DPad_North,
	k_EControllerActionOrigin_XBoxOne_DPad_South,
	k_EControllerActionOrigin_XBoxOne_DPad_West,
	k_EControllerActionOrigin_XBoxOne_DPad_East,

	// XBox 360
	k_EControllerActionOrigin_XBox360_A,
	k_EControllerActionOrigin_XBox360_B,
	k_EControllerActionOrigin_XBox360_X,
	k_EControllerActionOrigin_XBox360_Y,
	k_EControllerActionOrigin_XBox360_LeftBumper,
	k_EControllerActionOrigin_XBox360_RightBumper,
	k_EControllerActionOrigin_XBox360_Start,  //Start
	k_EControllerActionOrigin_XBox360_Back,  //Back
	k_EControllerActionOrigin_XBox360_LeftTrigger_Pull,
	k_EControllerActionOrigin_XBox360_LeftTrigger_Click,
	k_EControllerActionOrigin_XBox360_RightTrigger_Pull,
	k_EControllerActionOrigin_XBox360_RightTrigger_Click,
	k_EControllerActionOrigin_XBox360_LeftStick_Move,
	k_EControllerActionOrigin_XBox360_LeftStick_Click,
	k_EControllerActionOrigin_XBox360_LeftStick_DPadNorth,
	k_EControllerActionOrigin_XBox360_LeftStick_DPadSouth,
	k_EControllerActionOrigin_XBox360_LeftStick_DPadWest,
	k_EControllerActionOrigin_XBox360_LeftStick_DPadEast,
	k_EControllerActionOrigin_XBox360_RightStick_Move,
	k_EControllerActionOrigin_XBox360_RightStick_Click,
	k_EControllerActionOrigin_XBox360_RightStick_DPadNorth,
	k_EControllerActionOrigin_XBox360_RightStick_DPadSouth,
	k_EControllerActionOrigin_XBox360_RightStick_DPadWest,
	k_EControllerActionOrigin_XBox360_RightStick_DPadEast,
	k_EControllerActionOrigin_XBox360_DPad_North,
	k_EControllerActionOrigin_XBox360_DPad_South,
	k_EControllerActionOrigin_XBox360_DPad_West,
	k_EControllerActionOrigin_XBox360_DPad_East,	

	// SteamController V2
	k_EControllerActionOrigin_SteamV2_A,
	k_EControllerActionOrigin_SteamV2_B,
	k_EControllerActionOrigin_SteamV2_X,
	k_EControllerActionOrigin_SteamV2_Y,
	k_EControllerActionOrigin_SteamV2_LeftBumper,
	k_EControllerActionOrigin_SteamV2_RightBumper,
	k_EControllerActionOrigin_SteamV2_LeftGrip_Lower,
	k_EControllerActionOrigin_SteamV2_LeftGrip_Upper,
	k_EControllerActionOrigin_SteamV2_RightGrip_Lower,
	k_EControllerActionOrigin_SteamV2_RightGrip_Upper,
	k_EControllerActionOrigin_SteamV2_LeftBumper_Pressure,
	k_EControllerActionOrigin_SteamV2_RightBumper_Pressure,
	k_EControllerActionOrigin_SteamV2_LeftGrip_Pressure,
	k_EControllerActionOrigin_SteamV2_RightGrip_Pressure,
	k_EControllerActionOrigin_SteamV2_LeftGrip_Upper_Pressure,
	k_EControllerActionOrigin_SteamV2_RightGrip_Upper_Pressure,
	k_EControllerActionOrigin_SteamV2_Start,
	k_EControllerActionOrigin_SteamV2_Back,
	k_EControllerActionOrigin_SteamV2_LeftPad_Touch,
	k_EControllerActionOrigin_SteamV2_LeftPad_Swipe,
	k_EControllerActionOrigin_SteamV2_LeftPad_Click,
	k_EControllerActionOrigin_SteamV2_LeftPad_Pressure,
	k_EControllerActionOrigin_SteamV2_LeftPad_DPadNorth,
	k_EControllerActionOrigin_SteamV2_LeftPad_DPadSouth,
	k_EControllerActionOrigin_SteamV2_LeftPad_DPadWest,
	k_EControllerActionOrigin_SteamV2_LeftPad_DPadEast,
	k_EControllerActionOrigin_SteamV2_RightPad_Touch,
	k_EControllerActionOrigin_SteamV2_RightPad_Swipe,
	k_EControllerActionOrigin_SteamV2_RightPad_Click,
	k_EControllerActionOrigin_SteamV2_RightPad_Pressure,
	k_EControllerActionOrigin_SteamV2_RightPad_DPadNorth,
	k_EControllerActionOrigin_SteamV2_RightPad_DPadSouth,
	k_EControllerActionOrigin_SteamV2_RightPad_DPadWest,
	k_EControllerActionOrigin_SteamV2_RightPad_DPadEast,
	k_EControllerActionOrigin_SteamV2_LeftTrigger_Pull,
	k_EControllerActionOrigin_SteamV2_LeftTrigger_Click,
	k_EControllerActionOrigin_SteamV2_RightTrigger_Pull,
	k_EControllerActionOrigin_SteamV2_RightTrigger_Click,
	k_EControllerActionOrigin_SteamV2_LeftStick_Move,
	k_EControllerActionOrigin_SteamV2_LeftStick_Click,
	k_EControllerActionOrigin_SteamV2_LeftStick_DPadNorth,
	k_EControllerActionOrigin_SteamV2_LeftStick_DPadSouth,
	k_EControllerActionOrigin_SteamV2_LeftStick_DPadWest,
	k_EControllerActionOrigin_SteamV2_LeftStick_DPadEast,
	k_EControllerActionOrigin_SteamV2_Gyro_Move,
	k_EControllerActionOrigin_SteamV2_Gyro_Pitch,
	k_EControllerActionOrigin_SteamV2_Gyro_Yaw,
	k_EControllerActionOrigin_SteamV2_Gyro_Roll,

	// Switch - Pro or Joycons used as a single input device.
	// This does not apply to a single joycon
	k_EControllerActionOrigin_Switch_A,
	k_EControllerActionOrigin_Switch_B,
	k_EControllerActionOrigin_Switch_X,
	k_EControllerActionOrigin_Switch_Y,
	k_EControllerActionOrigin_Switch_LeftBumper,
	k_EControllerActionOrigin_Switch_RightBumper,
	k_EControllerActionOrigin_Switch_Plus,  //Start
	k_EControllerActionOrigin_Switch_Minus,	//Back
	k_EControllerActionOrigin_Switch_Capture,
	k_EControllerActionOrigin_Switch_LeftTrigger_Pull,
	k_EControllerActionOrigin_Switch_LeftTrigger_Click,
	k_EControllerActionOrigin_Switch_RightTrigger_Pull,
	k_EControllerActionOrigin_Switch_RightTrigger_Click,
	k_EControllerActionOrigin_Switch_LeftStick_Move,
	k_EControllerActionOrigin_Switch_LeftStick_Click,
	k_EControllerActionOrigin_Switch_LeftStick_DPadNorth,
	k_EControllerActionOrigin_Switch_LeftStick_DPadSouth,
	k_EControllerActionOrigin_Switch_LeftStick_DPadWest,
	k_EControllerActionOrigin_Switch_LeftStick_DPadEast,
	k_EControllerActionOrigin_Switch_RightStick_Move,
	k_EControllerActionOrigin_Switch_RightStick_Click,
	k_EControllerActionOrigin_Switch_RightStick_DPadNorth,
	k_EControllerActionOrigin_Switch_RightStick_DPadSouth,
	k_EControllerActionOrigin_Switch_RightStick_DPadWest,
	k_EControllerActionOrigin_Switch_RightStick_DPadEast,
	k_EControllerActionOrigin_Switch_DPad_North,
	k_EControllerActionOrigin_Switch_DPad_South,
	k_EControllerActionOrigin_Switch_DPad_West,
	k_EControllerActionOrigin_Switch_DPad_East,
	k_EControllerActionOrigin_Switch_ProGyro_Move,  // Primary Gyro in Pro Controller, or Right JoyCon
	k_EControllerActionOrigin_Switch_ProGyro_Pitch,  // Primary Gyro in Pro Controller, or Right JoyCon
	k_EControllerActionOrigin_Switch_ProGyro_Yaw,  // Primary Gyro in Pro Controller, or Right JoyCon
	k_EControllerActionOrigin_Switch_ProGyro_Roll,  // Primary Gyro in Pro Controller, or Right JoyCon
	// Switch JoyCon Specific
	k_EControllerActionOrigin_Switch_RightGyro_Move,  // Right JoyCon Gyro generally should correspond to Pro's single gyro
	k_EControllerActionOrigin_Switch_RightGyro_Pitch,  // Right JoyCon Gyro generally should correspond to Pro's single gyro
	k_EControllerActionOrigin_Switch_RightGyro_Yaw,  // Right JoyCon Gyro generally should correspond to Pro's single gyro
	k_EControllerActionOrigin_Switch_RightGyro_Roll,  // Right JoyCon Gyro generally should correspond to Pro's single gyro
	k_EControllerActionOrigin_Switch_LeftGyro_Move,
	k_EControllerActionOrigin_Switch_LeftGyro_Pitch,
	k_EControllerActionOrigin_Switch_LeftGyro_Yaw,
	k_EControllerActionOrigin_Switch_LeftGyro_Roll,
	k_EControllerActionOrigin_Switch_LeftGrip_Lower, // Left JoyCon SR Button
	k_EControllerActionOrigin_Switch_LeftGrip_Upper, // Left JoyCon SL Button
	k_EControllerActionOrigin_Switch_RightGrip_Lower,  // Right JoyCon SL Button
	k_EControllerActionOrigin_Switch_RightGrip_Upper,  // Right JoyCon SR Button

	// Added in SDK 1.45
	k_EControllerActionOrigin_PS4_DPad_Move,
	k_EControllerActionOrigin_XBoxOne_DPad_Move,
	k_EControllerActionOrigin_XBox360_DPad_Move,
	k_EControllerActionOrigin_Switch_DPad_Move,

	// Added in SDK 1.51
	k_EControllerActionOrigin_PS5_X,
	k_EControllerActionOrigin_PS5_Circle,
	k_EControllerActionOrigin_PS5_Triangle,
	k_EControllerActionOrigin_PS5_Square,
	k_EControllerActionOrigin_PS5_LeftBumper,
	k_EControllerActionOrigin_PS5_RightBumper,
	k_EControllerActionOrigin_PS5_Option,  //Start
	k_EControllerActionOrigin_PS5_Create,	//Back
	k_EControllerActionOrigin_PS5_Mute,
	k_EControllerActionOrigin_PS5_LeftPad_Touch,
	k_EControllerActionOrigin_PS5_LeftPad_Swipe,
	k_EControllerActionOrigin_PS5_LeftPad_Click,
	k_EControllerActionOrigin_PS5_LeftPad_DPadNorth,
	k_EControllerActionOrigin_PS5_LeftPad_DPadSouth,
	k_EControllerActionOrigin_PS5_LeftPad_DPadWest,
	k_EControllerActionOrigin_PS5_LeftPad_DPadEast,
	k_EControllerActionOrigin_PS5_RightPad_Touch,
	k_EControllerActionOrigin_PS5_RightPad_Swipe,
	k_EControllerActionOrigin_PS5_RightPad_Click,
	k_EControllerActionOrigin_PS5_RightPad_DPadNorth,
	k_EControllerActionOrigin_PS5_RightPad_DPadSouth,
	k_EControllerActionOrigin_PS5_RightPad_DPadWest,
	k_EControllerActionOrigin_PS5_RightPad_DPadEast,
	k_EControllerActionOrigin_PS5_CenterPad_Touch,
	k_EControllerActionOrigin_PS5_CenterPad_Swipe,
	k_EControllerActionOrigin_PS5_CenterPad_Click,
	k_EControllerActionOrigin_PS5_CenterPad_DPadNorth,
	k_EControllerActionOrigin_PS5_CenterPad_DPadSouth,
	k_EControllerActionOrigin_PS5_CenterPad_DPadWest,
	k_EControllerActionOrigin_PS5_CenterPad_DPadEast,
	k_EControllerActionOrigin_PS5_LeftTrigger_Pull,
	k_EControllerActionOrigin_PS5_LeftTrigger_Click,
	k_EControllerActionOrigin_PS5_RightTrigger_Pull,
	k_EControllerActionOrigin_PS5_RightTrigger_Click,
	k_EControllerActionOrigin_PS5_LeftStick_Move,
	k_EControllerActionOrigin_PS5_LeftStick_Click,
	k_EControllerActionOrigin_PS5_LeftStick_DPadNorth,
	k_EControllerActionOrigin_PS5_LeftStick_DPadSouth,
	k_EControllerActionOrigin_PS5_LeftStick_DPadWest,
	k_EControllerActionOrigin_PS5_LeftStick_DPadEast,
	k_EControllerActionOrigin_PS5_RightStick_Move,
	k_EControllerActionOrigin_PS5_RightStick_Click,
	k_EControllerActionOrigin_PS5_RightStick_DPadNorth,
	k_EControllerActionOrigin_PS5_RightStick_DPadSouth,
	k_EControllerActionOrigin_PS5_RightStick_DPadWest,
	k_EControllerActionOrigin_PS5_RightStick_DPadEast,
	k_EControllerActionOrigin_PS5_DPad_Move,
	k_EControllerActionOrigin_PS5_DPad_North,
	k_EControllerActionOrigin_PS5_DPad_South,
	k_EControllerActionOrigin_PS5_DPad_West,
	k_EControllerActionOrigin_PS5_DPad_East,
	k_EControllerActionOrigin_PS5_Gyro_Move,
	k_EControllerActionOrigin_PS5_Gyro_Pitch,
	k_EControllerActionOrigin_PS5_Gyro_Yaw,
	k_EControllerActionOrigin_PS5_Gyro_Roll,

	k_EControllerActionOrigin_XBoxOne_LeftGrip_Lower, 
	k_EControllerActionOrigin_XBoxOne_LeftGrip_Upper, 
	k_EControllerActionOrigin_XBoxOne_RightGrip_Lower,
	k_EControllerActionOrigin_XBoxOne_RightGrip_Upper,
	k_EControllerActionOrigin_XBoxOne_Share,

	// Added in SDK 1.53
	k_EControllerActionOrigin_SteamDeck_A,
	k_EControllerActionOrigin_SteamDeck_B,
	k_EControllerActionOrigin_SteamDeck_X,
	k_EControllerActionOrigin_SteamDeck_Y,
	k_EControllerActionOrigin_SteamDeck_L1,
	k_EControllerActionOrigin_SteamDeck_R1,
	k_EControllerActionOrigin_SteamDeck_Menu,
	k_EControllerActionOrigin_SteamDeck_View,
	k_EControllerActionOrigin_SteamDeck_LeftPad_Touch,
	k_EControllerActionOrigin_SteamDeck_LeftPad_Swipe,
	k_EControllerActionOrigin_SteamDeck_LeftPad_Click,
	k_EControllerActionOrigin_SteamDeck_LeftPad_DPadNorth,
	k_EControllerActionOrigin_SteamDeck_LeftPad_DPadSouth,
	k_EControllerActionOrigin_SteamDeck_LeftPad_DPadWest,
	k_EControllerActionOrigin_SteamDeck_LeftPad_DPadEast,
	k_EControllerActionOrigin_SteamDeck_RightPad_Touch,
	k_EControllerActionOrigin_SteamDeck_RightPad_Swipe,
	k_EControllerActionOrigin_SteamDeck_RightPad_Click,
	k_EControllerActionOrigin_SteamDeck_RightPad_DPadNorth,
	k_EControllerActionOrigin_SteamDeck_RightPad_DPadSouth,
	k_EControllerActionOrigin_SteamDeck_RightPad_DPadWest,
	k_EControllerActionOrigin_SteamDeck_RightPad_DPadEast,
	k_EControllerActionOrigin_SteamDeck_L2_SoftPull,
	k_EControllerActionOrigin_SteamDeck_L2,
	k_EControllerActionOrigin_SteamDeck_R2_SoftPull,
	k_EControllerActionOrigin_SteamDeck_R2,
	k_EControllerActionOrigin_SteamDeck_LeftStick_Move,
	k_EControllerActionOrigin_SteamDeck_L3,
	k_EControllerActionOrigin_SteamDeck_LeftStick_DPadNorth,
	k_EControllerActionOrigin_SteamDeck_LeftStick_DPadSouth,
	k_EControllerActionOrigin_SteamDeck_LeftStick_DPadWest,
	k_EControllerActionOrigin_SteamDeck_LeftStick_DPadEast,
	k_EControllerActionOrigin_SteamDeck_LeftStick_Touch,
	k_EControllerActionOrigin_SteamDeck_RightStick_Move,
	k_EControllerActionOrigin_SteamDeck_R3,
	k_EControllerActionOrigin_SteamDeck_RightStick_DPadNorth,
	k_EControllerActionOrigin_SteamDeck_RightStick_DPadSouth,
	k_EControllerActionOrigin_SteamDeck_RightStick_DPadWest,
	k_EControllerActionOrigin_SteamDeck_RightStick_DPadEast,
	k_EControllerActionOrigin_SteamDeck_RightStick_Touch,
	k_EControllerActionOrigin_SteamDeck_L4,
	k_EControllerActionOrigin_SteamDeck_R4,
	k_EControllerActionOrigin_SteamDeck_L5,
	k_EControllerActionOrigin_SteamDeck_R5,
	k_EControllerActionOrigin_SteamDeck_DPad_Move,
	k_EControllerActionOrigin_SteamDeck_DPad_North,
	k_EControllerActionOrigin_SteamDeck_DPad_South,
	k_EControllerActionOrigin_SteamDeck_DPad_West,
	k_EControllerActionOrigin_SteamDeck_DPad_East,
	k_EControllerActionOrigin_SteamDeck_Gyro_Move,
	k_EControllerActionOrigin_SteamDeck_Gyro_Pitch,
	k_EControllerActionOrigin_SteamDeck_Gyro_Yaw,
	k_EControllerActionOrigin_SteamDeck_Gyro_Roll,
	k_EControllerActionOrigin_SteamDeck_Reserved1,
	k_EControllerActionOrigin_SteamDeck_Reserved2,
	k_EControllerActionOrigin_SteamDeck_Reserved3,
	k_EControllerActionOrigin_SteamDeck_Reserved4,
	k_EControllerActionOrigin_SteamDeck_Reserved5,
	k_EControllerActionOrigin_SteamDeck_Reserved6,
	k_EControllerActionOrigin_SteamDeck_Reserved7,
	k_EControllerActionOrigin_SteamDeck_Reserved8,
	k_EControllerActionOrigin_SteamDeck_Reserved9,
	k_EControllerActionOrigin_SteamDeck_Reserved10,
	k_EControllerActionOrigin_SteamDeck_Reserved11,
	k_EControllerActionOrigin_SteamDeck_Reserved12,
	k_EControllerActionOrigin_SteamDeck_Reserved13,
	k_EControllerActionOrigin_SteamDeck_Reserved14,
	k_EControllerActionOrigin_SteamDeck_Reserved15,
	k_EControllerActionOrigin_SteamDeck_Reserved16,
	k_EControllerActionOrigin_SteamDeck_Reserved17,
	k_EControllerActionOrigin_SteamDeck_Reserved18,
	k_EControllerActionOrigin_SteamDeck_Reserved19,
	k_EControllerActionOrigin_SteamDeck_Reserved20,

	k_EControllerActionOrigin_Count, // If Steam has added support for new controllers origins will go here.
	k_EControllerActionOrigin_MaximumPossibleValue = 32767, // Origins are currently a maximum of 16 bits.
};

#ifndef ISTEAMINPUT_H
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
	k_ESteamInputType_Count,
	k_ESteamInputType_MaximumPossibleValue = 255,
};
#endif

enum ESteamControllerLEDFlag
{
	k_ESteamControllerLEDFlag_SetColor,
	k_ESteamControllerLEDFlag_RestoreUserDefault
};

// ControllerHandle_t is used to refer to a specific controller.
// This handle will consistently identify a controller, even if it is disconnected and re-connected
typedef uint64 ControllerHandle_t;


// These handles are used to refer to a specific in-game action or action set
// All action handles should be queried during initialization for performance reasons
typedef uint64 ControllerActionSetHandle_t;
typedef uint64 ControllerDigitalActionHandle_t;
typedef uint64 ControllerAnalogActionHandle_t;

#pragma pack( push, 1 )

#ifdef ISTEAMINPUT_H
#define ControllerAnalogActionData_t InputAnalogActionData_t
#define ControllerDigitalActionData_t InputDigitalActionData_t
#define ControllerMotionData_t  InputMotionData_t
#else
struct ControllerAnalogActionData_t
{
	// Type of data coming from this action, this will match what got specified in the action set
	EControllerSourceMode eMode;
	
	// The current state of this action; will be delta updates for mouse actions
	float x, y;
	
	// Whether or not this action is currently available to be bound in the active action set
	bool bActive;
};

struct ControllerDigitalActionData_t
{
	// The current state of this action; will be true if currently pressed
	bool bState;
	
	// Whether or not this action is currently available to be bound in the active action set
	bool bActive;
};

struct ControllerMotionData_t
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
#endif
#pragma pack( pop )


//-----------------------------------------------------------------------------
// Purpose: Steam Input API
//-----------------------------------------------------------------------------
class ISteamController
{
public:
	
	// Init and Shutdown must be called when starting/ending use of this interface
	virtual bool Init() = 0;
	virtual bool Shutdown() = 0;
	
	// Synchronize API state with the latest Steam Controller inputs available. This
	// is performed automatically by SteamAPI_RunCallbacks, but for the absolute lowest
	// possible latency, you call this directly before reading controller state. This must
	// be called from somewhere before GetConnectedControllers will return any handles
	virtual void RunFrame() = 0;

	// Enumerate currently connected controllers
	// handlesOut should point to a STEAM_CONTROLLER_MAX_COUNT sized array of ControllerHandle_t handles
	// Returns the number of handles written to handlesOut
	virtual int GetConnectedControllers( STEAM_OUT_ARRAY_COUNT( STEAM_CONTROLLER_MAX_COUNT, Receives list of connected controllers ) ControllerHandle_t *handlesOut ) = 0;
	
	//-----------------------------------------------------------------------------
	// ACTION SETS
	//-----------------------------------------------------------------------------

	// Lookup the handle for an Action Set. Best to do this once on startup, and store the handles for all future API calls.
	virtual ControllerActionSetHandle_t GetActionSetHandle( const char *pszActionSetName ) = 0;
	
	// Reconfigure the controller to use the specified action set (ie 'Menu', 'Walk' or 'Drive')
	// This is cheap, and can be safely called repeatedly. It's often easier to repeatedly call it in
	// your state loops, instead of trying to place it in all of your state transitions.
	virtual void ActivateActionSet( ControllerHandle_t controllerHandle, ControllerActionSetHandle_t actionSetHandle ) = 0;
	virtual ControllerActionSetHandle_t GetCurrentActionSet( ControllerHandle_t controllerHandle ) = 0;

	// ACTION SET LAYERS
	virtual void ActivateActionSetLayer( ControllerHandle_t controllerHandle, ControllerActionSetHandle_t actionSetLayerHandle ) = 0;
	virtual void DeactivateActionSetLayer( ControllerHandle_t controllerHandle, ControllerActionSetHandle_t actionSetLayerHandle ) = 0;
	virtual void DeactivateAllActionSetLayers( ControllerHandle_t controllerHandle ) = 0;
	// Enumerate currently active layers
	// handlesOut should point to a STEAM_CONTROLLER_MAX_ACTIVE_LAYERS sized array of ControllerActionSetHandle_t handles.
	// Returns the number of handles written to handlesOut
	virtual int GetActiveActionSetLayers( ControllerHandle_t controllerHandle, STEAM_OUT_ARRAY_COUNT( STEAM_CONTROLLER_MAX_ACTIVE_LAYERS, Receives list of active layers ) ControllerActionSetHandle_t *handlesOut ) = 0;

	//-----------------------------------------------------------------------------
	// ACTIONS
	//-----------------------------------------------------------------------------

	// Lookup the handle for a digital action. Best to do this once on startup, and store the handles for all future API calls.
	virtual ControllerDigitalActionHandle_t GetDigitalActionHandle( const char *pszActionName ) = 0;
	
	// Returns the current state of the supplied digital game action
	virtual ControllerDigitalActionData_t GetDigitalActionData( ControllerHandle_t controllerHandle, ControllerDigitalActionHandle_t digitalActionHandle ) = 0;
	
	// Get the origin(s) for a digital action within an action set. Returns the number of origins supplied in originsOut. Use this to display the appropriate on-screen prompt for the action.
	// originsOut should point to a STEAM_CONTROLLER_MAX_ORIGINS sized array of EControllerActionOrigin handles. The EControllerActionOrigin enum will get extended as support for new controller controllers gets added to
	// the Steam client and will exceed the values from this header, please check bounds if you are using a look up table.
	virtual int GetDigitalActionOrigins( ControllerHandle_t controllerHandle, ControllerActionSetHandle_t actionSetHandle, ControllerDigitalActionHandle_t digitalActionHandle, STEAM_OUT_ARRAY_COUNT( STEAM_CONTROLLER_MAX_ORIGINS, Receives list of aciton origins ) EControllerActionOrigin *originsOut ) = 0;
	
	// Lookup the handle for an analog action. Best to do this once on startup, and store the handles for all future API calls.
	virtual ControllerAnalogActionHandle_t GetAnalogActionHandle( const char *pszActionName ) = 0;
	
	// Returns the current state of these supplied analog game action
	virtual ControllerAnalogActionData_t GetAnalogActionData( ControllerHandle_t controllerHandle, ControllerAnalogActionHandle_t analogActionHandle ) = 0;

	// Get the origin(s) for an analog action within an action set. Returns the number of origins supplied in originsOut. Use this to display the appropriate on-screen prompt for the action.
	// originsOut should point to a STEAM_CONTROLLER_MAX_ORIGINS sized array of EControllerActionOrigin handles. The EControllerActionOrigin enum will get extended as support for new controller controllers gets added to
	// the Steam client and will exceed the values from this header, please check bounds if you are using a look up table.
	virtual int GetAnalogActionOrigins( ControllerHandle_t controllerHandle, ControllerActionSetHandle_t actionSetHandle, ControllerAnalogActionHandle_t analogActionHandle, STEAM_OUT_ARRAY_COUNT( STEAM_CONTROLLER_MAX_ORIGINS, Receives list of action origins ) EControllerActionOrigin *originsOut ) = 0;
	
	// Get a local path to art for on-screen glyph for a particular origin - this call is cheap
	virtual const char *GetGlyphForActionOrigin( EControllerActionOrigin eOrigin ) = 0;
	
	// Returns a localized string (from Steam's language setting) for the specified origin - this call is serialized
	virtual const char *GetStringForActionOrigin( EControllerActionOrigin eOrigin ) = 0;

	virtual void StopAnalogActionMomentum( ControllerHandle_t controllerHandle, ControllerAnalogActionHandle_t eAction ) = 0;

	// Returns raw motion data from the specified controller
	virtual ControllerMotionData_t GetMotionData( ControllerHandle_t controllerHandle ) = 0;

	//-----------------------------------------------------------------------------
	// OUTPUTS
	//-----------------------------------------------------------------------------

	// Trigger a haptic pulse on a controller
	virtual void TriggerHapticPulse( ControllerHandle_t controllerHandle, ESteamControllerPad eTargetPad, unsigned short usDurationMicroSec ) = 0;

	// Trigger a pulse with a duty cycle of usDurationMicroSec / usOffMicroSec, unRepeat times.
	// nFlags is currently unused and reserved for future use.
	virtual void TriggerRepeatedHapticPulse( ControllerHandle_t controllerHandle, ESteamControllerPad eTargetPad, unsigned short usDurationMicroSec, unsigned short usOffMicroSec, unsigned short unRepeat, unsigned int nFlags ) = 0;
	
	// Trigger a vibration event on supported controllers.  
	virtual void TriggerVibration( ControllerHandle_t controllerHandle, unsigned short usLeftSpeed, unsigned short usRightSpeed ) = 0;

	// Set the controller LED color on supported controllers.  
	virtual void SetLEDColor( ControllerHandle_t controllerHandle, uint8 nColorR, uint8 nColorG, uint8 nColorB, unsigned int nFlags ) = 0;

	//-----------------------------------------------------------------------------
	// Utility functions available without using the rest of Steam Input API
	//-----------------------------------------------------------------------------

	// Invokes the Steam overlay and brings up the binding screen if the user is using Big Picture Mode
	// If the user is not in Big Picture Mode it will open up the binding in a new window
	virtual bool ShowBindingPanel( ControllerHandle_t controllerHandle ) = 0;

	// Returns the input type for a particular handle - unlike EControllerActionOrigin which update with Steam and may return unrecognized values
	// ESteamInputType will remain static and only return valid values from your SDK version 
	virtual ESteamInputType GetInputTypeForHandle( ControllerHandle_t controllerHandle ) = 0;

	// Returns the associated controller handle for the specified emulated gamepad - can be used with the above 2 functions
	// to identify controllers presented to your game over Xinput. Returns 0 if the Xinput index isn't associated with Steam Input
	virtual ControllerHandle_t GetControllerForGamepadIndex( int nIndex ) = 0;

	// Returns the associated gamepad index for the specified controller, if emulating a gamepad or -1 if not associated with an Xinput index
	virtual int GetGamepadIndexForController( ControllerHandle_t ulControllerHandle ) = 0;
	
	// Returns a localized string (from Steam's language setting) for the specified Xbox controller origin.
	virtual const char *GetStringForXboxOrigin( EXboxOrigin eOrigin ) = 0;

	// Get a local path to art for on-screen glyph for a particular Xbox controller origin. 
	virtual const char *GetGlyphForXboxOrigin( EXboxOrigin eOrigin ) = 0;

	// Get the equivalent ActionOrigin for a given Xbox controller origin this can be chained with GetGlyphForActionOrigin to provide future proof glyphs for
	// non-Steam Input API action games. Note - this only translates the buttons directly and doesn't take into account any remapping a user has made in their configuration
	virtual EControllerActionOrigin GetActionOriginFromXboxOrigin( ControllerHandle_t controllerHandle, EXboxOrigin eOrigin ) = 0;

	// Convert an origin to another controller type - for inputs not present on the other controller type this will return k_EControllerActionOrigin_None
	virtual EControllerActionOrigin TranslateActionOrigin( ESteamInputType eDestinationInputType, EControllerActionOrigin eSourceOrigin ) = 0;

	// Get the binding revision for a given device. Returns false if the handle was not valid or if a mapping is not yet loaded for the device
	virtual bool GetControllerBindingRevision( ControllerHandle_t controllerHandle, int *pMajor, int *pMinor ) = 0;
};

#define STEAMCONTROLLER_INTERFACE_VERSION "SteamController008"

// Global interface accessor
inline ISteamController *SteamController();
STEAM_DEFINE_USER_INTERFACE_ACCESSOR( ISteamController *, SteamController, STEAMCONTROLLER_INTERFACE_VERSION );

#endif // ISTEAMCONTROLLER_H
