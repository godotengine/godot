/*
  Copyright (C) Valve Corporation

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

#ifndef CONTROLLER_TYPE_H
#define CONTROLLER_TYPE_H
#ifdef _WIN32
#pragma once
#endif

//-----------------------------------------------------------------------------
// Purpose: Steam Controller models 
// WARNING: DO NOT RENUMBER EXISTING VALUES - STORED IN A DATABASE
//-----------------------------------------------------------------------------
typedef enum
{
	k_eControllerType_None = -1,
	k_eControllerType_Unknown = 0,

	// Steam Controllers
	k_eControllerType_UnknownSteamController = 1,
	k_eControllerType_SteamController = 2,
	k_eControllerType_SteamControllerV2 = 3,
	k_eControllerType_SteamControllerNeptune = 4,

	// Other Controllers
	k_eControllerType_UnknownNonSteamController = 30,
	k_eControllerType_XBox360Controller = 31,
	k_eControllerType_XBoxOneController = 32,
	k_eControllerType_PS3Controller = 33,
	k_eControllerType_PS4Controller = 34,
	k_eControllerType_WiiController = 35,
	k_eControllerType_AppleController = 36,
	k_eControllerType_AndroidController = 37,
	k_eControllerType_SwitchProController = 38,
	k_eControllerType_SwitchJoyConLeft = 39,
	k_eControllerType_SwitchJoyConRight = 40,
	k_eControllerType_SwitchJoyConPair = 41,
	k_eControllerType_SwitchInputOnlyController = 42,
	k_eControllerType_MobileTouch = 43,
	k_eControllerType_XInputSwitchController = 44,  // Client-side only, used to mark Nintendo Switch style controllers as using XInput instead of the Nintendo Switch protocol
	k_eControllerType_PS5Controller = 45,
	k_eControllerType_XInputPS4Controller = 46,     // Client-side only, used to mark DualShock 4 style controllers using XInput instead of the DualShock 4 controller protocol
	k_eControllerType_LastController,			// Don't add game controllers below this enumeration - this enumeration can change value

	// Keyboards and Mice
	k_eControllertype_GenericKeyboard = 400,
	k_eControllertype_GenericMouse = 800,
} EControllerType;

typedef struct
{
	unsigned int m_unDeviceID;
	EControllerType m_eControllerType;
	const char *m_pszName;
} ControllerDescription_t;


extern EControllerType GuessControllerType( int nVID, int nPID );
extern const char *GuessControllerName( int nVID, int nPID );

#endif // CONTROLLER_TYPE_H
