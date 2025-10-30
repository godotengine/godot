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
#include "SDL_internal.h"


#include "controller_type.h"
#include "controller_list.h"


static const char *GetControllerTypeOverride( int nVID, int nPID )
{
	const char *hint = SDL_GetHint(SDL_HINT_GAMECONTROLLERTYPE);
	if (hint) {
		char key[32];
		const char *spot = NULL;

		SDL_snprintf(key, sizeof(key), "0x%.4x/0x%.4x=", nVID, nPID);
		spot = SDL_strstr(hint, key);
		if (!spot) {
			SDL_snprintf(key, sizeof(key), "0x%.4X/0x%.4X=", nVID, nPID);
			spot = SDL_strstr(hint, key);
		}
		if (spot) {
			spot += SDL_strlen(key);
			if (SDL_strncmp(spot, "k_eControllerType_", 18) == 0) {
				spot += 18;
			}
			return spot;
		}
	}
	return NULL;
}


EControllerType GuessControllerType( int nVID, int nPID )
{
#if 0//def _DEBUG
	// Verify that there are no duplicates in the controller list
	// If the list were sorted, we could do this much more efficiently, as well as improve lookup speed.
	static bool s_bCheckedForDuplicates;
	if ( !s_bCheckedForDuplicates )
	{
		s_bCheckedForDuplicates = true;
		int i, j;
		for ( i = 0; i < sizeof( arrControllers ) / sizeof( arrControllers[ 0 ] ); ++i )
		{
			for ( j = i + 1; j < sizeof( arrControllers ) / sizeof( arrControllers[ 0 ] ); ++j )
			{
				if ( arrControllers[ i ].m_unDeviceID == arrControllers[ j ].m_unDeviceID )
				{
					Log( "Duplicate controller entry found for VID 0x%.4x PID 0x%.4x\n", ( arrControllers[ i ].m_unDeviceID >> 16 ), arrControllers[ i ].m_unDeviceID & 0xFFFF );
				}
			}
		}
	}
#endif // _DEBUG

	unsigned int unDeviceID = MAKE_CONTROLLER_ID( nVID, nPID );
	int iIndex;

	const char *pszOverride = GetControllerTypeOverride( nVID, nPID );
	if ( pszOverride )
	{
		if ( SDL_strncasecmp( pszOverride, "Xbox360", 7 ) == 0 )
		{
			return k_eControllerType_XBox360Controller;
		}
		if ( SDL_strncasecmp( pszOverride, "XboxOne", 7 ) == 0 )
		{
			return k_eControllerType_XBoxOneController;
		}
		if ( SDL_strncasecmp( pszOverride, "PS3", 3 ) == 0 )
		{
			return k_eControllerType_PS3Controller;
		}
		if ( SDL_strncasecmp( pszOverride, "PS4", 3 ) == 0 )
		{
			return k_eControllerType_PS4Controller;
		}
		if ( SDL_strncasecmp( pszOverride, "PS5", 3 ) == 0 )
		{
			return k_eControllerType_PS5Controller;
		}
		if ( SDL_strncasecmp( pszOverride, "SwitchPro", 9 ) == 0 )
		{
			return k_eControllerType_SwitchProController;
		}
		if ( SDL_strncasecmp( pszOverride, "Steam", 5 ) == 0 )
		{
			return k_eControllerType_SteamController;
		}
		return k_eControllerType_UnknownNonSteamController;
	}

	for ( iIndex = 0; iIndex < sizeof( arrControllers ) / sizeof( arrControllers[0] ); ++iIndex )
	{
		if ( unDeviceID == arrControllers[ iIndex ].m_unDeviceID )
		{
			return arrControllers[ iIndex ].m_eControllerType;
		}
	}

	return k_eControllerType_UnknownNonSteamController;

}

const char *GuessControllerName( int nVID, int nPID )
{
	unsigned int unDeviceID = MAKE_CONTROLLER_ID( nVID, nPID );
	int iIndex;
	for ( iIndex = 0; iIndex < sizeof( arrControllers ) / sizeof( arrControllers[0] ); ++iIndex )
	{
		if ( unDeviceID == arrControllers[ iIndex ].m_unDeviceID )
		{
			return arrControllers[ iIndex ].m_pszName;
		}
	}

	return NULL;

}

#undef MAKE_CONTROLLER_ID
