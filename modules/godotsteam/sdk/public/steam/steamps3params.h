//====== Copyright 1996-2008, Valve Corporation, All rights reserved. =======
//
// Purpose: 
//
//=============================================================================

#ifndef STEAMPS3PARAMS_H
#define STEAMPS3PARAMS_H
#ifdef _WIN32
#pragma once
#endif

//----------------------------------------------------------------------------------------------------------------------------------------------------------//
//	PlayStation 3 initialization parameters
//
//	The following structure must be passed to when loading steam_api_ps3.prx
//----------------------------------------------------------------------------------------------------------------------------------------------------------//
#define STEAM_PS3_PATH_MAX 1055
#define STEAM_PS3_SERVICE_ID_MAX 32
#define STEAM_PS3_COMMUNICATION_ID_MAX 10
#define STEAM_PS3_COMMUNICATION_SIG_MAX 160
#define STEAM_PS3_LANGUAGE_MAX 64
#define STEAM_PS3_REGION_CODE_MAX 16
#define STEAM_PS3_CURRENT_PARAMS_VER 2
struct SteamPS3Params_t
{
	uint32 m_unVersion;										// set to STEAM_PS3_CURRENT_PARAMS_VER
	
	void *pReserved;
	uint32 m_nAppId;										// set to your game's appid

	char m_rgchInstallationPath[ STEAM_PS3_PATH_MAX ];		// directory containing latest steam prx's and sdata. Can be read only (BDVD)
	char m_rgchSystemCache[ STEAM_PS3_PATH_MAX ];			// temp working cache, not persistent 
	char m_rgchGameData[ STEAM_PS3_PATH_MAX ];				// persistent game data path for storing user data
	char m_rgchNpServiceID[ STEAM_PS3_SERVICE_ID_MAX ];
	char m_rgchNpCommunicationID[ STEAM_PS3_COMMUNICATION_ID_MAX ];
	char m_rgchNpCommunicationSig[ STEAM_PS3_COMMUNICATION_SIG_MAX ];

	// Language should be one of the following. must be zero terminated
	// danish
	// dutch
	// english
	// finnish
	// french
	// german
	// italian
	// korean
	// norwegian
	// polish
	// portuguese
	// russian
	// schinese
	// spanish
	// swedish
	// tchinese
	char m_rgchSteamLanguage[ STEAM_PS3_LANGUAGE_MAX ];

	// region codes are "SCEA", "SCEE", "SCEJ". must be zero terminated
	char m_rgchRegionCode[ STEAM_PS3_REGION_CODE_MAX ];

	// Should be SYS_TTYP3 through SYS_TTYP10, if it's 0 then Steam won't spawn a 
	// thread to read console input at all.  Using this let's you use Steam console commands
	// like: profile_on, profile_off, profile_dump, mem_stats, mem_validate.
	unsigned int m_cSteamInputTTY;

	struct Ps3netInit_t
	{
		bool m_bNeedInit;
		void *m_pMemory;
		int m_nMemorySize;
		int m_flags;
	} m_sysNetInitInfo;

	struct Ps3jpgInit_t
	{
		bool m_bNeedInit;
	} m_sysJpgInitInfo;

	struct Ps3pngInit_t
	{
		bool m_bNeedInit;
	} m_sysPngInitInfo;
	
	struct Ps3sysutilUserInfo_t
	{
		bool m_bNeedInit;
	} m_sysSysUtilUserInfo;

	bool m_bIncludeNewsPage;
};


//----------------------------------------------------------------------------------------------------------------------------------------------------------//
// PlayStation 3 memory structure
//----------------------------------------------------------------------------------------------------------------------------------------------------------//
#define STEAMPS3_MALLOC_INUSE 0x53D04A51
#define STEAMPS3_MALLOC_SYSTEM 0x0D102C48
#define STEAMPS3_MALLOC_OK 0xFFD04A51
struct SteamPS3Memory_t
{
	bool m_bSingleAllocation;		// If true, Steam will request one 6MB allocation and use the returned memory for all future allocations
									// If false, Steam will make call malloc for each allocation

	// required function pointers
	void* (*m_pfMalloc)(size_t);
	void* (*m_pfRealloc)(void *, size_t);
	void (*m_pfFree)(void *);
	size_t (*m_pUsable_size)(void*);
};


#endif // STEAMPS3PARAMS_H
