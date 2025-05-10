STEAM_PROC(void*, SteamAPI_SteamRemoteStorage_v016, (void))

STEAM_PROC(bool, SteamAPI_ISteamRemoteStorage_IsCloudEnabledForAccount, (void*))
STEAM_PROC(bool, SteamAPI_ISteamRemoteStorage_IsCloudEnabledForApp, (void*))

STEAM_PROC(bool, SteamAPI_ISteamRemoteStorage_BeginFileWriteBatch, (void*))
STEAM_PROC(bool, SteamAPI_ISteamRemoteStorage_EndFileWriteBatch, (void*))

STEAM_PROC(Sint32, SteamAPI_ISteamRemoteStorage_GetFileSize, (void*, const char*))
STEAM_PROC(Sint32, SteamAPI_ISteamRemoteStorage_FileRead, (void*, const char*, void*, Sint32))
STEAM_PROC(Sint32, SteamAPI_ISteamRemoteStorage_FileWrite, (void*, const char*, const void*, Sint32))
STEAM_PROC(bool, SteamAPI_ISteamRemoteStorage_GetQuota, (void*, Uint64*, Uint64*))

#undef STEAM_PROC
