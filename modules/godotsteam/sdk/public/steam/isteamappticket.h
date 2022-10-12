//====== Copyright 1996-2008, Valve Corporation, All rights reserved. =======
//
// Purpose: a private, but well versioned, interface to get at critical bits
// of a steam3 appticket - consumed by the simple drm wrapper to let it 
// ask about ownership with greater confidence.
//
//=============================================================================

#ifndef ISTEAMAPPTICKET_H
#define ISTEAMAPPTICKET_H
#pragma once

//-----------------------------------------------------------------------------
// Purpose: hand out a reasonable "future proof" view of an app ownership ticket
// the raw (signed) buffer, and indices into that buffer where the appid and 
// steamid are located.  the sizes of the appid and steamid are implicit in 
// (each version of) the interface - currently uin32 appid and uint64 steamid
//-----------------------------------------------------------------------------
class ISteamAppTicket
{
public:
    virtual uint32 GetAppOwnershipTicketData( uint32 nAppID, void *pvBuffer, uint32 cbBufferLength, uint32 *piAppId, uint32 *piSteamId, uint32 *piSignature, uint32 *pcbSignature ) = 0;
};

#define STEAMAPPTICKET_INTERFACE_VERSION "STEAMAPPTICKET_INTERFACE_VERSION001"


#endif // ISTEAMAPPTICKET_H
