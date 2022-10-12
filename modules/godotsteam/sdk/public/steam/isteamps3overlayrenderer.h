//====== Copyright © 1996-2010, Valve Corporation, All rights reserved. =======
//
// Purpose: interface the game must provide Steam with on PS3 in order for the
// Steam overlay to render.
//
//=============================================================================

#ifndef ISTEAMPS3OVERLAYRENDERER_H
#define ISTEAMPS3OVERLAYRENDERER_H
#ifdef _WIN32
#pragma once
#endif

#include "cell/pad.h"

//-----------------------------------------------------------------------------
// Purpose: Enum for supported gradient directions
//-----------------------------------------------------------------------------
enum EOverlayGradientDirection
{
	k_EOverlayGradientHorizontal = 1,
	k_EOverlayGradientVertical = 2,
	k_EOverlayGradientNone = 3,
};

// Helpers for fetching individual color components from ARGB packed DWORD colors Steam PS3 overlay renderer uses.
#define STEAM_COLOR_RED( color ) \
	(int)(((color)>>16)&0xff)

#define STEAM_COLOR_GREEN( color ) \
	(int)(((color)>>8)&0xff)

#define STEAM_COLOR_BLUE( color ) \
	(int)((color)&0xff)

#define STEAM_COLOR_ALPHA( color ) \
	(int)(((color)>>24)&0xff)


//-----------------------------------------------------------------------------
// Purpose: Interface the game must expose to Steam for rendering
//-----------------------------------------------------------------------------
class ISteamPS3OverlayRenderHost
{
public:

	// Interface for game engine to implement which Steam requires to render.

	// Draw a textured rect.  This may use only part of the texture and will pass texture coords, it will also possibly request a gradient and will specify colors for vertexes.
	virtual void DrawTexturedRect( int x0, int y0, int x1, int y1, float u0, float v0, float u1, float v1, int32 iTextureID, DWORD colorStart, DWORD colorEnd, EOverlayGradientDirection eDirection ) = 0;

	// Load a RGBA texture for Steam, or update a previously loaded one.  Updates may be partial.  You must not evict or remove this texture once Steam has uploaded it.
	virtual void LoadOrUpdateTexture( int32 iTextureID, bool bIsFullTexture, int x0, int y0, uint32 uWidth, uint32 uHeight, int32 iBytes, char *pData ) = 0;

	// Delete a texture Steam previously uploaded
	virtual void DeleteTexture( int32 iTextureID ) = 0;

	// Delete all previously uploaded textures
	virtual void DeleteAllTextures() = 0;
};


//-----------------------------------------------------------------------------
// Purpose: Interface Steam exposes for the game to tell it when to render, etc.
//-----------------------------------------------------------------------------
class ISteamPS3OverlayRender
{
public:

	// Call once at startup to initialize the Steam overlay and pass it your host interface ptr
	virtual bool BHostInitialize( uint32 unScreenWidth, uint32 unScreenHeight, uint32 unRefreshRate, ISteamPS3OverlayRenderHost *pRenderHost, void *CellFontLib ) = 0;

	// Call this once a frame when you are ready for the Steam overlay to render (ie, right before flipping buffers, after all your rendering)
	virtual void Render() = 0;

	// Call this everytime you read input on PS3.
	// 
	// If this returns true, then the overlay is active and has consumed the input, your game
	// should then ignore all the input until BHandleCellPadData once again returns false, which
	// will mean the overlay is deactivated.
	virtual bool BHandleCellPadData( const CellPadData &padData ) = 0;

	// Call this if you detect no controllers connected or that the XMB is intercepting input
	// 
	// This is important to clear input state for the overlay, so keys left down during XMB activation
	// are not continued to be processed.
	virtual bool BResetInputState() = 0;
};


#endif // ISTEAMPS3OVERLAYRENDERER_H