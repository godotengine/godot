/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2025 Sam Lantinga <slouken@libsdl.org>

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
/*
 * @author Mark Callow, www.edgewise-consulting.com.
 *
 * Thanks to @slime73 on GitHub for their gist showing how to add a CAMetalLayer
 * backed view.
 */
#include "SDL_internal.h"

#ifndef SDL_cocoametalview_h_
#define SDL_cocoametalview_h_

#if defined(SDL_VIDEO_DRIVER_COCOA) && (defined(SDL_VIDEO_VULKAN) || defined(SDL_VIDEO_METAL))

#import "../SDL_sysvideo.h"

#import "SDL_cocoawindow.h"

#import <Cocoa/Cocoa.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

@interface SDL3_cocoametalview : NSView

- (instancetype)initWithFrame:(NSRect)frame
                      highDPI:(BOOL)highDPI
                     windowID:(Uint32)windowID
                       opaque:(BOOL)opaque;

- (void)updateDrawableSize;
- (NSView *)hitTest:(NSPoint)point;

// Override superclass tag so this class can set it.
@property(assign, readonly) NSInteger tag;

@property(nonatomic) BOOL highDPI;
@property(nonatomic) Uint32 sdlWindowID;

@end

SDL_MetalView Cocoa_Metal_CreateView(SDL_VideoDevice *_this, SDL_Window *window);
void Cocoa_Metal_DestroyView(SDL_VideoDevice *_this, SDL_MetalView view);
void *Cocoa_Metal_GetLayer(SDL_VideoDevice *_this, SDL_MetalView view);

#endif // SDL_VIDEO_DRIVER_COCOA && (SDL_VIDEO_VULKAN || SDL_VIDEO_METAL)

#endif // SDL_cocoametalview_h_
