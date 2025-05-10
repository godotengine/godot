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

#ifndef SDL_uikitmetalview_h_
#define SDL_uikitmetalview_h_

#include "../SDL_sysvideo.h"
#include "SDL_uikitwindow.h"

#if defined(SDL_VIDEO_DRIVER_UIKIT) && (defined(SDL_VIDEO_VULKAN) || defined(SDL_VIDEO_METAL))

#import <UIKit/UIKit.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

@interface SDL_uikitmetalview : SDL_uikitview

- (instancetype)initWithFrame:(CGRect)frame
                        scale:(CGFloat)scale;

@end

SDL_MetalView UIKit_Metal_CreateView(SDL_VideoDevice *_this, SDL_Window *window);
void UIKit_Metal_DestroyView(SDL_VideoDevice *_this, SDL_MetalView view);
void *UIKit_Metal_GetLayer(SDL_VideoDevice *_this, SDL_MetalView view);

#endif // SDL_VIDEO_DRIVER_UIKIT && (SDL_VIDEO_VULKAN || SDL_VIDEO_METAL)

#endif // SDL_uikitmetalview_h_
