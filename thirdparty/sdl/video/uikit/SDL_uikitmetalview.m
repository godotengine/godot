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

#if defined(SDL_VIDEO_DRIVER_UIKIT) && (defined(SDL_VIDEO_VULKAN) || defined(SDL_VIDEO_METAL))

#include "../SDL_sysvideo.h"
#include "../../events/SDL_windowevents_c.h"

#import "SDL_uikitwindow.h"
#import "SDL_uikitmetalview.h"

@implementation SDL_uikitmetalview

// Returns a Metal-compatible layer.
+ (Class)layerClass
{
    return [CAMetalLayer class];
}

- (instancetype)initWithFrame:(CGRect)frame
                        scale:(CGFloat)scale
{
    if ((self = [super initWithFrame:frame])) {
        self.tag = SDL_METALVIEW_TAG;
        self.layer.contentsScale = scale;
        [self updateDrawableSize];
    }

    return self;
}

// Set the size of the metal drawables when the view is resized.
- (void)layoutSubviews
{
    [super layoutSubviews];
    [self updateDrawableSize];
}

- (void)updateDrawableSize
{
    CGSize size = self.bounds.size;
    size.width *= self.layer.contentsScale;
    size.height *= self.layer.contentsScale;

    CAMetalLayer *metallayer = ((CAMetalLayer *)self.layer);
    if (metallayer.drawableSize.width != size.width ||
        metallayer.drawableSize.height != size.height) {
        metallayer.drawableSize = size;
        SDL_SendWindowEvent([self getSDLWindow], SDL_EVENT_WINDOW_METAL_VIEW_RESIZED, 0, 0);
    }
}

@end

SDL_MetalView UIKit_Metal_CreateView(SDL_VideoDevice *_this, SDL_Window *window)
{
    @autoreleasepool {
        SDL_UIKitWindowData *data = (__bridge SDL_UIKitWindowData *)window->internal;
        CGFloat scale = 1.0;
        SDL_uikitmetalview *metalview;

        if (window->flags & SDL_WINDOW_HIGH_PIXEL_DENSITY) {
            /* Set the scale to the natural scale factor of the screen - then
             * the backing dimensions of the Metal view will match the pixel
             * dimensions of the screen rather than the dimensions in points
             * yielding high resolution on retine displays.
             */
#ifndef SDL_PLATFORM_VISIONOS
            scale = data.uiwindow.screen.nativeScale;
#else
            // VisionOS doesn't use the concept of "nativeScale" like other iOS devices.
            // We use a fixed scale factor of 2.0 to achieve better pixel density.
            // This is because VisionOS presents a virtual 1280x720 "screen", but we need
            // to render at a higher resolution for optimal visual quality.
            // TODO: Consider making this configurable or determining it dynamically
            // based on the specific visionOS device capabilities.
            scale = 2.0;
#endif
        }

        metalview = [[SDL_uikitmetalview alloc] initWithFrame:data.uiwindow.bounds
                                                        scale:scale];
        if (metalview == nil) {
            SDL_OutOfMemory();
            return NULL;
        }

        [metalview setSDLWindow:window];

        return (void *)CFBridgingRetain(metalview);
    }
}

void UIKit_Metal_DestroyView(SDL_VideoDevice *_this, SDL_MetalView view)
{
    @autoreleasepool {
        SDL_uikitmetalview *metalview = CFBridgingRelease(view);

        if ([metalview isKindOfClass:[SDL_uikitmetalview class]]) {
            [metalview setSDLWindow:NULL];
        }
    }
}

void *UIKit_Metal_GetLayer(SDL_VideoDevice *_this, SDL_MetalView view)
{
    @autoreleasepool {
        SDL_uikitview *uiview = (__bridge SDL_uikitview *)view;
        return (__bridge void *)uiview.layer;
    }
}

#endif // SDL_VIDEO_DRIVER_UIKIT && (SDL_VIDEO_VULKAN || SDL_VIDEO_METAL)
