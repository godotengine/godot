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

#include "../../events/SDL_windowevents_c.h"

#import "SDL_cocoametalview.h"

#if defined(SDL_VIDEO_DRIVER_COCOA) && (defined(SDL_VIDEO_VULKAN) || defined(SDL_VIDEO_METAL))

static bool SDLCALL SDL_MetalViewEventWatch(void *userdata, SDL_Event *event)
{
    /* Update the drawable size when SDL receives a size changed event for
     * the window that contains the metal view. It would be nice to use
     * - (void)resizeWithOldSuperviewSize:(NSSize)oldSize and
     * - (void)viewDidChangeBackingProperties instead, but SDL's size change
     * events don't always happen in the same frame (for example when a
     * resizable window exits a fullscreen Space via the user pressing the OS
     * exit-space button). */
    if (event->type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED) {
        @autoreleasepool {
            SDL3_cocoametalview *view = (__bridge SDL3_cocoametalview *)userdata;
            if (view.sdlWindowID == event->window.windowID) {
                [view updateDrawableSize];
            }
        }
    }
    return false;
}

@implementation SDL3_cocoametalview

// Return a Metal-compatible layer.
+ (Class)layerClass
{
    return NSClassFromString(@"CAMetalLayer");
}

// Indicate the view wants to draw using a backing layer instead of drawRect.
- (BOOL)wantsUpdateLayer
{
    return YES;
}

/* When the wantsLayer property is set to YES, this method will be invoked to
 * return a layer instance.
 */
- (CALayer *)makeBackingLayer
{
    return [self.class.layerClass layer];
}

- (instancetype)initWithFrame:(NSRect)frame
                      highDPI:(BOOL)highDPI
                     windowID:(Uint32)windowID
                       opaque:(BOOL)opaque
{
    self = [super initWithFrame:frame];
    if (self != nil) {
        self.highDPI = highDPI;
        self.sdlWindowID = windowID;
        self.wantsLayer = YES;

        // Allow resize.
        self.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;

        self.layer.opaque = opaque;

        SDL_AddWindowEventWatch(SDL_WINDOW_EVENT_WATCH_EARLY, SDL_MetalViewEventWatch, (__bridge void *)(self));

        [self updateDrawableSize];
    }

    return self;
}

- (void)dealloc
{
    SDL_RemoveWindowEventWatch(SDL_WINDOW_EVENT_WATCH_EARLY, SDL_MetalViewEventWatch, (__bridge void *)(self));
}

- (NSInteger)tag
{
    return SDL_METALVIEW_TAG;
}

- (void)updateDrawableSize
{
    CAMetalLayer *metalLayer = (CAMetalLayer *)self.layer;
    NSSize size = self.bounds.size;
    NSSize backingSize = size;

    if (self.highDPI) {
        /* Note: NSHighResolutionCapable must be set to true in the app's
         * Info.plist in order for the backing size to be high res.
         */
        backingSize = [self convertSizeToBacking:size];
    }

    metalLayer.contentsScale = backingSize.height / size.height;
    metalLayer.drawableSize = NSSizeToCGSize(backingSize);
}

- (NSView *)hitTest:(NSPoint)point
{
    return nil;
}

@end

SDL_MetalView Cocoa_Metal_CreateView(SDL_VideoDevice *_this, SDL_Window *window)
{
    @autoreleasepool {
        SDL_CocoaWindowData *data = (__bridge SDL_CocoaWindowData *)window->internal;
        NSView *view = data.nswindow.contentView;
        BOOL highDPI = (window->flags & SDL_WINDOW_HIGH_PIXEL_DENSITY) != 0;
        BOOL opaque = (window->flags & SDL_WINDOW_TRANSPARENT) == 0;
        Uint32 windowID = SDL_GetWindowID(window);
        SDL3_cocoametalview *newview;
        SDL_MetalView metalview;

        newview = [[SDL3_cocoametalview alloc] initWithFrame:view.frame
                                                    highDPI:highDPI
                                                   windowID:windowID
                                                     opaque:opaque];
        if (newview == nil) {
            SDL_OutOfMemory();
            return NULL;
        }

        [view addSubview:newview];

        // Make sure the drawable size is up to date after attaching the view.
        [newview updateDrawableSize];

        metalview = (SDL_MetalView)CFBridgingRetain(newview);

        return metalview;
    }
}

void Cocoa_Metal_DestroyView(SDL_VideoDevice *_this, SDL_MetalView view)
{
    @autoreleasepool {
        SDL3_cocoametalview *metalview = CFBridgingRelease(view);
        [metalview removeFromSuperview];
    }
}

void *Cocoa_Metal_GetLayer(SDL_VideoDevice *_this, SDL_MetalView view)
{
    @autoreleasepool {
        SDL3_cocoametalview *cocoaview = (__bridge SDL3_cocoametalview *)view;
        return (__bridge void *)cocoaview.layer;
    }
}

#endif // SDL_VIDEO_DRIVER_COCOA && (SDL_VIDEO_VULKAN || SDL_VIDEO_METAL)
