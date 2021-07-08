//
// Copyright 2019 Le Hoang Quyen. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#import "MGLKView+Private.h"
#import "MGLKViewController+Private.h"

#include <libGLESv2/entry_points_gles_2_0_autogen.h>

namespace
{
void Throw(NSString *msg)
{
    [NSException raise:@"MGLSurfaceException" format:@"%@", msg];
}
}

@implementation MGLKView

- (id)initWithCoder:(NSCoder *)coder
{
    if (self = [super initWithCoder:coder])
    {
#if TARGET_OS_OSX
        self.wantsLayer       = YES;
        self.autoresizingMask = kCALayerWidthSizable | kCALayerHeightSizable;
#else
        self.autoresizingMask = UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight;
        self.enableSetNeedsDisplay = YES;
#endif
    }
    return self;
}

- (id)initWithFrame:(CGRect)frame
{
    if (self = [super initWithFrame:frame])
    {
#if TARGET_OS_OSX
        self.wantsLayer       = YES;
        self.autoresizingMask = kCALayerWidthSizable | kCALayerHeightSizable;
#else
        self.autoresizingMask = UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight;
        self.enableSetNeedsDisplay = YES;
#endif
    }
    return self;
}

- (id)initWithFrame:(CGRect)frame context:(MGLContext *)context
{
    if (self = [self initWithFrame:frame])
    {
        [self setContext:context];
    }
    return self;
}

- (void)dealloc
{
    _context = nil;
}

#if TARGET_OS_OSX
- (CALayer *)makeBackingLayer
{
    return [MGLLayer layer];
}
#else
+ (Class)layerClass
{
    return MGLLayer.class;
}
#endif

- (MGLLayer *)glLayer
{
    return static_cast<MGLLayer *>(self.layer);
}

- (void)setContext:(MGLContext *)context
{
    if (_drawing)
    {
        Throw(@"Changing GL context when drawing is not allowed");
    }

    _context = context;
}

#if TARGET_OS_IOS || TARGET_OS_TV
- (void)setNeedsDisplay
{
    if (_enableSetNeedsDisplay)
    {
        [super setNeedsDisplay];
    }
}

- (void)setNeedsDisplayInRect:(CGRect)invalidRect
{
    if (_enableSetNeedsDisplay)
    {
        [super setNeedsDisplayInRect:invalidRect];
    }
}
#endif  // TARGET_OS_IOS || TARGET_OS_TV

- (void)setRetainedBacking:(BOOL)retainedBacking
{
    self.glLayer.retainedBacking = _retainedBacking = retainedBacking;
}

- (void)setDrawableColorFormat:(MGLDrawableColorFormat)drawableColorFormat
{
    self.glLayer.drawableColorFormat = _drawableColorFormat = drawableColorFormat;
}

- (void)setDrawableDepthFormat:(MGLDrawableDepthFormat)drawableDepthFormat
{
    self.glLayer.drawableDepthFormat = _drawableDepthFormat = drawableDepthFormat;
}

- (void)setDrawableStencilFormat:(MGLDrawableStencilFormat)drawableStencilFormat
{
    self.glLayer.drawableStencilFormat = _drawableStencilFormat = drawableStencilFormat;
}

- (void)setDrawableMultisample:(MGLDrawableMultisample)drawableMultisample
{
    self.glLayer.drawableMultisample = _drawableMultisample = drawableMultisample;
}

- (void)display
{
    [self displayAndCapture:nullptr];
}

- (void)bindDrawable
{
    [self.glLayer bindDefaultFrameBuffer];
}

- (CGSize)drawableSize
{
    if (!self.layer)
    {
        CGSize zero = {0};
        return zero;
    }
    return self.glLayer.drawableSize;
}

- (NSInteger)drawableWidth
{
    return self.drawableSize.width;
}

- (NSInteger)drawableHeight
{
    return self.drawableSize.height;
}

- (uint32_t)defaultOpenGLFrameBufferID
{
    return self.glLayer.defaultOpenGLFrameBufferID;
}

#if TARGET_OS_OSX
- (void)viewDidMoveToWindow
{
    [super viewDidMoveToWindow];
#else
- (void)didMoveToWindow
{
    [super didMoveToWindow];
#endif

    // notify view controller
    if (_controller)
    {
        [_controller viewDidMoveToWindow];
    }
}

- (void)drawRect:(CGRect)rect
{
    if (_delegate)
    {
        [_delegate mglkView:self drawInRect:rect];
    }
}

- (void)displayAndCapture:(uint8_t **)pPixels
{
    _drawing = YES;
    if (_context)
    {
        if (![MGLContext setCurrentContext:_context forLayer:self.glLayer])
        {
            Throw(@"Failed to setCurrentContext");
        }
    }

    [self drawRect:self.bounds];

    if (pPixels)
    {
        // Frame capture request
        int width  = static_cast<int>(self.drawableSize.width);
        int height = static_cast<int>(self.drawableSize.height);
        if (width && height)
        {
            // Capture framebuffer's content
            *pPixels = new uint8_t[4 * width * height];
            GLint prevPackAlignment;
            gl::GetIntegerv(GL_PACK_ALIGNMENT, &prevPackAlignment);
            gl::PixelStorei(GL_PACK_ALIGNMENT, 1);

            // Clear errors
            gl::GetError();

            gl::ReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, *pPixels);

            gl::PixelStorei(GL_PACK_ALIGNMENT, prevPackAlignment);

            // Clear errors
            gl::GetError();
        }
        else
        {
            *pPixels = nullptr;
        }
    }  // if (pPixels)

    if (![_context present:self.glLayer])
    {
        Throw(@"Failed to present framebuffer");
    }
    _drawing = NO;
}

#if TARGET_OS_IOS || TARGET_OS_TV
static void freeImageData(void *info, const void *data, size_t size)
{
    delete[] static_cast<uint8_t *>(info);
}

- (UIImage *)snapshot
{
    uint8_t *pixels = nullptr;
    [self displayAndCapture:&pixels];

    if (!pixels)
    {
        return nil;
    }

    int width      = static_cast<int>(self.drawableSize.width);
    int height     = static_cast<int>(self.drawableSize.height);
    size_t dataLen = width * height * 4;

    uint8_t *flippedPixels = new uint8_t[dataLen];
    for (int y1 = 0; y1 < height; y1++)
    {
        for (int x1 = 0; x1 < width * 4; x1++)
        {
            flippedPixels[(height - 1 - y1) * width * 4 + x1] = pixels[y1 * 4 * width + x1];
        }
    }
    delete[] pixels;

    CGDataProviderRef provider =
        CGDataProviderCreateWithData(flippedPixels, flippedPixels, dataLen, freeImageData);
    int bitsPerComponent                   = 8;
    int bitsPerPixel                       = 32;
    int bytesPerRow                        = 4 * width;
    CGColorSpaceRef colorSpaceRef          = CGColorSpaceCreateDeviceRGB();
    CGBitmapInfo bitmapInfo                = kCGBitmapByteOrderDefault | kCGImageAlphaNoneSkipLast;
    CGColorRenderingIntent renderingIntent = kCGRenderingIntentDefault;
    CGImageRef imageRef =
        CGImageCreate(width, height, bitsPerComponent, bitsPerPixel, bytesPerRow, colorSpaceRef,
                      bitmapInfo, provider, NULL, NO, renderingIntent);
    CGColorSpaceRelease(colorSpaceRef);
    CGDataProviderRelease(provider);
    UIImage *image = [UIImage imageWithCGImage:imageRef scale:1 orientation:UIImageOrientationUp];

    CGImageRelease(imageRef);
    return image;
}
#endif  // TARGET_OS_IOS || TARGET_OS_TV

@end
