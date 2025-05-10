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
#include "SDL_internal.h"

#if defined(SDL_VIDEO_DRIVER_UIKIT) && (defined(SDL_VIDEO_OPENGL_ES) || defined(SDL_VIDEO_OPENGL_ES2))

#include <OpenGLES/EAGLDrawable.h>
#include <OpenGLES/ES2/glext.h>
#import "SDL_uikitopenglview.h"
#include "SDL_uikitwindow.h"

@implementation SDL_uikitopenglview
{
    // The renderbuffer and framebuffer used to render to this layer.
    GLuint viewRenderbuffer, viewFramebuffer;

    // The depth buffer that is attached to viewFramebuffer, if it exists.
    GLuint depthRenderbuffer;

    GLenum colorBufferFormat;

    // format of depthRenderbuffer
    GLenum depthBufferFormat;

    // The framebuffer and renderbuffer used for rendering with MSAA.
    GLuint msaaFramebuffer, msaaRenderbuffer;

    // The number of MSAA samples.
    int samples;

    BOOL retainedBacking;
}

@synthesize context;
@synthesize backingWidth;
@synthesize backingHeight;

+ (Class)layerClass
{
    return [CAEAGLLayer class];
}

- (instancetype)initWithFrame:(CGRect)frame
                        scale:(CGFloat)scale
                retainBacking:(BOOL)retained
                        rBits:(int)rBits
                        gBits:(int)gBits
                        bBits:(int)bBits
                        aBits:(int)aBits
                    depthBits:(int)depthBits
                  stencilBits:(int)stencilBits
                         sRGB:(BOOL)sRGB
                 multisamples:(int)multisamples
                      context:(EAGLContext *)glcontext
{
    if ((self = [super initWithFrame:frame])) {
        const BOOL useStencilBuffer = (stencilBits != 0);
        const BOOL useDepthBuffer = (depthBits != 0);
        NSString *colorFormat = nil;

        context = glcontext;
        samples = multisamples;
        retainedBacking = retained;

        if (!context || ![EAGLContext setCurrentContext:context]) {
            SDL_SetError("Could not create OpenGL ES drawable (could not make context current)");
            return nil;
        }

        if (samples > 0) {
            GLint maxsamples = 0;
            glGetIntegerv(GL_MAX_SAMPLES, &maxsamples);

            // Clamp the samples to the max supported count.
            samples = SDL_min(samples, maxsamples);
        }

        if (sRGB) {
            colorFormat = kEAGLColorFormatSRGBA8;
            colorBufferFormat = GL_SRGB8_ALPHA8;
        } else if (rBits >= 8 || gBits >= 8 || bBits >= 8 || aBits > 0) {
            // if user specifically requests rbg888 or some color format higher than 16bpp
            colorFormat = kEAGLColorFormatRGBA8;
            colorBufferFormat = GL_RGBA8;
        } else {
            // default case (potentially faster)
            colorFormat = kEAGLColorFormatRGB565;
            colorBufferFormat = GL_RGB565;
        }

        CAEAGLLayer *eaglLayer = (CAEAGLLayer *)self.layer;

        eaglLayer.opaque = YES;
        eaglLayer.drawableProperties = @{
            kEAGLDrawablePropertyRetainedBacking : @(retained),
            kEAGLDrawablePropertyColorFormat : colorFormat
        };

        // Set the appropriate scale (for retina display support)
        self.contentScaleFactor = scale;

        // Create the color Renderbuffer Object
        glGenRenderbuffers(1, &viewRenderbuffer);
        glBindRenderbuffer(GL_RENDERBUFFER, viewRenderbuffer);

        if (![context renderbufferStorage:GL_RENDERBUFFER fromDrawable:eaglLayer]) {
            SDL_SetError("Failed to create OpenGL ES drawable");
            return nil;
        }

        // Create the Framebuffer Object
        glGenFramebuffers(1, &viewFramebuffer);
        glBindFramebuffer(GL_FRAMEBUFFER, viewFramebuffer);

        // attach the color renderbuffer to the FBO
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, viewRenderbuffer);

        glGetRenderbufferParameteriv(GL_RENDERBUFFER, GL_RENDERBUFFER_WIDTH, &backingWidth);
        glGetRenderbufferParameteriv(GL_RENDERBUFFER, GL_RENDERBUFFER_HEIGHT, &backingHeight);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            SDL_SetError("Failed creating OpenGL ES framebuffer");
            return nil;
        }

        /* When MSAA is used we'll use a separate framebuffer for rendering to,
         * since we'll need to do an explicit MSAA resolve before presenting. */
        if (samples > 0) {
            glGenFramebuffers(1, &msaaFramebuffer);
            glBindFramebuffer(GL_FRAMEBUFFER, msaaFramebuffer);

            glGenRenderbuffers(1, &msaaRenderbuffer);
            glBindRenderbuffer(GL_RENDERBUFFER, msaaRenderbuffer);

            glRenderbufferStorageMultisample(GL_RENDERBUFFER, samples, colorBufferFormat, backingWidth, backingHeight);

            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, msaaRenderbuffer);
        }

        if (useDepthBuffer || useStencilBuffer) {
            if (useStencilBuffer) {
                // Apparently you need to pack stencil and depth into one buffer.
                depthBufferFormat = GL_DEPTH24_STENCIL8_OES;
            } else if (useDepthBuffer) {
                /* iOS only uses 32-bit float (exposed as fixed point 24-bit)
                 * depth buffers. */
                depthBufferFormat = GL_DEPTH_COMPONENT24_OES;
            }

            glGenRenderbuffers(1, &depthRenderbuffer);
            glBindRenderbuffer(GL_RENDERBUFFER, depthRenderbuffer);

            if (samples > 0) {
                glRenderbufferStorageMultisample(GL_RENDERBUFFER, samples, depthBufferFormat, backingWidth, backingHeight);
            } else {
                glRenderbufferStorage(GL_RENDERBUFFER, depthBufferFormat, backingWidth, backingHeight);
            }

            if (useDepthBuffer) {
                glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthRenderbuffer);
            }
            if (useStencilBuffer) {
                glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_STENCIL_ATTACHMENT, GL_RENDERBUFFER, depthRenderbuffer);
            }
        }

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            SDL_SetError("Failed creating OpenGL ES framebuffer");
            return nil;
        }

        glBindRenderbuffer(GL_RENDERBUFFER, viewRenderbuffer);

        [self setDebugLabels];
    }

    return self;
}

- (GLuint)drawableRenderbuffer
{
    return viewRenderbuffer;
}

- (GLuint)drawableFramebuffer
{
    // When MSAA is used, the MSAA draw framebuffer is used for drawing.
    if (msaaFramebuffer) {
        return msaaFramebuffer;
    } else {
        return viewFramebuffer;
    }
}

- (GLuint)msaaResolveFramebuffer
{
    /* When MSAA is used, the MSAA draw framebuffer is used for drawing and the
     * view framebuffer is used as a MSAA resolve framebuffer. */
    if (msaaFramebuffer) {
        return viewFramebuffer;
    } else {
        return 0;
    }
}

- (void)updateFrame
{
    GLint prevRenderbuffer = 0;
    glGetIntegerv(GL_RENDERBUFFER_BINDING, &prevRenderbuffer);

    glBindRenderbuffer(GL_RENDERBUFFER, viewRenderbuffer);
    [context renderbufferStorage:GL_RENDERBUFFER fromDrawable:(CAEAGLLayer *)self.layer];

    glGetRenderbufferParameteriv(GL_RENDERBUFFER, GL_RENDERBUFFER_WIDTH, &backingWidth);
    glGetRenderbufferParameteriv(GL_RENDERBUFFER, GL_RENDERBUFFER_HEIGHT, &backingHeight);

    if (msaaRenderbuffer != 0) {
        glBindRenderbuffer(GL_RENDERBUFFER, msaaRenderbuffer);
        glRenderbufferStorageMultisample(GL_RENDERBUFFER, samples, colorBufferFormat, backingWidth, backingHeight);
    }

    if (depthRenderbuffer != 0) {
        glBindRenderbuffer(GL_RENDERBUFFER, depthRenderbuffer);

        if (samples > 0) {
            glRenderbufferStorageMultisample(GL_RENDERBUFFER, samples, depthBufferFormat, backingWidth, backingHeight);
        } else {
            glRenderbufferStorage(GL_RENDERBUFFER, depthBufferFormat, backingWidth, backingHeight);
        }
    }

    glBindRenderbuffer(GL_RENDERBUFFER, prevRenderbuffer);
}

- (void)setDebugLabels
{
    if (viewFramebuffer != 0) {
        glLabelObjectEXT(GL_FRAMEBUFFER, viewFramebuffer, 0, "context FBO");
    }

    if (viewRenderbuffer != 0) {
        glLabelObjectEXT(GL_RENDERBUFFER, viewRenderbuffer, 0, "context color buffer");
    }

    if (depthRenderbuffer != 0) {
        if (depthBufferFormat == GL_DEPTH24_STENCIL8_OES) {
            glLabelObjectEXT(GL_RENDERBUFFER, depthRenderbuffer, 0, "context depth-stencil buffer");
        } else {
            glLabelObjectEXT(GL_RENDERBUFFER, depthRenderbuffer, 0, "context depth buffer");
        }
    }

    if (msaaFramebuffer != 0) {
        glLabelObjectEXT(GL_FRAMEBUFFER, msaaFramebuffer, 0, "context MSAA FBO");
    }

    if (msaaRenderbuffer != 0) {
        glLabelObjectEXT(GL_RENDERBUFFER, msaaRenderbuffer, 0, "context MSAA renderbuffer");
    }
}

- (void)swapBuffers
{
    if (msaaFramebuffer) {
        const GLenum attachments[] = { GL_COLOR_ATTACHMENT0 };

        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, viewFramebuffer);

        /* OpenGL ES 3+ provides explicit MSAA resolves via glBlitFramebuffer.
         * In OpenGL ES 1 and 2, MSAA resolves must be done via an extension. */
        if (context.API >= kEAGLRenderingAPIOpenGLES3) {
            int w = backingWidth;
            int h = backingHeight;
            glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, GL_COLOR_BUFFER_BIT, GL_NEAREST);

            if (!retainedBacking) {
                // Discard the contents of the MSAA drawable color buffer.
                glInvalidateFramebuffer(GL_READ_FRAMEBUFFER, 1, attachments);
            }
        } else {
            glResolveMultisampleFramebufferAPPLE();

            if (!retainedBacking) {
                glDiscardFramebufferEXT(GL_READ_FRAMEBUFFER, 1, attachments);
            }
        }

        /* We assume the "drawable framebuffer" (MSAA draw framebuffer) was
         * previously bound... */
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, msaaFramebuffer);
    }

    /* viewRenderbuffer should always be bound here. Code that binds something
     * else is responsible for rebinding viewRenderbuffer, to reduce duplicate
     * state changes. */
    [context presentRenderbuffer:GL_RENDERBUFFER];
}

- (void)layoutSubviews
{
    [super layoutSubviews];

    int width = (int)(self.bounds.size.width * self.contentScaleFactor);
    int height = (int)(self.bounds.size.height * self.contentScaleFactor);

    // Update the color and depth buffer storage if the layer size has changed.
    if (width != backingWidth || height != backingHeight) {
        EAGLContext *prevContext = [EAGLContext currentContext];
        if (prevContext != context) {
            [EAGLContext setCurrentContext:context];
        }

        [self updateFrame];

        if (prevContext != context) {
            [EAGLContext setCurrentContext:prevContext];
        }
    }
}

- (void)destroyFramebuffer
{
    if (viewFramebuffer != 0) {
        glDeleteFramebuffers(1, &viewFramebuffer);
        viewFramebuffer = 0;
    }

    if (viewRenderbuffer != 0) {
        glDeleteRenderbuffers(1, &viewRenderbuffer);
        viewRenderbuffer = 0;
    }

    if (depthRenderbuffer != 0) {
        glDeleteRenderbuffers(1, &depthRenderbuffer);
        depthRenderbuffer = 0;
    }

    if (msaaFramebuffer != 0) {
        glDeleteFramebuffers(1, &msaaFramebuffer);
        msaaFramebuffer = 0;
    }

    if (msaaRenderbuffer != 0) {
        glDeleteRenderbuffers(1, &msaaRenderbuffer);
        msaaRenderbuffer = 0;
    }
}

- (void)dealloc
{
    if (context && context == [EAGLContext currentContext]) {
        [self destroyFramebuffer];
        [EAGLContext setCurrentContext:nil];
    }
}

@end

#endif // SDL_VIDEO_DRIVER_UIKIT
