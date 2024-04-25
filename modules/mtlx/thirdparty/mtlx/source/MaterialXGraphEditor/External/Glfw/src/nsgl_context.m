//========================================================================
// GLFW 3.4 macOS - www.glfw.org
//------------------------------------------------------------------------
// Copyright (c) 2009-2019 Camilla Löwy <elmindreda@glfw.org>
//
// This software is provided 'as-is', without any express or implied
// warranty. In no event will the authors be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you must not
//    claim that you wrote the original software. If you use this software
//    in a product, an acknowledgment in the product documentation would
//    be appreciated but is not required.
//
// 2. Altered source versions must be plainly marked as such, and must not
//    be misrepresented as being the original software.
//
// 3. This notice may not be removed or altered from any source
//    distribution.
//
//========================================================================
// It is fine to use C99 in this file because it will not be built with VS
//========================================================================

#include "internal.h"

#include <unistd.h>
#include <math.h>

static void makeContextCurrentNSGL(_GLFWwindow* window)
{
    @autoreleasepool {

    if (window)
        [window->context.nsgl.object makeCurrentContext];
    else
        [NSOpenGLContext clearCurrentContext];

    _glfwPlatformSetTls(&_glfw.contextSlot, window);

    } // autoreleasepool
}

static void swapBuffersNSGL(_GLFWwindow* window)
{
    @autoreleasepool {

    // HACK: Simulate vsync with usleep as NSGL swap interval does not apply to
    //       windows with a non-visible occlusion state
    if (!([window->ns.object occlusionState] & NSWindowOcclusionStateVisible))
    {
        int interval = 0;
        [window->context.nsgl.object getValues:&interval
                                  forParameter:NSOpenGLContextParameterSwapInterval];

        if (interval > 0)
        {
            const double framerate = 60.0;
            const uint64_t frequency = _glfwPlatformGetTimerFrequency();
            const uint64_t value = _glfwPlatformGetTimerValue();

            const double elapsed = value / (double) frequency;
            const double period = 1.0 / framerate;
            const double delay = period - fmod(elapsed, period);

            usleep(floorl(delay * 1e6));
        }
    }

    [window->context.nsgl.object flushBuffer];

    } // autoreleasepool
}

static void swapIntervalNSGL(int interval)
{
    @autoreleasepool {

    _GLFWwindow* window = _glfwPlatformGetTls(&_glfw.contextSlot);
    if (window)
    {
        [window->context.nsgl.object setValues:&interval
                                  forParameter:NSOpenGLContextParameterSwapInterval];
    }

    } // autoreleasepool
}

static int extensionSupportedNSGL(const char* extension)
{
    // There are no NSGL extensions
    return GLFW_FALSE;
}

static GLFWglproc getProcAddressNSGL(const char* procname)
{
    CFStringRef symbolName = CFStringCreateWithCString(kCFAllocatorDefault,
                                                       procname,
                                                       kCFStringEncodingASCII);

    GLFWglproc symbol = CFBundleGetFunctionPointerForName(_glfw.nsgl.framework,
                                                          symbolName);

    CFRelease(symbolName);

    return symbol;
}

static void destroyContextNSGL(_GLFWwindow* window)
{
    @autoreleasepool {

    [window->context.nsgl.pixelFormat release];
    window->context.nsgl.pixelFormat = nil;

    [window->context.nsgl.object release];
    window->context.nsgl.object = nil;

    } // autoreleasepool
}


//////////////////////////////////////////////////////////////////////////
//////                       GLFW internal API                      //////
//////////////////////////////////////////////////////////////////////////

// Initialize OpenGL support
//
GLFWbool _glfwInitNSGL(void)
{
    if (_glfw.nsgl.framework)
        return GLFW_TRUE;

    _glfw.nsgl.framework =
        CFBundleGetBundleWithIdentifier(CFSTR("com.apple.opengl"));
    if (_glfw.nsgl.framework == NULL)
    {
        _glfwInputError(GLFW_API_UNAVAILABLE,
                        "NSGL: Failed to locate OpenGL framework");
        return GLFW_FALSE;
    }

    return GLFW_TRUE;
}

// Terminate OpenGL support
//
void _glfwTerminateNSGL(void)
{
}

// Create the OpenGL context
//
GLFWbool _glfwCreateContextNSGL(_GLFWwindow* window,
                                const _GLFWctxconfig* ctxconfig,
                                const _GLFWfbconfig* fbconfig)
{
    if (ctxconfig->client == GLFW_OPENGL_ES_API)
    {
        _glfwInputError(GLFW_API_UNAVAILABLE,
                        "NSGL: OpenGL ES is not available on macOS");
        return GLFW_FALSE;
    }

    if (ctxconfig->major > 2)
    {
        if (ctxconfig->major == 3 && ctxconfig->minor < 2)
        {
            _glfwInputError(GLFW_VERSION_UNAVAILABLE,
                            "NSGL: The targeted version of macOS does not support OpenGL 3.0 or 3.1 but may support 3.2 and above");
            return GLFW_FALSE;
        }
    }

    // Context robustness modes (GL_KHR_robustness) are not yet supported by
    // macOS but are not a hard constraint, so ignore and continue

    // Context release behaviors (GL_KHR_context_flush_control) are not yet
    // supported by macOS but are not a hard constraint, so ignore and continue

    // Debug contexts (GL_KHR_debug) are not yet supported by macOS but are not
    // a hard constraint, so ignore and continue

    // No-error contexts (GL_KHR_no_error) are not yet supported by macOS but
    // are not a hard constraint, so ignore and continue

#define addAttrib(a) \
{ \
    assert((size_t) index < sizeof(attribs) / sizeof(attribs[0])); \
    attribs[index++] = a; \
}
#define setAttrib(a, v) { addAttrib(a); addAttrib(v); }

    NSOpenGLPixelFormatAttribute attribs[40];
    int index = 0;

    addAttrib(NSOpenGLPFAAccelerated);
    addAttrib(NSOpenGLPFAClosestPolicy);

    if (ctxconfig->nsgl.offline)
    {
        addAttrib(NSOpenGLPFAAllowOfflineRenderers);
        // NOTE: This replaces the NSSupportsAutomaticGraphicsSwitching key in
        //       Info.plist for unbundled applications
        // HACK: This assumes that NSOpenGLPixelFormat will remain
        //       a straightforward wrapper of its CGL counterpart
        addAttrib(kCGLPFASupportsAutomaticGraphicsSwitching);
    }

#if MAC_OS_X_VERSION_MAX_ALLOWED >= 101000
    if (ctxconfig->major >= 4)
    {
        setAttrib(NSOpenGLPFAOpenGLProfile, NSOpenGLProfileVersion4_1Core);
    }
    else
#endif /*MAC_OS_X_VERSION_MAX_ALLOWED*/
    if (ctxconfig->major >= 3)
    {
        setAttrib(NSOpenGLPFAOpenGLProfile, NSOpenGLProfileVersion3_2Core);
    }

    if (ctxconfig->major <= 2)
    {
        if (fbconfig->auxBuffers != GLFW_DONT_CARE)
            setAttrib(NSOpenGLPFAAuxBuffers, fbconfig->auxBuffers);

        if (fbconfig->accumRedBits != GLFW_DONT_CARE &&
            fbconfig->accumGreenBits != GLFW_DONT_CARE &&
            fbconfig->accumBlueBits != GLFW_DONT_CARE &&
            fbconfig->accumAlphaBits != GLFW_DONT_CARE)
        {
            const int accumBits = fbconfig->accumRedBits +
                                  fbconfig->accumGreenBits +
                                  fbconfig->accumBlueBits +
                                  fbconfig->accumAlphaBits;

            setAttrib(NSOpenGLPFAAccumSize, accumBits);
        }
    }

    if (fbconfig->redBits != GLFW_DONT_CARE &&
        fbconfig->greenBits != GLFW_DONT_CARE &&
        fbconfig->blueBits != GLFW_DONT_CARE)
    {
        int colorBits = fbconfig->redBits +
                        fbconfig->greenBits +
                        fbconfig->blueBits;

        // macOS needs non-zero color size, so set reasonable values
        if (colorBits == 0)
            colorBits = 24;
        else if (colorBits < 15)
            colorBits = 15;

        setAttrib(NSOpenGLPFAColorSize, colorBits);
    }

    if (fbconfig->alphaBits != GLFW_DONT_CARE)
        setAttrib(NSOpenGLPFAAlphaSize, fbconfig->alphaBits);

    if (fbconfig->depthBits != GLFW_DONT_CARE)
        setAttrib(NSOpenGLPFADepthSize, fbconfig->depthBits);

    if (fbconfig->stencilBits != GLFW_DONT_CARE)
        setAttrib(NSOpenGLPFAStencilSize, fbconfig->stencilBits);

    if (fbconfig->stereo)
    {
#if MAC_OS_X_VERSION_MAX_ALLOWED >= 101200
        _glfwInputError(GLFW_FORMAT_UNAVAILABLE,
                        "NSGL: Stereo rendering is deprecated");
        return GLFW_FALSE;
#else
        addAttrib(NSOpenGLPFAStereo);
#endif
    }

    if (fbconfig->doublebuffer)
        addAttrib(NSOpenGLPFADoubleBuffer);

    if (fbconfig->samples != GLFW_DONT_CARE)
    {
        if (fbconfig->samples == 0)
        {
            setAttrib(NSOpenGLPFASampleBuffers, 0);
        }
        else
        {
            setAttrib(NSOpenGLPFASampleBuffers, 1);
            setAttrib(NSOpenGLPFASamples, fbconfig->samples);
        }
    }

    // NOTE: All NSOpenGLPixelFormats on the relevant cards support sRGB
    //       framebuffer, so there's no need (and no way) to request it

    addAttrib(0);

#undef addAttrib
#undef setAttrib

    window->context.nsgl.pixelFormat =
        [[NSOpenGLPixelFormat alloc] initWithAttributes:attribs];
    if (window->context.nsgl.pixelFormat == nil)
    {
        _glfwInputError(GLFW_FORMAT_UNAVAILABLE,
                        "NSGL: Failed to find a suitable pixel format");
        return GLFW_FALSE;
    }

    NSOpenGLContext* share = nil;

    if (ctxconfig->share)
        share = ctxconfig->share->context.nsgl.object;

    window->context.nsgl.object =
        [[NSOpenGLContext alloc] initWithFormat:window->context.nsgl.pixelFormat
                                   shareContext:share];
    if (window->context.nsgl.object == nil)
    {
        _glfwInputError(GLFW_VERSION_UNAVAILABLE,
                        "NSGL: Failed to create OpenGL context");
        return GLFW_FALSE;
    }

    if (fbconfig->transparent)
    {
        GLint opaque = 0;
        [window->context.nsgl.object setValues:&opaque
                                  forParameter:NSOpenGLContextParameterSurfaceOpacity];
    }

    [window->ns.view setWantsBestResolutionOpenGLSurface:window->ns.retina];

    [window->context.nsgl.object setView:window->ns.view];

    window->context.makeCurrent = makeContextCurrentNSGL;
    window->context.swapBuffers = swapBuffersNSGL;
    window->context.swapInterval = swapIntervalNSGL;
    window->context.extensionSupported = extensionSupportedNSGL;
    window->context.getProcAddress = getProcAddressNSGL;
    window->context.destroy = destroyContextNSGL;

    return GLFW_TRUE;
}


//////////////////////////////////////////////////////////////////////////
//////                        GLFW native API                       //////
//////////////////////////////////////////////////////////////////////////

GLFWAPI id glfwGetNSGLContext(GLFWwindow* handle)
{
    _GLFWwindow* window = (_GLFWwindow*) handle;
    _GLFW_REQUIRE_INIT_OR_RETURN(nil);

    if (window->context.client == GLFW_NO_API)
    {
        _glfwInputError(GLFW_NO_WINDOW_CONTEXT, NULL);
        return nil;
    }

    return window->context.nsgl.object;
}

