//
//  MGLContext.m
//  OpenGLES
//
//  Created by Le Quyen on 16/10/19.
//  Copyright © 2019 Google. All rights reserved.
//

#import "MGLContext.h"
#import "MGLContext+Private.h"

#include <pthread.h>
#include <vector>

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <EGL/eglext_angle.h>
#include <EGL/eglplatform.h>
#include <common/debug.h>
#import "MGLLayer+Private.h"

namespace
{
struct ThreadLocalInfo
{
    __weak MGLContext *currentContext = nil;
    __weak MGLLayer *currentLayer     = nil;
};

#if TARGET_OS_SIMULATOR
pthread_key_t gThreadSpecificKey;
void ThreadTLSDestructor(void *data)
{
    auto tlsData = reinterpret_cast<ThreadLocalInfo *>(data);
    delete tlsData;
}

#endif

ThreadLocalInfo &CurrentTLS()
{
#if TARGET_OS_SIMULATOR
    // There are some issuess with C++11 TLS could not be compiled on iOS
    // simulator, so we have to fallback to use pthread TLS.
    static pthread_once_t sKeyOnce = PTHREAD_ONCE_INIT;
    pthread_once(&sKeyOnce, [] { pthread_key_create(&gThreadSpecificKey, ThreadTLSDestructor); });

    auto tlsData = reinterpret_cast<ThreadLocalInfo *>(pthread_getspecific(gThreadSpecificKey));
    if (!tlsData)
    {
        tlsData = new ThreadLocalInfo();
        pthread_setspecific(gThreadSpecificKey, tlsData);
    }
    return *tlsData;
#else  // TARGET_OS_SIMULATOR
    static thread_local ThreadLocalInfo tls;
    return tls;
#endif
}

void Throw(NSString *msg)
{
    [NSException raise:@"MGLSurfaceException" format:@"%@", msg];
}
}

// MGLSharegroup implementation
@interface MGLSharegroup ()
@property(atomic) MGLContext *firstContext;
@end

@implementation MGLSharegroup
@end

// MGLContext implementation

@implementation MGLContext

- (id)initWithAPI:(MGLRenderingAPI)api
{
    return [self initWithAPI:api sharegroup:nil];
}

- (id)initWithAPI:(MGLRenderingAPI)api sharegroup:(MGLSharegroup *)sharegroup
{
    if (self = [super init])
    {
        _renderingApi = api;
        _display      = [MGLDisplay defaultDisplay];
        if (sharegroup)
        {
            _sharegroup = sharegroup;
        }
        else
        {
            _sharegroup = [MGLSharegroup new];
        }

        if (!_sharegroup.firstContext)
        {
            _sharegroup.firstContext = self;
        }

        [self initContext];
    }
    return self;
}

- (MGLRenderingAPI)API
{
    return _renderingApi;
}

- (void)dealloc
{
    [self releaseContext];

    _display = nil;
}

- (void)releaseContext
{
    if (eglGetCurrentContext() == _eglContext)
    {
        eglMakeCurrent(_display.eglDisplay, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    }

    if (_dummySurface != EGL_NO_SURFACE)
    {
        eglDestroySurface(_display.eglDisplay, _dummySurface);
        _dummySurface = EGL_NO_SURFACE;
    }

    if (_eglContext != EGL_NO_CONTEXT)
    {
        eglDestroyContext(_display.eglDisplay, _eglContext);
        _eglContext = EGL_NO_CONTEXT;
    }
}

- (void)initContext
{
    // Init config
    std::vector<EGLint> surfaceAttribs = {
        EGL_RED_SIZE,       EGL_DONT_CARE, EGL_GREEN_SIZE,   EGL_DONT_CARE,
        EGL_BLUE_SIZE,      EGL_DONT_CARE, EGL_ALPHA_SIZE,   EGL_DONT_CARE,
        EGL_DEPTH_SIZE,     EGL_DONT_CARE, EGL_STENCIL_SIZE, EGL_DONT_CARE,
        EGL_SAMPLE_BUFFERS, EGL_DONT_CARE, EGL_SAMPLES,      EGL_DONT_CARE,
    };
    surfaceAttribs.push_back(EGL_NONE);
    EGLConfig config;
    EGLint numConfigs;
    if (!eglChooseConfig(_display.eglDisplay, surfaceAttribs.data(), &config, 1, &numConfigs) ||
        numConfigs < 1)
    {
        Throw(@"Failed to call eglChooseConfig()");
    }

    // Init context
    int ctxMajorVersion = 2;
    int ctxMinorVersion = 0;
    switch (_renderingApi)
    {
        case kMGLRenderingAPIOpenGLES1:
            ctxMajorVersion = 1;
            ctxMinorVersion = 0;
            break;
        case kMGLRenderingAPIOpenGLES2:
            ctxMajorVersion = 2;
            ctxMinorVersion = 0;
            break;
        case kMGLRenderingAPIOpenGLES3:
            ctxMajorVersion = 3;
            ctxMinorVersion = 0;
            break;
        default:
            UNREACHABLE();
    }
    EGLint ctxAttribs[] = {EGL_CONTEXT_MAJOR_VERSION, ctxMajorVersion, EGL_CONTEXT_MINOR_VERSION,
                           ctxMinorVersion, EGL_NONE};

    EGLContext sharedContext = EGL_NO_CONTEXT;
    if (_sharegroup.firstContext != self)
    {
        sharedContext = _sharegroup.firstContext.eglContext;
    }
    _eglContext = eglCreateContext(_display.eglDisplay, config, sharedContext, ctxAttribs);
    if (_eglContext == EGL_NO_CONTEXT)
    {
        Throw(@"Failed to call eglCreateContext()");
    }

    // Create dummy surface
    _dummyLayer       = [[CALayer alloc] init];
    _dummyLayer.frame = CGRectMake(0, 0, 1, 1);

    _dummySurface = eglCreateWindowSurface(_display.eglDisplay, config,
                                           (__bridge EGLNativeWindowType)_dummyLayer, nullptr);
    if (_dummySurface == EGL_NO_SURFACE)
    {
        Throw(@"Failed to call eglCreateWindowSurface()");
    }
}

- (BOOL)present:(MGLLayer *)layer
{
    return [layer present];
}

+ (MGLContext *)currentContext
{
    return CurrentTLS().currentContext;
}

+ (MGLLayer *)currentLayer
{
    return CurrentTLS().currentLayer;
}

+ (BOOL)setCurrentContext:(MGLContext *)context
{
    ThreadLocalInfo &tlsData = CurrentTLS();
    if (context)
    {
        return [context setCurrentContextForLayer:tlsData.currentLayer];
    }

    // No context
    tlsData.currentContext   = nil;
    tlsData.currentLayer     = nil;

    return eglMakeCurrent([MGLDisplay defaultDisplay].eglDisplay, EGL_NO_SURFACE, EGL_NO_SURFACE,
                          EGL_NO_CONTEXT);
}

+ (BOOL)setCurrentContext:(MGLContext *)context forLayer:(MGLLayer *)layer
{
    if (context)
    {
        return [context setCurrentContextForLayer:layer];
    }
    return [self setCurrentContext:nil];
}

- (BOOL)setCurrentContextForLayer:(MGLLayer *_Nullable)layer
{
    if (!layer)
    {
        if (eglGetCurrentContext() != _eglContext ||
            eglGetCurrentSurface(EGL_READ) != _dummySurface ||
            eglGetCurrentSurface(EGL_DRAW) != _dummySurface)
        {
            if (!eglMakeCurrent(_display.eglDisplay, _dummySurface, _dummySurface, _eglContext))
            {
                return NO;
            }
        }
    }
    else
    {
        if (![layer setCurrentContext:self])
        {
            return NO;
        }
    }

    ThreadLocalInfo &tlsData = CurrentTLS();
    tlsData.currentContext   = self;
    tlsData.currentLayer     = layer;

    return YES;
}

@end
