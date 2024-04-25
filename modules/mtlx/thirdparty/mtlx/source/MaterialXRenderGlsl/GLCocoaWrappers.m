//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#if defined (__APPLE__)

#import <Cocoa/Cocoa.h>
#import <AppKit/NSApplication.h>
#import "GLCocoaWrappers.h"

void* NSOpenGLChoosePixelFormatWrapper(bool allRenders, int bufferType, int colorSize, int depthFormat,
                                int stencilFormat, int auxBuffers, int accumSize, bool minimumPolicy,
                                bool accelerated, bool mp_safe,  bool stereo, bool supportMultiSample)
{
    // Create local autorelease pool for any objects that need to be autoreleased.
    NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];

    NSOpenGLPixelFormatAttribute list[50];
    int i = 0;
    if (allRenders)
    {
        list[i++] = NSOpenGLPFAAllRenderers;
    }
    if (bufferType == 1) // On screen
    {
        list[i++] = NSOpenGLPFADoubleBuffer;
    }
    if (colorSize != 0)
    {
        list[i++] = NSOpenGLPFAColorSize; list[i++] = colorSize;
    }
    if (depthFormat != 0)
    {
        list[i++] = NSOpenGLPFADepthSize; list[i++] = depthFormat;
    }
    if (stencilFormat != 0)
    {
        list[i++] = NSOpenGLPFAStencilSize; list[i++] = stencilFormat;
    }
    if (auxBuffers != 0)
    {
        list[i++] = NSOpenGLPFAAuxBuffers; list[i++] = auxBuffers;
    }
    if (accumSize != 0)
    {
        list[i++] = NSOpenGLPFAAccumSize; list[i++] = accumSize;
    }
    if (minimumPolicy)
    {
        list[i++] = NSOpenGLPFAMinimumPolicy;
    }
    if (accelerated)
    {
        list[i++] = NSOpenGLPFAAccelerated;
    }

    // Add multisample support to the list of attributes if supported
    //
    int multiSampleAttrIndex = i;
    if (supportMultiSample)
    {
        // Default to 4 samples
        list[i++] = NSOpenGLPFASampleBuffers; list[i++] = TRUE;
        list[i++] = NSOpenGLPFASamples; list[i++] = 4;
    }
#if MAC_OS_X_VERSION_MAX_ALLOWED >= 101000
    list[i++] = NSOpenGLPFAOpenGLProfile;
    list[i++] = NSOpenGLProfileVersion4_1Core;
#else
    list[i++] = NSOpenGLPFAOpenGLProfile;
    list[i++] = NSOpenGLProfileVersion3_2Core;
#endif
	list[ i++] = 0 ;

    NSOpenGLPixelFormat *pixelFormat = [[NSOpenGLPixelFormat alloc] initWithAttributes:list];
    if (!pixelFormat)
    {
        // Try again without multisample, if previous try failed
        //
        list[multiSampleAttrIndex++] = 0;    // NSOpenGLPFASampleBuffers
        list[multiSampleAttrIndex++] = 0;    // NSOpenGLPFASampleBuffers value
        list[multiSampleAttrIndex++] = 0;    // NSOpenGLPFASamplesB
        list[multiSampleAttrIndex++] = 0;    // NSOpenGLPFASamples value

        pixelFormat = [[NSOpenGLPixelFormat alloc] initWithAttributes:list];
    }

    // Free up memory
    [pool release];

    return pixelFormat;
}

void NSOpenGLReleasePixelFormat(void* pPixelFormat)
{
    NSOpenGLPixelFormat *pixelFormat = (NSOpenGLPixelFormat*)pPixelFormat;
    [pixelFormat release];
}

void NSOpenGLReleaseContext(void* pContext)
{
    NSOpenGLContext *context = (NSOpenGLContext*)pContext;
    [context release];
}

void* NSOpenGLCreateContextWrapper(void* pPixelFormat, void *pDummyContext)
{
    NSOpenGLPixelFormat *pixelFormat = (NSOpenGLPixelFormat*)pPixelFormat;
    NSOpenGLContext *dummyContext = (NSOpenGLContext*)pDummyContext;
    NSOpenGLContext *context = [[NSOpenGLContext alloc] initWithFormat:pixelFormat
                                    shareContext:dummyContext];

    return context;
}

void NSOpenGLSetDrawable(void* pContext, void* pWindow)
{
    // Create local autorelease pool for any objects that need to be autoreleased.
    NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];

    NSOpenGLContext *context = (NSOpenGLContext*)pContext;
    NSWindow *window = (NSWindow*)pWindow;
    NSView *view = [window contentView];
    [context setView:view];

    // Free up memory
    [pool release];
}

void NSOpenGLMakeCurrent(void* pContext)
{
    NSOpenGLContext* context = (NSOpenGLContext*)pContext;
    [context makeCurrentContext];
}

void* NSOpenGLGetCurrentContextWrapper()
{
    return [NSOpenGLContext currentContext];
}

void NSOpenGLSwapBuffers(void* pContext)
{
    NSOpenGLContext* context = (NSOpenGLContext*)pContext;
    [context flushBuffer];
}

void NSOpenGLClearCurrentContext()
{
    [NSOpenGLContext clearCurrentContext];
}

void NSOpenGLDestroyContext(void** pContext)
{
    NSOpenGLContext* context = (NSOpenGLContext*)*pContext;
    [context release];
    *pContext = NULL;
}

void NSOpenGLDestroyCurrentContext(void** pContext)
{
    [NSOpenGLContext clearCurrentContext];
    NSOpenGLContext* context = (NSOpenGLContext*)*pContext;
    [context release];
    *pContext = NULL;
}


void NSOpenGLClearDrawable(void* pContext)
{
    // Create local autorelease pool for any objects that need to be autoreleased.
    NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];

    NSOpenGLContext* context =(NSOpenGLContext*) pContext;
    [context clearDrawable];

    // Free up memory
    [pool release];
}

void NSOpenGLDescribePixelFormat(void* pPixelFormat, int attrib, int* vals)
{
    NSOpenGLPixelFormat *pixelFormat = (NSOpenGLPixelFormat*)pPixelFormat;
    [pixelFormat getValues:vals forAttribute:attrib forVirtualScreen:0];
}

void NSOpenGLGetInteger(void* pContext, int param, int* vals)
{
  NSOpenGLContext* context = (NSOpenGLContext*)pContext;
    [context getValues:vals forParameter:(NSOpenGLContextParameter)param];
}

void NSOpenGLUpdate(void* pContext)
{
  NSOpenGLContext* context = (NSOpenGLContext*)pContext;
    [context update];
}

void* NSOpenGLGetWindow(void* pView)
{
  NSView *view = (NSView*)pView;
    return [view window];
}

void NSOpenGLInitializeGLLibrary()
{
    // Create local autorelease pool for any objects that need to be autoreleased (needed in batch mode).
    NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];
    NSOpenGLPixelFormatAttribute attrib[] = {NSOpenGLPFAAllRenderers, NSOpenGLPFADoubleBuffer, 0};
    NSOpenGLPixelFormat *dummyPixelFormat = [[NSOpenGLPixelFormat alloc] initWithAttributes:attrib];
    if (nil != dummyPixelFormat)
    {
        [dummyPixelFormat release];
    }
    [pool release];
}

#endif
