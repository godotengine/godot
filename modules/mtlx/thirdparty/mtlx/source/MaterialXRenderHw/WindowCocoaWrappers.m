//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#if defined (__APPLE__)

#ifndef TARGET_OS_IOS

#import <Cocoa/Cocoa.h>
#import <AppKit/NSApplication.h>
#import <MaterialXRenderHw/WindowCocoaWrappers.h>

void* NSUtilGetView(void* pWindow)
{
    NSWindow* window = (NSWindow*)pWindow;
	NSView* view =  [window contentView];
	return (void*)view;
}

void* NSUtilCreateWindow(unsigned int width, unsigned int height, const char* title, bool batchMode)
{
	// In batch mode, ensure that Cocoa is initialized
	if (batchMode)
	{
		NSApplicationLoad();
	}

	// Create local autorelease pool for any objects that need to be autoreleased.
	NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];

	NSWindow *window = [[NSWindow alloc] initWithContentRect:NSMakeRect(0, 0, width, height)
		styleMask:NSWindowStyleMaskTitled  | NSWindowStyleMaskClosable	| NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable
		backing:NSBackingStoreBuffered defer:NO];
	NSString *string = [NSString stringWithUTF8String:title];

	[window setTitle:string];
	[window setAlphaValue:0.0];

	// Free up memory
	[pool release];

	return (void*)window;
}

void NSUtilShowWindow(void* pWindow)
{
    NSWindow* window = (NSWindow*) pWindow;
	[window orderFront:window];
}

void NSUtilHideWindow(void* pWindow)
{
    NSWindow* window = (NSWindow*)pWindow;
	[window orderOut:window];
}

void NSUtilSetFocus(void* pWindow)
{
    NSWindow* window = (NSWindow*)pWindow;
	[window makeKeyAndOrderFront:window];
}

void NSUtilDisposeWindow(void* pWindow)
{
	// Create local autorelease pool for any objects that need to be autoreleased.
	NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];

	NSWindow* window = (NSWindow*)pWindow;
    [window close];

	// Free up memory
	[pool release];
}

#endif

#endif
