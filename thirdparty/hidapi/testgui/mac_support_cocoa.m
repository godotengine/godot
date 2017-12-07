/*******************************
 Mac support for HID Test GUI
 
 Alan Ott
 Signal 11 Software
*******************************/

#include <fx.h>
#import <Cocoa/Cocoa.h>

extern FXMainWindow *g_main_window;


@interface MyAppDelegate : NSObject
{
} 
@end

@implementation MyAppDelegate
- (void) applicationWillBecomeActive:(NSNotification*)notif
{
	printf("WillBecomeActive\n");
	g_main_window->show();

}

- (void) applicationWillTerminate:(NSNotification*)notif
{
	/* Doesn't get called. Not sure why */
	printf("WillTerminate\n");
	FXApp::instance()->exit();
}

- (NSApplicationTerminateReply) applicationShouldTerminate:(NSApplication*)sender
{
	/* Doesn't get called. Not sure why */
	printf("ShouldTerminate\n");
	return YES;
}

- (void) applicationWillHide:(NSNotification*)notif
{
	printf("WillHide\n");
	g_main_window->hide();
}

- (void) handleQuitEvent:(NSAppleEventDescriptor*)event withReplyEvent:(NSAppleEventDescriptor*)replyEvent
{
	printf("QuitEvent\n");
	FXApp::instance()->exit();
}

@end

extern "C" {

void
init_apple_message_system()
{
	static MyAppDelegate *d = [MyAppDelegate new];

	[[NSApplication sharedApplication] setDelegate:d];

	/* Register for Apple Events. */
	/* This is from
	   http://stackoverflow.com/questions/1768497/application-exit-event */
	NSAppleEventManager *aem = [NSAppleEventManager sharedAppleEventManager];
	[aem setEventHandler:d
	     andSelector:@selector(handleQuitEvent:withReplyEvent:)
	     forEventClass:kCoreEventClass andEventID:kAEQuitApplication];
}

void
check_apple_events()
{
	NSApplication *app = [NSApplication sharedApplication];

	NSAutoreleasePool *pool = [NSAutoreleasePool new];
	while (1) {
		NSEvent* event = [NSApp nextEventMatchingMask:NSAnyEventMask
		                        untilDate:nil
                                        inMode:NSDefaultRunLoopMode
                                        dequeue:YES];
		if (event == NULL)
			break;
		else {
			//printf("Event happened: Type: %d\n", event->_type);
			[app sendEvent: event];
		}
	}
	[pool release];
}

} /* extern "C" */
