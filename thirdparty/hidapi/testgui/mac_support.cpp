/*******************************
 Mac support for HID Test GUI
 
 Alan Ott
 Signal 11 Software

 Some of this code is from Apple Documentation, most notably
 http://developer.apple.com/legacy/mac/library/documentation/AppleScript/Conceptual/AppleEvents/AppleEvents.pdf 
*******************************/

#include <Carbon/Carbon.h>
#include <fx.h>


extern FXMainWindow *g_main_window;

static pascal OSErr HandleQuitMessage(const AppleEvent *theAppleEvent, AppleEvent 
									  *reply, long handlerRefcon) 
{
	puts("Quitting\n");
	FXApp::instance()->exit();
	return 0;
}

static pascal OSErr HandleReopenMessage(const AppleEvent *theAppleEvent, AppleEvent 
									  *reply, long handlerRefcon) 
{
	puts("Showing");
	g_main_window->show();
	return 0;
}

static pascal OSErr HandleWildCardMessage(const AppleEvent *theAppleEvent, AppleEvent 
									  *reply, long handlerRefcon) 
{
	puts("WildCard\n");
	return 0;
}

OSStatus AEHandler(EventHandlerCallRef inCaller, EventRef inEvent, void* inRefcon) 
{ 
    Boolean     release = false; 
    EventRecord eventRecord; 
    OSErr       ignoreErrForThisSample; 
	
    // Events of type kEventAppleEvent must be removed from the queue 
    //  before being passed to AEProcessAppleEvent. 
    if (IsEventInQueue(GetMainEventQueue(), inEvent)) 
    { 
        // RemoveEventFromQueue will release the event, which will 
        //  destroy it if we don't retain it first. 
        RetainEvent(inEvent); 
        release = true; 
        RemoveEventFromQueue(GetMainEventQueue(), inEvent); 
    } 
    // Convert the event ref to the type AEProcessAppleEvent expects. 
    ConvertEventRefToEventRecord(inEvent, &eventRecord); 
    ignoreErrForThisSample = AEProcessAppleEvent(&eventRecord); 
    if (release) 
        ReleaseEvent(inEvent); 
    // This Carbon event has been handled, even if no AppleEvent handlers 
    //  were installed for the Apple event. 
    return noErr; 
}

static void HandleEvent(EventRecord *event) 
{ 
	//printf("What: %d message %x\n", event->what, event->message);
	if (event->what == osEvt) {
		if (((event->message >> 24) & 0xff) == suspendResumeMessage) {
			if (event->message & resumeFlag) {
				g_main_window->show();				
			}
		}
	}

#if 0
    switch (event->what) 
    { 
        case mouseDown: 
            //HandleMouseDown(event); 
            break; 
        case keyDown: 
        case autoKey: 
            //HandleKeyPress(event); 
            break; 
        case kHighLevelEvent: 
			puts("Calling ProcessAppleEvent\n");
            AEProcessAppleEvent(event); 
            break; 
    } 
#endif
} 

void
init_apple_message_system()
{
	OSErr err;
	static const EventTypeSpec appleEvents[] = 
	{
		{ kEventClassAppleEvent, kEventAppleEvent }
	};
	
	/* Install the handler for Apple Events */
	InstallApplicationEventHandler(NewEventHandlerUPP(AEHandler), 
	              GetEventTypeCount(appleEvents), appleEvents, 0, NULL); 

	/* Install handlers for the individual Apple Events that come
	   from the Dock icon: the Reopen (click), and the Quit messages. */
	err = AEInstallEventHandler(kCoreEventClass, kAEQuitApplication, 
	              NewAEEventHandlerUPP(HandleQuitMessage), 0, false);
	err = AEInstallEventHandler(kCoreEventClass, kAEReopenApplication, 
	              NewAEEventHandlerUPP(HandleReopenMessage), 0, false);
#if 0
	// Left as an example of a wild card match.
	err = AEInstallEventHandler(kCoreEventClass, typeWildCard, 
	              NewAEEventHandlerUPP(HandleWildMessage), 0, false);
#endif
}

void
check_apple_events()
{
	RgnHandle       cursorRgn = NULL; 
	Boolean         gotEvent=TRUE; 
	EventRecord     event; 

	while (gotEvent) { 
		gotEvent = WaitNextEvent(everyEvent, &event, 0L/*timeout*/, cursorRgn); 
		if (gotEvent) { 
			HandleEvent(&event); 
		} 
	}
}
