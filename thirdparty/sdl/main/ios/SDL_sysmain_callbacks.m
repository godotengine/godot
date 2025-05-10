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
#include "../SDL_main_callbacks.h"

#ifdef SDL_PLATFORM_IOS

#import <UIKit/UIKit.h>

#include "../../video/uikit/SDL_uikitevents.h"  // For SDL_UpdateLifecycleObserver()


@interface SDLIosMainCallbacksDisplayLink : NSObject
@property(nonatomic, retain) CADisplayLink *displayLink;
- (void)appIteration:(CADisplayLink *)sender;
- (instancetype)init:(SDL_AppIterate_func)_appiter quitfunc:(SDL_AppQuit_func)_appquit;
@end

static SDLIosMainCallbacksDisplayLink *globalDisplayLink;

@implementation SDLIosMainCallbacksDisplayLink

- (instancetype)init:(SDL_AppIterate_func)_appiter quitfunc:(SDL_AppQuit_func)_appquit;
{
    if ((self = [super init])) {
        self.displayLink = [CADisplayLink displayLinkWithTarget:self selector:@selector(appIteration:)];
        [self.displayLink addToRunLoop:[NSRunLoop currentRunLoop] forMode:NSDefaultRunLoopMode];
    }
    return self;
}

- (void)appIteration:(CADisplayLink *)sender
{
    const SDL_AppResult rc = SDL_IterateMainCallbacks(true);
    if (rc != SDL_APP_CONTINUE) {
        [self.displayLink invalidate];
        self.displayLink = nil;
        globalDisplayLink = nil;
        SDL_QuitMainCallbacks(rc);
        SDL_UpdateLifecycleObserver();
        exit((rc == SDL_APP_FAILURE) ? 1 : 0);
    }
}
@end

// SDL_RunApp will land in UIApplicationMain, which calls SDL_main from postFinishLaunch, which calls this.
// When we return from here, we're living in the RunLoop, and a CADisplayLink is firing regularly for us.
int SDL_EnterAppMainCallbacks(int argc, char* argv[], SDL_AppInit_func appinit, SDL_AppIterate_func appiter, SDL_AppEvent_func appevent, SDL_AppQuit_func appquit)
{
    SDL_AppResult rc = SDL_InitMainCallbacks(argc, argv, appinit, appiter, appevent, appquit);
    if (rc == SDL_APP_CONTINUE) {
        globalDisplayLink = [[SDLIosMainCallbacksDisplayLink alloc] init:appiter quitfunc:appquit];
        if (globalDisplayLink == nil) {
            rc = SDL_APP_FAILURE;
        } else {
            return 0;  // this will fall all the way out of SDL_main, where UIApplicationMain will keep running the RunLoop.
        }
    }

    // appinit requested quit, just bounce out now.
    SDL_QuitMainCallbacks(rc);
    exit((rc == SDL_APP_FAILURE) ? 1 : 0);

    return 1;  // just in case.
}

#endif


