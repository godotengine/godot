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

#ifdef SDL_VIDEO_DRIVER_UIKIT

#include "../SDL_sysvideo.h"

#import "SDL_uikitappdelegate.h"
#import "SDL_uikitmodes.h"
#import "SDL_uikitwindow.h"

#include "../../events/SDL_events_c.h"

#ifdef main
#undef main
#endif

static SDL_main_func forward_main;
static int forward_argc;
static char **forward_argv;
static int exit_status;

int SDL_RunApp(int argc, char* argv[], SDL_main_func mainFunction, void * reserved)
{
    int i;

    // store arguments
    /* Note that we need to be careful about how we allocate/free memory here.
     * If the application calls SDL_SetMemoryFunctions(), we can't rely on
     * SDL_free() to use the same allocator after SDL_main() returns.
     */
    forward_main = mainFunction;
    forward_argc = argc;
    forward_argv = (char **)malloc((argc + 1) * sizeof(char *)); // This should NOT be SDL_malloc()
    for (i = 0; i < argc; i++) {
        forward_argv[i] = malloc((strlen(argv[i]) + 1) * sizeof(char)); // This should NOT be SDL_malloc()
        strcpy(forward_argv[i], argv[i]);
    }
    forward_argv[i] = NULL;

    // Give over control to run loop, SDLUIKitDelegate will handle most things from here
    @autoreleasepool {
        UIApplicationMain(argc, argv, nil, [SDLUIKitDelegate getAppDelegateClassName]);
    }

    // free the memory we used to hold copies of argc and argv
    for (i = 0; i < forward_argc; i++) {
        free(forward_argv[i]); // This should NOT be SDL_free()
    }
    free(forward_argv); // This should NOT be SDL_free()

    return exit_status;
}

#if !defined(SDL_PLATFORM_TVOS) && !defined(SDL_PLATFORM_VISIONOS)
// Load a launch image using the old UILaunchImageFile-era naming rules.
static UIImage *SDL_LoadLaunchImageNamed(NSString *name, int screenh)
{
    UIInterfaceOrientation curorient = [UIApplication sharedApplication].statusBarOrientation;
    UIUserInterfaceIdiom idiom = [UIDevice currentDevice].userInterfaceIdiom;
    UIImage *image = nil;

    if (idiom == UIUserInterfaceIdiomPhone && screenh == 568) {
        // The image name for the iPhone 5 uses its height as a suffix.
        image = [UIImage imageNamed:[NSString stringWithFormat:@"%@-568h", name]];
    } else if (idiom == UIUserInterfaceIdiomPad) {
        // iPad apps can launch in any orientation.
        if (UIInterfaceOrientationIsLandscape(curorient)) {
            if (curorient == UIInterfaceOrientationLandscapeLeft) {
                image = [UIImage imageNamed:[NSString stringWithFormat:@"%@-LandscapeLeft", name]];
            } else {
                image = [UIImage imageNamed:[NSString stringWithFormat:@"%@-LandscapeRight", name]];
            }
            if (!image) {
                image = [UIImage imageNamed:[NSString stringWithFormat:@"%@-Landscape", name]];
            }
        } else {
            if (curorient == UIInterfaceOrientationPortraitUpsideDown) {
                image = [UIImage imageNamed:[NSString stringWithFormat:@"%@-PortraitUpsideDown", name]];
            }
            if (!image) {
                image = [UIImage imageNamed:[NSString stringWithFormat:@"%@-Portrait", name]];
            }
        }
    }

    if (!image) {
        image = [UIImage imageNamed:name];
    }

    return image;
}

@interface SDLLaunchStoryboardViewController : UIViewController
@property(nonatomic, strong) UIViewController *storyboardViewController;
- (instancetype)initWithStoryboardViewController:(UIViewController *)storyboardViewController;
@end

@implementation SDLLaunchStoryboardViewController

- (instancetype)initWithStoryboardViewController:(UIViewController *)storyboardViewController
{
    self = [super init];
    self.storyboardViewController = storyboardViewController;
    return self;
}

- (void)viewDidLoad
{
    [super viewDidLoad];

    [self addChildViewController:self.storyboardViewController];
    [self.view addSubview:self.storyboardViewController.view];
    self.storyboardViewController.view.autoresizingMask = UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight;
    self.storyboardViewController.view.frame = self.view.bounds;
    [self.storyboardViewController didMoveToParentViewController:self];

#ifndef SDL_PLATFORM_VISIONOS
    UIApplication.sharedApplication.statusBarHidden = self.prefersStatusBarHidden;
    UIApplication.sharedApplication.statusBarStyle = self.preferredStatusBarStyle;
#endif
}

- (BOOL)prefersStatusBarHidden
{
    return [[NSBundle.mainBundle objectForInfoDictionaryKey:@"UIStatusBarHidden"] boolValue];
}

- (UIStatusBarStyle)preferredStatusBarStyle
{
    NSString *statusBarStyle = [NSBundle.mainBundle objectForInfoDictionaryKey:@"UIStatusBarStyle"];
    if ([statusBarStyle isEqualToString:@"UIStatusBarStyleLightContent"]) {
        return UIStatusBarStyleLightContent;
    }
    if (@available(iOS 13.0, *)) {
        if ([statusBarStyle isEqualToString:@"UIStatusBarStyleDarkContent"]) {
            return UIStatusBarStyleDarkContent;
        }
    }
    return UIStatusBarStyleDefault;
}

@end
#endif // !SDL_PLATFORM_TVOS

@interface SDLLaunchScreenController ()

#ifndef SDL_PLATFORM_TVOS
- (NSUInteger)supportedInterfaceOrientations;
#endif

@end

@implementation SDLLaunchScreenController

- (instancetype)init
{
    return [self initWithNibName:nil bundle:[NSBundle mainBundle]];
}

- (instancetype)initWithNibName:(NSString *)nibNameOrNil bundle:(NSBundle *)nibBundleOrNil
{
    if (!(self = [super initWithNibName:nil bundle:nil])) {
        return nil;
    }

    NSString *screenname = nibNameOrNil;
    NSBundle *bundle = nibBundleOrNil;

    // A launch screen may not exist. Fall back to launch images in that case.
    if (screenname) {
        @try {
            self.view = [bundle loadNibNamed:screenname owner:self options:nil][0];
        }
        @catch (NSException *exception) {
            /* If a launch screen name is specified but it fails to load, iOS
             * displays a blank screen rather than falling back to an image. */
            return nil;
        }
    }

    if (!self.view) {
        NSArray *launchimages = [bundle objectForInfoDictionaryKey:@"UILaunchImages"];
        NSString *imagename = nil;
        UIImage *image = nil;

#ifdef SDL_PLATFORM_VISIONOS
        int screenw = SDL_XR_SCREENWIDTH;
        int screenh = SDL_XR_SCREENHEIGHT;
#else
        int screenw = (int)([UIScreen mainScreen].bounds.size.width + 0.5);
        int screenh = (int)([UIScreen mainScreen].bounds.size.height + 0.5);
#endif



#if !defined(SDL_PLATFORM_TVOS) && !defined(SDL_PLATFORM_VISIONOS)
        UIInterfaceOrientation curorient = [UIApplication sharedApplication].statusBarOrientation;

        // We always want portrait-oriented size, to match UILaunchImageSize.
        if (screenw > screenh) {
            int width = screenw;
            screenw = screenh;
            screenh = width;
        }
#endif

        // Xcode 5 introduced a dictionary of launch images in Info.plist.
        if (launchimages) {
            for (NSDictionary *dict in launchimages) {
                NSString *minversion = dict[@"UILaunchImageMinimumOSVersion"];
                NSString *sizestring = dict[@"UILaunchImageSize"];

                // Ignore this image if the current version is too low.
                if (minversion && !UIKit_IsSystemVersionAtLeast(minversion.doubleValue)) {
                    continue;
                }

                // Ignore this image if the size doesn't match.
                if (sizestring) {
                    CGSize size = CGSizeFromString(sizestring);
                    if ((int)(size.width + 0.5) != screenw || (int)(size.height + 0.5) != screenh) {
                        continue;
                    }
                }

#if !defined(SDL_PLATFORM_TVOS) && !defined(SDL_PLATFORM_VISIONOS)
                UIInterfaceOrientationMask orientmask = UIInterfaceOrientationMaskPortrait | UIInterfaceOrientationMaskPortraitUpsideDown;
                NSString *orientstring = dict[@"UILaunchImageOrientation"];

                if (orientstring) {
                    if ([orientstring isEqualToString:@"PortraitUpsideDown"]) {
                        orientmask = UIInterfaceOrientationMaskPortraitUpsideDown;
                    } else if ([orientstring isEqualToString:@"Landscape"]) {
                        orientmask = UIInterfaceOrientationMaskLandscape;
                    } else if ([orientstring isEqualToString:@"LandscapeLeft"]) {
                        orientmask = UIInterfaceOrientationMaskLandscapeLeft;
                    } else if ([orientstring isEqualToString:@"LandscapeRight"]) {
                        orientmask = UIInterfaceOrientationMaskLandscapeRight;
                    }
                }

                // Ignore this image if the orientation doesn't match.
                if ((orientmask & (1 << curorient)) == 0) {
                    continue;
                }
#endif

                imagename = dict[@"UILaunchImageName"];
            }

            if (imagename) {
                image = [UIImage imageNamed:imagename];
            }
        }
#if !defined(SDL_PLATFORM_TVOS) && !defined(SDL_PLATFORM_VISIONOS)
        else {
            imagename = [bundle objectForInfoDictionaryKey:@"UILaunchImageFile"];

            if (imagename) {
                image = SDL_LoadLaunchImageNamed(imagename, screenh);
            }

            if (!image) {
                image = SDL_LoadLaunchImageNamed(@"Default", screenh);
            }
        }
#endif

        if (image) {
#ifdef SDL_PLATFORM_VISIONOS
            CGRect viewFrame = CGRectMake(0, 0, screenw, screenh);
#else
            CGRect viewFrame = [UIScreen mainScreen].bounds;
#endif
            UIImageView *view = [[UIImageView alloc] initWithFrame:viewFrame];
            UIImageOrientation imageorient = UIImageOrientationUp;

#if !defined(SDL_PLATFORM_TVOS) && !defined(SDL_PLATFORM_VISIONOS)
            // Bugs observed / workaround tested in iOS 8.3.
            if (UIInterfaceOrientationIsLandscape(curorient)) {
                if (image.size.width < image.size.height) {
                    /* On iOS 8, portrait launch images displayed in forced-
                     * landscape mode (e.g. a standard Default.png on an iPhone
                     * when Info.plist only supports landscape orientations) need
                     * to be rotated to display in the expected orientation. */
                    if (curorient == UIInterfaceOrientationLandscapeLeft) {
                        imageorient = UIImageOrientationRight;
                    } else if (curorient == UIInterfaceOrientationLandscapeRight) {
                        imageorient = UIImageOrientationLeft;
                    }
                }
            }
#endif

            // Create the properly oriented image.
            view.image = [[UIImage alloc] initWithCGImage:image.CGImage scale:image.scale orientation:imageorient];

            self.view = view;
        }
    }

    return self;
}

- (void)loadView
{
    // Do nothing.
}

#ifndef SDL_PLATFORM_TVOS
- (BOOL)shouldAutorotate
{
    // If YES, the launch image will be incorrectly rotated in some cases.
    return NO;
}

- (NSUInteger)supportedInterfaceOrientations
{
    /* We keep the supported orientations unrestricted to avoid the case where
     * there are no common orientations between the ones set in Info.plist and
     * the ones set here (it will cause an exception in that case.) */
    return UIInterfaceOrientationMaskAll;
}
#endif // !SDL_PLATFORM_TVOS

@end

@implementation SDLUIKitDelegate
{
    UIWindow *launchWindow;
}

// convenience method
+ (id)sharedAppDelegate
{
    /* the delegate is set in UIApplicationMain(), which is guaranteed to be
     * called before this method */
    return [UIApplication sharedApplication].delegate;
}

+ (NSString *)getAppDelegateClassName
{
    /* subclassing notice: when you subclass this appdelegate, make sure to add
     * a category to override this method and return the actual name of the
     * delegate */
    return @"SDLUIKitDelegate";
}

- (void)hideLaunchScreen
{
    UIWindow *window = launchWindow;

    if (!window || window.hidden) {
        return;
    }

    launchWindow = nil;

    // Do a nice animated fade-out (roughly matches the real launch behavior.)
    [UIView animateWithDuration:0.2
        animations:^{
          window.alpha = 0.0;
        }
        completion:^(BOOL finished) {
          window.hidden = YES;
          UIKit_ForceUpdateHomeIndicator(); // Wait for launch screen to hide so settings are applied to the actual view controller.
        }];
}

- (void)postFinishLaunch
{
    /* Hide the launch screen the next time the run loop is run. SDL apps will
     * have a chance to load resources while the launch screen is still up. */
    [self performSelector:@selector(hideLaunchScreen) withObject:nil afterDelay:0.0];

    // run the user's application, passing argc and argv
    SDL_SetiOSEventPump(true);
    exit_status = forward_main(forward_argc, forward_argv);
    SDL_SetiOSEventPump(false);

    if (launchWindow) {
        launchWindow.hidden = YES;
        launchWindow = nil;
    }

    // exit, passing the return status from the user's application
    /* We don't actually exit to support applications that do setup in their
     * main function and then allow the Cocoa event loop to run. */
    // exit(exit_status);
}

- (BOOL)application:(UIApplication *)application didFinishLaunchingWithOptions:(NSDictionary *)launchOptions
{
    NSBundle *bundle = [NSBundle mainBundle];

#ifdef SDL_IPHONE_LAUNCHSCREEN
    /* The normal launch screen is displayed until didFinishLaunching returns,
     * but SDL_main is called after that happens and there may be a noticeable
     * delay between the start of SDL_main and when the first real frame is
     * displayed (e.g. if resources are loaded before SDL_GL_SwapWindow is
     * called), so we show the launch screen programmatically until the first
     * time events are pumped. */
    UIViewController *vc = nil;
    NSString *screenname = nil;

    // tvOS only uses a plain launch image.
#if !defined(SDL_PLATFORM_TVOS) && !defined(SDL_PLATFORM_VISIONOS)
    screenname = [bundle objectForInfoDictionaryKey:@"UILaunchStoryboardName"];

    if (screenname) {
        @try {
            /* The launch storyboard is actually a nib in some older versions of
             * Xcode. We'll try to load it as a storyboard first, as it's more
             * modern. */
            UIStoryboard *storyboard = [UIStoryboard storyboardWithName:screenname bundle:bundle];
            __auto_type storyboardVc = [storyboard instantiateInitialViewController];
            vc = [[SDLLaunchStoryboardViewController alloc] initWithStoryboardViewController:storyboardVc];
        }
        @catch (NSException *exception) {
            // Do nothing (there's more code to execute below).
        }
    }
#endif

    if (vc == nil) {
        vc = [[SDLLaunchScreenController alloc] initWithNibName:screenname bundle:bundle];
    }

    if (vc.view) {
#ifdef SDL_PLATFORM_VISIONOS
        CGRect viewFrame = CGRectMake(0, 0, SDL_XR_SCREENWIDTH, SDL_XR_SCREENHEIGHT);
#else
        CGRect viewFrame = [UIScreen mainScreen].bounds;
#endif
        launchWindow = [[UIWindow alloc] initWithFrame:viewFrame];

        /* We don't want the launch window immediately hidden when a real SDL
         * window is shown - we fade it out ourselves when we're ready. */
        launchWindow.windowLevel = UIWindowLevelNormal + 1.0;

        /* Show the window but don't make it key. Events should always go to
         * other windows when possible. */
        launchWindow.hidden = NO;

        launchWindow.rootViewController = vc;
    }
#endif

    // Set working directory to resource path
    [[NSFileManager defaultManager] changeCurrentDirectoryPath:[bundle resourcePath]];

    SDL_SetMainReady();
    [self performSelector:@selector(postFinishLaunch) withObject:nil afterDelay:0.0];

    return YES;
}

- (UIWindow *)window
{
    SDL_VideoDevice *_this = SDL_GetVideoDevice();
    if (_this) {
        SDL_Window *window = NULL;
        for (window = _this->windows; window != NULL; window = window->next) {
            SDL_UIKitWindowData *data = (__bridge SDL_UIKitWindowData *)window->internal;
            if (data != nil) {
                return data.uiwindow;
            }
        }
    }
    return nil;
}

- (void)setWindow:(UIWindow *)window
{
    // Do nothing.
}

- (void)sendDropFileForURL:(NSURL *)url fromSourceApplication:(NSString *)sourceApplication
{
    NSURL *fileURL = url.filePathURL;
    const char *sourceApplicationCString = sourceApplication ? [sourceApplication UTF8String] : NULL;
    if (fileURL != nil) {
        SDL_SendDropFile(NULL, sourceApplicationCString, fileURL.path.UTF8String);
    } else {
        SDL_SendDropFile(NULL, sourceApplicationCString, url.absoluteString.UTF8String);
    }
    SDL_SendDropComplete(NULL);
}

- (BOOL)application:(UIApplication *)app openURL:(NSURL *)url options:(NSDictionary<UIApplicationOpenURLOptionsKey, id> *)options
{
    // TODO: Handle options
    [self sendDropFileForURL:url fromSourceApplication:NULL];
    return YES;
}

@end

#endif // SDL_VIDEO_DRIVER_UIKIT
