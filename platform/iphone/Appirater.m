/*
 This file is part of Appirater.
 
 Copyright (c) 2012, Arash Payan
 All rights reserved.
 
 Permission is hereby granted, free of charge, to any person
 obtaining a copy of this software and associated documentation
 files (the "Software"), to deal in the Software without
 restriction, including without limitation the rights to use,
 copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the
 Software is furnished to do so, subject to the following
 conditions:
 
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 OTHER DEALINGS IN THE SOFTWARE.
 */
/*
 * Appirater.m
 * appirater
 *
 * Created by Arash Payan on 9/5/09.
 * http://arashpayan.com
 * Copyright 2012 Arash Payan. All rights reserved.
 */

#import "Appirater.h"
#import <SystemConfiguration/SCNetworkReachability.h>
#include <netinet/in.h>

#if ! __has_feature(objc_arc)
#warning This file must be compiled with ARC. Use -fobjc-arc flag (or convert project to ARC).
#endif

NSString *const kAppiraterFirstUseDate				= @"kAppiraterFirstUseDate";
NSString *const kAppiraterUseCount					= @"kAppiraterUseCount";
NSString *const kAppiraterSignificantEventCount		= @"kAppiraterSignificantEventCount";
NSString *const kAppiraterCurrentVersion			= @"kAppiraterCurrentVersion";
NSString *const kAppiraterRatedCurrentVersion		= @"kAppiraterRatedCurrentVersion";
NSString *const kAppiraterDeclinedToRate			= @"kAppiraterDeclinedToRate";
NSString *const kAppiraterReminderRequestDate		= @"kAppiraterReminderRequestDate";

NSString *templateReviewURL = @"itms-apps://ax.itunes.apple.com/WebObjects/MZStore.woa/wa/viewContentsUserReviews?type=Purple+Software&id=APP_ID";
NSString *templateReviewURLiOS7 = @"itms-apps://itunes.apple.com/app/idAPP_ID";
NSString *templateReviewURLiOS8 = @"itms-apps://itunes.apple.com/WebObjects/MZStore.woa/wa/viewContentsUserReviews?id=APP_ID&onlyLatestVersion=true&pageNumber=0&sortOrdering=1&type=Purple+Software";

static NSString *_appId;
static double _daysUntilPrompt = 30;
static NSInteger _usesUntilPrompt = 20;
static NSInteger _significantEventsUntilPrompt = -1;
static double _timeBeforeReminding = 1;
static BOOL _debug = NO;
#if __IPHONE_OS_VERSION_MIN_REQUIRED < __IPHONE_5_0
	static id<AppiraterDelegate> _delegate;
#else
	__weak static id<AppiraterDelegate> _delegate;
#endif
static BOOL _usesAnimation = TRUE;
static UIStatusBarStyle _statusBarStyle;
static BOOL _modalOpen = false;
static BOOL _alwaysUseMainBundle = NO;

@interface Appirater ()
@property (nonatomic, copy) NSString *alertTitle;
@property (nonatomic, copy) NSString *alertMessage;
@property (nonatomic, copy) NSString *alertCancelTitle;
@property (nonatomic, copy) NSString *alertRateTitle;
@property (nonatomic, copy) NSString *alertRateLaterTitle;
- (BOOL)connectedToNetwork;
+ (Appirater*)sharedInstance;
- (void)showPromptWithChecks:(BOOL)withChecks
      displayRateLaterButton:(BOOL)displayRateLaterButton;
- (void)showRatingAlert:(BOOL)displayRateLaterButton;
- (void)showRatingAlert;
- (BOOL)ratingAlertIsAppropriate;
- (BOOL)ratingConditionsHaveBeenMet;
- (void)incrementUseCount;
- (void)hideRatingAlert;
@end

@implementation Appirater 

@synthesize ratingAlert;

+ (void) setAppId:(NSString *)appId {
    _appId = appId;
}

+ (void) setDaysUntilPrompt:(double)value {
    _daysUntilPrompt = value;
}

+ (void) setUsesUntilPrompt:(NSInteger)value {
    _usesUntilPrompt = value;
}

+ (void) setSignificantEventsUntilPrompt:(NSInteger)value {
    _significantEventsUntilPrompt = value;
}

+ (void) setTimeBeforeReminding:(double)value {
    _timeBeforeReminding = value;
}

+ (void) setCustomAlertTitle:(NSString *)title
{
    [self sharedInstance].alertTitle = title;
}

+ (void) setCustomAlertMessage:(NSString *)message
{
    [self sharedInstance].alertMessage = message;
}

+ (void) setCustomAlertCancelButtonTitle:(NSString *)cancelTitle
{
    [self sharedInstance].alertCancelTitle = cancelTitle;
}

+ (void) setCustomAlertRateButtonTitle:(NSString *)rateTitle
{
    [self sharedInstance].alertRateTitle = rateTitle;
}

+ (void) setCustomAlertRateLaterButtonTitle:(NSString *)rateLaterTitle
{
    [self sharedInstance].alertRateLaterTitle = rateLaterTitle;
}

+ (void) setDebug:(BOOL)debug {
    _debug = debug;
}
+ (void)setDelegate:(id<AppiraterDelegate>)delegate{
	_delegate = delegate;
}
+ (void)setUsesAnimation:(BOOL)animation {
	_usesAnimation = animation;
}
+ (void)setOpenInAppStore:(BOOL)openInAppStore {
    [Appirater sharedInstance].openInAppStore = openInAppStore;
}
+ (void)setStatusBarStyle:(UIStatusBarStyle)style {
	_statusBarStyle = style;
}
+ (void)setModalOpen:(BOOL)open {
	_modalOpen = open;
}
+ (void)setAlwaysUseMainBundle:(BOOL)alwaysUseMainBundle {
    _alwaysUseMainBundle = alwaysUseMainBundle;
}

+ (NSBundle *)bundle
{
    NSBundle *bundle;

    if (_alwaysUseMainBundle) {
        bundle = [NSBundle mainBundle];
    } else {
        NSURL *appiraterBundleURL = [[NSBundle mainBundle] URLForResource:@"Appirater" withExtension:@"bundle"];

        if (appiraterBundleURL) {
            // Appirater.bundle will likely only exist when used via CocoaPods
            bundle = [NSBundle bundleWithURL:appiraterBundleURL];
        } else {
            bundle = [NSBundle mainBundle];
        }
    }

    return bundle;
}

- (NSString *)alertTitle
{
    return _alertTitle ? _alertTitle : APPIRATER_MESSAGE_TITLE;
}

- (NSString *)alertMessage
{
    return _alertMessage ? _alertMessage : APPIRATER_MESSAGE;
}

- (NSString *)alertCancelTitle
{
    return _alertCancelTitle ? _alertCancelTitle : APPIRATER_CANCEL_BUTTON;
}

- (NSString *)alertRateTitle
{
    return _alertRateTitle ? _alertRateTitle : APPIRATER_RATE_BUTTON;
}

- (NSString *)alertRateLaterTitle
{
    return _alertRateLaterTitle ? _alertRateLaterTitle : APPIRATER_RATE_LATER;
}

- (void)dealloc {
    [[NSNotificationCenter defaultCenter] removeObserver:self];
}

- (id)init {
    self = [super init];
    if (self) {
        if ([[UIDevice currentDevice].systemVersion floatValue] >= 7.0) {
            self.openInAppStore = YES;
        } else {
            self.openInAppStore = NO;
        }
    }
    
    return self;
}

- (BOOL)connectedToNetwork {
    // Create zero addy
    struct sockaddr_in zeroAddress;
    bzero(&zeroAddress, sizeof(zeroAddress));
    zeroAddress.sin_len = sizeof(zeroAddress);
    zeroAddress.sin_family = AF_INET;
	
    // Recover reachability flags
    SCNetworkReachabilityRef defaultRouteReachability = SCNetworkReachabilityCreateWithAddress(NULL, (struct sockaddr *)&zeroAddress);
    SCNetworkReachabilityFlags flags;
	
    Boolean didRetrieveFlags = SCNetworkReachabilityGetFlags(defaultRouteReachability, &flags);
    CFRelease(defaultRouteReachability);
	
    if (!didRetrieveFlags)
    {
        NSLog(@"Error. Could not recover network reachability flags");
        return NO;
    }
	
    BOOL isReachable = flags & kSCNetworkFlagsReachable;
    BOOL needsConnection = flags & kSCNetworkFlagsConnectionRequired;
	BOOL nonWiFi = flags & kSCNetworkReachabilityFlagsTransientConnection;
	
	NSURL *testURL = [NSURL URLWithString:@"http://www.apple.com/"];
	NSURLRequest *testRequest = [NSURLRequest requestWithURL:testURL  cachePolicy:NSURLRequestReloadIgnoringLocalCacheData timeoutInterval:20.0];
	NSURLConnection *testConnection = [[NSURLConnection alloc] initWithRequest:testRequest delegate:self];
	
    return ((isReachable && !needsConnection) || nonWiFi) ? (testConnection ? YES : NO) : NO;
}

+ (Appirater*)sharedInstance {
	static Appirater *appirater = nil;
	if (appirater == nil)
	{
        static dispatch_once_t onceToken;
        dispatch_once(&onceToken, ^{
            appirater = [[Appirater alloc] init];
			appirater.delegate = _delegate;
            [[NSNotificationCenter defaultCenter] addObserver:self selector:@selector(appWillResignActive) name:
                UIApplicationWillResignActiveNotification object:nil];
        });
	}
	
	return appirater;
}

- (void)showRatingAlert:(BOOL)displayRateLaterButton {
  UIAlertView *alertView = nil;
  id <AppiraterDelegate> delegate = _delegate;
    
  if(delegate && [delegate respondsToSelector:@selector(appiraterShouldDisplayAlert:)] && ![delegate appiraterShouldDisplayAlert:self]) {
      return;
  }
  
  if (displayRateLaterButton) {
  	alertView = [[UIAlertView alloc] initWithTitle:self.alertTitle
                                           message:self.alertMessage
                                          delegate:self
                                 cancelButtonTitle:self.alertCancelTitle
                                 otherButtonTitles:self.alertRateTitle, self.alertRateLaterTitle, nil];
  } else {
  	alertView = [[UIAlertView alloc] initWithTitle:self.alertTitle
                                           message:self.alertMessage
                                          delegate:self
                                 cancelButtonTitle:self.alertCancelTitle
                                 otherButtonTitles:self.alertRateTitle, nil];
  }

	self.ratingAlert = alertView;
    [alertView show];

    if (delegate && [delegate respondsToSelector:@selector(appiraterDidDisplayAlert:)]) {
             [delegate appiraterDidDisplayAlert:self];
    }
}

- (void)showRatingAlert
{
  [self showRatingAlert:true];
}

// is this an ok time to show the alert? (regardless of whether the rating conditions have been met)
//
// things checked here:
// * connectivity with network
// * whether user has rated before
// * whether user has declined to rate
// * whether rating alert is currently showing visibly
// things NOT checked here:
// * time since first launch
// * number of uses of app
// * number of significant events
// * time since last reminder
- (BOOL)ratingAlertIsAppropriate {
    return ([self connectedToNetwork]
            && ![self userHasDeclinedToRate]
            && !self.ratingAlert.visible
            && ![self userHasRatedCurrentVersion]);
}

// have the rating conditions been met/earned? (regardless of whether this would be a moment when it's appropriate to show a new rating alert)
//
// things checked here:
// * time since first launch
// * number of uses of app
// * number of significant events
// * time since last reminder
// things NOT checked here:
// * connectivity with network
// * whether user has rated before
// * whether user has declined to rate
// * whether rating alert is currently showing visibly
- (BOOL)ratingConditionsHaveBeenMet {
	if (_debug)
		return YES;
	
	NSUserDefaults *userDefaults = [NSUserDefaults standardUserDefaults];
	
	NSDate *dateOfFirstLaunch = [NSDate dateWithTimeIntervalSince1970:[userDefaults doubleForKey:kAppiraterFirstUseDate]];
	NSTimeInterval timeSinceFirstLaunch = [[NSDate date] timeIntervalSinceDate:dateOfFirstLaunch];
	NSTimeInterval timeUntilRate = 60 * 60 * 24 * _daysUntilPrompt;
	if (timeSinceFirstLaunch < timeUntilRate)
		return NO;
	
	// check if the app has been used enough
	NSInteger useCount = [userDefaults integerForKey:kAppiraterUseCount];
	if (useCount < _usesUntilPrompt)
		return NO;
	
	// check if the user has done enough significant events
	NSInteger sigEventCount = [userDefaults integerForKey:kAppiraterSignificantEventCount];
	if (sigEventCount < _significantEventsUntilPrompt)
		return NO;
	
	// if the user wanted to be reminded later, has enough time passed?
	NSDate *reminderRequestDate = [NSDate dateWithTimeIntervalSince1970:[userDefaults doubleForKey:kAppiraterReminderRequestDate]];
	NSTimeInterval timeSinceReminderRequest = [[NSDate date] timeIntervalSinceDate:reminderRequestDate];
	NSTimeInterval timeUntilReminder = 60 * 60 * 24 * _timeBeforeReminding;
	if (timeSinceReminderRequest < timeUntilReminder)
		return NO;
	
	return YES;
}

- (void)incrementUseCount {
	// get the app's version
	NSString *version = [[[NSBundle mainBundle] infoDictionary] objectForKey:(NSString*)kCFBundleVersionKey];
	
	// get the version number that we've been tracking
	NSUserDefaults *userDefaults = [NSUserDefaults standardUserDefaults];
	NSString *trackingVersion = [userDefaults stringForKey:kAppiraterCurrentVersion];
	if (trackingVersion == nil)
	{
		trackingVersion = version;
		[userDefaults setObject:version forKey:kAppiraterCurrentVersion];
	}
	
	if (_debug)
		NSLog(@"APPIRATER Tracking version: %@", trackingVersion);
	
	if ([trackingVersion isEqualToString:version])
	{
		// check if the first use date has been set. if not, set it.
		NSTimeInterval timeInterval = [userDefaults doubleForKey:kAppiraterFirstUseDate];
		if (timeInterval == 0)
		{
			timeInterval = [[NSDate date] timeIntervalSince1970];
			[userDefaults setDouble:timeInterval forKey:kAppiraterFirstUseDate];
		}
		
		// increment the use count
		NSInteger useCount = [userDefaults integerForKey:kAppiraterUseCount];
		useCount++;
		[userDefaults setInteger:useCount forKey:kAppiraterUseCount];
		if (_debug)
			NSLog(@"APPIRATER Use count: %@", @(useCount));
	}
	else
	{
		// it's a new version of the app, so restart tracking
		[userDefaults setObject:version forKey:kAppiraterCurrentVersion];
		[userDefaults setDouble:[[NSDate date] timeIntervalSince1970] forKey:kAppiraterFirstUseDate];
		[userDefaults setInteger:1 forKey:kAppiraterUseCount];
		[userDefaults setInteger:0 forKey:kAppiraterSignificantEventCount];
		[userDefaults setBool:NO forKey:kAppiraterRatedCurrentVersion];
		[userDefaults setBool:NO forKey:kAppiraterDeclinedToRate];
		[userDefaults setDouble:0 forKey:kAppiraterReminderRequestDate];
	}
	
	[userDefaults synchronize];
}

- (void)incrementSignificantEventCount {
	// get the app's version
	NSString *version = [[[NSBundle mainBundle] infoDictionary] objectForKey:(NSString*)kCFBundleVersionKey];
	
	// get the version number that we've been tracking
	NSUserDefaults *userDefaults = [NSUserDefaults standardUserDefaults];
	NSString *trackingVersion = [userDefaults stringForKey:kAppiraterCurrentVersion];
	if (trackingVersion == nil)
	{
		trackingVersion = version;
		[userDefaults setObject:version forKey:kAppiraterCurrentVersion];
	}
	
	if (_debug)
		NSLog(@"APPIRATER Tracking version: %@", trackingVersion);
	
	if ([trackingVersion isEqualToString:version])
	{
		// check if the first use date has been set. if not, set it.
		NSTimeInterval timeInterval = [userDefaults doubleForKey:kAppiraterFirstUseDate];
		if (timeInterval == 0)
		{
			timeInterval = [[NSDate date] timeIntervalSince1970];
			[userDefaults setDouble:timeInterval forKey:kAppiraterFirstUseDate];
		}
		
		// increment the significant event count
		NSInteger sigEventCount = [userDefaults integerForKey:kAppiraterSignificantEventCount];
		sigEventCount++;
		[userDefaults setInteger:sigEventCount forKey:kAppiraterSignificantEventCount];
		if (_debug)
			NSLog(@"APPIRATER Significant event count: %@", @(sigEventCount));
	}
	else
	{
		// it's a new version of the app, so restart tracking
		[userDefaults setObject:version forKey:kAppiraterCurrentVersion];
		[userDefaults setDouble:0 forKey:kAppiraterFirstUseDate];
		[userDefaults setInteger:0 forKey:kAppiraterUseCount];
		[userDefaults setInteger:1 forKey:kAppiraterSignificantEventCount];
		[userDefaults setBool:NO forKey:kAppiraterRatedCurrentVersion];
		[userDefaults setBool:NO forKey:kAppiraterDeclinedToRate];
		[userDefaults setDouble:0 forKey:kAppiraterReminderRequestDate];
	}
	
	[userDefaults synchronize];
}

- (void)incrementAndRate:(BOOL)canPromptForRating {
	[self incrementUseCount];
	
	if (canPromptForRating &&
        [self ratingConditionsHaveBeenMet] &&
        [self ratingAlertIsAppropriate])
	{
        dispatch_async(dispatch_get_main_queue(),
                       ^{
                           [self showRatingAlert];
                       });
	}
}

- (void)incrementSignificantEventAndRate:(BOOL)canPromptForRating {
	[self incrementSignificantEventCount];
	
    if (canPromptForRating &&
        [self ratingConditionsHaveBeenMet] &&
        [self ratingAlertIsAppropriate])
	{
        dispatch_async(dispatch_get_main_queue(),
                       ^{
                           [self showRatingAlert];
                       });
	}
}

- (BOOL)userHasDeclinedToRate {
    return [[NSUserDefaults standardUserDefaults] boolForKey:kAppiraterDeclinedToRate];
}

- (BOOL)userHasRatedCurrentVersion {
    return [[NSUserDefaults standardUserDefaults] boolForKey:kAppiraterRatedCurrentVersion];
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-implementations"
+ (void)appLaunched {
	[Appirater appLaunched:YES];
}
#pragma GCC diagnostic pop

+ (void)appLaunched:(BOOL)canPromptForRating {
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_LOW, 0),
                   ^{
                       Appirater *a = [Appirater sharedInstance];
                       if (_debug) {
                           dispatch_async(dispatch_get_main_queue(),
                                          ^{
                                              [a showRatingAlert];
                                          });
                       } else {
                           [a incrementAndRate:canPromptForRating]; 
                       }
                   });
}

- (void)hideRatingAlert {
	if (self.ratingAlert.visible) {
		if (_debug)
			NSLog(@"APPIRATER Hiding Alert");
		[self.ratingAlert dismissWithClickedButtonIndex:-1 animated:NO];
	}	
}

+ (void)appWillResignActive {
	if (_debug)
		NSLog(@"APPIRATER appWillResignActive");
	[[Appirater sharedInstance] hideRatingAlert];
}

+ (void)appEnteredForeground:(BOOL)canPromptForRating {
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_LOW, 0),
                   ^{
                       [[Appirater sharedInstance] incrementAndRate:canPromptForRating];
                   });
}

+ (void)userDidSignificantEvent:(BOOL)canPromptForRating {
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_LOW, 0),
                   ^{
                       [[Appirater sharedInstance] incrementSignificantEventAndRate:canPromptForRating];
                   });
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-implementations"
+ (void)showPrompt {
  [Appirater tryToShowPrompt];
}
#pragma GCC diagnostic pop

+ (void)tryToShowPrompt {
  [[Appirater sharedInstance] showPromptWithChecks:true
                            displayRateLaterButton:true];
}

+ (void)forceShowPrompt:(BOOL)displayRateLaterButton {
  [[Appirater sharedInstance] showPromptWithChecks:false
                            displayRateLaterButton:displayRateLaterButton];
}

- (void)showPromptWithChecks:(BOOL)withChecks
      displayRateLaterButton:(BOOL)displayRateLaterButton {
  if (withChecks == NO || [self ratingAlertIsAppropriate]) {
    [self showRatingAlert:displayRateLaterButton];
  }
}

+ (id)getRootViewController {
    UIWindow *window = [[UIApplication sharedApplication] keyWindow];
    if (window.windowLevel != UIWindowLevelNormal) {
        NSArray *windows = [[UIApplication sharedApplication] windows];
        for(window in windows) {
            if (window.windowLevel == UIWindowLevelNormal) {
                break;
            }
        }
    }
    
    return [Appirater iterateSubViewsForViewController:window]; // iOS 8+ deep traverse
}

+ (id)iterateSubViewsForViewController:(UIView *) parentView {
    for (UIView *subView in [parentView subviews]) {
        UIResponder *responder = [subView nextResponder];
        if([responder isKindOfClass:[UIViewController class]]) {
            return [self topMostViewController: (UIViewController *) responder];
        }
        id found = [Appirater iterateSubViewsForViewController:subView];
        if( nil != found) {
            return found;
        }
    }
    return nil;
}

+ (UIViewController *) topMostViewController: (UIViewController *) controller {
	BOOL isPresenting = NO;
	do {
		// this path is called only on iOS 6+, so -presentedViewController is fine here.
		UIViewController *presented = [controller presentedViewController];
		isPresenting = presented != nil;
		if(presented != nil) {
			controller = presented;
		}
		
	} while (isPresenting);
	
	return controller;
}

+ (void)rateApp {
	
	NSUserDefaults *userDefaults = [NSUserDefaults standardUserDefaults];
	[userDefaults setBool:YES forKey:kAppiraterRatedCurrentVersion];
	[userDefaults synchronize];

	//Use the in-app StoreKit view if available (iOS 6) and imported. This works in the simulator.
	if (![Appirater sharedInstance].openInAppStore && NSStringFromClass([SKStoreProductViewController class]) != nil) {
		
		SKStoreProductViewController *storeViewController = [[SKStoreProductViewController alloc] init];
		NSNumber *appId = [NSNumber numberWithInteger:_appId.integerValue];
		[storeViewController loadProductWithParameters:@{SKStoreProductParameterITunesItemIdentifier:appId} completionBlock:nil];
		storeViewController.delegate = self.sharedInstance;
        
        id <AppiraterDelegate> delegate = self.sharedInstance.delegate;
		if ([delegate respondsToSelector:@selector(appiraterWillPresentModalView:animated:)]) {
			[delegate appiraterWillPresentModalView:self.sharedInstance animated:_usesAnimation];
		}
		[[self getRootViewController] presentViewController:storeViewController animated:_usesAnimation completion:^{
			[self setModalOpen:YES];
			//Temporarily use a black status bar to match the StoreKit view.
			[self setStatusBarStyle:[UIApplication sharedApplication].statusBarStyle];
#if __IPHONE_OS_VERSION_MAX_ALLOWED >= 70000
			[[UIApplication sharedApplication]setStatusBarStyle:UIStatusBarStyleLightContent animated:_usesAnimation];
#endif
		}];
	
	//Use the standard openUrl method if StoreKit is unavailable.
	} else {
		
		#if TARGET_IPHONE_SIMULATOR
		NSLog(@"APPIRATER NOTE: iTunes App Store is not supported on the iOS simulator. Unable to open App Store page.");
		#else
		NSString *reviewURL = [templateReviewURL stringByReplacingOccurrencesOfString:@"APP_ID" withString:[NSString stringWithFormat:@"%@", _appId]];

		// iOS 7 needs a different templateReviewURL @see https://github.com/arashpayan/appirater/issues/131
        // Fixes condition @see https://github.com/arashpayan/appirater/issues/205
		if ([[[UIDevice currentDevice] systemVersion] floatValue] >= 7.0 && [[[UIDevice currentDevice] systemVersion] floatValue] < 8.0) {
			reviewURL = [templateReviewURLiOS7 stringByReplacingOccurrencesOfString:@"APP_ID" withString:[NSString stringWithFormat:@"%@", _appId]];
		}
        // iOS 8 needs a different templateReviewURL also @see https://github.com/arashpayan/appirater/issues/182
        else if ([[[UIDevice currentDevice] systemVersion] floatValue] >= 8.0)
        {
            reviewURL = [templateReviewURLiOS8 stringByReplacingOccurrencesOfString:@"APP_ID" withString:[NSString stringWithFormat:@"%@", _appId]];
        }

		[[UIApplication sharedApplication] openURL:[NSURL URLWithString:reviewURL]];
		#endif
	}
}

- (void)alertView:(UIAlertView *)alertView didDismissWithButtonIndex:(NSInteger)buttonIndex {
	NSUserDefaults *userDefaults = [NSUserDefaults standardUserDefaults];
    
    id <AppiraterDelegate> delegate = _delegate;
	
	switch (buttonIndex) {
		case 0:
		{
			// they don't want to rate it
			[userDefaults setBool:YES forKey:kAppiraterDeclinedToRate];
			[userDefaults synchronize];
			if(delegate && [delegate respondsToSelector:@selector(appiraterDidDeclineToRate:)]){
				[delegate appiraterDidDeclineToRate:self];
			}
			break;
		}
		case 1:
		{
			// they want to rate it
			[Appirater rateApp];
			if(delegate&& [delegate respondsToSelector:@selector(appiraterDidOptToRate:)]){
				[delegate appiraterDidOptToRate:self];
			}
			break;
		}
		case 2:
			// remind them later
			[userDefaults setDouble:[[NSDate date] timeIntervalSince1970] forKey:kAppiraterReminderRequestDate];
			[userDefaults synchronize];
			if(delegate && [delegate respondsToSelector:@selector(appiraterDidOptToRemindLater:)]){
				[delegate appiraterDidOptToRemindLater:self];
			}
			break;
		default:
			break;
	}
}

//Delegate call from the StoreKit view.
- (void)productViewControllerDidFinish:(SKStoreProductViewController *)viewController {
	[Appirater closeModal];
}

//Close the in-app rating (StoreKit) view and restore the previous status bar style.
+ (void)closeModal {
	if (_modalOpen) {
		[[UIApplication sharedApplication]setStatusBarStyle:_statusBarStyle animated:_usesAnimation];
		BOOL usedAnimation = _usesAnimation;
		[self setModalOpen:NO];
		
		// get the top most controller (= the StoreKit Controller) and dismiss it
		UIViewController *presentingController = [UIApplication sharedApplication].keyWindow.rootViewController;
		presentingController = [self topMostViewController: presentingController];
		[presentingController dismissViewControllerAnimated:_usesAnimation completion:^{
            id <AppiraterDelegate> delegate = self.sharedInstance.delegate;
			if ([delegate respondsToSelector:@selector(appiraterDidDismissModalView:animated:)]) {
				[delegate appiraterDidDismissModalView:(Appirater *)self animated:usedAnimation];
			}
		}];
		[self.class setStatusBarStyle:(UIStatusBarStyle)nil];
	}
}

@end
