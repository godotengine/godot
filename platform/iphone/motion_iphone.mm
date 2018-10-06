#import "motion_iphone.h"
#import "os_iphone.h"
#import <UIKit/UIKit.h>

@interface MotionTracker ()

@property(nonatomic, strong) CMMotionManager *manager;

@end

@implementation MotionTracker

+ (instancetype)sharedTracker {
	static dispatch_once_t onceToken;
	static MotionTracker *gMotionTracker = nil;
	dispatch_once(&onceToken, ^{
		gMotionTracker = [[MotionTracker alloc] init];
	});
	return gMotionTracker;
}

- (id)init {
	if (self = [super init]) {
		self.manager = [[CMMotionManager alloc] init];
		self.manager.deviceMotionUpdateInterval = 1.f / 70;
		[self.manager startDeviceMotionUpdatesUsingReferenceFrame:
							  CMAttitudeReferenceFrameXMagneticNorthZVertical];
	}
	return self;
}

- (void)updateMotion {
	// Just using polling approach for now, we can set this up so it sends
	// data to us in intervals, might be better. See Apple reference pages
	// for more details:
	// https://developer.apple.com/reference/coremotion/cmmotionmanager?language=objc

	// Apple splits our accelerometer date into a gravity and user movement
	// component. We add them back together
	CMAcceleration gravity = self.manager.deviceMotion.gravity;
	CMAcceleration acceleration = self.manager.deviceMotion.userAcceleration;

	///@TODO We don't seem to be getting data here, is my device broken or
	/// is this code incorrect?
	CMMagneticField magnetic = self.manager.deviceMotion.magneticField.field;

	///@TODO we can access rotationRate as a CMRotationRate variable
	///(processed date) or CMGyroData (raw data), have to see what works
	/// best
	CMRotationRate rotation = self.manager.deviceMotion.rotationRate;

	// Adjust for screen orientation.
	// [[UIDevice currentDevice] orientation] changes even if we've fixed
	// our orientation which is not a good thing when you're trying to get
	// your user to move the screen in all directions and want consistent
	// output

	///@TODO Using [[UIApplication sharedApplication] statusBarOrientation]
	/// is a bit of a hack.
	switch ([[UIApplication sharedApplication] statusBarOrientation]) {
		case UIDeviceOrientationLandscapeLeft: {
			OSIPhone::get_singleton()->update_gravity(-gravity.y, gravity.x,
					gravity.z);
			OSIPhone::get_singleton()->update_accelerometer(
					-(acceleration.y + gravity.y), (acceleration.x + gravity.x),
					acceleration.z + gravity.z);
			OSIPhone::get_singleton()->update_magnetometer(
					-magnetic.y, magnetic.x, magnetic.z);
			OSIPhone::get_singleton()->update_gyroscope(-rotation.y, rotation.x,
					rotation.z);
			break;
		}
		case UIDeviceOrientationLandscapeRight: {
			OSIPhone::get_singleton()->update_gravity(gravity.y, -gravity.x,
					gravity.z);
			OSIPhone::get_singleton()->update_accelerometer(
					(acceleration.y + gravity.y), -(acceleration.x + gravity.x),
					acceleration.z + gravity.z);
			OSIPhone::get_singleton()->update_magnetometer(
					magnetic.y, -magnetic.x, magnetic.z);
			OSIPhone::get_singleton()->update_gyroscope(rotation.y, -rotation.x,
					rotation.z);
			break;
		}
		case UIDeviceOrientationPortraitUpsideDown: {
			OSIPhone::get_singleton()->update_gravity(-gravity.x, gravity.y,
					gravity.z);
			OSIPhone::get_singleton()->update_accelerometer(
					-(acceleration.x + gravity.x), (acceleration.y + gravity.y),
					acceleration.z + gravity.z);
			OSIPhone::get_singleton()->update_magnetometer(
					-magnetic.x, magnetic.y, magnetic.z);
			OSIPhone::get_singleton()->update_gyroscope(-rotation.x, rotation.y,
					rotation.z);
			break;
		}
		default: { // assume portrait
			OSIPhone::get_singleton()->update_gravity(gravity.x, gravity.y,
					gravity.z);
			OSIPhone::get_singleton()->update_accelerometer(
					acceleration.x + gravity.x, acceleration.y + gravity.y,
					acceleration.z + gravity.z);
			OSIPhone::get_singleton()->update_magnetometer(magnetic.x, magnetic.y,
					magnetic.z);
			OSIPhone::get_singleton()->update_gyroscope(rotation.x, rotation.y,
					rotation.z);
			break;
		}
	}
}

@end