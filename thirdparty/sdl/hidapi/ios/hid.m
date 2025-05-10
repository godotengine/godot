/*
  Simple DirectMedia Layer
  Copyright (C) 2021 Valve Corporation

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

#if defined(SDL_PLATFORM_IOS) || defined(SDL_PLATFORM_TVOS)

#ifndef SDL_HIDAPI_DISABLED

#include "../SDL_hidapi_c.h"

#define hid_close                    PLATFORM_hid_close
#define hid_device                   PLATFORM_hid_device
#define hid_device_                  PLATFORM_hid_device_
#define hid_enumerate                PLATFORM_hid_enumerate
#define hid_error                    PLATFORM_hid_error
#define hid_exit                     PLATFORM_hid_exit
#define hid_free_enumeration         PLATFORM_hid_free_enumeration
#define hid_get_device_info          PLATFORM_hid_get_device_info
#define hid_get_feature_report       PLATFORM_hid_get_feature_report
#define hid_get_indexed_string       PLATFORM_hid_get_indexed_string
#define hid_get_input_report         PLATFORM_hid_get_input_report
#define hid_get_manufacturer_string  PLATFORM_hid_get_manufacturer_string
#define hid_get_product_string       PLATFORM_hid_get_product_string
#define hid_get_report_descriptor    PLATFORM_hid_get_report_descriptor
#define hid_get_serial_number_string PLATFORM_hid_get_serial_number_string
#define hid_init                     PLATFORM_hid_init
#define hid_open_path                PLATFORM_hid_open_path
#define hid_open                     PLATFORM_hid_open
#define hid_read                     PLATFORM_hid_read
#define hid_read_timeout             PLATFORM_hid_read_timeout
#define hid_send_feature_report      PLATFORM_hid_send_feature_report
#define hid_set_nonblocking          PLATFORM_hid_set_nonblocking
#define hid_version                  PLATFORM_hid_version
#define hid_version_str              PLATFORM_hid_version_str
#define hid_write                    PLATFORM_hid_write

#include <CoreBluetooth/CoreBluetooth.h>
#include <QuartzCore/QuartzCore.h>
#import <UIKit/UIKit.h>
#import <mach/mach_time.h>
#include <pthread.h>
#include <sys/time.h>
#include <unistd.h>
#include "../hidapi/hidapi.h"

#define VALVE_USB_VID       0x28DE
#define D0G_BLE2_PID        0x1106

typedef uint32_t uint32;
typedef uint64_t uint64;

// enables detailed NSLog logging of feature reports
#define FEATURE_REPORT_LOGGING	0

#define REPORT_SEGMENT_DATA_FLAG	0x80
#define REPORT_SEGMENT_LAST_FLAG	0x40

#define VALVE_SERVICE		@"100F6C32-1735-4313-B402-38567131E5F3"

// (READ/NOTIFICATIONS)
#define VALVE_INPUT_CHAR	@"100F6C33-1735-4313-B402-38567131E5F3"

//  (READ/WRITE)
#define VALVE_REPORT_CHAR	@"100F6C34-1735-4313-B402-38567131E5F3"

// TODO: create CBUUID's in __attribute__((constructor)) rather than doing [CBUUID UUIDWithString:...] everywhere

#pragma pack(push,1)

typedef struct
{
	uint8_t		segmentHeader;
	uint8_t		featureReportMessageID;
	uint8_t		length;
	uint8_t		settingIdentifier;
	union {
		uint16_t	usPayload;
		uint32_t	uPayload;
		uint64_t	ulPayload;
		uint8_t		ucPayload[15];
	};
} bluetoothSegment;

typedef struct {
	uint8_t		id;
	union {
		bluetoothSegment segment;
		struct {
			uint8_t		segmentHeader;
			uint8_t		featureReportMessageID;
			uint8_t		length;
			uint8_t		settingIdentifier;
			union {
				uint16_t	usPayload;
				uint32_t	uPayload;
				uint64_t	ulPayload;
				uint8_t		ucPayload[15];
			};
		};
	};
} hidFeatureReport;

#pragma pack(pop)

size_t GetBluetoothSegmentSize(bluetoothSegment *segment)
{
    return segment->length + 3;
}

#define RingBuffer_cbElem   19
#define RingBuffer_nElem    4096

typedef struct {
	int _first, _last;
	uint8_t _data[ ( RingBuffer_nElem * RingBuffer_cbElem ) ];
	pthread_mutex_t accessLock;
} RingBuffer;

static void RingBuffer_init( RingBuffer *this )
{
    this->_first = -1;
    this->_last = 0;
    pthread_mutex_init( &this->accessLock, 0 );
}

static bool RingBuffer_write( RingBuffer *this, const uint8_t *src )
{
    pthread_mutex_lock( &this->accessLock );
    memcpy( &this->_data[ this->_last ], src, RingBuffer_cbElem );
    if ( this->_first == -1 )
    {
        this->_first = this->_last;
    }
    this->_last = ( this->_last + RingBuffer_cbElem ) % (RingBuffer_nElem * RingBuffer_cbElem);
    if ( this->_last == this->_first )
    {
        this->_first = ( this->_first + RingBuffer_cbElem ) % (RingBuffer_nElem * RingBuffer_cbElem);
        pthread_mutex_unlock( &this->accessLock );
        return false;
    }
    pthread_mutex_unlock( &this->accessLock );
    return true;
}

static bool RingBuffer_read( RingBuffer *this, uint8_t *dst )
{
    pthread_mutex_lock( &this->accessLock );
    if ( this->_first == -1 )
    {
        pthread_mutex_unlock( &this->accessLock );
        return false;
    }
    memcpy( dst, &this->_data[ this->_first ], RingBuffer_cbElem );
    this->_first = ( this->_first + RingBuffer_cbElem ) % (RingBuffer_nElem * RingBuffer_cbElem);
    if ( this->_first == this->_last )
    {
        this->_first = -1;
    }
    pthread_mutex_unlock( &this->accessLock );
    return true;
}


#pragma mark HIDBLEDevice Definition

typedef enum
{
	BLEDeviceWaitState_None,
	BLEDeviceWaitState_Waiting,
	BLEDeviceWaitState_Complete,
	BLEDeviceWaitState_Error
} BLEDeviceWaitState;

@interface HIDBLEDevice : NSObject <CBPeripheralDelegate>
{
	RingBuffer _inputReports;
	uint8_t	_featureReport[20];
	BLEDeviceWaitState	_waitStateForReadFeatureReport;
	BLEDeviceWaitState	_waitStateForWriteFeatureReport;
}

@property (nonatomic, readwrite) bool connected;
@property (nonatomic, readwrite) bool ready;

@property (nonatomic, strong) CBPeripheral     *bleSteamController;
@property (nonatomic, strong) CBCharacteristic *bleCharacteristicInput;
@property (nonatomic, strong) CBCharacteristic *bleCharacteristicReport;

- (id)initWithPeripheral:(CBPeripheral *)peripheral;

@end


@interface HIDBLEManager : NSObject <CBCentralManagerDelegate>

@property (nonatomic) int nPendingScans;
@property (nonatomic) int nPendingPairs;
@property (nonatomic, strong) CBCentralManager *centralManager;
@property (nonatomic, strong) NSMapTable<CBPeripheral *, HIDBLEDevice *> *deviceMap;
@property (nonatomic, retain) dispatch_queue_t bleSerialQueue;

+ (instancetype)sharedInstance;
- (void)startScan:(int)duration;
- (void)stopScan;
- (int)updateConnectedSteamControllers:(BOOL) bForce;
- (void)appWillResignActiveNotification:(NSNotification *)note;
- (void)appDidBecomeActiveNotification:(NSNotification *)note;

@end


// singleton class - access using HIDBLEManager.sharedInstance
@implementation HIDBLEManager

+ (instancetype)sharedInstance
{
	static HIDBLEManager *sharedInstance = nil;
	static dispatch_once_t onceToken;
	dispatch_once(&onceToken, ^{
		sharedInstance = [HIDBLEManager new];
		sharedInstance.nPendingScans = 0;
		sharedInstance.nPendingPairs = 0;

        // Bluetooth is currently only used for Steam Controllers, so check that hint
        // before initializing Bluetooth, which will prompt the user for permission.
		if ( SDL_GetHintBoolean( SDL_HINT_JOYSTICK_HIDAPI_STEAM, false ) )
		{
			[[NSNotificationCenter defaultCenter] addObserver:sharedInstance selector:@selector(appWillResignActiveNotification:) name: UIApplicationWillResignActiveNotification object:nil];
			[[NSNotificationCenter defaultCenter] addObserver:sharedInstance selector:@selector(appDidBecomeActiveNotification:) name:UIApplicationDidBecomeActiveNotification object:nil];

			// receive reports on a high-priority serial-queue. optionally put writes on the serial queue to avoid logical
			// race conditions talking to the controller from multiple threads, although BLE fragmentation/assembly means
			// that we can still screw this up.
			// most importantly we need to consume reports at a high priority to avoid the OS thinking we aren't really
			// listening to the BLE device, as iOS on slower devices may stop delivery of packets to the app WITHOUT ACTUALLY
			// DISCONNECTING FROM THE DEVICE if we don't react quickly enough to their delivery.
			// see also the error-handling states in the peripheral delegate to re-open the device if it gets closed
			sharedInstance.bleSerialQueue = dispatch_queue_create( "com.valvesoftware.steamcontroller.ble", DISPATCH_QUEUE_SERIAL );
			dispatch_set_target_queue( sharedInstance.bleSerialQueue, dispatch_get_global_queue( DISPATCH_QUEUE_PRIORITY_HIGH, 0 ) );

			// creating a CBCentralManager will always trigger a future centralManagerDidUpdateState:
			// where any scanning gets started or connecting to existing peripherals happens, it's never already in a
			// powered-on state for a newly launched application.
			sharedInstance.centralManager = [[CBCentralManager alloc] initWithDelegate:sharedInstance queue:sharedInstance.bleSerialQueue];
		}
		sharedInstance.deviceMap = [[NSMapTable alloc] initWithKeyOptions:NSMapTableWeakMemory valueOptions:NSMapTableStrongMemory capacity:4];
	});
	return sharedInstance;
}

// called for NSNotification UIApplicationWillResignActiveNotification
- (void)appWillResignActiveNotification:(NSNotification *)note
{
	// we'll get resign-active notification if pairing is happening.
	if ( self.nPendingPairs > 0 )
		return;

	for ( CBPeripheral *peripheral in self.deviceMap )
	{
		HIDBLEDevice *steamController = [self.deviceMap objectForKey:peripheral];
		if ( steamController )
		{
			steamController.connected = NO;
			steamController.ready = NO;
			[self.centralManager cancelPeripheralConnection:peripheral];
		}
	}
	[self.deviceMap removeAllObjects];
}

// called for NSNotification UIApplicationDidBecomeActiveNotification
//  whenever the application comes back from being inactive, trigger a 20s pairing scan and reconnect
//  any devices that may have paired while we were inactive.
- (void)appDidBecomeActiveNotification:(NSNotification *)note
{
	[self updateConnectedSteamControllers:true];
	[self startScan:20];
}

- (int)updateConnectedSteamControllers:(BOOL) bForce
{
	static uint64_t s_unLastUpdateTick = 0;
	static mach_timebase_info_data_t s_timebase_info;

	if ( self.centralManager == nil )
    {
		return 0;
    }

	if (s_timebase_info.denom == 0)
	{
		mach_timebase_info( &s_timebase_info );
	}

	uint64_t ticksNow = mach_approximate_time();
	if ( !bForce && ( ( (ticksNow - s_unLastUpdateTick) * s_timebase_info.numer ) / s_timebase_info.denom ) < (5ull * NSEC_PER_SEC) )
		return (int)self.deviceMap.count;

	// we can see previously connected BLE peripherals but can't connect until the CBCentralManager
	// is fully powered up - only do work when we are in that state
	if ( self.centralManager.state != CBManagerStatePoweredOn )
		return (int)self.deviceMap.count;

	// only update our last-check-time if we actually did work, otherwise there can be a long delay during initial power-up
	s_unLastUpdateTick = mach_approximate_time();

	// if a pair is in-flight, the central manager may still give it back via retrieveConnected... and
	// cause the SDL layer to attempt to initialize it while some of its endpoints haven't yet been established
	if ( self.nPendingPairs > 0 )
		return (int)self.deviceMap.count;

	NSArray<CBPeripheral *> *peripherals = [self.centralManager retrieveConnectedPeripheralsWithServices: @[ [CBUUID UUIDWithString:@"180A"]]];
	for ( CBPeripheral *peripheral in peripherals )
	{
		// we already know this peripheral
		if ( [self.deviceMap objectForKey: peripheral] != nil )
			continue;

		NSLog( @"connected peripheral: %@", peripheral );
		if ( [peripheral.name hasPrefix:@"Steam"] )
		{
			self.nPendingPairs += 1;
			HIDBLEDevice *steamController = [[HIDBLEDevice alloc] initWithPeripheral:peripheral];
			[self.deviceMap setObject:steamController forKey:peripheral];
			[self.centralManager connectPeripheral:peripheral options:nil];
		}
	}

	return (int)self.deviceMap.count;
}

// manual API for folks to start & stop scanning
- (void)startScan:(int)duration
{
	if ( self.centralManager == nil )
	{
		return;
	}

	NSLog( @"BLE: requesting scan for %d seconds", duration );
	@synchronized (self)
	{
		if ( _nPendingScans++ == 0 )
		{
			[self.centralManager scanForPeripheralsWithServices:nil options:nil];
		}
	}

	if ( duration != 0 )
	{
		dispatch_after( dispatch_time( DISPATCH_TIME_NOW, (int64_t)(duration * NSEC_PER_SEC)), dispatch_get_main_queue(), ^{
			[self stopScan];
		});
	}
}

- (void)stopScan
{
	if ( self.centralManager == nil )
	{
		return;
	}

	NSLog( @"BLE: stopping scan" );
	@synchronized (self)
	{
		if ( --_nPendingScans <= 0 )
		{
			_nPendingScans = 0;
			[self.centralManager stopScan];
		}
	}
}


#pragma mark CBCentralManagerDelegate Implementation

// called whenever the BLE hardware state changes.
- (void)centralManagerDidUpdateState:(CBCentralManager *)central
{
	switch ( central.state )
	{
		case CBManagerStatePoweredOn:
		{
			NSLog( @"CoreBluetooth BLE hardware is powered on and ready" );

			// at startup, if we have no already attached peripherals, do a 20s scan for new unpaired devices,
			// otherwise callers should occaisionally do additional scans. we don't want to continuously be
			// scanning because it drains battery, causes other nearby people to have a hard time pairing their
			// Steam Controllers, and may also trigger firmware weirdness when a device attempts to start
			// the pairing sequence multiple times concurrently
			if ( [self updateConnectedSteamControllers:false] == 0 )
			{
				// TODO: we could limit our scan to only peripherals supporting the SteamController service, but
				//  that service doesn't currently fit in the base advertising packet, we'd need to put it into an
				//  extended scan packet. Useful optimization downstream, but not currently necessary
				//	NSArray *services = @[[CBUUID UUIDWithString:VALVE_SERVICE]];
				[self startScan:20];
			}
			break;
		}

		case CBManagerStatePoweredOff:
			NSLog( @"CoreBluetooth BLE hardware is powered off" );
			break;

		case CBManagerStateUnauthorized:
			NSLog( @"CoreBluetooth BLE state is unauthorized" );
			break;

		case CBManagerStateUnknown:
			NSLog( @"CoreBluetooth BLE state is unknown" );
			break;

		case CBManagerStateUnsupported:
			NSLog( @"CoreBluetooth BLE hardware is unsupported on this platform" );
			break;

		case CBManagerStateResetting:
			NSLog( @"CoreBluetooth BLE manager is resetting" );
			break;
	}
}

- (void)centralManager:(CBCentralManager *)central didConnectPeripheral:(CBPeripheral *)peripheral
{
	HIDBLEDevice *steamController = [_deviceMap objectForKey:peripheral];
	steamController.connected = YES;
	self.nPendingPairs -= 1;
}

- (void)centralManager:(CBCentralManager *)central didFailToConnectPeripheral:(CBPeripheral *)peripheral error:(NSError *)error
{
	NSLog( @"Failed to connect: %@", error );
	[_deviceMap removeObjectForKey:peripheral];
	self.nPendingPairs -= 1;
}

- (void)centralManager:(CBCentralManager *)central didDiscoverPeripheral:(CBPeripheral *)peripheral advertisementData:(NSDictionary *)advertisementData RSSI:(NSNumber *)RSSI
{
	NSString *localName = [advertisementData objectForKey:CBAdvertisementDataLocalNameKey];
	NSString *log = [NSString stringWithFormat:@"Found '%@'", localName];

	if ( [localName hasPrefix:@"Steam"] )
	{
		NSLog( @"%@ : %@ - %@", log, peripheral, advertisementData );
		self.nPendingPairs += 1;
		HIDBLEDevice *steamController = [[HIDBLEDevice alloc] initWithPeripheral:peripheral];
		[self.deviceMap setObject:steamController forKey:peripheral];
		[self.centralManager connectPeripheral:peripheral options:nil];
	}
}

- (void)centralManager:(CBCentralManager *)central didDisconnectPeripheral:(CBPeripheral *)peripheral error:(NSError *)error
{
	HIDBLEDevice *steamController = [self.deviceMap objectForKey:peripheral];
	if ( steamController )
	{
		steamController.connected = NO;
		steamController.ready = NO;
		[self.deviceMap removeObjectForKey:peripheral];
	}
}

@end


// Core Bluetooth devices calling back on event boundaries of their run-loops. so annoying.
static void process_pending_events(void)
{
	CFRunLoopRunResult res;
	do
	{
		res = CFRunLoopRunInMode( kCFRunLoopDefaultMode, 0.001, FALSE );
	}
	while( res != kCFRunLoopRunFinished && res != kCFRunLoopRunTimedOut );
}

@implementation HIDBLEDevice

- (id)init
{
	if ( self = [super init] )
	{
        RingBuffer_init( &_inputReports );
		self.bleSteamController = nil;
		self.bleCharacteristicInput = nil;
		self.bleCharacteristicReport = nil;
		_connected = NO;
		_ready = NO;
	}
	return self;
}

- (id)initWithPeripheral:(CBPeripheral *)peripheral
{
	if ( self = [super init] )
	{
        RingBuffer_init( &_inputReports );
		_connected = NO;
		_ready = NO;
		self.bleSteamController = peripheral;
		if ( peripheral )
		{
			peripheral.delegate = self;
		}
		self.bleCharacteristicInput = nil;
		self.bleCharacteristicReport = nil;
	}
	return self;
}

- (void)setConnected:(bool)connected
{
	_connected = connected;
	if ( _connected )
	{
		[_bleSteamController discoverServices:nil];
	}
	else
	{
		NSLog( @"Disconnected" );
	}
}

- (size_t)read_input_report:(uint8_t *)dst
{
	if ( RingBuffer_read( &_inputReports, dst+1 ) )
	{
		*dst = 0x03;
		return 20;
	}
	return 0;
}

- (int)send_report:(const uint8_t *)data length:(size_t)length
{
	[_bleSteamController writeValue:[NSData dataWithBytes:data length:length] forCharacteristic:_bleCharacteristicReport type:CBCharacteristicWriteWithResponse];
	return (int)length;
}

- (int)send_feature_report:(hidFeatureReport *)report
{
#if FEATURE_REPORT_LOGGING
	uint8_t *reportBytes = (uint8_t *)report;

	NSLog( @"HIDBLE:send_feature_report (%02zu/19) [%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x]", GetBluetoothSegmentSize( report->segment ),
		  reportBytes[1], reportBytes[2], reportBytes[3], reportBytes[4], reportBytes[5], reportBytes[6],
		  reportBytes[7], reportBytes[8], reportBytes[9], reportBytes[10], reportBytes[11], reportBytes[12],
		  reportBytes[13], reportBytes[14], reportBytes[15], reportBytes[16], reportBytes[17], reportBytes[18],
		  reportBytes[19] );
#endif

	int sendSize = (int)GetBluetoothSegmentSize( &report->segment );
	if ( sendSize > 20 )
		sendSize = 20;

#if 1
	// fire-and-forget - we are going to not wait for the response here because all Steam Controller BLE send_feature_report's are ignored,
	//  except errors.
	[_bleSteamController writeValue:[NSData dataWithBytes:&report->segment length:sendSize] forCharacteristic:_bleCharacteristicReport type:CBCharacteristicWriteWithResponse];

	// pretend we received a result anybody cares about
	return 19;

#else
	// this is technically the correct send_feature_report logic if you want to make sure it gets through and is
	// acknowledged or errors out
	_waitStateForWriteFeatureReport = BLEDeviceWaitState_Waiting;
	[_bleSteamController writeValue:[NSData dataWithBytes:&report->segment length:sendSize
									 ] forCharacteristic:_bleCharacteristicReport type:CBCharacteristicWriteWithResponse];

	while ( _waitStateForWriteFeatureReport == BLEDeviceWaitState_Waiting )
	{
		process_pending_events();
	}

	if ( _waitStateForWriteFeatureReport == BLEDeviceWaitState_Error )
	{
		_waitStateForWriteFeatureReport = BLEDeviceWaitState_None;
		return -1;
	}

	_waitStateForWriteFeatureReport = BLEDeviceWaitState_None;
	return 19;
#endif
}

- (int)get_feature_report:(uint8_t)feature into:(uint8_t *)buffer
{
	_waitStateForReadFeatureReport = BLEDeviceWaitState_Waiting;
	[_bleSteamController readValueForCharacteristic:_bleCharacteristicReport];

	while ( _waitStateForReadFeatureReport == BLEDeviceWaitState_Waiting )
		process_pending_events();

	if ( _waitStateForReadFeatureReport == BLEDeviceWaitState_Error )
	{
		_waitStateForReadFeatureReport = BLEDeviceWaitState_None;
		return -1;
	}

	memcpy( buffer, _featureReport, sizeof(_featureReport) );

	_waitStateForReadFeatureReport = BLEDeviceWaitState_None;

#if FEATURE_REPORT_LOGGING
	NSLog( @"HIDBLE:get_feature_report (19) [%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x]",
		  buffer[1], buffer[2], buffer[3], buffer[4], buffer[5], buffer[6],
		  buffer[7], buffer[8], buffer[9], buffer[10], buffer[11], buffer[12],
		  buffer[13], buffer[14], buffer[15], buffer[16], buffer[17], buffer[18],
		  buffer[19] );
#endif

	return 19;
}

#pragma mark CBPeripheralDelegate Implementation

- (void)peripheral:(CBPeripheral *)peripheral didDiscoverServices:(NSError *)error
{
	for (CBService *service in peripheral.services)
	{
		NSLog( @"Found Service: %@", service );
		if ( [service.UUID isEqual:[CBUUID UUIDWithString:VALVE_SERVICE]] )
		{
			[peripheral discoverCharacteristics:nil forService:service];
		}
	}
}

- (void)peripheral:(CBPeripheral *)peripheral didDiscoverDescriptorsForCharacteristic:(CBCharacteristic *)characteristic error:(NSError *)error
{
	// nothing yet needed here, enable for logging
	if ( /* DISABLES CODE */ (0) )
	{
		for ( CBDescriptor *descriptor in characteristic.descriptors )
		{
			NSLog( @" - Descriptor '%@'", descriptor );
		}
	}
}

- (void)peripheral:(CBPeripheral *)peripheral didDiscoverCharacteristicsForService:(CBService *)service error:(NSError *)error
{
	if ([service.UUID isEqual:[CBUUID UUIDWithString:VALVE_SERVICE]])
	{
		for (CBCharacteristic *aChar in service.characteristics)
		{
			NSLog( @"Found Characteristic %@", aChar );

			if ( [aChar.UUID isEqual:[CBUUID UUIDWithString:VALVE_INPUT_CHAR]] )
			{
				self.bleCharacteristicInput = aChar;
			}
			else if ( [aChar.UUID isEqual:[CBUUID UUIDWithString:VALVE_REPORT_CHAR]] )
			{
				self.bleCharacteristicReport = aChar;
				[self.bleSteamController discoverDescriptorsForCharacteristic: aChar];
			}
		}
	}
}

- (void)peripheral:(CBPeripheral *)peripheral didUpdateValueForCharacteristic:(CBCharacteristic *)characteristic error:(NSError *)error
{
	static uint64_t s_ticksLastOverflowReport = 0;

	// receiving an input report is the final indicator that the user accepted a pairing
	// request and that we successfully established notification. CoreBluetooth has no
	// notification of the pairing acknowledgement, which is a bad oversight.
	if ( self.ready == NO )
	{
		self.ready = YES;
		HIDBLEManager.sharedInstance.nPendingPairs -= 1;
	}

	if ( [characteristic.UUID isEqual:_bleCharacteristicInput.UUID] )
	{
		NSData *data = [characteristic value];
		if ( data.length != 19 )
		{
			NSLog( @"HIDBLE: incoming data is %lu bytes should be exactly 19", (unsigned long)data.length );
		}
		if ( !RingBuffer_write( &_inputReports, (const uint8_t *)data.bytes ) )
		{
			uint64_t ticksNow = mach_approximate_time();
			if ( ticksNow - s_ticksLastOverflowReport > (5ull * NSEC_PER_SEC / 10) )
			{
				NSLog( @"HIDBLE: input report buffer overflow" );
				s_ticksLastOverflowReport = ticksNow;
			}
		}
	}
	else if ( [characteristic.UUID isEqual:_bleCharacteristicReport.UUID] )
	{
		memset( _featureReport, 0, sizeof(_featureReport) );

		if ( error != nil )
		{
			NSLog( @"HIDBLE: get_feature_report error: %@", error );
			_waitStateForReadFeatureReport = BLEDeviceWaitState_Error;
		}
		else
		{
			NSData *data = [characteristic value];
			if ( data.length != 20 )
			{
				NSLog( @"HIDBLE: incoming data is %lu bytes should be exactly 20", (unsigned long)data.length );
			}
			memcpy( _featureReport, data.bytes, MIN( data.length, sizeof(_featureReport) ) );
			_waitStateForReadFeatureReport = BLEDeviceWaitState_Complete;
		}
	}
}

- (void)peripheral:(CBPeripheral *)peripheral didWriteValueForCharacteristic:(CBCharacteristic *)characteristic error:(NSError *)error
{
	if ( [characteristic.UUID isEqual:[CBUUID UUIDWithString:VALVE_REPORT_CHAR]] )
	{
		if ( error != nil )
		{
			NSLog( @"HIDBLE: write_feature_report error: %@", error );
			_waitStateForWriteFeatureReport = BLEDeviceWaitState_Error;
		}
		else
		{
			_waitStateForWriteFeatureReport = BLEDeviceWaitState_Complete;
		}
	}
}

- (void)peripheral:(CBPeripheral *)peripheral didUpdateNotificationStateForCharacteristic:(CBCharacteristic *)characteristic error:(NSError *)error
{
	NSLog( @"didUpdateNotifcationStateForCharacteristic %@ (%@)", characteristic, error );
}

@end


#pragma mark hid_api implementation

struct hid_device_ {
	void *device_handle;
	int blocking;
	struct hid_device_info* device_info;
	hid_device *next;
};

int HID_API_EXPORT HID_API_CALL hid_init(void)
{
	return ( HIDBLEManager.sharedInstance == nil ) ? -1 : 0;
}

int HID_API_EXPORT HID_API_CALL hid_exit(void)
{
	return 0;
}

void HID_API_EXPORT HID_API_CALL hid_ble_scan( int bStart )
{
	HIDBLEManager *bleManager = HIDBLEManager.sharedInstance;
	if ( bStart )
	{
		[bleManager startScan:0];
	}
	else
	{
		[bleManager stopScan];
	}
}

HID_API_EXPORT hid_device * HID_API_CALL hid_open(unsigned short vendor_id, unsigned short product_id, const wchar_t *serial_number)
{
	return NULL;
}

HID_API_EXPORT hid_device * HID_API_CALL hid_open_path( const char *path )
{
	hid_device *result = NULL;
	NSString *nssPath = [NSString stringWithUTF8String:path];
	HIDBLEManager *bleManager = HIDBLEManager.sharedInstance;
	NSEnumerator<HIDBLEDevice *> *devices = [bleManager.deviceMap objectEnumerator];

	for ( HIDBLEDevice *device in devices )
	{
		// we have the device but it hasn't found its service or characteristics until it is connected
		if ( !device.ready || !device.connected || !device.bleCharacteristicInput )
			continue;

		if ( [device.bleSteamController.identifier.UUIDString isEqualToString:nssPath] )
		{
			result = (hid_device *)malloc( sizeof( hid_device ) );
			memset( result, 0, sizeof( hid_device ) );
			result->device_handle = (void*)CFBridgingRetain( device );
			result->blocking = NO;
			// enable reporting input events on the characteristic
			[device.bleSteamController setNotifyValue:YES forCharacteristic:device.bleCharacteristicInput];
			return result;
		}
	}
	return result;
}

void  HID_API_EXPORT hid_free_enumeration(struct hid_device_info *devs)
{
	/* This function is identical to the Linux version. Platform independent. */
	struct hid_device_info *d = devs;
	while (d) {
		struct hid_device_info *next = d->next;
		free(d->path);
		free(d->serial_number);
		free(d->manufacturer_string);
		free(d->product_string);
		free(d);
		d = next;
	}
}

int HID_API_EXPORT hid_set_nonblocking(hid_device *dev, int nonblock)
{
	/* All Nonblocking operation is handled by the library. */
	dev->blocking = !nonblock;

	return 0;
}

static struct hid_device_info *create_device_info_for_hid_device(HIDBLEDevice *device)
{
    // We currently only support the Steam Controller
    struct hid_device_info *device_info = (struct hid_device_info *)malloc( sizeof(struct hid_device_info) );
    memset( device_info, 0, sizeof(struct hid_device_info) );
    device_info->path = strdup( device.bleSteamController.identifier.UUIDString.UTF8String );
    device_info->vendor_id = VALVE_USB_VID;
    device_info->product_id = D0G_BLE2_PID;
    device_info->product_string = wcsdup( L"Steam Controller" );
    device_info->manufacturer_string = wcsdup( L"Valve Corporation" );
    device_info->bus_type = HID_API_BUS_BLUETOOTH;
    return device_info;
}

struct hid_device_info  HID_API_EXPORT *hid_enumerate(unsigned short vendor_id, unsigned short product_id)
{ @autoreleasepool {
	struct hid_device_info *root = NULL;

	/* See if there are any devices we should skip in enumeration */
	if (SDL_HIDAPI_ShouldIgnoreDevice(HID_API_BUS_BLUETOOTH, VALVE_USB_VID, D0G_BLE2_PID, 0, 0)) {
		return NULL;
	}

	if ( ( vendor_id == 0 || vendor_id == VALVE_USB_VID ) &&
	     ( product_id == 0 || product_id == D0G_BLE2_PID ) )
	{
		HIDBLEManager *bleManager = HIDBLEManager.sharedInstance;
		[bleManager updateConnectedSteamControllers:false];
		NSEnumerator<HIDBLEDevice *> *devices = [bleManager.deviceMap objectEnumerator];
		for ( HIDBLEDevice *device in devices )
		{
			// there are several brief windows in connecting to an already paired device and
			// one long window waiting for users to confirm pairing where we don't want
			// to consider a device ready - if we hand it back to SDL or another
			// Steam Controller consumer, their additional SC setup work will fail
			// in unusual/silent ways and we can actually corrupt the BLE stack for
			// the entire system and kill the appletv remote's Menu button (!)
			if ( device.bleSteamController.state != CBPeripheralStateConnected ||
				 device.connected == NO || device.ready == NO )
			{
				if ( device.ready == NO && device.bleCharacteristicInput != nil )
				{
					// attempt to register for input reports. this call will silently fail
					// until the pairing finalizes with user acceptance. oh, apple.
					[device.bleSteamController setNotifyValue:YES forCharacteristic:device.bleCharacteristicInput];
				}
				continue;
			}
			struct hid_device_info *device_info = create_device_info_for_hid_device(device);
			device_info->next = root;
			root = device_info;
		}
	}
	return root;
}}

int HID_API_EXPORT_CALL hid_get_manufacturer_string(hid_device *dev, wchar_t *string, size_t maxlen)
{
	static wchar_t s_wszManufacturer[] = L"Valve Corporation";
	wcsncpy( string, s_wszManufacturer, sizeof(s_wszManufacturer)/sizeof(s_wszManufacturer[0]) );
	return 0;
}

int HID_API_EXPORT_CALL hid_get_product_string(hid_device *dev, wchar_t *string, size_t maxlen)
{
	static wchar_t s_wszProduct[] = L"Steam Controller";
	wcsncpy( string, s_wszProduct, sizeof(s_wszProduct)/sizeof(s_wszProduct[0]) );
	return 0;
}

int HID_API_EXPORT_CALL hid_get_serial_number_string(hid_device *dev, wchar_t *string, size_t maxlen)
{
	static wchar_t s_wszSerial[] = L"12345";
	wcsncpy( string, s_wszSerial, sizeof(s_wszSerial)/sizeof(s_wszSerial[0]) );
	return 0;
}

int HID_API_EXPORT_CALL hid_get_indexed_string(hid_device *dev, int string_index, wchar_t *string, size_t maxlen)
{
	return -1;
}

struct hid_device_info *hid_get_device_info(hid_device *dev)
{
    HIDBLEDevice *device_handle = (__bridge HIDBLEDevice *)dev->device_handle;

	if (!dev->device_info) {
		// Lazy initialize device_info
		dev->device_info = create_device_info_for_hid_device(device_handle);
	}

	// create_device_info_for_hid_device will set an error if needed
	return dev->device_info;
}

int hid_get_report_descriptor(hid_device *device, unsigned char *buf, size_t buf_size)
{
    // Not implemented
    return -1;
}

int HID_API_EXPORT hid_write(hid_device *dev, const unsigned char *data, size_t length)
{
    HIDBLEDevice *device_handle = (__bridge HIDBLEDevice *)dev->device_handle;

	if ( !device_handle.connected )
		return -1;

	return [device_handle send_report:data length:length];
}

void HID_API_EXPORT hid_close(hid_device *dev)
{
    HIDBLEDevice *device_handle = CFBridgingRelease( dev->device_handle );

	// disable reporting input events on the characteristic
	if ( device_handle.connected ) {
		[device_handle.bleSteamController setNotifyValue:NO forCharacteristic:device_handle.bleCharacteristicInput];
	}

    hid_free_enumeration(dev->device_info);

	free( dev );
}

int HID_API_EXPORT hid_send_feature_report(hid_device *dev, const unsigned char *data, size_t length)
{
    HIDBLEDevice *device_handle = (__bridge HIDBLEDevice *)dev->device_handle;

	if ( !device_handle.connected )
		return -1;

	return [device_handle send_feature_report:(hidFeatureReport *)(void *)data];
}

int HID_API_EXPORT hid_get_feature_report(hid_device *dev, unsigned char *data, size_t length)
{
    HIDBLEDevice *device_handle = (__bridge HIDBLEDevice *)dev->device_handle;

	if ( !device_handle.connected )
		return -1;

	size_t written = [device_handle get_feature_report:data[0] into:data];

	return written == length-1 ? (int)length : (int)written;
}

int HID_API_EXPORT hid_get_input_report(hid_device *dev, unsigned char *data, size_t length)
{
    // Not supported
    return -1;
}

int HID_API_EXPORT hid_read(hid_device *dev, unsigned char *data, size_t length)
{
    HIDBLEDevice *device_handle = (__bridge HIDBLEDevice *)dev->device_handle;

	if ( !device_handle.connected )
		return -1;

	return hid_read_timeout(dev, data, length, 0);
}

int HID_API_EXPORT hid_read_timeout(hid_device *dev, unsigned char *data, size_t length, int milliseconds)
{
    HIDBLEDevice *device_handle = (__bridge HIDBLEDevice *)dev->device_handle;

	if ( !device_handle.connected )
		return -1;

	if ( milliseconds != 0 )
	{
		NSLog( @"hid_read_timeout with non-zero wait" );
	}
	int result = (int)[device_handle read_input_report:data];
#if FEATURE_REPORT_LOGGING
	NSLog( @"HIDBLE:hid_read_timeout (%d) [%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x]", result,
		  data[1], data[2], data[3], data[4], data[5], data[6],
		  data[7], data[8], data[9], data[10], data[11], data[12],
		  data[13], data[14], data[15], data[16], data[17], data[18],
		  data[19] );
#endif
	return result;
}

HID_API_EXPORT const wchar_t* HID_API_CALL hid_error(hid_device *dev)
{
	return NULL;
}

#endif /* !SDL_HIDAPI_DISABLED */

#endif /* SDL_PLATFORM_IOS || SDL_PLATFORM_TVOS */
