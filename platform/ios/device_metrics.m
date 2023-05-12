/**************************************************************************/
/*  device_metrics.m                                                      */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#import "device_metrics.h"

@implementation GodotDeviceMetrics

+ (NSDictionary *)dpiList {
	return @{
		@[
			@"iPad1,1", // iPad 1th Gen
			@"iPad1,2", // iPad 1th Gen (3G)
			@"iPad2,1", // iPad 2nd Gen
			@"iPad2,2", // iPad 2nd Gen (GSM)
			@"iPad2,3", // iPad 2nd Gen (CDMA)
			@"iPad2,4", // iPad 2nd Gen
		] : @132,
		@[
			@"iPhone1,1", // iPhone 1st Gen
			@"iPhone1,2", // iPhone 3G
			@"iPhone2,1", // iPhone 3GS
			@"iPad2,5", // iPad mini
			@"iPad2,6", // iPad mini (GSM+LTE)
			@"iPad2,7", // iPad mini (CDMA+LTE)
			@"iPod1,1", // iPod 1st Gen
			@"iPod2,1", // iPod 2nd Gen
			@"iPod3,1", // iPod 3rd Gen
		] : @163,
		@[
			@"iPad3,1", // iPad 3rd Gen
			@"iPad3,2", // iPad 3rd Gen (CDMA)
			@"iPad3,3", // iPad 3rd Gen (GSM)
			@"iPad3,4", // iPad 4th Gen
			@"iPad3,5", // iPad 4th Gen (GSM+LTE)
			@"iPad3,6", // iPad 4th Gen (CDMA+LTE)
			@"iPad4,1", // iPad Air (WiFi)
			@"iPad4,2", // iPad Air (GSM+CDMA)
			@"iPad4,3", // iPad Air (China)
			@"iPad4,7", // iPad mini 3 (WiFi)
			@"iPad4,8", // iPad mini 3 (GSM+CDMA)
			@"iPad4,9", // iPad Mini 3 (China)
			@"iPad6,3", // iPad Pro (9.7 inch, WiFi)
			@"iPad6,4", // iPad Pro (9.7 inch, WiFi+LTE)
			@"iPad6,7", // iPad Pro (12.9 inch, WiFi)
			@"iPad6,8", // iPad Pro (12.9 inch, WiFi+LTE)
			@"iPad6,11", // iPad 5th Gen (2017)
			@"iPad6,12", // iPad 5th Gen (2017)
			@"iPad7,1", // iPad Pro 2nd Gen (WiFi)
			@"iPad7,2", // iPad Pro 2nd Gen (WiFi+Cellular)
			@"iPad7,3", // iPad Pro 10.5-inch 2nd Gen
			@"iPad7,4", // iPad Pro 10.5-inch 2nd Gen
			@"iPad7,5", // iPad 6th Gen (WiFi)
			@"iPad7,6", // iPad 6th Gen (WiFi+Cellular)
			@"iPad7,11", // iPad 7th Gen 10.2-inch (WiFi)
			@"iPad7,12", // iPad 7th Gen 10.2-inch (WiFi+Cellular)
			@"iPad8,1", // iPad Pro 11 inch 3rd Gen (WiFi)
			@"iPad8,2", // iPad Pro 11 inch 3rd Gen (1TB, WiFi)
			@"iPad8,3", // iPad Pro 11 inch 3rd Gen (WiFi+Cellular)
			@"iPad8,4", // iPad Pro 11 inch 3rd Gen (1TB, WiFi+Cellular)
			@"iPad8,5", // iPad Pro 12.9 inch 3rd Gen (WiFi)
			@"iPad8,6", // iPad Pro 12.9 inch 3rd Gen (1TB, WiFi)
			@"iPad8,7", // iPad Pro 12.9 inch 3rd Gen (WiFi+Cellular)
			@"iPad8,8", // iPad Pro 12.9 inch 3rd Gen (1TB, WiFi+Cellular)
			@"iPad8,9", // iPad Pro 11 inch 4th Gen (WiFi)
			@"iPad8,10", // iPad Pro 11 inch 4th Gen (WiFi+Cellular)
			@"iPad8,11", // iPad Pro 12.9 inch 4th Gen (WiFi)
			@"iPad8,12", // iPad Pro 12.9 inch 4th Gen (WiFi+Cellular)
			@"iPad11,3", // iPad Air 3rd Gen (WiFi)
			@"iPad11,4", // iPad Air 3rd Gen
			@"iPad11,6", // iPad 8th Gen (WiFi)
			@"iPad11,7", // iPad 8th Gen (WiFi+Cellular)
			@"iPad12,1", // iPad 9th Gen (WiFi)
			@"iPad12,2", // iPad 9th Gen (WiFi+Cellular)
			@"iPad13,1", // iPad Air 4th Gen (WiFi)
			@"iPad13,2", // iPad Air 4th Gen (WiFi+Cellular)
			@"iPad13,4", // iPad Pro 11 inch 5th Gen
			@"iPad13,5", // iPad Pro 11 inch 5th Gen
			@"iPad13,6", // iPad Pro 11 inch 5th Gen
			@"iPad13,7", // iPad Pro 11 inch 5th Gen
			@"iPad13,8", // iPad Pro 12.9 inch 5th Gen
			@"iPad13,9", // iPad Pro 12.9 inch 5th Gen
			@"iPad13,10", // iPad Pro 12.9 inch 5th Gen
			@"iPad13,11", // iPad Pro 12.9 inch 5th Gen
			@"iPad13,16", // iPad Air 5th Gen (WiFi)
			@"iPad13,17", // iPad Air 5th Gen (WiFi+Cellular)
			@"iPad13,18", // iPad 10th Gen
			@"iPad13,19", // iPad 10th Gen
			@"iPad14,3", // iPad Pro 11 inch 6th Gen
			@"iPad14,4", // iPad Pro 11 inch 6th Gen
			@"iPad14,5", // iPad Pro 12.9 inch 6th Gen
			@"iPad14,6", // iPad Pro 12.9 inch 6th Gen
		] : @264,
		@[
			@"iPhone3,1", // iPhone 4
			@"iPhone3,2", // iPhone 4 (GSM)
			@"iPhone3,3", // iPhone 4 (CDMA)
			@"iPhone4,1", // iPhone 4S
			@"iPhone5,1", // iPhone 5 (GSM)
			@"iPhone5,2", // iPhone 5 (GSM+CDMA)
			@"iPhone5,3", // iPhone 5C (GSM)
			@"iPhone5,4", // iPhone 5C (Global)
			@"iPhone6,1", // iPhone 5S (GSM)
			@"iPhone6,2", // iPhone 5S (Global)
			@"iPhone7,2", // iPhone 6
			@"iPhone8,1", // iPhone 6s
			@"iPhone8,4", // iPhone SE (GSM)
			@"iPhone9,1", // iPhone 7
			@"iPhone9,3", // iPhone 7
			@"iPhone10,1", // iPhone 8
			@"iPhone10,4", // iPhone 8
			@"iPhone11,8", // iPhone XR
			@"iPhone12,1", // iPhone 11
			@"iPhone12,8", // iPhone SE 2nd gen
			@"iPhone14,6", // iPhone SE 3rd gen
			@"iPad4,4", // iPad mini Retina (WiFi)
			@"iPad4,5", // iPad mini Retina (GSM+CDMA)
			@"iPad4,6", // iPad mini Retina (China)
			@"iPad5,1", // iPad mini 4th Gen (WiFi)
			@"iPad5,2", // iPad mini 4th Gen
			@"iPad5,3", // iPad Air 2 (WiFi)
			@"iPad5,4", // iPad Air 2
			@"iPad11,1", // iPad mini 5th Gen (WiFi)
			@"iPad11,2", // iPad mini 5th Gen
			@"iPad14,1", // iPad mini 6th Gen (WiFi)
			@"iPad14,2", // iPad mini 6th Gen
			@"iPod4,1", // iPod 4th Gen
			@"iPod5,1", // iPod 5th Gen
			@"iPod7,1", // iPod 6th Gen
			@"iPod9,1", // iPod 7th Gen
		] : @326,
		@[
			@"iPhone7,1", // iPhone 6 Plus
			@"iPhone8,2", // iPhone 6s Plus
			@"iPhone9,2", // iPhone 7 Plus
			@"iPhone9,4", // iPhone 7 Plus
			@"iPhone10,2", // iPhone 8 Plus
			@"iPhone10,5", // iPhone 8 Plus
		] : @401,
		@[
			@"iPhone10,3", // iPhone X Global
			@"iPhone10,6", // iPhone X GSM
			@"iPhone11,2", // iPhone XS
			@"iPhone11,4", // iPhone XS Max
			@"iPhone11,6", // iPhone XS Max Global
			@"iPhone12,3", // iPhone 11 Pro
			@"iPhone12,5", // iPhone 11 Pro Max
			@"iPhone13,4", // iPhone 12 Pro Max
			@"iPhone14,3", // iPhone 13 Pro Max
			@"iPhone14,8", // iPhone 14 Plus
		] : @458,
		@[
			@"iPhone13,2", // iPhone 12
			@"iPhone13,3", // iPhone 12 Pro
			@"iPhone14,2", // iPhone 13 Pro
			@"iPhone14,5", // iPhone 13
			@"iPhone14,7", // iPhone 14
			@"iPhone15,2", // iPhone 14 Pro
			@"iPhone15,3", // iPhone 14 Pro Max
		] : @460,
		@[
			@"iPhone13,1", // iPhone 12 Mini
			@"iPhone14,4", // iPhone 13 Mini
		] : @476
	};
}

@end
