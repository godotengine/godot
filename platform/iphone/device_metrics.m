/*************************************************************************/
/*  device_metrics.m                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#import "device_metrics.h"

@implementation GodotDeviceMetrics

+ (NSDictionary *)dpiList {
	return @{
		@[
			@"iPad1,1",
			@"iPad2,1",
			@"iPad2,2",
			@"iPad2,3",
			@"iPad2,4",
		] : @132,
		@[
			@"iPhone1,1",
			@"iPhone1,2",
			@"iPhone2,1",
			@"iPad2,5",
			@"iPad2,6",
			@"iPad2,7",
			@"iPod1,1",
			@"iPod2,1",
			@"iPod3,1",
		] : @163,
		@[
			@"iPad3,1",
			@"iPad3,2",
			@"iPad3,3",
			@"iPad3,4",
			@"iPad3,5",
			@"iPad3,6",
			@"iPad4,1",
			@"iPad4,2",
			@"iPad4,3",
			@"iPad5,3",
			@"iPad5,4",
			@"iPad6,3",
			@"iPad6,4",
			@"iPad6,7",
			@"iPad6,8",
			@"iPad6,11",
			@"iPad6,12",
			@"iPad7,1",
			@"iPad7,2",
			@"iPad7,3",
			@"iPad7,4",
			@"iPad7,5",
			@"iPad7,6",
			@"iPad7,11",
			@"iPad7,12",
			@"iPad8,1",
			@"iPad8,2",
			@"iPad8,3",
			@"iPad8,4",
			@"iPad8,5",
			@"iPad8,6",
			@"iPad8,7",
			@"iPad8,8",
			@"iPad8,9",
			@"iPad8,10",
			@"iPad8,11",
			@"iPad8,12",
			@"iPad11,3",
			@"iPad11,4",
		] : @264,
		@[
			@"iPhone3,1",
			@"iPhone3,2",
			@"iPhone3,3",
			@"iPhone4,1",
			@"iPhone5,1",
			@"iPhone5,2",
			@"iPhone5,3",
			@"iPhone5,4",
			@"iPhone6,1",
			@"iPhone6,2",
			@"iPhone7,2",
			@"iPhone8,1",
			@"iPhone8,4",
			@"iPhone9,1",
			@"iPhone9,3",
			@"iPhone10,1",
			@"iPhone10,4",
			@"iPhone11,8",
			@"iPhone12,1",
			@"iPhone12,8",
			@"iPad4,4",
			@"iPad4,5",
			@"iPad4,6",
			@"iPad4,7",
			@"iPad4,8",
			@"iPad4,9",
			@"iPad5,1",
			@"iPad5,2",
			@"iPad11,1",
			@"iPad11,2",
			@"iPod4,1",
			@"iPod5,1",
			@"iPod7,1",
			@"iPod9,1",
		] : @326,
		@[
			@"iPhone7,1",
			@"iPhone8,2",
			@"iPhone9,2",
			@"iPhone9,4",
			@"iPhone10,2",
			@"iPhone10,5",
		] : @401,
		@[
			@"iPhone10,3",
			@"iPhone10,6",
			@"iPhone11,2",
			@"iPhone11,4",
			@"iPhone11,6",
			@"iPhone12,3",
			@"iPhone12,5",
		] : @458,
	};
}

@end
