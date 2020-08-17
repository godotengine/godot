/*************************************************************************/
/*  godot_vars.h                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef GODOT_VARS_H
#define GODOT_VARS_H

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface GDBool : NSObject
- (id)initWithBool:(BOOL)boolValue;
- (BOOL)getValue;
@end

@interface GDInteger : NSObject
- (id)initWithInt:(NSNumber *)intValue;
- (int)getValue;
@end

@interface GDFloat : NSObject
- (id)initWithFloat:(NSNumber *)floatValue;
- (float)getValue;
@end

@interface GDDouble : NSObject
- (id)initWithDouble:(NSNumber *)doubleValue;
- (double)getValue;
@end

@interface GDString : NSObject
- (id)initWithString:(NSString *)stringValue;
- (NSString *)getValue;
@end

@interface GDIntegerArray : NSObject
- (id)initWithIntegerArray:(NSArray<NSNumber *> *)intArrayValue;
- (NSArray<NSNumber *> *)getValue;
@end

@interface GDFloatArray : NSObject
- (id)initWithFloatArray:(NSArray<NSNumber *> *)floatArrayValue;
- (NSArray<NSNumber *> *)getValue;
@end

@interface GDDoubleArray : NSObject
- (id)initWithDoubleArray:(NSArray<NSNumber *> *)doubleArrayValue;
- (NSArray<NSNumber *> *)getValue;
@end

@interface GDByteArray : NSObject
- (id)initWithByteArray:(NSData *)byteArrayValue;
- (NSData *)getValue;
@end

@interface GDStringArray : NSObject
- (id)initWithStringArray:(NSArray<NSString *> *)stringArrayValue;
- (NSArray<NSString *> *)getValue;
@end

@interface GDDictionary : NSObject
- (id)initWithDictionary:(NSDictionary *)dictionaryValue;
- (NSDictionary *)getValue;
@end
NS_ASSUME_NONNULL_END

#endif
