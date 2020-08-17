/*************************************************************************/
/*  godot_vars.mm                                                    */
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

#include "godot_vars.h"

@interface GDBool ()
@property(nonatomic, assign) BOOL value;
@end
@implementation GDBool
- (id)initWithBool:(BOOL)boolValue {
	self = [super init];
	if (self) {
		self.value = boolValue;
	}
	return self;
}

- (BOOL)getValue {
	return self.value;
}
@end

@interface GDInteger ()
@property(nonatomic, strong) NSNumber *value;
@end

@implementation GDInteger
- (id)initWithInt:(NSNumber *)intValue {
	self = [super init];
	if (self) {
		self.value = intValue;
	}
	return self;
}

- (int)getValue {
	return [self.value intValue];
}
@end

@interface GDFloat ()
@property(nonatomic, strong) NSNumber *value;
@end

@implementation GDFloat
- (id)initWithFloat:(NSNumber *)floatValue {
	self = [super init];
	if (self) {
		self.value = floatValue;
	}
	return self;
}

- (float)getValue {
	return [self.value floatValue];
}
@end

@interface GDDouble ()
@property(nonatomic, strong) NSNumber *value;
@end

@implementation GDDouble
- (id)initWithDouble:(NSNumber *)doubleValue {
	self = [super init];
	if (self) {
		self.value = doubleValue;
	}
	return self;
}

- (double)getValue {
	return [self.value doubleValue];
}
@end

@interface GDString ()
@property(nonatomic, strong) NSString *value;
@end

@implementation GDString
- (id)initWithString:(NSString *)stringValue {
	self = [super init];
	if (self) {
		self.value = stringValue;
	}
	return self;
}

- (NSString *)getValue {
	return self.value;
}
@end

@interface GDIntegerArray ()
@property(nonatomic, strong) NSArray<NSNumber *> *value;
@end

@implementation GDIntegerArray
- (id)initWithIntegerArray:(NSArray<NSNumber *> *)intArrayValue {
	self = [super init];
	if (self) {
		self.value = intArrayValue;
	}
	return self;
}

- (NSArray<NSNumber *> *)getValue {
	return self.value;
}
@end

@interface GDFloatArray ()
@property(nonatomic, strong) NSArray<NSNumber *> *value;
@end

@implementation GDFloatArray
- (id)initWithFloatArray:(NSArray<NSNumber *> *)floatArrayValue {
	self = [super init];
	if (self) {
		self.value = floatArrayValue;
	}
	return self;
}

- (NSArray<NSNumber *> *)getValue {
	return self.value;
}
@end

@interface GDDoubleArray ()
@property(nonatomic, strong) NSArray<NSNumber *> *value;
@end

@implementation GDDoubleArray
- (id)initWithDoubleArray:(NSArray<NSNumber *> *)doubleArrayValue {
	self = [super init];
	if (self) {
		self.value = doubleArrayValue;
	}
	return self;
}

- (NSArray<NSNumber *> *)getValue {
	return self.value;
}

@end

@interface GDByteArray ()
@property(nonatomic, strong) NSData *value;
@end

@implementation GDByteArray

- (id)initWithByteArray:(NSData *)byteArrayValue {
	self = [super init];
	if (self) {
		self.value = byteArrayValue;
	}
	return self;
}

- (NSData *)getValue {
	return self.value;
}
@end

@interface GDStringArray ()
@property(nonatomic, strong) NSArray<NSString *> *value;
@end

@implementation GDStringArray
- (id)initWithStringArray:(NSArray<NSString *> *)stringArrayValue {
	self = [super init];
	if (self) {
		self.value = stringArrayValue;
	}
	return self;
}

- (NSArray<NSString *> *)getValue {
	return self.value;
}
@end

@interface GDDictionary ()
@property(nonatomic, strong) NSDictionary *value;
@end

@implementation GDDictionary

- (id)initWithDictionary:(NSDictionary *)dictionaryValue {
	self = [super init];
	if (self) {
		self.value = dictionaryValue;
	}
	return self;
}

- (NSDictionary *)getValue {
	return self.value;
}
@end
