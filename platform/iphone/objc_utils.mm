/*************************************************************************/
/*  jni_utils.cpp                                                        */
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

#include "objc_utils.h"
#include "plugin/godot_vars.h"

#import <Foundation/Foundation.h>

id _variant_to_id(Variant::Type p_type, const Variant *p_arg, bool empty_obj) {

	id v;

	switch (p_type) {

		case Variant::BOOL: {
			if ((bool)(*p_arg)) {
				v = [NSNumber numberWithInteger:1];
			} else {
				v = [NSNumber numberWithInteger:0];
			}

		} break;
		case Variant::INT: {
			v = [NSNumber numberWithInteger:(int)(*p_arg)];

		} break;
		case Variant::REAL: {
			v = [NSNumber numberWithDouble:(double)(*p_arg)];

		} break;
		case Variant::STRING: {
			String s = *p_arg;
			v = [[[NSString alloc] initWithUTF8String:s.utf8().get_data()] autorelease];

		} break;
		case Variant::POOL_STRING_ARRAY: {
			PoolVector<String> sarray = *p_arg;
			NSMutableArray *string_array = [[NSMutableArray alloc] init];
			for (int j = 0; j < sarray.size(); j++) {
				[string_array addObject:[[[NSString alloc] initWithUTF8String:sarray[j].utf8().get_data()] autorelease]];
			}
			v = string_array;
		} break;
		case Variant::POOL_INT_ARRAY: {
			PoolVector<int> array = *p_arg;
			NSMutableArray *int_array = [[NSMutableArray alloc] init];
			for (int j = 0; j < array.size(); j++) {
				[int_array addObject:[NSNumber numberWithInt:array[j]]];
			}
			v = int_array;
		} break;
		case Variant::POOL_REAL_ARRAY: {
			PoolVector<float> array = *p_arg;
			NSMutableArray *float_array = [[NSMutableArray alloc] init];
			for (int j = 0; j < array.size(); j++) {
				[float_array addObject:[NSNumber numberWithFloat:array[j]]];
			}
			v = float_array;
		} break;
		case Variant::DICTIONARY: {
			NSMutableDictionary *dictionary = [NSMutableDictionary dictionary];
			Dictionary dict = *p_arg;
			Array keys = dict.keys();
			for (int j = 0; j < keys.size(); j++) {
				Variant var = dict[keys[j]];
				id val = _variant_to_id(var.get_type(), &var, true);
				[dictionary setObject:val forKey:[[[NSString alloc] initWithUTF8String:String(keys[j]).utf8().get_data()] autorelease]];
			}
			v = dictionary;
		} break;
		case Variant::POOL_BYTE_ARRAY: {
			PoolVector<uint8_t> array = *p_arg;
			PoolVector<uint8_t>::Read r = array.read();

			NSMutableData *byte_data = [NSMutableData dataWithLength:array.size()];
			const uint8_t *src_data_ptr = r.ptr();
			[byte_data appendBytes:src_data_ptr length:array.size()];

			v = byte_data;
		} break;
		default: {
			v = nil;
		} break;
	}
	return v;
}

Variant _id_to_variant(id p_objc_type, Variant::Type p_type) {
	switch (p_type) {

		case Variant::BOOL: {
			if ([((NSNumber *)p_objc_type) boolValue]) {
				return true;
			} else {
				return false;
			}
		} break;
		case Variant::INT: {
			return [((NSNumber *)p_objc_type) intValue];
		} break;
		case Variant::REAL: {
			return [((NSNumber *)p_objc_type) doubleValue];
		} break;
		case Variant::STRING: {
			return String([((NSString *)p_objc_type) UTF8String]);
		} break;
		case Variant::POOL_STRING_ARRAY: {
			PoolVector<String> sarray;
			for (NSString *element in ((NSArray *)p_objc_type)) {
				sarray.push_back(String([element UTF8String]));
			}
			return sarray;
		} break;
		case Variant::POOL_INT_ARRAY: {
			PoolVector<int> int_array;
			for (NSNumber *element in ((NSArray *)p_objc_type)) {
				int_array.push_back([element intValue]);
			}
			return int_array;
		} break;
		case Variant::POOL_REAL_ARRAY: {
			NSArray *objc_real_array = ((NSArray *)p_objc_type);
			PoolRealArray double_array;
			double_array.resize([objc_real_array count]);

			PoolRealArray::Write w = double_array.write();

			for (int i = 0; i < [objc_real_array count]; i++) {
				w.ptr()[i] = [objc_real_array[i] doubleValue];
			}

			return double_array;
		} break;
		case Variant::DICTIONARY: {
			Dictionary ret;
			NSArray *keys = [((NSDictionary *)p_objc_type) allKeys];
			for (int i = 0; i < [keys count]; i++) {
				ret[keys[i]] = [((NSDictionary *)p_objc_type) objectForKey:keys[i]];
			}
			return ret;

		} break;
		case Variant::POOL_BYTE_ARRAY: {
			NSData *byte_data = ((NSData *)p_objc_type);
			PoolVector<uint8_t> sarr;
			sarr.resize([byte_data length]);

			PoolVector<uint8_t>::Write w = sarr.write();
			[byte_data getBytes:w.ptr() length:[byte_data length]];

			return sarr;
		} break;
		default: {
			return Variant::NIL;
		} break;
	}
}

id _create_objc_object(Variant::Type p_type) {

	id v;

	switch (p_type) {

		case Variant::BOOL: {
			v = [NSNumber numberWithInteger:0];

		} break;
		case Variant::INT: {
			NSNumber *ret = [NSNumber numberWithInteger:0];
			v = ret;

		} break;
		case Variant::REAL: {
			NSNumber *ret = [NSNumber numberWithDouble:0.0];
			v = ret;

		} break;
		case Variant::STRING: {
			NSString *ret = [[NSString alloc] init];
			v = ret;
		} break;
		case Variant::POOL_STRING_ARRAY: {
			NSMutableArray *ret = [[NSMutableArray alloc] init];
			v = ret;

		} break;
		case Variant::POOL_INT_ARRAY: {
			NSMutableArray *ret = [[NSMutableArray alloc] init];
			v = ret;

		} break;
		case Variant::POOL_REAL_ARRAY: {
			NSMutableArray *ret = [[NSMutableArray alloc] init];
			v = ret;

		} break;
		case Variant::DICTIONARY: {
			NSMutableDictionary *ret = [NSMutableDictionary dictionary];
			v = ret;

		} break;
		case Variant::POOL_BYTE_ARRAY: {
			NSMutableData *ret = [NSMutableData new];
			v = ret;

		} break;

		default: {
			v = nil;
		} break;
	}
	return v;
}

Variant::Type get_objc_type(const String &p_type) {

	static struct {
		const char *name;
		Variant::Type type;
	} _type_to_vtype[] = {
		{ "void", Variant::NIL },
		{ "bool", Variant::BOOL },
		{ "int", Variant::INT },
		{ "float", Variant::REAL },
		{ "double", Variant::REAL },
		{ "NSString", Variant::STRING },
		{ "[int]", Variant::POOL_INT_ARRAY },
		{ "[byte]", Variant::POOL_BYTE_ARRAY },
		{ "[float]", Variant::POOL_REAL_ARRAY },
		{ "[double]", Variant::POOL_REAL_ARRAY },
		{ "[NSString]", Variant::POOL_STRING_ARRAY },
		{ "NSDictionary", Variant::DICTIONARY },
		{ NULL, Variant::NIL }
	};

	int idx = 0;

	while (_type_to_vtype[idx].name) {

		if (p_type == _type_to_vtype[idx].name)
			return _type_to_vtype[idx].type;

		idx++;
	}

	return Variant::NIL;
}

Variant get_objc_class_type(const String &p_name, id p_objc_type) {
	if (p_objc_type == nil) {
		return Variant();
	}

	if (p_name == "GDBool") {
		GDBool *gbObj = (GDBool *)p_objc_type;
		return [gbObj getValue] ? true : false;
	};

	if (p_name == "GDInteger") {
		GDInteger *gbObj = (GDInteger *)p_objc_type;
		return [gbObj getValue];
	}

	if (p_name == "GDFloat") {
		GDFloat *gbObj = (GDFloat *)p_objc_type;
		return [gbObj getValue];
	}

	if (p_name == "GDDouble") {
		GDDouble *gbObj = (GDDouble *)p_objc_type;
		return [gbObj getValue];
	}

	if (p_name == "GDString") {
		GDString *gbObj = (GDString *)p_objc_type;
		return String([[gbObj getValue] UTF8String]);
	}

	if (p_name == "NSString" || p_name == "__NSCFConstantString") {
		return String([((NSString *)p_objc_type) UTF8String]);
	};

	if (p_name == "GDIntegerArray") {
		GDIntegerArray *gbObj = (GDIntegerArray *)p_objc_type;
		PoolVector<int> int_array;
		for (NSNumber *element in [gbObj getValue]) {
			int_array.push_back([element intValue]);
		}
		return int_array;
	}

	if (p_name == "GDFloatArray") {
		GDFloatArray *gbObj = (GDFloatArray *)p_objc_type;
		PoolRealArray double_array;
		double_array.resize([[gbObj getValue] count]);

		PoolRealArray::Write w = double_array.write();

		for (int i = 0; i < [[gbObj getValue] count]; i++) {
			w.ptr()[i] = [[gbObj getValue][i] floatValue];
		}

		return double_array;
	}

	if (p_name == "GDDoubleArray") {
		GDDoubleArray *gbObj = (GDDoubleArray *)p_objc_type;
		PoolRealArray double_array;
		double_array.resize([[gbObj getValue] count]);

		PoolRealArray::Write w = double_array.write();

		for (int i = 0; i < [[gbObj getValue] count]; i++) {
			w.ptr()[i] = [[gbObj getValue][i] doubleValue];
		}

		return double_array;
	}

	if (p_name == "GDByteArray") {
		GDByteArray *gbObj = (GDByteArray *)p_objc_type;
		NSData *byte_data = [gbObj getValue];
		PoolVector<uint8_t> sarr;
		sarr.resize([[gbObj getValue] length]);

		PoolVector<uint8_t>::Write w = sarr.write();
		[byte_data getBytes:w.ptr() length:[[gbObj getValue] length]];

		return sarr;
	}

	if (p_name == "GDStringArray") {
		GDStringArray *gbObj = (GDStringArray *)p_objc_type;
		PoolVector<String> sarray;
		for (NSString *element in ((NSArray *)[gbObj getValue])) {
			sarray.push_back(String([element UTF8String]));
		}
		return sarray;
	}

	if (p_name == "GDDictionary") {
		GDDictionary *gbObj = (GDDictionary *)p_objc_type;
		Dictionary ret;
		NSArray *keys = [[gbObj getValue] allKeys];
		for (int i = 0; i < [keys count]; i++) {
			Variant v = get_objc_class_type(String(class_getName([[[gbObj getValue] objectForKey:keys[i]] class])), [[gbObj getValue] objectForKey:keys[i]]);
			ret[String([keys[i] UTF8String])] = v;
		}
		return ret;
	}

	return Variant();
}

Variant::Type get_objc_to_variant(const String &p_type) {

	static struct {
		const char *name;
		Variant::Type type;
	} _type_to_vtype[] = {
		{ "void", Variant::NIL },
		{ "GDBOOL", Variant::BOOL },
		{ "GDInt", Variant::INT },
		{ "GDFloat", Variant::REAL },
		{ "GDDouble", Variant::REAL },
		{ "NSString", Variant::STRING },
		{ "GDIntArray", Variant::POOL_INT_ARRAY },
		{ "GBByteArray", Variant::POOL_BYTE_ARRAY },
		{ "GDFloatArray", Variant::POOL_REAL_ARRAY },
		{ "GDStringArray", Variant::POOL_STRING_ARRAY },
		{ "NSDictionary", Variant::DICTIONARY },
		{ NULL, Variant::NIL }
	};

	int idx = 0;

	while (_type_to_vtype[idx].name) {

		if (p_type == _type_to_vtype[idx].name)
			return _type_to_vtype[idx].type;

		idx++;
	}

	return Variant::NIL;
}