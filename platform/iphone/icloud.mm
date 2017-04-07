/*************************************************************************/
/*  icloud.mm                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifdef ICLOUD_ENABLED

#include "icloud.h"

#ifndef __IPHONE_9_0
extern "C" {
#endif

#import <Foundation/Foundation.h>
#import "app_delegate.h"

#ifndef __IPHONE_9_0
};
#endif

ICloud* ICloud::instance = NULL;

void ICloud::_bind_methods() {
	ClassDB::bind_method(D_METHOD("remove_key"),&ICloud::remove_key);
	ClassDB::bind_method(D_METHOD("set_key_values"),&ICloud::set_key_values);
	ClassDB::bind_method(D_METHOD("get_key_value"),&ICloud::get_key_value);
	ClassDB::bind_method(D_METHOD("synchronize_key_values"),&ICloud::synchronize_key_values);
	ClassDB::bind_method(D_METHOD("get_all_key_values"),&ICloud::get_all_key_values);

	ClassDB::bind_method(D_METHOD("get_pending_event_count"),&ICloud::get_pending_event_count);
	ClassDB::bind_method(D_METHOD("pop_pending_event"),&ICloud::pop_pending_event);
};

int ICloud::get_pending_event_count() {

	return pending_events.size();
};

Variant ICloud::pop_pending_event() {

	Variant front = pending_events.front()->get();
	pending_events.pop_front();

	return front;
};

ICloud* ICloud::get_singleton() {
	return instance;
};

//convert from apple's abstract type to godot's abstract type....
Variant nsobject_to_variant(NSObject* object) {
	if ([object isKindOfClass:[NSString class]]) {
		const char* str = [(NSString*)object UTF8String];
		return String::utf8(str != NULL ? str : "");
	}
	else if ([object isKindOfClass:[NSData class]]) {
		PoolByteArray ret;
		NSData* data = (NSData*)object;
		if ([data length] > 0) {
			ret.resize([data length]);
			{
				PoolByteArray::Write w = ret.write();
				copymem(w.ptr(), [data bytes], [data length]);
			}
		}
		return ret;
	}
	else if ([object isKindOfClass:[NSArray class]]) {
		Array result;
		NSArray* array = (NSArray*)object;
		for (unsigned int i = 0; i < [array count]; ++i) {
			NSObject* value = [array objectAtIndex:i];
			result.push_back(nsobject_to_variant(value));
		}
		return result;
	}
	else if ([object isKindOfClass:[NSDictionary class]]) {
		Dictionary result;
		NSDictionary* dic = (NSDictionary*)object;


		NSArray* keys =  [dic allKeys];
		int count = [keys count];
		for (int i=0; i < count; ++i) {
			NSObject* k = [ keys objectAtIndex:i];
			NSObject* v = [dic objectForKey:k];

			result[nsobject_to_variant(k)] = nsobject_to_variant(v);
		}
		return result;
	}
	else if ([object isKindOfClass:[NSNumber class]]) {
		//Every type except numbers can reliably identify its type.  The following is comparing to the *internal* representation, which isn't guaranteed to match the type that was used to create it, and is not advised, particularly when dealing with potential platform differences (ie, 32/64 bit)
		//To avoid errors, we'll cast as broadly as possible, and only return int or float.
		//bool, char, int, uint, longlong -> int
		//float, double -> float
		NSNumber* num = (NSNumber*)object;
		if(strcmp([num objCType], @encode(BOOL)) == 0) {
			return Variant((int)[num boolValue]);
		}
		else if(strcmp([num objCType], @encode(char)) == 0) {
			return Variant((int)[num charValue]);
		}
		else if(strcmp([num objCType], @encode(int)) == 0) {
			return Variant([num intValue]);
		}
		else if(strcmp([num objCType], @encode(unsigned int)) == 0) {
			return Variant((int)[num unsignedIntValue]);
		}
		else if(strcmp([num objCType], @encode(long long)) == 0) {
			return Variant((int)[num longValue]);
		}
		else if(strcmp([num objCType], @encode(float)) == 0) {
			return Variant([num floatValue]);
		}
		else if(strcmp([num objCType], @encode(double)) == 0) {
			return Variant((float)[num doubleValue]);
		}
	}
	else if ([object isKindOfClass:[NSDate class]]) {
		//this is a type that icloud supports...but how did you submit it in the first place?
		//I guess this is a type that *might* show up, if you were, say, trying to make your game
		//compatible with existing cloud data written by another engine's version of your game
		WARN_PRINT("NSDate unsupported, returning null Variant")
		return Variant();
	}
	else if ([object isKindOfClass:[NSNull class]] or object == nil) {
		return Variant();
	}
	else {
		WARN_PRINT("Trying to convert unknown NSObject type to Variant");
		return Variant();
	}
}

NSObject* variant_to_nsobject(Variant v) {
	if (v.get_type() == Variant::STRING) {
		return [[[NSString alloc] initWithUTF8String:((String)v).utf8().get_data()] autorelease];
	}
	else if (v.get_type() == Variant::REAL) {
		return [NSNumber numberWithDouble:(double)v];
	}
	else if (v.get_type() == Variant::INT) {
		return [NSNumber numberWithLongLong:(long)(int)v];
	}
	else if (v.get_type() == Variant::BOOL) {
		return [NSNumber numberWithBool:BOOL((bool)v)];
	}
	else if (v.get_type() == Variant::DICTIONARY) {
		NSMutableDictionary* result = [[[NSMutableDictionary alloc] init] autorelease];
		Dictionary dic = v;
		Array keys = dic.keys();
		for (unsigned int i = 0; i < keys.size(); ++i) {
			NSString* key = [[[NSString alloc] initWithUTF8String:((String)(keys[i])).utf8().get_data()] autorelease];
			NSObject* value = variant_to_nsobject(dic[keys[i]]);

			if (key == NULL || value == NULL) {
				return NULL;
			}

			[result setObject:value forKey:key];
		}
		return result;
	}
	else if (v.get_type() == Variant::ARRAY) {
		NSMutableArray* result = [[[NSMutableArray alloc] init] autorelease];
		Array arr = v;
		for (unsigned int i = 0; i < arr.size(); ++i) {
			NSObject* value = variant_to_nsobject(arr[i]);
			if (value == NULL) {
				//trying to add something unsupported to the array. cancel the whole array
				return NULL;
			}
			[result addObject:value];
		}
		return result;
	}
	else if (v.get_type() == Variant::POOL_BYTE_ARRAY) {
		PoolByteArray arr = v;
		PoolByteArray::Read r = arr.read();
		NSData* result = [NSData dataWithBytes:r.ptr() length:arr.size()];
		return result;
	}
	WARN_PRINT(String("Could not add unsupported type to iCloud: '" + Variant::get_type_name(v.get_type())+"'").utf8().get_data());
	return NULL;
}


Error ICloud::remove_key(Variant p_param) {
	String param = p_param;
	NSString* key = [[[NSString alloc] initWithUTF8String:param.utf8().get_data()] autorelease];

	NSUbiquitousKeyValueStore *store = [NSUbiquitousKeyValueStore defaultStore];

	if (![[store dictionaryRepresentation] objectForKey:key]) {
		return ERR_INVALID_PARAMETER;
	}

	[store removeObjectForKey:key];
	return OK;
}

//return an array of the keys that could not be set
Variant ICloud::set_key_values(Variant p_params) {
	Dictionary params = p_params;
	Array keys = params.keys();

	Array error_keys;

	for (unsigned int i = 0; i < keys.size(); ++i) {
		String variant_key = keys[i];
		Variant variant_value = params[variant_key];

		NSString* key = [[[NSString alloc] initWithUTF8String:variant_key.utf8().get_data()] autorelease];
		if (key == NULL) {
			error_keys.push_back(variant_key);
			continue;
		}

		NSObject* value = variant_to_nsobject(variant_value);

		if (value == NULL) {
			error_keys.push_back(variant_key);
			continue;
		}

		NSUbiquitousKeyValueStore *store = [NSUbiquitousKeyValueStore defaultStore];
		[store setObject:value forKey:key];
	}

	return error_keys;
}

Variant ICloud::get_key_value(Variant p_param) {
	String param = p_param;

	NSString* key = [[[NSString alloc] initWithUTF8String:param.utf8().get_data()] autorelease];
	NSUbiquitousKeyValueStore *store = [NSUbiquitousKeyValueStore defaultStore];

	if (![[store dictionaryRepresentation] objectForKey:key]) {
		return Variant();
	}

	Variant result = nsobject_to_variant([[store dictionaryRepresentation] objectForKey:key]);

	return result;
}

Variant ICloud::get_all_key_values() {
	Dictionary result;

	NSUbiquitousKeyValueStore* store = [NSUbiquitousKeyValueStore defaultStore];
	NSDictionary* store_dictionary = [store dictionaryRepresentation];

	NSArray* keys =  [store_dictionary allKeys];
	int count = [keys count];
	for (int i=0; i < count; ++i) {
		NSString* k = [ keys objectAtIndex:i];
		NSObject* v = [store_dictionary objectForKey:k];

		const char* str = [k UTF8String];
		if (str != NULL) {
			result[String::utf8(str)] = nsobject_to_variant(v);
		}
	}

	return result;
}

Error ICloud::synchronize_key_values() {
	NSUbiquitousKeyValueStore *store = [NSUbiquitousKeyValueStore defaultStore];
	BOOL result = [store synchronize];
	if (result == YES) {
		return OK;
	}
	else {
		return FAILED;
	}
}
/*
Error ICloud::initial_sync() {
	//you sometimes have to write something to the store to get it to download new data.  go apple!
	NSUbiquitousKeyValueStore *store = [NSUbiquitousKeyValueStore defaultStore];
	if ([store boolForKey:@"isb"])
    {
        [store setBool:NO forKey:@"isb"];
    }
    else
    {
        [store setBool:YES forKey:@"isb"];
    }
    return synchronize();
}
*/
ICloud::ICloud() {
	ERR_FAIL_COND(instance != NULL);
	instance = this;
	//connected = false;

	[
		//[NSNotificationCenter defaultCenter] addObserverForName: @"notify"
		[NSNotificationCenter defaultCenter] addObserverForName: NSUbiquitousKeyValueStoreDidChangeExternallyNotification
		object: [NSUbiquitousKeyValueStore defaultStore]
		queue: nil
		usingBlock: ^ (NSNotification * notification) {
			NSDictionary* userInfo = [notification userInfo];
			NSInteger change = [[userInfo objectForKey:NSUbiquitousKeyValueStoreChangeReasonKey] integerValue];

			Dictionary ret;
			ret["type"] = "key_value_changed";

			//PoolStringArray result_keys;
			//Array result_values;
			Dictionary keyValues;
			String reason = "";

			if (change == NSUbiquitousKeyValueStoreServerChange) {
				reason = "server";
			}
			else if (change == NSUbiquitousKeyValueStoreInitialSyncChange) {
				reason = "initial_sync";
			}
			else if (change == NSUbiquitousKeyValueStoreQuotaViolationChange) {
				reason = "quota_violation";
			}
			else if (change == NSUbiquitousKeyValueStoreAccountChange) {
				reason = "account";
			}

			ret["reason"] = reason;


			NSUbiquitousKeyValueStore *store = [NSUbiquitousKeyValueStore defaultStore];

			NSArray * keys = [userInfo objectForKey:NSUbiquitousKeyValueStoreChangedKeysKey];
			for (NSString* key in keys) {
				const char* str = [key UTF8String];
				if (str == NULL) {
					continue;
				}

				NSObject* object = [store objectForKey:key];

				//figure out what kind of object it is
				Variant value = nsobject_to_variant(object);

				keyValues[String::utf8(str)] = value;
			}

			ret["changed_values"] = keyValues;
			pending_events.push_back(ret);
		}
	];
}


ICloud::~ICloud() {

};

#endif
