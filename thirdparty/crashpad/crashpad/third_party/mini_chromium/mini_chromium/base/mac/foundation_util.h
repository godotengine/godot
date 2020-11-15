// Copyright 2008 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef MINI_CHROMIUM_BASE_MAC_FOUNDATION_UTIL_H_
#define MINI_CHROMIUM_BASE_MAC_FOUNDATION_UTIL_H_

#include <ApplicationServices/ApplicationServices.h>

#include "base/logging.h"

#if defined(__OBJC__)
#import <Foundation/Foundation.h>
#else  // defined(__OBJC__)
#include <CoreFoundation/CoreFoundation.h>
#endif  // defined(__OBJC__)

#if !defined(__OBJC__)
#define OBJC_CPP_CLASS_DECL(x) class x;
#else  // defined(__OBJC__)
#define OBJC_CPP_CLASS_DECL(x)
#endif  // defined(__OBJC__)

// Convert toll-free bridged CFTypes to NSTypes and vice-versa. This does not
// autorelease |cf_val|. This is useful for the case where there is a CFType in
// a call that expects an NSType and the compiler is complaining about const
// casting problems.
// The calls are used like this:
// NSString *foo = CFToNSCast(CFSTR("Hello"));
// CFStringRef foo2 = NSToCFCast(@"Hello");
// The macro magic below is to enforce safe casting. It could possibly have
// been done using template function specialization, but template function
// specialization doesn't always work intuitively,
// (http://www.gotw.ca/publications/mill17.htm) so the trusty combination
// of macros and function overloading is used instead.

#define CF_TO_NS_CAST_DECL(TypeCF, TypeNS) \
OBJC_CPP_CLASS_DECL(TypeNS) \
\
namespace base { \
namespace mac { \
TypeNS* CFToNSCast(TypeCF##Ref cf_val); \
TypeCF##Ref NSToCFCast(TypeNS* ns_val); \
} \
}
#define CF_TO_NS_MUTABLE_CAST_DECL(name) \
CF_TO_NS_CAST_DECL(CF##name, NS##name) \
OBJC_CPP_CLASS_DECL(NSMutable##name) \
\
namespace base { \
namespace mac { \
NSMutable##name* CFToNSCast(CFMutable##name##Ref cf_val); \
CFMutable##name##Ref NSToCFCast(NSMutable##name* ns_val); \
} \
}

// List of toll-free bridged types taken from:
// http://www.cocoadev.com/index.pl?TollFreeBridged

CF_TO_NS_MUTABLE_CAST_DECL(Array);
CF_TO_NS_MUTABLE_CAST_DECL(AttributedString);
CF_TO_NS_CAST_DECL(CFCalendar, NSCalendar);
CF_TO_NS_MUTABLE_CAST_DECL(CharacterSet);
CF_TO_NS_MUTABLE_CAST_DECL(Data);
CF_TO_NS_CAST_DECL(CFDate, NSDate);
CF_TO_NS_MUTABLE_CAST_DECL(Dictionary);
CF_TO_NS_CAST_DECL(CFError, NSError);
CF_TO_NS_CAST_DECL(CFLocale, NSLocale);
CF_TO_NS_CAST_DECL(CFNumber, NSNumber);
CF_TO_NS_CAST_DECL(CFRunLoopTimer, NSTimer);
CF_TO_NS_CAST_DECL(CFTimeZone, NSTimeZone);
CF_TO_NS_MUTABLE_CAST_DECL(Set);
CF_TO_NS_CAST_DECL(CFReadStream, NSInputStream);
CF_TO_NS_CAST_DECL(CFWriteStream, NSOutputStream);
CF_TO_NS_MUTABLE_CAST_DECL(String);
CF_TO_NS_CAST_DECL(CFURL, NSURL);

#undef CF_TO_NS_CAST_DECL
#undef CF_TO_NS_MUTABLE_CAST_DECL
#undef OBJC_CPP_CLASS_DECL

namespace base {
namespace mac {

// CFCast<>() and CFCastStrict<>() cast a basic CFTypeRef to a more
// specific CoreFoundation type. The compatibility of the passed
// object is found by comparing its opaque type against the
// requested type identifier. If the supplied object is not
// compatible with the requested return type, CFCast<>() returns
// NULL and CFCastStrict<>() will DCHECK. Providing a NULL pointer
// to either variant results in NULL being returned without
// triggering any DCHECK.
//
// Example usage:
// CFNumberRef some_number = base::mac::CFCast<CFNumberRef>(
//     CFArrayGetValueAtIndex(array, index));
//
// CFTypeRef hello = CFSTR("hello world");
// CFStringRef some_string = base::mac::CFCastStrict<CFStringRef>(hello);

template<typename T>
T CFCast(const CFTypeRef& cf_val);

template<typename T>
T CFCastStrict(const CFTypeRef& cf_val);

#define CF_CAST_DECL(TypeCF) \
template<> TypeCF##Ref \
CFCast<TypeCF##Ref>(const CFTypeRef& cf_val);\
\
template<> TypeCF##Ref \
CFCastStrict<TypeCF##Ref>(const CFTypeRef& cf_val);

CF_CAST_DECL(CFArray);
CF_CAST_DECL(CFBag);
CF_CAST_DECL(CFBoolean);
CF_CAST_DECL(CFData);
CF_CAST_DECL(CFDate);
CF_CAST_DECL(CFDictionary);
CF_CAST_DECL(CFNull);
CF_CAST_DECL(CFNumber);
CF_CAST_DECL(CFSet);
CF_CAST_DECL(CFString);
CF_CAST_DECL(CFURL);
CF_CAST_DECL(CFUUID);

CF_CAST_DECL(CGColor);

CF_CAST_DECL(CTFont);
CF_CAST_DECL(CTRun);

CF_CAST_DECL(SecACL);
CF_CAST_DECL(SecTrustedApplication);

#undef CF_CAST_DECL

#if defined(__OBJC__)

// ObjCCast<>() and ObjCCastStrict<>() cast a basic id to a more
// specific (NSObject-derived) type. The compatibility of the passed
// object is found by checking if it's a kind of the requested type
// identifier. If the supplied object is not compatible with the
// requested return type, ObjCCast<>() returns nil and
// ObjCCastStrict<>() will DCHECK. Providing a nil pointer to either
// variant results in nil being returned without triggering any DCHECK.
//
// The strict variant is useful when retrieving a value from a
// collection which only has values of a specific type, e.g. an
// NSArray of NSStrings. The non-strict variant is useful when
// retrieving values from data that you can't fully control. For
// example, a plist read from disk may be beyond your exclusive
// control, so you'd only want to check that the values you retrieve
// from it are of the expected types, but not crash if they're not.
//
// Example usage:
// NSString* version = base::mac::ObjCCast<NSString>(
//     [bundle objectForInfoDictionaryKey:@"CFBundleShortVersionString"]);
//
// NSString* str = base::mac::ObjCCastStrict<NSString>(
//     [ns_arr_of_ns_strs objectAtIndex:0]);
template<typename T>
T* ObjCCast(id objc_val) {
  if ([objc_val isKindOfClass:[T class]]) {
    return reinterpret_cast<T*>(objc_val);
  }
  return nil;
}

template<typename T>
T* ObjCCastStrict(id objc_val) {
  T* rv = ObjCCast<T>(objc_val);
  DCHECK(objc_val == nil || rv);
  return rv;
}

#endif  // defined(__OBJC__)

}  // namespace mac
}  // namespace base

#endif  // MINI_CHROMIUM_BASE_MAC_FOUNDATION_UTIL_H_
