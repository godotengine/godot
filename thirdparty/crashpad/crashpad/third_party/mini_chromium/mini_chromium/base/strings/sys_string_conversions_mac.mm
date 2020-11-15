// Copyright 2008 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "base/strings/sys_string_conversions.h"

#include "base/logging.h"
#include "base/mac/foundation_util.h"

namespace base {

namespace {

const CFStringEncoding kNarrowStringEncoding = kCFStringEncodingUTF8;

template<typename StringType>
StringType CFStringToSTLStringWithEncodingT(CFStringRef cfstring,
                                            CFStringEncoding encoding) {
  CFIndex length = CFStringGetLength(cfstring);
  if (length == 0) {
    return StringType();
  }

  CFRange whole_string = CFRangeMake(0, length);
  CFIndex out_size;
  CFIndex converted = CFStringGetBytes(cfstring,
                                       whole_string,
                                       encoding,
                                       0,
                                       FALSE,
                                       NULL,
                                       0,
                                       &out_size);
  if (converted == 0 || out_size == 0) {
    return StringType();
  }

  DCHECK_EQ(out_size % sizeof(typename StringType::value_type), 0u);
  typename StringType::size_type elements =
      out_size * sizeof(UInt8) / sizeof(typename StringType::value_type);
  StringType out(elements, typename StringType::value_type());

  converted = CFStringGetBytes(cfstring,
                               whole_string,
                               encoding,
                               0,
                               FALSE,
                               reinterpret_cast<UInt8*>(&out[0]),
                               out_size,
                               NULL);
  if (converted == 0) {
    return StringType();
  }

  return out;
}

template<typename StringType>
static CFStringRef STLStringToCFStringWithEncodingsT(
    const StringType& in,
    CFStringEncoding in_encoding) {
  typename StringType::size_type in_length = in.length();
  if (in_length == 0) {
    return CFSTR("");
  }

  return CFStringCreateWithBytes(kCFAllocatorDefault,
                                 reinterpret_cast<const UInt8*>(in.c_str()),
                                 in_length *
                                     sizeof(typename StringType::value_type),
                                 in_encoding,
                                 FALSE);
}

}  // namespace

std::string SysCFStringRefToUTF8(CFStringRef ref) {
  return CFStringToSTLStringWithEncodingT<std::string>(ref,
                                                       kNarrowStringEncoding);
}

std::string SysNSStringToUTF8(NSString *nsstring) {
  if (!nsstring) {
    return std::string();
  }
  return SysCFStringRefToUTF8(base::mac::NSToCFCast(nsstring));
}

CFStringRef SysUTF8ToCFStringRef(const std::string& utf8) {
  return STLStringToCFStringWithEncodingsT(utf8, kNarrowStringEncoding);
}

NSString* SysUTF8ToNSString(const std::string& utf8) {
  return [base::mac::CFToNSCast(SysUTF8ToCFStringRef(utf8)) autorelease];
}

}  // namespace base
