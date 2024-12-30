// Copyright 2020 Google LLC
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google LLC nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#import "HTTPRequest.h"

#include <Availability.h>
#include <AvailabilityMacros.h>

#if (defined(__IPHONE_OS_VERSION_MIN_REQUIRED) && defined(__IPHONE_7_0) && \
     __IPHONE_OS_VERSION_MIN_REQUIRED >= __IPHONE_7_0)
#import <UIKit/UIKit.h>
#define HAS_BACKGROUND_TASK_API 1
#else
#define HAS_BACKGROUND_TASK_API 0
#endif

#import "encoding_util.h"

#if (defined(__IPHONE_OS_VERSION_MIN_REQUIRED) && defined(__IPHONE_7_0) && \
     __IPHONE_OS_VERSION_MIN_REQUIRED >= __IPHONE_7_0) ||                  \
    (defined(MAC_OS_X_VERSION_MIN_REQUIRED) &&                             \
     defined(MAC_OS_X_VERSION_10_11) &&                                    \
     MAC_OS_X_VERSION_MIN_REQUIRED >= MAC_OS_X_VERSION_10_11)
#define USE_NSURLSESSION 1
#else
#define USE_NSURLSESSION 0
#endif

// As -[NSURLConnection sendSynchronousRequest:returningResponse:error:] has
// been deprecated with iOS 9.0 / OS X 10.11 SDKs, this function re-implements
// it using -[NSURLSession dataTaskWithRequest:completionHandler:] which is
// available on iOS 7+.
static NSData* SendSynchronousNSURLRequest(NSURLRequest* req,
                                           NSURLResponse** outResponse,
                                           NSError** outError) {
#if USE_NSURLSESSION
  __block NSData* result = nil;
  __block NSError* error = nil;
  __block NSURLResponse* response = nil;
  dispatch_semaphore_t waitSemaphone = dispatch_semaphore_create(0);

  NSURLSessionConfiguration* config =
      [NSURLSessionConfiguration defaultSessionConfiguration];
  [config setTimeoutIntervalForRequest:240.0];
  NSURLSession* session = [NSURLSession sessionWithConfiguration:config];
  NSURLSessionDataTask *task = [session
      dataTaskWithRequest:req
        completionHandler:^(NSData* data, NSURLResponse* resp, NSError* err) {
          if (outError)
            error = [err retain];
          if (outResponse)
            response = [resp retain];
          if (err == nil)
            result = [data retain];
          dispatch_semaphore_signal(waitSemaphone);
        }];
  [task resume];

#if HAS_BACKGROUND_TASK_API
  // Used to guard against ending the background task twice, which UIKit
  // considers to be an error.
  __block BOOL isBackgroundTaskActive = YES;
  __block UIBackgroundTaskIdentifier backgroundTaskIdentifier =
      UIBackgroundTaskInvalid;
  backgroundTaskIdentifier = [UIApplication.sharedApplication
      beginBackgroundTaskWithName:@"Breakpad Upload"
                expirationHandler:^{
                  if (!isBackgroundTaskActive) {
                    return;
                  }
                  isBackgroundTaskActive = NO;

                  [task cancel];
                  [UIApplication.sharedApplication
                      endBackgroundTask:backgroundTaskIdentifier];
                }];
#endif  // HAS_BACKGROUND_TASK_API

  dispatch_semaphore_wait(waitSemaphone, DISPATCH_TIME_FOREVER);
  dispatch_release(waitSemaphone);

#if HAS_BACKGROUND_TASK_API
  if (backgroundTaskIdentifier != UIBackgroundTaskInvalid) {
    // Dispatch to main queue in order to synchronize access to
    // `isBackgroundTaskActive` with the background task expiration handler,
    // which is always run on the main thread.
    dispatch_async(dispatch_get_main_queue(), ^{
      if (!isBackgroundTaskActive) {
        return;
      }
      isBackgroundTaskActive = NO;

      [UIApplication.sharedApplication
          endBackgroundTask:backgroundTaskIdentifier];
    });
  }
#endif  // HAS_BACKGROUND_TASK_API

  if (outError)
    *outError = [error autorelease];
  if (outResponse)
    *outResponse = [response autorelease];
  return [result autorelease];
#else  // USE_NSURLSESSION
  return [NSURLConnection sendSynchronousRequest:req
                               returningResponse:outResponse
                                           error:outError];
#endif  // USE_NSURLSESSION
}

@implementation HTTPRequest

//=============================================================================
- (id)initWithURL:(NSURL*)URL {
  if ((self = [super init])) {
    URL_ = [URL copy];
  }

  return self;
}

//=============================================================================
- (void)dealloc {
  [URL_ release];
  [response_ release];

  [super dealloc];
}

//=============================================================================
- (NSURL*)URL {
  return URL_;
}

//=============================================================================
- (NSHTTPURLResponse*)response {
  return response_;
}

//=============================================================================
- (NSString*)HTTPMethod {
  @throw [NSException
      exceptionWithName:NSInternalInconsistencyException
                 reason:[NSString stringWithFormat:@"You must"
                                                    "override %@ in a subclass",
                                                   NSStringFromSelector(_cmd)]
               userInfo:nil];
}

//=============================================================================
- (NSString*)contentType {
  return nil;
}

//=============================================================================
- (NSData*)bodyData {
  return nil;
}

//=============================================================================
- (NSData*)send:(NSError**)withError {
  NSMutableURLRequest* req = [[NSMutableURLRequest alloc]
          initWithURL:URL_
          cachePolicy:NSURLRequestUseProtocolCachePolicy
      timeoutInterval:60.0];

  NSString* contentType = [self contentType];
  if ([contentType length] > 0) {
    [req setValue:contentType forHTTPHeaderField:@"Content-type"];
  }

  NSData* bodyData = [self bodyData];
  if ([bodyData length] > 0) {
    [req setHTTPBody:bodyData];
  }

  [req setHTTPMethod:[self HTTPMethod]];

  [response_ release];
  response_ = nil;

  NSData* data = nil;
  if ([[req URL] isFileURL]) {
    [[req HTTPBody] writeToURL:[req URL] options:0 error:withError];
  } else {
    NSURLResponse* response = nil;
    data = SendSynchronousNSURLRequest(req, &response, withError);
    response_ = (NSHTTPURLResponse*)[response retain];
  }
  [req release];

  return data;
}

//=============================================================================
+ (NSData*)formDataForFileContents:(NSData*)contents withName:(NSString*)name {
  NSMutableData* data = [NSMutableData data];
  NSString* escaped = PercentEncodeNSString(name);
  NSString* fmt = @"Content-Disposition: form-data; name=\"%@\"; "
                   "filename=\"minidump.dmp\"\r\nContent-Type: "
                   "application/octet-stream\r\n\r\n";
  NSString* pre = [NSString stringWithFormat:fmt, escaped];

  [data appendData:[pre dataUsingEncoding:NSUTF8StringEncoding]];
  [data appendData:contents];

  return data;
}

//=============================================================================
+ (NSData*)formDataForFile:(NSString*)file withName:(NSString*)name {
  NSData* contents = [NSData dataWithContentsOfFile:file];

  return [HTTPRequest formDataForFileContents:contents withName:name];
}

//=============================================================================
+ (NSData*)formDataForKey:(NSString*)key value:(NSString*)value {
  NSString* escaped = PercentEncodeNSString(key);
  NSString* fmt = @"Content-Disposition: form-data; name=\"%@\"\r\n\r\n%@\r\n";
  NSString* form = [NSString stringWithFormat:fmt, escaped, value];

  return [form dataUsingEncoding:NSUTF8StringEncoding];
}

//=============================================================================
+ (void)appendFileToBodyData:(NSMutableData*)data
                    withName:(NSString*)name
              withFileOrData:(id)fileOrData {
  NSData* fileData;

  // The object can be either the path to a file (NSString) or the contents
  // of the file (NSData).
  if ([fileOrData isKindOfClass:[NSData class]])
    fileData = [self formDataForFileContents:fileOrData withName:name];
  else
    fileData = [HTTPRequest formDataForFile:fileOrData withName:name];

  [data appendData:fileData];
}

@end
