// Copyright (c) 2020, Google Inc.
// All rights reserved.
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
//     * Neither the name of Google Inc. nor the names of its
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

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN
/**
 Represents a single HTTP request. Sending the request is synchronous.
 Once the send is complete, the response will be set.

 This is a base interface that specific HTTP requests derive from.
 It is not intended to be instantiated directly.
 */
@interface HTTPRequest : NSObject {
 @protected
  NSURL* URL_;                   // The destination URL (STRONG)
  NSHTTPURLResponse* response_;  // The response from the send (STRONG)
}

/**
 Initializes the HTTPRequest and sets its URL.
 */
- (id)initWithURL:(NSURL*)URL;

- (NSURL*)URL;

- (NSHTTPURLResponse*)response;

- (NSString*)HTTPMethod;  // Internal, don't call outside class hierarchy.

- (NSString*)contentType;  // Internal, don't call outside class hierarchy.

- (NSData*)bodyData;  // Internal, don't call outside class hierarchy.

- (NSData*)send:(NSError**)error;

/**
 Appends a file to the HTTP request, either by filename or by file content
 (in the form of NSData).
 */
+ (void)appendFileToBodyData:(NSMutableData*)data
                    withName:(NSString*)name
              withFileOrData:(id)fileOrData;

@end

NS_ASSUME_NONNULL_END
