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

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/**
 Represents a response from a sym-upload-v2 server to a createUploadURLOnServer
 call.
 */
@interface UploadURLResponse : NSObject {
 @protected
  NSString* uploadURL_;
  NSString* uploadKey_;
}

- (id)initWithUploadURL:(NSString*)uploadURL withUploadKey:(NSString*)uploadKey;

- (NSString*)uploadURL;
- (NSString*)uploadKey;
@end

/**
 Possible return statuses from a sym-upload-v2 server to a
 completeUploadOnServer call.
 */
typedef NS_ENUM(NSInteger, CompleteUploadResult) {
  CompleteUploadResultOk,
  CompleteUploadResultDuplicateData,
  CompleteUploadResultError
};

/**
 Possible return statuses from a sym-upload-v2 server to a
 checkSymbolStatusOnServer call.
 */
typedef NS_ENUM(NSInteger, SymbolStatus) {
  SymbolStatusFound,
  SymbolStatusMissing,
  SymbolStatusUnknown
};

/**
 Interface to help a client interact with a sym-upload-v2 server, over HTTP.
 For details of the API and protocol, see :/docs/sym_upload_v2_protocol.md.
 */
@interface SymbolCollectorClient : NSObject
;

/**
 Calls the /v1/symbols/{debug_file}/{debug_id}:checkStatus API on the server.
 */
+ (SymbolStatus)checkSymbolStatusOnServer:(NSString*)APIURL
                               withAPIKey:(NSString*)APIKey
                            withDebugFile:(NSString*)debugFile
                              withDebugID:(NSString*)debugID;

/**
 Calls the /v1/uploads:create API on the server.
 */
+ (UploadURLResponse*)createUploadURLOnServer:(NSString*)APIURL
                                   withAPIKey:(NSString*)APIKey;

/**
 Calls the /v1/uploads/{key}:complete API on the server.
 */
+ (CompleteUploadResult)completeUploadOnServer:(NSString*)APIURL
                                    withAPIKey:(NSString*)APIKey
                                 withUploadKey:(NSString*)uploadKey
                                 withDebugFile:(NSString*)debugFile
                                   withDebugID:(NSString*)debugID
                                      withType:(NSString*)type
                               withProductName:(NSString*)productName;

@end

NS_ASSUME_NONNULL_END
