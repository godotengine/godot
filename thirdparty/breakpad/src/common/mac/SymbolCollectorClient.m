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

#import "SymbolCollectorClient.h"

#import "HTTPGetRequest.h"
#import "HTTPSimplePostRequest.h"

@implementation UploadURLResponse

//=============================================================================
- (id)initWithUploadURL:(NSString*)uploadURL
          withUploadKey:(NSString*)uploadKey {
  if (self = [super init]) {
    uploadURL_ = [uploadURL copy];
    uploadKey_ = [uploadKey copy];
  }
  return self;
}

//=============================================================================
- (void)dealloc {
  [uploadURL_ release];
  [uploadKey_ release];

  [super dealloc];
}

//=============================================================================
- (NSString*)uploadURL {
  return uploadURL_;
}

//=============================================================================
- (NSString*)uploadKey {
  return uploadKey_;
}
@end

@implementation SymbolCollectorClient

//=============================================================================
+ (SymbolStatus)checkSymbolStatusOnServer:(NSString*)APIURL
                               withAPIKey:(NSString*)APIKey
                            withDebugFile:(NSString*)debugFile
                              withDebugID:(NSString*)debugID {
  // Note that forward-slash is listed as a character to escape here, for
  // completeness, however it is illegal in a debugFile input.
  NSMutableCharacterSet* allowedDebugFileCharacters = [NSMutableCharacterSet
      characterSetWithCharactersInString:@" \"\\/#%:?@|^`{}<>[]&=;"];
  [allowedDebugFileCharacters
      formUnionWithCharacterSet:[NSCharacterSet controlCharacterSet]];
  [allowedDebugFileCharacters invert];
  NSString* escapedDebugFile =
      [debugFile stringByAddingPercentEncodingWithAllowedCharacters:
                     allowedDebugFileCharacters];

  NSURL* URL = [NSURL
      URLWithString:[NSString
                        stringWithFormat:@"%@/v1/symbols/%@/%@:checkStatus"
                                         @"?key=%@",
                                         APIURL, escapedDebugFile, debugID,
                                         APIKey]];

  HTTPGetRequest* getRequest = [[HTTPGetRequest alloc] initWithURL:URL];
  NSError* error = nil;
  NSData* data = [getRequest send:&error];
  NSString* result = [[NSString alloc] initWithData:data
                                           encoding:NSUTF8StringEncoding];
  int responseCode = [[getRequest response] statusCode];
  [getRequest release];

  if (error || responseCode != 200) {
    fprintf(stdout, "Failed to check symbol status.\n");
    fprintf(stdout, "Response code: %d\n", responseCode);
    fprintf(stdout, "Response:\n");
    fprintf(stdout, "%s\n", [result UTF8String]);
    return SymbolStatusUnknown;
  }

  error = nil;
  NSRegularExpression* statusRegex = [NSRegularExpression
      regularExpressionWithPattern:@"\"status\": \"([^\"]+)\""
                           options:0
                             error:&error];
  NSArray* matches =
      [statusRegex matchesInString:result
                           options:0
                             range:NSMakeRange(0, [result length])];
  if ([matches count] != 1) {
    fprintf(stdout, "Failed to parse check symbol status response.");
    fprintf(stdout, "Response:\n");
    fprintf(stdout, "%s\n", [result UTF8String]);
    return SymbolStatusUnknown;
  }

  NSString* status = [result substringWithRange:[matches[0] rangeAtIndex:1]];
  [result release];

  return [status isEqualToString:@"FOUND"] ? SymbolStatusFound
                                           : SymbolStatusMissing;
}

//=============================================================================
+ (UploadURLResponse*)createUploadURLOnServer:(NSString*)APIURL
                                   withAPIKey:(NSString*)APIKey {
  NSURL* URL = [NSURL
      URLWithString:[NSString stringWithFormat:@"%@/v1/uploads:create?key=%@",
                                               APIURL, APIKey]];

  HTTPSimplePostRequest* postRequest =
      [[HTTPSimplePostRequest alloc] initWithURL:URL];
  NSError* error = nil;
  NSData* data = [postRequest send:&error];
  NSString* result = [[NSString alloc] initWithData:data
                                           encoding:NSUTF8StringEncoding];
  int responseCode = [[postRequest response] statusCode];
  [postRequest release];

  if (error || responseCode != 200) {
    fprintf(stdout, "Failed to create upload URL.\n");
    fprintf(stdout, "Response code: %d\n", responseCode);
    fprintf(stdout, "Response:\n");
    fprintf(stdout, "%s\n", [result UTF8String]);
    return nil;
  }

  // Note camel-case rather than underscores.
  NSRegularExpression* uploadURLRegex = [NSRegularExpression
      regularExpressionWithPattern:@"\"uploadUrl\": \"([^\"]+)\""
                           options:0
                             error:&error];
  NSRegularExpression* uploadKeyRegex = [NSRegularExpression
      regularExpressionWithPattern:@"\"uploadKey\": \"([^\"]+)\""
                           options:0
                             error:&error];

  NSArray* uploadURLMatches =
      [uploadURLRegex matchesInString:result
                              options:0
                                range:NSMakeRange(0, [result length])];
  NSArray* uploadKeyMatches =
      [uploadKeyRegex matchesInString:result
                              options:0
                                range:NSMakeRange(0, [result length])];
  if ([uploadURLMatches count] != 1 || [uploadKeyMatches count] != 1) {
    fprintf(stdout, "Failed to parse create url response.");
    fprintf(stdout, "Response:\n");
    fprintf(stdout, "%s\n", [result UTF8String]);
    return nil;
  }
  NSString* uploadURL =
      [result substringWithRange:[uploadURLMatches[0] rangeAtIndex:1]];
  NSString* uploadKey =
      [result substringWithRange:[uploadKeyMatches[0] rangeAtIndex:1]];

  return [[UploadURLResponse alloc] initWithUploadURL:uploadURL
                                        withUploadKey:uploadKey];
}

//=============================================================================
+ (CompleteUploadResult)completeUploadOnServer:(NSString*)APIURL
                                    withAPIKey:(NSString*)APIKey
                                 withUploadKey:(NSString*)uploadKey
                                 withDebugFile:(NSString*)debugFile
                                   withDebugID:(NSString*)debugID
                                      withType:(NSString*)type
                               withProductName:(NSString*)productName {
  NSURL* URL = [NSURL
      URLWithString:[NSString
                        stringWithFormat:@"%@/v1/uploads/%@:complete?key=%@",
                                         APIURL, uploadKey, APIKey]];

  NSMutableDictionary* jsonDictionary = [@{
    @"symbol_id" : @{@"debug_file" : debugFile, @"debug_id" : debugID},
    @"symbol_upload_type" : type, @"use_async_processing" : @"true"
  } mutableCopy];

  if (productName != nil) {
    jsonDictionary[@"metadata"] = @{@"product_name": productName};
  }

  NSError* error = nil;
  NSData* jsonData =
      [NSJSONSerialization dataWithJSONObject:jsonDictionary
                                      options:NSJSONWritingPrettyPrinted
                                        error:&error];
  if (jsonData == nil) {
    fprintf(stdout, "Error: %s\n", [[error localizedDescription] UTF8String]);
    fprintf(stdout,
            "Failed to complete upload. Could not write JSON payload.\n");
    return CompleteUploadResultError;
  }

  NSString* body = [[NSString alloc] initWithData:jsonData
                                         encoding:NSUTF8StringEncoding];
  HTTPSimplePostRequest* postRequest =
      [[HTTPSimplePostRequest alloc] initWithURL:URL];
  [postRequest setBody:body];
  [postRequest setContentType:@"application/json"];

  NSData* data = [postRequest send:&error];
  if (data == nil) {
    fprintf(stdout, "Error: %s\n", [[error localizedDescription] UTF8String]);
    fprintf(stdout, "Failed to complete upload URL.\n");
    return CompleteUploadResultError;
  }

  NSString* result = [[NSString alloc] initWithData:data
                                           encoding:NSUTF8StringEncoding];
  int responseCode = [[postRequest response] statusCode];
  [postRequest release];
  if (responseCode != 200) {
    fprintf(stdout, "Failed to complete upload URL.\n");
    fprintf(stdout, "Response code: %d\n", responseCode);
    fprintf(stdout, "Response:\n");
    fprintf(stdout, "%s\n", [result UTF8String]);
    return CompleteUploadResultError;
  }

  // Note camel-case rather than underscores.
  NSRegularExpression* completeResultRegex = [NSRegularExpression
      regularExpressionWithPattern:@"\"result\": \"([^\"]+)\""
                           options:0
                             error:&error];

  NSArray* completeResultMatches =
      [completeResultRegex matchesInString:result
                                   options:0
                                     range:NSMakeRange(0, [result length])];

  if ([completeResultMatches count] != 1) {
    fprintf(stdout, "Failed to parse complete upload response.");
    fprintf(stdout, "Response:\n");
    fprintf(stdout, "%s\n", [result UTF8String]);
    return CompleteUploadResultError;
  }
  NSString* completeResult =
      [result substringWithRange:[completeResultMatches[0] rangeAtIndex:1]];
  [result release];

  return ([completeResult isEqualToString:@"DUPLICATE_DATA"])
             ? CompleteUploadResultDuplicateData
             : CompleteUploadResultOk;
}
@end
