// Copyright (c) 2012, Google Inc.
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

#ifndef CLIENT_IOS_HANDLER_IOS_BREAKPAD_CONTROLLER_H_
#define CLIENT_IOS_HANDLER_IOS_BREAKPAD_CONTROLLER_H_

#import <Foundation/Foundation.h>

#import "client/ios/Breakpad.h"

// This class is used to offer a higher level API around BreakpadRef. It
// configures it, ensures thread-safety, and sends crash reports back to the
// collecting server. By default, no crash reports are sent, the user must call
// |setUploadingEnabled:YES| to start the uploading.
@interface BreakpadController : NSObject {
 @private
  // The dispatch queue that will own the breakpad reference.
  dispatch_queue_t queue_;

  // Instance of Breakpad crash reporter. This is owned by the queue, but can
  // be created on the main thread at startup.
  BreakpadRef breakpadRef_;

  // The dictionary that contains configuration for breakpad. Modifying it
  // should only happen when the controller is not started. The initial value
  // is the infoDictionary of the bundle of the application.
  NSMutableDictionary* configuration_;

  // Whether or not crash reports should be uploaded.
  BOOL enableUploads_;

  // Whether the controller has been started on the main thread. This is only
  // used to assert the initialization order is correct.
  BOOL started_;

  // The interval to wait between two uploads. Value is 0 if no upload must be
  // done.
  int uploadIntervalInSeconds_;

  // The dictionary that contains additional server parameters to send when
  // uploading crash reports.
  NSDictionary* uploadTimeParameters_;

  // The callback to call on report upload completion.
  BreakpadUploadCompletionCallback uploadCompleteCallback_;
}

// Singleton.
+ (BreakpadController*)sharedInstance;

// Update the controller configuration. Merges its old configuration with the
// new one. Merge is done by replacing the old values by the new values.
- (void)updateConfiguration:(NSDictionary*)configuration;

// Reset the controller configuration to its initial value, which is the
// infoDictionary of the bundle of the application.
- (void)resetConfiguration;

// Configure the URL to upload the report to. This must be called at least once
// if the URL is not in the bundle information.
- (void)setUploadingURL:(NSString*)url;

// Set the minimal interval between two uploads in seconds. This must be called
// at least once if the interval is not in the bundle information. A value of 0
// will prevent uploads.
- (void)setUploadInterval:(int)intervalInSeconds;

// Set additional server parameters to send when uploading crash reports.
- (void)setParametersToAddAtUploadTime:(NSDictionary*)uploadTimeParameters;

// Specify an upload parameter that will be added to the crash report when a
// crash report is generated. See |BreakpadAddUploadParameter|.
- (void)addUploadParameter:(NSString*)value forKey:(NSString*)key;

// Sets the callback to be called after uploading a crash report to the server.
// Only the latest callback registered will be called.
- (void)setUploadCallback:(BreakpadUploadCompletionCallback)callback;

// Remove a previously-added parameter from the upload parameter set. See
// |BreakpadRemoveUploadParameter|.
- (void)removeUploadParameterForKey:(NSString*)key;

// Access the underlying BreakpadRef. This method is asynchronous, and will be
// executed on the thread owning the BreakpadRef variable. Moreover, if the
// controller is not started, the block will be called with a NULL parameter.
- (void)withBreakpadRef:(void(^)(BreakpadRef))callback;

// Starts the BreakpadController by registering crash handlers. If
// |onCurrentThread| is YES, all setup is done on the current thread, otherwise
// it is done on a private queue.
- (void)start:(BOOL)onCurrentThread;

// Unregisters the crash handlers.
- (void)stop;

// Returns whether or not the controller is started.
- (BOOL)isStarted;

// Enables or disables uploading of crash reports, but does not stop the
// BreakpadController.
- (void)setUploadingEnabled:(BOOL)enabled;

// Check if there is currently a crash report to upload.
- (void)hasReportToUpload:(void(^)(BOOL))callback;

// Get the number of crash reports waiting to upload.
- (void)getCrashReportCount:(void(^)(int))callback;

// Get the next report to upload.
// - If upload is disabled, callback will be called with (nil, -1).
// - If a delay is to be waited before sending, callback will be called with
//   (nil, n), with n (> 0) being the number of seconds to wait.
// - if no delay is needed, callback will be called with (0, configuration),
//   configuration being next report to upload, or nil if none is pending.
- (void)getNextReportConfigurationOrSendDelay:
    (void(^)(NSDictionary*, int))callback;

// Get the date of the most recent crash report.
- (void)getDateOfMostRecentCrashReport:(void(^)(NSDate *))callback;

// Sends synchronously the report specified by |configuration|. This method is
// NOT thread safe and must be called from the breakpad thread.
- (void)threadUnsafeSendReportWithConfiguration:(NSDictionary*)configuration
                                withBreakpadRef:(BreakpadRef)ref;

@end

#endif  // CLIENT_IOS_HANDLER_IOS_BREAKPAD_CONTROLLER_H_
