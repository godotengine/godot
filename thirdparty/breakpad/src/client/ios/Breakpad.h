// Copyright 2011 Google LLC
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

// Framework to provide a simple C API to crash reporting for
// applications.  By default, if any machine-level exception (e.g.,
// EXC_BAD_ACCESS) occurs, it will be handled by the BreakpadRef
// object as follows:
//
// 1. Create a minidump file (see Breakpad for details)
// 2. Create a config file.
//
// These files can then be uploaded to a server.

typedef void* BreakpadRef;

#ifdef __cplusplus
extern "C" {
#endif

#include <Foundation/Foundation.h>

#include <client/apple/Framework/BreakpadDefines.h>

// The keys in the dictionary returned by |BreakpadGenerateReport|.
#define BREAKPAD_OUTPUT_DUMP_FILE   "BreakpadDumpFile"
#define BREAKPAD_OUTPUT_CONFIG_FILE "BreakpadConfigFile"

// Optional user-defined function to decide if we should handle this crash or
// forward it along.
// Return true if you want Breakpad to handle it.
// Return false if you want Breakpad to skip it
// The exception handler always returns false, as if SEND_AND_EXIT were false
// (which means the next exception handler will take the exception)
typedef bool (*BreakpadFilterCallback)(int exception_type,
                                       int exception_code,
                                       mach_port_t crashing_thread,
                                       void* context);

// Optional user-defined function that will be called after a network upload
// of a crash report.
// |report_id| will be the id returned by the server, or "ERR" if an error
// occurred.
// |error| will contain the error, or nil if no error occured.
typedef void (*BreakpadUploadCompletionCallback)(NSString* report_id,
                                                 NSError* error);

// Create a new BreakpadRef object and install it as an exception
// handler.  The |parameters| will typically be the contents of your
// bundle's Info.plist.
//
// You can also specify these additional keys for customizable behavior:
// Key:                           Value:
// BREAKPAD_PRODUCT               Product name (e.g., "MyAwesomeProduct")
//                                This one is used as the key to identify
//                                the product when uploading. Falls back to
//                                CFBundleName if not specified.
//                                REQUIRED
//
// BREAKPAD_PRODUCT_DISPLAY       This is the display name, e.g. a pretty
//                                name for the product when the crash_sender
//                                pops up UI for the user. Falls back first to
//                                CFBundleDisplayName and then to
//                                BREAKPAD_PRODUCT if not specified.
//
// BREAKPAD_VERSION               Product version (e.g., 1.2.3), used
//                                as metadata for crash report. Falls back to
//                                CFBundleVersion if not specified.
//                                REQUIRED
//
// BREAKPAD_VENDOR                Vendor name, used in UI (e.g. "A report has
//                                been created that you can send to <vendor>")
//
// BREAKPAD_URL                   URL destination for reporting
//                                REQUIRED
//
// BREAKPAD_DUMP_DIRECTORY        The directory to store crash-dumps
//                                in. By default, we use
//                                ~/Library/Cache/Breakpad/<BREAKPAD_PRODUCT>
//                                The path you specify here is tilde-expanded.
//
// BREAKPAD_SERVER_TYPE           A parameter that tells Breakpad how to
//                                rewrite the upload parameters for a specific
//                                server type.  The currently valid values are
//                                'socorro' or 'google'.  If you want to add
//                                other types, see the function in
//                                crash_report_sender.m that maps parameters to
//                                URL parameters.  Defaults to 'google'.
//
// BREAKPAD_SERVER_PARAMETER_DICT A plist dictionary of static
//                                parameters that are uploaded to the
//                                server.  The parameters are sent as
//                                is to the crash server.  Their
//                                content isn't added to the minidump
//                                but pass as URL parameters when
//                                uploading theminidump to the crash
//                                server.
//=============================================================================
// The BREAKPAD_PRODUCT, BREAKPAD_VERSION and BREAKPAD_URL are
// required to have non-NULL values.  By default, the BREAKPAD_PRODUCT
// will be the CFBundleName and the BREAKPAD_VERSION will be the
// CFBundleVersion when these keys are present in the bundle's
// Info.plist, which is usually passed in to BreakpadCreate() as an
// NSDictionary (you could also pass in another dictionary that had
// the same keys configured).  If the BREAKPAD_PRODUCT or
// BREAKPAD_VERSION are ultimately undefined, BreakpadCreate() will
// fail.  You have been warned.
//
// If you are running in a debugger, Breakpad will not install, unless the
// BREAKPAD_IGNORE_DEBUGGER envionment variable is set and/or non-zero.
//
//=============================================================================
// The following are NOT user-supplied but are documented here for
// completeness.  They are calculated by Breakpad during initialization &
// crash-dump generation, or entered in by the user.
//
// BREAKPAD_PROCESS_START_TIME       The time, in seconds since the Epoch, the
//                                   process started
//
// BREAKPAD_PROCESS_CRASH_TIME       The time, in seconds since the Epoch, the
//                                   process crashed.
//
// BREAKPAD_PROCESS_UP_TIME          The total time in milliseconds the process
//                                   has been running.  This parameter is not
//                                   set until the crash-dump-generation phase.
//
// BREAKPAD_SERVER_PARAMETER_PREFIX  This prefix is used by Breakpad
//                                   internally, because Breakpad uses
//                                   the same dictionary internally to
//                                   track both its internal
//                                   configuration parameters and
//                                   parameters meant to be uploaded
//                                   to the server.  This string is
//                                   used internally by Breakpad to
//                                   prefix user-supplied parameter
//                                   names so those can be sent to the
//                                   server without leaking Breakpad's
//                                   internal values.

// Returns a new BreakpadRef object on success, NULL otherwise.
BreakpadRef BreakpadCreate(NSDictionary* parameters);

// Uninstall and release the data associated with |ref|.
void BreakpadRelease(BreakpadRef ref);

// User defined key and value string storage.  Generally this is used
// to configure Breakpad's internal operation, such as whether the
// crash_sender should prompt the user, or the filesystem location for
// the minidump file.  See Breakpad.h for some parameters that can be
// set.  Anything longer than 255 bytes will be truncated. Note that
// the string is converted to UTF8 before truncation, so any multibyte
// character that straddles the 255(256 - 1 for terminator) byte limit
// will be mangled.
//
// A maximum number of 64 key/value pairs are supported.  An assert()
// will fire if more than this number are set.  Unfortunately, right
// now, the same dictionary is used for both Breakpad's parameters AND
// the Upload parameters.
//
// TODO (nealsid): Investigate how necessary this is if we don't
// automatically upload parameters to the server anymore.
// TODO (nealsid): separate server parameter dictionary from the
// dictionary used to configure Breakpad, and document limits for each
// independently.
void BreakpadSetKeyValue(BreakpadRef ref, NSString* key, NSString* value);
NSString* BreakpadKeyValue(BreakpadRef ref, NSString* key);
void BreakpadRemoveKeyValue(BreakpadRef ref, NSString* key);

// You can use this method to specify parameters that will be uploaded
// to the crash server.  They will be automatically encoded as
// necessary.  Note that as mentioned above there are limits on both
// the number of keys and their length.
void BreakpadAddUploadParameter(BreakpadRef ref, NSString* key,
                                NSString* value);

// This method will remove a previously-added parameter from the
// upload parameter set.
void BreakpadRemoveUploadParameter(BreakpadRef ref, NSString* key);

// Method to handle uploading data to the server

// Returns the number of crash reports waiting to send to the server.
int BreakpadGetCrashReportCount(BreakpadRef ref);

// Returns the next upload configuration. The report file is deleted.
NSDictionary* BreakpadGetNextReportConfiguration(BreakpadRef ref);

// Returns the date of the most recent crash report.
NSDate* BreakpadGetDateOfMostRecentCrashReport(BreakpadRef ref);

// Upload next report to the server.
void BreakpadUploadNextReport(BreakpadRef ref);

// Upload next report to the server.
// |server_parameters| is additional server parameters to send.
void BreakpadUploadNextReportWithParameters(
    BreakpadRef ref,
    NSDictionary* server_parameters,
    BreakpadUploadCompletionCallback callback);

// Upload a report to the server.
// |server_parameters| is additional server parameters to send.
// |configuration| is the configuration of the breakpad report to send.
void BreakpadUploadReportWithParametersAndConfiguration(
    BreakpadRef ref,
    NSDictionary* server_parameters,
    NSDictionary* configuration,
    BreakpadUploadCompletionCallback callback);

// Handles the network response of a breakpad upload. This function is needed if
// the actual upload is done by the Breakpad client.
// |configuration| is the configuration of the upload. It must contain the same
// fields as the configuration passed to
// BreakpadUploadReportWithParametersAndConfiguration.
// |data| and |error| contain the network response.
void BreakpadHandleNetworkResponse(BreakpadRef ref,
                                   NSDictionary* configuration,
                                   NSData* data,
                                   NSError* error);

// Upload a file to the server. |data| is the content of the file to sent.
// |server_parameters| is additional server parameters to send.
void BreakpadUploadData(BreakpadRef ref, NSData* data, NSString* name,
                        NSDictionary* server_parameters);

// Generate a breakpad minidump and configuration file in the dump directory.
// The report will be available for uploading. The paths of the created files
// are returned in the dictionary. |server_parameters| is additional server
// parameters to add in the config file.
NSDictionary* BreakpadGenerateReport(BreakpadRef ref,
                                     NSDictionary* server_parameters);

#ifdef __cplusplus
}
#endif
