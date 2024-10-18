// Copyright 2006 Google LLC
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
//
// Framework to provide a simple C API to crash reporting for
// applications.  By default, if any machine-level exception (e.g.,
// EXC_BAD_ACCESS) occurs, it will be handled by the BreakpadRef
// object as follows:
//
// 1. Create a minidump file (see Breakpad for details)
// 2. Prompt the user (using CFUserNotification)
// 3. Invoke a command line reporting tool to send the minidump to a
//    server
//
// By specifying parameters to the BreakpadCreate function, you can
// modify the default behavior to suit your needs and wants and
// desires.

// A service name associated with the original bootstrap parent port, saved in
// OnDemandServer and restored in Inspector.
#define BREAKPAD_BOOTSTRAP_PARENT_PORT    "com.Breakpad.BootstrapParent"

typedef void* BreakpadRef;

#ifdef __cplusplus
extern "C" {
#endif

#include <CoreFoundation/CoreFoundation.h>
#include <Foundation/Foundation.h>

#include "BreakpadDefines.h"

// Optional user-defined function to dec to decide if we should handle
// this crash or forward it along.
// Return true if you want Breakpad to handle it.
// Return false if you want Breakpad to skip it
// The exception handler always returns false, as if SEND_AND_EXIT were false
// (which means the next exception handler will take the exception)
typedef bool (*BreakpadFilterCallback)(int exception_type,
                                       int exception_code,
                                       mach_port_t crashing_thread,
                                       void* context);

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
// BREAKPAD_REPORT_INTERVAL       # of seconds between sending
//                                reports.  If an additional report is
//                                generated within this time, it will
//                                be ignored.  Default: 3600sec.
//                                Specify 0 to send all reports.
//
// BREAKPAD_SKIP_CONFIRM          If true, the reporter will send the report
//                                without any user intervention.
//                                Defaults to NO
//
// BREAKPAD_CONFIRM_TIMEOUT       Number of seconds before the upload
//                                confirmation dialog will be automatically
//                                dismissed (cancelling the upload).
//                                Default: 300 seconds (min of 60).
//                                Specify 0 to prevent timeout.
//
// BREAKPAD_SEND_AND_EXIT         If true, the handler will exit after sending.
//                                This will prevent any other handler (e.g.,
//                                CrashReporter) from getting the crash.
//                                Defaults TO YES
//
// BREAKPAD_DUMP_DIRECTORY        The directory to store crash-dumps
//                                in. By default, we use
//                                ~/Library/Breakpad/<BREAKPAD_PRODUCT>
//                                The path you specify here is tilde-expanded.
//
// BREAKPAD_INSPECTOR_LOCATION    The full path to the Inspector executable.
//                                Defaults to <Framework resources>/Inspector
//
// BREAKPAD_REPORTER_EXE_LOCATION The full path to the Reporter/sender
//                                executable.
//                                Default:
//                                <Framework Resources>/crash_report_sender.app
//
// BREAKPAD_LOGFILES              Indicates an array of log file paths that
//                                should be uploaded at crash time.
//
// BREAKPAD_REQUEST_COMMENTS      If true, the message dialog will have a
//                                text box for the user to enter comments.
//                                Default: NO
//
// BREAKPAD_REQUEST_EMAIL         If true and BREAKPAD_REQUEST_COMMENTS is also
//                                true, the message dialog will have a text
//                                box for the user to enter their email address.
//                                Default: NO
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
//
// BREAKPAD_IN_PROCESS            A boolean NSNumber value. If YES, Breakpad
//                                will write the dump file in-process and then
//                                launch the reporter executable as a child
//                                process.
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
// The BREAKPAD_SKIP_CONFIRM and BREAKPAD_SEND_AND_EXIT default
// values are NO and YES.  However, they can be controlled by setting their
// values in a user or global plist.
//
// It's easiest to use Breakpad via the Framework, but if you're compiling the
// code in directly, BREAKPAD_INSPECTOR_LOCATION and
// BREAKPAD_REPORTER_EXE_LOCATION allow you to specify custom paths
// to the helper executables.
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
// BREAKPAD_LOGFILE_KEY_PREFIX       Used to find out which parameters in the
//                                   parameter dictionary correspond to log
//                                   file paths.
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
//
// BREAKPAD_ON_DEMAND                Used internally to indicate to the
//                                   Reporter that we're sending on-demand,
//                                   not as result of a crash.
//
// BREAKPAD_COMMENTS                 The text the user provided as comments.
//                                   Only used in crash_report_sender.

// Returns a new BreakpadRef object on success, NULL otherwise.
BreakpadRef BreakpadCreate(NSDictionary* parameters);

// Uninstall and release the data associated with |ref|.
void BreakpadRelease(BreakpadRef ref);

// Clients may set an optional callback which gets called when a crash
// occurs.  The callback function should return |true| if we should
// handle the crash, generate a crash report, etc. or |false| if we
// should ignore it and forward the crash (normally to CrashReporter).
// Context is a pointer to arbitrary data to make the callback with.
void BreakpadSetFilterCallback(BreakpadRef ref,
                               BreakpadFilterCallback callback,
                               void* context);

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

// Add a log file for Breakpad to read and send upon crash dump
void BreakpadAddLogFile(BreakpadRef ref, NSString* logPathname);

// Generate a minidump and send
void BreakpadGenerateAndSendReport(BreakpadRef ref);

#ifdef __cplusplus
}
#endif
