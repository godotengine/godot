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


#define IGNORE_DEBUGGER "BREAKPAD_IGNORE_DEBUGGER"

#import "client/mac/Framework/Breakpad.h"

#include <assert.h>
#import <Foundation/Foundation.h>
#include <pthread.h>
#include <sys/stat.h>
#include <sys/sysctl.h>

#import "client/mac/crash_generation/Inspector.h"
#import "client/mac/handler/exception_handler.h"
#import "client/mac/Framework/Breakpad.h"
#import "client/mac/Framework/OnDemandServer.h"
#import "client/mac/handler/protected_memory_allocator.h"
#include "common/mac/launch_reporter.h"
#import "common/mac/MachIPC.h"
#import "common/simple_string_dictionary.h"

#if !defined(__EXCEPTIONS) || (__clang__ && !__has_feature(cxx_exceptions))
// This file uses C++ try/catch (but shouldn't). Duplicate the macros from
// <c++/4.2.1/exception_defines.h> allowing this file to work properly with
// exceptions disabled even when other C++ libraries are used. #undef the try
// and catch macros first in case libstdc++ is in use and has already provided
// its own definitions.
#undef try
#define try       if (true)
#undef catch
#define catch(X)  if (false)
#endif  // __EXCEPTIONS

using google_breakpad::MachPortSender;
using google_breakpad::MachReceiveMessage;
using google_breakpad::MachSendMessage;
using google_breakpad::ReceivePort;
using google_breakpad::SimpleStringDictionary;

//=============================================================================
// We want any memory allocations which are used by breakpad during the
// exception handling process (after a crash has happened) to be read-only
// to prevent them from being smashed before a crash occurs.  Unfortunately
// we cannot protect against smashes to our exception handling thread's
// stack.
//
// NOTE: Any memory allocations which are not used during the exception
// handling process may be allocated in the normal ways.
//
// The ProtectedMemoryAllocator class provides an Allocate() method which
// we'll using in conjunction with placement operator new() to control
// allocation of C++ objects.  Note that we don't use operator delete()
// but instead call the objects destructor directly:  object->~ClassName();
//
ProtectedMemoryAllocator* gMasterAllocator = NULL;
ProtectedMemoryAllocator* gKeyValueAllocator = NULL;
ProtectedMemoryAllocator* gBreakpadAllocator = NULL;

// Mutex for thread-safe access to the key/value dictionary used by breakpad.
// It's a global instead of an instance variable of Breakpad
// since it can't live in a protected memory area.
pthread_mutex_t gDictionaryMutex;

//=============================================================================
// Stack-based object for thread-safe access to a memory-protected region.
// It's assumed that normally the memory block (allocated by the allocator)
// is protected (read-only).  Creating a stack-based instance of
// ProtectedMemoryLocker will unprotect this block after taking the lock.
// Its destructor will first re-protect the memory then release the lock.
class ProtectedMemoryLocker {
 public:
  ProtectedMemoryLocker(pthread_mutex_t* mutex,
                        ProtectedMemoryAllocator* allocator)
      : mutex_(mutex),
        allocator_(allocator) {
    // Lock the mutex
    __attribute__((unused)) int rv = pthread_mutex_lock(mutex_);
    assert(rv == 0);

    // Unprotect the memory
    allocator_->Unprotect();
  }

  ~ProtectedMemoryLocker() {
    // First protect the memory
    allocator_->Protect();

    // Then unlock the mutex
    __attribute__((unused)) int rv = pthread_mutex_unlock(mutex_);
    assert(rv == 0);
  }

 private:
  ProtectedMemoryLocker();
  ProtectedMemoryLocker(const ProtectedMemoryLocker&);
  ProtectedMemoryLocker& operator=(const ProtectedMemoryLocker&);

  pthread_mutex_t* mutex_;
  ProtectedMemoryAllocator* allocator_;
};

//=============================================================================
class Breakpad {
 public:
  // factory method
  static Breakpad* Create(NSDictionary* parameters) {
    // Allocate from our special allocation pool
    Breakpad* breakpad =
      new (gBreakpadAllocator->Allocate(sizeof(Breakpad)))
        Breakpad();

    if (!breakpad)
      return NULL;

    if (!breakpad->Initialize(parameters)) {
      // Don't use operator delete() here since we allocated from special pool
      breakpad->~Breakpad();
      return NULL;
    }

    return breakpad;
  }

  ~Breakpad();

  void SetKeyValue(NSString* key, NSString* value);
  NSString* KeyValue(NSString* key);
  void RemoveKeyValue(NSString* key);

  void GenerateAndSendReport();

  void SetFilterCallback(BreakpadFilterCallback callback, void* context) {
    filter_callback_ = callback;
    filter_callback_context_ = context;
  }

 private:
  Breakpad()
    : handler_(NULL),
      config_params_(NULL),
      send_and_exit_(true),
      filter_callback_(NULL),
      filter_callback_context_(NULL) {
    inspector_path_[0] = 0;
  }

  bool Initialize(NSDictionary* parameters);
  bool InitializeInProcess(NSDictionary* parameters);
  bool InitializeOutOfProcess(NSDictionary* parameters);

  bool ExtractParameters(NSDictionary* parameters);

  // Dispatches to HandleException()
  static bool ExceptionHandlerDirectCallback(void* context,
                                             int exception_type,
                                             int exception_code,
                                             int exception_subcode,
                                             mach_port_t crashing_thread);

  bool HandleException(int exception_type,
                       int exception_code,
                       int exception_subcode,
                       mach_port_t crashing_thread);

  // Dispatches to HandleMinidump().
  // This gets called instead of ExceptionHandlerDirectCallback when running
  // with the BREAKPAD_IN_PROCESS option.
  static bool HandleMinidumpCallback(const char* dump_dir,
                                     const char* minidump_id,
                                     void* context,
                                     bool succeeded);

  // This is only used when BREAKPAD_IN_PROCESS is YES.
  bool HandleMinidump(const char* dump_dir, const char* minidump_id);

  // Since ExceptionHandler (w/o namespace) is defined as typedef in OSX's
  // MachineExceptions.h, we have to explicitly name the handler.
  google_breakpad::ExceptionHandler* handler_; // The actual handler (STRONG)

  char                    inspector_path_[PATH_MAX];  // Path to inspector tool

  SimpleStringDictionary* config_params_; // Create parameters (STRONG)

  OnDemandServer          inspector_;

  bool                    send_and_exit_;  // Exit after sending, if true

  BreakpadFilterCallback  filter_callback_;
  void*                   filter_callback_context_;
};

#pragma mark -
#pragma mark Helper functions

//=============================================================================
// Helper functions

//=============================================================================
static BOOL IsDebuggerActive() {
  BOOL result = NO;
  NSUserDefaults* stdDefaults = [NSUserDefaults standardUserDefaults];

  // We check both defaults and the environment variable here

  BOOL ignoreDebugger = [stdDefaults boolForKey:@IGNORE_DEBUGGER];

  if (!ignoreDebugger) {
    char* ignoreDebuggerStr = getenv(IGNORE_DEBUGGER);
    ignoreDebugger = (ignoreDebuggerStr ? strtol(ignoreDebuggerStr, NULL, 10) : 0) != 0;
  }

  if (!ignoreDebugger) {
    pid_t pid = getpid();
    int mib[4] = {CTL_KERN, KERN_PROC, KERN_PROC_PID, pid};
    int mibSize = sizeof(mib) / sizeof(int);
    size_t actualSize;

    if (sysctl(mib, mibSize, NULL, &actualSize, NULL, 0) == 0) {
      struct kinfo_proc* info = (struct kinfo_proc*)malloc(actualSize);

      if (info) {
        // This comes from looking at the Darwin xnu Kernel
        if (sysctl(mib, mibSize, info, &actualSize, NULL, 0) == 0)
          result = (info->kp_proc.p_flag & P_TRACED) ? YES : NO;

        free(info);
      }
    }
  }

  return result;
}

//=============================================================================
bool Breakpad::ExceptionHandlerDirectCallback(void* context,
                                              int exception_type,
                                              int exception_code,
                                              int exception_subcode,
                                              mach_port_t crashing_thread) {
  Breakpad* breakpad = (Breakpad*)context;

  // If our context is damaged or something, just return false to indicate that
  // the handler should continue without us.
  if (!breakpad)
    return false;

  return breakpad->HandleException( exception_type,
                                    exception_code,
                                    exception_subcode,
                                    crashing_thread);
}

//=============================================================================
bool Breakpad::HandleMinidumpCallback(const char* dump_dir,
                                      const char* minidump_id,
                                      void* context,
                                      bool succeeded) {
  Breakpad* breakpad = (Breakpad*)context;

  // If our context is damaged or something, just return false to indicate that
  // the handler should continue without us.
  if (!breakpad || !succeeded)
    return false;

  return breakpad->HandleMinidump(dump_dir, minidump_id);
}

//=============================================================================
#pragma mark -

#include <dlfcn.h>

//=============================================================================
// Returns the pathname to the Resources directory for this version of
// Breakpad which we are now running.
//
// Don't make the function static, since _dyld_lookup_and_bind_fully needs a
// simple non-static C name
//
extern "C" {
NSString* GetResourcePath();
NSString* GetResourcePath() {
  NSString* resourcePath = nil;

  // If there are multiple breakpads installed then calling bundleWithIdentifier
  // will not work properly, so only use that as a backup plan.
  // We want to find the bundle containing the code where this function lives
  // and work from there
  //

  // Get the pathname to the code which contains this function
  Dl_info info;
  if (dladdr((const void*)GetResourcePath, &info) != 0) {
    NSFileManager* filemgr = [NSFileManager defaultManager];
    NSString* filePath =
        [filemgr stringWithFileSystemRepresentation:info.dli_fname
                                             length:strlen(info.dli_fname)];
    NSString* bundlePath = [filePath stringByDeletingLastPathComponent];
    // The "Resources" directory should be in the same directory as the
    // executable code, since that's how the Breakpad framework is built.
    resourcePath = [bundlePath stringByAppendingPathComponent:@"Resources/"];
  } else {
    // fallback plan
    NSBundle* bundle =
        [NSBundle bundleWithIdentifier:@"com.Google.BreakpadFramework"];
    resourcePath = [bundle resourcePath];
  }

  return resourcePath;
}
}  // extern "C"

//=============================================================================
bool Breakpad::Initialize(NSDictionary* parameters) {
  // Initialize
  config_params_ = NULL;
  handler_ = NULL;

  // Check for debugger
  if (IsDebuggerActive()) {
    return true;
  }

  // Gather any user specified parameters
  if (!ExtractParameters(parameters)) {
    return false;
  }

  if ([[parameters objectForKey:@BREAKPAD_IN_PROCESS] boolValue])
    return InitializeInProcess(parameters);
  else
    return InitializeOutOfProcess(parameters);
}

//=============================================================================
bool Breakpad::InitializeInProcess(NSDictionary* parameters) {
  handler_ =
      new (gBreakpadAllocator->Allocate(
          sizeof(google_breakpad::ExceptionHandler)))
          google_breakpad::ExceptionHandler(
              config_params_->GetValueForKey(BREAKPAD_DUMP_DIRECTORY),
              0, &HandleMinidumpCallback, this, true, 0);
  return true;    
}

//=============================================================================
bool Breakpad::InitializeOutOfProcess(NSDictionary* parameters) {
  // Get path to Inspector executable.
  NSString* inspectorPathString = KeyValue(@BREAKPAD_INSPECTOR_LOCATION);

  // Standardize path (resolve symlinkes, etc.)  and escape spaces
  inspectorPathString = [inspectorPathString stringByStandardizingPath];
  inspectorPathString = [[inspectorPathString componentsSeparatedByString:@" "]
                                              componentsJoinedByString:@"\\ "];

  // Create an on-demand server object representing the Inspector.
  // In case of a crash, we simply need to call the LaunchOnDemand()
  // method on it, then send a mach message to its service port.
  // It will then launch and perform a process inspection of our crashed state.
  // See the HandleException() method for the details.
#define RECEIVE_PORT_NAME "com.Breakpad.Inspector"

  name_t portName;
  snprintf(portName, sizeof(name_t),  "%s%d", RECEIVE_PORT_NAME, getpid());

  // Save the location of the Inspector
  strlcpy(inspector_path_, [inspectorPathString fileSystemRepresentation],
          sizeof(inspector_path_));

  // Append a single command-line argument to the Inspector path
  // representing the bootstrap name of the launch-on-demand receive port.
  // When the Inspector is launched, it can use this to lookup the port
  // by calling bootstrap_check_in().
  strlcat(inspector_path_, " ", sizeof(inspector_path_));
  strlcat(inspector_path_, portName, sizeof(inspector_path_));

  kern_return_t kr = inspector_.Initialize(inspector_path_,
                                           portName,
                                           true);        // shutdown on exit

  if (kr != KERN_SUCCESS) {
    return false;
  }

  // Create the handler (allocating it in our special protected pool)
  handler_ =
      new (gBreakpadAllocator->Allocate(
          sizeof(google_breakpad::ExceptionHandler)))
          google_breakpad::ExceptionHandler(
              Breakpad::ExceptionHandlerDirectCallback, this, true);
  return true;
}

//=============================================================================
Breakpad::~Breakpad() {
  // Note that we don't use operator delete() on these pointers,
  // since they were allocated by ProtectedMemoryAllocator objects.
  //
  if (config_params_) {
    config_params_->~SimpleStringDictionary();
  }

  if (handler_)
    handler_->~ExceptionHandler();
}

//=============================================================================
bool Breakpad::ExtractParameters(NSDictionary* parameters) {
  NSUserDefaults* stdDefaults = [NSUserDefaults standardUserDefaults];
  NSString* skipConfirm = [stdDefaults stringForKey:@BREAKPAD_SKIP_CONFIRM];
  NSString* sendAndExit = [stdDefaults stringForKey:@BREAKPAD_SEND_AND_EXIT];

  NSString* serverType = [parameters objectForKey:@BREAKPAD_SERVER_TYPE];
  NSString* display = [parameters objectForKey:@BREAKPAD_PRODUCT_DISPLAY];
  NSString* product = [parameters objectForKey:@BREAKPAD_PRODUCT];
  NSString* version = [parameters objectForKey:@BREAKPAD_VERSION];
  NSString* urlStr = [parameters objectForKey:@BREAKPAD_URL];
  NSString* interval = [parameters objectForKey:@BREAKPAD_REPORT_INTERVAL];
  NSString* inspectorPathString =
      [parameters objectForKey:@BREAKPAD_INSPECTOR_LOCATION];
  NSString* reporterPathString =
      [parameters objectForKey:@BREAKPAD_REPORTER_EXE_LOCATION];
  NSString* timeout = [parameters objectForKey:@BREAKPAD_CONFIRM_TIMEOUT];
  NSArray*  logFilePaths = [parameters objectForKey:@BREAKPAD_LOGFILES];
  NSString* logFileTailSize =
      [parameters objectForKey:@BREAKPAD_LOGFILE_UPLOAD_SIZE];
  NSString* requestUserText =
      [parameters objectForKey:@BREAKPAD_REQUEST_COMMENTS];
  NSString* requestEmail = [parameters objectForKey:@BREAKPAD_REQUEST_EMAIL];
  NSString* vendor =
      [parameters objectForKey:@BREAKPAD_VENDOR];
  NSString* dumpSubdirectory =
      [parameters objectForKey:@BREAKPAD_DUMP_DIRECTORY];

  NSDictionary* serverParameters =
      [parameters objectForKey:@BREAKPAD_SERVER_PARAMETER_DICT];

  // These may have been set above as user prefs, which take priority.
  if (!skipConfirm) {
    skipConfirm = [parameters objectForKey:@BREAKPAD_SKIP_CONFIRM];
  }
  if (!sendAndExit) {
    sendAndExit = [parameters objectForKey:@BREAKPAD_SEND_AND_EXIT];
  }

  if (!product)
    product = [parameters objectForKey:@"CFBundleName"];

  if (!display) {
    display = [parameters objectForKey:@"CFBundleDisplayName"];
    if (!display) {
      display = product;
    }
  }

  if (!version)
    version = [parameters objectForKey:@"CFBundleVersion"];

  if (!interval)
    interval = @"3600";

  if (!timeout)
    timeout = @"300";

  if (!logFileTailSize)
    logFileTailSize = @"200000";

  if (!vendor) {
    vendor = @"Vendor not specified";
  }

  // Normalize the values.
  if (skipConfirm) {
    skipConfirm = [skipConfirm uppercaseString];

    if ([skipConfirm isEqualToString:@"YES"] ||
        [skipConfirm isEqualToString:@"TRUE"] ||
        [skipConfirm isEqualToString:@"1"])
      skipConfirm = @"YES";
    else
      skipConfirm = @"NO";
  } else {
    skipConfirm = @"NO";
  }

  send_and_exit_ = true;
  if (sendAndExit) {
    sendAndExit = [sendAndExit uppercaseString];

    if ([sendAndExit isEqualToString:@"NO"] ||
        [sendAndExit isEqualToString:@"FALSE"] ||
        [sendAndExit isEqualToString:@"0"])
      send_and_exit_ = false;
  }

  if (requestUserText) {
    requestUserText = [requestUserText uppercaseString];

    if ([requestUserText isEqualToString:@"YES"] ||
        [requestUserText isEqualToString:@"TRUE"] ||
        [requestUserText isEqualToString:@"1"])
      requestUserText = @"YES";
    else
      requestUserText = @"NO";
  } else {
    requestUserText = @"NO";
  }

  // Find the helper applications if not specified in user config.
  NSString* resourcePath = nil;
  if (!inspectorPathString || !reporterPathString) {
    resourcePath = GetResourcePath();
    if (!resourcePath) {
      return false;
    }
  }

  // Find Inspector.
  if (!inspectorPathString) {
    inspectorPathString =
        [resourcePath stringByAppendingPathComponent:@"Inspector"];
  }

  // Verify that there is an Inspector tool.
  if (![[NSFileManager defaultManager] fileExistsAtPath:inspectorPathString]) {
    return false;
  }

  // Find Reporter.
  if (!reporterPathString) {
    reporterPathString =
        [resourcePath
         stringByAppendingPathComponent:@"crash_report_sender.app"];
    reporterPathString =
        [[NSBundle bundleWithPath:reporterPathString] executablePath];
  }

  // Verify that there is a Reporter application.
  if (![[NSFileManager defaultManager]
             fileExistsAtPath:reporterPathString]) {
    return false;
  }

  if (!dumpSubdirectory) {
    dumpSubdirectory = @"";
  }

  // The product, version, and URL are required values.
  if (![product length]) {
    return false;
  }

  if (![version length]) {
    return false;
  }

  if (![urlStr length]) {
    return false;
  }

  config_params_ =
      new (gKeyValueAllocator->Allocate(sizeof(SimpleStringDictionary)) )
        SimpleStringDictionary();

  SimpleStringDictionary& dictionary = *config_params_;

  dictionary.SetKeyValue(BREAKPAD_SERVER_TYPE,     [serverType UTF8String]);
  dictionary.SetKeyValue(BREAKPAD_PRODUCT_DISPLAY, [display UTF8String]);
  dictionary.SetKeyValue(BREAKPAD_PRODUCT,         [product UTF8String]);
  dictionary.SetKeyValue(BREAKPAD_VERSION,         [version UTF8String]);
  dictionary.SetKeyValue(BREAKPAD_URL,             [urlStr UTF8String]);
  dictionary.SetKeyValue(BREAKPAD_REPORT_INTERVAL, [interval UTF8String]);
  dictionary.SetKeyValue(BREAKPAD_SKIP_CONFIRM,    [skipConfirm UTF8String]);
  dictionary.SetKeyValue(BREAKPAD_CONFIRM_TIMEOUT, [timeout UTF8String]);
  dictionary.SetKeyValue(BREAKPAD_INSPECTOR_LOCATION,
                         [inspectorPathString fileSystemRepresentation]);
  dictionary.SetKeyValue(BREAKPAD_REPORTER_EXE_LOCATION,
                         [reporterPathString fileSystemRepresentation]);
  dictionary.SetKeyValue(BREAKPAD_LOGFILE_UPLOAD_SIZE,
                         [logFileTailSize UTF8String]);
  dictionary.SetKeyValue(BREAKPAD_REQUEST_COMMENTS,
                         [requestUserText UTF8String]);
  dictionary.SetKeyValue(BREAKPAD_REQUEST_EMAIL, [requestEmail UTF8String]);
  dictionary.SetKeyValue(BREAKPAD_VENDOR, [vendor UTF8String]);
  dictionary.SetKeyValue(BREAKPAD_DUMP_DIRECTORY,
                         [dumpSubdirectory UTF8String]);

  struct timeval tv;
  gettimeofday(&tv, NULL);
  char timeStartedString[32];
  sprintf(timeStartedString, "%zd", tv.tv_sec);
  dictionary.SetKeyValue(BREAKPAD_PROCESS_START_TIME,
                         timeStartedString);

  if (logFilePaths) {
    char logFileKey[255];
    for(unsigned int i = 0; i < [logFilePaths count]; i++) {
      sprintf(logFileKey,"%s%d", BREAKPAD_LOGFILE_KEY_PREFIX, i);
      dictionary.SetKeyValue(logFileKey,
                             [[logFilePaths objectAtIndex:i]
                               fileSystemRepresentation]);
    }
  }

  if (serverParameters) {
    // For each key-value pair, call BreakpadAddUploadParameter()
    NSEnumerator* keyEnumerator = [serverParameters keyEnumerator];
    NSString* aParameter;
    while ((aParameter = [keyEnumerator nextObject])) {
      BreakpadAddUploadParameter(this, aParameter,
				 [serverParameters objectForKey:aParameter]);
    }
  }
  return true;
}

//=============================================================================
void Breakpad::SetKeyValue(NSString* key, NSString* value) {
  // We allow nil values. This is the same as removing the keyvalue.
  if (!config_params_ || !key)
    return;

  config_params_->SetKeyValue([key UTF8String], [value UTF8String]);
}

//=============================================================================
NSString* Breakpad::KeyValue(NSString* key) {
  if (!config_params_ || !key)
    return nil;

  const char* value = config_params_->GetValueForKey([key UTF8String]);
  return value ? [NSString stringWithUTF8String:value] : nil;
}

//=============================================================================
void Breakpad::RemoveKeyValue(NSString* key) {
  if (!config_params_ || !key) return;

  config_params_->RemoveKey([key UTF8String]);
}

//=============================================================================
void Breakpad::GenerateAndSendReport() {
  config_params_->SetKeyValue(BREAKPAD_ON_DEMAND, "YES");
  HandleException(0, 0, 0, mach_thread_self());
  config_params_->SetKeyValue(BREAKPAD_ON_DEMAND, "NO");
}

//=============================================================================
bool Breakpad::HandleException(int exception_type,
                               int exception_code,
                               int exception_subcode,
                               mach_port_t crashing_thread) {
  if (filter_callback_) {
    bool should_handle = filter_callback_(exception_type,
                                          exception_code,
                                          crashing_thread,
                                          filter_callback_context_);
    if (!should_handle) return false;
  }

  // We need to reset the memory protections to be read/write,
  // since LaunchOnDemand() requires changing state.
  gBreakpadAllocator->Unprotect();
  // Configure the server to launch when we message the service port.
  // The reason we do this here, rather than at startup, is that we
  // can leak a bootstrap service entry if this method is called and
  // there never ends up being a crash.
  inspector_.LaunchOnDemand();
  gBreakpadAllocator->Protect();

  // The Inspector should send a message to this port to verify it
  // received our information and has finished the inspection.
  ReceivePort acknowledge_port;

  // Send initial information to the Inspector.
  MachSendMessage message(kMsgType_InspectorInitialInfo);
  message.AddDescriptor(mach_task_self());          // our task
  message.AddDescriptor(crashing_thread);           // crashing thread
  message.AddDescriptor(mach_thread_self());        // exception-handling thread
  message.AddDescriptor(acknowledge_port.GetPort());// message receive port

  InspectorInfo info;
  info.exception_type = exception_type;
  info.exception_code = exception_code;
  info.exception_subcode = exception_subcode;
  info.parameter_count = config_params_->GetCount();
  message.SetData(&info, sizeof(info));

  MachPortSender sender(inspector_.GetServicePort());

  kern_return_t result = sender.SendMessage(message, 2000);

  if (result == KERN_SUCCESS) {
    // Now, send a series of key-value pairs to the Inspector.
    const SimpleStringDictionary::Entry* entry = NULL;
    SimpleStringDictionary::Iterator iter(*config_params_);

    while ( (entry = iter.Next()) ) {
      KeyValueMessageData keyvalue_data(*entry);

      MachSendMessage keyvalue_message(kMsgType_InspectorKeyValuePair);
      keyvalue_message.SetData(&keyvalue_data, sizeof(keyvalue_data));

      result = sender.SendMessage(keyvalue_message, 2000);

      if (result != KERN_SUCCESS) {
        break;
      }
    }

    if (result == KERN_SUCCESS) {
      // Wait for acknowledgement that the inspection has finished.
      MachReceiveMessage acknowledge_messsage;
      result = acknowledge_port.WaitForMessage(&acknowledge_messsage, 5000);
    }
  }

#if VERBOSE
  PRINT_MACH_RESULT(result, "Breakpad: SendMessage ");
  printf("Breakpad: Inspector service port = %#x\n",
    inspector_.GetServicePort());
#endif

  // If we don't want any forwarding, return true here to indicate that we've
  // processed things as much as we want.
  if (send_and_exit_) return true;

  return false;
}

//=============================================================================
bool Breakpad::HandleMinidump(const char* dump_dir, const char* minidump_id) {
  google_breakpad::ConfigFile config_file;
  config_file.WriteFile(dump_dir, config_params_, dump_dir, minidump_id);
  google_breakpad::LaunchReporter(
      config_params_->GetValueForKey(BREAKPAD_REPORTER_EXE_LOCATION),
      config_file.GetFilePath());
  return true;
}

//=============================================================================
//=============================================================================

#pragma mark -
#pragma mark Public API

//=============================================================================
BreakpadRef BreakpadCreate(NSDictionary* parameters) {
  try {
    // This is confusing.  Our two main allocators for breakpad memory are:
    //    - gKeyValueAllocator for the key/value memory
    //    - gBreakpadAllocator for the Breakpad, ExceptionHandler, and other
    //      breakpad allocations which are accessed at exception handling time.
    //
    // But in order to avoid these two allocators themselves from being smashed,
    // we'll protect them as well by allocating them with gMasterAllocator.
    //
    // gMasterAllocator itself will NOT be protected, but this doesn't matter,
    // since once it does its allocations and locks the memory, smashes to itself
    // don't affect anything we care about.
    gMasterAllocator =
        new ProtectedMemoryAllocator(sizeof(ProtectedMemoryAllocator) * 2);

    gKeyValueAllocator =
        new (gMasterAllocator->Allocate(sizeof(ProtectedMemoryAllocator)))
            ProtectedMemoryAllocator(sizeof(SimpleStringDictionary));

    // Create a mutex for use in accessing the SimpleStringDictionary
    int mutexResult = pthread_mutex_init(&gDictionaryMutex, NULL);
    if (mutexResult == 0) {

      // With the current compiler, gBreakpadAllocator is allocating 1444 bytes.
      // Let's round up to the nearest page size.
      //
      int breakpad_pool_size = 4096;

      /*
       sizeof(Breakpad)
       + sizeof(google_breakpad::ExceptionHandler)
       + sizeof( STUFF ALLOCATED INSIDE ExceptionHandler )
       */

      gBreakpadAllocator =
          new (gMasterAllocator->Allocate(sizeof(ProtectedMemoryAllocator)))
              ProtectedMemoryAllocator(breakpad_pool_size);

      // Stack-based autorelease pool for Breakpad::Create() obj-c code.
      NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
      Breakpad* breakpad = Breakpad::Create(parameters);

      if (breakpad) {
        // Make read-only to protect against memory smashers
        gMasterAllocator->Protect();
        gKeyValueAllocator->Protect();
        gBreakpadAllocator->Protect();
        // Can uncomment this line to figure out how much space was actually
        // allocated using this allocator
        //     printf("gBreakpadAllocator allocated size = %d\n",
        //         gBreakpadAllocator->GetAllocatedSize() );
        [pool release];
        return (BreakpadRef)breakpad;
      }

      [pool release];
    }
  } catch(...) {    // don't let exceptions leave this C API
    fprintf(stderr, "BreakpadCreate() : error\n");
  }

  if (gKeyValueAllocator) {
    gKeyValueAllocator->~ProtectedMemoryAllocator();
    gKeyValueAllocator = NULL;
  }

  if (gBreakpadAllocator) {
    gBreakpadAllocator->~ProtectedMemoryAllocator();
    gBreakpadAllocator = NULL;
  }

  delete gMasterAllocator;
  gMasterAllocator = NULL;

  return NULL;
}

//=============================================================================
void BreakpadRelease(BreakpadRef ref) {
  try {
    Breakpad* breakpad = (Breakpad*)ref;

    if (gMasterAllocator) {
      gMasterAllocator->Unprotect();
      gKeyValueAllocator->Unprotect();
      gBreakpadAllocator->Unprotect();

      breakpad->~Breakpad();

      // Unfortunately, it's not possible to deallocate this stuff
      // because the exception handling thread is still finishing up
      // asynchronously at this point...  OK, it could be done with
      // locks, etc.  But since BreakpadRelease() should usually only
      // be called right before the process exits, it's not worth
      // deallocating this stuff.
#if 0
      gKeyValueAllocator->~ProtectedMemoryAllocator();
      gBreakpadAllocator->~ProtectedMemoryAllocator();
      delete gMasterAllocator;

      gMasterAllocator = NULL;
      gKeyValueAllocator = NULL;
      gBreakpadAllocator = NULL;
#endif

      pthread_mutex_destroy(&gDictionaryMutex);
    }
  } catch(...) {    // don't let exceptions leave this C API
    fprintf(stderr, "BreakpadRelease() : error\n");
  }
}

//=============================================================================
void BreakpadSetKeyValue(BreakpadRef ref, NSString* key, NSString* value) {
  try {
    // Not called at exception time
    Breakpad* breakpad = (Breakpad*)ref;

    if (breakpad && key && gKeyValueAllocator) {
      ProtectedMemoryLocker locker(&gDictionaryMutex, gKeyValueAllocator);

      breakpad->SetKeyValue(key, value);
    }
  } catch(...) {    // don't let exceptions leave this C API
    fprintf(stderr, "BreakpadSetKeyValue() : error\n");
  }
}

void BreakpadAddUploadParameter(BreakpadRef ref,
                                NSString* key,
                                NSString* value) {
  // The only difference, internally, between an upload parameter and
  // a key value one that is set with BreakpadSetKeyValue is that we
  // prepend the keyname with a special prefix.  This informs the
  // crash sender that the parameter should be sent along with the
  // POST of the crash dump upload.
  try {
    Breakpad* breakpad = (Breakpad*)ref;

    if (breakpad && key && gKeyValueAllocator) {
      ProtectedMemoryLocker locker(&gDictionaryMutex, gKeyValueAllocator);

      NSString* prefixedKey = [@BREAKPAD_SERVER_PARAMETER_PREFIX
				stringByAppendingString:key];
      breakpad->SetKeyValue(prefixedKey, value);
    }
  } catch(...) {    // don't let exceptions leave this C API
    fprintf(stderr, "BreakpadSetKeyValue() : error\n");
  }
}

void BreakpadRemoveUploadParameter(BreakpadRef ref,
                                   NSString* key) {
  try {
    // Not called at exception time
    Breakpad* breakpad = (Breakpad*)ref;

    if (breakpad && key && gKeyValueAllocator) {
      ProtectedMemoryLocker locker(&gDictionaryMutex, gKeyValueAllocator);

      NSString* prefixedKey = [NSString stringWithFormat:@"%@%@",
                                        @BREAKPAD_SERVER_PARAMETER_PREFIX, key];
      breakpad->RemoveKeyValue(prefixedKey);
    }
  } catch(...) {    // don't let exceptions leave this C API
    fprintf(stderr, "BreakpadRemoveKeyValue() : error\n");
  }
}
//=============================================================================
NSString* BreakpadKeyValue(BreakpadRef ref, NSString* key) {
  NSString* value = nil;

  try {
    // Not called at exception time
    Breakpad* breakpad = (Breakpad*)ref;

    if (!breakpad || !key || !gKeyValueAllocator)
      return nil;

    ProtectedMemoryLocker locker(&gDictionaryMutex, gKeyValueAllocator);

    value = breakpad->KeyValue(key);
  } catch(...) {    // don't let exceptions leave this C API
    fprintf(stderr, "BreakpadKeyValue() : error\n");
  }

  return value;
}

//=============================================================================
void BreakpadRemoveKeyValue(BreakpadRef ref, NSString* key) {
  try {
    // Not called at exception time
    Breakpad* breakpad = (Breakpad*)ref;

    if (breakpad && key && gKeyValueAllocator) {
      ProtectedMemoryLocker locker(&gDictionaryMutex, gKeyValueAllocator);

      breakpad->RemoveKeyValue(key);
    }
  } catch(...) {    // don't let exceptions leave this C API
    fprintf(stderr, "BreakpadRemoveKeyValue() : error\n");
  }
}

//=============================================================================
void BreakpadGenerateAndSendReport(BreakpadRef ref) {
  try {
    Breakpad* breakpad = (Breakpad*)ref;

    if (breakpad && gKeyValueAllocator) {
      ProtectedMemoryLocker locker(&gDictionaryMutex, gKeyValueAllocator);

      gBreakpadAllocator->Unprotect();
      breakpad->GenerateAndSendReport();
      gBreakpadAllocator->Protect();
    }
  } catch(...) {    // don't let exceptions leave this C API
    fprintf(stderr, "BreakpadGenerateAndSendReport() : error\n");
  }
}

//=============================================================================
void BreakpadSetFilterCallback(BreakpadRef ref,
                               BreakpadFilterCallback callback,
                               void* context) {

  try {
    Breakpad* breakpad = (Breakpad*)ref;

    if (breakpad && gBreakpadAllocator) {
      // share the dictionary mutex here (we really don't need a mutex)
      ProtectedMemoryLocker locker(&gDictionaryMutex, gBreakpadAllocator);

      breakpad->SetFilterCallback(callback, context);
    }
  } catch(...) {    // don't let exceptions leave this C API
    fprintf(stderr, "BreakpadSetFilterCallback() : error\n");
  }
}

//============================================================================
void BreakpadAddLogFile(BreakpadRef ref, NSString* logPathname) {
  int logFileCounter = 0;

  NSString* logFileKey = [NSString stringWithFormat:@"%@%d",
                                   @BREAKPAD_LOGFILE_KEY_PREFIX,
                                   logFileCounter];

  NSString* existingLogFilename = nil;
  existingLogFilename = BreakpadKeyValue(ref, logFileKey);
  // Find the first log file key that we can use by testing for existence
  while (existingLogFilename) {
    if ([existingLogFilename isEqualToString:logPathname]) {
      return;
    }
    logFileCounter++;
    logFileKey = [NSString stringWithFormat:@"%@%d",
                           @BREAKPAD_LOGFILE_KEY_PREFIX,
                           logFileCounter];
    existingLogFilename = BreakpadKeyValue(ref, logFileKey);
  }

  BreakpadSetKeyValue(ref, logFileKey, logPathname);
}
