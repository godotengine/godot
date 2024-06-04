//
//  GTMLogger.h
//
//  Copyright 2007-2008 Google Inc.
//
//  Licensed under the Apache License, Version 2.0 (the "License"); you may not
//  use this file except in compliance with the License.  You may obtain a copy
//  of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
//  WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
//  License for the specific language governing permissions and limitations under
//  the License.
//

// Key Abstractions
// ----------------
//
// This file declares multiple classes and protocols that are used by the
// GTMLogger logging system. The 4 main abstractions used in this file are the
// following:
//
//   * logger (GTMLogger) - The main logging class that users interact with. It
//   has methods for logging at different levels and uses a log writer, a log
//   formatter, and a log filter to get the job done.
//
//   * log writer (GTMLogWriter) - Writes a given string to some log file, where
//   a "log file" can be a physical file on disk, a POST over HTTP to some URL,
//   or even some in-memory structure (e.g., a ring buffer).
//
//   * log formatter (GTMLogFormatter) - Given a format string and arguments as
//   a va_list, returns a single formatted NSString. A "formatted string" could
//   be a string with the date prepended, a string with values in a CSV format,
//   or even a string of XML.
//
//   * log filter (GTMLogFilter) - Given a formatted log message as an NSString
//   and the level at which the message is to be logged, this class will decide
//   whether the given message should be logged or not. This is a flexible way
//   to filter out messages logged at a certain level, messages that contain
//   certain text, or filter nothing out at all. This gives the caller the
//   flexibility to dynamically enable debug logging in Release builds.
//
// This file also declares some classes to handle the common log writer, log
// formatter, and log filter cases. Callers can also create their own writers,
// formatters, and filters and they can even build them on top of the ones
// declared here. Keep in mind that your custom writer/formatter/filter may be
// called from multiple threads, so it must be thread-safe.

#import <Foundation/Foundation.h>
#import "GTMDefines.h"

// Predeclaration of used protocols that are declared later in this file.
@protocol GTMLogWriter, GTMLogFormatter, GTMLogFilter;

// GTMLogger
//
// GTMLogger is the primary user-facing class for an object-oriented logging
// system. It is built on the concept of log formatters (GTMLogFormatter), log
// writers (GTMLogWriter), and log filters (GTMLogFilter). When a message is
// sent to a GTMLogger to log a message, the message is formatted using the log
// formatter, then the log filter is consulted to see if the message should be
// logged, and if so, the message is sent to the log writer to be written out.
//
// GTMLogger is intended to be a flexible and thread-safe logging solution. Its
// flexibility comes from the fact that GTMLogger instances can be customized
// with user defined formatters, filters, and writers. And these writers,
// filters, and formatters can be combined, stacked, and customized in arbitrary
// ways to suit the needs at hand. For example, multiple writers can be used at
// the same time, and a GTMLogger instance can even be used as another
// GTMLogger's writer. This allows for arbitrarily deep logging trees.
//
// A standard GTMLogger uses a writer that sends messages to standard out, a
// formatter that smacks a timestamp and a few other bits of interesting
// information on the message, and a filter that filters out debug messages from
// release builds. Using the standard log settings, a log message will look like
// the following:
//
//   2007-12-30 10:29:24.177 myapp[4588/0xa07d0f60] [lvl=1] foo=<Foo: 0x123>
//
// The output contains the date and time of the log message, the name of the
// process followed by its process ID/thread ID, the log level at which the
// message was logged (in the previous example the level was 1:
// kGTMLoggerLevelDebug), and finally, the user-specified log message itself (in
// this case, the log message was @"foo=%@", foo).
//
// Multiple instances of GTMLogger can be created, each configured their own
// way.  Though GTMLogger is not a singleton (in the GoF sense), it does provide
// access to a shared (i.e., globally accessible) GTMLogger instance. This makes
// it convenient for all code in a process to use the same GTMLogger instance.
// The shared GTMLogger instance can also be configured in an arbitrary, and
// these configuration changes will affect all code that logs through the shared
// instance.

//
// Log Levels
// ----------
// GTMLogger has 3 different log levels: Debug, Info, and Error. GTMLogger
// doesn't take any special action based on the log level; it simply forwards
// this information on to formatters, filters, and writers, each of which may
// optionally take action based on the level. Since log level filtering is
// performed at runtime, log messages are typically not filtered out at compile
// time.  The exception to this rule is that calls to the GTMLoggerDebug() macro
// *ARE* filtered out of non-DEBUG builds. This is to be backwards compatible
// with behavior that many developers are currently used to. Note that this
// means that GTMLoggerDebug(@"hi") will be compiled out of Release builds, but
// [[GTMLogger sharedLogger] logDebug:@"hi"] will NOT be compiled out.
//
// Standard loggers are created with the GTMLogLevelFilter log filter, which
// filters out certain log messages based on log level, and some other settings.
//
// In addition to the -logDebug:, -logInfo:, and -logError: methods defined on
// GTMLogger itself, there are also C macros that make usage of the shared
// GTMLogger instance very convenient. These macros are:
//
//   GTMLoggerDebug(...)
//   GTMLoggerInfo(...)
//   GTMLoggerError(...)
//
// Again, a notable feature of these macros is that GTMLogDebug() calls *will be
// compiled out of non-DEBUG builds*.
//
// Standard Loggers
// ----------------
// GTMLogger has the concept of "standard loggers". A standard logger is simply
// a logger that is pre-configured with some standard/common writer, formatter,
// and filter combination. Standard loggers are created using the creation
// methods beginning with "standard". The alternative to a standard logger is a
// regular logger, which will send messages to stdout, with no special
// formatting, and no filtering.
//
// How do I use GTMLogger?
// ----------------------
// The typical way you will want to use GTMLogger is to simply use the
// GTMLogger*() macros for logging from code. That way we can easily make
// changes to the GTMLogger class and simply update the macros accordingly. Only
// your application startup code (perhaps, somewhere in main()) should use the
// GTMLogger class directly in order to configure the shared logger, which all
// of the code using the macros will be using. Again, this is just the typical
// situation.
//
// To be complete, there are cases where you may want to use GTMLogger directly,
// or even create separate GTMLogger instances for some reason. That's fine,
// too.
//
// Examples
// --------
// The following show some common GTMLogger use cases.
//
// 1. You want to log something as simply as possible. Also, this call will only
//    appear in debug builds. In non-DEBUG builds it will be completely removed.
//
//      GTMLoggerDebug(@"foo = %@", foo);
//
// 2. The previous example is similar to the following. The major difference is
//    that the previous call (example 1) will be compiled out of Release builds
//    but this statement will not be compiled out.
//
//      [[GTMLogger sharedLogger] logDebug:@"foo = %@", foo];
//
// 3. Send all logging output from the shared logger to a file. We do this by
//    creating an NSFileHandle for writing associated with a file, and setting
//    that file handle as the logger's writer.
//
//      NSFileHandle *f = [NSFileHandle fileHandleForWritingAtPath:@"/tmp/f.log"
//                                                          create:YES];
//      [[GTMLogger sharedLogger] setWriter:f];
//      GTMLoggerError(@"hi");  // This will be sent to /tmp/f.log
//
// 4. Create a new GTMLogger that will log to a file. This example differs from
//    the previous one because here we create a new GTMLogger that is different
//    from the shared logger.
//
//      GTMLogger *logger = [GTMLogger standardLoggerWithPath:@"/tmp/temp.log"];
//      [logger logInfo:@"hi temp log file"];
//
// 5. Create a logger that writes to stdout and does NOT do any formatting to
//    the log message. This might be useful, for example, when writing a help
//    screen for a command-line tool to standard output.
//
//      GTMLogger *logger = [GTMLogger logger];
//      [logger logInfo:@"%@ version 0.1 usage", progName];
//
// 6. Send log output to stdout AND to a log file. The trick here is that
//    NSArrays function as composite log writers, which means when an array is
//    set as the log writer, it forwards all logging messages to all of its
//    contained GTMLogWriters.
//
//      // Create array of GTMLogWriters
//      NSArray *writers = [NSArray arrayWithObjects:
//          [NSFileHandle fileHandleForWritingAtPath:@"/tmp/f.log" create:YES],
//          [NSFileHandle fileHandleWithStandardOutput], nil];
//
//      GTMLogger *logger = [GTMLogger standardLogger];
//      [logger setWriter:writers];
//      [logger logInfo:@"hi"];  // Output goes to stdout and /tmp/f.log
//
// For futher details on log writers, formatters, and filters, see the
// documentation below.
//
// NOTE: GTMLogger is application level logging.  By default it does nothing
// with _GTMDevLog/_GTMDevAssert (see GTMDefines.h).  An application can choose
// to bridge _GTMDevLog/_GTMDevAssert to GTMLogger by providing macro
// definitions in its prefix header (see GTMDefines.h for how one would do
// that).
//
@interface GTMLogger : NSObject {
 @private
  id<GTMLogWriter> writer_;
  id<GTMLogFormatter> formatter_;
  id<GTMLogFilter> filter_;
}

//
// Accessors for the shared logger instance
//

// Returns a shared/global standard GTMLogger instance. Callers should typically
// use this method to get a GTMLogger instance, unless they explicitly want
// their own instance to configure for their own needs. This is the only method
// that returns a shared instance; all the rest return new GTMLogger instances.
+ (id)sharedLogger;

// Sets the shared logger instance to |logger|. Future calls to +sharedLogger
// will return |logger| instead.
+ (void)setSharedLogger:(GTMLogger *)logger;

//
// Creation methods
//

// Returns a new autoreleased GTMLogger instance that will log to stdout, using
// the GTMLogStandardFormatter, and the GTMLogLevelFilter filter.
+ (id)standardLogger;

// Same as +standardLogger, but logs to stderr.
+ (id)standardLoggerWithStderr;

// Same as +standardLogger but levels >= kGTMLoggerLevelError are routed to
// stderr, everything else goes to stdout.
+ (id)standardLoggerWithStdoutAndStderr;

// Returns a new standard GTMLogger instance with a log writer that will
// write to the file at |path|, and will use the GTMLogStandardFormatter and
// GTMLogLevelFilter classes. If |path| does not exist, it will be created.
+ (id)standardLoggerWithPath:(NSString *)path;

// Returns an autoreleased GTMLogger instance that will use the specified
// |writer|, |formatter|, and |filter|.
+ (id)loggerWithWriter:(id<GTMLogWriter>)writer
             formatter:(id<GTMLogFormatter>)formatter
                filter:(id<GTMLogFilter>)filter;

// Returns an autoreleased GTMLogger instance that logs to stdout, with the
// basic formatter, and no filter. The returned logger differs from the logger
// returned by +standardLogger because this one does not do any filtering and
// does not do any special log formatting; this is the difference between a
// "regular" logger and a "standard" logger.
+ (id)logger;

// Designated initializer. This method returns a GTMLogger initialized with the
// specified |writer|, |formatter|, and |filter|. See the setter methods below
// for what values will be used if nil is passed for a parameter.
- (id)initWithWriter:(id<GTMLogWriter>)writer
           formatter:(id<GTMLogFormatter>)formatter
              filter:(id<GTMLogFilter>)filter;

//
// Logging  methods
//

// Logs a message at the debug level (kGTMLoggerLevelDebug).
- (void)logDebug:(NSString *)fmt, ... NS_FORMAT_FUNCTION(1, 2);
// Logs a message at the info level (kGTMLoggerLevelInfo).
- (void)logInfo:(NSString *)fmt, ... NS_FORMAT_FUNCTION(1, 2);
// Logs a message at the error level (kGTMLoggerLevelError).
- (void)logError:(NSString *)fmt, ... NS_FORMAT_FUNCTION(1, 2);
// Logs a message at the assert level (kGTMLoggerLevelAssert).
- (void)logAssert:(NSString *)fmt, ... NS_FORMAT_FUNCTION(1, 2);


//
// Accessors
//

// Accessor methods for the log writer. If the log writer is set to nil,
// [NSFileHandle fileHandleWithStandardOutput] is used.
- (id<GTMLogWriter>)writer;
- (void)setWriter:(id<GTMLogWriter>)writer;

// Accessor methods for the log formatter. If the log formatter is set to nil,
// GTMLogBasicFormatter is used. This formatter will format log messages in a
// plain printf style.
- (id<GTMLogFormatter>)formatter;
- (void)setFormatter:(id<GTMLogFormatter>)formatter;

// Accessor methods for the log filter. If the log filter is set to nil,
// GTMLogNoFilter is used, which allows all log messages through.
- (id<GTMLogFilter>)filter;
- (void)setFilter:(id<GTMLogFilter>)filter;

@end  // GTMLogger


// Helper functions that are used by the convenience GTMLogger*() macros that
// enable the logging of function names.
@interface GTMLogger (GTMLoggerMacroHelpers)
- (void)logFuncDebug:(const char *)func msg:(NSString *)fmt, ...
  NS_FORMAT_FUNCTION(2, 3);
- (void)logFuncInfo:(const char *)func msg:(NSString *)fmt, ...
  NS_FORMAT_FUNCTION(2, 3);
- (void)logFuncError:(const char *)func msg:(NSString *)fmt, ...
  NS_FORMAT_FUNCTION(2, 3);
- (void)logFuncAssert:(const char *)func msg:(NSString *)fmt, ...
  NS_FORMAT_FUNCTION(2, 3);
@end  // GTMLoggerMacroHelpers


// The convenience macros are only defined if they haven't already been defined.
#ifndef GTMLoggerInfo

// Convenience macros that log to the shared GTMLogger instance. These macros
// are how users should typically log to GTMLogger. Notice that GTMLoggerDebug()
// calls will be compiled out of non-Debug builds.
#define GTMLoggerDebug(...)  \
  [[GTMLogger sharedLogger] logFuncDebug:__func__ msg:__VA_ARGS__]
#define GTMLoggerInfo(...)   \
  [[GTMLogger sharedLogger] logFuncInfo:__func__ msg:__VA_ARGS__]
#define GTMLoggerError(...)  \
  [[GTMLogger sharedLogger] logFuncError:__func__ msg:__VA_ARGS__]
#define GTMLoggerAssert(...) \
  [[GTMLogger sharedLogger] logFuncAssert:__func__ msg:__VA_ARGS__]

// If we're not in a debug build, remove the GTMLoggerDebug statements. This
// makes calls to GTMLoggerDebug "compile out" of Release builds
#ifndef DEBUG
#undef GTMLoggerDebug
#define GTMLoggerDebug(...) do {} while(0)
#endif

#endif  // !defined(GTMLoggerInfo)

// Log levels.
typedef enum {
  kGTMLoggerLevelUnknown,
  kGTMLoggerLevelDebug,
  kGTMLoggerLevelInfo,
  kGTMLoggerLevelError,
  kGTMLoggerLevelAssert,
} GTMLoggerLevel;


//
//   Log Writers
//

// Protocol to be implemented by a GTMLogWriter instance.
@protocol GTMLogWriter <NSObject>
// Writes the given log message to where the log writer is configured to write.
- (void)logMessage:(NSString *)msg level:(GTMLoggerLevel)level;
@end  // GTMLogWriter


// Simple category on NSFileHandle that makes NSFileHandles valid log writers.
// This is convenient because something like, say, +fileHandleWithStandardError
// now becomes a valid log writer. Log messages are written to the file handle
// with a newline appended.
@interface NSFileHandle (GTMFileHandleLogWriter) <GTMLogWriter>
// Opens the file at |path| in append mode, and creates the file with |mode|
// if it didn't previously exist.
+ (id)fileHandleForLoggingAtPath:(NSString *)path mode:(mode_t)mode;
@end  // NSFileHandle


// This category makes NSArray a GTMLogWriter that can be composed of other
// GTMLogWriters. This is the classic Composite GoF design pattern. When the
// GTMLogWriter -logMessage:level: message is sent to the array, the array
// forwards the message to all of its elements that implement the GTMLogWriter
// protocol.
//
// This is useful in situations where you would like to send log output to
// multiple log writers at the same time. Simply create an NSArray of the log
// writers you wish to use, then set the array as the "writer" for your
// GTMLogger instance.
@interface NSArray (GTMArrayCompositeLogWriter) <GTMLogWriter>
@end  // GTMArrayCompositeLogWriter


// This category adapts the GTMLogger interface so that it can be used as a log
// writer; it's an "adapter" in the GoF Adapter pattern sense.
//
// This is useful when you want to configure a logger to log to a specific
// writer with a specific formatter and/or filter. But you want to also compose
// that with a different log writer that may have its own formatter and/or
// filter.
@interface GTMLogger (GTMLoggerLogWriter) <GTMLogWriter>
@end  // GTMLoggerLogWriter


//
//   Log Formatters
//

// Protocol to be implemented by a GTMLogFormatter instance.
@protocol GTMLogFormatter <NSObject>
// Returns a formatted string using the format specified in |fmt| and the va
// args specified in |args|.
- (NSString *)stringForFunc:(NSString *)func
                 withFormat:(NSString *)fmt
                     valist:(va_list)args
                      level:(GTMLoggerLevel)level NS_FORMAT_FUNCTION(2, 0);
@end  // GTMLogFormatter


// A basic log formatter that formats a string the same way that NSLog (or
// printf) would. It does not do anything fancy, nor does it add any data of its
// own.
@interface GTMLogBasicFormatter : NSObject <GTMLogFormatter>

// Helper method for prettying C99 __func__ and GCC __PRETTY_FUNCTION__
- (NSString *)prettyNameForFunc:(NSString *)func;

@end  // GTMLogBasicFormatter


// A log formatter that formats the log string like the basic formatter, but
// also prepends a timestamp and some basic process info to the message, as
// shown in the following sample output.
//   2007-12-30 10:29:24.177 myapp[4588/0xa07d0f60] [lvl=1] log mesage here
@interface GTMLogStandardFormatter : GTMLogBasicFormatter {
 @private
  NSDateFormatter *dateFormatter_;  // yyyy-MM-dd HH:mm:ss.SSS
  NSString *pname_;
  pid_t pid_;
}
@end  // GTMLogStandardFormatter


//
//   Log Filters
//

// Protocol to be imlemented by a GTMLogFilter instance.
@protocol GTMLogFilter <NSObject>
// Returns YES if |msg| at |level| should be filtered out; NO otherwise.
- (BOOL)filterAllowsMessage:(NSString *)msg level:(GTMLoggerLevel)level;
@end  // GTMLogFilter


// A log filter that filters messages at the kGTMLoggerLevelDebug level out of
// non-debug builds. Messages at the kGTMLoggerLevelInfo level are also filtered
// out of non-debug builds unless GTMVerboseLogging is set in the environment or
// the processes's defaults. Messages at the kGTMLoggerLevelError level are
// never filtered.
@interface GTMLogLevelFilter : NSObject <GTMLogFilter>
@end  // GTMLogLevelFilter

// A simple log filter that does NOT filter anything out;
// -filterAllowsMessage:level will always return YES. This can be a convenient
// way to enable debug-level logging in release builds (if you so desire).
@interface GTMLogNoFilter : NSObject <GTMLogFilter>
@end  // GTMLogNoFilter


// Base class for custom level filters. Not for direct use, use the minimum
// or maximum level subclasses below.
@interface GTMLogAllowedLevelFilter : NSObject <GTMLogFilter> {
 @private
  NSIndexSet *allowedLevels_;
}
@end

// A log filter that allows you to set a minimum log level. Messages below this
// level will be filtered.
@interface GTMLogMininumLevelFilter : GTMLogAllowedLevelFilter

// Designated initializer, logs at levels < |level| will be filtered.
- (id)initWithMinimumLevel:(GTMLoggerLevel)level;

@end

// A log filter that allows you to set a maximum log level. Messages whose level
// exceeds this level will be filtered. This is really only useful if you have
// a composite GTMLogger that is sending the other messages elsewhere.
@interface GTMLogMaximumLevelFilter : GTMLogAllowedLevelFilter

// Designated initializer, logs at levels > |level| will be filtered.
- (id)initWithMaximumLevel:(GTMLoggerLevel)level;

@end


// For subclasses only
@interface GTMLogger (PrivateMethods)

- (void)logInternalFunc:(const char *)func
                 format:(NSString *)fmt
                 valist:(va_list)args
                  level:(GTMLoggerLevel)level NS_FORMAT_FUNCTION(2, 0);

@end

