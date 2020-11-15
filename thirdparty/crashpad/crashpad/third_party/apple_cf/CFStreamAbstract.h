/*
 * Copyright (c) 2010 Apple Inc. All rights reserved.
 *
 * @APPLE_LICENSE_HEADER_START@
 * 
 * This file contains Original Code and/or Modifications of Original Code
 * as defined in and that are subject to the Apple Public Source License
 * Version 2.0 (the 'License'). You may not use this file except in
 * compliance with the License. Please obtain a copy of the License at
 * http://www.opensource.apple.com/apsl/ and read it before using this
 * file.
 * 
 * The Original Code and all software distributed under the License are
 * distributed on an 'AS IS' basis, WITHOUT WARRANTY OF ANY KIND, EITHER
 * EXPRESS OR IMPLIED, AND APPLE HEREBY DISCLAIMS ALL SUCH WARRANTIES,
 * INCLUDING WITHOUT LIMITATION, ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE, QUIET ENJOYMENT OR NON-INFRINGEMENT.
 * Please see the License for the specific language governing rights and
 * limitations under the License.
 * 
 * @APPLE_LICENSE_HEADER_END@
 */

/*	CFStreamAbstract.h
	Copyright (c) 2000-2009, Apple Inc. All rights reserved.
*/

#if !defined(__COREFOUNDATION_CFSTREAMABSTRACT__)
#define __COREFOUNDATION_CFSTREAMABSTRACT__ 1

#include <CoreFoundation/CFStream.h>

CF_EXTERN_C_BEGIN

/*  During a stream's lifetime, the open callback will be called once, followed by any number of openCompleted calls (until openCompleted returns TRUE).  Then any number of read/canRead or write/canWrite calls, then a single close call.  copyProperty can be called at any time.  prepareAsynch will be called exactly once when the stream's client is first configured.

    Expected semantics:
    - open reserves any system resources that are needed.  The stream may start the process of opening, returning TRUE immediately and setting openComplete to FALSE.  When the open completes, _CFStreamSignalEvent should be called passing kCFStreamOpenCompletedEvent.  openComplete should be set to TRUE only if the open operation completed in its entirety.
    - openCompleted will only be called after open has been called, but before any kCFStreamOpenCompletedEvent has been received.  Return TRUE, setting error.code to 0, if the open operation has completed.  Return TRUE, setting error to the correct error code and domain if the open operation completed, but failed.  Return FALSE if the open operation is still in-progress.  If your open ever fails to complete (i.e. sets openComplete to FALSE), you must be implement the openCompleted callback.
    - read should read into the given buffer, returning the number of bytes successfully read.  read must block until at least one byte is available, but should not block until the entire buffer is filled; zero should only be returned if end-of-stream is encountered. atEOF should be set to true if the EOF is encountered, false otherwise.  error.code should be set to zero if no error occurs; otherwise, error should be set to the appropriate values.
    - getBuffer is an optimization to return an internal buffer of bytes read from the stream, and may return NULL.  getBuffer itself may be NULL if the concrete implementation does not wish to provide an internal buffer.  If implemented, it should set numBytesRead to the number of bytes available in the internal buffer (but should not exceed maxBytesToRead) and return a pointer to the base of the bytes.
    - canRead will only be called once openCompleted reports that the stream has been successfully opened (or the initial open call succeeded).  It should return whether there are bytes that can be read without blocking.
    - write should write the bytes in the given buffer to the device, returning the number of bytes successfully written.  write must block until at least one byte is written.  error.code should be set to zero if no error occurs; otherwise, error should be set to the appropriate values.
    - close should close the device, releasing any reserved system resources.  close cannot fail (it may be called to abort the stream), and may be called at any time after open has been called.  It will only be called once.
    - copyProperty should return the value for the given property, or NULL if none exists.  Composite streams (streams built on top of other streams) should take care to call CFStreamCopyProperty on the base stream if they do not recognize the property given, to give the underlying stream a chance to respond.

    In all cases, errors returned by reference will be initialized to NULL by the caller, and if they are set to non-NULL, will
    be released by the caller 
*/
   
typedef struct {
    CFIndex version; /* == 2 */

    void *(*create)(CFReadStreamRef stream, void *info);
    void (*finalize)(CFReadStreamRef stream, void *info);
    CFStringRef (*copyDescription)(CFReadStreamRef stream, void *info);

    Boolean (*open)(CFReadStreamRef stream, CFErrorRef *error, Boolean *openComplete, void *info);
    Boolean (*openCompleted)(CFReadStreamRef stream, CFErrorRef *error, void *info);
    CFIndex (*read)(CFReadStreamRef stream, UInt8 *buffer, CFIndex bufferLength, CFErrorRef *error, Boolean *atEOF, void *info);
    const UInt8 *(*getBuffer)(CFReadStreamRef stream, CFIndex maxBytesToRead, CFIndex *numBytesRead, CFErrorRef *error, Boolean *atEOF, void *info);
    Boolean (*canRead)(CFReadStreamRef stream, CFErrorRef *error, void *info);
    void (*close)(CFReadStreamRef stream, void *info);

    CFTypeRef (*copyProperty)(CFReadStreamRef stream, CFStringRef propertyName, void *info);
    Boolean (*setProperty)(CFReadStreamRef stream, CFStringRef propertyName, CFTypeRef propertyValue, void *info);

    void (*requestEvents)(CFReadStreamRef stream, CFOptionFlags streamEvents, void *info);
    void (*schedule)(CFReadStreamRef stream, CFRunLoopRef runLoop, CFStringRef runLoopMode, void *info);
    void (*unschedule)(CFReadStreamRef stream, CFRunLoopRef runLoop, CFStringRef runLoopMode, void *info);
} CFReadStreamCallBacks;

typedef struct {
    CFIndex version; /* == 2 */

    void *(*create)(CFWriteStreamRef stream, void *info);
    void (*finalize)(CFWriteStreamRef stream, void *info);
    CFStringRef (*copyDescription)(CFWriteStreamRef stream, void *info);

    Boolean (*open)(CFWriteStreamRef stream, CFErrorRef *error, Boolean *openComplete, void *info);
    Boolean (*openCompleted)(CFWriteStreamRef stream, CFErrorRef *error, void *info);
    CFIndex (*write)(CFWriteStreamRef stream, const UInt8 *buffer, CFIndex bufferLength, CFErrorRef *error, void *info);
    Boolean (*canWrite)(CFWriteStreamRef stream, CFErrorRef *error, void *info); 
    void (*close)(CFWriteStreamRef stream, void *info);

    CFTypeRef (*copyProperty)(CFWriteStreamRef stream, CFStringRef propertyName, void *info);
    Boolean (*setProperty)(CFWriteStreamRef stream, CFStringRef propertyName, CFTypeRef propertyValue, void *info);

    void (*requestEvents)(CFWriteStreamRef stream, CFOptionFlags streamEvents, void *info);
    void (*schedule)(CFWriteStreamRef stream, CFRunLoopRef runLoop, CFStringRef runLoopMode, void *info);
    void (*unschedule)(CFWriteStreamRef stream, CFRunLoopRef runLoop, CFStringRef runLoopMode, void *info);
} CFWriteStreamCallBacks;

// Primitive creation mechanisms.
CF_EXPORT
CFReadStreamRef CFReadStreamCreate(CFAllocatorRef alloc, const CFReadStreamCallBacks *callbacks, void *info);
CF_EXPORT
CFWriteStreamRef CFWriteStreamCreate(CFAllocatorRef alloc, const CFWriteStreamCallBacks *callbacks, void *info);

/* All the functions below can only be called when you are sure the stream in question was created via
   CFReadStreamCreate() or CFWriteStreamCreate(), above.  They are NOT safe for toll-free bridged objects, 
   so the caller must be sure the argument passed is not such an object. */

// To be called by the concrete stream implementation (the callbacks) when an event occurs. error may be NULL if event != kCFStreamEventErrorOccurred
// error should be a CFErrorRef if the callbacks are version 2 or later; otherwise it should be a (CFStreamError *).
CF_EXPORT
void CFReadStreamSignalEvent(CFReadStreamRef stream, CFStreamEventType event, const void *error);
CF_EXPORT
void CFWriteStreamSignalEvent(CFWriteStreamRef stream, CFStreamEventType event, const void *error);

// These require that the stream allow the run loop to run once before delivering the event to its client.  
// See the comment above CFRead/WriteStreamSignalEvent for interpretation of the error argument.
CF_EXPORT
void _CFReadStreamSignalEventDelayed(CFReadStreamRef stream, CFStreamEventType event, const void *error);
CF_EXPORT
void _CFWriteStreamSignalEventDelayed(CFWriteStreamRef stream, CFStreamEventType event, const void *error);

CF_EXPORT
void _CFReadStreamClearEvent(CFReadStreamRef stream, CFStreamEventType event);
// Write variant not currently needed
//CF_EXPORT
//void _CFWriteStreamClearEvent(CFWriteStreamRef stream, CFStreamEventType event);

// Convenience for concrete implementations to extract the info pointer given the stream.
CF_EXPORT
void *CFReadStreamGetInfoPointer(CFReadStreamRef stream);
CF_EXPORT
void *CFWriteStreamGetInfoPointer(CFWriteStreamRef stream);

// Returns the client info pointer currently set on the stream.  These should probably be made public one day.
CF_EXPORT
void *_CFReadStreamGetClient(CFReadStreamRef readStream);
CF_EXPORT
void *_CFWriteStreamGetClient(CFWriteStreamRef writeStream);

// Returns an array of the runloops and modes on which the stream is currently scheduled
CF_EXPORT
CFArrayRef _CFReadStreamGetRunLoopsAndModes(CFReadStreamRef readStream);
CF_EXPORT
CFArrayRef _CFWriteStreamGetRunLoopsAndModes(CFWriteStreamRef writeStream);

/* Deprecated versions; here for backwards compatibility. */
typedef struct {
    CFIndex version; /* == 1 */
    void *(*create)(CFReadStreamRef stream, void *info);
    void (*finalize)(CFReadStreamRef stream, void *info);
    CFStringRef (*copyDescription)(CFReadStreamRef stream, void *info);
    Boolean (*open)(CFReadStreamRef stream, CFStreamError *error, Boolean *openComplete, void *info);
    Boolean (*openCompleted)(CFReadStreamRef stream, CFStreamError *error, void *info);
    CFIndex (*read)(CFReadStreamRef stream, UInt8 *buffer, CFIndex bufferLength, CFStreamError *error, Boolean *atEOF, void *info);
    const UInt8 *(*getBuffer)(CFReadStreamRef stream, CFIndex maxBytesToRead, CFIndex *numBytesRead, CFStreamError *error, Boolean *atEOF, void *info);
    Boolean (*canRead)(CFReadStreamRef stream, void *info);
    void (*close)(CFReadStreamRef stream, void *info);
    CFTypeRef (*copyProperty)(CFReadStreamRef stream, CFStringRef propertyName, void *info);
    Boolean (*setProperty)(CFReadStreamRef stream, CFStringRef propertyName, CFTypeRef propertyValue, void *info);
    void (*requestEvents)(CFReadStreamRef stream, CFOptionFlags streamEvents, void *info);
    void (*schedule)(CFReadStreamRef stream, CFRunLoopRef runLoop, CFStringRef runLoopMode, void *info);
    void (*unschedule)(CFReadStreamRef stream, CFRunLoopRef runLoop, CFStringRef runLoopMode, void *info);
} CFReadStreamCallBacksV1;

typedef struct {
    CFIndex version; /* == 1 */
    void *(*create)(CFWriteStreamRef stream, void *info);
    void (*finalize)(CFWriteStreamRef stream, void *info);
    CFStringRef (*copyDescription)(CFWriteStreamRef stream, void *info);
    Boolean (*open)(CFWriteStreamRef stream, CFStreamError *error, Boolean *openComplete, void *info);
    Boolean (*openCompleted)(CFWriteStreamRef stream, CFStreamError *error, void *info);
    CFIndex (*write)(CFWriteStreamRef stream, const UInt8 *buffer, CFIndex bufferLength, CFStreamError *error, void *info);
    Boolean (*canWrite)(CFWriteStreamRef stream, void *info); 
    void (*close)(CFWriteStreamRef stream, void *info);
    CFTypeRef (*copyProperty)(CFWriteStreamRef stream, CFStringRef propertyName, void *info);
    Boolean (*setProperty)(CFWriteStreamRef stream, CFStringRef propertyName, CFTypeRef propertyValue, void *info);
    void (*requestEvents)(CFWriteStreamRef stream, CFOptionFlags streamEvents, void *info);
    void (*schedule)(CFWriteStreamRef stream, CFRunLoopRef runLoop, CFStringRef runLoopMode, void *info);
    void (*unschedule)(CFWriteStreamRef stream, CFRunLoopRef runLoop, CFStringRef runLoopMode, void *info);
} CFWriteStreamCallBacksV1;

typedef struct {
    CFIndex version; /* == 0 */
    Boolean (*open)(CFReadStreamRef stream, CFStreamError *error, Boolean *openComplete, void *info);
    Boolean (*openCompleted)(CFReadStreamRef stream, CFStreamError *error, void *info);
    CFIndex (*read)(CFReadStreamRef stream, UInt8 *buffer, CFIndex bufferLength, CFStreamError *error, Boolean *atEOF, void *info);
    const UInt8 *(*getBuffer)(CFReadStreamRef stream, CFIndex maxBytesToRead, CFIndex *numBytesRead, CFStreamError *error, Boolean *atEOF, void *info);
    Boolean (*canRead)(CFReadStreamRef stream, void *info);
    void (*close)(CFReadStreamRef stream, void *info);
    CFTypeRef (*copyProperty)(CFReadStreamRef stream, CFStringRef propertyName, void *info);
    void (*schedule)(CFReadStreamRef stream, CFRunLoopRef runLoop, CFStringRef runLoopMode, void *info);
    void (*unschedule)(CFReadStreamRef stream, CFRunLoopRef runLoop, CFStringRef runLoopMode, void *info);
} CFReadStreamCallBacksV0;

typedef struct {
    CFIndex version; /* == 0 */
    Boolean (*open)(CFWriteStreamRef stream, CFStreamError *error, Boolean *openComplete, void *info);
    Boolean (*openCompleted)(CFWriteStreamRef stream, CFStreamError *error, void *info);
    CFIndex (*write)(CFWriteStreamRef stream, const UInt8 *buffer, CFIndex bufferLength, CFStreamError *error, void *info);
    Boolean (*canWrite)(CFWriteStreamRef stream, void *info); 
    void (*close)(CFWriteStreamRef stream, void *info);
    CFTypeRef (*copyProperty)(CFWriteStreamRef stream, CFStringRef propertyName, void *info);
    void (*schedule)(CFWriteStreamRef stream, CFRunLoopRef runLoop, CFStringRef runLoopMode, void *info);
    void (*unschedule)(CFWriteStreamRef stream, CFRunLoopRef runLoop, CFStringRef runLoopMode, void *info);
} CFWriteStreamCallBacksV0;

CF_EXTERN_C_END

#endif /* ! __COREFOUNDATION_CFSTREAMABSTRACT__ */
