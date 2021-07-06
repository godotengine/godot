//
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// Platform.h: The public interface ANGLE exposes to the API layer, for
//   doing platform-specific tasks like gathering data, or for tracing.

#ifndef ANGLE_PLATFORM_H
#define ANGLE_PLATFORM_H

#include <stdint.h>
#include <array>

#define EGL_PLATFORM_ANGLE_PLATFORM_METHODS_ANGLEX 0x3482

#if defined(_WIN32)
#    if !defined(LIBANGLE_IMPLEMENTATION)
#        define ANGLE_PLATFORM_EXPORT __declspec(dllimport)
#    else
#        define ANGLE_PLATFORM_EXPORT __declspec(dllexport)
#    endif
#elif defined(__GNUC__) || defined(__clang__)
#    define ANGLE_PLATFORM_EXPORT __attribute__((visibility("default")))
#endif
#if !defined(ANGLE_PLATFORM_EXPORT)
#    define ANGLE_PLATFORM_EXPORT
#endif

#if defined(_WIN32)
#    define ANGLE_APIENTRY __stdcall
#else
#    define ANGLE_APIENTRY
#endif

namespace angle
{
struct FeaturesD3D;
struct FeaturesVk;
struct FeaturesMtl;
using TraceEventHandle = uint64_t;
using EGLDisplayType   = void *;
struct PlatformMethods;

// Use a C-like API to not trigger undefined calling behaviour.
// Avoid using decltype here to work around sanitizer limitations.
// TODO(jmadill): Use decltype here if/when UBSAN is fixed.

// System --------------------------------------------------------------

// Wall clock time in seconds since the epoch.
// TODO(jmadill): investigate using an ANGLE internal time library
using CurrentTimeFunc = double (*)(PlatformMethods *platform);
inline double DefaultCurrentTime(PlatformMethods *platform)
{
    return 0.0;
}

// Monotonically increasing time in seconds from an arbitrary fixed point in the past.
// This function is expected to return at least millisecond-precision values. For this reason,
// it is recommended that the fixed point be no further in the past than the epoch.
using MonotonicallyIncreasingTimeFunc = double (*)(PlatformMethods *platform);
inline double DefaultMonotonicallyIncreasingTime(PlatformMethods *platform)
{
    return 0.0;
}

// Logging ------------------------------------------------------------

// Log an error message within the platform implementation.
using LogErrorFunc = void (*)(PlatformMethods *platform, const char *errorMessage);
inline void DefaultLogError(PlatformMethods *platform, const char *errorMessage) {}

// Log a warning message within the platform implementation.
using LogWarningFunc = void (*)(PlatformMethods *platform, const char *warningMessage);
inline void DefaultLogWarning(PlatformMethods *platform, const char *warningMessage) {}

// Log an info message within the platform implementation.
using LogInfoFunc = void (*)(PlatformMethods *platform, const char *infoMessage);
inline void DefaultLogInfo(PlatformMethods *platform, const char *infoMessage) {}

// Tracing --------

// Get a pointer to the enabled state of the given trace category. The
// embedder can dynamically change the enabled state as trace event
// recording is started and stopped by the application. Only long-lived
// literal strings should be given as the category name. The implementation
// expects the returned pointer to be held permanently in a local static. If
// the unsigned char is non-zero, tracing is enabled. If tracing is enabled,
// addTraceEvent is expected to be called by the trace event macros.
using GetTraceCategoryEnabledFlagFunc = const unsigned char *(*)(PlatformMethods *platform,
                                                                 const char *categoryName);
inline const unsigned char *DefaultGetTraceCategoryEnabledFlag(PlatformMethods *platform,
                                                               const char *categoryName)
{
    return nullptr;
}

//
// Add a trace event to the platform tracing system. Depending on the actual
// enabled state, this event may be recorded or dropped.
// - phase specifies the type of event:
//   - BEGIN ('B'): Marks the beginning of a scoped event.
//   - END ('E'): Marks the end of a scoped event.
//   - COMPLETE ('X'): Marks the beginning of a scoped event, but doesn't
//     need a matching END event. Instead, at the end of the scope,
//     updateTraceEventDuration() must be called with the TraceEventHandle
//     returned from addTraceEvent().
//   - INSTANT ('I'): Standalone, instantaneous event.
//   - START ('S'): Marks the beginning of an asynchronous event (the end
//     event can occur in a different scope or thread). The id parameter is
//     used to match START/FINISH pairs.
//   - FINISH ('F'): Marks the end of an asynchronous event.
//   - COUNTER ('C'): Used to trace integer quantities that change over
//     time. The argument values are expected to be of type int.
//   - METADATA ('M'): Reserved for internal use.
// - categoryEnabled is the pointer returned by getTraceCategoryEnabledFlag.
// - name is the name of the event. Also used to match BEGIN/END and
//   START/FINISH pairs.
// - id optionally allows events of the same name to be distinguished from
//   each other. For example, to trace the construction and destruction of
//   objects, specify the pointer as the id parameter.
// - timestamp should be a time value returned from monotonicallyIncreasingTime.
// - numArgs specifies the number of elements in argNames, argTypes, and
//   argValues.
// - argNames is the array of argument names. Use long-lived literal strings
//   or specify the COPY flag.
// - argTypes is the array of argument types:
//   - BOOL (1): bool
//   - UINT (2): unsigned long long
//   - INT (3): long long
//   - DOUBLE (4): double
//   - POINTER (5): void*
//   - STRING (6): char* (long-lived null-terminated char* string)
//   - COPY_STRING (7): char* (temporary null-terminated char* string)
//   - CONVERTABLE (8): WebConvertableToTraceFormat
// - argValues is the array of argument values. Each value is the unsigned
//   long long member of a union of all supported types.
// - flags can be 0 or one or more of the following, ORed together:
//   - COPY (0x1): treat all strings (name, argNames and argValues of type
//     string) as temporary so that they will be copied by addTraceEvent.
//   - HAS_ID (0x2): use the id argument to uniquely identify the event for
//     matching with other events of the same name.
//   - MANGLE_ID (0x4): specify this flag if the id parameter is the value
//     of a pointer.
using AddTraceEventFunc = angle::TraceEventHandle (*)(PlatformMethods *platform,
                                                      char phase,
                                                      const unsigned char *categoryEnabledFlag,
                                                      const char *name,
                                                      unsigned long long id,
                                                      double timestamp,
                                                      int numArgs,
                                                      const char **argNames,
                                                      const unsigned char *argTypes,
                                                      const unsigned long long *argValues,
                                                      unsigned char flags);
inline angle::TraceEventHandle DefaultAddTraceEvent(PlatformMethods *platform,
                                                    char phase,
                                                    const unsigned char *categoryEnabledFlag,
                                                    const char *name,
                                                    unsigned long long id,
                                                    double timestamp,
                                                    int numArgs,
                                                    const char **argNames,
                                                    const unsigned char *argTypes,
                                                    const unsigned long long *argValues,
                                                    unsigned char flags)
{
    return 0;
}

// Set the duration field of a COMPLETE trace event.
using UpdateTraceEventDurationFunc = void (*)(PlatformMethods *platform,
                                              const unsigned char *categoryEnabledFlag,
                                              const char *name,
                                              angle::TraceEventHandle eventHandle);
inline void DefaultUpdateTraceEventDuration(PlatformMethods *platform,
                                            const unsigned char *categoryEnabledFlag,
                                            const char *name,
                                            angle::TraceEventHandle eventHandle)
{}

// Callbacks for reporting histogram data.
// CustomCounts histogram has exponential bucket sizes, so that min=1, max=1000000, bucketCount=50
// would do.
using HistogramCustomCountsFunc = void (*)(PlatformMethods *platform,
                                           const char *name,
                                           int sample,
                                           int min,
                                           int max,
                                           int bucketCount);
inline void DefaultHistogramCustomCounts(PlatformMethods *platform,
                                         const char *name,
                                         int sample,
                                         int min,
                                         int max,
                                         int bucketCount)
{}
// Enumeration histogram buckets are linear, boundaryValue should be larger than any possible sample
// value.
using HistogramEnumerationFunc = void (*)(PlatformMethods *platform,
                                          const char *name,
                                          int sample,
                                          int boundaryValue);
inline void DefaultHistogramEnumeration(PlatformMethods *platform,
                                        const char *name,
                                        int sample,
                                        int boundaryValue)
{}
// Unlike enumeration histograms, sparse histograms only allocate memory for non-empty buckets.
using HistogramSparseFunc = void (*)(PlatformMethods *platform, const char *name, int sample);
inline void DefaultHistogramSparse(PlatformMethods *platform, const char *name, int sample) {}
// Boolean histograms track two-state variables.
using HistogramBooleanFunc = void (*)(PlatformMethods *platform, const char *name, bool sample);
inline void DefaultHistogramBoolean(PlatformMethods *platform, const char *name, bool sample) {}

// Allows us to programatically override ANGLE's default workarounds for testing purposes.
using OverrideWorkaroundsD3DFunc = void (*)(PlatformMethods *platform,
                                            angle::FeaturesD3D *featuresD3D);
inline void DefaultOverrideWorkaroundsD3D(PlatformMethods *platform,
                                          angle::FeaturesD3D *featuresD3D)
{}

using OverrideFeaturesVkFunc = void (*)(PlatformMethods *platform,
                                        angle::FeaturesVk *featuresVulkan);
inline void DefaultOverrideFeaturesVk(PlatformMethods *platform, angle::FeaturesVk *featuresVulkan)
{}

using OverrideFeaturesMtlFunc = void (*)(PlatformMethods *platform,
                                         angle::FeaturesMtl *featuresMetal);
inline void DefaultOverrideFeaturesMtl(PlatformMethods *platform, angle::FeaturesMtl *featuresMetal)
{}

// Callback on a successful program link with the program binary. Can be used to store
// shaders to disk. Keys are a 160-bit SHA-1 hash.
using ProgramKeyType   = std::array<uint8_t, 20>;
using CacheProgramFunc = void (*)(PlatformMethods *platform,
                                  const ProgramKeyType &key,
                                  size_t programSize,
                                  const uint8_t *programBytes);
inline void DefaultCacheProgram(PlatformMethods *platform,
                                const ProgramKeyType &key,
                                size_t programSize,
                                const uint8_t *programBytes)
{}

// Platform methods are enumerated here once.
#define ANGLE_PLATFORM_OP(OP)                                    \
    OP(currentTime, CurrentTime)                                 \
    OP(monotonicallyIncreasingTime, MonotonicallyIncreasingTime) \
    OP(logError, LogError)                                       \
    OP(logWarning, LogWarning)                                   \
    OP(logInfo, LogInfo)                                         \
    OP(getTraceCategoryEnabledFlag, GetTraceCategoryEnabledFlag) \
    OP(addTraceEvent, AddTraceEvent)                             \
    OP(updateTraceEventDuration, UpdateTraceEventDuration)       \
    OP(histogramCustomCounts, HistogramCustomCounts)             \
    OP(histogramEnumeration, HistogramEnumeration)               \
    OP(histogramSparse, HistogramSparse)                         \
    OP(histogramBoolean, HistogramBoolean)                       \
    OP(overrideWorkaroundsD3D, OverrideWorkaroundsD3D)           \
    OP(overrideFeaturesVk, OverrideFeaturesVk)                   \
    OP(overrideFeaturesMtl, OverrideFeaturesMtl)                 \
    OP(cacheProgram, CacheProgram)

#define ANGLE_PLATFORM_METHOD_DEF(Name, CapsName) CapsName##Func Name = Default##CapsName;

struct ANGLE_PLATFORM_EXPORT PlatformMethods
{
    inline PlatformMethods();

    // User data pointer for any implementation specific members. Put it at the start of the
    // platform structure so it doesn't become overwritten if one version of the platform
    // adds or removes new members.
    void *context = 0;

    ANGLE_PLATFORM_OP(ANGLE_PLATFORM_METHOD_DEF)
};

inline PlatformMethods::PlatformMethods() = default;

#undef ANGLE_PLATFORM_METHOD_DEF

// Subtract one to account for the context pointer.
constexpr unsigned int g_NumPlatformMethods = (sizeof(PlatformMethods) / sizeof(uintptr_t)) - 1;

#define ANGLE_PLATFORM_METHOD_STRING(Name) #Name
#define ANGLE_PLATFORM_METHOD_STRING2(Name, CapsName) ANGLE_PLATFORM_METHOD_STRING(Name),

constexpr const char *const g_PlatformMethodNames[g_NumPlatformMethods] = {
    ANGLE_PLATFORM_OP(ANGLE_PLATFORM_METHOD_STRING2)};

#undef ANGLE_PLATFORM_METHOD_STRING2
#undef ANGLE_PLATFORM_METHOD_STRING

}  // namespace angle

extern "C" {

// Gets the platform methods on the passed-in EGL display. If the method name signature does not
// match the compiled signature for this ANGLE, false is returned. On success true is returned.
// The application should set any platform methods it cares about on the returned pointer.
// If display is not valid, behaviour is undefined.

ANGLE_PLATFORM_EXPORT bool ANGLE_APIENTRY ANGLEGetDisplayPlatform(angle::EGLDisplayType display,
                                                                  const char *const methodNames[],
                                                                  unsigned int methodNameCount,
                                                                  void *context,
                                                                  void *platformMethodsOut);

// Sets the platform methods back to their defaults.
// If display is not valid, behaviour is undefined.
ANGLE_PLATFORM_EXPORT void ANGLE_APIENTRY ANGLEResetDisplayPlatform(angle::EGLDisplayType display);

}  // extern "C"

namespace angle
{
typedef bool(ANGLE_APIENTRY *GetDisplayPlatformFunc)(angle::EGLDisplayType,
                                                     const char *const *,
                                                     unsigned int,
                                                     void *,
                                                     void *);
typedef void(ANGLE_APIENTRY *ResetDisplayPlatformFunc)(angle::EGLDisplayType);
}  // namespace angle

// This function is not exported
angle::PlatformMethods *ANGLEPlatformCurrent();

#endif  // ANGLE_PLATFORM_H
