//
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// histogram_macros.h:
//   Helpers for making histograms, to keep consistency with Chromium's
//   histogram_macros.h.

#ifndef LIBANGLE_HISTOGRAM_MACROS_H_
#define LIBANGLE_HISTOGRAM_MACROS_H_

#include <platform/Platform.h>

#define ANGLE_HISTOGRAM_TIMES(name, sample) ANGLE_HISTOGRAM_CUSTOM_TIMES(name, sample, 1, 10000, 50)

#define ANGLE_HISTOGRAM_MEDIUM_TIMES(name, sample) \
    ANGLE_HISTOGRAM_CUSTOM_TIMES(name, sample, 10, 180000, 50)

// Use this macro when times can routinely be much longer than 10 seconds.
#define ANGLE_HISTOGRAM_LONG_TIMES(name, sample) \
    ANGLE_HISTOGRAM_CUSTOM_TIMES(name, sample, 1, 3600000, 50)

// Use this macro when times can routinely be much longer than 10 seconds and
// you want 100 buckets.
#define ANGLE_HISTOGRAM_LONG_TIMES_100(name, sample) \
    ANGLE_HISTOGRAM_CUSTOM_TIMES(name, sample, 1, 3600000, 100)

// For folks that need real specific times, use this to select a precise range
// of times you want plotted, and the number of buckets you want used.
#define ANGLE_HISTOGRAM_CUSTOM_TIMES(name, sample, min, max, bucket_count) \
    ANGLE_HISTOGRAM_CUSTOM_COUNTS(name, sample, min, max, bucket_count)

#define ANGLE_HISTOGRAM_COUNTS(name, sample) \
    ANGLE_HISTOGRAM_CUSTOM_COUNTS(name, sample, 1, 1000000, 50)

#define ANGLE_HISTOGRAM_COUNTS_100(name, sample) \
    ANGLE_HISTOGRAM_CUSTOM_COUNTS(name, sample, 1, 100, 50)

#define ANGLE_HISTOGRAM_COUNTS_10000(name, sample) \
    ANGLE_HISTOGRAM_CUSTOM_COUNTS(name, sample, 1, 10000, 50)

#define ANGLE_HISTOGRAM_CUSTOM_COUNTS(name, sample, min, max, bucket_count)                       \
    ANGLEPlatformCurrent()->histogramCustomCounts(ANGLEPlatformCurrent(), name, sample, min, max, \
                                                  bucket_count)

#define ANGLE_HISTOGRAM_PERCENTAGE(name, under_one_hundred) \
    ANGLE_HISTOGRAM_ENUMERATION(name, under_one_hundred, 101)

#define ANGLE_HISTOGRAM_BOOLEAN(name, sample) \
    ANGLEPlatformCurrent()->histogramBoolean(ANGLEPlatformCurrent(), name, sample)

#define ANGLE_HISTOGRAM_ENUMERATION(name, sample, boundary_value)                      \
    ANGLEPlatformCurrent()->histogramEnumeration(ANGLEPlatformCurrent(), name, sample, \
                                                 boundary_value)

#define ANGLE_HISTOGRAM_MEMORY_KB(name, sample) \
    ANGLE_HISTOGRAM_CUSTOM_COUNTS(name, sample, 1000, 500000, 50)

#define ANGLE_HISTOGRAM_MEMORY_MB(name, sample) \
    ANGLE_HISTOGRAM_CUSTOM_COUNTS(name, sample, 1, 1000, 50)

#define ANGLE_HISTOGRAM_SPARSE_SLOWLY(name, sample) \
    ANGLEPlatformCurrent()->histogramSparse(ANGLEPlatformCurrent(), name, sample)

// Scoped class which logs its time on this earth as a UMA statistic. This is
// recommended for when you want a histogram which measures the time it takes
// for a method to execute. This measures up to 10 seconds.
#define SCOPED_ANGLE_HISTOGRAM_TIMER(name) \
    SCOPED_ANGLE_HISTOGRAM_TIMER_EXPANDER(name, false, __COUNTER__)

// Similar scoped histogram timer, but this uses ANGLE_HISTOGRAM_LONG_TIMES_100,
// which measures up to an hour, and uses 100 buckets. This is more expensive
// to store, so only use if this often takes >10 seconds.
#define SCOPED_ANGLE_HISTOGRAM_LONG_TIMER(name) \
    SCOPED_ANGLE_HISTOGRAM_TIMER_EXPANDER(name, true, __COUNTER__)

// This nested macro is necessary to expand __COUNTER__ to an actual value.
#define SCOPED_ANGLE_HISTOGRAM_TIMER_EXPANDER(name, is_long, key) \
    SCOPED_ANGLE_HISTOGRAM_TIMER_UNIQUE(name, is_long, key)

#define SCOPED_ANGLE_HISTOGRAM_TIMER_UNIQUE(name, is_long, key)                         \
    class ScopedHistogramTimer##key                                                     \
    {                                                                                   \
      public:                                                                           \
        ScopedHistogramTimer##key()                                                     \
            : constructed_(ANGLEPlatformCurrent()->currentTime(ANGLEPlatformCurrent())) \
        {}                                                                              \
        ~ScopedHistogramTimer##key()                                                    \
        {                                                                               \
            if (constructed_ == 0)                                                      \
                return;                                                                 \
            auto *platform = ANGLEPlatformCurrent();                                    \
            double elapsed = platform->currentTime(platform) - constructed_;            \
            int elapsedMS  = static_cast<int>(elapsed * 1000.0);                        \
            if (is_long)                                                                \
            {                                                                           \
                ANGLE_HISTOGRAM_LONG_TIMES_100(name, elapsedMS);                        \
            }                                                                           \
            else                                                                        \
            {                                                                           \
                ANGLE_HISTOGRAM_TIMES(name, elapsedMS);                                 \
            }                                                                           \
        }                                                                               \
                                                                                        \
      private:                                                                          \
        double constructed_;                                                            \
    } scoped_histogram_timer_##key

#endif  // LIBANGLE_HISTOGRAM_MACROS_H_
