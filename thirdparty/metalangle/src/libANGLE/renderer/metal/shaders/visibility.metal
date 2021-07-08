//
// Copyright 2019 The ANGLE Project. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#include "common.h"

constant bool kCombineWithExistingResult [[function_constant(1)]];

// Combine the visibility result of current render pass with previous value from previous render
// pass
struct CombineVisibilityResultOptions
{
    // Start offset in the render pass's visibility buffer allocated for the query.
    uint startOffset;
    // How many offsets in the render pass's visibility buffer is used for the query?
    uint numOffsets;
};

kernel void combineVisibilityResult(uint idx [[thread_position_in_grid]],
                                    constant CombineVisibilityResultOptions &options [[buffer(0)]],
                                    constant ushort4 *renderpassVisibilityResult [[buffer(1)]],
                                    device ushort4 *finalResults [[buffer(2)]])
{
    if (idx > 0)
    {
        // NOTE(hqle):
        // This is a bit wasteful to use a WARP of multiple threads just for combining one integer.
        // Consider a better approach.
        return;
    }
    ushort4 finalResult16x4;

    if (kCombineWithExistingResult)
    {
        finalResult16x4 = finalResults[0];
    }
    else
    {
        finalResult16x4 = ushort4(0, 0, 0, 0);
    }

    for (uint i = 0; i < options.numOffsets; ++i)
    {
        uint offset              = options.startOffset + i;
        ushort4 renderpassResult = renderpassVisibilityResult[offset];

        // Only boolean result is required, so bitwise OR is enough
        finalResult16x4          = finalResult16x4 | renderpassResult;
    }
    finalResults[0] = finalResult16x4;
}
