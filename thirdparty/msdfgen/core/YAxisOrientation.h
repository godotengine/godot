
#pragma once

#include "base.h"

namespace msdfgen {

/// Specifies whether the Y component of the coordinate system increases in the upward or downward direction.
enum YAxisOrientation {
    Y_UPWARD,
    Y_DOWNWARD
};

}

#define MSDFGEN_Y_AXIS_DEFAULT_ORIENTATION msdfgen::Y_UPWARD
#define MSDFGEN_Y_AXIS_NONDEFAULT_ORIENTATION (MSDFGEN_Y_AXIS_DEFAULT_ORIENTATION == msdfgen::Y_DOWNWARD ? msdfgen::Y_UPWARD : msdfgen::Y_DOWNWARD)
