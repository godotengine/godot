
#pragma once

#include <cstdlib>
#include <cstdio>
#include "Shape.h"

namespace msdfgen {

/// Deserializes a text description of a vector shape into output.
bool readShapeDescription(FILE *input, Shape &output, bool *colorsSpecified = NULL);
bool readShapeDescription(const char *input, Shape &output, bool *colorsSpecified = NULL);
/// Serializes a shape object into a text description.
bool writeShapeDescription(FILE *output, const Shape &shape);

}
