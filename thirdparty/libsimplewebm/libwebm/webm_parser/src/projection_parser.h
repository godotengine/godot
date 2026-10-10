// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#ifndef SRC_PROJECTION_PARSER_H_
#define SRC_PROJECTION_PARSER_H_

#include "src/byte_parser.h"
#include "src/float_parser.h"
#include "src/int_parser.h"
#include "src/master_value_parser.h"
#include "webm/dom_types.h"
#include "webm/id.h"

namespace webm {

// Spec reference:
// https://github.com/google/spatial-media/blob/master/docs/spherical-video-v2-rfc.md#projection-master-element
class ProjectionParser : public MasterValueParser<Projection> {
 public:
  ProjectionParser()
      : MasterValueParser(
            MakeChild<IntParser<ProjectionType>>(Id::kProjectionType,
                                                 &Projection::type),
            MakeChild<BinaryParser>(Id::kProjectionPrivate,
                                    &Projection::projection_private),
            MakeChild<FloatParser>(Id::kProjectionPoseYaw,
                                   &Projection::pose_yaw),
            MakeChild<FloatParser>(Id::kProjectionPosePitch,
                                   &Projection::pose_pitch),
            MakeChild<FloatParser>(Id::kProjectionPoseRoll,
                                   &Projection::pose_roll)) {}
};

}  // namespace webm

#endif  // SRC_PROJECTION_PARSER_H_
