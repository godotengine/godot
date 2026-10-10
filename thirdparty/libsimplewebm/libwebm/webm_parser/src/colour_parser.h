// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#ifndef SRC_COLOUR_PARSER_H_
#define SRC_COLOUR_PARSER_H_

#include "src/int_parser.h"
#include "src/master_value_parser.h"
#include "src/mastering_metadata_parser.h"
#include "webm/dom_types.h"
#include "webm/id.h"

namespace webm {

// Spec reference:
// http://matroska.org/technical/specs/index.html#Colour
// http://www.webmproject.org/docs/container/#Colour
class ColourParser : public MasterValueParser<Colour> {
 public:
  ColourParser()
      : MasterValueParser<Colour>(
            MakeChild<IntParser<MatrixCoefficients>>(
                Id::kMatrixCoefficients, &Colour::matrix_coefficients),
            MakeChild<UnsignedIntParser>(Id::kBitsPerChannel,
                                         &Colour::bits_per_channel),
            MakeChild<UnsignedIntParser>(Id::kChromaSubsamplingHorz,
                                         &Colour::chroma_subsampling_x),
            MakeChild<UnsignedIntParser>(Id::kChromaSubsamplingVert,
                                         &Colour::chroma_subsampling_y),
            MakeChild<UnsignedIntParser>(Id::kCbSubsamplingHorz,
                                         &Colour::cb_subsampling_x),
            MakeChild<UnsignedIntParser>(Id::kCbSubsamplingVert,
                                         &Colour::cb_subsampling_y),
            MakeChild<UnsignedIntParser>(Id::kChromaSitingHorz,
                                         &Colour::chroma_siting_x),
            MakeChild<UnsignedIntParser>(Id::kChromaSitingVert,
                                         &Colour::chroma_siting_y),
            MakeChild<IntParser<Range>>(Id::kRange, &Colour::range),
            MakeChild<IntParser<TransferCharacteristics>>(
                Id::kTransferCharacteristics,
                &Colour::transfer_characteristics),
            MakeChild<IntParser<Primaries>>(Id::kPrimaries, &Colour::primaries),
            MakeChild<UnsignedIntParser>(Id::kMaxCll, &Colour::max_cll),
            MakeChild<UnsignedIntParser>(Id::kMaxFall, &Colour::max_fall),
            MakeChild<MasteringMetadataParser>(Id::kMasteringMetadata,
                                               &Colour::mastering_metadata)) {}
};

}  // namespace webm

#endif  // SRC_COLOUR_PARSER_H_
