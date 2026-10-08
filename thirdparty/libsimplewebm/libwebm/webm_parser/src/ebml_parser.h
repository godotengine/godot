// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#ifndef SRC_EBML_PARSER_H_
#define SRC_EBML_PARSER_H_

#include "src/byte_parser.h"
#include "src/int_parser.h"
#include "src/master_value_parser.h"
#include "webm/dom_types.h"
#include "webm/id.h"

namespace webm {

// Spec references:
// http://matroska.org/technical/specs/index.html#EBML
// https://github.com/Matroska-Org/ebml-specification/blob/master/specification.markdown#ebml-header-elements
// http://www.webmproject.org/docs/container/#EBML
class EbmlParser : public MasterValueParser<Ebml> {
 public:
  EbmlParser()
      : MasterValueParser<Ebml>(
            MakeChild<UnsignedIntParser>(Id::kEbmlVersion, &Ebml::ebml_version),
            MakeChild<UnsignedIntParser>(Id::kEbmlReadVersion,
                                         &Ebml::ebml_read_version),
            MakeChild<UnsignedIntParser>(Id::kEbmlMaxIdLength,
                                         &Ebml::ebml_max_id_length),
            MakeChild<UnsignedIntParser>(Id::kEbmlMaxSizeLength,
                                         &Ebml::ebml_max_size_length),
            MakeChild<StringParser>(Id::kDocType, &Ebml::doc_type),
            MakeChild<UnsignedIntParser>(Id::kDocTypeVersion,
                                         &Ebml::doc_type_version),
            MakeChild<UnsignedIntParser>(Id::kDocTypeReadVersion,
                                         &Ebml::doc_type_read_version)) {}

 protected:
  Status OnParseCompleted(Callback* callback) override {
    return callback->OnEbml(metadata(Id::kEbml), value());
  }
};

}  // namespace webm

#endif  // SRC_EBML_PARSER_H_
