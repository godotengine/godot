// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#ifndef SRC_CLUSTER_PARSER_H_
#define SRC_CLUSTER_PARSER_H_

#include "src/block_group_parser.h"
#include "src/block_parser.h"
#include "src/int_parser.h"
#include "src/master_value_parser.h"
#include "webm/dom_types.h"
#include "webm/id.h"

namespace webm {

// Spec reference:
// http://matroska.org/technical/specs/index.html#Cluster
// http://www.webmproject.org/docs/container/#Cluster
class ClusterParser : public MasterValueParser<Cluster> {
 public:
  ClusterParser()
      : MasterValueParser<Cluster>(
            MakeChild<UnsignedIntParser>(Id::kTimecode, &Cluster::timecode),
            MakeChild<UnsignedIntParser>(Id::kPrevSize,
                                         &Cluster::previous_size),
            MakeChild<SimpleBlockParser>(Id::kSimpleBlock,
                                         &Cluster::simple_blocks)
                .UseAsStartEvent(),
            MakeChild<BlockGroupParser>(Id::kBlockGroup, &Cluster::block_groups)
                .UseAsStartEvent()) {}

 protected:
  Status OnParseStarted(Callback* callback, Action* action) override {
    return callback->OnClusterBegin(metadata(Id::kCluster), value(), action);
  }

  Status OnParseCompleted(Callback* callback) override {
    return callback->OnClusterEnd(metadata(Id::kCluster), value());
  }
};

}  // namespace webm

#endif  // SRC_CLUSTER_PARSER_H_
