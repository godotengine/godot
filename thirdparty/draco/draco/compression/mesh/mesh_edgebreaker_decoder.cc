// Copyright 2016 The Draco Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
#include "draco/compression/mesh/mesh_edgebreaker_decoder.h"

#include "draco/compression/mesh/mesh_edgebreaker_decoder_impl.h"
#include "draco/compression/mesh/mesh_edgebreaker_traversal_predictive_decoder.h"
#include "draco/compression/mesh/mesh_edgebreaker_traversal_valence_decoder.h"

namespace draco {

MeshEdgebreakerDecoder::MeshEdgebreakerDecoder() {}

bool MeshEdgebreakerDecoder::CreateAttributesDecoder(int32_t att_decoder_id) {
  return impl_->CreateAttributesDecoder(att_decoder_id);
}

bool MeshEdgebreakerDecoder::InitializeDecoder() {
  uint8_t traversal_decoder_type;
  if (!buffer()->Decode(&traversal_decoder_type)) {
    return false;
  }
  impl_ = nullptr;
  if (traversal_decoder_type == MESH_EDGEBREAKER_STANDARD_ENCODING) {
#ifdef DRACO_STANDARD_EDGEBREAKER_SUPPORTED
    impl_ = std::unique_ptr<MeshEdgebreakerDecoderImplInterface>(
        new MeshEdgebreakerDecoderImpl<MeshEdgebreakerTraversalDecoder>());
#endif
  } else if (traversal_decoder_type == MESH_EDGEBREAKER_PREDICTIVE_ENCODING) {
#ifdef DRACO_BACKWARDS_COMPATIBILITY_SUPPORTED
#ifdef DRACO_PREDICTIVE_EDGEBREAKER_SUPPORTED
    impl_ = std::unique_ptr<MeshEdgebreakerDecoderImplInterface>(
        new MeshEdgebreakerDecoderImpl<
            MeshEdgebreakerTraversalPredictiveDecoder>());
#endif
#endif
  } else if (traversal_decoder_type == MESH_EDGEBREAKER_VALENCE_ENCODING) {
    impl_ = std::unique_ptr<MeshEdgebreakerDecoderImplInterface>(
        new MeshEdgebreakerDecoderImpl<
            MeshEdgebreakerTraversalValenceDecoder>());
  }
  if (!impl_) {
    return false;
  }
  if (!impl_->Init(this)) {
    return false;
  }
  return true;
}

bool MeshEdgebreakerDecoder::DecodeConnectivity() {
  return impl_->DecodeConnectivity();
}

bool MeshEdgebreakerDecoder::OnAttributesDecoded() {
  return impl_->OnAttributesDecoded();
}

}  // namespace draco
