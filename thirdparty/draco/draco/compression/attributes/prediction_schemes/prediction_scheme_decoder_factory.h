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
// Functions for creating prediction schemes for decoders using the provided
// prediction method id.

#ifndef DRACO_COMPRESSION_ATTRIBUTES_PREDICTION_SCHEMES_PREDICTION_SCHEME_DECODER_FACTORY_H_
#define DRACO_COMPRESSION_ATTRIBUTES_PREDICTION_SCHEMES_PREDICTION_SCHEME_DECODER_FACTORY_H_

#include "draco/compression/attributes/prediction_schemes/mesh_prediction_scheme_constrained_multi_parallelogram_decoder.h"
#include "draco/draco_features.h"
#ifdef DRACO_NORMAL_ENCODING_SUPPORTED
#include "draco/compression/attributes/prediction_schemes/mesh_prediction_scheme_geometric_normal_decoder.h"
#endif
#include "draco/compression/attributes/prediction_schemes/mesh_prediction_scheme_multi_parallelogram_decoder.h"
#include "draco/compression/attributes/prediction_schemes/mesh_prediction_scheme_parallelogram_decoder.h"
#include "draco/compression/attributes/prediction_schemes/mesh_prediction_scheme_tex_coords_decoder.h"
#include "draco/compression/attributes/prediction_schemes/mesh_prediction_scheme_tex_coords_portable_decoder.h"
#include "draco/compression/attributes/prediction_schemes/prediction_scheme_decoder.h"
#include "draco/compression/attributes/prediction_schemes/prediction_scheme_delta_decoder.h"
#include "draco/compression/attributes/prediction_schemes/prediction_scheme_factory.h"
#include "draco/compression/mesh/mesh_decoder.h"

namespace draco {

// Factory class for creating mesh prediction schemes. The factory implements
// operator() that is used to create an appropriate mesh prediction scheme in
// CreateMeshPredictionScheme() function in prediction_scheme_factory.h
template <typename DataTypeT>
struct MeshPredictionSchemeDecoderFactory {
  // Operator () specialized for the wrap transform. Wrap transform can be used
  // for all mesh prediction schemes. The specialization is done in compile time
  // to prevent instantiations of unneeded combinations of prediction schemes +
  // prediction transforms.
  template <class TransformT, class MeshDataT,
            PredictionSchemeTransformType Method>
  struct DispatchFunctor {
    std::unique_ptr<PredictionSchemeDecoder<DataTypeT, TransformT>> operator()(
        PredictionSchemeMethod method, const PointAttribute *attribute,
        const TransformT &transform, const MeshDataT &mesh_data,
        uint16_t bitstream_version) {
      if (method == MESH_PREDICTION_PARALLELOGRAM) {
        return std::unique_ptr<PredictionSchemeDecoder<DataTypeT, TransformT>>(
            new MeshPredictionSchemeParallelogramDecoder<DataTypeT, TransformT,
                                                         MeshDataT>(
                attribute, transform, mesh_data));
      }
#ifdef DRACO_BACKWARDS_COMPATIBILITY_SUPPORTED
      else if (method == MESH_PREDICTION_MULTI_PARALLELOGRAM) {
        return std::unique_ptr<PredictionSchemeDecoder<DataTypeT, TransformT>>(
            new MeshPredictionSchemeMultiParallelogramDecoder<
                DataTypeT, TransformT, MeshDataT>(attribute, transform,
                                                  mesh_data));
      }
#endif
      else if (method == MESH_PREDICTION_CONSTRAINED_MULTI_PARALLELOGRAM) {
        return std::unique_ptr<PredictionSchemeDecoder<DataTypeT, TransformT>>(
            new MeshPredictionSchemeConstrainedMultiParallelogramDecoder<
                DataTypeT, TransformT, MeshDataT>(attribute, transform,
                                                  mesh_data));
      }
#ifdef DRACO_BACKWARDS_COMPATIBILITY_SUPPORTED
      else if (method == MESH_PREDICTION_TEX_COORDS_DEPRECATED) {
        return std::unique_ptr<PredictionSchemeDecoder<DataTypeT, TransformT>>(
            new MeshPredictionSchemeTexCoordsDecoder<DataTypeT, TransformT,
                                                     MeshDataT>(
                attribute, transform, mesh_data, bitstream_version));
      }
#endif
      else if (method == MESH_PREDICTION_TEX_COORDS_PORTABLE) {
        return std::unique_ptr<PredictionSchemeDecoder<DataTypeT, TransformT>>(
            new MeshPredictionSchemeTexCoordsPortableDecoder<
                DataTypeT, TransformT, MeshDataT>(attribute, transform,
                                                  mesh_data));
      }
#ifdef DRACO_NORMAL_ENCODING_SUPPORTED
      else if (method == MESH_PREDICTION_GEOMETRIC_NORMAL) {
        return std::unique_ptr<PredictionSchemeDecoder<DataTypeT, TransformT>>(
            new MeshPredictionSchemeGeometricNormalDecoder<
                DataTypeT, TransformT, MeshDataT>(attribute, transform,
                                                  mesh_data));
      }
#endif
      return nullptr;
    }
  };

#ifdef DRACO_NORMAL_ENCODING_SUPPORTED
  // Operator () specialized for normal octahedron transforms. These transforms
  // are currently used only by the geometric normal prediction scheme (the
  // transform is also used by delta coding, but delta predictor is not
  // constructed in this function).
  template <class TransformT, class MeshDataT>
  struct DispatchFunctor<TransformT, MeshDataT,
                         PREDICTION_TRANSFORM_NORMAL_OCTAHEDRON_CANONICALIZED> {
    std::unique_ptr<PredictionSchemeDecoder<DataTypeT, TransformT>> operator()(
        PredictionSchemeMethod method, const PointAttribute *attribute,
        const TransformT &transform, const MeshDataT &mesh_data,
        uint16_t bitstream_version) {
      if (method == MESH_PREDICTION_GEOMETRIC_NORMAL) {
        return std::unique_ptr<PredictionSchemeDecoder<DataTypeT, TransformT>>(
            new MeshPredictionSchemeGeometricNormalDecoder<
                DataTypeT, TransformT, MeshDataT>(attribute, transform,
                                                  mesh_data));
      }
      return nullptr;
    }
  };
  template <class TransformT, class MeshDataT>
  struct DispatchFunctor<TransformT, MeshDataT,
                         PREDICTION_TRANSFORM_NORMAL_OCTAHEDRON> {
    std::unique_ptr<PredictionSchemeDecoder<DataTypeT, TransformT>> operator()(
        PredictionSchemeMethod method, const PointAttribute *attribute,
        const TransformT &transform, const MeshDataT &mesh_data,
        uint16_t bitstream_version) {
      if (method == MESH_PREDICTION_GEOMETRIC_NORMAL) {
        return std::unique_ptr<PredictionSchemeDecoder<DataTypeT, TransformT>>(
            new MeshPredictionSchemeGeometricNormalDecoder<
                DataTypeT, TransformT, MeshDataT>(attribute, transform,
                                                  mesh_data));
      }
      return nullptr;
    }
  };
#endif

  template <class TransformT, class MeshDataT>
  std::unique_ptr<PredictionSchemeDecoder<DataTypeT, TransformT>> operator()(
      PredictionSchemeMethod method, const PointAttribute *attribute,
      const TransformT &transform, const MeshDataT &mesh_data,
      uint16_t bitstream_version) {
    return DispatchFunctor<TransformT, MeshDataT, TransformT::GetType()>()(
        method, attribute, transform, mesh_data, bitstream_version);
  }
};

// Creates a prediction scheme for a given decoder and given prediction method.
// The prediction schemes are automatically initialized with decoder specific
// data if needed.
template <typename DataTypeT, class TransformT>
std::unique_ptr<PredictionSchemeDecoder<DataTypeT, TransformT>>
CreatePredictionSchemeForDecoder(PredictionSchemeMethod method, int att_id,
                                 const PointCloudDecoder *decoder,
                                 const TransformT &transform) {
  if (method == PREDICTION_NONE) {
    return nullptr;
  }
  const PointAttribute *const att = decoder->point_cloud()->attribute(att_id);
  if (decoder->GetGeometryType() == TRIANGULAR_MESH) {
    // Cast the decoder to mesh decoder. This is not necessarily safe if there
    // is some other decoder decides to use TRIANGULAR_MESH as the return type,
    // but unfortunately there is not nice work around for this without using
    // RTTI (double dispatch and similar concepts will not work because of the
    // template nature of the prediction schemes).
    const MeshDecoder *const mesh_decoder =
        static_cast<const MeshDecoder *>(decoder);

    auto ret = CreateMeshPredictionScheme<
        MeshDecoder, PredictionSchemeDecoder<DataTypeT, TransformT>,
        MeshPredictionSchemeDecoderFactory<DataTypeT>>(
        mesh_decoder, method, att_id, transform, decoder->bitstream_version());
    if (ret) {
      return ret;
    }
    // Otherwise try to create another prediction scheme.
  }
  // Create delta decoder.
  return std::unique_ptr<PredictionSchemeDecoder<DataTypeT, TransformT>>(
      new PredictionSchemeDeltaDecoder<DataTypeT, TransformT>(att, transform));
}

// Create a prediction scheme using a default transform constructor.
template <typename DataTypeT, class TransformT>
std::unique_ptr<PredictionSchemeDecoder<DataTypeT, TransformT>>
CreatePredictionSchemeForDecoder(PredictionSchemeMethod method, int att_id,
                                 const PointCloudDecoder *decoder) {
  return CreatePredictionSchemeForDecoder<DataTypeT, TransformT>(
      method, att_id, decoder, TransformT());
}

}  // namespace draco

#endif  // DRACO_COMPRESSION_ATTRIBUTES_PREDICTION_SCHEMES_PREDICTION_SCHEME_DECODER_FACTORY_H_
