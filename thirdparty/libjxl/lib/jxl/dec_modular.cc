// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/dec_modular.h"

#include <jxl/memory_manager.h>

#include <cstdint>
#include <vector>

#include "lib/jxl/frame_header.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/dec_modular.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/printf_macros.h"
#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/compressed_dc.h"
#include "lib/jxl/epf.h"
#include "lib/jxl/modular/encoding/encoding.h"
#include "lib/jxl/modular/modular_image.h"
#include "lib/jxl/modular/transform/transform.h"

HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

// These templates are not found via ADL.
using hwy::HWY_NAMESPACE::Add;
using hwy::HWY_NAMESPACE::Mul;
using hwy::HWY_NAMESPACE::Rebind;

void MultiplySum(const size_t xsize,
                 const pixel_type* const JXL_RESTRICT row_in,
                 const pixel_type* const JXL_RESTRICT row_in_Y,
                 const float factor, float* const JXL_RESTRICT row_out) {
  const HWY_FULL(float) df;
  const Rebind<pixel_type, HWY_FULL(float)> di;  // assumes pixel_type <= float
  const auto factor_v = Set(df, factor);
  for (size_t x = 0; x < xsize; x += Lanes(di)) {
    const auto in = Add(Load(di, row_in + x), Load(di, row_in_Y + x));
    const auto out = Mul(ConvertTo(df, in), factor_v);
    Store(out, df, row_out + x);
  }
}

void RgbFromSingle(const size_t xsize,
                   const pixel_type* const JXL_RESTRICT row_in,
                   const float factor, float* out_r, float* out_g,
                   float* out_b) {
  const HWY_FULL(float) df;
  const Rebind<pixel_type, HWY_FULL(float)> di;  // assumes pixel_type <= float

  const auto factor_v = Set(df, factor);
  for (size_t x = 0; x < xsize; x += Lanes(di)) {
    const auto in = Load(di, row_in + x);
    const auto out = Mul(ConvertTo(df, in), factor_v);
    Store(out, df, out_r + x);
    Store(out, df, out_g + x);
    Store(out, df, out_b + x);
  }
}

void SingleFromSingle(const size_t xsize,
                      const pixel_type* const JXL_RESTRICT row_in,
                      const float factor, float* row_out) {
  const HWY_FULL(float) df;
  const Rebind<pixel_type, HWY_FULL(float)> di;  // assumes pixel_type <= float

  const auto factor_v = Set(df, factor);
  for (size_t x = 0; x < xsize; x += Lanes(di)) {
    const auto in = Load(di, row_in + x);
    const auto out = Mul(ConvertTo(df, in), factor_v);
    Store(out, df, row_out + x);
  }
}
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {
HWY_EXPORT(MultiplySum);       // Local function
HWY_EXPORT(RgbFromSingle);     // Local function
HWY_EXPORT(SingleFromSingle);  // Local function

// Slow conversion using double precision multiplication, only
// needed when the bit depth is too high for single precision
void SingleFromSingleAccurate(const size_t xsize,
                              const pixel_type* const JXL_RESTRICT row_in,
                              const double factor, float* row_out) {
  for (size_t x = 0; x < xsize; x++) {
    row_out[x] = row_in[x] * factor;
  }
}

// convert custom [bits]-bit float (with [exp_bits] exponent bits) stored as int
// back to binary32 float
Status int_to_float(const pixel_type* const JXL_RESTRICT row_in,
                    float* const JXL_RESTRICT row_out, const size_t xsize,
                    const int bits, const int exp_bits) {
  static_assert(sizeof(pixel_type) == sizeof(float));
  if (bits == 32) {
    JXL_ENSURE(exp_bits == 8);
    memcpy(row_out, row_in, xsize * sizeof(float));
    return true;
  }
  int exp_bias = (1 << (exp_bits - 1)) - 1;
  int sign_shift = bits - 1;
  int mant_bits = bits - exp_bits - 1;
  int mant_shift = 23 - mant_bits;
  for (size_t x = 0; x < xsize; ++x) {
    uint32_t f;
    memcpy(&f, &row_in[x], 4);
    int signbit = (f >> sign_shift);
    f &= (1 << sign_shift) - 1;
    if (f == 0) {
      row_out[x] = (signbit ? -0.f : 0.f);
      continue;
    }
    int exp = (f >> mant_bits);
    int mantissa = (f & ((1 << mant_bits) - 1));
    mantissa <<= mant_shift;
    // Try to normalize only if there is space for maneuver.
    if (exp == 0 && exp_bits < 8) {
      // subnormal number
      while ((mantissa & 0x800000) == 0) {
        mantissa <<= 1;
        exp--;
      }
      exp++;
      // remove leading 1 because it is implicit now
      mantissa &= 0x7fffff;
    }
    exp -= exp_bias;
    // broke up the arbitrary float into its parts, now reassemble into
    // binary32
    exp += 127;
    JXL_ENSURE(exp >= 0);
    f = (signbit ? 0x80000000 : 0);
    f |= (exp << 23);
    f |= mantissa;
    memcpy(&row_out[x], &f, 4);
  }
  return true;
}

#if JXL_DEBUG_V_LEVEL >= 1
std::string ModularStreamId::DebugString() const {
  std::ostringstream os;
  os << (kind == GlobalData   ? "ModularGlobal"
         : kind == VarDCTDC   ? "VarDCTDC"
         : kind == ModularDC  ? "ModularDC"
         : kind == ACMetadata ? "ACMeta"
         : kind == QuantTable ? "QuantTable"
         : kind == ModularAC  ? "ModularAC"
                              : "");
  if (kind == VarDCTDC || kind == ModularDC || kind == ACMetadata ||
      kind == ModularAC) {
    os << " group " << group_id;
  }
  if (kind == ModularAC) {
    os << " pass " << pass_id;
  }
  if (kind == QuantTable) {
    os << " " << quant_table_id;
  }
  return os.str();
}
#endif

Status ModularFrameDecoder::DecodeGlobalInfo(BitReader* reader,
                                             const FrameHeader& frame_header,
                                             bool allow_truncated_group) {
  JxlMemoryManager* memory_manager = this->memory_manager();
  bool decode_color = frame_header.encoding == FrameEncoding::kModular;
  const auto& metadata = frame_header.nonserialized_metadata->m;
  bool is_gray = metadata.color_encoding.IsGray();
  size_t nb_chans = 3;
  if (is_gray && frame_header.color_transform == ColorTransform::kNone) {
    nb_chans = 1;
  }
  do_color = decode_color;
  size_t nb_extra = metadata.extra_channel_info.size();
  bool has_tree = static_cast<bool>(reader->ReadBits(1));
  if (!allow_truncated_group ||
      reader->TotalBitsConsumed() < reader->TotalBytes() * kBitsPerByte) {
    if (has_tree) {
      size_t tree_size_limit =
          std::min(static_cast<size_t>(1 << 22),
                   1024 + frame_dim.xsize * frame_dim.ysize *
                              (nb_chans + nb_extra) / 16);
      JXL_RETURN_IF_ERROR(
          DecodeTree(memory_manager, reader, &tree, tree_size_limit));
      JXL_RETURN_IF_ERROR(DecodeHistograms(
          memory_manager, reader, (tree.size() + 1) / 2, &code, &context_map));
    }
  }
  if (!do_color) nb_chans = 0;

  bool fp = metadata.bit_depth.floating_point_sample;

  // bits_per_sample is just metadata for XYB images.
  if (metadata.bit_depth.bits_per_sample >= 32 && do_color &&
      frame_header.color_transform != ColorTransform::kXYB) {
    if (metadata.bit_depth.bits_per_sample == 32 && fp == false) {
      return JXL_FAILURE("uint32_t not supported in dec_modular");
    } else if (metadata.bit_depth.bits_per_sample > 32) {
      return JXL_FAILURE("bits_per_sample > 32 not supported");
    }
  }

  JXL_ASSIGN_OR_RETURN(
      Image gi,
      Image::Create(memory_manager, frame_dim.xsize, frame_dim.ysize,
                    metadata.bit_depth.bits_per_sample, nb_chans + nb_extra));

  all_same_shift = true;
  if (frame_header.color_transform == ColorTransform::kYCbCr) {
    for (size_t c = 0; c < nb_chans; c++) {
      gi.channel[c].hshift = frame_header.chroma_subsampling.HShift(c);
      gi.channel[c].vshift = frame_header.chroma_subsampling.VShift(c);
      size_t xsize_shifted =
          DivCeil(frame_dim.xsize, 1 << gi.channel[c].hshift);
      size_t ysize_shifted =
          DivCeil(frame_dim.ysize, 1 << gi.channel[c].vshift);
      JXL_RETURN_IF_ERROR(gi.channel[c].shrink(xsize_shifted, ysize_shifted));
      if (gi.channel[c].hshift != gi.channel[0].hshift ||
          gi.channel[c].vshift != gi.channel[0].vshift)
        all_same_shift = false;
    }
  }

  for (size_t ec = 0, c = nb_chans; ec < nb_extra; ec++, c++) {
    size_t ecups = frame_header.extra_channel_upsampling[ec];
    JXL_RETURN_IF_ERROR(
        gi.channel[c].shrink(DivCeil(frame_dim.xsize_upsampled, ecups),
                             DivCeil(frame_dim.ysize_upsampled, ecups)));
    gi.channel[c].hshift = gi.channel[c].vshift =
        CeilLog2Nonzero(ecups) - CeilLog2Nonzero(frame_header.upsampling);
    if (gi.channel[c].hshift != gi.channel[0].hshift ||
        gi.channel[c].vshift != gi.channel[0].vshift)
      all_same_shift = false;
  }

  JXL_DEBUG_V(6, "DecodeGlobalInfo: full_image (w/o transforms) %s",
              gi.DebugString().c_str());
  ModularOptions options;
  options.max_chan_size = frame_dim.group_dim;
  options.group_dim = frame_dim.group_dim;
  Status dec_status = ModularGenericDecompress(
      reader, gi, &global_header, ModularStreamId::Global().ID(frame_dim),
      &options,
      /*undo_transforms=*/false, &tree, &code, &context_map,
      allow_truncated_group);
  if (!allow_truncated_group) JXL_RETURN_IF_ERROR(dec_status);
  if (dec_status.IsFatalError()) {
    return JXL_FAILURE("Failed to decode global modular info");
  }

  // TODO(eustas): are we sure this can be done after partial decode?
  have_something = false;
  for (size_t c = 0; c < gi.channel.size(); c++) {
    Channel& gic = gi.channel[c];
    if (c >= gi.nb_meta_channels && gic.w <= frame_dim.group_dim &&
        gic.h <= frame_dim.group_dim)
      have_something = true;
  }
  // move global transforms to groups if possible
  if (!have_something && all_same_shift) {
    if (gi.transform.size() == 1 && gi.transform[0].id == TransformId::kRCT) {
      global_transform = gi.transform;
      gi.transform.clear();
      // TODO(jon): also move no-delta-palette out (trickier though)
    }
  }
  full_image = std::move(gi);
  JXL_DEBUG_V(6, "DecodeGlobalInfo: full_image (with transforms) %s",
              full_image.DebugString().c_str());
  return dec_status;
}

void ModularFrameDecoder::MaybeDropFullImage() {
  if (full_image.transform.empty() && !have_something && all_same_shift) {
    use_full_image = false;
    JXL_DEBUG_V(6, "Dropping full image");
    for (auto& ch : full_image.channel) {
      // keep metadata on channels around, but dealloc their planes
      ch.plane = Plane<pixel_type>();
    }
  }
}

Status ModularFrameDecoder::DecodeGroup(
    const FrameHeader& frame_header, const Rect& rect, BitReader* reader,
    int minShift, int maxShift, const ModularStreamId& stream, bool zerofill,
    PassesDecoderState* dec_state, RenderPipelineInput* render_pipeline_input,
    bool allow_truncated, bool* should_run_pipeline) {
  JXL_DEBUG_V(6, "Decoding %s with rect %s and shift bracket %d..%d %s",
              stream.DebugString().c_str(), Description(rect).c_str(), minShift,
              maxShift, zerofill ? "using zerofill" : "");
  JXL_ENSURE(stream.kind == ModularStreamId::Kind::ModularDC ||
             stream.kind == ModularStreamId::Kind::ModularAC);
  const size_t xsize = rect.xsize();
  const size_t ysize = rect.ysize();
  JXL_ASSIGN_OR_RETURN(Image gi, Image::Create(memory_manager_, xsize, ysize,
                                               full_image.bitdepth, 0));
  // start at the first bigger-than-groupsize non-metachannel
  size_t c = full_image.nb_meta_channels;
  for (; c < full_image.channel.size(); c++) {
    Channel& fc = full_image.channel[c];
    if (fc.w > frame_dim.group_dim || fc.h > frame_dim.group_dim) break;
  }
  size_t beginc = c;
  for (; c < full_image.channel.size(); c++) {
    Channel& fc = full_image.channel[c];
    int shift = std::min(fc.hshift, fc.vshift);
    if (shift > maxShift) continue;
    if (shift < minShift) continue;
    Rect r(rect.x0() >> fc.hshift, rect.y0() >> fc.vshift,
           rect.xsize() >> fc.hshift, rect.ysize() >> fc.vshift, fc.w, fc.h);
    if (r.xsize() == 0 || r.ysize() == 0) continue;
    if (zerofill && use_full_image) {
      for (size_t y = 0; y < r.ysize(); ++y) {
        pixel_type* const JXL_RESTRICT row_out = r.Row(&fc.plane, y);
        memset(row_out, 0, r.xsize() * sizeof(*row_out));
      }
    } else {
      JXL_ASSIGN_OR_RETURN(
          Channel gc, Channel::Create(memory_manager_, r.xsize(), r.ysize()));
      if (zerofill) ZeroFillImage(&gc.plane);
      gc.hshift = fc.hshift;
      gc.vshift = fc.vshift;
      gi.channel.emplace_back(std::move(gc));
    }
  }
  if (zerofill && use_full_image) return true;
  // Return early if there's nothing to decode. Otherwise there might be
  // problems later (in ModularImageToDecodedRect).
  if (gi.channel.empty()) {
    if (dec_state && should_run_pipeline) {
      const auto* metadata = frame_header.nonserialized_metadata;
      if (do_color || metadata->m.num_extra_channels > 0) {
        // Signal to FrameDecoder that we do not have some of the required input
        // for the render pipeline.
        *should_run_pipeline = false;
      }
    }
    JXL_DEBUG_V(6, "Nothing to decode, returning early.");
    return true;
  }
  ModularOptions options;
  if (!zerofill) {
    auto status = ModularGenericDecompress(
        reader, gi, /*header=*/nullptr, stream.ID(frame_dim), &options,
        /*undo_transforms=*/true, &tree, &code, &context_map, allow_truncated);
    if (!allow_truncated) JXL_RETURN_IF_ERROR(status);
    if (status.IsFatalError()) return status;
  }
  // Undo global transforms that have been pushed to the group level
  if (!use_full_image) {
    JXL_ENSURE(render_pipeline_input);
    for (const auto& t : global_transform) {
      JXL_RETURN_IF_ERROR(t.Inverse(gi, global_header.wp_header));
    }
    JXL_RETURN_IF_ERROR(ModularImageToDecodedRect(
        frame_header, gi, dec_state, nullptr, *render_pipeline_input,
        Rect(0, 0, gi.w, gi.h)));
    return true;
  }
  int gic = 0;
  for (c = beginc; c < full_image.channel.size(); c++) {
    Channel& fc = full_image.channel[c];
    int shift = std::min(fc.hshift, fc.vshift);
    if (shift > maxShift) continue;
    if (shift < minShift) continue;
    Rect r(rect.x0() >> fc.hshift, rect.y0() >> fc.vshift,
           rect.xsize() >> fc.hshift, rect.ysize() >> fc.vshift, fc.w, fc.h);
    if (r.xsize() == 0 || r.ysize() == 0) continue;
    JXL_ENSURE(use_full_image);
    JXL_RETURN_IF_ERROR(
        CopyImageTo(/*rect_from=*/Rect(0, 0, r.xsize(), r.ysize()),
                    /*from=*/gi.channel[gic].plane,
                    /*rect_to=*/r, /*to=*/&fc.plane));
    gic++;
  }
  return true;
}

Status ModularFrameDecoder::DecodeVarDCTDC(const FrameHeader& frame_header,
                                           size_t group_id, BitReader* reader,
                                           PassesDecoderState* dec_state) {
  JxlMemoryManager* memory_manager = dec_state->memory_manager();
  const Rect r = dec_state->shared->frame_dim.DCGroupRect(group_id);
  JXL_DEBUG_V(6, "Decoding VarDCT DC with rect %s", Description(r).c_str());
  // TODO(eustas): investigate if we could reduce the impact of
  //               EvalRationalPolynomial; generally speaking, the limit is
  //               2**(128/(3*magic)), where 128 comes from IEEE 754 exponent,
  //               3 comes from XybToRgb that cubes the values, and "magic" is
  //               the sum of all other contributions. 2**18 is known to lead
  //               to NaN on input found by fuzzing (see commit message).
  JXL_ASSIGN_OR_RETURN(Image image,
                       Image::Create(memory_manager, r.xsize(), r.ysize(),
                                     full_image.bitdepth, 3));
  size_t stream_id = ModularStreamId::VarDCTDC(group_id).ID(frame_dim);
  reader->Refill();
  size_t extra_precision = reader->ReadFixedBits<2>();
  float mul = 1.0f / (1 << extra_precision);
  ModularOptions options;
  for (size_t c = 0; c < 3; c++) {
    Channel& ch = image.channel[c < 2 ? c ^ 1 : c];
    ch.w >>= frame_header.chroma_subsampling.HShift(c);
    ch.h >>= frame_header.chroma_subsampling.VShift(c);
    JXL_RETURN_IF_ERROR(ch.shrink());
  }
  if (!ModularGenericDecompress(
          reader, image, /*header=*/nullptr, stream_id, &options,
          /*undo_transforms=*/true, &tree, &code, &context_map)) {
    return JXL_FAILURE("Failed to decode VarDCT DC group (DC group id %d)",
                       static_cast<int>(group_id));
  }
  DequantDC(r, &dec_state->shared_storage.dc_storage,
            &dec_state->shared_storage.quant_dc, image,
            dec_state->shared->quantizer.MulDC(), mul,
            dec_state->shared->cmap.base().DCFactors(),
            frame_header.chroma_subsampling, dec_state->shared->block_ctx_map);
  return true;
}

Status ModularFrameDecoder::DecodeAcMetadata(const FrameHeader& frame_header,
                                             size_t group_id, BitReader* reader,
                                             PassesDecoderState* dec_state) {
  JxlMemoryManager* memory_manager = dec_state->memory_manager();
  const Rect r = dec_state->shared->frame_dim.DCGroupRect(group_id);
  JXL_DEBUG_V(6, "Decoding AcMetadata with rect %s", Description(r).c_str());
  size_t upper_bound = r.xsize() * r.ysize();
  reader->Refill();
  size_t count = reader->ReadBits(CeilLog2Nonzero(upper_bound)) + 1;
  size_t stream_id = ModularStreamId::ACMetadata(group_id).ID(frame_dim);
  // YToX, YToB, ACS + QF, EPF
  JXL_ASSIGN_OR_RETURN(Image image,
                       Image::Create(memory_manager, r.xsize(), r.ysize(),
                                     full_image.bitdepth, 4));
  static_assert(kColorTileDimInBlocks == 8, "Color tile size changed");
  Rect cr(r.x0() >> 3, r.y0() >> 3, (r.xsize() + 7) >> 3, (r.ysize() + 7) >> 3);
  JXL_ASSIGN_OR_RETURN(
      image.channel[0],
      Channel::Create(memory_manager, cr.xsize(), cr.ysize(), 3, 3));
  JXL_ASSIGN_OR_RETURN(
      image.channel[1],
      Channel::Create(memory_manager, cr.xsize(), cr.ysize(), 3, 3));
  JXL_ASSIGN_OR_RETURN(image.channel[2],
                       Channel::Create(memory_manager, count, 2, 0, 0));
  ModularOptions options;
  if (!ModularGenericDecompress(
          reader, image, /*header=*/nullptr, stream_id, &options,
          /*undo_transforms=*/true, &tree, &code, &context_map)) {
    return JXL_FAILURE("Failed to decode AC metadata");
  }
  JXL_RETURN_IF_ERROR(
      ConvertPlaneAndClamp(Rect(image.channel[0].plane), image.channel[0].plane,
                           cr, &dec_state->shared_storage.cmap.ytox_map));
  JXL_RETURN_IF_ERROR(
      ConvertPlaneAndClamp(Rect(image.channel[1].plane), image.channel[1].plane,
                           cr, &dec_state->shared_storage.cmap.ytob_map));
  size_t num = 0;
  bool is444 = frame_header.chroma_subsampling.Is444();
  auto& ac_strategy = dec_state->shared_storage.ac_strategy;
  size_t xlim = std::min(ac_strategy.xsize(), r.x0() + r.xsize());
  size_t ylim = std::min(ac_strategy.ysize(), r.y0() + r.ysize());
  uint32_t local_used_acs = 0;
  for (size_t iy = 0; iy < r.ysize(); iy++) {
    size_t y = r.y0() + iy;
    int32_t* row_qf = r.Row(&dec_state->shared_storage.raw_quant_field, iy);
    uint8_t* row_epf = r.Row(&dec_state->shared_storage.epf_sharpness, iy);
    int32_t* row_in_1 = image.channel[2].plane.Row(0);
    int32_t* row_in_2 = image.channel[2].plane.Row(1);
    int32_t* row_in_3 = image.channel[3].plane.Row(iy);
    for (size_t ix = 0; ix < r.xsize(); ix++) {
      size_t x = r.x0() + ix;
      int sharpness = row_in_3[ix];
      if (sharpness < 0 || sharpness >= LoopFilter::kEpfSharpEntries) {
        return JXL_FAILURE("Corrupted sharpness field");
      }
      row_epf[ix] = sharpness;
      if (ac_strategy.IsValid(x, y)) {
        continue;
      }

      if (num >= count) return JXL_FAILURE("Corrupted stream");

      if (!AcStrategy::IsRawStrategyValid(row_in_1[num])) {
        return JXL_FAILURE("Invalid AC strategy");
      }
      local_used_acs |= 1u << row_in_1[num];
      AcStrategy acs = AcStrategy::FromRawStrategy(row_in_1[num]);
      if ((acs.covered_blocks_x() > 1 || acs.covered_blocks_y() > 1) &&
          !is444) {
        return JXL_FAILURE(
            "AC strategy not compatible with chroma subsampling");
      }
      // Ensure that blocks do not overflow *AC* groups.
      size_t next_x_ac_block = (x / kGroupDimInBlocks + 1) * kGroupDimInBlocks;
      size_t next_y_ac_block = (y / kGroupDimInBlocks + 1) * kGroupDimInBlocks;
      size_t next_x_dct_block = x + acs.covered_blocks_x();
      size_t next_y_dct_block = y + acs.covered_blocks_y();
      if (next_x_dct_block > next_x_ac_block || next_x_dct_block > xlim) {
        return JXL_FAILURE("Invalid AC strategy, x overflow");
      }
      if (next_y_dct_block > next_y_ac_block || next_y_dct_block > ylim) {
        return JXL_FAILURE("Invalid AC strategy, y overflow");
      }
      JXL_RETURN_IF_ERROR(
          ac_strategy.SetNoBoundsCheck(x, y, AcStrategyType(row_in_1[num])));
      row_qf[ix] = 1 + std::max<int32_t>(0, std::min(Quantizer::kQuantMax - 1,
                                                     row_in_2[num]));
      num++;
    }
  }
  dec_state->used_acs |= local_used_acs;
  if (frame_header.loop_filter.epf_iters > 0) {
    JXL_RETURN_IF_ERROR(ComputeSigma(frame_header.loop_filter, r, dec_state));
  }
  return true;
}

Status ModularFrameDecoder::ModularImageToDecodedRect(
    const FrameHeader& frame_header, Image& gi, PassesDecoderState* dec_state,
    jxl::ThreadPool* pool, RenderPipelineInput& render_pipeline_input,
    Rect modular_rect) const {
  const auto* metadata = frame_header.nonserialized_metadata;
  JXL_ENSURE(gi.transform.empty());

  auto get_row = [&](size_t c, size_t y) {
    const auto& buffer = render_pipeline_input.GetBuffer(c);
    return buffer.second.Row(buffer.first, y);
  };

  size_t c = 0;
  if (do_color) {
    const bool rgb_from_gray =
        metadata->m.color_encoding.IsGray() &&
        frame_header.color_transform == ColorTransform::kNone;
    const bool fp = metadata->m.bit_depth.floating_point_sample &&
                    frame_header.color_transform != ColorTransform::kXYB;
    for (; c < 3; c++) {
      double factor = full_image.bitdepth < 32
                          ? 1.0 / ((1u << full_image.bitdepth) - 1)
                          : 0;
      size_t c_in = c;
      if (frame_header.color_transform == ColorTransform::kXYB) {
        factor = dec_state->shared->matrices.DCQuants()[c];
        // XYB is encoded as YX(B-Y)
        if (c < 2) c_in = 1 - c;
      } else if (rgb_from_gray) {
        c_in = 0;
      }
      JXL_ENSURE(c_in < gi.channel.size());
      Channel& ch_in = gi.channel[c_in];
      // TODO(eustas): could we detect it on earlier stage?
      if (ch_in.w == 0 || ch_in.h == 0) {
        return JXL_FAILURE("Empty image");
      }
      JXL_ENSURE(ch_in.hshift <= 3 && ch_in.vshift <= 3);
      Rect r = render_pipeline_input.GetBuffer(c).second;
      Rect mr(modular_rect.x0() >> ch_in.hshift,
              modular_rect.y0() >> ch_in.vshift,
              DivCeil(modular_rect.xsize(), 1 << ch_in.hshift),
              DivCeil(modular_rect.ysize(), 1 << ch_in.vshift));
      mr = mr.Crop(ch_in.plane);
      size_t xsize_shifted = r.xsize();
      size_t ysize_shifted = r.ysize();
      if (r.ysize() != mr.ysize() || r.xsize() != mr.xsize()) {
        return JXL_FAILURE("Dimension mismatch: trying to fit a %" PRIuS
                           "x%" PRIuS
                           " modular channel into "
                           "a %" PRIuS "x%" PRIuS " rect",
                           mr.xsize(), mr.ysize(), r.xsize(), r.ysize());
      }
      if (frame_header.color_transform == ColorTransform::kXYB && c == 2) {
        JXL_ENSURE(!fp);
        const auto process_row = [&](const uint32_t task,
                                     size_t /* thread */) -> Status {
          const size_t y = task;
          const pixel_type* const JXL_RESTRICT row_in = mr.Row(&ch_in.plane, y);
          const pixel_type* const JXL_RESTRICT row_in_Y =
              mr.Row(&gi.channel[0].plane, y);
          float* const JXL_RESTRICT row_out = get_row(c, y);
          HWY_DYNAMIC_DISPATCH(MultiplySum)
          (xsize_shifted, row_in, row_in_Y, factor, row_out);
          return true;
        };
        JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, ysize_shifted,
                                      ThreadPool::NoInit, process_row,
                                      "ModularIntToFloat"));
      } else if (fp) {
        int bits = metadata->m.bit_depth.bits_per_sample;
        int exp_bits = metadata->m.bit_depth.exponent_bits_per_sample;
        const auto process_row = [&](const uint32_t task,
                                     size_t /* thread */) -> Status {
          const size_t y = task;
          const pixel_type* const JXL_RESTRICT row_in = mr.Row(&ch_in.plane, y);
          if (rgb_from_gray) {
            for (size_t cc = 0; cc < 3; cc++) {
              float* const JXL_RESTRICT row_out = get_row(cc, y);
              JXL_RETURN_IF_ERROR(
                  int_to_float(row_in, row_out, xsize_shifted, bits, exp_bits));
            }
          } else {
            float* const JXL_RESTRICT row_out = get_row(c, y);
            JXL_RETURN_IF_ERROR(
                int_to_float(row_in, row_out, xsize_shifted, bits, exp_bits));
          }
          return true;
        };
        JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, ysize_shifted,
                                      ThreadPool::NoInit, process_row,
                                      "ModularIntToFloat_losslessfloat"));
      } else {
        const auto process_row = [&](const uint32_t task,
                                     size_t /* thread */) -> Status {
          const size_t y = task;
          const pixel_type* const JXL_RESTRICT row_in = mr.Row(&ch_in.plane, y);
          if (rgb_from_gray) {
            if (full_image.bitdepth < 23) {
              HWY_DYNAMIC_DISPATCH(RgbFromSingle)
              (xsize_shifted, row_in, factor, get_row(0, y), get_row(1, y),
               get_row(2, y));
            } else {
              SingleFromSingleAccurate(xsize_shifted, row_in, factor,
                                       get_row(0, y));
              SingleFromSingleAccurate(xsize_shifted, row_in, factor,
                                       get_row(1, y));
              SingleFromSingleAccurate(xsize_shifted, row_in, factor,
                                       get_row(2, y));
            }
          } else {
            float* const JXL_RESTRICT row_out = get_row(c, y);
            if (full_image.bitdepth < 23) {
              HWY_DYNAMIC_DISPATCH(SingleFromSingle)
              (xsize_shifted, row_in, factor, row_out);
            } else {
              SingleFromSingleAccurate(xsize_shifted, row_in, factor, row_out);
            }
          }
          return true;
        };
        JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, ysize_shifted,
                                      ThreadPool::NoInit, process_row,
                                      "ModularIntToFloat"));
      }
      if (rgb_from_gray) {
        break;
      }
    }
    if (rgb_from_gray) {
      c = 1;
    }
  }
  size_t num_extra_channels = metadata->m.num_extra_channels;
  for (size_t ec = 0; ec < num_extra_channels; ec++, c++) {
    const ExtraChannelInfo& eci = metadata->m.extra_channel_info[ec];
    int bits = eci.bit_depth.bits_per_sample;
    int exp_bits = eci.bit_depth.exponent_bits_per_sample;
    bool fp = eci.bit_depth.floating_point_sample;
    JXL_ENSURE(fp || bits < 32);
    const double factor = fp ? 0 : (1.0 / ((1u << bits) - 1));
    JXL_ENSURE(c < gi.channel.size());
    Channel& ch_in = gi.channel[c];
    const auto& buffer = render_pipeline_input.GetBuffer(3 + ec);
    Rect r = buffer.second;
    Rect mr(modular_rect.x0() >> ch_in.hshift,
            modular_rect.y0() >> ch_in.vshift,
            DivCeil(modular_rect.xsize(), 1 << ch_in.hshift),
            DivCeil(modular_rect.ysize(), 1 << ch_in.vshift));
    mr = mr.Crop(ch_in.plane);
    if (r.ysize() != mr.ysize() || r.xsize() != mr.xsize()) {
      return JXL_FAILURE("Dimension mismatch: trying to fit a %" PRIuS
                         "x%" PRIuS
                         " modular channel into "
                         "a %" PRIuS "x%" PRIuS " rect",
                         mr.xsize(), mr.ysize(), r.xsize(), r.ysize());
    }
    for (size_t y = 0; y < r.ysize(); ++y) {
      float* const JXL_RESTRICT row_out = r.Row(buffer.first, y);
      const pixel_type* const JXL_RESTRICT row_in = mr.Row(&ch_in.plane, y);
      if (fp) {
        JXL_RETURN_IF_ERROR(
            int_to_float(row_in, row_out, r.xsize(), bits, exp_bits));
      } else {
        if (full_image.bitdepth < 23) {
          HWY_DYNAMIC_DISPATCH(SingleFromSingle)
          (r.xsize(), row_in, factor, row_out);
        } else {
          SingleFromSingleAccurate(r.xsize(), row_in, factor, row_out);
        }
      }
    }
  }
  return true;
}

Status ModularFrameDecoder::FinalizeDecoding(const FrameHeader& frame_header,
                                             PassesDecoderState* dec_state,
                                             jxl::ThreadPool* pool,
                                             bool inplace) {
  if (!use_full_image) return true;
  JxlMemoryManager* memory_manager = dec_state->memory_manager();
  Image gi{memory_manager};
  if (inplace) {
    gi = std::move(full_image);
  } else {
    JXL_ASSIGN_OR_RETURN(gi, Image::Clone(full_image));
  }
  size_t xsize = gi.w;
  size_t ysize = gi.h;

  JXL_DEBUG_V(3, "Finalizing decoding for modular image: %s",
              gi.DebugString().c_str());

  // Don't use threads if total image size is smaller than a group
  if (xsize * ysize < frame_dim.group_dim * frame_dim.group_dim) pool = nullptr;

  // Undo the global transforms
  gi.undo_transforms(global_header.wp_header, pool);
  JXL_ENSURE(global_transform.empty());
  if (gi.error) return JXL_FAILURE("Undoing transforms failed");

  for (size_t i = 0; i < dec_state->shared->frame_dim.num_groups; i++) {
    dec_state->render_pipeline->ClearDone(i);
  }

  const auto init = [&](size_t num_threads) -> Status {
    bool use_group_ids = (frame_header.encoding == FrameEncoding::kVarDCT ||
                          (frame_header.flags & FrameHeader::kNoise));
    JXL_RETURN_IF_ERROR(dec_state->render_pipeline->PrepareForThreads(
        num_threads, use_group_ids));
    return true;
  };
  const auto process_group = [&](const uint32_t group,
                                 size_t thread_id) -> Status {
    RenderPipelineInput input =
        dec_state->render_pipeline->GetInputBuffers(group, thread_id);
    JXL_RETURN_IF_ERROR(ModularImageToDecodedRect(
        frame_header, gi, dec_state, nullptr, input,
        dec_state->shared->frame_dim.GroupRect(group)));
    JXL_RETURN_IF_ERROR(input.Done());
    return true;
  };
  JXL_RETURN_IF_ERROR(RunOnPool(pool, 0,
                                dec_state->shared->frame_dim.num_groups, init,
                                process_group, "ModularToRect"));
  return true;
}

static constexpr const float kAlmostZero = 1e-8f;

Status ModularFrameDecoder::DecodeQuantTable(
    JxlMemoryManager* memory_manager, size_t required_size_x,
    size_t required_size_y, BitReader* br, QuantEncoding* encoding, size_t idx,
    ModularFrameDecoder* modular_frame_decoder) {
  JXL_RETURN_IF_ERROR(F16Coder::Read(br, &encoding->qraw.qtable_den));
  if (encoding->qraw.qtable_den < kAlmostZero) {
    // qtable[] values are already checked for <= 0 so the denominator may not
    // be negative.
    return JXL_FAILURE("Invalid qtable_den: value too small");
  }
  JXL_ASSIGN_OR_RETURN(
      Image image,
      Image::Create(memory_manager, required_size_x, required_size_y, 8, 3));
  ModularOptions options;
  if (modular_frame_decoder) {
    JXL_ASSIGN_OR_RETURN(ModularStreamId qt, ModularStreamId::QuantTable(idx));
    JXL_RETURN_IF_ERROR(ModularGenericDecompress(
        br, image, /*header=*/nullptr, qt.ID(modular_frame_decoder->frame_dim),
        &options, /*undo_transforms=*/true, &modular_frame_decoder->tree,
        &modular_frame_decoder->code, &modular_frame_decoder->context_map));
  } else {
    JXL_RETURN_IF_ERROR(ModularGenericDecompress(br, image, /*header=*/nullptr,
                                                 0, &options,
                                                 /*undo_transforms=*/true));
  }
  if (!encoding->qraw.qtable) {
    encoding->qraw.qtable =
        new std::vector<int>(required_size_x * required_size_y * 3);
  } else {
    JXL_ENSURE(encoding->qraw.qtable->size() ==
               required_size_x * required_size_y * 3);
  }
  int* qtable = encoding->qraw.qtable->data();
  for (size_t c = 0; c < 3; c++) {
    for (size_t y = 0; y < required_size_y; y++) {
      int32_t* JXL_RESTRICT row = image.channel[c].Row(y);
      for (size_t x = 0; x < required_size_x; x++) {
        qtable[c * required_size_x * required_size_y + y * required_size_x +
               x] = row[x];
        if (row[x] <= 0) {
          return JXL_FAILURE("Invalid raw quantization table");
        }
      }
    }
  }
  return true;
}

}  // namespace jxl
#endif  // HWY_ONCE
