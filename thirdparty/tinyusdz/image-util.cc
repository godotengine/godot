// SPDX-License-Identifier: Apache 2.0
// Copyright 2023-Present, Light Transport Entertainment Inc.
//
// TODO
// - [ ] Optimize Rec.709 conversion
//
#include <cmath>
#include <sstream>

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#endif

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Warray-bounds"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#endif

#if !defined(TINYUSDZ_NO_STB_IMAGE_RESIZE_IMPLEMENTATION)
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#endif

//#include "external/stb_image_resize.h"
#include "external/stb_image_resize2.h"

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#include "image-util.hh"
#include "value-types.hh"
#include "common-macros.inc"
#include "tiny-format.hh"

#if defined(TINYUSDZ_WITH_COLORIO)
#include "external/tiny-color-io.h"
#endif

#define PushError(msg) \
  if (err) { \
    (*err) += msg; \
  }

// From https://www.nayuki.io/page/srgb-transform-library --------------------
/*
 * sRGB transform (C++)
 *
 * Copyright (c) 2017 Project Nayuki. (MIT License)
 * https://www.nayuki.io/page/srgb-transform-library
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 * - The above copyright notice and this permission notice shall be included in
 *   all copies or substantial portions of the Software.
 * - The Software is provided "as is", without warranty of any kind, express or
 *   implied, including but not limited to the warranties of merchantability,
 *   fitness for a particular purpose and noninfringement. In no event shall the
 *   authors or copyright holders be liable for any claim, damages or other
 *   liability, whether in an action of contract, tort or otherwise, arising from,
 *   out of or in connection with the Software or the use or other dealings in the
 *   Software.
 */

namespace SrgbTransform {

/*---- sRGB values to linear intensities ----*/
float srgbToLinear(float x);
double srgbToLinear(double x);
float linearToSrgb(float x);
double linearToSrgb(double x);
uint8_t linearToSrgb8bit(float x);
uint8_t linearToSrgb8bit(double x);

float srgbToLinear(float x) {
	if (x <= 0.0f)
		return 0.0f;
	else if (x >= 1.0f)
		return 1.0f;
	else if (x < 0.04045f)
		return x / 12.92f;
	else
		return std::pow((x + 0.055f) / 1.055f, 2.4f);
}


double srgbToLinear(double x) {
	if (x <= 0.0)
		return 0.0;
	else if (x >= 1.0)
		return 1.0;
	else if (x < 0.04045)
		return x / 12.92;
	else
		return std::pow((x + 0.055) / 1.055, 2.4);
}


const float SRGB_8BIT_TO_LINEAR_FLOAT[1 << 8] = {
	0.0f, 3.03527e-4f, 6.07054e-4f, 9.10581e-4f,
	0.001214108f, 0.001517635f, 0.001821162f, 0.0021246888f,
	0.002428216f, 0.002731743f, 0.00303527f, 0.0033465358f,
	0.0036765074f, 0.004024717f, 0.004391442f, 0.0047769537f,
	0.005181517f, 0.005605392f, 0.0060488335f, 0.006512091f,
	0.0069954107f, 0.007499032f, 0.008023193f, 0.008568126f,
	0.009134059f, 0.009721218f, 0.010329823f, 0.010960095f,
	0.011612245f, 0.012286489f, 0.0129830325f, 0.013702083f,
	0.014443845f, 0.015208516f, 0.015996294f, 0.016807377f,
	0.017641956f, 0.018500222f, 0.019382363f, 0.020288564f,
	0.021219011f, 0.022173885f, 0.023153368f, 0.024157634f,
	0.025186861f, 0.026241222f, 0.027320893f, 0.02842604f,
	0.029556835f, 0.030713445f, 0.031896032f, 0.033104766f,
	0.034339808f, 0.035601314f, 0.036889452f, 0.038204372f,
	0.039546236f, 0.0409152f, 0.04231141f, 0.04373503f,
	0.045186203f, 0.046665087f, 0.048171826f, 0.049706567f,
	0.051269464f, 0.05286065f, 0.05448028f, 0.056128494f,
	0.057805438f, 0.059511244f, 0.06124606f, 0.06301002f,
	0.06480327f, 0.066625945f, 0.068478175f, 0.0703601f,
	0.07227185f, 0.07421357f, 0.07618539f, 0.07818743f,
	0.08021983f, 0.082282715f, 0.084376216f, 0.086500466f,
	0.08865559f, 0.09084172f, 0.093058966f, 0.09530747f,
	0.097587354f, 0.09989873f, 0.10224174f, 0.10461649f,
	0.107023105f, 0.10946172f, 0.111932434f, 0.11443538f,
	0.11697067f, 0.119538434f, 0.122138776f, 0.12477182f,
	0.12743768f, 0.13013647f, 0.13286832f, 0.13563333f,
	0.13843162f, 0.14126329f, 0.14412847f, 0.14702727f,
	0.14995979f, 0.15292616f, 0.15592647f, 0.15896083f,
	0.16202939f, 0.1651322f, 0.1682694f, 0.17144111f,
	0.1746474f, 0.17788842f, 0.18116425f, 0.18447499f,
	0.18782078f, 0.19120169f, 0.19461784f, 0.19806932f,
	0.20155625f, 0.20507874f, 0.20863687f, 0.21223076f,
	0.21586053f, 0.21952623f, 0.22322798f, 0.2269659f,
	0.23074007f, 0.23455061f, 0.2383976f, 0.24228115f,
	0.24620135f, 0.2501583f, 0.25415212f, 0.25818288f,
	0.2622507f, 0.26635563f, 0.27049783f, 0.27467734f,
	0.2788943f, 0.28314877f, 0.28744087f, 0.29177067f,
	0.2961383f, 0.3005438f, 0.30498734f, 0.30946895f,
	0.31398875f, 0.3185468f, 0.32314324f, 0.32777813f,
	0.33245155f, 0.33716366f, 0.34191445f, 0.3467041f,
	0.35153264f, 0.35640016f, 0.36130682f, 0.36625263f,
	0.3712377f, 0.37626216f, 0.38132605f, 0.38642946f,
	0.3915725f, 0.39675525f, 0.4019778f, 0.40724024f,
	0.41254264f, 0.4178851f, 0.4232677f, 0.42869052f,
	0.43415368f, 0.4396572f, 0.44520122f, 0.45078582f,
	0.45641103f, 0.46207702f, 0.4677838f, 0.4735315f,
	0.4793202f, 0.48514995f, 0.4910209f, 0.496933f,
	0.5028865f, 0.50888133f, 0.5149177f, 0.5209956f,
	0.52711517f, 0.53327644f, 0.5394795f, 0.5457245f,
	0.55201143f, 0.55834043f, 0.5647115f, 0.57112485f,
	0.57758045f, 0.58407843f, 0.59061885f, 0.5972018f,
	0.60382736f, 0.61049557f, 0.6172066f, 0.62396044f,
	0.63075715f, 0.6375969f, 0.6444797f, 0.65140563f,
	0.65837485f, 0.66538733f, 0.67244315f, 0.6795425f,
	0.6866853f, 0.6938718f, 0.7011019f, 0.7083758f,
	0.71569353f, 0.7230551f, 0.73046076f, 0.73791045f,
	0.74540424f, 0.7529422f, 0.7605245f, 0.76815116f,
	0.7758222f, 0.7835378f, 0.791298f, 0.7991027f,
	0.8069523f, 0.8148466f, 0.82278574f, 0.8307699f,
	0.838799f, 0.8468732f, 0.8549926f, 0.8631572f,
	0.8713671f, 0.8796224f, 0.8879231f, 0.8962694f,
	0.9046612f, 0.91309863f, 0.92158186f, 0.9301109f,
	0.9386857f, 0.9473065f, 0.9559733f, 0.9646863f,
	0.9734453f, 0.9822506f, 0.9911021f, 1.0f,
};


const double SRGB_8BIT_TO_LINEAR_DOUBLE[1 << 8] = {
	0.0, 3.035269835488375e-4, 6.07053967097675e-4, 9.105809506465125e-4,
	0.00121410793419535, 0.0015176349177441874, 0.001821161901293025, 0.0021246888848418626,
	0.0024282158683907, 0.0027317428519395373, 0.003035269835488375, 0.003346535763899161,
	0.003676507324047436, 0.004024717018496307, 0.004391442037410293, 0.004776953480693729,
	0.005181516702338386, 0.005605391624202723, 0.006048833022857054, 0.006512090792594475,
	0.006995410187265387, 0.007499032043226175, 0.008023192985384994, 0.008568125618069307,
	0.009134058702220787, 0.00972121732023785, 0.010329823029626936, 0.010960094006488246,
	0.011612245179743885, 0.012286488356915872, 0.012983032342173012, 0.013702083047289686,
	0.014443843596092545, 0.01520851442291271, 0.01599629336550963, 0.016807375752887384,
	0.017641954488384078, 0.018500220128379697, 0.019382360956935723, 0.0202885630566524,
	0.021219010376003555, 0.022173884793387385, 0.02315336617811041, 0.024157632448504756,
	0.02518685962736163, 0.026241221894849898, 0.027320891639074894, 0.028426039504420793,
	0.0295568344378088, 0.030713443732993635, 0.03189603307301153, 0.033104766570885055,
	0.03433980680868217, 0.03560131487502034, 0.03688945040110004, 0.0382043715953465,
	0.03954623527673284, 0.04091519690685319, 0.042311410620809675, 0.043735029256973465,
	0.04518620438567554, 0.046665086336880095, 0.04817182422688942, 0.04970656598412723,
	0.05126945837404324, 0.052860647023180246, 0.05448027644244237, 0.05612849004960009,
	0.05780543019106723, 0.0595112381629812, 0.06124605423161761, 0.06301001765316767,
	0.06480326669290577, 0.06662593864377289, 0.06847816984440017, 0.07036009569659588,
	0.07227185068231748, 0.07421356838014963, 0.07618538148130785, 0.07818742180518633,
	0.08021982031446832, 0.0822827071298148, 0.08437621154414882, 0.08650046203654976,
	0.08865558628577294, 0.09084171118340768, 0.09305896284668745, 0.0953074666309647,
	0.09758734714186246, 0.09989872824711389, 0.10224173308810132, 0.10461648409110419,
	0.10702310297826761, 0.10946171077829933, 0.1119324278369056, 0.11443537382697373,
	0.11697066775851084, 0.11953842798834562, 0.12213877222960187, 0.12477181756095049,
	0.12743768043564743, 0.1301364766903643, 0.13286832155381798, 0.13563332965520566,
	0.13843161503245183, 0.14126329114027164, 0.14412847085805777, 0.14702726649759498,
	0.14995978981060856, 0.15292615199615017, 0.1559264637078274, 0.1589608350608804,
	0.162029375639111, 0.1651321945016676, 0.16826940018969075, 0.1714411007328226,
	0.17464740365558504, 0.17788841598362912, 0.18116424424986022, 0.184474994500441,
	0.18782077230067787, 0.19120168274079138, 0.1946178304415758, 0.19806931955994886,
	0.20155625379439707, 0.20507873639031693, 0.20863687014525575, 0.21223075741405523,
	0.21586050011389926, 0.2195261997292692, 0.2232279573168085, 0.22696587351009836,
	0.23074004852434915, 0.23455058216100522, 0.238397573812271, 0.24228112246555486,
	0.24620132670783548, 0.25015828472995344, 0.25415209433082675, 0.2581828529215958,
	0.26225065752969623, 0.26635560480286247, 0.2704977910130658, 0.27467731206038465,
	0.2788942634768104, 0.2831487404299921, 0.2874408377269175, 0.29177064981753587,
	0.2961382707983211, 0.3005437944157765, 0.3049873140698863, 0.30946892281750854,
	0.31398871337571754, 0.31854677812509186, 0.32314320911295075, 0.3277780980565422,
	0.33245153634617935, 0.33716361504833037, 0.3419144249086609, 0.3467040563550296,
	0.35153259950043936, 0.3564001441459435, 0.3613067797835095, 0.3662525955988395,
	0.3712376804741491, 0.3762621229909065, 0.38132601143253014, 0.386429433787049,
	0.39157247774972326, 0.39675523072562685, 0.4019777798321958, 0.4072402119017367,
	0.41254261348390375, 0.4178850708481375, 0.4232676699860717, 0.4286904966139066,
	0.43415363617474895, 0.4396571738409188, 0.44520119451622786, 0.45078578283822346,
	0.45641102318040466, 0.4620769996544071, 0.467783796112159, 0.47353149614800955,
	0.4793201831008268, 0.4851499400560704, 0.4910208498478356, 0.4969329950608704,
	0.5028864580325687, 0.5088813208549338, 0.5149176653765214, 0.5209955732043543,
	0.5271151257058131, 0.5332764040105052, 0.5394794890121072, 0.5457244613701866,
	0.5520114015120001, 0.5583403896342679, 0.5647115057049292, 0.5711248294648731,
	0.5775804404296506, 0.5840784178911641, 0.5906188409193369, 0.5972017883637634,
	0.6038273388553378, 0.6104955708078648, 0.6172065624196511, 0.6239603916750761,
	0.6307571363461468, 0.6375968739940326, 0.6444796819705821, 0.6514056374198242,
	0.6583748172794485, 0.665387298282272, 0.6724431569576875, 0.6795424696330938,
	0.6866853124353135, 0.6938717612919899, 0.7011018919329731, 0.7083757798916868,
	0.7156935005064807, 0.7230551289219693, 0.7304607400903537, 0.7379104087727308,
	0.7454042095403874, 0.7529422167760779, 0.7605245046752924, 0.768151147247507,
	0.7758222183174236, 0.7835377915261935, 0.7912979403326302, 0.799102738014409,
	0.8069522576692516, 0.8148465722161012, 0.8227857543962835, 0.8307698767746546,
	0.83879901174074, 0.846873231509858, 0.8549926081242338, 0.8631572134541023,
	0.8713671191987972, 0.8796223968878317, 0.8879231178819663, 0.8962693533742664,
	0.9046611743911496, 0.9130986517934192, 0.9215818562772946, 0.9301108583754237,
	0.938685728457888, 0.9473065367331999, 0.9559733532492861, 0.9646862478944651,
	0.9734452903984125, 0.9822505503331171, 0.9911020971138298, 1.0,
};



/*---- Linear intensities to sRGB values ----*/

float linearToSrgb(float x) {
	if (x <= 0.0f)
		return 0.0f;
	else if (x >= 1.0f)
		return 1.0f;
	else if (x < 0.0031308f)
		return x * 12.92f;
	else
		return std::pow(x, 1.0f / 2.4f) * 1.055f - 0.055f;
}


double linearToSrgb(double x) {
	if (x <= 0.0)
		return 0.0;
	else if (x >= 1.0)
		return 1.0;
	else if (x < 0.0031308)
		return x * 12.92;
	else
		return std::pow(x, 1.0 / 2.4) * 1.055 - 0.055;
}

uint8_t linearToSrgb8bit(float x) {
	if (x <= 0.0f)
		return 0;
	if (x >= 1.0f)
		return 255;
	const float *TABLE = SRGB_8BIT_TO_LINEAR_FLOAT;
	int y = 0;
	for (int i = 128; i != 0; i >>= 1) {
		if (TABLE[y + i] <= x)
			y += i;
	}
	if (x - TABLE[y] <= TABLE[y + 1] - x)
		return static_cast<uint8_t>((std::max)(0, (std::min)(255, y)));
	else
		return static_cast<uint8_t>((std::max)(0, (std::min)(255, y + 1)));
}

uint8_t linearToSrgb8bit(double x) {
	if (x <= 0.0)
		return 0;
	if (x >= 1.0)
		return 255;
	const double *TABLE = SRGB_8BIT_TO_LINEAR_DOUBLE;
	int y = 0;
	for (int i = 128; i != 0; i >>= 1) {
		if (TABLE[y + i] <= x)
			y += i;
	}
	if (x - TABLE[y] <= TABLE[y + 1] - x)
		return static_cast<uint8_t>((std::max)(0, (std::min)(255, y)));
	else
		return static_cast<uint8_t>((std::max)(0, (std::min)(255, y + 1)));
}

// ----------------------------------------------------------------------------

} // SrgbTransform

namespace tinyusdz {

namespace detail {

uint8_t f32_to_u8(float x);
uint8_t linearToRec709_8bit(float L);
float Rec709ToLinear(uint8_t v);

uint8_t f32_to_u8(float x) {
  return static_cast<uint8_t>((std::max)(0, (std::min)(int(x * 255.0f), 255)));
}

// Naiive implementation of Rec.709
//
// https://en.wikipedia.org/wiki/Rec._709

uint8_t linearToRec709_8bit(float L) {
  float V;
  if (L > 1.0f) {
    V = 1.0f;
  } else if (L < 0.018f) {
    V = 4.5f * L;
  } else {
    // 0.45 ~= 1/2.2
    V = 1.099f * std::pow(L, 0.45f) - 0.099f;
  }

  return static_cast<uint8_t>((std::max)(0, (std::min)(255, int(V))));
}

float Rec709ToLinear(uint8_t v) {
  float V = v / 255.0f;

  float L;
  if (V > 0.081f) {
    L = V / 4.5f;
  } else {
    L = std::pow((V + 0.099f)/1.099f, (1.0f/0.45f));
  }

  return L;

}

} // namespace detail

bool linear_f32_to_srgb_8bit(const std::vector<float> &in_img, size_t width,
                         size_t height,
                         size_t channels, size_t channel_stride,
                         std::vector<uint8_t> *out_img, std::string *err) {

  if (width == 0) {
    PUSH_ERROR_AND_RETURN("width is zero.");
  }

  if (height == 0) {
    PUSH_ERROR_AND_RETURN("height is zero.");
  }

  if (channels == 0) {
    PUSH_ERROR_AND_RETURN("channels is zero.");
  }

  if (out_img == nullptr) {
    PUSH_ERROR_AND_RETURN("`out_img` is nullptr.");
  }

  if (channel_stride == 0) {
    channel_stride = channels;
  } else {
    if (channel_stride < channels) {
      PUSH_ERROR_AND_RETURN(fmt::format("channel_stride {} is smaller than input channels {}", channel_stride, channels));
    }
  }

  size_t dest_size = size_t(width) * size_t(height) * channel_stride;
  if (dest_size > in_img.size()) {
    PUSH_ERROR_AND_RETURN(fmt::format("Insufficient input buffer size. must be the same or larger than {} but has {}", dest_size, in_img.size()));
  }

  out_img->resize(dest_size);

  for (size_t y = 0; y < height; y++) {
    for (size_t x = 0; x < width; x++) {
      for (size_t c = 0; c < channels; c++) {
        size_t idx = channel_stride * width * y + channel_stride * x + c;
        (*out_img)[idx] = SrgbTransform::linearToSrgb8bit(in_img[idx]);
      }

      // remainder(usually alpha channel)
      // Apply linear conversion.
      for (size_t c = channels; c < channel_stride; c++) {
        size_t idx = channel_stride * width * y + channel_stride * x + c;
        (*out_img)[idx] = detail::f32_to_u8(in_img[idx]);
      }
    }
  }

  return true;
}

bool srgb_8bit_to_linear_f32(const std::vector<uint8_t> &in_img, size_t width,
                         size_t height,
                         size_t channels, size_t channel_stride,
                         std::vector<float> *out_img, std::string *err) {

  if (width == 0) {
    PUSH_ERROR_AND_RETURN("width is zero.");
  }

  if (height == 0) {
    PUSH_ERROR_AND_RETURN("height is zero.");
  }

  if (channels == 0) {
    PUSH_ERROR_AND_RETURN("channels is zero.");
  }

  if (out_img == nullptr) {
    PUSH_ERROR_AND_RETURN("`out_img` is nullptr.");
  }

  if (channel_stride == 0) {
    channel_stride = channels;
  } else {
    if (channel_stride < channels) {
      PUSH_ERROR_AND_RETURN(fmt::format("channel_stride {} is smaller than input channels {}", channel_stride, channels));
    }
  }

  size_t dest_size = size_t(width) * size_t(height) * channel_stride;
  if (dest_size > in_img.size()) {
    PUSH_ERROR_AND_RETURN(fmt::format("Insufficient input buffer size. must be the same or larger than {} but has {}", dest_size, in_img.size()));
  }

  out_img->resize(dest_size);

  // TODO: Use table approach for larger image size?

  for (size_t y = 0; y < height; y++) {
    for (size_t x = 0; x < width; x++) {
      for (size_t c = 0; c < channels; c++) {
        size_t idx = channel_stride * width * y + channel_stride * x + c;
        (*out_img)[idx] = SrgbTransform::srgbToLinear(float(in_img[idx]) / 255.0f);
      }

      // remainder(usually alpha channel)
      // Apply linear conversion.
      for (size_t c = channels; c < channel_stride; c++) {
        size_t idx = channel_stride * width * y + channel_stride * x + c;
        (*out_img)[idx] = float(in_img[idx]) / 255.0f;
      }
    }
  }

  return true;
}

bool srgb_f32_to_linear_f32(const std::vector<float> &in_img, size_t width,
                         size_t height,
                         size_t channels, size_t channel_stride,
                         std::vector<float> *out_img, const float scale_factor, const float bias, const float alpha_scale_factor, const float alpha_bias, std::string *err) {

  if (width == 0) {
    PUSH_ERROR_AND_RETURN("width is zero.");
  }

  if (height == 0) {
    PUSH_ERROR_AND_RETURN("height is zero.");
  }

  if (channels == 0) {
    PUSH_ERROR_AND_RETURN("channels is zero.");
  }

  if (out_img == nullptr) {
    PUSH_ERROR_AND_RETURN("`out_img` is nullptr.");
  }

  if (channel_stride == 0) {
    channel_stride = channels;
  } else {
    if (channel_stride < channels) {
      PUSH_ERROR_AND_RETURN(fmt::format("channel_stride {} is smaller than input channels {}", channel_stride, channels));
    }
  }

  size_t dest_size = size_t(width) * size_t(height) * channel_stride;
  if (dest_size > in_img.size()) {
    PUSH_ERROR_AND_RETURN(fmt::format("Insufficient input buffer size. must be the same or larger than {} but has {}", dest_size, in_img.size()));
  }

  out_img->resize(dest_size);

  // assume input is in [0.0, 1.0]
  for (size_t y = 0; y < height; y++) {
    for (size_t x = 0; x < width; x++) {
      for (size_t c = 0; c < channels; c++) {
        size_t idx = channel_stride * width * y + channel_stride * x + c;
        float f = in_img[idx] * scale_factor + bias;
        (*out_img)[idx] = SrgbTransform::srgbToLinear(f);
      }

      // remainder(usually alpha channel)
      // Apply linear conversion.
      for (size_t c = channels; c < channel_stride; c++) {
        size_t idx = channel_stride * width * y + channel_stride * x + c;
        float f = in_img[idx] * alpha_scale_factor + alpha_bias;
        (*out_img)[idx] = f;
      }
    }
  }

  return true;
}

bool srgb_8bit_to_linear_8bit(const std::vector<uint8_t> &in_img, size_t width,
                         size_t height,
                         size_t channels, size_t channel_stride,
                         std::vector<uint8_t> *out_img, std::string *err) {

  if (width == 0) {
    PUSH_ERROR_AND_RETURN("width is zero.");
  }

  if (height == 0) {
    PUSH_ERROR_AND_RETURN("height is zero.");
  }

  if (channels == 0) {
    PUSH_ERROR_AND_RETURN("channels is zero.");
  }

  if (out_img == nullptr) {
    PUSH_ERROR_AND_RETURN("`out_img` is nullptr.");
  }

  if (channel_stride == 0) {
    channel_stride = channels;
  } else {
    if (channel_stride < channels) {
      PUSH_ERROR_AND_RETURN(fmt::format("channel_stride {} is smaller than input channels {}", channel_stride, channels));
    }
  }

  size_t dest_size = size_t(width) * size_t(height) * channel_stride;
  if (dest_size > in_img.size()) {
    PUSH_ERROR_AND_RETURN(fmt::format("Insufficient input buffer size. must be the same or larger than {} but has {}", dest_size, in_img.size()));
  }

  out_img->resize(dest_size);

  // TODO: Precompute table.
  uint8_t linearlization_table[256];
  for (size_t u = 0; u < 256; u++) {
    float f = float(u) / 255.0f;
    linearlization_table[u] = detail::f32_to_u8(SrgbTransform::srgbToLinear(f));
  }

  for (size_t y = 0; y < height; y++) {
    for (size_t x = 0; x < width; x++) {
      for (size_t c = 0; c < channels; c++) {
        size_t idx = channel_stride * width * y + channel_stride * x + c;
        (*out_img)[idx] = linearlization_table[in_img[idx]];
      }

      // remainder(usually alpha channel)
      // no op.
      for (size_t c = channels; c < channel_stride; c++) {
        size_t idx = channel_stride * width * y + channel_stride * x + c;
        (*out_img)[idx] = in_img[idx];
      }
    }
  }

  return true;
}

bool u8_to_f32_image(const std::vector<uint8_t> &in_img, size_t width,
                         size_t height,
                         size_t channels,
                         std::vector<float> *out_img, std::string *err) {
  if (width == 0) {
    PUSH_ERROR_AND_RETURN("width is zero.");
  }

  if (height == 0) {
    PUSH_ERROR_AND_RETURN("height is zero.");
  }

  if (channels == 0) {
    PUSH_ERROR_AND_RETURN("channels is zero.");
  }

  if (out_img == nullptr) {
    PUSH_ERROR_AND_RETURN("`out_img` is nullptr.");
  }

  size_t num_pixels = size_t(width) * size_t(height) * channels;
  if (num_pixels > in_img.size()) {
    PUSH_ERROR_AND_RETURN(fmt::format("Insufficient input buffer size. must be the same or larger than {} but has {}", num_pixels, in_img.size()));
  }

  out_img->resize(num_pixels);

  for (size_t i = 0; i < num_pixels; i++) {
    (*out_img)[i] = float(in_img[i]) / 255.0f;
  }

  return true;
}

bool f32_to_u8_image(const std::vector<float> &in_img, size_t width,
                         size_t height,
                         size_t channels,
                         std::vector<uint8_t> *out_img, float scale, float bias, std::string *err) {
  if (width == 0) {
    PUSH_ERROR_AND_RETURN("width is zero.");
  }

  if (height == 0) {
    PUSH_ERROR_AND_RETURN("height is zero.");
  }

  if (channels == 0) {
    PUSH_ERROR_AND_RETURN("channels is zero.");
  }

  if (out_img == nullptr) {
    PUSH_ERROR_AND_RETURN("`out_img` is nullptr.");
  }

  size_t num_pixels = size_t(width) * size_t(height) * channels;
  if (num_pixels > in_img.size()) {
    PUSH_ERROR_AND_RETURN(fmt::format("Insufficient input buffer size. must be the same or larger than {} but has {}", num_pixels, in_img.size()));
  }

  out_img->resize(num_pixels);

  for (size_t i = 0; i < num_pixels; i++) {
    float f = scale * in_img[i] + bias;
    (*out_img)[i] = detail::f32_to_u8(f);
  }

  return true;
}

bool linear_displayp3_to_linear_sRGB(const std::vector<float> &in_img, size_t width,
                         size_t height, size_t channels,
                         std::vector<float> *out_img, std::string *err) {

  if (width == 0) {
    PUSH_ERROR_AND_RETURN("width is zero.");
  }

  if (height == 0) {
    PUSH_ERROR_AND_RETURN("height is zero.");
  }

  if (channels == 0) {
    PUSH_ERROR_AND_RETURN("channels is zero.");
  }

  if ((channels != 3) && (channels != 4)) {
    PUSH_ERROR_AND_RETURN(fmt::format("channels must be 3 or 4, but got {}", channels));
  }

  if (out_img == nullptr) {
    PUSH_ERROR_AND_RETURN("`out_img` is nullptr.");
  }

  if (in_img.size() != (width * height * channels)) {
    PUSH_ERROR_AND_RETURN(fmt::format("Input buffer size must be {}, but got {}", (width * height * channels), in_img.size()));
  }

  out_img->resize(in_img.size());

  // http://endavid.com/index.php?entry=79
  // https://tech.metail.com/introduction-colour-spaces-dci-p3/



  if (channels == 3) {
    for (size_t y = 0; y < height; y++) {
      for (size_t x = 0; x < width; x++) {
        float r, g, b;
        r = in_img[3 * (y * width + x) + 0];
        g = in_img[3 * (y * width + x) + 1];
        b = in_img[3 * (y * width + x) + 2];

        float out_rgb[3];
        out_rgb[0] = 1.2249f * r - 0.2247f * g;
        out_rgb[1] = -0.0420f * r + 1.0419f * g;
        out_rgb[2] = -0.0197f * r - 0.0786f * g + 1.0979f * b;

        // clamp
        out_rgb[0] = (out_rgb[0] < 0.0f) ? 0.0f : out_rgb[0];
        out_rgb[1] = (out_rgb[1] < 0.0f) ? 0.0f : out_rgb[1];
        out_rgb[2] = (out_rgb[2] < 0.0f) ? 0.0f : out_rgb[2];

        (*out_img)[3 * (y * width + x) + 0] = out_rgb[0];
        (*out_img)[3 * (y * width + x) + 1] = out_rgb[1];
        (*out_img)[3 * (y * width + x) + 2] = out_rgb[2];
      }
    }

  } else { // rgba
    for (size_t y = 0; y < height; y++) {
      for (size_t x = 0; x < width; x++) {
        float r, g, b, a;
        r = in_img[4 * (y * width + x) + 0];
        g = in_img[4 * (y * width + x) + 1];
        b = in_img[4 * (y * width + x) + 2];
        a = in_img[4 * (y * width + x) + 3];

        float out_rgb[3];
        out_rgb[0] = 1.2249f * r - 0.2247f * g;
        out_rgb[1] = -0.0420f * r + 1.0419f * g;
        out_rgb[2] = -0.0197f * r - 0.0786f * g + 1.0979f * b;

        // clamp
        out_rgb[0] = (out_rgb[0] < 0.0f) ? 0.0f : out_rgb[0];
        out_rgb[1] = (out_rgb[1] < 0.0f) ? 0.0f : out_rgb[1];
        out_rgb[2] = (out_rgb[2] < 0.0f) ? 0.0f : out_rgb[2];

        (*out_img)[4 * (y * width + x) + 0] = out_rgb[0];
        (*out_img)[4 * (y * width + x) + 1] = out_rgb[1];
        (*out_img)[4 * (y * width + x) + 2] = out_rgb[2];
        (*out_img)[4 * (y * width + x) + 3] = a;
      }
    }
  }

  return true;
}

bool linear_sRGB_to_linear_displayp3(const std::vector<float> &in_img, size_t width,
                         size_t height, size_t channels,
                         std::vector<float> *out_img, std::string *err) {

  if (width == 0) {
    PUSH_ERROR_AND_RETURN("width is zero.");
  }

  if (height == 0) {
    PUSH_ERROR_AND_RETURN("height is zero.");
  }

  if (channels == 0) {
    PUSH_ERROR_AND_RETURN("channels is zero.");
  }

  if ((channels != 3) && (channels != 4)) {
    PUSH_ERROR_AND_RETURN(fmt::format("channels must be 3 or 4, but got {}", channels));
  }

  if (out_img == nullptr) {
    PUSH_ERROR_AND_RETURN("`out_img` is nullptr.");
  }


  if (in_img.size() != (width * height * channels)) {
    PUSH_ERROR_AND_RETURN(fmt::format("Input buffer size must be {}, but got {}", (width * height * channels), in_img.size()));
  }

  out_img->resize(in_img.size());

  // http://endavid.com/index.php?entry=79
  // https://tech.metail.com/introduction-colour-spaces-dci-p3/

  if (channels == 3) {
    for (size_t y = 0; y < height; y++) {
      for (size_t x = 0; x < width; x++) {
        float r, g, b;
        r = in_img[3 * (y * width + x) + 0];
        g = in_img[3 * (y * width + x) + 1];
        b = in_img[3 * (y * width + x) + 2];

        float out_rgb[3];
        out_rgb[0] = 0.8225f * r + 0.1774f * g;
        out_rgb[1] = 0.0332f * r + 0.9669f * g;
        out_rgb[2] = 0.0171f * r + 0.0724f * g + 0.9108f * b;

        // clamp for just in case.
        out_rgb[0] = (out_rgb[0] < 0.0f) ? 0.0f : out_rgb[0];
        out_rgb[1] = (out_rgb[1] < 0.0f) ? 0.0f : out_rgb[1];
        out_rgb[2] = (out_rgb[2] < 0.0f) ? 0.0f : out_rgb[2];

        (*out_img)[3 * (y * width + x) + 0] = out_rgb[0];
        (*out_img)[3 * (y * width + x) + 1] = out_rgb[1];
        (*out_img)[3 * (y * width + x) + 2] = out_rgb[2];
      }
    }

  } else { // rgba
    for (size_t y = 0; y < height; y++) {
      for (size_t x = 0; x < width; x++) {
        float r, g, b, a;
        r = in_img[4 * (y * width + x) + 0];
        g = in_img[4 * (y * width + x) + 1];
        b = in_img[4 * (y * width + x) + 2];
        a = in_img[4 * (y * width + x) + 3];

        float out_rgb[3];
        out_rgb[0] = 0.8225f * r + 0.1774f * g;
        out_rgb[1] = 0.0332f * r + 0.9669f * g;
        out_rgb[2] = 0.0171f * r + 0.0724f * g + 0.9108f * b;

        // clamp for just in case.
        out_rgb[0] = (out_rgb[0] < 0.0f) ? 0.0f : out_rgb[0];
        out_rgb[1] = (out_rgb[1] < 0.0f) ? 0.0f : out_rgb[1];
        out_rgb[2] = (out_rgb[2] < 0.0f) ? 0.0f : out_rgb[2];

        (*out_img)[4 * (y * width + x) + 0] = out_rgb[0];
        (*out_img)[4 * (y * width + x) + 1] = out_rgb[1];
        (*out_img)[4 * (y * width + x) + 2] = out_rgb[2];
        (*out_img)[4 * (y * width + x) + 3] = a;
      }
    }
  }

  return true;
}

bool linear_sRGB_to_ACEScg(const std::vector<float> &in_img, size_t width,
                         size_t height, size_t channels,
                         std::vector<float> *out_img, std::string *err) {

  if (width == 0) {
    PUSH_ERROR_AND_RETURN("width is zero.");
  }

  if (height == 0) {
    PUSH_ERROR_AND_RETURN("height is zero.");
  }

  if (channels == 0) {
    PUSH_ERROR_AND_RETURN("channels is zero.");
  }

  if ((channels != 3) && (channels != 4)) {
    PUSH_ERROR_AND_RETURN(fmt::format("channels must be 3 or 4, but got {}", channels));
  }

  if (out_img == nullptr) {
    PUSH_ERROR_AND_RETURN("`out_img` is nullptr.");
  }


  if (in_img.size() != (width * height * channels)) {
    PUSH_ERROR_AND_RETURN(fmt::format("Input buffer size must be {}, but got {}", (width * height * channels), in_img.size()));
  }

  out_img->resize(in_img.size());

  // sRGB(D65) > XYZ -> D65toD50(Chromatic adaptation) -> ACEScg(AP1, D50)
  // 
  // https://www.shadertoy.com/view/WltSRB
  // https://computergraphics.stackexchange.com/questions/9834/how-to-convert-from-xyz-or-srgb-to-acescg-ap1
  // https://gist.github.com/Opioid/442d4975a23eed9a9e129bc3de97ea2a

  if (channels == 3) {
    for (size_t y = 0; y < height; y++) {
      for (size_t x = 0; x < width; x++) {
        float r, g, b;
        r = in_img[3 * (y * width + x) + 0];
        g = in_img[3 * (y * width + x) + 1];
        b = in_img[3 * (y * width + x) + 2];

        float out_rgb[3];
        out_rgb[0] = 0.6130973f * r + 0.33952285f * g +  0.04737928f * b;
        out_rgb[1] = 0.07019422f * r + 0.91635557f * g + 0.01345259f * b;
        out_rgb[2] = 0.0206156f * r + 0.10956983f * g + 0.86981512f  *b;

        // clamp negative value just in case.
        out_rgb[0] = (out_rgb[0] < 0.0f) ? 0.0f : out_rgb[0];
        out_rgb[1] = (out_rgb[1] < 0.0f) ? 0.0f : out_rgb[1];
        out_rgb[2] = (out_rgb[2] < 0.0f) ? 0.0f : out_rgb[2];

        (*out_img)[3 * (y * width + x) + 0] = out_rgb[0];
        (*out_img)[3 * (y * width + x) + 1] = out_rgb[1];
        (*out_img)[3 * (y * width + x) + 2] = out_rgb[2];
      }
    }

  } else { // rgba
    for (size_t y = 0; y < height; y++) {
      for (size_t x = 0; x < width; x++) {
        float r, g, b, a;
        r = in_img[4 * (y * width + x) + 0];
        g = in_img[4 * (y * width + x) + 1];
        b = in_img[4 * (y * width + x) + 2];
        a = in_img[4 * (y * width + x) + 3];

        float out_rgb[3];
        out_rgb[0] = 0.6130973f * r + 0.33952285f * g +  0.04737928f * b;
        out_rgb[1] = 0.07019422f * r + 0.91635557f * g + 0.01345259f * b;
        out_rgb[2] = 0.0206156f * r + 0.10956983f * g + 0.86981512f  *b;

        // clamp for just in case.
        out_rgb[0] = (out_rgb[0] < 0.0f) ? 0.0f : out_rgb[0];
        out_rgb[1] = (out_rgb[1] < 0.0f) ? 0.0f : out_rgb[1];
        out_rgb[2] = (out_rgb[2] < 0.0f) ? 0.0f : out_rgb[2];

        (*out_img)[4 * (y * width + x) + 0] = out_rgb[0];
        (*out_img)[4 * (y * width + x) + 1] = out_rgb[1];
        (*out_img)[4 * (y * width + x) + 2] = out_rgb[2];
        (*out_img)[4 * (y * width + x) + 3] = a;
      }
    }
  }

  return true;
}

bool ACEScg_to_linear_sRGB(const std::vector<float> &in_img, size_t width,
                         size_t height, size_t channels,
                         std::vector<float> *out_img, std::string *err) {

  if (width == 0) {
    PUSH_ERROR_AND_RETURN("width is zero.");
  }

  if (height == 0) {
    PUSH_ERROR_AND_RETURN("height is zero.");
  }

  if (channels == 0) {
    PUSH_ERROR_AND_RETURN("channels is zero.");
  }

  if ((channels != 3) && (channels != 4)) {
    PUSH_ERROR_AND_RETURN(fmt::format("channels must be 3 or 4, but got {}", channels));
  }

  if (out_img == nullptr) {
    PUSH_ERROR_AND_RETURN("`out_img` is nullptr.");
  }


  if (in_img.size() != (width * height * channels)) {
    PUSH_ERROR_AND_RETURN(fmt::format("Input buffer size must be {}, but got {}", (width * height * channels), in_img.size()));
  }

  out_img->resize(in_img.size());

  // inv(ACEScg_to_lin_sRGB)
  // 
  // https://www.shadertoy.com/view/WltSRB

  if (channels == 3) {
    for (size_t y = 0; y < height; y++) {
      for (size_t x = 0; x < width; x++) {
        float r, g, b;
        r = in_img[3 * (y * width + x) + 0];
        g = in_img[3 * (y * width + x) + 1];
        b = in_img[3 * (y * width + x) + 2];

        float out_rgb[3];
        out_rgb[0] =  1.705052f * r -0.621792f * g   -0.083258f * b;
        out_rgb[1] = -0.130257f * r + 1.140805f *  -0.010548f * b;
        out_rgb[2] = -0.024004f * r -0.128969f *g + 1.152972f * b;

        // clamp negative
        out_rgb[0] = (out_rgb[0] < 0.0f) ? 0.0f : out_rgb[0];
        out_rgb[1] = (out_rgb[1] < 0.0f) ? 0.0f : out_rgb[1];
        out_rgb[2] = (out_rgb[2] < 0.0f) ? 0.0f : out_rgb[2];

        (*out_img)[3 * (y * width + x) + 0] = out_rgb[0];
        (*out_img)[3 * (y * width + x) + 1] = out_rgb[1];
        (*out_img)[3 * (y * width + x) + 2] = out_rgb[2];
      }
    }

  } else { // rgba
    for (size_t y = 0; y < height; y++) {
      for (size_t x = 0; x < width; x++) {
        float r, g, b, a;
        r = in_img[4 * (y * width + x) + 0];
        g = in_img[4 * (y * width + x) + 1];
        b = in_img[4 * (y * width + x) + 2];
        a = in_img[4 * (y * width + x) + 3];

        float out_rgb[3];
        out_rgb[0] =  1.705052f * r -0.621792f * g   -0.083258f * b;
        out_rgb[1] = -0.130257f * r + 1.140805f *  -0.010548f * b;
        out_rgb[2] = -0.024004f * r -0.128969f *g + 1.152972f * b;

        // clamp negative value
        out_rgb[0] = (out_rgb[0] < 0.0f) ? 0.0f : out_rgb[0];
        out_rgb[1] = (out_rgb[1] < 0.0f) ? 0.0f : out_rgb[1];
        out_rgb[2] = (out_rgb[2] < 0.0f) ? 0.0f : out_rgb[2];

        (*out_img)[4 * (y * width + x) + 0] = out_rgb[0];
        (*out_img)[4 * (y * width + x) + 1] = out_rgb[1];
        (*out_img)[4 * (y * width + x) + 2] = out_rgb[2];
        (*out_img)[4 * (y * width + x) + 3] = a;
      }
    }
  }

  return true;
}

bool displayp3_f16_to_linear_f32(const std::vector<value::half> &in_img, size_t width,
                         size_t height,
                         size_t channels, size_t channel_stride,
                         std::vector<float> *out_img, const float scale_factor, const float bias, const float alpha_scale_factor, const float alpha_bias, std::string *err) {

  if (width == 0) {
    PUSH_ERROR_AND_RETURN("width is zero.");
  }

  if (height == 0) {
    PUSH_ERROR_AND_RETURN("height is zero.");
  }

  if (channels == 0) {
    PUSH_ERROR_AND_RETURN("channels is zero.");
  }

  if (out_img == nullptr) {
    PUSH_ERROR_AND_RETURN("`out_img` is nullptr.");
  }

  if (channel_stride == 0) {
    channel_stride = channels;
  } else {
    if (channel_stride < channels) {
      PUSH_ERROR_AND_RETURN(fmt::format("channel_stride {} is smaller than input channels {}", channel_stride, channels));
    }
  }

  size_t dest_size = size_t(width) * size_t(height) * channel_stride;
  if (dest_size > in_img.size()) {
    PUSH_ERROR_AND_RETURN(fmt::format("Insufficient input buffer size. must be the same or larger than {} but has {}", dest_size, in_img.size()));
  }

  out_img->resize(dest_size);

  // assume input is in [0.0, 1.0]
  for (size_t y = 0; y < height; y++) {
    for (size_t x = 0; x < width; x++) {
      for (size_t c = 0; c < channels; c++) {
        size_t idx = channel_stride * width * y + channel_stride * x + c;
        float in_val = value::half_to_float(in_img[idx]);
        float f = in_val * scale_factor + bias;
        (*out_img)[idx] = SrgbTransform::srgbToLinear(f); // Display P3 use the same transfer function with sRGB
      }

      // remainder(usually alpha channel)
      // Apply linear conversion.
      for (size_t c = channels; c < channel_stride; c++) {
        size_t idx = channel_stride * width * y + channel_stride * x + c;
        float in_val = value::half_to_float(in_img[idx]);
        float f = in_val * alpha_scale_factor + alpha_bias;
        (*out_img)[idx] = f;
      }
    }
  }

  return true;
}

} // namespace tinyusdz
