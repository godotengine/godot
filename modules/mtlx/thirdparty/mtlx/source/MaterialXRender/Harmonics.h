//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_HARMONICS_H
#define MATERIALX_HARMONICS_H

/// @file
/// Spherical harmonics functionality

#include <MaterialXRender/Export.h>
#include <MaterialXRender/Image.h>
#include <MaterialXRender/Types.h>

MATERIALX_NAMESPACE_BEGIN

/// Class template for a vector of spherical harmonic coefficients.
///
/// Template parameter C is the coefficient type (e.g. double, Color3).
/// Template parameter B is the number of spherical harmonic bands.
template <class C, size_t B> class ShCoeffs
{
  public:
    static const size_t NUM_BANDS = B;
    static const size_t NUM_COEFFS = B * B;

  public:
    ShCoeffs() = default;
    explicit ShCoeffs(const std::array<C, NUM_COEFFS>& arr) :
        _arr(arr) { }

    /// @name Comparison Operators
    /// @{

    /// Return true if the given vector is identical to this one.
    bool operator==(const ShCoeffs& rhs) const { return _arr == rhs._arr; }

    /// Return true if the given vector differs from this one.
    bool operator!=(const ShCoeffs& rhs) const { return _arr != rhs._arr; }

    /// @}
    /// @name Indexing Operators
    /// @{

    /// Return the coefficient at the given index.
    C& operator[](size_t i) { return _arr.at(i); }

    /// Return the const coefficient at the given index.
    const C& operator[](size_t i) const { return _arr.at(i); }

    /// @}

  protected:
    std::array<C, NUM_COEFFS> _arr;
};

/// Double-precision scalar coefficients for third-order spherical harmonics.
using Sh3ScalarCoeffs = ShCoeffs<double, 3>;

/// Double-precision color coefficients for third-order spherical harmonics.
using Sh3ColorCoeffs = ShCoeffs<Color3d, 3>;

/// Project an environment map to third-order SH, with an optional convolution
/// to convert radiance to irradiance.
/// @param env An environment map in lat-long format.
/// @param irradiance If true, then the returned signal will be convolved
///    by a clamped cosine kernel to generate irradiance.
/// @return The projection of the environment to third-order SH.
MX_RENDER_API Sh3ColorCoeffs projectEnvironment(ConstImagePtr env, bool irradiance = false);

/// Normalize an environment to the given radiance.
/// @param env An environment map in lat-long format.
/// @param envRadiance The radiance to which the environment map should be normalized.
/// @param maxTexelRadiance The maximum radiance allowed for any individual texel of the map.
/// @return A new normalized environment map, in the same format as the original.
MX_RENDER_API ImagePtr normalizeEnvironment(ConstImagePtr env, float envRadiance, float maxTexelRadiance);

/// Compute the dominant light direction and color of an environment map.
/// @param env An environment map in lat-long format.
/// @param lightDir Returns the dominant light direction of the environment.
/// @param lightColor Returns the color of the light from the dominant direction.
MX_RENDER_API void computeDominantLight(ConstImagePtr env, Vector3& lightDir, Color3& lightColor);

/// Render the given spherical harmonic signal to an environment map.
/// @param shEnv The color signal of the environment encoded as third-order SH.
/// @param width The width of the output environment map.
/// @param height The height of the output environment map.
/// @return An environment map in the lat-long format.
MX_RENDER_API ImagePtr renderEnvironment(const Sh3ColorCoeffs& shEnv, unsigned int width, unsigned int height);

/// Render a reference irradiance map from the given environment map,
/// using brute-force computations for a slow but accurate result.
/// @param env An environment map in lat-long format.
/// @param width The width of the output irradiance map.
/// @param height The height of the output irradiance map.
/// @return An irradiance map in the lat-long format.
MX_RENDER_API ImagePtr renderReferenceIrradiance(ConstImagePtr env, unsigned int width, unsigned int height);

MATERIALX_NAMESPACE_END

#endif
