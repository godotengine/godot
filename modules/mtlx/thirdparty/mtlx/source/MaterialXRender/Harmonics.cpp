//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXRender/Harmonics.h>

#include <iostream>

MATERIALX_NAMESPACE_BEGIN

namespace
{

const double PI = std::acos(-1.0);

const double BASIS_CONSTANT_0 = std::sqrt( 1.0 / ( 4.0 * PI));
const double BASIS_CONSTANT_1 = std::sqrt( 3.0 / ( 4.0 * PI));
const double BASIS_CONSTANT_2 = std::sqrt(15.0 / ( 4.0 * PI));
const double BASIS_CONSTANT_3 = std::sqrt( 5.0 / (16.0 * PI));
const double BASIS_CONSTANT_4 = std::sqrt(15.0 / (16.0 * PI));

const double COSINE_CONSTANT_0 = 1.0;
const double COSINE_CONSTANT_1 = 2.0 / 3.0;
const double COSINE_CONSTANT_2 = 1.0 / 4.0;

const Color3d LUMA_COEFFS_REC709(0.2126, 0.7152, 0.0722);

double imageXToPhi(unsigned int x, unsigned int width)
{
    // Align spherical coordinates with texel centers by adding 0.5.
    return 2.0 * PI * (x + 0.5) / width;
}

double imageYToTheta(unsigned int y, unsigned int height)
{
    return PI * (y + 0.5) / height;
}

Vector3d sphericalToCartesian(double theta, double phi)
{
    double r = std::sin(theta);
    return Vector3d(-r * std::sin(phi), -std::cos(theta), r * std::cos(phi));
}

double texelSolidAngle(unsigned int y, unsigned int width, unsigned int height)
{
    // Return the solid angle of a texel within a lat-long environment map.
    //
    // Reference:
    //   https://en.wikipedia.org/wiki/Solid_angle#Latitude-longitude_rectangle

    double dTheta = std::cos(y * PI / height) - std::cos((y + 1) * PI / height);
    double dPhi = 2.0 * PI / width;
    return dTheta * dPhi;
}

Sh3ScalarCoeffs evalDirection(const Vector3d& dir)
{
    // Evaluate the spherical harmonic basis functions for the given direction,
    // returning the first three bands of coefficients.
    //
    // References:
    //   https://cseweb.ucsd.edu/~ravir/papers/envmap/envmap.pdf
    //   http://orlandoaguilar.github.io/sh/spherical/harmonics/irradiance/map/2017/02/12/SphericalHarmonics.html

    const double& x = dir[0];
    const double& y = dir[1];
    const double& z = dir[2];

    return Sh3ScalarCoeffs(
    {
        BASIS_CONSTANT_0,
        BASIS_CONSTANT_1 * y,
        BASIS_CONSTANT_1 * z,
        BASIS_CONSTANT_1 * x,
        BASIS_CONSTANT_2 * x * y,
        BASIS_CONSTANT_2 * y * z,
        BASIS_CONSTANT_3 * (3.0 * z * z - 1.0),
        BASIS_CONSTANT_2 * x * z,
        BASIS_CONSTANT_4 * (x * x - y * y)
    });
}

} // anonymous namespace

Sh3ColorCoeffs projectEnvironment(ConstImagePtr env, bool irradiance)
{
    Sh3ColorCoeffs shEnv;

    for (unsigned int y = 0; y < env->getHeight(); y++)
    {
        double theta = imageYToTheta(y, env->getHeight());
        double texelWeight = texelSolidAngle(y, env->getWidth(), env->getHeight());

        for (unsigned int x = 0; x < env->getWidth(); x++)
        {
            // Sample the color at these coordinates.
            Color4 color = env->getTexelColor(x, y);

            // Compute the direction vector.
            double phi = imageXToPhi(x, env->getWidth());
            Vector3d dir = sphericalToCartesian(theta, phi);

            // Evaluate the given direction as SH coefficients.
            Sh3ScalarCoeffs shDir = evalDirection(dir);

            // Combine color with texel weight.
            Color3d weightedColor(color[0] * texelWeight,
                                  color[1] * texelWeight,
                                  color[2] * texelWeight);

            // Update coefficients for the influence of this texel.
            for (size_t i = 0; i < shEnv.NUM_COEFFS; i++)
            {
                shEnv[i] += weightedColor * shDir[i];
            }
        }
    }

    // If irradiance is requested, then apply constant factors to convolve the
    // signal by a clamped cosine kernel.
    if (irradiance)
    {
        shEnv[0] *= COSINE_CONSTANT_0;
        shEnv[1] *= COSINE_CONSTANT_1;
        shEnv[2] *= COSINE_CONSTANT_1;
        shEnv[3] *= COSINE_CONSTANT_1;
        shEnv[4] *= COSINE_CONSTANT_2;
        shEnv[5] *= COSINE_CONSTANT_2;
        shEnv[6] *= COSINE_CONSTANT_2;
        shEnv[7] *= COSINE_CONSTANT_2;
        shEnv[8] *= COSINE_CONSTANT_2;
    }

    return shEnv;
}

ImagePtr normalizeEnvironment(ConstImagePtr env, float envRadiance, float maxTexelRadiance)
{
    // Compute the radiance of the original environment map.
    double origEnvRadiance = 0.0;
    for (unsigned int y = 0; y < env->getHeight(); y++)
    {
        double texelWeight = texelSolidAngle(y, env->getWidth(), env->getHeight());

        for (unsigned int x = 0; x < env->getWidth(); x++)
        {
            // Sample the color at these coordinates.
            Color4 color = env->getTexelColor(x, y);

            // Apply maximum texel radiance.
            double texelRadiance = Color3d(color[0], color[1], color[2]).dot(LUMA_COEFFS_REC709);
            if ((float) texelRadiance > maxTexelRadiance)
            {
                color *= maxTexelRadiance / (float) texelRadiance;
            }

            // Combine color with texel weight.
            Color3d weightedColor(color[0] * texelWeight,
                                  color[1] * texelWeight,
                                  color[2] * texelWeight);

            // Add to environment radiance.
            double texelContribution = weightedColor.dot(LUMA_COEFFS_REC709);
            origEnvRadiance += texelContribution;
        }
    }

    // Generate the normalized map.
    ImagePtr normEnv = Image::create(env->getWidth(), env->getHeight(), env->getChannelCount(), env->getBaseType());
    normEnv->createResourceBuffer();
    float envNormFactor = origEnvRadiance ? (float) (envRadiance / origEnvRadiance) : 1.0f;
    for (unsigned int y = 0; y < env->getHeight(); y++)
    {
        for (unsigned int x = 0; x < env->getWidth(); x++)
        {
            // Sample the color at these coordinates.
            Color4 color = env->getTexelColor(x, y);

            // Apply maximum texel radiance.
            double texelRadiance = Color3d(color[0], color[1], color[2]).dot(LUMA_COEFFS_REC709);
            if ((float) texelRadiance > maxTexelRadiance)
            {
                color *= maxTexelRadiance / (float) texelRadiance;
            }

            // Store the normalized color.
            normEnv->setTexelColor(x, y, color * envNormFactor);
        }
    }

    return normEnv;
}

void computeDominantLight(ConstImagePtr env, Vector3& lightDir, Color3& lightColor)
{
    // Reference:
    //   https://seblagarde.wordpress.com/2011/10/09/dive-in-sh-buffer-idea/

    // Project the environment to spherical harmonics.
    Sh3ColorCoeffs shEnv = projectEnvironment(env);

    // Handle empty environments.
    if (shEnv == Sh3ColorCoeffs())
    {
        lightDir = Vector3(0.0f, -1.0f, 0.0f);
        lightColor = Color3(0.0f);
        return;
    }

    // Compute the dominant light direction.
    Vector3d dir = Vector3d(shEnv[3].dot(LUMA_COEFFS_REC709),
                            shEnv[1].dot(LUMA_COEFFS_REC709),
                            shEnv[2].dot(LUMA_COEFFS_REC709)).getNormalized();

    // Evaluate the dominant direction as spherical harmonics.
    Sh3ScalarCoeffs shDir = evalDirection(dir);
    Vector4d vDir(shDir[0], shDir[1], shDir[2], shDir[3]);

    // Compute the dominant light color.
    Vector4d vEnvR(shEnv[0][0], shEnv[1][0], shEnv[2][0], shEnv[3][0]);
    Vector4d vEnvG(shEnv[0][1], shEnv[1][1], shEnv[2][1], shEnv[3][1]);
    Vector4d vEnvB(shEnv[0][2], shEnv[1][2], shEnv[2][2], shEnv[3][2]);
    Color3d color = Color3d(
        std::max(vDir.dot(vEnvR), 0.0),
        std::max(vDir.dot(vEnvG), 0.0),
        std::max(vDir.dot(vEnvB), 0.0)) / vDir.dot(vDir);

    // Convert to single-precision floats.
    lightDir = Vector3((float) dir[0], (float) dir[1], (float) dir[2]);
    lightColor = Color3((float) color[0], (float) color[1], (float) color[2]);
}

ImagePtr renderEnvironment(const Sh3ColorCoeffs& shEnv, unsigned int width, unsigned int height)
{
    ImagePtr env = Image::create(width, height, 3, Image::BaseType::FLOAT);
    env->createResourceBuffer();

    for (unsigned int y = 0; y < env->getHeight(); y++)
    {
        double theta = imageYToTheta(y, env->getHeight());
        for (unsigned int x = 0; x < env->getWidth(); x++)
        {
            // Compute the direction vector.
            double phi = imageXToPhi(x, env->getWidth());
            Vector3d dir = sphericalToCartesian(theta, phi);

            // Evaluate the given direction as SH coefficients.
            Sh3ScalarCoeffs shDir = evalDirection(dir);

            // Compute the signal color in this direction.
            Color3d signalColor;
            for (size_t i = 0; i < shEnv.NUM_COEFFS; i++)
            {
                signalColor += shEnv[i] * shDir[i];
            }

            // Clamp the color and store as an environment texel.
            Color4 outputColor(
                (float) std::max(signalColor[0], 0.0),
                (float) std::max(signalColor[1], 0.0),
                (float) std::max(signalColor[2], 0.0),
                1.0f);
            env->setTexelColor(x, y, outputColor);
        }
    }

    return env;
}

ImagePtr renderReferenceIrradiance(ConstImagePtr env, unsigned int width, unsigned int height)
{
    std::cout << "Rendering reference irradiance map..." << std::endl;
    ImagePtr outImage = Image::create(width, height, 3, Image::BaseType::FLOAT);
    outImage->createResourceBuffer();

    // Iterate through output texels.
    for (unsigned int outY = 0; outY < outImage->getHeight(); outY++)
    {
        std::cout << "Rendering irradiance map row " << outY << " of " << outImage->getHeight() << "..." << std::endl;
        double outTheta = imageYToTheta(outY, outImage->getHeight());
        for (unsigned int outX = 0; outX < outImage->getWidth(); outX++)
        {
            // Compute the output direction vector.
            double outPhi = imageXToPhi(outX, outImage->getWidth());
            Vector3d outDir = sphericalToCartesian(outTheta, outPhi);

            // Initialize output texel color.
            Color3d outColor;

            // Iterate through input texels.
            for (unsigned int inY = 0; inY < env->getHeight(); inY++)
            {
                double inTheta = imageYToTheta(inY, env->getHeight());
                if (std::abs(inTheta - outTheta) >= PI / 2.0)
                {
                    continue;
                }

                double inTexelWeight = texelSolidAngle(inY, env->getWidth(), env->getHeight());
                for (unsigned int inX = 0; inX < env->getWidth(); inX++)
                {
                    // Compute the input direction vector.
                    double inPhi = imageXToPhi(inX, env->getWidth());
                    Vector3d inDir = sphericalToCartesian(inTheta, inPhi);

                    // Compute the cosine weight.
                    double cosineWeight = inDir.dot(outDir);
                    if (cosineWeight <= 0.0)
                    {
                        continue;
                    }

                    // Sample the input environment at these coordinates.
                    Color4 envColor = env->getTexelColor(inX, inY);

                    // Apply the influence of this input texel.
                    outColor += Color3d(envColor[0], envColor[1], envColor[2]) * inTexelWeight * cosineWeight;
                }
            }

            // Normalize and store the output texel.
            outImage->setTexelColor(outX, outY, Color4((float) (outColor[0] / PI),
                                                       (float) (outColor[1] / PI),
                                                       (float) (outColor[2] / PI),
                                                       1.0f));
        }
    }

    return outImage;
}

MATERIALX_NAMESPACE_END
