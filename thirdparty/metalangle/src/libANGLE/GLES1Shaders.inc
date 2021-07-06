//
// Copyright 2018 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// GLES1Shaders.inc: Defines GLES1 emulation shader.

constexpr char kGLES1DrawVShader[] = R"(#version 300 es
precision highp float;

#define kMaxTexUnits 4

in vec4 pos;
in vec3 normal;
in vec4 color;
in float pointsize;
in vec4 texcoord0;
in vec4 texcoord1;
in vec4 texcoord2;
in vec4 texcoord3;

uniform mat4 projection;
uniform mat4 modelview;
uniform mat4 modelview_invtr;
uniform mat4 texture_matrix[kMaxTexUnits];

uniform bool enable_rescale_normal;
uniform bool enable_normalize;

// Point rasterization//////////////////////////////////////////////////////////

uniform bool point_rasterization;
uniform float point_size_min;
uniform float point_size_max;
uniform vec3 point_distance_attenuation;

// GL_OES_draw_texture uniforms/////////////////////////////////////////////////

uniform bool enable_draw_texture;
uniform vec4 draw_texture_coords;
uniform vec2 draw_texture_dims;
uniform vec4 draw_texture_normalized_crop_rect[kMaxTexUnits];

// Varyings/////////////////////////////////////////////////////////////////////

out vec4 pos_varying;
out vec3 normal_varying;
out vec4 color_varying;
flat out vec4 color_varying_flat;
out vec4 texcoord0_varying;
out vec4 texcoord1_varying;
out vec4 texcoord2_varying;
out vec4 texcoord3_varying;

const vec4 drawTextureVertices[6] = vec4[](
    vec4(0.0, 0.0, 0.0, 1.0),
    vec4(1.0, 0.0, 0.0, 1.0),
    vec4(1.0, 1.0, 0.0, 1.0),
    vec4(0.0, 0.0, 0.0, 1.0),
    vec4(1.0, 1.0, 0.0, 1.0),
    vec4(0.0, 1.0, 0.0, 1.0));

vec4 drawTexturePosition(int vertexId)
{

    float drawTexX = draw_texture_coords[0];
    float drawTexY = draw_texture_coords[1];
    float drawTexZ = draw_texture_coords[2];
    float drawTexW = draw_texture_dims[0];
    float drawTexH = draw_texture_dims[1];

    return vec4(drawTexX, drawTexY, drawTexZ, 0.0) +
           drawTextureVertices[vertexId] *
           vec4(drawTexW, drawTexH, 1.0, 1.0);
}

vec4 drawTextureTexCoord(int vertexId, int textureUnit)
{
    float texCropU = draw_texture_normalized_crop_rect[textureUnit].x;
    float texCropV = draw_texture_normalized_crop_rect[textureUnit].y;
    float texCropW = draw_texture_normalized_crop_rect[textureUnit].z;
    float texCropH = draw_texture_normalized_crop_rect[textureUnit].w;

    return vec4(texCropU, texCropV, 0.0, 0.0) +
           drawTextureVertices[vertexId] *
           vec4(texCropW, texCropH, 0.0, 0.0);
}

vec4 calcWorldPosition(vec4 posInput)
{
    return modelview * posInput;
}

vec4 calcNdcFromWorldPosition(vec4 worldPos)
{
    return projection * worldPos;
}

float calcPointSize(vec4 ndcPos)
{
    float dist         = length(ndcPos.z);
    float attConst     = point_distance_attenuation[0];
    float attLinear    = point_distance_attenuation[1];
    float attQuad      = point_distance_attenuation[2];
    float attPart      = attConst + attLinear * dist + attQuad * dist * dist;
    float attPointSize = pointsize / pow(attPart, 0.5);

    return clamp(attPointSize, point_size_min, point_size_max);
}

vec3 calcNormal(vec3 normalInput)
{
    mat3 mvInvTr3 = mat3(modelview_invtr);
    vec3 result   = mvInvTr3 * normalInput;

    if (enable_rescale_normal)
    {
        float rescale   = 1.0;
        vec3 rescaleVec = vec3(mvInvTr3[2]);
        float len       = length(rescaleVec);
        if (len > 0.0)
        {
            rescale = 1.0 / len;
        }
        result *= rescale;
    }

    if (enable_normalize)
    {
        result = normalize(result);
    }

    return result;
}

void main()
{
    color_varying      = color;
    color_varying_flat = color;

    if (enable_draw_texture)
    {
        int vertexId        = gl_VertexID;
        vec4 posDrawTexture = drawTexturePosition(vertexId);

        gl_Position = posDrawTexture;
        pos_varying = posDrawTexture;

        normal_varying = normal;

        gl_PointSize = pointsize;

        texcoord0_varying = drawTextureTexCoord(vertexId, 0);
        texcoord1_varying = drawTextureTexCoord(vertexId, 1);
        texcoord2_varying = drawTextureTexCoord(vertexId, 2);
        texcoord3_varying = drawTextureTexCoord(vertexId, 3);
    }
    else
    {
        vec4 worldPos = calcWorldPosition(pos);
        vec4 ndcPos   = calcNdcFromWorldPosition(worldPos);

        gl_Position = ndcPos;
        pos_varying = worldPos;

        normal_varying = calcNormal(normal);

        // Avoid calculating point size stuff
        // if we are not rendering points.
        if (point_rasterization)
        {
            gl_PointSize = calcPointSize(ndcPos);
        }
        else
        {
            gl_PointSize = pointsize;
        }

        texcoord0_varying = texture_matrix[0] * texcoord0;
        texcoord1_varying = texture_matrix[1] * texcoord1;
        texcoord2_varying = texture_matrix[2] * texcoord2;
        texcoord3_varying = texture_matrix[3] * texcoord3;
    }
}
)";

// version, flat,
constexpr char kGLES1DrawFShaderHeader[] = R"(#version 300 es
precision highp float;

// Defines for GL constants
#define kMaxLights                           8
#define kMaxTexUnits                         4
#define kMaxClipPlanes                       6

#define kModulate                       0x2100
#define kDecal                          0x2101
#define kCombine                        0x8570
#define kReplace                        0x1E01
#define kBlend                          0x0BE2
#define kAdd                            0x0104

#define kAddSigned                      0x8574
#define kInterpolate                    0x8575
#define kSubtract                       0x84E7
#define kDot3Rgb                        0x86AE
#define kDot3Rgba                       0x86AF

#define kAlpha                          0x1906
#define kRGB                            0x1907
#define kRGBA                           0x1908
#define kLuminance                      0x1909
#define kLuminanceAlpha                 0x190A

#define kTexture                        0x1702
#define kConstant                       0x8576
#define kPrimaryColor                   0x8577
#define kPrevious                       0x8578

#define kSrcColor                       0x0300
#define kOneMinusSrcColor               0x0301
#define kSrcAlpha                       0x0302
#define kOneMinusSrcAlpha               0x0303

#define kLinear                         0x2601
#define kExp                            0x0800
#define kExp2                           0x0801

#define kNever                          0x0200
#define kLess                           0x0201
#define kEqual                          0x0202
#define kLequal                         0x0203
#define kGreater                        0x0204
#define kNotequal                       0x0205
#define kGequal                         0x0206
#define kAlways                         0x0207
#define kZero                              0x0
#define kOne                               0x1

#define kClear                          0x1500
#define kAnd                            0x1501
#define kAnd_reverse                    0x1502
#define kCopy                           0x1503
#define kAnd_inverted                   0x1504
#define kNoop                           0x1505
#define kXor                            0x1506
#define kOr                             0x1507
#define kNor                            0x1508
#define kEquiv                          0x1509
#define kInvert                         0x150A
#define kOr_reverse                     0x150B
#define kCopy_inverted                  0x150C
#define kOr_inverted                    0x150D
#define kNand                           0x150E
#define kSet                            0x150F)";

constexpr char kGLES1DrawFShaderUniformDefs[] = R"(

// Texture units ///////////////////////////////////////////////////////////////

uniform bool enable_texture_2d[kMaxTexUnits];
uniform bool enable_texture_cube_map[kMaxTexUnits];

// These are not arrays because hw support for arrays
// of samplers is rather lacking.

uniform sampler2D tex_sampler0;
uniform samplerCube tex_cube_sampler0;

uniform sampler2D tex_sampler1;
uniform samplerCube tex_cube_sampler1;

uniform sampler2D tex_sampler2;
uniform samplerCube tex_cube_sampler2;

uniform sampler2D tex_sampler3;
uniform samplerCube tex_cube_sampler3;

uniform int texture_format[kMaxTexUnits];

uniform int texture_env_mode[kMaxTexUnits];
uniform int combine_rgb[kMaxTexUnits];
uniform int combine_alpha[kMaxTexUnits];
uniform int src0_rgb[kMaxTexUnits];
uniform int src0_alpha[kMaxTexUnits];
uniform int src1_rgb[kMaxTexUnits];
uniform int src1_alpha[kMaxTexUnits];
uniform int src2_rgb[kMaxTexUnits];
uniform int src2_alpha[kMaxTexUnits];
uniform int op0_rgb[kMaxTexUnits];
uniform int op0_alpha[kMaxTexUnits];
uniform int op1_rgb[kMaxTexUnits];
uniform int op1_alpha[kMaxTexUnits];
uniform int op2_rgb[kMaxTexUnits];
uniform int op2_alpha[kMaxTexUnits];
uniform vec4 texture_env_color[kMaxTexUnits];
uniform float texture_env_rgb_scale[kMaxTexUnits];
uniform float texture_env_alpha_scale[kMaxTexUnits];
uniform bool point_sprite_coord_replace[kMaxTexUnits];

// Vertex attributes////////////////////////////////////////////////////////////

in vec4 pos_varying;
in vec3 normal_varying;
in vec4 color_varying;
flat in vec4 color_varying_flat;
in vec4 texcoord0_varying;
in vec4 texcoord1_varying;
in vec4 texcoord2_varying;
in vec4 texcoord3_varying;

// Alpha test///////////////////////////////////////////////////////////////////

uniform bool enable_alpha_test;
uniform int alpha_func;
uniform float alpha_test_ref;

// Shading: flat shading, lighting, and materials///////////////////////////////

uniform bool shade_model_flat;
uniform bool enable_lighting;
uniform bool enable_color_material;

uniform vec4 material_ambient;
uniform vec4 material_diffuse;
uniform vec4 material_specular;
uniform vec4 material_emissive;
uniform float material_specular_exponent;

uniform vec4 light_model_scene_ambient;
uniform bool light_model_two_sided;

uniform bool light_enables[kMaxLights];
uniform vec4 light_ambients[kMaxLights];
uniform vec4 light_diffuses[kMaxLights];
uniform vec4 light_speculars[kMaxLights];
uniform vec4 light_positions[kMaxLights];
uniform vec3 light_directions[kMaxLights];
uniform float light_spotlight_exponents[kMaxLights];
uniform float light_spotlight_cutoff_angles[kMaxLights];
uniform float light_attenuation_consts[kMaxLights];
uniform float light_attenuation_linears[kMaxLights];
uniform float light_attenuation_quadratics[kMaxLights];

// Fog /////////////////////////////////////////////////////////////////////////

uniform bool enable_fog;
uniform int fog_mode;
uniform float fog_density;
uniform float fog_start;
uniform float fog_end;
uniform vec4 fog_color;

// User clip plane /////////////////////////////////////////////////////////////

uniform bool enable_clip_planes;
uniform bool clip_plane_enables[kMaxClipPlanes];
uniform vec4 clip_planes[kMaxClipPlanes];

// Point rasterization//////////////////////////////////////////////////////////

uniform bool point_rasterization;
uniform bool point_sprite_enabled;

// GL_OES_draw_texture//////////////////////////////////////////////////////////

uniform bool enable_draw_texture;

// Outgoing fragment////////////////////////////////////////////////////////////

out vec4 frag_color;
)";

constexpr char kGLES1DrawFShaderFunctions[] = R"(

float posDot(vec3 a, vec3 b)
{
    return max(dot(a, b), 0.0);
}

vec4 doLighting(vec4 currentFragment)
{
    vec4 materialAmbientActual = material_ambient;
    vec4 materialDiffuseActual = material_diffuse;

    if (enable_color_material || enable_texture_2d[0] || enable_texture_cube_map[0])
    {
        materialAmbientActual = currentFragment;
        materialDiffuseActual = currentFragment;
    }

    vec4 lightingResult = material_emissive + materialAmbientActual * light_model_scene_ambient;

    for (int i = 0; i < kMaxLights; i++)
    {

        if (!light_enables[i])
            continue;

        vec4 lightAmbient  = light_ambients[i];
        vec4 lightDiffuse  = light_diffuses[i];
        vec4 lightSpecular = light_speculars[i];
        vec4 lightPos      = light_positions[i];
        vec3 lightDir      = light_directions[i];
        float attConst     = light_attenuation_consts[i];
        float attLinear    = light_attenuation_linears[i];
        float attQuadratic = light_attenuation_quadratics[i];
        float spotAngle    = light_spotlight_cutoff_angles[i];
        float spotExponent = light_spotlight_exponents[i];

        vec3 toLight;
        if (lightPos.w == 0.0)
        {
            toLight = lightPos.xyz;
        }
        else
        {
            toLight = (lightPos.xyz / lightPos.w - pos_varying.xyz);
        }

        float lightDist = length(toLight);
        vec3 h          = normalize(toLight) + vec3(0.0, 0.0, 1.0);
        float ndotL     = posDot(normal_varying, normalize(toLight));
        float ndoth     = posDot(normal_varying, normalize(h));

        float specAtt;

        if (ndotL != 0.0)
        {
            specAtt = 1.0;
        }
        else
        {
            specAtt = 0.0;
        }

        float att;

        if (lightPos.w != 0.0)
        {
            float attDenom =
                (attConst + attLinear * lightDist + attQuadratic * lightDist * lightDist);
            att = 1.0 / attDenom;
        }
        else
        {
            att = 1.0;
        }

        float spot;

        float spotAngleCos = cos(radians(spotAngle));
        vec3 toSurfaceDir  = -normalize(toLight);
        float spotDot      = posDot(toSurfaceDir, normalize(lightDir));

        if (spotAngle == 180.0 || lightPos.w == 0.0)
        {
            spot = 1.0;
        }
        else
        {
            if (spotDot < spotAngleCos)
            {
                spot = 0.0;
            }
            else
            {
                spot = pow(spotDot, spotExponent);
            }
        }

        vec4 contrib = materialAmbientActual * lightAmbient;
        contrib += ndotL * materialDiffuseActual * lightDiffuse;
        if (ndoth > 0.0 && material_specular_exponent > 0.0)
        {
            contrib += specAtt * pow(ndoth, material_specular_exponent) * material_specular *
                       lightSpecular;
        }
        else
        {
            if (ndoth > 0.0)
            {
                contrib += specAtt * material_specular * lightSpecular;
            }
        }
        contrib *= att * spot;
        lightingResult += contrib;
    }

    return lightingResult;
}

bool doAlphaTest(vec4 currentFragment)
{
    bool shouldPassAlpha = false;
    float incAlpha       = currentFragment.a;

    switch (alpha_func)
    {
        case kNever:
            shouldPassAlpha = false;
            break;
        case kLess:
            shouldPassAlpha = incAlpha < alpha_test_ref;
            break;
        case kLequal:
            shouldPassAlpha = incAlpha <= alpha_test_ref;
            break;
        case kEqual:
            shouldPassAlpha = incAlpha == alpha_test_ref;
            break;
        case kGequal:
            shouldPassAlpha = incAlpha >= alpha_test_ref;
            break;
        case kGreater:
            shouldPassAlpha = incAlpha > alpha_test_ref;
            break;
        case kNotequal:
            shouldPassAlpha = incAlpha != alpha_test_ref;
            break;
        case kAlways:
        default:
            shouldPassAlpha = true;
            break;
    }

    return shouldPassAlpha;
}

bool doClipPlaneTest()
{
    bool res = true;

    for (int i = 0; i < kMaxClipPlanes; i++)
    {
        if (clip_plane_enables[i])
        {
            float dist = dot(clip_planes[i].xyz, pos_varying.xyz) + clip_planes[i].w * pos_varying.w;
            res        = res && (dist >= 0.0);
        }
    }

    return res;
}

vec4 doFog(vec4 currentFragment)
{

    float eyeDist = -pos_varying.z / pos_varying.w;
    float f       = 1.0;
    switch (fog_mode)
    {
        case kExp:
            f = exp(-fog_density * eyeDist);
            break;
        case kExp2:
            f = exp(-(pow(fog_density * eyeDist, 2.0)));
            break;
        case kLinear:
            f = (fog_end - eyeDist) / (fog_end - fog_start);
            break;
        default:
            break;
    }

    vec4 result = f * currentFragment + (1.0 - f) * fog_color;
    return result;
}

)";

constexpr char kGLES1DrawFShaderMultitexturing[] = R"(

bool isTextureUnitEnabled(int unit)
{
    return enable_texture_2d[unit] || enable_texture_cube_map[unit];
}

vec4 getTextureColor(int unit)
{
    vec4 res;

    switch (unit)
    {
        case 0:
            if (enable_texture_2d[0])
            {
                res = texture(tex_sampler0, texcoord0_varying.xy);
            }
            else if (enable_texture_cube_map[0])
            {
                res = texture(tex_cube_sampler0, texcoord0_varying.xyz);
            }
            break;
        case 1:
            if (enable_texture_2d[1])
            {
                res = texture(tex_sampler1, texcoord1_varying.xy);
            }
            else if (enable_texture_cube_map[1])
            {
                res = texture(tex_cube_sampler1, texcoord1_varying.xyz);
            }
            break;
        case 2:
            if (enable_texture_2d[2])
            {
                res = texture(tex_sampler2, texcoord2_varying.xy);
            }
            else if (enable_texture_cube_map[2])
            {
                res = texture(tex_cube_sampler2, texcoord2_varying.xyz);
            }
            break;
        case 3:
            if (enable_texture_2d[3])
            {
                res = texture(tex_sampler3, texcoord3_varying.xy);
            }
            else if (enable_texture_cube_map[3])
            {
                // TODO: Weird stuff happens
                // res = texture(tex_cube_sampler3, texcoord3_varying.xyz);
            }
            break;
        default:
            break;
    }

    return res;
}

vec4 getPointSpriteTextureColor(int unit)
{
    vec4 res;

    switch (unit)
    {
        case 0:
            if (enable_texture_2d[0])
            {
                res = texture(tex_sampler0, gl_PointCoord.xy);
            }
            break;
        case 1:
            if (enable_texture_2d[1])
            {
                res = texture(tex_sampler1, gl_PointCoord.xy);
            }
            break;
        case 2:
            if (enable_texture_2d[2])
            {
                res = texture(tex_sampler2, gl_PointCoord.xy);
            }
            break;
        case 3:
            if (enable_texture_2d[3])
            {
                res = texture(tex_sampler3, gl_PointCoord.xy);
            }
            break;
        default:
            break;
    }

    return res;
}

vec3 textureCombineSrcnOpnRgb(int srcnRgb,
                              int opnRgb,
                              vec4 textureEnvColor,
                              vec4 vertexColor,
                              vec4 texturePrevColor,
                              vec4 textureColor)
{
    vec3 res;
    vec4 op;

    switch (srcnRgb)
    {
        case kTexture:
            op = textureColor;
            break;
        case kConstant:
            op = textureEnvColor;
            break;
        case kPrimaryColor:
            op = vertexColor;
            break;
        case kPrevious:
            op = texturePrevColor;
            break;
        default:
            op = texturePrevColor;
            break;
    }

    switch (opnRgb)
    {
        case kSrcColor:
            res = op.rgb;
            break;
        case kOneMinusSrcColor:
            res = 1.0 - op.rgb;
            break;
        case kSrcAlpha:
            res = vec3(op.a, op.a, op.a);
            break;
        case kOneMinusSrcAlpha:
            res = vec3(1.0 - op.a, 1.0 - op.a, 1.0 - op.a);
            break;
        default:
            break;
    }

    return res;
}

float textureCombineSrcnOpnAlpha(int srcn,
                                 int opn,
                                 vec4 textureEnvColor,
                                 vec4 vertexColor,
                                 vec4 texturePrevColor,
                                 vec4 textureColor)
{
    float res;
    vec4 op;

    switch (srcn)
    {
        case kTexture:
            op = textureColor;
            break;
        case kConstant:
            op = textureEnvColor;
            break;
        case kPrimaryColor:
            op = vertexColor;
            break;
        case kPrevious:
            op = texturePrevColor;
            break;
        default:
            op = texturePrevColor;
            break;
    }

    switch (opn)
    {
        case kSrcAlpha:
            res = op.a;
            break;
        case kOneMinusSrcAlpha:
            res = 1.0 - op.a;
            break;
        default:
            break;
    }

    return res;
}

vec4 textureCombine(int combineRgb,
                    int combineAlpha,
                    int src0Rgb,
                    int src0Alpha,
                    int src1Rgb,
                    int src1Alpha,
                    int src2Rgb,
                    int src2Alpha,
                    int op0Rgb,
                    int op0Alpha,
                    int op1Rgb,
                    int op1Alpha,
                    int op2Rgb,
                    int op2Alpha,
                    vec4 textureEnvColor,
                    float rgbScale,
                    float alphaScale,
                    vec4 vertexColor,
                    vec4 texturePrevColor,
                    vec4 textureColor)
{

    vec3 resRgb;
    float resAlpha;

    vec3 arg0Rgb;
    float arg0Alpha;
    vec3 arg1Rgb;
    float arg1Alpha;
    vec3 arg2Rgb;
    float arg2Alpha;
    float dotVal;

    arg0Rgb   = textureCombineSrcnOpnRgb(src0Rgb, op0Rgb, textureEnvColor, vertexColor,
                                       texturePrevColor, textureColor);
    arg0Alpha = textureCombineSrcnOpnAlpha(src0Alpha, op0Alpha, textureEnvColor, vertexColor,
                                           texturePrevColor, textureColor);

    if (combineRgb != kReplace)
    {
        arg1Rgb = textureCombineSrcnOpnRgb(src1Rgb, op1Rgb, textureEnvColor, vertexColor,
                                           texturePrevColor, textureColor);
    }

    if (combineAlpha != kReplace)
    {
        arg1Alpha = textureCombineSrcnOpnAlpha(src1Alpha, op1Alpha, textureEnvColor, vertexColor,
                                               texturePrevColor, textureColor);
    }

    if (combineRgb == kInterpolate)
    {
        arg2Rgb = textureCombineSrcnOpnRgb(src2Rgb, op2Rgb, textureEnvColor, vertexColor,
                                           texturePrevColor, textureColor);
    }

    if (combineAlpha == kInterpolate)
    {
        arg2Alpha = textureCombineSrcnOpnAlpha(src2Alpha, op2Alpha, textureEnvColor, vertexColor,
                                               texturePrevColor, textureColor);
    }

    switch (combineRgb)
    {
        case kReplace:
            resRgb = arg0Rgb;
            break;
        case kModulate:
            resRgb = arg0Rgb * arg1Rgb;
            break;
        case kAdd:
            resRgb = arg0Rgb + arg1Rgb;
            break;
        case kAddSigned:
            resRgb = arg0Rgb + arg1Rgb - 0.5;
            break;
        case kInterpolate:
            resRgb = arg0Rgb * arg2Rgb + arg1Rgb * (1.0 - arg2Rgb);
            break;
        case kSubtract:
            resRgb = arg0Rgb - arg1Rgb;
            break;
        default:
            break;
    }

    switch (combineAlpha)
    {
        case kReplace:
            resAlpha = arg0Alpha;
            break;
        case kModulate:
            resAlpha = arg0Alpha * arg1Alpha;
            break;
        case kAdd:
            resAlpha = arg0Alpha + arg1Alpha;
            break;
        case kAddSigned:
            resAlpha = arg0Alpha + arg1Alpha - 0.5;
            break;
        case kInterpolate:
            resAlpha = arg0Alpha * arg2Alpha + arg1Alpha * (1.0 - arg2Alpha);
            break;
        case kSubtract:
            resAlpha = arg0Alpha - arg1Alpha;
            break;
        default:
            break;
    }

    if (combineRgb == kDot3Rgb || combineRgb == kDot3Rgba)
    {
        dotVal = 4.0 * dot(arg0Rgb - 0.5, arg1Rgb - 0.5);

        if (combineRgb == kDot3Rgb)
        {
            return vec4(dotVal, dotVal, dotVal, resAlpha);
        }
        else
        {
            return vec4(dotVal, dotVal, dotVal, dotVal);
        }
    }
    else
    {
        return vec4(resRgb, resAlpha);
    }
}

vec4 textureFunction(int unit,
                     int texFormat,
                     int envMode,
                     int combineRgb,
                     int combineAlpha,
                     int src0Rgb,
                     int src0Alpha,
                     int src1Rgb,
                     int src1Alpha,
                     int src2Rgb,
                     int src2Alpha,
                     int op0Rgb,
                     int op0Alpha,
                     int op1Rgb,
                     int op1Alpha,
                     int op2Rgb,
                     int op2Alpha,
                     vec4 textureEnvColor,
                     float rgbScale,
                     float alphaScale,
                     vec4 vertexColor,
                     vec4 texturePrevColor,
                     vec4 textureColor)
{

    if (!isTextureUnitEnabled(unit))
    {
        return texturePrevColor;
    }

    vec4 res;

    switch (envMode)
    {
        case kReplace:
            switch (texFormat)
            {
                case kAlpha:
                    res.rgb = texturePrevColor.rgb;
                    res.a   = textureColor.a;
                    break;
                case kRGBA:
                case kLuminanceAlpha:
                    res.rgba = textureColor.rgba;
                    break;
                case kRGB:
                case kLuminance:
                default:
                    res.rgb = textureColor.rgb;
                    res.a   = texturePrevColor.a;
                    break;
            }
            break;
        case kModulate:
            switch (texFormat)
            {
                case kAlpha:
                    res.rgb = texturePrevColor.rgb;
                    res.a   = texturePrevColor.a * textureColor.a;
                    break;
                case kRGBA:
                case kLuminanceAlpha:
                    res.rgba = texturePrevColor.rgba * textureColor.rgba;
                    break;
                case kRGB:
                case kLuminance:
                default:
                    res.rgb = texturePrevColor.rgb * textureColor.rgb;
                    res.a   = texturePrevColor.a;
                    break;
            }
            break;
        case kDecal:
            switch (texFormat)
            {
                case kRGB:
                    res.rgb = textureColor.rgb;
                    res.a   = texturePrevColor.a;
                    break;
                case kRGBA:
                    res.rgb = texturePrevColor.rgb * (1.0 - textureColor.a) +
                              textureColor.rgb * textureColor.a;
                    res.a = texturePrevColor.a;
                    break;
                case kAlpha:
                case kLuminance:
                case kLuminanceAlpha:
                default:
                    res.rgb = texturePrevColor.rgb * textureColor.rgb;
                    res.a   = texturePrevColor.a;
                    break;
            }
            break;
        case kBlend:
            switch (texFormat)
            {
                case kAlpha:
                    res.rgb = texturePrevColor.rgb;
                    res.a   = textureColor.a * texturePrevColor.a;
                    break;
                case kLuminance:
                case kRGB:
                    res.rgb = texturePrevColor.rgb * (1.0 - textureColor.rgb) +
                              textureEnvColor.rgb * textureColor.rgb;
                    res.a = texturePrevColor.a;
                    break;
                case kLuminanceAlpha:
                case kRGBA:
                default:
                    res.rgb = texturePrevColor.rgb * (1.0 - textureColor.rgb) +
                              textureEnvColor.rgb * textureColor.rgb;
                    res.a = textureColor.a * texturePrevColor.a;
                    break;
            }
            break;
        case kAdd:
            switch (texFormat)
            {
                case kAlpha:
                    res.rgb = texturePrevColor.rgb;
                    res.a   = textureColor.a * texturePrevColor.a;
                    break;
                case kLuminance:
                case kRGB:
                    res.rgb = texturePrevColor.rgb + textureColor.rgb;
                    res.a   = texturePrevColor.a;
                    break;
                case kLuminanceAlpha:
                case kRGBA:
                default:
                    res.rgb = texturePrevColor.rgb + textureColor.rgb;
                    res.a   = textureColor.a * texturePrevColor.a;
                    break;
            }
            break;
        case kCombine:
            res = textureCombine(combineRgb, combineAlpha, src0Rgb, src0Alpha, src1Rgb, src1Alpha,
                                 src2Rgb, src2Alpha, op0Rgb, op0Alpha, op1Rgb, op1Alpha, op2Rgb,
                                 op2Alpha, textureEnvColor, rgbScale, alphaScale, vertexColor,
                                 texturePrevColor, textureColor);
            res.rgb *= rgbScale;
            res.a *= alphaScale;
            break;
        default:
            break;
    }

    return clamp(res, 0.0, 1.0);
}
)";

constexpr char kGLES1DrawFShaderMain[] = R"(
void main()
{
    if (enable_clip_planes && !enable_draw_texture)
    {
        if (!doClipPlaneTest())
        {
            discard;
        }
    }

    vec4 currentFragment;
    vec4 vertex_color;

    if (shade_model_flat)
    {
        vertex_color = color_varying_flat;
    }
    else
    {
        vertex_color = color_varying;
    }

    currentFragment = vertex_color;

    vec4 texturePrevColor = currentFragment;

    for (int i = 0; i < kMaxTexUnits; i++)
    {
        vec4 textureColor;

        if (point_rasterization && point_sprite_enabled &&
            point_sprite_coord_replace[i]) {
            textureColor = getPointSpriteTextureColor(i);
        } else {
            textureColor = getTextureColor(i);
        }

        currentFragment = textureFunction(
            i, texture_format[i], texture_env_mode[i], combine_rgb[i], combine_alpha[i],
            src0_rgb[i], src0_alpha[i], src1_rgb[i], src1_alpha[i], src2_rgb[i], src2_alpha[i],
            op0_rgb[i], op0_alpha[i], op1_rgb[i], op1_alpha[i], op2_rgb[i], op2_alpha[i],
            texture_env_color[i], texture_env_rgb_scale[i], texture_env_alpha_scale[i],
            vertex_color, texturePrevColor, textureColor);

        texturePrevColor = currentFragment;
    }

    if (enable_lighting)
    {
        currentFragment = doLighting(currentFragment);
    }

    if (enable_fog)
    {
        currentFragment = doFog(currentFragment);
    }

    if (enable_alpha_test && !doAlphaTest(currentFragment))
    {
        discard;
    }

    frag_color = currentFragment;
}
)";
