/*
 * Copyright (c) 2023 - 2026 ThorVG project. All rights reserved.

 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:

 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.

 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "tvgWgShaderSrc.h"
#include <string>

//************************************************************************
// graphics shader source: stencil
//************************************************************************

const char* cShaderSrc_Stencil = R"(
struct VertexInput { @location(0) position: vec2f };
struct VertexOutput { @builtin(position) position: vec4f };

@group(0) @binding(0) var<uniform> uViewMat  : mat4x4f;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = uViewMat * vec4f(in.position.xy, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    return vec4f(0.0, 0.0, 0.0, 1.0);
}
)";

//************************************************************************
// graphics shader source: depth
//************************************************************************

const char* cShaderSrc_Depth = R"(
struct VertexInput { @location(0) position: vec2f };
struct VertexOutput { @builtin(position) position: vec4f };

@group(0) @binding(0) var<uniform> uViewMat  : mat4x4f;
@group(1) @binding(0) var<uniform> uDepth : f32;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = uViewMat * vec4f(in.position.xy, 0.0, 1.0);
    out.position.z = uDepth;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    return vec4f(1.0, 0.5, 0.0, 1.0);
}
)";

//************************************************************************
// graphics shader source: solid normal blend
//************************************************************************

const char* cShaderSrc_Solid = R"(
struct VertexInput { @location(0) position: vec2f, @location(1) color: vec4f };
struct VertexOutput { @builtin(position) position: vec4f, @location(0) color: vec4f };

@group(0) @binding(0) var<uniform> uViewMat : mat4x4f;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = uViewMat * vec4f(in.position.xy, 0.0, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let Sc = in.color;
    let So = 1.0;
    return vec4f(Sc.rgb * Sc.a * So, Sc.a * So);
}
)";

//************************************************************************
// graphics shader source: linear normal blend
//************************************************************************

const char* cShaderSrc_Linear = R"(
struct VertexInput { @location(0) position: vec2f };
struct VertexOutput { @builtin(position) position : vec4f, @location(0) vGradCoord : vec4f };
struct GradSettings  { transform: mat4x4f, coords: vec4f, focal: vec4f };
struct PaintSettings { options: vec4f, color: vec4f, gradient: GradSettings };

// uniforms
@group(0) @binding(0) var<uniform> uViewMat : mat4x4f;
@group(1) @binding(0) var<uniform> uPaintSettings : PaintSettings;
@group(2) @binding(0) var uSamplerGrad : sampler;
@group(2) @binding(1) var uTextureGrad : texture_2d<f32>;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = uViewMat * vec4f(in.position.xy, 0.0, 1.0);
    out.vGradCoord = uPaintSettings.gradient.transform * vec4f(in.position.xy, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let pos = in.vGradCoord.xy;
    let st = uPaintSettings.gradient.coords.xy;
    let ed = uPaintSettings.gradient.coords.zw;
    let ba = ed - st;
    let t = dot(pos - st, ba) / dot(ba, ba);
    let Sc = textureSample(uTextureGrad, uSamplerGrad, vec2f(t, 0.5));
    let So = uPaintSettings.options.a;
    return vec4f(Sc.rgb * Sc.a * So, Sc.a * So);
}
)";

//************************************************************************
// graphics shader source: radial normal blend
//************************************************************************

const char* cShaderSrc_Radial = R"(
struct VertexInput { @location(0) position: vec2f };
struct VertexOutput { @builtin(position) position : vec4f, @location(0) vGradCoord : vec4f };
struct GradSettings  { transform: mat4x4f, coords: vec4f, focal: vec4f };
struct PaintSettings { options: vec4f, color: vec4f, gradient: GradSettings };

@group(0) @binding(0) var<uniform> uViewMat : mat4x4f;
@group(1) @binding(0) var<uniform> uPaintSettings : PaintSettings;
@group(2) @binding(0) var uSamplerGrad : sampler;
@group(2) @binding(1) var uTextureGrad : texture_2d<f32>;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = uViewMat * vec4f(in.position.xy, 0.0, 1.0);
    out.vGradCoord = uPaintSettings.gradient.transform * vec4f(in.position.xy, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    // original data
    let d0 = in.vGradCoord.xy - uPaintSettings.gradient.coords.xy;
    let d1 = uPaintSettings.gradient.coords.xy - uPaintSettings.gradient.focal.xy;
    let r0 = uPaintSettings.gradient.coords.z;
    let rd = uPaintSettings.gradient.focal.z - uPaintSettings.gradient.coords.z;
    let a = 1.0*dot(d1, d1) - 1.0*rd*rd;
    let b = 2.0*dot(d0, d1) - 2.0*r0*rd;
    let c = 1.0*dot(d0, d0) - 1.0*r0*r0;
    let d = b*b - 4*a*c;
    var t = 0.0;
    if (d >= 0) { t = min(1.0, (-b + sqrt(d))/(2*a)); }
    if ((c > 0) && (t >= 1.0)) { t = 0.0; }
    let Sc = textureSample(uTextureGrad, uSamplerGrad, vec2f(1.0 - t, 0.5));
    let So = uPaintSettings.options.a;
    return vec4f(Sc.rgb * Sc.a * So, Sc.a * So);
}
)";

//************************************************************************
// graphics shader source: image normal blend
//************************************************************************

const char* cShaderSrc_Image = R"(
struct VertexInput { @location(0) position: vec2f, @location(1) texCoord: vec2f };
struct VertexOutput { @builtin(position) position: vec4f, @location(0) vTexCoord: vec2f };
struct PaintSettings { options: vec4f };

@group(0) @binding(0) var<uniform> uViewMat : mat4x4f;
@group(1) @binding(0) var<uniform> uPaintSettings : PaintSettings;
@group(2) @binding(0) var uSampler     : sampler;
@group(2) @binding(1) var uTextureView : texture_2d<f32>;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = uViewMat * vec4f(in.position.xy, 0.0, 1.0);
    out.vTexCoord = in.texCoord;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    var Sc: vec4f = textureSample(uTextureView, uSampler, in.vTexCoord.xy);
    let So: f32 = uPaintSettings.options.a;
    return vec4f(Sc.rgb * Sc.a * So, Sc.a * So);
};
)";

//************************************************************************
// graphics shader source: scene normal blend
//************************************************************************

const char* cShaderSrc_Scene = R"(
struct VertexInput { @location(0) position: vec2f, @location(1) texCoord: vec2f };
struct VertexOutput { @builtin(position) position: vec4f, @location(0) vTexCoord: vec2f };

@group(0) @binding(0) var uSamplerSrc : sampler;
@group(0) @binding(1) var uTextureSrc : texture_2d<f32>;
@group(1) @binding(0) var<uniform> So : f32;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4f(in.position.xy, 0.0, 1.0);
    out.vTexCoord = in.texCoord;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let Sc = textureSample(uTextureSrc, uSamplerSrc, in.vTexCoord.xy);
    return vec4f(Sc.rgb * So, Sc.a * So);
};
)";

//************************************************************************
// graphics shader source: custom shaders
//************************************************************************

const char* cShaderSrc_Solid_Blend = R"(
struct VertexInput { @location(0) position: vec2f, @location(1) color: vec4f };
struct VertexOutput { @builtin(position) position: vec4f, @location(0) vColor: vec4f, @location(1) vScrCoord: vec2f };

@group(0) @binding(0) var<uniform> uViewMat : mat4x4f;
@group(1) @binding(0) var uSamplerDst : sampler;
@group(1) @binding(1) var uTextureDst : texture_2d<f32>;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let pos = uViewMat * vec4f(in.position.xy, 0.0, 1.0);
    out.position = pos;
    out.vColor = in.color;
    out.vScrCoord = vec2f(0.5 + pos.x * 0.5, 0.5 - pos.y * 0.5);
    return out;
}

struct FragData { Sc: vec3f, Sa: f32, So: f32, Dc: vec3f, Da: f32 };
fn getFragData(in: VertexOutput) -> FragData {
    // get source data
    let colorSrc = in.vColor;
    let colorDst = textureSample(uTextureDst, uSamplerDst, in.vScrCoord.xy);
    // fill fragment data
    var data: FragData;
    data.Sc = colorSrc.rgb;
    data.Sa = colorSrc.a;
    data.So = 1.0;
    data.Dc = colorDst.rgb;
    data.Da = colorDst.a;
    data.Sc = data.Sa * data.So * data.Sc;
    data.Sa = data.Sa * data.So;
    return data;
};

fn postProcess(d: FragData, R: vec4f) -> vec4f { return R; };
)";

const char* cShaderSrc_Linear_Blend = R"(
struct VertexInput { @location(0) position: vec2f };
struct VertexOutput { @builtin(position) position: vec4f, @location(0) vGradCoord : vec4f, @location(1) vScrCoord: vec2f };
struct GradSettings  { transform: mat4x4f, coords: vec4f, focal: vec4f };
struct PaintSettings { options: vec4f, color: vec4f, gradient: GradSettings };

@group(0) @binding(0) var<uniform> uViewMat : mat4x4f;
@group(1) @binding(0) var<uniform> uPaintSettings : PaintSettings;
@group(2) @binding(0) var uSamplerGrad : sampler;
@group(2) @binding(1) var uTextureGrad : texture_2d<f32>;
@group(3) @binding(0) var uSamplerDst : sampler;
@group(3) @binding(1) var uTextureDst : texture_2d<f32>;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let pos = uViewMat * vec4f(in.position.xy, 0.0, 1.0);
    out.position = pos;
    out.vGradCoord = uPaintSettings.gradient.transform * vec4f(in.position.xy, 0.0, 1.0);
    out.vScrCoord = vec2f(0.5 + pos.x * 0.5, 0.5 - pos.y * 0.5);
    return out;
}

struct FragData { Sc: vec3f, Sa: f32, So: f32, Dc: vec3f, Da: f32 };
fn getFragData(in: VertexOutput) -> FragData {
    // get source data
    let pos = in.vGradCoord.xy;
    let st = uPaintSettings.gradient.coords.xy;
    let ed = uPaintSettings.gradient.coords.zw;
    let ba = ed - st;
    let t = dot(pos - st, ba) / dot(ba, ba);
    let colorSrc = textureSample(uTextureGrad, uSamplerGrad, vec2f(t, 0.5));
    let colorDst = textureSample(uTextureDst, uSamplerDst, in.vScrCoord.xy);
    // fill fragment data
    var data: FragData;
    data.Sc = colorSrc.rgb;
    data.Sa = colorSrc.a;
    data.So = uPaintSettings.options.a;
    data.Dc = colorDst.rgb;
    data.Da = colorDst.a;
    data.Sc = mix(data.Dc, data.Sc, data.Sa * data.So);
    data.Sa = mix(data.Da,     1.0, data.Sa * data.So);
    return data;
};

fn postProcess(d: FragData, R: vec4f) -> vec4f { return R; };
)";

const char* cShaderSrc_Radial_Blend = R"(
struct VertexInput { @location(0) position: vec2f };
struct VertexOutput { @builtin(position) position: vec4f, @location(0) vGradCoord : vec4f, @location(1) vScrCoord: vec2f };
struct GradSettings  { transform: mat4x4f, coords: vec4f, focal: vec4f };
struct PaintSettings { options: vec4f, color: vec4f, gradient: GradSettings };

@group(0) @binding(0) var<uniform> uViewMat : mat4x4f;
@group(1) @binding(0) var<uniform> uPaintSettings : PaintSettings;
@group(2) @binding(0) var uSamplerGrad : sampler;
@group(2) @binding(1) var uTextureGrad : texture_2d<f32>;
@group(3) @binding(0) var uSamplerDst : sampler;
@group(3) @binding(1) var uTextureDst : texture_2d<f32>;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let pos = uViewMat * vec4f(in.position.xy, 0.0, 1.0);
    out.position = pos;
    out.vGradCoord = uPaintSettings.gradient.transform * vec4f(in.position.xy, 0.0, 1.0);
    out.vScrCoord = vec2f(0.5 + pos.x * 0.5, 0.5 - pos.y * 0.5);
    return out;
}

struct FragData { Sc: vec3f, Sa: f32, So: f32, Dc: vec3f, Da: f32 };
fn getFragData(in: VertexOutput) -> FragData {
    let d0 = in.vGradCoord.xy - uPaintSettings.gradient.coords.xy;
    let d1 = uPaintSettings.gradient.coords.xy - uPaintSettings.gradient.focal.xy;
    let r0 = uPaintSettings.gradient.coords.z;
    let rd = uPaintSettings.gradient.focal.z - uPaintSettings.gradient.coords.z;
    let a = 1.0*dot(d1, d1) - 1.0*rd*rd;
    let b = 2.0*dot(d0, d1) - 2.0*r0*rd;
    let c = 1.0*dot(d0, d0) - 1.0*r0*r0;
    let d = b*b - 4*a*c;
    var t = 0.0;
    if (d >= 0) { t = min(1.0, (-b + sqrt(d))/(2*a)); }
    if ((c > 0) && (t >= 1.0)) { t = 0.0; }
    let colorSrc = textureSample(uTextureGrad, uSamplerGrad, vec2f(1.0 - t, 0.5));
    let colorDst = textureSample(uTextureDst, uSamplerDst, in.vScrCoord.xy);
    // fill fragment data
    var data: FragData;
    data.Sc = colorSrc.rgb;
    data.Sa = colorSrc.a;
    data.So = uPaintSettings.options.a;
    data.Dc = colorDst.rgb;
    data.Da = colorDst.a;
    data.Sc = mix(data.Dc, data.Sc, data.Sa * data.So);
    data.Sa = mix(data.Da,     1.0, data.Sa * data.So);
    return data;
};

fn postProcess(d: FragData, R: vec4f) -> vec4f { return R; };
)";

const char* cShaderSrc_Image_Blend = R"(
struct VertexInput { @location(0) position: vec2f, @location(1) texCoord: vec2f };
struct VertexOutput { @builtin(position) position: vec4f, @location(0) vTexCoord : vec2f, @location(1) vScrCoord: vec2f };
struct PaintSettings { options: vec4f };

@group(0) @binding(0) var<uniform> uViewMat : mat4x4f;
@group(1) @binding(0) var<uniform> uPaintSettings : PaintSettings;
@group(2) @binding(0) var uSamplerSrc : sampler;
@group(2) @binding(1) var uTextureSrc : texture_2d<f32>;
@group(3) @binding(0) var uSamplerDst : sampler;
@group(3) @binding(1) var uTextureDst : texture_2d<f32>;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let pos = uViewMat * vec4f(in.position.xy, 0.0, 1.0);
    out.position = pos;
    out.vTexCoord = in.texCoord;
    out.vScrCoord = vec2f(0.5 + pos.x * 0.5, 0.5 - pos.y * 0.5);
    return out;
}

struct FragData { Sc: vec3f, Sa: f32, So: f32, Dc: vec3f, Da: f32 };
fn getFragData(in: VertexOutput) -> FragData {
    // get source data
    let colorSrc = textureSample(uTextureSrc, uSamplerSrc, in.vTexCoord.xy);
    let colorDst = textureSample(uTextureDst, uSamplerDst, in.vScrCoord.xy);
    // fill fragment data
    var data: FragData;
    data.Sc = colorSrc.rgb;
    data.Sa = colorSrc.a;
    data.So = uPaintSettings.options.a;
    data.Dc = colorDst.rgb;
    data.Da = colorDst.a;
    data.Sc = data.Sc * data.So;
    data.Sa = data.Sa * data.So;
    return data;
};

fn postProcess(d: FragData, R: vec4f) -> vec4f { return mix(vec4(d.Dc, d.Da), R, d.Sa); };
)";

const char* cShaderSrc_Scene_Blend = R"(
struct VertexInput { @location(0) position: vec2f, @location(1) texCoord: vec2f };
struct VertexOutput { @builtin(position) position: vec4f, @location(0) vScrCoord: vec2f };

@group(0) @binding(0) var uSamplerSrc : sampler;
@group(0) @binding(1) var uTextureSrc : texture_2d<f32>;
@group(1) @binding(0) var uSamplerDst : sampler;
@group(1) @binding(1) var uTextureDst : texture_2d<f32>;
@group(2) @binding(0) var<uniform> So : f32;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4f(in.position.xy, 0.0, 1.0);
    out.vScrCoord = in.texCoord;
    return out;
}

struct FragData { Sc: vec3f, Sa: f32, So: f32, Dc: vec3f, Da: f32 };
fn getFragData(in: VertexOutput) -> FragData {
    // get source data
    let colorSrc = textureSample(uTextureSrc, uSamplerSrc, in.vScrCoord.xy);
    let colorDst = textureSample(uTextureDst, uSamplerDst, in.vScrCoord.xy);
    // fill fragment data
    var data: FragData;
    data.Sc = min(vec3f(1.0, 1.0, 1.0), colorSrc.rgb / select(colorSrc.a, 1.0, (colorSrc.a == 0.0) || (colorSrc.a == 1.0)));
    data.Sa = colorSrc.a;
    data.Dc = colorDst.rgb;
    data.Da = colorDst.a;
    return data;
};

fn postProcess(d: FragData, R: vec4f) -> vec4f { return mix(vec4(d.Dc, d.Da), R, d.Sa * So); };
)";

const char* cShaderSrc_BlendFuncs = R"(
const One = vec3f(1.0, 1.0, 1.0);

const LUM_W = vec3f(0.3, 0.59, 0.11);

fn setLum(colorIn: vec3f, l: f32) -> vec3f {
    var color = colorIn + vec3f(l - dot(colorIn, LUM_W));
    let ll = dot(color, LUM_W);
    let n = min(color.r, min(color.g, color.b));
    let x = max(color.r, max(color.g, color.b));

    if (n < 0.0) {
        color = vec3f(ll) + (color - vec3f(ll)) * (ll / (ll - n));
    }
    if (x > 1.0) {
        color = vec3f(ll) + (color - vec3f(ll)) * ((1.0 - ll) / (x - ll));
    }
    return color;
};

fn sat(color: vec3f) -> f32 {
    return max(color.r, max(color.g, color.b)) - min(color.r, min(color.g, color.b));
};

fn setSat(colorIn: vec3f, s: f32) -> vec3f {
    let rMin = step(colorIn.r, colorIn.g) * step(colorIn.r, colorIn.b);
    let gMin = (1.0 - rMin) * step(colorIn.g, colorIn.r) * step(colorIn.g, colorIn.b);
    let minMask = vec3f(rMin, gMin, 1.0 - rMin - gMin);

    let bMax = step(colorIn.r, colorIn.b) * step(colorIn.g, colorIn.b);
    let gMax = (1.0 - bMax) * step(colorIn.r, colorIn.g) * step(colorIn.b, colorIn.g);
    let maxMask = vec3f(1.0 - bMax - gMax, gMax, bMax);
    let midMask = vec3f(1.0) - minMask - maxMask;

    let cMin = dot(colorIn, minMask);
    let cMid = dot(colorIn, midMask);
    let cMax = dot(colorIn, maxMask);
    let delta = cMax - cMin;
    let deltaMask = sign(delta);
    let scale = deltaMask * s / max(delta, 1e-6);

    return maxMask * vec3f(s * deltaMask) + midMask * vec3f((cMid - cMin) * scale);
};

@fragment
fn fs_main_Normal(in: VertexOutput) -> @location(0) vec4f {
    // used as debug blend method
    return vec4f(1.0, 0.0, 0.0, 1.0);
}

@fragment
fn fs_main_Multiply(in: VertexOutput) -> @location(0) vec4f {
    let d: FragData = getFragData(in);
    var Rc = d.Sc;
    if (d.Da > 0.0) {
        Rc = d.Sc * min(One, d.Dc / d.Da);
        Rc = mix(d.Sc, Rc, d.Da);
    };
    return postProcess(d, vec4f(Rc, 1.0));
}


@fragment
fn fs_main_Screen(in: VertexOutput) -> @location(0) vec4f {
    let d: FragData = getFragData(in);
    let Rc = d.Sc + d.Dc - d.Sc * d.Dc;
    return postProcess(d, vec4f(Rc, 1.0));
}

@fragment
fn fs_main_Overlay(in: VertexOutput) -> @location(0) vec4f {
    let d: FragData = getFragData(in);
    var Rc = d.Sc;
    if (d.Da > 0.0) {
        let Dc = min(One, d.Dc / d.Da);
        Rc.r = select(1.0 - min(1.0, 2 * (1 - d.Sc.r) * (1 - Dc.r)), min(1.0, 2 * d.Sc.r * Dc.r), (Dc.r < 0.5));
        Rc.g = select(1.0 - min(1.0, 2 * (1 - d.Sc.g) * (1 - Dc.g)), min(1.0, 2 * d.Sc.g * Dc.g), (Dc.g < 0.5));
        Rc.b = select(1.0 - min(1.0, 2 * (1 - d.Sc.b) * (1 - Dc.b)), min(1.0, 2 * d.Sc.b * Dc.b), (Dc.b < 0.5));
        Rc = mix(d.Sc, Rc, d.Da);
    }
    return postProcess(d, vec4f(Rc, 1.0));
}

@fragment
fn fs_main_Darken(in: VertexOutput) -> @location(0) vec4f {
    let d: FragData = getFragData(in);
    var Rc = d.Sc;
    if (d.Da > 0.0) {
        Rc = min(d.Sc, min(One, d.Dc / d.Da));
        Rc = mix(d.Sc, Rc, d.Da);
    };
    return postProcess(d, vec4f(Rc, 1.0));
}

@fragment
fn fs_main_Lighten(in: VertexOutput) -> @location(0) vec4f {
    let d: FragData = getFragData(in);
    let Rc = max(d.Sc, d.Dc);
    return postProcess(d, vec4f(Rc, 1.0));
}

@fragment
fn fs_main_ColorDodge(in: VertexOutput) -> @location(0) vec4f {
    let d: FragData = getFragData(in);
    var Rc = d.Sc;
    if (d.Da > 0.0) {
        let Dc = min(One, d.Dc / d.Da);
        Rc.r = select(0.0, select(1.0, min(1.0, Dc.r / (1.0 - d.Sc.r)), d.Sc.r < 1.0), Dc.r > 0.0);
        Rc.g = select(0.0, select(1.0, min(1.0, Dc.g / (1.0 - d.Sc.g)), d.Sc.g < 1.0), Dc.g > 0.0);
        Rc.b = select(0.0, select(1.0, min(1.0, Dc.b / (1.0 - d.Sc.b)), d.Sc.b < 1.0), Dc.b > 0.0);
        Rc = mix(d.Sc, Rc, d.Da);
    }
    return postProcess(d, vec4f(Rc, 1.0));
}

@fragment
fn fs_main_ColorBurn(in: VertexOutput) -> @location(0) vec4f {
    let d: FragData = getFragData(in);
    var Rc = d.Sc;
    if (d.Da > 0.0) {
        let Dc = min(One, d.Dc / d.Da);
        Rc.r = select(select(1.0, 0.0, Dc.r < 1), 1.0 - min(1.0, (1.0 - Dc.r) / d.Sc.r), d.Sc.r > 0);
        Rc.g = select(select(1.0, 0.0, Dc.g < 1), 1.0 - min(1.0, (1.0 - Dc.g) / d.Sc.g), d.Sc.g > 0);
        Rc.b = select(select(1.0, 0.0, Dc.b < 1), 1.0 - min(1.0, (1.0 - Dc.b) / d.Sc.b), d.Sc.b > 0);
        Rc = mix(d.Sc, Rc, d.Da);
    }
    return postProcess(d, vec4f(Rc, 1.0));
}

@fragment
fn fs_main_HardLight(in: VertexOutput) -> @location(0) vec4f {
    let d: FragData = getFragData(in);
    var Rc = d.Sc;
    if (d.Da > 0.0) {
        let Dc = min(One, d.Dc / d.Da);
        Rc.r = select(1.0 - min(1.0, 2 * (1 - d.Sc.r) * (1 - Dc.r)), min(1.0, 2 * d.Sc.r * Dc.r), (d.Sc.r < 0.5));
        Rc.g = select(1.0 - min(1.0, 2 * (1 - d.Sc.g) * (1 - Dc.g)), min(1.0, 2 * d.Sc.g * Dc.g), (d.Sc.g < 0.5));
        Rc.b = select(1.0 - min(1.0, 2 * (1 - d.Sc.b) * (1 - Dc.b)), min(1.0, 2 * d.Sc.b * Dc.b), (d.Sc.b < 0.5));
        Rc = mix(d.Sc, Rc, d.Da);
    }
    return postProcess(d, vec4f(Rc, 1.0));
}

@fragment
fn fs_main_SoftLight(in: VertexOutput) -> @location(0) vec4f {
    let d: FragData = getFragData(in);
    var Rc = d.Sc;
    if (d.Da > 0.0) {
        let Dc = min(One, d.Dc / d.Da);
        let Dlow = ((16.0 * Dc - 12.0) * Dc + 4.0) * Dc;
        let Dhigh = sqrt(Dc);
        let D = select(Dhigh, Dlow, Dc <= vec3f(0.25));
        let low = Dc - (1.0 - 2.0 * d.Sc) * Dc * (1.0 - Dc);
        let high = Dc + (2.0 * d.Sc - 1.0) * (D - Dc);
        Rc = select(high, low, d.Sc <= vec3f(0.5));
        Rc = clamp(Rc, vec3f(0.0), One);
        Rc = mix(d.Sc, Rc, d.Da);
    };
    return postProcess(d, vec4f(Rc, 1.0));
}

@fragment
fn fs_main_Difference(in: VertexOutput) -> @location(0) vec4f {
    let d: FragData = getFragData(in);
    let Rc = abs(d.Dc - d.Sc);
    return postProcess(d, vec4f(Rc, 1.0));
}

@fragment
fn fs_main_Exclusion(in: VertexOutput) -> @location(0) vec4f {
    let d: FragData = getFragData(in);
    let Rc = d.Sc + d.Dc - 2 * d.Sc * d.Dc;
    return postProcess(d, vec4f(Rc, 1.0));
}

@fragment
fn fs_main_Hue(in: VertexOutput) -> @location(0) vec4f {
    let d: FragData = getFragData(in);
    var Rc = d.Sc;
    if (d.Da > 0.0) {
        let Dc = min(One, d.Dc / d.Da);
        Rc = setSat(d.Sc, sat(Dc));
        Rc = setLum(Rc, dot(Dc, LUM_W));
        Rc = mix(d.Sc, Rc, d.Da);
    };
    return postProcess(d, vec4f(Rc, 1.0));
}

@fragment
fn fs_main_Saturation(in: VertexOutput) -> @location(0) vec4f {
    let d: FragData = getFragData(in);
    var Rc = d.Sc;
    if (d.Da > 0.0) {
        let Dc = min(One, d.Dc / d.Da);
        let s = max(d.Sc.r, max(d.Sc.g, d.Sc.b)) - min(d.Sc.r, min(d.Sc.g, d.Sc.b));
        let n = min(Dc.r, min(Dc.g, Dc.b));
        let x = max(Dc.r, max(Dc.g, Dc.b));
        Rc = vec3f(0.0);
        if (x > n) {
            Rc = (Dc - vec3f(n)) * (s / (x - n));
        }
        Rc = setLum(Rc, dot(Dc, LUM_W));

        Rc = mix(d.Sc, Rc, d.Da);
    };
    return postProcess(d, vec4f(Rc, 1.0));
}

@fragment
fn fs_main_Color(in: VertexOutput) -> @location(0) vec4f {
    let d: FragData = getFragData(in);
    var Rc = d.Sc;
    if (d.Da > 0.0) {
        let Dc = min(One, d.Dc / d.Da);
        Rc = setLum(d.Sc, dot(Dc, LUM_W));

        Rc = mix(d.Sc, Rc, d.Da);
    };
    return postProcess(d, vec4f(Rc, 1.0));
}

@fragment
fn fs_main_Luminosity(in: VertexOutput) -> @location(0) vec4f {
    let d: FragData = getFragData(in);
    var Rc = d.Sc;
    if (d.Da > 0.0) {
        let Dc = min(One, d.Dc / d.Da);
        Rc = setLum(Dc, dot(d.Sc, LUM_W));

        Rc = mix(d.Sc, Rc, d.Da);
    };
    return postProcess(d, vec4f(Rc, 1.0));
}

@fragment
fn fs_main_Add(in: VertexOutput) -> @location(0) vec4f {
    let d: FragData = getFragData(in);
    let Rc = min(One, d.Sc + d.Dc);
    return postProcess(d, vec4f(Rc, 1.0));
}
)";

//************************************************************************
// graphics shader source: scene compose
//************************************************************************

const char* cShaderSrc_Scene_Compose = R"(
struct VertexInput { @location(0) position: vec2f, @location(1) texCoord: vec2f };
struct VertexOutput { @builtin(position) position: vec4f, @location(0) texCoord: vec2f };

@group(0) @binding(0) var uSamplerSrc : sampler;
@group(0) @binding(1) var uTextureSrc : texture_2d<f32>;
@group(1) @binding(0) var uSamplerMsk : sampler;
@group(1) @binding(1) var uTextureMsk : texture_2d<f32>;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4f(in.position.xy, 0.0, 1.0);
    out.texCoord = in.texCoord;
    return out;
}

@fragment
fn fs_main_None(in: VertexOutput) -> @location(0) vec4f {
    let src: vec4f = textureSample(uTextureSrc, uSamplerSrc, in.texCoord.xy);
    return vec4f(src);
};

@fragment
fn fs_main_ClipPath(in: VertexOutput) -> @location(0) vec4f {
    let src: vec4f = textureSample(uTextureSrc, uSamplerSrc, in.texCoord.xy);
    let msk: vec4f = textureSample(uTextureMsk, uSamplerMsk, in.texCoord.xy);
    return vec4f(src * msk.a);
};

@fragment
fn fs_main_AlphaMask(in: VertexOutput) -> @location(0) vec4f {
    let src: vec4f = textureSample(uTextureSrc, uSamplerSrc, in.texCoord.xy);
    let msk: vec4f = textureSample(uTextureMsk, uSamplerMsk, in.texCoord.xy);
    return vec4f(src * msk.a);
};

@fragment
fn fs_main_InvAlphaMask(in: VertexOutput) -> @location(0) vec4f {
    let src: vec4f = textureSample(uTextureSrc, uSamplerSrc, in.texCoord.xy);
    let msk: vec4f = textureSample(uTextureMsk, uSamplerMsk, in.texCoord.xy);
    return vec4f(src * (1.0 - msk.a));
};

@fragment
fn fs_main_LumaMask(in: VertexOutput) -> @location(0) vec4f {
    let src: vec4f = textureSample(uTextureSrc, uSamplerSrc, in.texCoord.xy);
    let msk: vec4f = textureSample(uTextureMsk, uSamplerMsk, in.texCoord.xy);
    let luma: f32 = dot(msk.rgb, vec3f(0.2125, 0.7154, 0.0721));
    return vec4f(src * luma);
};

@fragment
fn fs_main_InvLumaMask(in: VertexOutput) -> @location(0) vec4f {
    let src: vec4f = textureSample(uTextureSrc, uSamplerSrc, in.texCoord.xy);
    let msk: vec4f = textureSample(uTextureMsk, uSamplerMsk, in.texCoord.xy);
    let luma: f32 = dot(msk.rgb, vec3f(0.2125, 0.7154, 0.0721));
    return vec4f(src * (1.0 - luma));
};

@fragment
fn fs_main_AddMask(in: VertexOutput) -> @location(0) vec4f {
    let src: vec4f = textureSample(uTextureSrc, uSamplerSrc, in.texCoord.xy);
    let msk: vec4f = textureSample(uTextureMsk, uSamplerMsk, in.texCoord.xy);
    return vec4f(src.rgb, src.a + msk.a * (1.0 - src.a));
};

@fragment
fn fs_main_SubtractMask(in: VertexOutput) -> @location(0) vec4f {
    let src: vec4f = textureSample(uTextureSrc, uSamplerSrc, in.texCoord.xy);
    let msk: vec4f = textureSample(uTextureMsk, uSamplerMsk, in.texCoord.xy);
    return vec4f(src.rgb, src.a * (1.0 - msk.a));
};

@fragment
fn fs_main_IntersectMask(in: VertexOutput) -> @location(0) vec4f {
    let src: vec4f = textureSample(uTextureSrc, uSamplerSrc, in.texCoord.xy);
    let msk: vec4f = textureSample(uTextureMsk, uSamplerMsk, in.texCoord.xy);
    return vec4f(src.rgb, src.a * msk.a);
};

@fragment
fn fs_main_DifferenceMask(in: VertexOutput) -> @location(0) vec4f {
    let src: vec4f = textureSample(uTextureSrc, uSamplerSrc, in.texCoord.xy);
    let msk: vec4f = textureSample(uTextureMsk, uSamplerMsk, in.texCoord.xy);
    return vec4f(src.rgb, src.a * (1.0 - msk.a) + msk.a * (1.0 - src.a));
};

@fragment
fn fs_main_LightenMask(in: VertexOutput) -> @location(0) vec4f {
    let src: vec4f = textureSample(uTextureSrc, uSamplerSrc, in.texCoord.xy);
    let msk: vec4f = textureSample(uTextureMsk, uSamplerMsk, in.texCoord.xy);
    return vec4f(src.rgb, max(src.a, msk.a));
};

@fragment
fn fs_main_DarkenMask(in: VertexOutput) -> @location(0) vec4f {
    let src: vec4f = textureSample(uTextureSrc, uSamplerSrc, in.texCoord.xy);
    let msk: vec4f = textureSample(uTextureMsk, uSamplerMsk, in.texCoord.xy);
    return vec4f(src.rgb, min(src.a, msk.a));
};
)";

//************************************************************************
// graphics shader source: texture blit
//************************************************************************

const char* cShaderSrc_Blit = R"(
struct VertexInput { @location(0) position: vec2f, @location(1) texCoord: vec2f };
struct VertexOutput { @builtin(position) position: vec4f, @location(0) texCoord: vec2f };

@group(0) @binding(0) var uSamplerSrc : sampler;
@group(0) @binding(1) var uTextureSrc : texture_2d<f32>;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4f(in.position.xy, 0.0, 1.0);
    out.texCoord = in.texCoord;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    return textureSample(uTextureSrc, uSamplerSrc, in.texCoord.xy);
};
)";

//************************************************************************
// shader source: effects
//************************************************************************

const char* cShaderSrc_Shadow = R"(
struct VertexInput { @location(0) position: vec2f, @location(1) texCoord: vec2f };
struct VertexOutput { @builtin(position) position: vec4f, @location(0) texCoord: vec2f };

@group(0) @binding(0) var uSamplerSrc : sampler;
@group(0) @binding(1) var uTextureSrc : texture_2d<f32>;
@group(1) @binding(0) var uSamplerSdw : sampler;
@group(1) @binding(1) var uTextureSdw : texture_2d<f32>;
@group(2) @binding(0) var<uniform> settings: array<vec4f, 3>;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4f(in.position.xy, 0.0, 1.0);
    out.texCoord = in.texCoord;
    return out;
}

@fragment
fn fs_main_shadow(in: VertexOutput) -> @location(0) vec4f {
    let texelSize: vec2f = 1.0 / vec2f(textureDimensions(uTextureSrc));
    let offset: vec2f = settings[2].xy * texelSize;
    let orig: vec4f = textureSample(uTextureSrc, uSamplerSrc, in.texCoord.xy);
    let blur: vec4f = textureSample(uTextureSdw, uSamplerSdw, in.texCoord.xy - offset);
    let shad: vec4f = settings[1] * blur.a;
    return orig + shad * (1.0 - orig.a);
};
)";

const char* cShaderSrc_Effects = R"(
struct VertexInput { @location(0) position: vec2f, @location(1) texCoord: vec2f };
struct VertexOutput { @builtin(position) position: vec4f, @location(0) texCoord: vec2f };

@group(0) @binding(0) var uSamplerSrc : sampler;
@group(0) @binding(1) var uTextureSrc : texture_2d<f32>;
@group(1) @binding(0) var<uniform> settings: array<vec4f, 3>;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4f(in.position.xy, 0.0, 1.0);
    out.texCoord = in.texCoord;
    return out;
}

fn gaussian(x: f32, sigma: f32) -> f32 {
    let exponent: f32 = -x * x / (2.0 * sigma * sigma);
    return exp(exponent) * 0.39894f / sigma;
}

@fragment
fn fs_main_vert(in: VertexOutput) -> @location(0) vec4f {
    let texelSize: vec2f = 1.0 / vec2f(textureDimensions(uTextureSrc));
    var colorSum: vec4f = vec4(0.0);
    let sigma: f32 = settings[0].x * settings[0].y;
    var weightSum: f32 = 0.0;
    let radius: i32 = i32(settings[0].z);

    for (var y: i32 = -radius; y <= radius; y++) {
        let offset: vec2f = vec2f(0.0, f32(y) * texelSize.y);
        let coord = in.texCoord.xy + offset;
        let weight: f32 = select(0.0, gaussian(f32(y), sigma), saturate(coord.y) == coord.y);
        colorSum += textureSample(uTextureSrc, uSamplerSrc, coord) * weight;
        weightSum += weight;
    }

    return colorSum / weightSum;
};

@fragment
fn fs_main_horz(in: VertexOutput) -> @location(0) vec4f {
    let texelSize: vec2f = 1.0 / vec2f(textureDimensions(uTextureSrc));
    var colorSum: vec4f = vec4(0.0);
    let sigma: f32 = settings[0].x * settings[0].y;
    var weightSum: f32 = 0.0;
    let radius: i32 = i32(settings[0].z);

    for (var y: i32 = -radius; y <= radius; y++) {
        let offset: vec2f = vec2f(f32(y) * texelSize.x, 0.0);
        let coord = in.texCoord.xy + offset;
        let weight: f32 = select(0.0, gaussian(f32(y), sigma), saturate(coord.x) == coord.x);
        colorSum += textureSample(uTextureSrc, uSamplerSrc, coord) * weight;
        weightSum += weight;
    }

    return colorSum / weightSum;
};

@fragment
fn fs_main_fill(in: VertexOutput) -> @location(0) vec4f {
    let orig = textureSample(uTextureSrc, uSamplerSrc, in.texCoord.xy);
    let fill = settings[0];
    return fill * orig.a * fill.a;
};

@fragment
fn fs_main_tint(in: VertexOutput) -> @location(0) vec4f {
    let orig = textureSample(uTextureSrc, uSamplerSrc, in.texCoord.xy);
    let luma: f32 = dot(orig.rgb, vec3f(0.2126, 0.7152, 0.0722));
    return vec4f(mix(orig.rgb, mix(settings[0].rgb, settings[1].rgb, luma), settings[2].r) * orig.a, orig.a);
};

@fragment
fn fs_main_tritone(in: VertexOutput) -> @location(0) vec4f {
    let orig = textureSample(uTextureSrc, uSamplerSrc, in.texCoord.xy);
    let luma: f32 = dot(orig.rgb, vec3f(0.2126, 0.7152, 0.0722));
    let isBright: bool = luma >= 0.5f;
    let t = select(luma * 2.0f, (luma - 0.5) * 2.0f, isBright);
    let frm: vec3f = select(settings[0].rgb, settings[1].rgb, isBright);
    let to:  vec3f = select(settings[1].rgb, settings[2].rgb, isBright);
    var tmp: vec4f = vec4f(mix(frm, to, t), 1.0f);

    if (settings[2].a > 0.0f) { tmp = mix(tmp, orig, settings[2].a); }
    return tmp * orig.a;
};
)";
