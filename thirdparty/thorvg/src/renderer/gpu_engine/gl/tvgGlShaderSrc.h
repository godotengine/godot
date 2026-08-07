/*
 * Copyright (c) 2020 - 2026 ThorVG project. All rights reserved.

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

#ifndef _TVG_GL_SHADERSRC_H_
#define _TVG_GL_SHADERSRC_H_

extern const char* COLOR_VERT_SHADER;
extern const char* COLOR_FRAG_SHADER;
extern const char* GRADIENT_VERT_SHADER;
extern const char* STR_GRADIENT_FRAG_COMMON_VARIABLES;
extern const char* STR_GRADIENT_FRAG_COMMON_FUNCTIONS;
extern const char* STR_LINEAR_GRADIENT_VARIABLES;
extern const char* STR_LINEAR_GRADIENT_FUNCTIONS;
extern const char* STR_LINEAR_GRADIENT_MAIN;
extern const char* STR_RADIAL_GRADIENT_VARIABLES;
extern const char* STR_RADIAL_GRADIENT_FUNCTIONS;
extern const char* STR_RADIAL_GRADIENT_MAIN;
extern const char* IMAGE_VERT_SHADER;
extern const char* IMAGE_FRAG_SHADER;
extern const char* MASK_VERT_SHADER;
extern const char* MASK_ALPHA_FRAG_SHADER;
extern const char* MASK_INV_ALPHA_FRAG_SHADER;
extern const char* MASK_LUMA_FRAG_SHADER;
extern const char* MASK_INV_LUMA_FRAG_SHADER;
extern const char* MASK_ADD_FRAG_SHADER;
extern const char* MASK_SUB_FRAG_SHADER;
extern const char* MASK_INTERSECT_FRAG_SHADER;
extern const char* MASK_DIFF_FRAG_SHADER;
extern const char* MASK_DARKEN_FRAG_SHADER;
extern const char* MASK_LIGHTEN_FRAG_SHADER;
extern const char* STENCIL_VERT_SHADER;
extern const char* STENCIL_FRAG_SHADER;
extern const char* BLIT_VERT_SHADER;
extern const char* BLIT_FRAG_SHADER;

extern const char* BLEND_IMAGE_FRAG_HEADER;
extern const char* BLEND_SCENE_FRAG_HEADER;
extern const char* BLEND_SHAPE_SOLID_FRAG_HEADER;
extern const char* BLEND_SHAPE_LINEAR_FRAG_HEADER;
extern const char* BLEND_SHAPE_RADIAL_FRAG_HEADER;

extern const char* BLEND_FRAG_LUM_HELPER;
extern const char* BLEND_FRAG_SAT_HELPER;

extern const char* NORMAL_BLEND_FRAG;
extern const char* MULTIPLY_BLEND_FRAG;
extern const char* SCREEN_BLEND_FRAG;
extern const char* OVERLAY_BLEND_FRAG;
extern const char* DARKEN_BLEND_FRAG;
extern const char* LIGHTEN_BLEND_FRAG;
extern const char* COLOR_DODGE_BLEND_FRAG;
extern const char* COLOR_BURN_BLEND_FRAG;
extern const char* HARD_LIGHT_BLEND_FRAG;
extern const char* SOFT_LIGHT_BLEND_FRAG;
extern const char* DIFFERENCE_BLEND_FRAG;
extern const char* EXCLUSION_BLEND_FRAG;
extern const char* HUE_BLEND_FRAG;
extern const char* SATURATION_BLEND_FRAG;
extern const char* COLOR_BLEND_FRAG;
extern const char* LUMINOSITY_BLEND_FRAG;
extern const char* ADD_BLEND_FRAG;

extern const char* EFFECT_VERTEX;
extern const char* GAUSSIAN_VERTICAL;
extern const char* GAUSSIAN_HORIZONTAL;
extern const char* EFFECT_DROPSHADOW;
extern const char* EFFECT_FILL;
extern const char* EFFECT_TINT;
extern const char* EFFECT_TRITONE;

#endif /* _TVG_GL_SHADERSRC_H_ */
