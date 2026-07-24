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

#ifndef _TVG_WG_SHADER_SRC_H_
#define _TVG_WG_SHADER_SRC_H_

// helper shaders
extern const char* cShaderSrc_Stencil;
extern const char* cShaderSrc_Depth;
// shaders normal blend
extern const char* cShaderSrc_Solid;
extern const char* cShaderSrc_Linear;
extern const char* cShaderSrc_Radial;
extern const char* cShaderSrc_Image;
extern const char* cShaderSrc_Scene;
// shaders custrom blend
extern const char* cShaderSrc_Solid_Blend;
extern const char* cShaderSrc_Linear_Blend;
extern const char* cShaderSrc_Radial_Blend;
extern const char* cShaderSrc_Image_Blend;
extern const char* cShaderSrc_Scene_Blend;
extern const char* cShaderSrc_BlendFuncs;
// shaders scene compose
extern const char* cShaderSrc_Scene_Compose;
// shaders blit
extern const char* cShaderSrc_Blit;

// shader sources effects
extern const char* cShaderSrc_Shadow;
extern const char* cShaderSrc_Effects;

#endif // _TVG_WG_SHEDER_SRC_H_
