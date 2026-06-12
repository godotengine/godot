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

#ifndef _TVG_GL_PROGRAM_H_
#define _TVG_GL_PROGRAM_H_

#include "tvgGlShader.h"

class GlProgram
{
public:
    GlProgram(const char* vertSrc, const char* fragSrc);
    ~GlProgram();

    void load();
    static void unload();
    int32_t getAttributeLocation(const char* name);
    int32_t getUniformLocation(const char* name);
    int32_t getUniformBlockIndex(const char* name);
    uint32_t getProgramId();
    void setUniform1Value(int32_t location, int count, const int* values);
    void setUniform2Value(int32_t location, int count, const int* values);
    void setUniform3Value(int32_t location, int count, const int* values);
    void setUniform4Value(int32_t location, int count, const int* values);
    void setUniform1Value(int32_t location, int count, const float* values);
    void setUniform2Value(int32_t location, int count, const float* values);
    void setUniform3Value(int32_t location, int count, const float* values);
    void setUniform4Value(int32_t location, int count, const float* values);

private:
    uint32_t mProgramObj;
    static uint32_t mCurrentProgram;
};

#endif /* _TVG_GL_PROGRAM_H_ */
