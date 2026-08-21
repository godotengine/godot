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

#include "tvgGlProgram.h"

/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/

uint32_t GlProgram::mCurrentProgram = 0;


/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/

GlProgram::GlProgram(const char* vertSrc, const char* fragSrc)
{
    auto shader = GlShader(vertSrc, fragSrc);

    // Create the program object
    uint32_t progObj = glCreateProgram();
    assert(progObj);

    glAttachShader(progObj, shader.getVertexShader());
    glAttachShader(progObj, shader.getFragmentShader());

    // Link the program
    glLinkProgram(progObj);

    // Check the link status
    GLint linked;
    glGetProgramiv(progObj, GL_LINK_STATUS, &linked);

    if (!linked) {
        GLint infoLen = 0;
        glGetProgramiv(progObj, GL_INFO_LOG_LENGTH, &infoLen);
        if (infoLen > 0)
        {
            auto infoLog = tvg::malloc<char>(sizeof(char) * infoLen);
            glGetProgramInfoLog(progObj, infoLen, NULL, infoLog);
            TVGERR("GL_ENGINE", "Error linking shader: %s", infoLog);
            tvg::free(infoLog);

        }
        glDeleteProgram(progObj);
        progObj = 0;
        assert(0);
    }
    mProgramObj = progObj;
}


GlProgram::~GlProgram()
{
    if (mCurrentProgram == mProgramObj) unload();
    glDeleteProgram(mProgramObj);
}


void GlProgram::load()
{
    if (mCurrentProgram == mProgramObj) return;
    mCurrentProgram = mProgramObj;
    GL_CHECK(glUseProgram(mProgramObj));

}


void GlProgram::unload()
{
    mCurrentProgram = 0;
}


int32_t GlProgram::getAttributeLocation(const char* name)
{
    GL_CHECK(int32_t location = glGetAttribLocation(mCurrentProgram, name));
    return location;
}


int32_t GlProgram::getUniformLocation(const char* name)
{
    GL_CHECK(int32_t location = glGetUniformLocation(mProgramObj, name));
    return location;
}

int32_t GlProgram::getUniformBlockIndex(const char* name)
{
    GL_CHECK(int32_t index = glGetUniformBlockIndex(mProgramObj, name));
    return index;
}

uint32_t GlProgram::getProgramId()
{
    return mProgramObj;
}

void GlProgram::setUniform1Value(int32_t location, int count, const int* values)
{
    GL_CHECK(glUniform1iv(location, count, values));
}


void GlProgram::setUniform2Value(int32_t location, int count, const int* values)
{
    GL_CHECK(glUniform2iv(location, count, values));
}


void GlProgram::setUniform3Value(int32_t location, int count, const int* values)
{
    GL_CHECK(glUniform3iv(location, count, values));
}


void GlProgram::setUniform4Value(int32_t location, int count, const int* values)
{
    GL_CHECK(glUniform4iv(location, count, values));
}


void GlProgram::setUniform1Value(int32_t location, int count, const float* values)
{
    GL_CHECK(glUniform1fv(location, count, values));
}


void GlProgram::setUniform2Value(int32_t location, int count, const float* values)
{
    GL_CHECK(glUniform2fv(location, count, values));
}


void GlProgram::setUniform3Value(int32_t location, int count, const float* values)
{
    GL_CHECK(glUniform3fv(location, count, values));
}


void GlProgram::setUniform4Value(int32_t location, int count, const float* values)
{
    GL_CHECK(glUniform4fv(location, count, values));
}
