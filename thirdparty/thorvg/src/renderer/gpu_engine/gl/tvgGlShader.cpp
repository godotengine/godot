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

#include "tvgGlShader.h"

/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/

uint32_t GlShader::compileShader(uint32_t type, char* shaderSrc)
{
    GLuint shader;
    GLint compiled;

    // Create the shader object
    shader = glCreateShader(type);

    /**
     * [0] shader version string
     * [1] precision declaration
     * [2] shader source
     */
    const char* shaderPack[3];
    // but in general All Desktop GPU should use OpenGL version ( #version 330 core )
#if defined (THORVG_GL_TARGET_GLES)
    shaderPack[0] ="#version 300 es\n";
#else
    shaderPack[0] ="#version 330 core\n";
#endif
    shaderPack[1] = "precision highp float;\n precision highp int;\n";
    shaderPack[2] = shaderSrc;

    // Load the shader source
    glShaderSource(shader, 3, shaderPack, NULL);

    // Compile the shader
    glCompileShader(shader);

    // Check the compile status
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);

    if (!compiled)
    {
        GLint infoLen = 0;

        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLen);

        if (infoLen > 0)
        {
            auto infoLog = tvg::malloc<char>(sizeof(char)*infoLen);
            glGetShaderInfoLog(shader, infoLen, NULL, infoLog);
            TVGERR("GL_ENGINE", "Error compiling shader: %s", infoLog);
            tvg::free(infoLog);
        }
        glDeleteShader(shader);
    }

    return shader;
}

/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/

GlShader::GlShader(const char* vertSrc, const char* fragSrc)
{
    mVtShader = compileShader(GL_VERTEX_SHADER, const_cast<char*>(vertSrc));
    mFrShader = compileShader(GL_FRAGMENT_SHADER, const_cast<char*>(fragSrc));
}


GlShader::~GlShader()
{
    glDeleteShader(mVtShader);
    glDeleteShader(mFrShader);
}

uint32_t GlShader::getVertexShader()
{
    return mVtShader;
}


uint32_t GlShader::getFragmentShader()
{
    return mFrShader;
}
