//
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// VertexAttribute.inc: Inline vertex attribute methods
//

namespace gl
{

inline VertexAttribCurrentValueData::VertexAttribCurrentValueData()
    : Type(gl::VertexAttribType::Float)
{
    Values.FloatValues[0] = 0.0f;
    Values.FloatValues[1] = 0.0f;
    Values.FloatValues[2] = 0.0f;
    Values.FloatValues[3] = 1.0f;
}

inline void VertexAttribCurrentValueData::setFloatValues(const GLfloat floatValues[4])
{
    for (unsigned int valueIndex = 0; valueIndex < 4; valueIndex++)
    {
        Values.FloatValues[valueIndex] = floatValues[valueIndex];
    }
    Type = gl::VertexAttribType::Float;
}

inline void VertexAttribCurrentValueData::setIntValues(const GLint intValues[4])
{
    for (unsigned int valueIndex = 0; valueIndex < 4; valueIndex++)
    {
        Values.IntValues[valueIndex] = intValues[valueIndex];
    }
    Type = gl::VertexAttribType::Int;
}

inline void VertexAttribCurrentValueData::setUnsignedIntValues(const GLuint unsignedIntValues[4])
{
    for (unsigned int valueIndex = 0; valueIndex < 4; valueIndex++)
    {
        Values.UnsignedIntValues[valueIndex] = unsignedIntValues[valueIndex];
    }
    Type = gl::VertexAttribType::UnsignedInt;
}

inline bool operator==(const VertexAttribCurrentValueData &a, const VertexAttribCurrentValueData &b)
{
    return (a.Type == b.Type && memcmp(&a.Values, &b.Values, sizeof(VertexAttribCurrentValueData::Values)) == 0);
}

inline bool operator!=(const VertexAttribCurrentValueData &a, const VertexAttribCurrentValueData &b)
{
    return !(a == b);
}

}  // namespace gl
