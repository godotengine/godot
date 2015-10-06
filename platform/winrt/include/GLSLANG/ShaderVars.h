//
// Copyright (c) 2013-2014 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// ShaderVars.h:
//  Types to represent GL variables (varyings, uniforms, etc)
//

#ifndef _COMPILER_INTERFACE_VARIABLES_
#define _COMPILER_INTERFACE_VARIABLES_

#include <string>
#include <vector>
#include <algorithm>

// Assume ShaderLang.h is included before ShaderVars.h, for sh::GLenum
// Note: make sure to increment ANGLE_SH_VERSION when changing ShaderVars.h

namespace sh
{

// Varying interpolation qualifier, see section 4.3.9 of the ESSL 3.00.4 spec
enum InterpolationType
{
    INTERPOLATION_SMOOTH,
    INTERPOLATION_CENTROID,
    INTERPOLATION_FLAT
};

// Uniform block layout qualifier, see section 4.3.8.3 of the ESSL 3.00.4 spec
enum BlockLayoutType
{
    BLOCKLAYOUT_STANDARD,
    BLOCKLAYOUT_PACKED,
    BLOCKLAYOUT_SHARED
};

// Base class for all variables defined in shaders, including Varyings, Uniforms, etc
// Note: we must override the copy constructor and assignment operator so we can
// work around excessive GCC binary bloating:
// See https://code.google.com/p/angleproject/issues/detail?id=697
struct COMPILER_EXPORT ShaderVariable
{
    ShaderVariable();
    ShaderVariable(GLenum typeIn, unsigned int arraySizeIn);
    ~ShaderVariable();
    ShaderVariable(const ShaderVariable &other);
    ShaderVariable &operator=(const ShaderVariable &other);

    bool isArray() const { return arraySize > 0; }
    unsigned int elementCount() const { return std::max(1u, arraySize); }
    bool isStruct() const { return !fields.empty(); }

    // All of the shader's variables are described using nested data
    // structures. This is needed in order to disambiguate similar looking
    // types, such as two structs containing the same fields, but in
    // different orders. "findInfoByMappedName" provides an easy query for
    // users to dive into the data structure and fetch the unique variable
    // instance corresponding to a dereferencing chain of the top-level
    // variable.
    // Given a mapped name like 'a[0].b.c[0]', return the ShaderVariable
    // that defines 'c' in |leafVar|, and the original name 'A[0].B.C[0]'
    // in |originalName|, based on the assumption that |this| defines 'a'.
    // If no match is found, return false.
    bool findInfoByMappedName(const std::string &mappedFullName,
                              const ShaderVariable **leafVar,
                              std::string* originalFullName) const;

    GLenum type;
    GLenum precision;
    std::string name;
    std::string mappedName;
    unsigned int arraySize;
    bool staticUse;
    std::vector<ShaderVariable> fields;
    std::string structName;

  protected:
    bool isSameVariableAtLinkTime(const ShaderVariable &other,
                                  bool matchPrecision) const;

    bool operator==(const ShaderVariable &other) const;
    bool operator!=(const ShaderVariable &other) const
    {
        return !operator==(other);
    }
};

struct COMPILER_EXPORT Uniform : public ShaderVariable
{
    Uniform();
    ~Uniform();
    Uniform(const Uniform &other);
    Uniform &operator=(const Uniform &other);
    bool operator==(const Uniform &other) const;
    bool operator!=(const Uniform &other) const
    {
        return !operator==(other);
    }

    // Decide whether two uniforms are the same at shader link time,
    // assuming one from vertex shader and the other from fragment shader.
    // See GLSL ES Spec 3.00.3, sec 4.3.5.
    bool isSameUniformAtLinkTime(const Uniform &other) const;
};

struct COMPILER_EXPORT Attribute : public ShaderVariable
{
    Attribute();
    ~Attribute();
    Attribute(const Attribute &other);
    Attribute &operator=(const Attribute &other);
    bool operator==(const Attribute &other) const;
    bool operator!=(const Attribute &other) const
    {
        return !operator==(other);
    }

    int location;
};

struct COMPILER_EXPORT InterfaceBlockField : public ShaderVariable
{
    InterfaceBlockField();
    ~InterfaceBlockField();
    InterfaceBlockField(const InterfaceBlockField &other);
    InterfaceBlockField &operator=(const InterfaceBlockField &other);
    bool operator==(const InterfaceBlockField &other) const;
    bool operator!=(const InterfaceBlockField &other) const
    {
        return !operator==(other);
    }

    // Decide whether two InterfaceBlock fields are the same at shader
    // link time, assuming one from vertex shader and the other from
    // fragment shader.
    // See GLSL ES Spec 3.00.3, sec 4.3.7.
    bool isSameInterfaceBlockFieldAtLinkTime(
        const InterfaceBlockField &other) const;

    bool isRowMajorLayout;
};

struct COMPILER_EXPORT Varying : public ShaderVariable
{
    Varying();
    ~Varying();
    Varying(const Varying &otherg);
    Varying &operator=(const Varying &other);
    bool operator==(const Varying &other) const;
    bool operator!=(const Varying &other) const
    {
        return !operator==(other);
    }

    // Decide whether two varyings are the same at shader link time,
    // assuming one from vertex shader and the other from fragment shader.
    // See GLSL ES Spec 3.00.3, sec 4.3.9.
    bool isSameVaryingAtLinkTime(const Varying &other) const;

    InterpolationType interpolation;
    bool isInvariant;
};

struct COMPILER_EXPORT InterfaceBlock
{
    InterfaceBlock();
    ~InterfaceBlock();
    InterfaceBlock(const InterfaceBlock &other);
    InterfaceBlock &operator=(const InterfaceBlock &other);

    std::string name;
    std::string mappedName;
    std::string instanceName;
    unsigned int arraySize;
    BlockLayoutType layout;
    bool isRowMajorLayout;
    bool staticUse;
    std::vector<InterfaceBlockField> fields;
};

}

#endif // _COMPILER_INTERFACE_VARIABLES_
