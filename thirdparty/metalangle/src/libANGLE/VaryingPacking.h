//
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// VaryingPacking:
//   Class which describes a mapping from varyings to registers, according
//   to the spec, or using custom packing algorithms. We also keep a register
//   allocation list for the D3D renderer.
//

#ifndef LIBANGLE_VARYINGPACKING_H_
#define LIBANGLE_VARYINGPACKING_H_

#include <GLSLANG/ShaderVars.h>

#include "angle_gl.h"
#include "common/angleutils.h"

#include <map>

namespace gl
{
class InfoLog;
struct ProgramVaryingRef;

using ProgramMergedVaryings = std::map<std::string, ProgramVaryingRef>;

struct PackedVarying
{
    PackedVarying(const sh::ShaderVariable &varyingIn, sh::InterpolationType interpolationIn)
        : PackedVarying(varyingIn, interpolationIn, "", false)
    {}
    PackedVarying(const sh::ShaderVariable &varyingIn,
                  sh::InterpolationType interpolationIn,
                  const std::string &parentStructNameIn,
                  GLuint fieldIndexIn)
        : varying(&varyingIn),
          vertexOnly(false),
          interpolation(interpolationIn),
          parentStructName(parentStructNameIn),
          arrayIndex(GL_INVALID_INDEX),
          fieldIndex(fieldIndexIn)
    {}

    bool isStructField() const { return !parentStructName.empty(); }

    bool isArrayElement() const { return arrayIndex != GL_INVALID_INDEX; }

    std::string fullName() const
    {
        std::stringstream fullNameStr;
        if (isStructField())
        {
            fullNameStr << parentStructName << ".";
        }

        fullNameStr << varying->name;
        if (arrayIndex != GL_INVALID_INDEX)
        {
            fullNameStr << "[" << arrayIndex << "]";
        }
        return fullNameStr.str();
    }

    const sh::ShaderVariable *varying;

    // Transform feedback varyings can be only referenced in the VS.
    bool vertexOnly;

    // Cached so we can store sh::ShaderVariable to point to varying fields.
    sh::InterpolationType interpolation;

    // Struct name
    std::string parentStructName;

    GLuint arrayIndex;

    // Field index in the struct.  In Vulkan, this is used to assign a
    // struct-typed varying location to the location of its first field.
    GLuint fieldIndex;
};

struct PackedVaryingRegister final
{
    PackedVaryingRegister()
        : packedVarying(nullptr),
          varyingArrayIndex(0),
          varyingRowIndex(0),
          registerRow(0),
          registerColumn(0)
    {}

    PackedVaryingRegister(const PackedVaryingRegister &) = default;
    PackedVaryingRegister &operator=(const PackedVaryingRegister &) = default;

    bool operator<(const PackedVaryingRegister &other) const
    {
        return sortOrder() < other.sortOrder();
    }

    unsigned int sortOrder() const
    {
        // TODO(jmadill): Handle interpolation types
        return registerRow * 4 + registerColumn;
    }

    std::string tfVaryingName() const
    {
        if (packedVarying->isArrayElement() || packedVarying->isStructField())
        {
            return packedVarying->fullName();
        }
        else
        {
            return packedVarying->varying->name;
        }
    }

    // Index to the array of varyings.
    const PackedVarying *packedVarying;

    // The array element of the packed varying.
    unsigned int varyingArrayIndex;

    // The row of the array element of the packed varying.
    unsigned int varyingRowIndex;

    // The register row to which we've assigned this packed varying.
    unsigned int registerRow;

    // The column of the register row into which we've packed this varying.
    unsigned int registerColumn;
};

// Supported packing modes:
enum class PackMode
{
    // We treat mat2 arrays as taking two full rows.
    WEBGL_STRICT,

    // We allow mat2 to take a 2x2 chunk.
    ANGLE_RELAXED,

    // Each varying takes a separate register. No register sharing.
    ANGLE_NON_CONFORMANT_D3D9,
};

class VaryingPacking final : angle::NonCopyable
{
  public:
    VaryingPacking(GLuint maxVaryingVectors, PackMode packMode);
    ~VaryingPacking();

    bool packUserVaryings(gl::InfoLog &infoLog, const std::vector<PackedVarying> &packedVaryings);

    bool collectAndPackUserVaryings(gl::InfoLog &infoLog,
                                    const ProgramMergedVaryings &mergedVaryings,
                                    const std::vector<std::string> &tfVaryings);

    struct Register
    {
        Register() { data[0] = data[1] = data[2] = data[3] = false; }

        bool &operator[](unsigned int index) { return data[index]; }
        bool operator[](unsigned int index) const { return data[index]; }

        bool data[4];
    };

    Register &operator[](unsigned int index) { return mRegisterMap[index]; }
    const Register &operator[](unsigned int index) const { return mRegisterMap[index]; }

    const std::vector<PackedVaryingRegister> &getRegisterList() const { return mRegisterList; }
    unsigned int getMaxSemanticIndex() const
    {
        return static_cast<unsigned int>(mRegisterList.size());
    }

    const std::vector<std::string> &getInactiveVaryingNames() const
    {
        return mInactiveVaryingNames;
    }

  private:
    bool packVarying(const PackedVarying &packedVarying);
    bool isFree(unsigned int registerRow,
                unsigned int registerColumn,
                unsigned int varyingRows,
                unsigned int varyingColumns) const;
    void insert(unsigned int registerRow,
                unsigned int registerColumn,
                const PackedVarying &packedVarying);

    std::vector<Register> mRegisterMap;
    std::vector<PackedVaryingRegister> mRegisterList;
    std::vector<PackedVarying> mPackedVaryings;
    std::vector<std::string> mInactiveVaryingNames;

    PackMode mPackMode;
};

}  // namespace gl

#endif  // LIBANGLE_VARYINGPACKING_H_
