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

#include "libANGLE/VaryingPacking.h"

#include "common/utilities.h"
#include "libANGLE/Program.h"
#include "libANGLE/Shader.h"

namespace gl
{

namespace
{

// true if varying x has a higher priority in packing than y
bool ComparePackedVarying(const PackedVarying &x, const PackedVarying &y)
{
    // If the PackedVarying 'x' or 'y' to be compared is an array element, this clones an equivalent
    // non-array shader variable 'vx' or 'vy' for actual comparison instead.
    sh::ShaderVariable vx, vy;
    const sh::ShaderVariable *px, *py;
    if (x.isArrayElement())
    {
        vx = *x.varying;
        vx.arraySizes.clear();
        px = &vx;
    }
    else
    {
        px = x.varying;
    }

    if (y.isArrayElement())
    {
        vy = *y.varying;
        vy.arraySizes.clear();
        py = &vy;
    }
    else
    {
        py = y.varying;
    }

    return gl::CompareShaderVar(*px, *py);
}

}  // anonymous namespace

// Implementation of VaryingPacking
VaryingPacking::VaryingPacking(GLuint maxVaryingVectors, PackMode packMode)
    : mRegisterMap(maxVaryingVectors), mPackMode(packMode)
{}

VaryingPacking::~VaryingPacking() = default;

// Packs varyings into generic varying registers, using the algorithm from
// See [OpenGL ES Shading Language 1.00 rev. 17] appendix A section 7 page 111
// Also [OpenGL ES Shading Language 3.00 rev. 4] Section 11 page 119
// Returns false if unsuccessful.
bool VaryingPacking::packVarying(const PackedVarying &packedVarying)
{
    const sh::ShaderVariable &varying = *packedVarying.varying;

    // "Non - square matrices of type matCxR consume the same space as a square matrix of type matN
    // where N is the greater of C and R."
    // Here we are a bit more conservative and allow packing non-square matrices more tightly.
    // Make sure we use transposed matrix types to count registers correctly.
    ASSERT(!varying.isStruct());
    GLenum transposedType       = gl::TransposeMatrixType(varying.type);
    unsigned int varyingRows    = gl::VariableRowCount(transposedType);
    unsigned int varyingColumns = gl::VariableColumnCount(transposedType);

    // Special pack mode for D3D9. Each varying takes a full register, no sharing.
    // TODO(jmadill): Implement more sophisticated component packing in D3D9.
    if (mPackMode == PackMode::ANGLE_NON_CONFORMANT_D3D9)
    {
        varyingColumns = 4;
    }

    // "Variables of type mat2 occupies 2 complete rows."
    // For non-WebGL contexts, we allow mat2 to occupy only two columns per row.
    else if (mPackMode == PackMode::WEBGL_STRICT && varying.type == GL_FLOAT_MAT2)
    {
        varyingColumns = 4;
    }

    // "Arrays of size N are assumed to take N times the size of the base type"
    // GLSL ES 3.10 section 4.3.6: Output variables cannot be arrays of arrays or arrays of
    // structures, so we may use getBasicTypeElementCount().
    const unsigned int elementCount = varying.getBasicTypeElementCount();
    varyingRows *= (packedVarying.isArrayElement() ? 1 : elementCount);

    unsigned int maxVaryingVectors = static_cast<unsigned int>(mRegisterMap.size());

    // Fail if we are packing a single over-large varying.
    if (varyingRows > maxVaryingVectors)
    {
        return false;
    }

    // "For 2, 3 and 4 component variables packing is started using the 1st column of the 1st row.
    // Variables are then allocated to successive rows, aligning them to the 1st column."
    if (varyingColumns >= 2 && varyingColumns <= 4)
    {
        for (unsigned int row = 0; row <= maxVaryingVectors - varyingRows; ++row)
        {
            if (isFree(row, 0, varyingRows, varyingColumns))
            {
                insert(row, 0, packedVarying);
                return true;
            }
        }

        // "For 2 component variables, when there are no spare rows, the strategy is switched to
        // using the highest numbered row and the lowest numbered column where the variable will
        // fit."
        if (varyingColumns == 2)
        {
            for (unsigned int r = maxVaryingVectors - varyingRows + 1; r-- >= 1;)
            {
                if (isFree(r, 2, varyingRows, 2))
                {
                    insert(r, 2, packedVarying);
                    return true;
                }
            }
        }

        return false;
    }

    // "1 component variables have their own packing rule. They are packed in order of size, largest
    // first. Each variable is placed in the column that leaves the least amount of space in the
    // column and aligned to the lowest available rows within that column."
    ASSERT(varyingColumns == 1);
    unsigned int contiguousSpace[4]     = {0};
    unsigned int bestContiguousSpace[4] = {0};
    unsigned int totalSpace[4]          = {0};

    for (unsigned int row = 0; row < maxVaryingVectors; ++row)
    {
        for (unsigned int column = 0; column < 4; ++column)
        {
            if (mRegisterMap[row][column])
            {
                contiguousSpace[column] = 0;
            }
            else
            {
                contiguousSpace[column]++;
                totalSpace[column]++;

                if (contiguousSpace[column] > bestContiguousSpace[column])
                {
                    bestContiguousSpace[column] = contiguousSpace[column];
                }
            }
        }
    }

    unsigned int bestColumn = 0;
    for (unsigned int column = 1; column < 4; ++column)
    {
        if (bestContiguousSpace[column] >= varyingRows &&
            (bestContiguousSpace[bestColumn] < varyingRows ||
             totalSpace[column] < totalSpace[bestColumn]))
        {
            bestColumn = column;
        }
    }

    if (bestContiguousSpace[bestColumn] >= varyingRows)
    {
        for (unsigned int row = 0; row < maxVaryingVectors; row++)
        {
            if (isFree(row, bestColumn, varyingRows, 1))
            {
                for (unsigned int arrayIndex = 0; arrayIndex < varyingRows; ++arrayIndex)
                {
                    // If varyingRows > 1, it must be an array.
                    PackedVaryingRegister registerInfo;
                    registerInfo.packedVarying  = &packedVarying;
                    registerInfo.registerRow    = row + arrayIndex;
                    registerInfo.registerColumn = bestColumn;
                    registerInfo.varyingArrayIndex =
                        (packedVarying.isArrayElement() ? packedVarying.arrayIndex : arrayIndex);
                    registerInfo.varyingRowIndex = 0;
                    // Do not record register info for builtins.
                    // TODO(jmadill): Clean this up.
                    if (!packedVarying.varying->isBuiltIn())
                    {
                        mRegisterList.push_back(registerInfo);
                    }
                    mRegisterMap[row + arrayIndex][bestColumn] = true;
                }
                break;
            }
        }
        return true;
    }

    return false;
}

bool VaryingPacking::isFree(unsigned int registerRow,
                            unsigned int registerColumn,
                            unsigned int varyingRows,
                            unsigned int varyingColumns) const
{
    for (unsigned int row = 0; row < varyingRows; ++row)
    {
        ASSERT(registerRow + row < mRegisterMap.size());
        for (unsigned int column = 0; column < varyingColumns; ++column)
        {
            ASSERT(registerColumn + column < 4);
            if (mRegisterMap[registerRow + row][registerColumn + column])
            {
                return false;
            }
        }
    }

    return true;
}

void VaryingPacking::insert(unsigned int registerRow,
                            unsigned int registerColumn,
                            const PackedVarying &packedVarying)
{
    unsigned int varyingRows    = 0;
    unsigned int varyingColumns = 0;

    const auto &varying = *packedVarying.varying;
    ASSERT(!varying.isStruct());
    GLenum transposedType = gl::TransposeMatrixType(varying.type);
    varyingRows           = gl::VariableRowCount(transposedType);
    varyingColumns        = gl::VariableColumnCount(transposedType);

    PackedVaryingRegister registerInfo;
    registerInfo.packedVarying  = &packedVarying;
    registerInfo.registerColumn = registerColumn;

    // GLSL ES 3.10 section 4.3.6: Output variables cannot be arrays of arrays or arrays of
    // structures, so we may use getBasicTypeElementCount().
    const unsigned int arrayElementCount = varying.getBasicTypeElementCount();
    for (unsigned int arrayElement = 0; arrayElement < arrayElementCount; ++arrayElement)
    {
        if (packedVarying.isArrayElement() && arrayElement != packedVarying.arrayIndex)
        {
            continue;
        }
        for (unsigned int varyingRow = 0; varyingRow < varyingRows; ++varyingRow)
        {
            registerInfo.registerRow     = registerRow + (arrayElement * varyingRows) + varyingRow;
            registerInfo.varyingRowIndex = varyingRow;
            registerInfo.varyingArrayIndex = arrayElement;
            // Do not record register info for builtins.
            // TODO(jmadill): Clean this up.
            if (!packedVarying.varying->isBuiltIn())
            {
                mRegisterList.push_back(registerInfo);
            }

            for (unsigned int columnIndex = 0; columnIndex < varyingColumns; ++columnIndex)
            {
                mRegisterMap[registerInfo.registerRow][registerColumn + columnIndex] = true;
            }
        }
    }
}

bool VaryingPacking::collectAndPackUserVaryings(gl::InfoLog &infoLog,
                                                const ProgramMergedVaryings &mergedVaryings,
                                                const std::vector<std::string> &tfVaryings)
{
    std::set<std::string> uniqueFullNames;
    mPackedVaryings.clear();

    for (const auto &ref : mergedVaryings)
    {
        const sh::ShaderVariable *input  = ref.second.frontShader;
        const sh::ShaderVariable *output = ref.second.backShader;

        // Only pack statically used varyings that have a matched input or output, plus special
        // builtins. Note that we pack all statically used user-defined varyings even if they are
        // not active. GLES specs are a bit vague on whether it's allowed to only pack active
        // varyings, though GLES 3.1 spec section 11.1.2.1 says that "device-dependent
        // optimizations" may be used to make vertex shader outputs fit.
        if ((input && output && output->staticUse) ||
            (input && input->isBuiltIn() && input->active) ||
            (output && output->isBuiltIn() && output->active))
        {
            const sh::ShaderVariable *varying = output ? output : input;

            // Don't count gl_Position. Also don't count gl_PointSize for D3D9.
            if (varying->name != "gl_Position" &&
                !(varying->name == "gl_PointSize" &&
                  mPackMode == PackMode::ANGLE_NON_CONFORMANT_D3D9))
            {
                // Will get the vertex shader interpolation by default.
                auto interpolation = ref.second.get()->interpolation;

                // Note that we lose the vertex shader static use information here. The data for the
                // variable is taken from the fragment shader.
                if (varying->isStruct())
                {
                    ASSERT(!(varying->isArray() && varying == input));
                    for (GLuint fieldIndex = 0; fieldIndex < varying->fields.size(); ++fieldIndex)
                    {
                        const sh::ShaderVariable &field = varying->fields[fieldIndex];

                        ASSERT(!field.isStruct() && !field.isArray());
                        mPackedVaryings.emplace_back(field, interpolation, varying->name,
                                                     fieldIndex);
                        uniqueFullNames.insert(mPackedVaryings.back().fullName());
                    }
                }
                else
                {
                    mPackedVaryings.emplace_back(*varying, interpolation);
                    uniqueFullNames.insert(mPackedVaryings.back().fullName());
                }
                continue;
            }
        }

        // If the varying is not used in the VS, we know it is inactive.
        if (!input)
        {
            mInactiveVaryingNames.push_back(ref.first);
            continue;
        }

        // Keep Transform FB varyings in the merged list always.
        for (const std::string &tfVarying : tfVaryings)
        {
            std::vector<unsigned int> subscripts;
            std::string baseName = ParseResourceName(tfVarying, &subscripts);
            size_t subscript     = GL_INVALID_INDEX;
            if (!subscripts.empty())
            {
                subscript = subscripts.back();
            }
            // Already packed for fragment shader.
            if (uniqueFullNames.count(tfVarying) > 0 || uniqueFullNames.count(baseName) > 0)
            {
                continue;
            }
            if (input->isStruct())
            {
                GLuint fieldIndex = 0;
                const sh::ShaderVariable *field =
                    FindShaderVarField(*input, tfVarying, &fieldIndex);
                if (field != nullptr)
                {
                    ASSERT(!field->isStruct() && !field->isArray());
                    mPackedVaryings.emplace_back(*field, input->interpolation, input->name,
                                                 fieldIndex);
                    mPackedVaryings.back().vertexOnly = true;
                    mPackedVaryings.back().arrayIndex = GL_INVALID_INDEX;
                    uniqueFullNames.insert(tfVarying);
                }
            }
            // Array as a whole and array element conflict has already been checked in
            // linkValidateTransformFeedback.
            else if (baseName == input->name)
            {
                // only pack varyings that are not builtins.
                if (tfVarying.compare(0, 3, "gl_") != 0)
                {
                    mPackedVaryings.emplace_back(*input, input->interpolation);
                    mPackedVaryings.back().vertexOnly = true;
                    mPackedVaryings.back().arrayIndex = static_cast<GLuint>(subscript);
                    uniqueFullNames.insert(tfVarying);
                }
                // Continue to match next array element for 'input' if the current match is array
                // element.
                if (subscript == GL_INVALID_INDEX)
                {
                    break;
                }
            }
        }

        if (uniqueFullNames.count(ref.first) == 0)
        {
            mInactiveVaryingNames.push_back(ref.first);
        }
    }

    std::sort(mPackedVaryings.begin(), mPackedVaryings.end(), ComparePackedVarying);

    return packUserVaryings(infoLog, mPackedVaryings);
}

// See comment on packVarying.
bool VaryingPacking::packUserVaryings(gl::InfoLog &infoLog,
                                      const std::vector<PackedVarying> &packedVaryings)
{
    // "Variables are packed into the registers one at a time so that they each occupy a contiguous
    // subrectangle. No splitting of variables is permitted."
    for (const PackedVarying &packedVarying : packedVaryings)
    {
        if (!packVarying(packedVarying))
        {
            infoLog << "Could not pack varying " << packedVarying.fullName();

            // TODO(jmadill): Implement more sophisticated component packing in D3D9.
            if (mPackMode == PackMode::ANGLE_NON_CONFORMANT_D3D9)
            {
                infoLog << "Note: Additional non-conformant packing restrictions are enforced on "
                           "D3D9.";
            }

            return false;
        }
    }

    // Sort the packed register list
    std::sort(mRegisterList.begin(), mRegisterList.end());

    return true;
}
}  // namespace gl
