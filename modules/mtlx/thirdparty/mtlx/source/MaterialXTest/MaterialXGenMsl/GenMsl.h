//
// Copyright (c) 2023 Apple Inc.
// Licensed under the Apache License v2.0
//

#ifndef GENGLSL_H
#define GENGLSL_H

#include <MaterialXTest/MaterialXGenShader/GenShaderUtil.h>

#include <MaterialXFormat/File.h>
#include <MaterialXFormat/Util.h>

#include "CompileMsl.h"

#include <cassert>

namespace mx = MaterialX;

class MslShaderGeneratorTester : public GenShaderUtil::ShaderGeneratorTester
{
  public:
    using ParentClass = GenShaderUtil::ShaderGeneratorTester;

    MslShaderGeneratorTester(mx::ShaderGeneratorPtr shaderGenerator, const mx::FilePathVec& testRootPaths, 
                             const mx::FileSearchPath& searchPath, const mx::FilePath& logFilePath, bool writeShadersToDisk) :
        GenShaderUtil::ShaderGeneratorTester(shaderGenerator, testRootPaths, searchPath, logFilePath, writeShadersToDisk)
    {}

    void setTestStages() override
    {
        _testStages.push_back(mx::Stage::VERTEX);
        _testStages.push_back(mx::Stage::PIXEL);
    }

    // Ignore trying to create shader code for displacementshaders
    void addSkipNodeDefs() override
    {
        _skipNodeDefs.insert("ND_displacement_float");
        _skipNodeDefs.insert("ND_displacement_vector3");
        ParentClass::addSkipNodeDefs();
    }

    void setupDependentLibraries() override
    {
        ParentClass::setupDependentLibraries();

        mx::FilePath lightDir = mx::getDefaultDataSearchPath().find("resources/Materials/TestSuite/lights");
        loadLibrary(lightDir / mx::FilePath("light_compound_test.mtlx"), _dependLib);
        loadLibrary(lightDir / mx::FilePath("light_rig_test_1.mtlx"), _dependLib);
    }

    void compileSource(const std::vector<mx::FilePath>& sourceCodePaths) override
    {
        int i = 0;
        for(const mx::FilePath& sourceCodePath : sourceCodePaths)
        {
            assert(i == 0 || i == 1);
            CompileMslShader(sourceCodePath.asString().c_str(), i == 0 ?  "VertexMain" : "FragmentMain");
            ++i;
        }
    }
    
  protected:
    void getImplementationWhiteList(mx::StringSet& whiteList) override
    {
        whiteList =
        {
            "ambientocclusion", "arrayappend", "backfacing", "screen", "curveadjust", "displacementshader",
            "volumeshader", "IM_constant_", "IM_dot_", "IM_geompropvalue_boolean", "IM_geompropvalue_string",
            "IM_light_genmsl", "IM_point_light_genmsl", "IM_spot_light_genmsl", "IM_directional_light_genmsl",
            "IM_angle", "surfacematerial", "volumematerial", "ND_surfacematerial", "ND_volumematerial", "ND_backface_util", "IM_backface_util_genmsl"
        };
    }
};

#endif // GENGLSL_H
