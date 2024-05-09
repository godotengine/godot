//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXView/Viewer.h>

#include <MaterialXRender/Util.h>
#include <MaterialXFormat/Util.h>
#include <MaterialXCore/Util.h>

#include <iostream>

NANOGUI_FORCE_DISCRETE_GPU();

const std::string options =
    " Options: \n"
    "    --material [FILENAME]          Specify the filename of the MTLX document to be displayed in the viewer\n"
    "    --mesh [FILENAME]              Specify the filename of the OBJ mesh to be displayed in the viewer\n"
    "    --meshRotation [VECTOR3]       Specify the rotation of the displayed mesh as three comma-separated floats, representing rotations in degrees about the X, Y, and Z axes (defaults to 0,0,0)\n"
    "    --meshScale [FLOAT]            Specify the uniform scale of the displayed mesh\n"
    "    --enableTurntable[BOOLEAN]     Specify whether to enable turntable rendering of the scene\n"
    "    --turntableSteps [INTEGER]     Specify the number of steps for a complete turntable rotation. Defaults to 360\n"
    "    --cameraPosition [VECTOR3]     Specify the position of the camera as three comma-separated floats (defaults to 0,0,5)\n"
    "    --cameraTarget [VECTOR3]       Specify the position of the camera target as three comma-separated floats (defaults to 0,0,0)\n"
    "    --cameraViewAngle [FLOAT]      Specify the view angle of the camera, or zero for an orthographic projection (defaults to 45)\n"
    "    --cameraZoom [FLOAT]           Specify the zoom factor for the camera, implemented as a mesh scale multiplier (defaults to 1)\n"
    "    --envRad [FILENAME]            Specify the filename of the environment light to display, stored as HDR environment radiance in the latitude-longitude format\n"
    "    --envMethod [INTEGER]          Specify the environment lighting method (0 = filtered importance sampling, 1 = prefiltered environment maps, defaults to 0)\n"
    "    --envSampleCount [INTEGER]     Specify the environment sample count (defaults to 16)\n"
    "    --envLightIntensity [FLOAT]    Specify the environment light intensity (defaults to 1)\n"
    "    --lightRotation [FLOAT]        Specify the rotation in degrees of the lighting environment about the Y axis (defaults to 0)\n"
    "    --shadowMap [BOOLEAN]          Specify whether shadow mapping is enabled (defaults to true)\n"
    "    --path [FILEPATH]              Specify an additional data search path location (e.g. '/projects/MaterialX').  This absolute path will be queried when locating data libraries, XInclude references, and referenced images.\n"
    "    --library [FILEPATH]           Specify an additional data library folder (e.g. 'vendorlib', 'studiolib').  This relative path will be appended to each location in the data search path when loading data libraries.\n"
    "    --screenWidth [INTEGER]        Specify the width of the screen image in pixels (defaults to 1280)\n"
    "    --screenHeight [INTEGER]       Specify the height of the screen image in pixels (defaults to 960)\n"
    "    --screenColor [VECTOR3]        Specify the background color of the viewer as three comma-separated floats (defaults to 0.3,0.3,0.32)\n"
    "    --drawEnvironment [BOOLEAN]    Specify whether to render the environment as the background (defaults to false)\n"
    "    --captureFilename [FILENAME]   Specify the filename to which the first rendered frame should be written\n"
    "    --bakeWidth [INTEGER]          Specify the target width for texture baking (defaults to maximum image width of the source document)\n"
    "    --bakeHeight [INTEGER]         Specify the target height for texture baking (defaults to maximum image height of the source document)\n"
    "    --bakeFilename [STRING]        Specify the output document filename for texture baking\n"
    "    --refresh [FLOAT]              Specify the refresh period for the viewer in milliseconds (defaults to 50, set to -1 to disable)\n"
    "    --remap [TOKEN1:TOKEN2]        Specify the remapping from one token to another when MaterialX document is loaded\n"
    "    --skip [NAME]                  Specify to skip elements matching the given name attribute\n"
    "    --terminator [STRING]          Specify to enforce the given terminator string for file prefixes\n"
    "    --help                         Display the complete list of command-line options\n";

template <class T> void parseToken(std::string token, std::string type, T& res)
{
    if (token.empty())
    {
        return;
    }

    mx::ValuePtr value = mx::Value::createValueFromStrings(token, type);
    if (!value)
    {
        std::cout << "Unable to parse token " << token << " as type " << type << std::endl;
        return;
    }

    res = value->asA<T>();
}

int main(int argc, char* const argv[])
{
    std::vector<std::string> tokens;
    for (int i = 1; i < argc; i++)
    {
        tokens.emplace_back(argv[i]);
    }

    std::string materialFilename = "resources/Materials/Examples/StandardSurface/standard_surface_default.mtlx";
    std::string meshFilename = "resources/Geometry/shaderball.glb";
    std::string envRadianceFilename = "resources/Lights/san_giuseppe_bridge_split.hdr";
    mx::FileSearchPath searchPath = mx::getDefaultDataSearchPath();
    mx::FilePathVec libraryFolders;

    mx::Vector3 meshRotation;
    float meshScale = 1.0f;
    bool turntableEnabled = false;
    int turntableSteps = 360;
    mx::Vector3 cameraPosition(DEFAULT_CAMERA_POSITION);
    mx::Vector3 cameraTarget;
    float cameraViewAngle(DEFAULT_CAMERA_VIEW_ANGLE);
    float cameraZoom(DEFAULT_CAMERA_ZOOM);
    mx::HwSpecularEnvironmentMethod specularEnvironmentMethod = mx::SPECULAR_ENVIRONMENT_FIS;
    int envSampleCount = mx::DEFAULT_ENV_SAMPLE_COUNT;
    float envLightIntensity = 1.0f;
    float lightRotation = 0.0f;
    bool shadowMap = true;
    DocumentModifiers modifiers;
    int screenWidth = 1280;
    int screenHeight = 960;
    mx::Color3 screenColor(mx::DEFAULT_SCREEN_COLOR_SRGB);
    bool drawEnvironment = false;
    std::string captureFilename;
    int bakeWidth = 0;
    int bakeHeight = 0;
    std::string bakeFilename;
    float refresh = 50.0f;

    for (size_t i = 0; i < tokens.size(); i++)
    {
        const std::string& token = tokens[i];
        const std::string& nextToken = i + 1 < tokens.size() ? tokens[i + 1] : mx::EMPTY_STRING;
        if (token == "--material")
        {
            materialFilename = nextToken;
        }
        else if (token == "--mesh")
        {
            meshFilename = nextToken;
        }
        else if (token == "--envRad")
        {
            envRadianceFilename = nextToken;
        }
        else if (token == "--meshRotation")
        {
            parseToken(nextToken, "vector3", meshRotation);
        }
        else if (token == "--meshScale")
        {
            parseToken(nextToken, "float", meshScale);
        }
        else if (token == "--enableTurntable")
        {
            parseToken(nextToken, "boolean", turntableEnabled);
        }
        else if (token == "--turntableSteps")
        {
            parseToken(nextToken, "integer", turntableSteps);
            turntableSteps = std::clamp(turntableSteps, 2, 360);
        }
        else if (token == "--cameraPosition")
        {
            parseToken(nextToken, "vector3", cameraPosition);
        }
        else if (token == "--cameraTarget")
        {
            parseToken(nextToken, "vector3", cameraTarget);
        }
        else if (token == "--cameraViewAngle")
        {
            parseToken(nextToken, "float", cameraViewAngle);
        }
        else if (token == "--cameraZoom")
        {
            parseToken(nextToken, "float", cameraZoom);
        }
        else if (token == "--envMethod")
        {
            if (std::stoi(nextToken) == 1)
            {
                specularEnvironmentMethod = mx::SPECULAR_ENVIRONMENT_PREFILTER;
            }
        }
        else if (token == "--envSampleCount")
        {
            parseToken(nextToken, "integer", envSampleCount);
        }
        else if (token == "--envLightIntensity")
        {
            parseToken(nextToken, "float", envLightIntensity);
        }
        else if (token == "--lightRotation")
        {
            parseToken(nextToken, "float", lightRotation);
        }
        else if (token == "--shadowMap")
        {
            parseToken(nextToken, "boolean", shadowMap);
        }
        else if (token == "--path")
        {
            searchPath.append(mx::FileSearchPath(nextToken));
        }
        else if (token == "--library")
        {
            libraryFolders.push_back(nextToken);
        }
        else if (token == "--screenWidth")
        {
            parseToken(nextToken, "integer", screenWidth);
        }
        else if (token == "--screenHeight")
        {
            parseToken(nextToken, "integer", screenHeight);
        }
        else if (token == "--screenColor")
        {
            parseToken(nextToken, "color3", screenColor);
        }
        else if (token == "--drawEnvironment")
        {
            parseToken(nextToken, "boolean", drawEnvironment);
        }
        else if (token == "--captureFilename")
        {
            parseToken(nextToken, "string", captureFilename);
        }
        else if (token == "--bakeWidth")
        {
            parseToken(nextToken, "integer", bakeWidth);
        }
        else if (token == "--bakeHeight")
        {
            parseToken(nextToken, "integer", bakeHeight);
        }
        else if (token == "--bakeFilename")
        {
            parseToken(nextToken, "string", bakeFilename);
        }
        else if (token == "--refresh")
        {
            parseToken(nextToken, "float", refresh);
        }
        else if (token == "--remap")
        {
            mx::StringVec vec = mx::splitString(nextToken, ":");
            if (vec.size() == 2)
            {
                modifiers.remapElements[vec[0]] = vec[1];
            }
            else if (!nextToken.empty())
            {
                std::cout << "Unable to parse token following command-line option: " << token << std::endl;
            }
        }
        else if (token == "--skip")
        {
            modifiers.skipElements.insert(nextToken);
        }
        else if (token == "--terminator")
        {
            modifiers.filePrefixTerminator = nextToken;
        }
        else if (token == "--help")
        {
            std::cout << " MaterialXView version " << mx::getVersionString() << std::endl;
            std::cout << options << std::endl;
            return 0;
        }
        else
        {
            std::cout << "Unrecognized command-line option: " << token << std::endl;
            std::cout << "Launch the viewer with '--help' for a complete list of supported options." << std::endl;
            continue;
        }

        if (nextToken.empty())
        {
            std::cout << "Expected another token following command-line option: " << token << std::endl;
        }
        else
        {
            i++;
        }
    }

    // Append the standard library folder, giving it a lower precedence than user-supplied libraries.
    libraryFolders.push_back("libraries");

    ng::init();
    {
        ng::ref<Viewer> viewer = new Viewer(materialFilename,
                                            meshFilename,
                                            envRadianceFilename,
                                            searchPath,
                                            libraryFolders,
                                            screenWidth,
                                            screenHeight,
                                            screenColor);
        viewer->setMeshRotation(meshRotation);
        viewer->setMeshScale(meshScale);
        viewer->setTurntableEnabled(turntableEnabled);
        viewer->setTurntableSteps(turntableSteps);
        viewer->setCameraPosition(cameraPosition);
        viewer->setCameraTarget(cameraTarget);
        viewer->setCameraViewAngle(cameraViewAngle);
        viewer->setCameraZoom(cameraZoom);
        viewer->setSpecularEnvironmentMethod(specularEnvironmentMethod);
        viewer->setEnvSampleCount(envSampleCount);
        viewer->setEnvLightIntensity(envLightIntensity);
        viewer->setLightRotation(lightRotation);
        viewer->setShadowMapEnable(shadowMap);
        viewer->setDrawEnvironment(drawEnvironment);
        viewer->setDocumentModifiers(modifiers);
        viewer->setBakeWidth(bakeWidth);
        viewer->setBakeHeight(bakeHeight);
        viewer->setBakeFilename(bakeFilename);
        viewer->initialize();

        if (!captureFilename.empty())
        {
            viewer->requestFrameCapture(captureFilename);
            viewer->draw_all();
            viewer->requestExit();
        }
        else if (!bakeFilename.empty())
        {
            viewer->bakeTextures();
            viewer->requestExit();
        }
        else
        {
            viewer->set_visible(true);
        }
        ng::mainloop(refresh);
    }
    ng::shutdown();

    return 0;
}
