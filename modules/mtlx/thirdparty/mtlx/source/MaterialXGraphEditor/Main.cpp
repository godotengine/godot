//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXGraphEditor/Graph.h>
#include <MaterialXFormat/Environ.h>
#include <MaterialXFormat/File.h>
#include <MaterialXFormat/Util.h>

#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <GLFW/glfw3.h>

#include <iostream>

namespace
{

static void errorCallback(int error, const char* description)
{
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

const std::string options =
    " Options: \n"
    "    --material [FILENAME]          Specify the filename of the MTLX document to be displayed in the graph editor\n"
    "    --mesh [FILENAME]              Specify the filename of the OBJ or glTF mesh to be displayed in the graph editor\n"
    "    --path [FILEPATH]              Specify an additional data search path location (e.g. '/projects/MaterialX').  This absolute path will be queried when locating data libraries, XInclude references, and referenced images.\n"
    "    --library [FILEPATH]           Specify an additional data library folder (e.g. 'vendorlib', 'studiolib').  This relative path will be appended to each location in the data search path when loading data libraries.\n"
    "    --captureFilename [FILENAME]   Specify the filename to which the first rendered frame should be written\n"
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

} // anonymous namespace

int main(int argc, char* const argv[])
{
    std::vector<std::string> tokens;
    for (int i = 1; i < argc; i++)
    {
        tokens.emplace_back(argv[i]);
    }

    std::string materialFilename = "resources/Materials/Examples/StandardSurface/standard_surface_marble_solid.mtlx";
    std::string meshFilename = "resources/Geometry/shaderball.glb";
    mx::FileSearchPath searchPath = mx::getDefaultDataSearchPath();
    mx::FilePathVec libraryFolders;
    int viewWidth = 256;
    int viewHeight = 256;
    std::string captureFilename;

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
        else if (token == "--path")
        {
            searchPath.append(mx::FileSearchPath(nextToken));
        }
        else if (token == "--library")
        {
            libraryFolders.push_back(nextToken);
        }
        else if (token == "--viewWidth")
        {
            parseToken(nextToken, "integer", viewWidth);
        }
        else if (token == "--viewHeight")
        {
            parseToken(nextToken, "integer", viewHeight);
        }
        else if (token == "--captureFilename")
        {
            parseToken(nextToken, "string", captureFilename);
        }
        else if (token == "--help")
        {
            std::cout << " MaterialXGraphEditor version " << mx::getVersionString() << std::endl;
            std::cout << options << std::endl;
            return 0;
        }
        else
        {
            std::cout << "Unrecognized command-line option: " << token << std::endl;
            std::cout << "Launch the graph editor with '--help' for a complete list of supported options." << std::endl;
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

    // Setup window
    glfwSetErrorCallback(errorCallback);
    if (!glfwInit())
    {
        return 1;
    }

    // Determine GL and GLSL versions
#if defined(__APPLE__)
    // GL 3.2 + GLSL 150
    const char* glsl_version = "#version 150";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // 3.2+ only
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);           // Required on Mac
#else
    // GL 3.0 + GLSL 130
    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
#endif

    // Create window with graphics context
    GLFWwindow* window = glfwCreateWindow(1280, 960, "MaterialX Graph Editor", NULL, NULL);
    if (!window)
    {
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();

    // Set ini and log filename to NULL. This will prevent the automatic creation of imgui.ini
    // in the same folder as the saved material.
    // TODO: Consider setting the ini and log file paths to an application directory.
    io.IniFilename = NULL;
    io.LogFilename = NULL;

    io.Fonts->AddFontDefault();

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Create graph editor.
    Graph* graph = new Graph(materialFilename,
                             meshFilename,
                             searchPath,
                             libraryFolders,
                             viewWidth,
                             viewHeight);
    if (!captureFilename.empty())
    {
        graph->getRenderer()->requestFrameCapture(captureFilename);
        graph->getRenderer()->requestExit();
    }

    // Handle DPI scaling
    // Note that ScaleAllSizes() only handles things like spacing of elements.
    // Fonts are not handled, so a global font scale factor is set.
    // There appears to be no multi-monitor solution so use the primary monitor
    // for now.
    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    if (monitor)
    {
        float xscale = 1.0f, yscale = 1.0f;
        glfwGetMonitorContentScale(monitor, &xscale, &yscale);
        ImGuiStyle& style = ImGui::GetStyle();
        float dpiScale = xscale > yscale ? xscale : yscale;
        style.ScaleAllSizes(dpiScale);
        graph->setFontScale(dpiScale);
    }

    // Create editor config and context.
    ed::Config config;
    config.SettingsFile = nullptr;
    ed::EditorContext* editorContext = ed::CreateEditor(&config);
    const float ZOOM_LEVELS[] = { 0.1f, 0.15f, 0.20f, 0.25f, 0.33f, 0.5f, 0.75f, 1.0f };
    for (auto& level : ZOOM_LEVELS)
    {
        config.CustomZoomLevels.push_back(level);
    }
    ed::SetCurrentEditor(editorContext);

    // Main loop
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        graph->getRenderer()->drawContents();
        if (!captureFilename.empty())
        {
            break;
        }

        double xpos = 0.0;
        double ypos = 0.0;
        glfwGetCursorPos(window, &xpos, &ypos);
        graph->drawGraph(ImVec2((float) xpos, (float) ypos));
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    if (editorContext)
    {
        ed::DestroyEditor(editorContext);
        editorContext = nullptr;
    }
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
