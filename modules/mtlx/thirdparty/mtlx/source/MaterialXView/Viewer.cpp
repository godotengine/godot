//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXView/Viewer.h>
#include <MaterialXView/RenderPipeline.h>

#ifdef MATERIALXVIEW_METAL_BACKEND
#include <MaterialXView/RenderPipelineMetal.h>
#include <MaterialXGenMsl/MslShaderGenerator.h>
#include <nanogui/metal.h>
#else
#include <MaterialXRenderGlsl/GLUtil.h>
#include <MaterialXView/RenderPipelineGL.h>
#include <MaterialXGenGlsl/GlslShaderGenerator.h>
#endif

#include <MaterialXRender/ShaderRenderer.h>
#include <MaterialXRender/CgltfLoader.h>
#include <MaterialXRender/Harmonics.h>
#include <MaterialXRender/OiioImageLoader.h>
#include <MaterialXRender/StbImageLoader.h>
#include <MaterialXRender/TinyObjLoader.h>

#include <MaterialXGenShader/DefaultColorManagementSystem.h>
#include <MaterialXGenShader/ShaderTranslator.h>

#if MATERIALX_BUILD_GEN_MDL
#include <MaterialXGenMdl/MdlShaderGenerator.h>
#endif
#if MATERIALX_BUILD_GEN_OSL
#include <MaterialXGenOsl/OslShaderGenerator.h>
#endif
#include <MaterialXGenGlsl/EsslShaderGenerator.h>

#include <MaterialXFormat/Environ.h>
#include <MaterialXFormat/Util.h>

#include <nanogui/icons.h>
#include <nanogui/messagedialog.h>
#include <nanogui/opengl.h>
#include <nanogui/vscrollpanel.h>

#include <fstream>
#include <iostream>
#include <iomanip>

const mx::Vector3 DEFAULT_CAMERA_POSITION(0.0f, 0.0f, 5.0f);
const float DEFAULT_CAMERA_VIEW_ANGLE = 45.0f;
const float DEFAULT_CAMERA_ZOOM = 1.0f;

namespace
{

const int MIN_ENV_SAMPLE_COUNT = 4;
const int MAX_ENV_SAMPLE_COUNT = 1024;

const int SHADOW_MAP_SIZE = 2048;
const int ALBEDO_TABLE_SIZE = 128;
const int IRRADIANCE_MAP_WIDTH = 256;
const int IRRADIANCE_MAP_HEIGHT = 128;

const std::string DIR_LIGHT_NODE_CATEGORY = "directional_light";
const std::string IRRADIANCE_MAP_FOLDER = "irradiance";
const float ORTHO_VIEW_DISTANCE = 1000.0f;
const float ORTHO_PROJECTION_HEIGHT = 1.8f;

const float ENV_MAP_SPLIT_RADIANCE = 16.0f;
const float MAX_ENV_TEXEL_RADIANCE = 100000.0f;
const float IDEAL_ENV_MAP_RADIANCE = 6.0f;

const float IDEAL_MESH_SPHERE_RADIUS = 2.0f;

const float PI = std::acos(-1.0f);

void writeTextFile(const std::string& text, const std::string& filePath)
{
    std::ofstream file;
    file.open(filePath);
    file << text;
    file.close();
}

void applyModifiers(mx::DocumentPtr doc, const DocumentModifiers& modifiers)
{
    for (mx::ElementPtr elem : doc->traverseTree())
    {
        if (modifiers.remapElements.count(elem->getCategory()))
        {
            elem->setCategory(modifiers.remapElements.at(elem->getCategory()));
        }
        if (modifiers.remapElements.count(elem->getName()))
        {
            elem->setName(modifiers.remapElements.at(elem->getName()));
        }
        mx::StringVec attrNames = elem->getAttributeNames();
        for (const std::string& attrName : attrNames)
        {
            if (modifiers.remapElements.count(elem->getAttribute(attrName)))
            {
                elem->setAttribute(attrName, modifiers.remapElements.at(elem->getAttribute(attrName)));
            }
        }
        if (elem->hasFilePrefix() && !modifiers.filePrefixTerminator.empty())
        {
            std::string filePrefix = elem->getFilePrefix();
            if (!mx::stringEndsWith(filePrefix, modifiers.filePrefixTerminator))
            {
                elem->setFilePrefix(filePrefix + modifiers.filePrefixTerminator);
            }
        }
        std::vector<mx::ElementPtr> children = elem->getChildren();
        for (mx::ElementPtr child : children)
        {
            if (modifiers.skipElements.count(child->getCategory()) ||
                modifiers.skipElements.count(child->getName()))
            {
                elem->removeChild(child->getName());
            }
        }
    }

    // Remap unsupported texture coordinate indices.
    for (mx::ElementPtr elem : doc->traverseTree())
    {
        mx::NodePtr node = elem->asA<mx::Node>();
        if (node && node->getCategory() == "texcoord")
        {
            mx::InputPtr index = node->getInput("index");
            mx::ValuePtr value = index ? index->getValue() : nullptr;
            if (value && value->isA<int>() && value->asA<int>() != 0)
            {
                index->setValue(0);
            }
        }
    }
}

} // anonymous namespace

//
// Viewer methods
//

Viewer::Viewer(const std::string& materialFilename,
               const std::string& meshFilename,
               const std::string& envRadianceFilename,
               const mx::FileSearchPath& searchPath,
               const mx::FilePathVec& libraryFolders,
               int screenWidth,
               int screenHeight,
               const mx::Color3& screenColor) :
    ng::Screen(ng::Vector2i(screenWidth, screenHeight), "MaterialXView",
        true, false, true, true, false, 4, 0),
    _window(nullptr),
    _materialFilename(materialFilename),
    _meshFilename(meshFilename),
    _envRadianceFilename(envRadianceFilename),
    _searchPath(searchPath),
    _libraryFolders(libraryFolders),
    _meshScale(1.0f),
    _turntableEnabled(false),
    _turntableSteps(360),
    _turntableStep(0),
    _cameraPosition(DEFAULT_CAMERA_POSITION),
    _cameraUp(0.0f, 1.0f, 0.0f),
    _cameraViewAngle(DEFAULT_CAMERA_VIEW_ANGLE),
    _cameraNearDist(0.05f),
    _cameraFarDist(5000.0f),
    _cameraZoom(DEFAULT_CAMERA_ZOOM),
    _userCameraEnabled(true),
    _userTranslationActive(false),
    _lightRotation(0.0f),
    _normalizeEnvironment(false),
    _splitDirectLight(false),
    _generateReferenceIrradiance(false),
    _saveGeneratedLights(false),
    _shadowSoftness(1),
    _ambientOcclusionGain(0.6f),
    _selectedGeom(0),
    _geomLabel(nullptr),
    _geometrySelectionBox(nullptr),
    _selectedMaterial(0),
    _materialLabel(nullptr),
    _materialSelectionBox(nullptr),
    _identityCamera(mx::Camera::create()),
    _viewCamera(mx::Camera::create()),
    _envCamera(mx::Camera::create()),
    _shadowCamera(mx::Camera::create()),
    _lightHandler(mx::LightHandler::create()),
#ifndef MATERIALXVIEW_METAL_BACKEND
    _genContext(mx::GlslShaderGenerator::create()),
    _genContextEssl(mx::EsslShaderGenerator::create()),
#else
    _genContext(mx::MslShaderGenerator::create()),
#endif
#if MATERIALX_BUILD_GEN_OSL
    _genContextOsl(mx::OslShaderGenerator::create()),
#endif
#if MATERIALX_BUILD_GEN_MDL
    _genContextMdl(mx::MdlShaderGenerator::create()),
#endif
    _unitRegistry(mx::UnitConverterRegistry::create()),
    _drawEnvironment(false),
    _outlineSelection(false),
    _renderTransparency(true),
    _renderDoubleSided(true),
    _colorTexture(nullptr),
    _splitByUdims(true),
    _mergeMaterials(false),
    _showAllInputs(false),
    _flattenSubgraphs(false),
    _targetShader("standard_surface"),
    _captureRequested(false),
    _exitRequested(false),
    _wedgeRequested(false),
    _wedgePropertyName("base"),
    _wedgePropertyMin(0.0f),
    _wedgePropertyMax(1.0f),
    _wedgeImageCount(8),
    _bakeHdr(false),
    _bakeAverage(false),
    _bakeOptimize(true),
    _bakeRequested(false),
    _bakeWidth(0),
    _bakeHeight(0),
    _bakeDocumentPerMaterial(false)
{
    // Resolve input filenames, taking both the provided search path and
    // current working directory into account.
    mx::FileSearchPath localSearchPath = searchPath;
    localSearchPath.append(mx::FilePath::getCurrentPath());
    _materialFilename = localSearchPath.find(_materialFilename);
    _meshFilename = localSearchPath.find(_meshFilename);
    _envRadianceFilename = localSearchPath.find(_envRadianceFilename);

    // Set the requested background color.
    set_background(ng::Color(screenColor[0], screenColor[1], screenColor[2], 1.0f));

    // Set default Glsl generator options.
    _genContext.getOptions().targetColorSpaceOverride = "lin_rec709";
    _genContext.getOptions().fileTextureVerticalFlip = true;
    _genContext.getOptions().hwShadowMap = true;
    _genContext.getOptions().hwImplicitBitangents = false;

#ifdef MATERIALXVIEW_METAL_BACKEND
    _renderPipeline = MetalRenderPipeline::create(this);
    _renderPipeline->initialize(ng::metal_device(),
                                ng::metal_command_queue());
#else
    _renderPipeline = GLRenderPipeline::create(this);
    
    // Set Essl generator options
    _genContextEssl.getOptions().targetColorSpaceOverride = "lin_rec709";
    _genContextEssl.getOptions().fileTextureVerticalFlip = false;
    _genContextEssl.getOptions().hwMaxActiveLightSources = 1;
#endif
#if MATERIALX_BUILD_GEN_OSL
    // Set OSL generator options.
    _genContextOsl.getOptions().targetColorSpaceOverride = "lin_rec709";
    _genContextOsl.getOptions().fileTextureVerticalFlip = false;
#endif
#if MATERIALX_BUILD_GEN_MDL
    // Set MDL generator options.
    _genContextMdl.getOptions().targetColorSpaceOverride = "lin_rec709";
    _genContextMdl.getOptions().fileTextureVerticalFlip = false;
#endif
}

void Viewer::initialize()
{
    _window = new ng::Window(this, "Viewer Options");
    _window->set_position(ng::Vector2i(15, 15));
    _window->set_layout(new ng::GroupLayout());

    // Initialize the standard libraries and color/unit management.
    loadStandardLibraries();

    // Initialize image handler.
    _imageHandler = _renderPipeline->createImageHandler();
#if MATERIALX_BUILD_OIIO
    _imageHandler->addLoader(mx::OiioImageLoader::create());
#endif
    _imageHandler->setSearchPath(_searchPath);

    // Initialize user interfaces.
    createLoadMeshInterface(_window, "Load Mesh");
    createLoadMaterialsInterface(_window, "Load Material");
    createLoadEnvironmentInterface(_window, "Load Environment");
    createPropertyEditorInterface(_window, "Property Editor");
    createAdvancedSettings(_window);

    // Create geometry selection box.
    _geomLabel = new ng::Label(_window, "Select Geometry");
    _geometrySelectionBox = new ng::ComboBox(_window, { "None" });
    _geometrySelectionBox->set_chevron_icon(-1);
    _geometrySelectionBox->set_callback([this](int choice)
    {
        size_t index = (size_t) choice;
        if (index < _geometryList.size())
        {
            _selectedGeom = index;
            if (_materialAssignments.count(getSelectedGeometry()))
            {
                setSelectedMaterial(_materialAssignments[getSelectedGeometry()]);
            }
            updateDisplayedProperties();
            updateMaterialSelectionUI();
        }
    });

    // Create material selection box.
    _materialLabel = new ng::Label(_window, "Assigned Material");
    _materialSelectionBox = new ng::ComboBox(_window, { "None" });
    _materialSelectionBox->set_chevron_icon(-1);
    _materialSelectionBox->set_callback([this](int choice)
    {
        size_t index = (size_t) choice;
        if (index < _materials.size())
        {
            // Update selected material index
            _selectedMaterial = index;

            // Assign selected material to geometry
            assignMaterial(getSelectedGeometry(), _materials[index]);
        }
    });

    // Create geometry handler.
    mx::TinyObjLoaderPtr objLoader = mx::TinyObjLoader::create();
    mx::CgltfLoaderPtr gltfLoader = mx::CgltfLoader::create();
    _geometryHandler = mx::GeometryHandler::create();
    _geometryHandler->addLoader(objLoader);
    _geometryHandler->addLoader(gltfLoader);
    loadMesh(_searchPath.find(_meshFilename));

    _renderPipeline->initFramebuffer(width(), height(), nullptr);
    
    // Create environment geometry handler.
    _envGeometryHandler = mx::GeometryHandler::create();
    _envGeometryHandler->addLoader(objLoader);
    mx::FilePath envSphere("resources/Geometry/sphere.obj");
    _envGeometryHandler->loadGeometry(_searchPath.find(envSphere));

    // Initialize environment light.
    loadEnvironmentLight();

    // Initialize camera.
    initCamera();
    set_resize_callback([this](ng::Vector2i size)
    {
#ifdef MATERIALXVIEW_METAL_BACKEND
        _colorTexture = metal_texture();
#endif
        _renderPipeline->resizeFramebuffer(size.x(), size.y(), _colorTexture);
        _viewCamera->setViewportSize(mx::Vector2(static_cast<float>(size[0]), static_cast<float>(size[1])));
    });

    // Update geometry selections.
    updateGeometrySelections();

    // Load the requested material document.
    loadDocument(_materialFilename, _stdLib);

    // Finalize the UI.
    _propertyEditor.setVisible(false);
    perform_layout();

    _turntableTimer.startTimer();
}

void Viewer::loadEnvironmentLight()
{
    // Load the requested radiance map.
    mx::ImagePtr envRadianceMap = _imageHandler->acquireImage(_envRadianceFilename);
    if (!envRadianceMap)
    {
        new ng::MessageDialog(this, ng::MessageDialog::Type::Warning, "Failed to load environment light");
        return;
    }

    // If requested, normalize the environment upon loading.
    if (_normalizeEnvironment)
    {
        envRadianceMap = mx::normalizeEnvironment(envRadianceMap, IDEAL_ENV_MAP_RADIANCE, MAX_ENV_TEXEL_RADIANCE);
        if (_saveGeneratedLights)
        {
            _imageHandler->saveImage("NormalizedRadiance.hdr", envRadianceMap);
        }
    }

    // If requested, split the environment into indirect and direct components.
    if (_splitDirectLight)
    {
        splitDirectLight(envRadianceMap, envRadianceMap, _lightRigDoc);
        if (_saveGeneratedLights)
        {
            _imageHandler->saveImage("IndirectRadiance.hdr", envRadianceMap);
            mx::writeToXmlFile(_lightRigDoc, "DirectLightRig.mtlx");
        }
    }

    // Look for an irradiance map using an expected filename convention.
    mx::ImagePtr envIrradianceMap;
    if (!_normalizeEnvironment && !_splitDirectLight)
    {
        mx::FilePath envIrradiancePath = _envRadianceFilename.getParentPath() / IRRADIANCE_MAP_FOLDER / _envRadianceFilename.getBaseName();
        envIrradianceMap = _imageHandler->acquireImage(envIrradiancePath);
    }

    // If not found, then generate an irradiance map via spherical harmonics.
    if (!envIrradianceMap || envIrradianceMap->getWidth() == 1)
    {
        if (_generateReferenceIrradiance)
        {
            envIrradianceMap = mx::renderReferenceIrradiance(envRadianceMap, IRRADIANCE_MAP_WIDTH, IRRADIANCE_MAP_HEIGHT);
            if (_saveGeneratedLights)
            {
                _imageHandler->saveImage("ReferenceIrradiance.hdr", envIrradianceMap);
            }
        }
        else
        {
            mx::Sh3ColorCoeffs shIrradiance = mx::projectEnvironment(envRadianceMap, true);
            envIrradianceMap = mx::renderEnvironment(shIrradiance, IRRADIANCE_MAP_WIDTH, IRRADIANCE_MAP_HEIGHT);
            if (_saveGeneratedLights)
            {
                _imageHandler->saveImage("SphericalHarmonicIrradiance.hdr", envIrradianceMap);
            }
        }
    }

    // Release any existing environment maps and store the new ones.
    _imageHandler->releaseRenderResources(_lightHandler->getEnvRadianceMap());
    _imageHandler->releaseRenderResources(_lightHandler->getEnvPrefilteredMap());
    _imageHandler->releaseRenderResources(_lightHandler->getEnvIrradianceMap());

    _lightHandler->setEnvRadianceMap(envRadianceMap);
    _lightHandler->setEnvIrradianceMap(envIrradianceMap);
    _lightHandler->setEnvPrefilteredMap(nullptr);

    // Look for a light rig using an expected filename convention.
    if (!_splitDirectLight)
    {
        _lightRigFilename = _envRadianceFilename;
        _lightRigFilename.removeExtension();
        _lightRigFilename.addExtension(mx::MTLX_EXTENSION);
        _lightRigFilename = _searchPath.find(_lightRigFilename);
        if (_lightRigFilename.exists())
        {
            _lightRigDoc = mx::createDocument();
            mx::readFromXmlFile(_lightRigDoc, _lightRigFilename, _searchPath);
        }
        else
        {
            _lightRigDoc = nullptr;
        }
    }

    // Invalidate the existing environment material, if any.
    _envMaterial = nullptr;
}

void Viewer::applyDirectLights(mx::DocumentPtr doc)
{
    if (_lightRigDoc)
    {
        doc->importLibrary(_lightRigDoc);
        _xincludeFiles.insert(_lightRigFilename);
    }

    try
    {
        std::vector<mx::NodePtr> lights;
        _lightHandler->findLights(doc, lights);
        _lightHandler->registerLights(doc, lights, _genContext);
#ifndef MATERIALXVIEW_METAL_BACKEND
        _lightHandler->registerLights(doc, lights, _genContextEssl);
#endif
        _lightHandler->setLightSources(lights);
    }
    catch (std::exception& e)
    {
        new ng::MessageDialog(this, ng::MessageDialog::Type::Warning, "Failed to set up lighting", e.what());
    }
}

void Viewer::assignMaterial(mx::MeshPartitionPtr geometry, mx::MaterialPtr material)
{
    if (!geometry || _geometryHandler->getMeshes().empty())
    {
        return;
    }

    if (geometry == getSelectedGeometry())
    {
        setSelectedMaterial(material);
        if (material)
        {
            updateDisplayedProperties();
        }
    }

    if (material)
    {
        _materialAssignments[geometry] = material;
        material->unbindGeometry();
    }
    else
    {
        _materialAssignments.erase(geometry);
    }
}

mx::FilePath Viewer::getBaseOutputPath()
{
    mx::FilePath baseFilename = _searchPath.find(_materialFilename);
    baseFilename.removeExtension();
    mx::FilePath outputPath = mx::getEnviron("MATERIALX_VIEW_OUTPUT_PATH");
    if (!outputPath.isEmpty())
    {
        baseFilename = outputPath / baseFilename.getBaseName();
    }
    return baseFilename;
}

mx::ElementPredicate Viewer::getElementPredicate()
{
    return [this](mx::ConstElementPtr elem)
    {
        if (elem->hasSourceUri())
        {
            return (_xincludeFiles.count(elem->getSourceUri()) == 0);
        }
        return true;
    };
}

void Viewer::createLoadMeshInterface(Widget* parent, const std::string& label)
{
    ng::Button* meshButton = new ng::Button(parent, label);
    meshButton->set_icon(FA_FOLDER);
    meshButton->set_callback([this]()
    {
        m_process_events = false;
        std::string filename = ng::file_dialog(
        {
            { "obj", "Wavefront OBJ" },
            { "gltf", "GLTF ASCII" },
            { "glb", "GLTF Binary"}
        }, false);
        if (!filename.empty())
        {
            loadMesh(filename);

            _meshRotation = mx::Vector3();
            _meshScale = 1.0f;
            _cameraTarget = mx::Vector3();

            initCamera();
        }
        m_process_events = true;
    });
}

void Viewer::createLoadMaterialsInterface(Widget* parent, const std::string& label)
{
    ng::Button* materialButton = new ng::Button(parent, label);
    materialButton->set_icon(FA_FOLDER);
    materialButton->set_callback([this]()
    {
        m_process_events = false;
        std::string filename = ng::file_dialog({ { "mtlx", "MaterialX" } }, false);
        if (!filename.empty())
        {
            _materialFilename = filename;
            loadDocument(_materialFilename, _stdLib);
        }
        m_process_events = true;
    });
}

void Viewer::createLoadEnvironmentInterface(Widget* parent, const std::string& label)
{
    ng::Button* envButton = new ng::Button(parent, label);
    envButton->set_icon(FA_FOLDER);
    envButton->set_callback([this]()
    {
        m_process_events = false;
        mx::StringSet extensions = _imageHandler->supportedExtensions();
        std::vector<std::pair<std::string, std::string>> filetypes;
        for (const auto& extension : extensions)
        {
            filetypes.emplace_back(extension, extension);
        }
        std::string filename = ng::file_dialog(filetypes, false);
        if (!filename.empty())
        {
            _envRadianceFilename = filename;
            loadEnvironmentLight();
            loadDocument(_materialFilename, _stdLib);
            invalidateShadowMap();
        }
        m_process_events = true;
    });
}

void Viewer::createSaveMaterialsInterface(Widget* parent, const std::string& label)
{
    ng::Button* materialButton = new ng::Button(parent, label);
    materialButton->set_icon(FA_SAVE);
    materialButton->set_callback([this]()
    {
        m_process_events = false;
        mx::MaterialPtr material = getSelectedMaterial();
        mx::FilePath filename = ng::file_dialog({ { "mtlx", "MaterialX" } }, true);

        // Save document
        if (material && !filename.isEmpty())
        {
            if (filename.getExtension() != mx::MTLX_EXTENSION)
            {
                filename.addExtension(mx::MTLX_EXTENSION);
            }

            mx::XmlWriteOptions writeOptions;
            writeOptions.elementPredicate = getElementPredicate();
            mx::writeToXmlFile(material->getDocument(), filename, &writeOptions);

            // Update material file name
            _materialFilename = filename;
        }
        m_process_events = true;
    });
}

void Viewer::createPropertyEditorInterface(Widget* parent, const std::string& label)
{
    ng::Button* editorButton = new ng::Button(parent, label);
    editorButton->set_flags(ng::Button::ToggleButton);
    editorButton->set_change_callback([this](bool state)
    {
        _propertyEditor.setVisible(state);
        perform_layout();
    });
}

void Viewer::createAdvancedSettings(Widget* parent)
{
    ng::PopupButton* advancedButton = new ng::PopupButton(parent, "Advanced Settings");
    advancedButton->set_icon(FA_TOOLS);
    advancedButton->set_chevron_icon(-1);
    ng::Popup* advancedPopupParent = advancedButton->popup();
    advancedPopupParent->set_layout(new ng::GroupLayout());

    ng::VScrollPanel* scrollPanel = new ng::VScrollPanel(advancedPopupParent);
    scrollPanel->set_fixed_height(500);
    ng::Widget* advancedPopup = new ng::Widget(scrollPanel);
    advancedPopup->set_layout(new ng::GroupLayout(13));

    ng::Label* viewLabel = new ng::Label(advancedPopup, "Viewing Options");
    viewLabel->set_font_size(20);
    viewLabel->set_font("sans-bold");

    ng::CheckBox* drawEnvironmentBox = new ng::CheckBox(advancedPopup, "Draw Environment");
    drawEnvironmentBox->set_checked(_drawEnvironment);
    drawEnvironmentBox->set_callback([this](bool enable)
    {
        _drawEnvironment = enable;
    });

    ng::CheckBox* outlineSelectedGeometryBox = new ng::CheckBox(advancedPopup, "Outline Selected Geometry");
    outlineSelectedGeometryBox->set_checked(_outlineSelection);
    outlineSelectedGeometryBox->set_callback([this](bool enable)
    {
        _outlineSelection = enable;
    });

    ng::Label* renderLabel = new ng::Label(advancedPopup, "Render Options");
    renderLabel->set_font_size(20);
    renderLabel->set_font("sans-bold");

    ng::CheckBox* transparencyBox = new ng::CheckBox(advancedPopup, "Render Transparency");
    transparencyBox->set_checked(_renderTransparency);
    transparencyBox->set_callback([this](bool enable)
    {
        _renderTransparency = enable;
    });

    ng::CheckBox* doubleSidedBox = new ng::CheckBox(advancedPopup, "Render Double-Sided");
    doubleSidedBox->set_checked(_renderDoubleSided);
    doubleSidedBox->set_callback([this](bool enable)
    {
        _renderDoubleSided = enable;
    });

    ng::CheckBox* importanceSampleBox = new ng::CheckBox(advancedPopup, "Environment FIS");
    importanceSampleBox->set_checked(_genContext.getOptions().hwSpecularEnvironmentMethod == mx::SPECULAR_ENVIRONMENT_FIS);
    _lightHandler->setUsePrefilteredMap(_genContext.getOptions().hwSpecularEnvironmentMethod != mx::SPECULAR_ENVIRONMENT_FIS);
    importanceSampleBox->set_callback([this](bool enable)
    {
        _genContext.getOptions().hwSpecularEnvironmentMethod = enable ? mx::SPECULAR_ENVIRONMENT_FIS : mx::SPECULAR_ENVIRONMENT_PREFILTER;
#ifndef MATERIALXVIEW_METAL_BACKEND
        _genContextEssl.getOptions().hwSpecularEnvironmentMethod = _genContext.getOptions().hwSpecularEnvironmentMethod;
#endif
        _lightHandler->setUsePrefilteredMap(!enable);
        reloadShaders();
    });

    ng::CheckBox* refractionBox = new ng::CheckBox(advancedPopup, "Transmission Refraction");
    refractionBox->set_checked(_genContext.getOptions().hwTransmissionRenderMethod == mx::TRANSMISSION_REFRACTION);
    refractionBox->set_callback([this](bool enable)
    {
        _genContext.getOptions().hwTransmissionRenderMethod = enable ? mx::TRANSMISSION_REFRACTION : mx::TRANSMISSION_OPACITY;
#ifndef MATERIALXVIEW_METAL_BACKEND
        _genContextEssl.getOptions().hwTransmissionRenderMethod = _genContext.getOptions().hwTransmissionRenderMethod;
#endif
        reloadShaders();
    });

    ng::CheckBox* refractionSidedBox = new ng::CheckBox(advancedPopup, "Refraction Two-Sided");
    refractionSidedBox->set_checked(_lightHandler->getRefractionTwoSided());
    refractionSidedBox->set_callback([this](bool enable)
    {
        _lightHandler->setRefractionTwoSided(enable);
    });

    ng::CheckBox* shaderInterfaceBox = new ng::CheckBox(advancedPopup, "Reduce Shader Interface");
    shaderInterfaceBox->set_checked(_genContext.getOptions().shaderInterfaceType == mx::SHADER_INTERFACE_REDUCED);
    shaderInterfaceBox->set_callback([this](bool enable)
    {
        mx::ShaderInterfaceType interfaceType = enable ? mx::SHADER_INTERFACE_REDUCED : mx::SHADER_INTERFACE_COMPLETE;
        setShaderInterfaceType(interfaceType);
    });

    Widget* albedoGroup = new Widget(advancedPopup);
    albedoGroup->set_layout(new ng::BoxLayout(ng::Orientation::Horizontal));
    new ng::Label(albedoGroup, "Albedo Method:");
    mx::StringVec albedoOptions = { "Analytic", "Table", "MC" };
    ng::ComboBox* albedoBox = new ng::ComboBox(albedoGroup, albedoOptions);
    albedoBox->set_chevron_icon(-1);
    albedoBox->set_selected_index((int) _genContext.getOptions().hwDirectionalAlbedoMethod );
    albedoBox->set_callback([this](int index)
    {
        _genContext.getOptions().hwDirectionalAlbedoMethod = (mx::HwDirectionalAlbedoMethod) index;
        reloadShaders();
        try
        {
            _renderPipeline->updateAlbedoTable(ALBEDO_TABLE_SIZE);
        }
        catch (mx::ExceptionRenderError& e)
        {
            for (const std::string& error : e.errorLog())
            {
                std::cerr << error << std::endl;
            }
            new ng::MessageDialog(this, ng::MessageDialog::Type::Warning, "Shader generation error", e.what());
            _materialAssignments.clear();
        }
        catch (std::exception& e)
        {
            new ng::MessageDialog(this, ng::MessageDialog::Type::Warning, "Failed to update albedo table", e.what());
            _materialAssignments.clear();
        }
    });

    Widget* sampleGroup = new Widget(advancedPopup);
    sampleGroup->set_layout(new ng::BoxLayout(ng::Orientation::Horizontal));
    new ng::Label(sampleGroup, "Environment Samples:");
    mx::StringVec sampleOptions;
    for (int i = MIN_ENV_SAMPLE_COUNT; i <= MAX_ENV_SAMPLE_COUNT; i *= 4)
    {
        m_process_events = false;
        sampleOptions.push_back(std::to_string(i));
        m_process_events = true;
    }
    ng::ComboBox* sampleBox = new ng::ComboBox(sampleGroup, sampleOptions);
    sampleBox->set_chevron_icon(-1);
    sampleBox->set_selected_index((int)std::log2(_lightHandler->getEnvSampleCount() / MIN_ENV_SAMPLE_COUNT) / 2);
    sampleBox->set_callback([this](int index)
    {
        _lightHandler->setEnvSampleCount(MIN_ENV_SAMPLE_COUNT * (int) std::pow(4, index));
    });

    ng::Label* lightingLabel = new ng::Label(advancedPopup, "Lighting Options");
    lightingLabel->set_font_size(20);
    lightingLabel->set_font("sans-bold");

    ng::CheckBox* directLightingBox = new ng::CheckBox(advancedPopup, "Direct Lighting");
    directLightingBox->set_checked(_lightHandler->getDirectLighting());
    directLightingBox->set_callback([this](bool enable)
    {
        _lightHandler->setDirectLighting(enable);
    });

    ng::CheckBox* indirectLightingBox = new ng::CheckBox(advancedPopup, "Indirect Lighting");
    indirectLightingBox->set_checked(_lightHandler->getIndirectLighting());
    indirectLightingBox->set_callback([this](bool enable)
    {
        _lightHandler->setIndirectLighting(enable);
    });

    ng::Widget* lightRotationRow = new ng::Widget(advancedPopup);
    lightRotationRow->set_layout(new ng::BoxLayout(ng::Orientation::Horizontal));
    mx::UIProperties ui;
    ui.uiMin = mx::Value::createValue(0.0f);
    ui.uiMax = mx::Value::createValue(360.0f);
    ng::FloatBox<float>* lightRotationBox = createFloatWidget(lightRotationRow, "Light Rotation:",
        _lightRotation, &ui, [this](float value)
    {
        _lightRotation = value;
        invalidateShadowMap();
    });
    lightRotationBox->set_editable(true);

    ng::Label* shadowingLabel = new ng::Label(advancedPopup, "Shadowing Options");
    shadowingLabel->set_font_size(20);
    shadowingLabel->set_font("sans-bold");

    ng::CheckBox* shadowMapBox = new ng::CheckBox(advancedPopup, "Shadow Map");
    shadowMapBox->set_checked(_genContext.getOptions().hwShadowMap);
    shadowMapBox->set_callback([this](bool enable)
    {
        _genContext.getOptions().hwShadowMap = enable;
        reloadShaders();
    });

    ng::CheckBox* ambientOcclusionBox = new ng::CheckBox(advancedPopup, "Ambient Occlusion");
    ambientOcclusionBox->set_checked(_genContext.getOptions().hwAmbientOcclusion);
    ambientOcclusionBox->set_callback([this](bool enable)
    {
        _genContext.getOptions().hwAmbientOcclusion = enable;
        reloadShaders();
    });

    ng::Widget* ambientOcclusionGainRow = new ng::Widget(advancedPopup);
    ambientOcclusionGainRow->set_layout(new ng::BoxLayout(ng::Orientation::Horizontal));
    ng::FloatBox<float>* ambientOcclusionGainBox = createFloatWidget(ambientOcclusionGainRow, "AO Gain:",
        _ambientOcclusionGain, nullptr, [this](float value)
    {
        _ambientOcclusionGain = value;
    });
    ambientOcclusionGainBox->set_editable(true);

    ng::Label* sceneLabel = new ng::Label(advancedPopup, "Scene Options");
    sceneLabel->set_font_size(20);
    sceneLabel->set_font("sans-bold");

    Widget* unitGroup = new Widget(advancedPopup);
    unitGroup->set_layout(new ng::BoxLayout(ng::Orientation::Horizontal));
    new ng::Label(unitGroup, "Distance Unit:");
    ng::ComboBox* distanceUnitBox = new ng::ComboBox(unitGroup, _distanceUnitOptions);
    distanceUnitBox->set_fixed_size(ng::Vector2i(100, 20));
    distanceUnitBox->set_chevron_icon(-1);
    if (_distanceUnitConverter)
    {
        distanceUnitBox->set_selected_index(_distanceUnitConverter->getUnitAsInteger("meter"));
    }
    distanceUnitBox->set_callback([this](int index)
    {
        m_process_events = false;
        _genContext.getOptions().targetDistanceUnit = _distanceUnitOptions[index];
#ifndef MATERIALXVIEW_METAL_BACKEND
        _genContextEssl.getOptions().targetDistanceUnit = _distanceUnitOptions[index];
#endif
#if MATERIALX_BUILD_GEN_OSL
        _genContextOsl.getOptions().targetDistanceUnit = _distanceUnitOptions[index];
#endif
#if MATERIALX_BUILD_GEN_MDL
        _genContextMdl.getOptions().targetDistanceUnit = _distanceUnitOptions[index];
#endif
        reloadShaders();
        m_process_events = true;
    });

    ng::Label* meshLoading = new ng::Label(advancedPopup, "Mesh Loading Options");
    meshLoading->set_font_size(20);
    meshLoading->set_font("sans-bold");

    ng::CheckBox* splitUdimsBox = new ng::CheckBox(advancedPopup, "Split By UDIMs");
    splitUdimsBox->set_checked(_splitByUdims);
    splitUdimsBox->set_callback([this](bool enable)
    {
        _splitByUdims = enable;
    });

    ng::Label* materialLoading = new ng::Label(advancedPopup, "Material Loading Options");
    materialLoading->set_font_size(20);
    materialLoading->set_font("sans-bold");

    ng::CheckBox* mergeMaterialsBox = new ng::CheckBox(advancedPopup, "Merge Materials");
    mergeMaterialsBox->set_checked(_mergeMaterials);
    mergeMaterialsBox->set_callback([this](bool enable)
    {
        _mergeMaterials = enable;
    });

    ng::CheckBox* showInputsBox = new ng::CheckBox(advancedPopup, "Show All Inputs");
    showInputsBox->set_checked(_showAllInputs);
    showInputsBox->set_callback([this](bool enable)
    {
        _showAllInputs = enable;
    });

    ng::CheckBox* flattenBox = new ng::CheckBox(advancedPopup, "Flatten Subgraphs");
    flattenBox->set_checked(_flattenSubgraphs);
    flattenBox->set_callback([this](bool enable)
    {
        _flattenSubgraphs = enable;
    });

    ng::Label* envLoading = new ng::Label(advancedPopup, "Environment Loading Options");
    envLoading->set_font_size(20);
    envLoading->set_font("sans-bold");

    ng::CheckBox* normalizeEnvBox = new ng::CheckBox(advancedPopup, "Normalize Environment");
    normalizeEnvBox->set_checked(_normalizeEnvironment);
    normalizeEnvBox->set_callback([this](bool enable)
    {
        _normalizeEnvironment = enable;
    });

    ng::CheckBox* splitDirectLightBox = new ng::CheckBox(advancedPopup, "Split Direct Light");
    splitDirectLightBox->set_checked(_splitDirectLight);
    splitDirectLightBox->set_callback([this](bool enable)
    {
        _splitDirectLight = enable;
    });

    ng::Label* translationLabel = new ng::Label(advancedPopup, "Translation Options (T)");
    translationLabel->set_font_size(20);
    translationLabel->set_font("sans-bold");

    ng::Widget* targetShaderGroup = new ng::Widget(advancedPopup);
    targetShaderGroup->set_layout(new ng::BoxLayout(ng::Orientation::Horizontal));
    new ng::Label(targetShaderGroup, "Target Shader");
    ng::TextBox* targetShaderBox = new ng::TextBox(targetShaderGroup, _targetShader);
    targetShaderBox->set_callback([this](const std::string& choice)
    {
        _targetShader = choice;
        return true;
    });
    targetShaderBox->set_font_size(16);
    targetShaderBox->set_editable(true);

    ng::Label* textureLabel = new ng::Label(advancedPopup, "Texture Baking Options (B)");
    textureLabel->set_font_size(20);
    textureLabel->set_font("sans-bold");

    ng::CheckBox* bakeHdrBox = new ng::CheckBox(advancedPopup, "Bake HDR Textures");
    bakeHdrBox->set_checked(_bakeHdr);
    bakeHdrBox->set_callback([this](bool enable)
    {
        _bakeHdr = enable;
    });

    ng::CheckBox* bakeAverageBox = new ng::CheckBox(advancedPopup, "Bake Averaged Textures");
    bakeAverageBox->set_checked(_bakeAverage);
    bakeAverageBox->set_callback([this](bool enable)
    {
        _bakeAverage = enable;
    });

    ng::CheckBox* bakeOptimized = new ng::CheckBox(advancedPopup, "Optimize Baked Constants");
    bakeOptimized->set_checked(_bakeOptimize);
    bakeOptimized->set_callback([this](bool enable)
    {
        _bakeOptimize = enable;
    });

    ng::CheckBox* bakeDocumentPerMaterial= new ng::CheckBox(advancedPopup, "Bake Document Per Material");
    bakeDocumentPerMaterial->set_checked(_bakeDocumentPerMaterial);
    bakeDocumentPerMaterial->set_callback([this](bool enable)
    {
        _bakeDocumentPerMaterial = enable;
    });    

    ng::Label* wedgeLabel = new ng::Label(advancedPopup, "Wedge Render Options (W)");
    wedgeLabel->set_font_size(20);
    wedgeLabel->set_font("sans-bold");

    ng::Widget* wedgeNameGroup = new ng::Widget(advancedPopup);
    wedgeNameGroup->set_layout(new ng::BoxLayout(ng::Orientation::Horizontal));
    new ng::Label(wedgeNameGroup, "Property Name");
    ng::TextBox* wedgeNameBox = new ng::TextBox(wedgeNameGroup, _wedgePropertyName);
    wedgeNameBox->set_callback([this](const std::string& choice)
    {
        _wedgePropertyName = choice;
        return true;
    });
    wedgeNameBox->set_font_size(16);
    wedgeNameBox->set_editable(true);

    ng::Widget* wedgeMinGroup = new ng::Widget(advancedPopup);
    wedgeMinGroup->set_layout(new ng::BoxLayout(ng::Orientation::Horizontal));
    mx::UIProperties wedgeProp;
    wedgeProp.uiSoftMin = mx::Value::createValue(0.0f);
    wedgeProp.uiSoftMax = mx::Value::createValue(1.0f);
    ng::FloatBox<float>* wedgeMinBox = createFloatWidget(wedgeMinGroup, "Property Min:",
        _wedgePropertyMax, &wedgeProp, [this](float value)
    {
        _wedgePropertyMin = value;
    });
    wedgeMinBox->set_value(0.0);
    wedgeMinBox->set_editable(true);

    ng::Widget* wedgeMaxGroup = new ng::Widget(advancedPopup);
    wedgeMaxGroup->set_layout(new ng::BoxLayout(ng::Orientation::Horizontal));
    ng::FloatBox<float>* wedgeMaxBox = createFloatWidget(wedgeMaxGroup, "Property Max:",
        _wedgePropertyMax, &wedgeProp, [this](float value)
    {
        _wedgePropertyMax = value;
    });
    wedgeMaxBox->set_value(1.0);
    wedgeMaxBox->set_editable(true);

    ng::Widget* wedgeCountGroup = new ng::Widget(advancedPopup);
    wedgeCountGroup->set_layout(new ng::BoxLayout(ng::Orientation::Horizontal));
    mx::UIProperties wedgeCountProp;
    wedgeCountProp.uiMin = mx::Value::createValue(1);
    wedgeCountProp.uiSoftMax = mx::Value::createValue(8);
    wedgeCountProp.uiStep = mx::Value::createValue(1);
    ng::IntBox<int>* wedgeCountBox = createIntWidget(wedgeCountGroup, "Image Count:",
        _wedgeImageCount, &wedgeCountProp, [this](int value)
    {
        _wedgeImageCount = value;
    });
    wedgeCountBox->set_value(8);
    wedgeCountBox->set_editable(true);
}

void Viewer::updateGeometrySelections()
{
    _geometryList.clear();
    if (_geometryHandler->getMeshes().empty())
    {
        return;
    }
    for (auto mesh : _geometryHandler->getMeshes())
    {
        for (size_t partIndex = 0; partIndex < mesh->getPartitionCount(); partIndex++)
        {
            mx::MeshPartitionPtr part = mesh->getPartition(partIndex);
            _geometryList.push_back(part);
        }
    }

    std::vector<std::string> items;
    for (const mx::MeshPartitionPtr& part : _geometryList)
    {
        std::string geomName = part->getName();
        mx::StringVec geomSplit = mx::splitString(geomName, ":");
        if (!geomSplit.empty() && !geomSplit[geomSplit.size() - 1].empty())
        {
            geomName = geomSplit[geomSplit.size() - 1];
        }
        items.push_back(geomName);
    }
    _geometrySelectionBox->set_items(items);

    _geomLabel->set_visible(items.size() > 1);
    _geometrySelectionBox->set_visible(items.size() > 1);
    _selectedGeom = 0;

    perform_layout();
}

void Viewer::updateMaterialSelections()
{
    std::vector<std::string> items;
    for (const auto& material : _materials)
    {
        mx::ElementPtr displayElem = material->getMaterialNode() ?
                                     material->getMaterialNode() :
                                     material->getElement();
        std::string displayName = displayElem->getName();
        if (displayName == "out")
        {
            displayName = displayElem->getParent()->getName();
        }
        if (!material->getUdim().empty())
        {
            displayName += " (" + material->getUdim() + ")";
        }
        items.push_back(displayName);
    }
    _materialSelectionBox->set_items(items);

    _materialLabel->set_visible(items.size() > 1);
    _materialSelectionBox->set_visible(items.size() > 1);

    perform_layout();
}

void Viewer::updateMaterialSelectionUI()
{
    for (const auto& pair : _materialAssignments)
    {
        if (pair.first == getSelectedGeometry())
        {
            for (size_t i = 0; i < _materials.size(); i++)
            {
                if (_materials[i] == pair.second)
                {
                    _materialSelectionBox->set_selected_index((int) i);
                    break;
                }
            }
        }
    }
    perform_layout();
}

void Viewer::loadMesh(const mx::FilePath& filename)
{
    _geometryHandler->clearGeometry();
    if (_geometryHandler->loadGeometry(filename))
    {
        _meshFilename = filename;
        if (_splitByUdims)
        {
            for (auto mesh : _geometryHandler->getMeshes())
            {
                mesh->splitByUdims();
            }
        }

        updateGeometrySelections();

        // Assign the selected material to all geometries.
        _materialAssignments.clear();
        mx::MaterialPtr material = getSelectedMaterial();
        if (material)
        {
            for (mx::MeshPartitionPtr geom : _geometryList)
            {
                assignMaterial(geom, material);
            }
        }

        // Unbind utility materials from the previous geometry.
        if (_wireMaterial)
        {
            _wireMaterial->unbindGeometry();
        }
        if (_shadowMaterial)
        {
            _shadowMaterial->unbindGeometry();
        }

        invalidateShadowMap();
    }
    else
    {
        new ng::MessageDialog(this, ng::MessageDialog::Type::Warning, "Mesh Loading Error", filename);
    }
}

void Viewer::loadDocument(const mx::FilePath& filename, mx::DocumentPtr libraries)
{
    // Set up read options.
    mx::XmlReadOptions readOptions;
    readOptions.readXIncludeFunction = [](mx::DocumentPtr doc, const mx::FilePath& filename,
                                          const mx::FileSearchPath& searchPath, const mx::XmlReadOptions* options)
    {
        mx::FilePath resolvedFilename = searchPath.find(filename);
        if (resolvedFilename.exists())
        {
            readFromXmlFile(doc, resolvedFilename, searchPath, options);
        }
        else
        {
            std::cerr << "Include file not found: " << filename.asString() << std::endl;
        }
    };

    // Clear user data on the generator.
    _genContext.clearUserData();
#ifndef MATERIALXVIEW_METAL_BACKEND
    _genContextEssl.clearUserData();
#endif

    // Clear materials if merging is not requested.
    if (!_mergeMaterials)
    {
        for (mx::MeshPartitionPtr geom : _geometryList)
        {
            if (_materialAssignments.count(geom))
            {
                assignMaterial(geom, nullptr);
            }
        }
        _materials.clear();
    }

    std::vector<mx::MaterialPtr> newMaterials;
    try
    {
        // Load source document.
        mx::DocumentPtr doc = mx::createDocument();
        mx::readFromXmlFile(doc, filename, _searchPath, &readOptions);
        _materialSearchPath = mx::getSourceSearchPath(doc);

        // Import libraries.
        doc->importLibrary(libraries);

        // Apply direct lights.
        applyDirectLights(doc);

        // Apply modifiers to the content document.
        applyModifiers(doc, _modifiers);

        // Flatten subgraphs if requested.
        if (_flattenSubgraphs)
        {
            doc->flattenSubgraphs();
            for (mx::NodeGraphPtr graph : doc->getNodeGraphs())
            {
                if (graph->getActiveSourceUri() == doc->getActiveSourceUri())
                {
                    graph->flattenSubgraphs();
                }
            }
        }

        // Validate the document.
        std::string message;
        if (!doc->validate(&message))
        {
            std::cerr << "*** Validation warnings for " << _materialFilename.getBaseName() << " ***" << std::endl;
            std::cerr << message;
        }

        // If requested, add implicit inputs to top-level nodes.
        if (_showAllInputs)
        {
            for (mx::NodePtr node : doc->getNodes())
            {
                node->addInputsFromNodeDef();
            }
        }

        // Find new renderable elements.
        mx::StringVec renderablePaths;
        std::vector<mx::TypedElementPtr> elems = mx::findRenderableElements(doc);
        if (elems.empty())
        {
            throw mx::Exception("No renderable elements found in " + _materialFilename.getBaseName());
        }
        std::vector<mx::NodePtr> materialNodes;
        for (mx::TypedElementPtr elem : elems)
        {
            mx::TypedElementPtr renderableElem = elem;
            mx::NodePtr node = elem->asA<mx::Node>();
            materialNodes.push_back(node && node->getType() == mx::MATERIAL_TYPE_STRING ? node : nullptr);
            renderablePaths.push_back(renderableElem->getNamePath());
        }

        // Check for any udim set.
        mx::ValuePtr udimSetValue = doc->getGeomPropValue(mx::UDIM_SET_PROPERTY);

        // Create new materials.
        mx::TypedElementPtr udimElement;
        for (size_t i=0; i<renderablePaths.size(); i++)
        {
            const auto& renderablePath = renderablePaths[i];
            mx::ElementPtr elem = doc->getDescendant(renderablePath);
            mx::TypedElementPtr typedElem = elem ? elem->asA<mx::TypedElement>() : nullptr;
            if (!typedElem)
            {
                continue;
            }
            if (udimSetValue && udimSetValue->isA<mx::StringVec>())
            {
                for (const std::string& udim : udimSetValue->asA<mx::StringVec>())
                {
                    mx::MaterialPtr mat = _renderPipeline->createMaterial();
                    mat->setDocument(doc);
                    mat->setElement(typedElem);
                    mat->setMaterialNode(materialNodes[i]);
                    mat->setUdim(udim);
                    newMaterials.push_back(mat);
                    
                    udimElement = typedElem;
                }
            }
            else
            {
                mx::MaterialPtr mat = _renderPipeline->createMaterial();
                mat->setDocument(doc);
                mat->setElement(typedElem);
                mat->setMaterialNode(materialNodes[i]);
                newMaterials.push_back(mat);
            }
        }

        if (!newMaterials.empty())
        {
            // Extend the image search path to include material source folders.
            mx::FileSearchPath extendedSearchPath = _searchPath;
            extendedSearchPath.append(_materialSearchPath);
            _imageHandler->setSearchPath(extendedSearchPath);

            // Add new materials to the global vector.
            _materials.insert(_materials.end(), newMaterials.begin(), newMaterials.end());

            mx::MaterialPtr udimMaterial = nullptr;
            for (mx::MaterialPtr mat : newMaterials)
            {
                // Clear cached implementations, in case libraries on the file system have changed.
                _genContext.clearNodeImplementations();
#ifndef MATERIALXVIEW_METAL_BACKEND
                _genContextEssl.clearNodeImplementations();
#endif

                mx::TypedElementPtr elem = mat->getElement();

                std::string udim = mat->getUdim();
                if (!udim.empty())
                {
                    if ((udimElement == elem) && udimMaterial)
                    {
                        // Reuse existing material for all udims
                        mat->copyShader(udimMaterial);
                    }
                    else
                    {
                        // Generate a shader for the new material.
                        mat->generateShader(_genContext);
                        if (udimElement == elem)
                        {
                            udimMaterial = mat;
                        }
                    }
                }
                else
                {
                    // Generate a shader for the new material.
                    mat->generateShader(_genContext);
                }
            }
       
            // Apply material assignments in the order in which they are declared within the document,
            // with later assignments superseding earlier ones.
            for (mx::LookPtr look : doc->getLooks())
            {
                for (mx::MaterialAssignPtr matAssign : look->getActiveMaterialAssigns())
                {
                    const std::string& activeGeom = matAssign->getActiveGeom();
                    for (mx::MeshPartitionPtr part : _geometryList)
                    {
                        std::string geom = part->getName();
                        for (const std::string& id : part->getSourceNames())
                        {
                            geom += mx::ARRAY_PREFERRED_SEPARATOR + id;
                        }
                        if (mx::geomStringsMatch(activeGeom, geom, true))
                        {
                            for (mx::MaterialPtr mat : newMaterials)
                            {
                                if (mat->getMaterialNode() == matAssign->getReferencedMaterial())
                                {
                                    assignMaterial(part, mat);
                                    break;
                                }
                            }
                        }
                        mx::CollectionPtr coll = matAssign->getCollection();
                        if (coll && coll->matchesGeomString(geom))
                        {
                            for (mx::MaterialPtr mat : newMaterials)
                            {
                                if (mat->getMaterialNode() == matAssign->getReferencedMaterial())
                                {
                                    assignMaterial(part, mat);
                                    break;
                                }
                            }
                        }
                    }
                }
            }

            // Apply implicit udim assignments, if any.
            for (mx::MaterialPtr mat : newMaterials)
            {
                mx::NodePtr materialNode = mat->getMaterialNode();
                if (materialNode)
                {
                    std::string udim = mat->getUdim();
                    if (!udim.empty())
                    {
                        for (mx::MeshPartitionPtr geom : _geometryList)
                        {
                            if (geom->getName() == udim)
                            {
                                assignMaterial(geom, mat);
                            }
                        }
                    }
                }
            }

            // Apply fallback assignments.
            mx::MaterialPtr fallbackMaterial = newMaterials[0];
            if (!_mergeMaterials || fallbackMaterial->getUdim().empty())
            {
                for (mx::MeshPartitionPtr geom : _geometryList)
                {
                    if (!_materialAssignments[geom])
                    {
                        assignMaterial(geom, fallbackMaterial);
                    }
                }
            }
        }
    }
    catch (mx::ExceptionRenderError& e)
    {
        for (const std::string& error : e.errorLog())
        {
            std::cerr << error << std::endl;
        }
        new ng::MessageDialog(this, ng::MessageDialog::Type::Warning, "Shader generation error", e.what());
        _materialAssignments.clear();
    }
    catch (std::exception& e)
    {
        new ng::MessageDialog(this, ng::MessageDialog::Type::Warning, "Failed to load material", e.what());
        _materialAssignments.clear();
    }

    // Update material UI.
    updateMaterialSelections();
    updateMaterialSelectionUI();

    invalidateShadowMap();
    perform_layout();
}

void Viewer::reloadShaders()
{
    try
    {
        for (mx::MaterialPtr material : _materials)
        {
            material->generateShader(_genContext);
        }
        return;
    }
    catch (mx::ExceptionRenderError& e)
    {
        for (const std::string& error : e.errorLog())
        {
            std::cerr << error << std::endl;
        }
        new ng::MessageDialog(this, ng::MessageDialog::Type::Warning, "Shader generation error", e.what());
    }
    catch (std::exception& e)
    {
        new ng::MessageDialog(this, ng::MessageDialog::Type::Warning, "Failed to reload shaders", e.what());
    }

    _materialAssignments.clear();
}

void Viewer::saveShaderSource(mx::GenContext& context)
{
    try
    {
        mx::MaterialPtr material = getSelectedMaterial();
        mx::TypedElementPtr elem = material ? material->getElement() : nullptr;
        if (elem)
        {
            mx::FilePath sourceFilename = getBaseOutputPath();
#ifndef MATERIALXVIEW_METAL_BACKEND
            if (context.getShaderGenerator().getTarget() == mx::GlslShaderGenerator::TARGET)
            {
                mx::ShaderPtr shader = material->getShader();
                const std::string& pixelShader = shader->getSourceCode(mx::Stage::PIXEL);
                const std::string& vertexShader = shader->getSourceCode(mx::Stage::VERTEX);
                writeTextFile(pixelShader, sourceFilename.asString() + "_ps.glsl");
                writeTextFile(vertexShader, sourceFilename.asString() + "_vs.glsl");
                new ng::MessageDialog(this, ng::MessageDialog::Type::Information, "Saved GLSL source: ",
                    sourceFilename.asString() + "_*.glsl");
            }
            else if (context.getShaderGenerator().getTarget() == mx::EsslShaderGenerator::TARGET)
            {
                mx::ShaderPtr shader = createShader(elem->getNamePath(), context, elem);
                const std::string& pixelShader = shader->getSourceCode(mx::Stage::PIXEL);
                const std::string& vertexShader = shader->getSourceCode(mx::Stage::VERTEX);
                writeTextFile(vertexShader, sourceFilename.asString() + "_essl_vs.glsl");
                writeTextFile(pixelShader, sourceFilename.asString() + "_essl_ps.glsl");
                new ng::MessageDialog(this, ng::MessageDialog::Type::Information, "Saved ESSL source: ",
                    sourceFilename.asString() + "_essl_*.glsl");
            }
#else
            if (context.getShaderGenerator().getTarget() == mx::MslShaderGenerator::TARGET)
            {
                mx::ShaderPtr shader = material->getShader();
                const std::string& pixelShader = shader->getSourceCode(mx::Stage::PIXEL);
                const std::string& vertexShader = shader->getSourceCode(mx::Stage::VERTEX);
                writeTextFile(pixelShader, sourceFilename.asString() + "_ps.metal");
                writeTextFile(vertexShader, sourceFilename.asString() + "_vs.metal");
                new ng::MessageDialog(this, ng::MessageDialog::Type::Information, "Saved GLSL source: ",
                    sourceFilename.asString() + "_*.metal");
            }
#endif
#if MATERIALX_BUILD_GEN_OSL
            else if (context.getShaderGenerator().getTarget() == mx::OslShaderGenerator::TARGET)
            {
                mx::ShaderPtr shader = createShader(elem->getNamePath(), context, elem);
                const std::string& pixelShader = shader->getSourceCode(mx::Stage::PIXEL);
                sourceFilename.addExtension("osl");
                writeTextFile(pixelShader, sourceFilename);
                new ng::MessageDialog(this, ng::MessageDialog::Type::Information, "Saved OSL source: ", sourceFilename);
            }
#endif
#if MATERIALX_BUILD_GEN_MDL
            else if (context.getShaderGenerator().getTarget() == mx::MdlShaderGenerator::TARGET)
            {
                mx::ShaderPtr shader = createShader(elem->getNamePath(), context, elem);
                const std::string& pixelShader = shader->getSourceCode(mx::Stage::PIXEL);
                sourceFilename.addExtension("mdl");
                writeTextFile(pixelShader, sourceFilename);
                new ng::MessageDialog(this, ng::MessageDialog::Type::Information, "Saved MDL source: ", sourceFilename);
            }
#endif
        }
    }
    catch (std::exception& e)
    {
        new ng::MessageDialog(this, ng::MessageDialog::Type::Warning, "Cannot save source for material", e.what());
    }
}

void Viewer::loadShaderSource()
{
    try
    {
        mx::MaterialPtr material = getSelectedMaterial();
        mx::TypedElementPtr elem = material ? material->getElement() : nullptr;
        if (elem)
        {
            mx::FilePath sourceFilename = getBaseOutputPath();
            mx::FilePath pixelSourceFilename = sourceFilename.asString() + "_ps.glsl";
            mx::FilePath vertexSourceFilename = sourceFilename.asString() + "_vs.glsl";
            if (material->loadSource(vertexSourceFilename, pixelSourceFilename, material->hasTransparency()))
            {
                assignMaterial(getSelectedGeometry(), material);
            }
        }
    }
    catch (std::exception& e)
    {
        new ng::MessageDialog(this, ng::MessageDialog::Type::Warning, "Cannot load source for material", e.what());
    }
}

void Viewer::saveDotFiles()
{
    try
    {
        mx::MaterialPtr material = getSelectedMaterial();
        mx::TypedElementPtr elem = material ? material->getElement() : nullptr;
        mx::NodePtr shaderNode = elem->asA<mx::Node>();
        if (shaderNode)
        {
            mx::FilePath baseFilename = getBaseOutputPath();
            for (mx::InputPtr input : shaderNode->getInputs())
            {
                mx::OutputPtr output = input->getConnectedOutput();
                mx::ConstNodeGraphPtr nodeGraph = output ? output->getAncestorOfType<mx::NodeGraph>() : nullptr;
                if (nodeGraph)
                {
                    std::string dotString = nodeGraph->asStringDot();
                    std::string dotFilename = baseFilename.asString() + "_" + nodeGraph->getName() + ".dot";
                    writeTextFile(dotString, dotFilename);
                }
            }

            mx::NodeDefPtr nodeDef = shaderNode->getNodeDef();
            mx::InterfaceElementPtr implement = nodeDef ? nodeDef->getImplementation() : nullptr;
            mx::NodeGraphPtr nodeGraph = implement ? implement->asA<mx::NodeGraph>() : nullptr;
            if (nodeGraph)
            {
                std::string dotString = nodeGraph->asStringDot();
                std::string dotFilename = baseFilename.asString() + "_" + nodeDef->getName() + ".dot";
                writeTextFile(dotString, dotFilename);
            }
            new ng::MessageDialog(this, ng::MessageDialog::Type::Information, "Saved dot files: ", baseFilename.asString() + "_*.dot");
        }
    }
    catch (std::exception& e)
    {
        new ng::MessageDialog(this, ng::MessageDialog::Type::Warning, "Cannot save dot file for material", e.what());
    }
}

mx::UnsignedIntPair Viewer::computeBakingResolution(mx::ConstDocumentPtr doc)
{
    mx::ImageVec imageVec = _imageHandler->getReferencedImages(doc);
    mx::UnsignedIntPair bakingRes = mx::getMaxDimensions(imageVec);
    bakingRes.first = std::max(bakingRes.first, (unsigned int) 4);
    bakingRes.second = std::max(bakingRes.second, (unsigned int) 4);
    if (_bakeWidth)
    {
        bakingRes.first = std::max(_bakeWidth, (unsigned int) 4);
    }
    if (_bakeHeight)
    {
        bakingRes.second = std::max(_bakeHeight, (unsigned int) 4);
    }
    return bakingRes;
}

mx::DocumentPtr Viewer::translateMaterial()
{
    mx::MaterialPtr material = getSelectedMaterial();
    mx::DocumentPtr doc = material ? material->getDocument() : nullptr;
    if (!doc)
    {
        return nullptr;
    }

    mx::DocumentPtr translatedDoc = doc->copy();
    mx::ShaderTranslatorPtr translator = mx::ShaderTranslator::create();
    try
    {
        translator->translateAllMaterials(translatedDoc, _targetShader);
    }
    catch (std::exception& e)
    {
        new ng::MessageDialog(this, ng::MessageDialog::Type::Warning, "Failed to translate material", e.what());
        return nullptr;
    }

    return translatedDoc;
}

void Viewer::initContext(mx::GenContext& context)
{
    // Initialize search path.
    context.registerSourceCodeSearchPath(_searchPath);

    // Initialize color management.
    mx::DefaultColorManagementSystemPtr cms = mx::DefaultColorManagementSystem::create(context.getShaderGenerator().getTarget());
    cms->loadLibrary(_stdLib);
    context.getShaderGenerator().setColorManagementSystem(cms);

    // Initialize unit management.
    mx::UnitSystemPtr unitSystem = mx::UnitSystem::create(context.getShaderGenerator().getTarget());
    unitSystem->loadLibrary(_stdLib);
    unitSystem->setUnitConverterRegistry(_unitRegistry);
    context.getShaderGenerator().setUnitSystem(unitSystem);
    context.getOptions().targetDistanceUnit = "meter";
}

void Viewer::loadStandardLibraries()
{
    // Initialize the standard library.
    try
    {
        _stdLib = mx::createDocument();
        _xincludeFiles = mx::loadLibraries(_libraryFolders, _searchPath, _stdLib);
        if (_xincludeFiles.empty())
        {
            std::cerr << "Could not find standard data libraries on the given search path: " << _searchPath.asString() << std::endl;
        }
    }
    catch (std::exception& e)
    {
        std::cerr << "Failed to load standard data libraries: " << e.what() << std::endl;
        return;
    }

    // Initialize unit management.
    mx::UnitTypeDefPtr distanceTypeDef = _stdLib->getUnitTypeDef("distance");
    _distanceUnitConverter = mx::LinearUnitConverter::create(distanceTypeDef);
    _unitRegistry->addUnitConverter(distanceTypeDef, _distanceUnitConverter);
    mx::UnitTypeDefPtr angleTypeDef = _stdLib->getUnitTypeDef("angle");
    mx::LinearUnitConverterPtr angleConverter = mx::LinearUnitConverter::create(angleTypeDef);
    _unitRegistry->addUnitConverter(angleTypeDef, angleConverter);

    // Create the list of supported distance units.
    auto unitScales = _distanceUnitConverter->getUnitScale();
    _distanceUnitOptions.resize(unitScales.size());
    for (auto unitScale : unitScales)
    {
        int location = _distanceUnitConverter->getUnitAsInteger(unitScale.first);
        _distanceUnitOptions[location] = unitScale.first;
    }

    // Initialize the generator contexts.
    initContext(_genContext);
#ifndef MATERIALXVIEW_METAL_BACKEND
    initContext(_genContextEssl);
#endif
#if MATERIALX_BUILD_GEN_OSL
    initContext(_genContextOsl);
#endif
#if MATERIALX_BUILD_GEN_MDL
    initContext(_genContextMdl);
#endif
}

bool Viewer::keyboard_event(int key, int scancode, int action, int modifiers)
{
    if (Screen::keyboard_event(key, scancode, action, modifiers))
    {
        return true;
    }

    // Adjust camera zoom.
    if (_userCameraEnabled)
    {
        if (key == GLFW_KEY_KP_ADD && action == GLFW_PRESS)
        {
            _cameraZoom *= 1.1f;
        }
        if (key == GLFW_KEY_KP_SUBTRACT && action == GLFW_PRESS)
        {
            _cameraZoom = std::max(0.1f, _cameraZoom * 0.9f);
        }
    }

    // Reload the current document, and optionally the standard libraries, from
    // the file system.
    if (key == GLFW_KEY_R && action == GLFW_PRESS)
    {
        mx::MaterialPtr material = getSelectedMaterial();
        mx::DocumentPtr doc = material ? material->getDocument() : nullptr;
        mx::FilePath filename = doc ? mx::FilePath(doc->getSourceUri()) : _materialFilename;
        if (modifiers == GLFW_MOD_SHIFT)
        {
            loadStandardLibraries();
        }
        loadDocument(filename, _stdLib);
        return true;
    }

    // Save GLSL shader source to file.
    if (key == GLFW_KEY_G && action == GLFW_PRESS)
    {
        saveShaderSource(_genContext);
        return true;
    }

#if MATERIALX_BUILD_GEN_OSL
    // Save OSL shader source to file.
    if (key == GLFW_KEY_O && action == GLFW_PRESS)
    {
        saveShaderSource(_genContextOsl);
        return true;
    }
#endif

#if MATERIALX_BUILD_GEN_MDL
    // Save MDL shader source to file.
    if (key == GLFW_KEY_M && action == GLFW_PRESS)
    {
        saveShaderSource(_genContextMdl);
        return true;
    }
#endif

    // Save Essl shader source to file.
    if (key == GLFW_KEY_E && action == GLFW_PRESS)
    {
#ifndef MATERIALXVIEW_METAL_BACKEND
        saveShaderSource(_genContextEssl);
#endif
        return true;
    }

    // Load GLSL shader source from file.  Editing the source files before
    // loading provides a way to debug and experiment with shader source code.
    if (key == GLFW_KEY_L && action == GLFW_PRESS)
    {
        loadShaderSource();
        return true;
    }

    // Clear the image cache, reloading all required images from the file system.
    if (key == GLFW_KEY_I && action == GLFW_PRESS && modifiers == GLFW_MOD_SHIFT)
    {
        _imageHandler->clearImageCache();
        return true;
    }

    // Save each node graph in the current material as a dot file.
    if (key == GLFW_KEY_D && action == GLFW_PRESS)
    {
        saveDotFiles();
        return true;
    }

    // Capture the current frame and save as an image file.
    if (key == GLFW_KEY_F && action == GLFW_PRESS)
    {
        _captureFilename = ng::file_dialog({ { mx::ImageLoader::PNG_EXTENSION, "Image File" } }, true);
        if (!_captureFilename.isEmpty())
        {
            if (_captureFilename.getExtension().empty())
            {
                _captureFilename.addExtension(mx::ImageLoader::PNG_EXTENSION);
            }
            _captureRequested = true;
        }
    }

    // Render a wedge for the current material.
    if (key == GLFW_KEY_W && action == GLFW_PRESS)
    {
        _wedgeFilename = ng::file_dialog({ { mx::ImageLoader::PNG_EXTENSION, "Image File" } }, true);
        if (!_wedgeFilename.isEmpty())
        {
            if (_wedgeFilename.getExtension().empty())
            {
                _wedgeFilename.addExtension(mx::ImageLoader::PNG_EXTENSION);
            }
            _wedgeRequested = true;
        }
    }

    // Request shader translation for the current material.
    if (key == GLFW_KEY_T && action == GLFW_PRESS)
    {
        mx::DocumentPtr translatedDoc = translateMaterial();
        if (translatedDoc)
        {
            mx::FilePath translatedFilename = getBaseOutputPath();
            translatedFilename = translatedFilename.asString() + "_" + _targetShader;
            translatedFilename.addExtension(mx::MTLX_EXTENSION);

            mx::XmlWriteOptions writeOptions;
            writeOptions.elementPredicate = getElementPredicate();
            mx::writeToXmlFile(translatedDoc, translatedFilename, &writeOptions);

            new ng::MessageDialog(this, ng::MessageDialog::Type::Information, "Saved translated material: ", translatedFilename);
        }
        return true;
    }

    // Request a bake of the current material.
    if (key == GLFW_KEY_B && action == GLFW_PRESS)
    {
        mx::FilePath filename = ng::file_dialog({ { "mtlx", "MaterialX" } }, true);
        if (!filename.isEmpty())
        {
            if (filename.getExtension() != mx::MTLX_EXTENSION)
            {
                filename.addExtension(mx::MTLX_EXTENSION);
            }
            _bakeRequested = true;
            _bakeFilename = filename;
        }
    }

    // Up and down keys cycle through selected geometries.
    if ((key == GLFW_KEY_UP || key == GLFW_KEY_DOWN) && action == GLFW_PRESS)
    {
        if (_geometryList.size() > 1)
        {
            if (key == GLFW_KEY_DOWN)
            {
                _selectedGeom = (_selectedGeom < _geometryList.size() - 1) ? _selectedGeom + 1 : 0;
            }
            else
            {
                _selectedGeom = (_selectedGeom > 0) ? _selectedGeom - 1 : _geometryList.size() - 1;
            }
        }
        _geometrySelectionBox->set_selected_index((int) _selectedGeom);
        updateMaterialSelectionUI();
        return true;
    }

    // Left and right keys cycle through selected materials.
    if ((key == GLFW_KEY_LEFT || key == GLFW_KEY_RIGHT) && action == GLFW_PRESS)
    {
        if (_materials.size() > 1)
        {
            if (key == GLFW_KEY_RIGHT)
            {
                _selectedMaterial = (_selectedMaterial < _materials.size() - 1) ? _selectedMaterial + 1 : 0;
            }
            else
            {
                _selectedMaterial = (_selectedMaterial > 0) ? _selectedMaterial - 1 : _materials.size() - 1;
            }
            assignMaterial(getSelectedGeometry(), getSelectedMaterial());
            updateMaterialSelectionUI();
        }
        return true;
    }

    if (key == GLFW_KEY_U && action == GLFW_PRESS)
    {
        _window->set_visible(!_window->visible());
    }

    return false;
}

mx::ImagePtr Viewer::renderWedge()
{
    mx::MaterialPtr material = getSelectedMaterial();
    mx::ShaderPort* uniform = material ? material->findUniform(_wedgePropertyName) : nullptr;
    if (!uniform)
    {
        new ng::MessageDialog(this, ng::MessageDialog::Type::Warning, "Material property not found", _wedgePropertyName);
        return nullptr;
    }

    mx::ValuePtr origPropertyValue = uniform->getValue();
    if (origPropertyValue)
    {
        if (!origPropertyValue->isA<int>() && !origPropertyValue->isA<float>() &&
            !origPropertyValue->isA<mx::Vector2>() &&
            !origPropertyValue->isA<mx::Color3>() && !origPropertyValue->isA<mx::Vector3>() &&
            !origPropertyValue->isA<mx::Color4>() && !origPropertyValue->isA<mx::Vector4>())
        {
            new ng::MessageDialog(this, ng::MessageDialog::Type::Warning, "Material property type not supported", _wedgePropertyName);
            return nullptr;
        }

        std::vector<mx::ImagePtr> imageVec;
        float wedgePropertyStep = (_wedgePropertyMax - _wedgePropertyMin) / (_wedgeImageCount - 1);
        for (unsigned int i = 0; i < _wedgeImageCount; i++)
        {
            bool setValue = false;
            float propertyValue = (i == _wedgeImageCount - 1) ? _wedgePropertyMax : _wedgePropertyMin + wedgePropertyStep * i;
            if (origPropertyValue->isA<int>())
            {
                material->modifyUniform(_wedgePropertyName, mx::Value::createValue((int) propertyValue));
                setValue = true;
            }
            else if (origPropertyValue->isA<float>())
            {
                material->modifyUniform(_wedgePropertyName, mx::Value::createValue(propertyValue));
                setValue = true;
            }
            else if (origPropertyValue->isA<mx::Vector2>())
            {
                material->modifyUniform(_wedgePropertyName, mx::Value::createValue(mx::Vector2(propertyValue)));
                setValue = true;
            }
            else if (origPropertyValue->isA<mx::Color3>() ||
                     origPropertyValue->isA<mx::Vector3>())
            {
                material->modifyUniform(_wedgePropertyName, mx::Value::createValue(mx::Vector3(propertyValue)));
                setValue = true;
            }
            else if (origPropertyValue->isA<mx::Color4>() ||
                     origPropertyValue->isA<mx::Vector4>())
            {
                mx::Vector4 val(propertyValue, propertyValue, propertyValue, origPropertyValue->isA<mx::Color4>() ? 1.0f : propertyValue);
                material->modifyUniform(_wedgePropertyName, mx::Value::createValue(val));
                setValue = true;
            }
            if (setValue)
            {
#ifdef MATERIALXVIEW_METAL_BACKEND
                _colorTexture = metal_texture();
#endif
                if (!_geometryList.empty() && !_materialAssignments.empty())
                {
                    _renderPipeline->renderFrame(_colorTexture,
                                                 SHADOW_MAP_SIZE,
                                                 DIR_LIGHT_NODE_CATEGORY.c_str());
                }

                imageVec.push_back(_renderPipeline->getFrameImage());
            }
        }

        material->modifyUniform(_wedgePropertyName, origPropertyValue);

        return mx::createImageStrip(imageVec);
    }
    return nullptr;
}

void Viewer::renderTurnable()
{
    int frameCount = abs(_turntableSteps);

    float currentRotation = _meshRotation[1];
    _meshRotation[1] = 0.0f;
    int currentTurntableStep = _turntableStep;

    mx::FilePath turnableFileName = _captureFilename;
    const std::string extension = turnableFileName.getExtension();
    turnableFileName.removeExtension();

    for (_turntableStep = 0; _turntableStep < frameCount; _turntableStep++)
    {
        updateCameras();
        clear();
        invalidateShadowMap();
        _renderPipeline->renderFrame(_colorTexture,
                                     SHADOW_MAP_SIZE,
                                     DIR_LIGHT_NODE_CATEGORY.c_str());

        mx::ImagePtr frameImage = _renderPipeline->getFrameImage();
        if (frameImage)
        {
            std::stringstream intfmt;
            intfmt << std::setfill('0') << std::setw(4) << _turntableStep;
            std::string saveName = turnableFileName.asString() + "_" + intfmt.str() + "." + extension;
            if (_imageHandler->saveImage(saveName, frameImage, true))
            {
                std::cout << "Wrote turntable frame to file: " << saveName << std::endl;
            }
        }
    }

    _turntableStep = currentTurntableStep;
    _meshRotation[1] = currentRotation;
}

void Viewer::renderScreenSpaceQuad(mx::MaterialPtr material)
{
    if (!_quadMesh)
    {
        _quadMesh = mx::GeometryHandler::createQuadMesh();
    }
    
    material->bindMesh(_quadMesh);
    material->drawPartition(_quadMesh->getPartition(0));
}

void Viewer::draw_contents()
{
    updateCameras();

#ifndef MATERIALXVIEW_METAL_BACKEND
    mx::checkGlErrors("before viewer render");

    // Clear the screen.
    clear();
#endif

    if (_turntableEnabled && _turntableSteps)
    {
        if (!_captureRequested)
        {
            const double updateTime = 1.0 / 24.0;
            if (_turntableTimer.elapsedTime() > updateTime)
            {
                _turntableStep++;
                _turntableTimer.startTimer();
                invalidateShadowMap();
            }
        }
        else
        {
            _captureRequested = false;
            renderTurnable();
        }
    }

    // Render a wedge for the current material.
    if (_wedgeRequested)
    {
        _wedgeRequested = false;
        mx::ImagePtr wedgeImage = renderWedge();
        if (wedgeImage && _imageHandler->saveImage(_wedgeFilename, wedgeImage, true))
        {
            std::cout << "Wrote wedge to disk: " << _wedgeFilename.asString() << std::endl;
        }
        else
        {
            new ng::MessageDialog(this, ng::MessageDialog::Type::Information,
                "Failed to write wedge to disk: ", _wedgeFilename.asString());
        }
    }

    // Render the current frame.
    try
    {
        _renderPipeline->renderFrame(_colorTexture,
                                     SHADOW_MAP_SIZE,
                                     DIR_LIGHT_NODE_CATEGORY.c_str());
    }
    catch (std::exception& e)
    {
        new ng::MessageDialog(this, ng::MessageDialog::Type::Warning,
            "Failed to render frame: ", e.what());
        _materialAssignments.clear();
#ifndef MATERIALXVIEW_METAL_BACKEND
        glDisable(GL_FRAMEBUFFER_SRGB);
#endif
    }

    // Capture the current frame.
    if (_captureRequested && !_turntableEnabled)
    {
        _captureRequested = false;
        mx::ImagePtr frameImage = _renderPipeline->getFrameImage();
        if (frameImage && _imageHandler->saveImage(_captureFilename, frameImage, true))
        {
            std::cout << "Wrote frame to disk: " << _captureFilename.asString() << std::endl;
        }
        else
        {
            new ng::MessageDialog(this, ng::MessageDialog::Type::Information,
                "Failed to write frame to disk: ", _captureFilename.asString());
        }
    }

    // Bake textures for the current material.
    if (_bakeRequested)
    {
        _bakeRequested = false;
        bakeTextures();
    }

    // Handle exit requests.
    if (_exitRequested)
    {
        ng::leave();
        set_visible(false);
    }

#ifndef MATERIALXVIEW_METAL_BACKEND
    mx::checkGlErrors("after viewer render");
#endif
}

bool Viewer::scroll_event(const ng::Vector2i& p, const ng::Vector2f& rel)
{
    if (Screen::scroll_event(p, rel))
    {
        return true;
    }

    if (_userCameraEnabled)
    {
        _cameraZoom = std::max(0.1f, _cameraZoom * ((rel.y() > 0) ? 1.1f : 0.9f));
        return true;
    }

    return false;
}

bool Viewer::mouse_motion_event(const ng::Vector2i& p,
                                const ng::Vector2i& rel,
                                int button,
                                int modifiers)
{
    if (Screen::mouse_motion_event(p, rel, button, modifiers))
    {
        return true;
    }

    mx::Vector2 pos((float) p.x(), (float) p.y());
    if (_viewCamera->applyArcballMotion(pos))
    {
        return true;
    }

    if (_userTranslationActive)
    {
        updateCameras();

        mx::Vector3 boxMin = _geometryHandler->getMinimumBounds();
        mx::Vector3 boxMax = _geometryHandler->getMaximumBounds();
        mx::Vector3 sphereCenter = (boxMax + boxMin) / 2.0f;

        float viewZ = _viewCamera->projectToViewport(sphereCenter)[2];
        mx::Vector3 pos1 = _viewCamera->unprojectFromViewport(
            mx::Vector3(pos[0], (float) m_size.y() - pos[1], viewZ));
        mx::Vector3 pos0 = _viewCamera->unprojectFromViewport(
            mx::Vector3(_userTranslationPixel[0], (float) m_size.y() - _userTranslationPixel[1], viewZ));
        mx::Vector3 translation = (pos1 - pos0);
#ifdef MATERIALXVIEW_METAL_BACKEND
        translation[1] = -translation[1];
#endif
        _userTranslation = _userTranslationStart + translation;

        return true;
    }

    return false;
}

bool Viewer::mouse_button_event(const ng::Vector2i& p, int button, bool down, int modifiers)
{
    if (Screen::mouse_button_event(p, button, down, modifiers))
    {
        return true;
    }

    mx::Vector2 pos((float) p.x(), (float) p.y());
    if (button == GLFW_MOUSE_BUTTON_1 && !modifiers)
    {
        _viewCamera->arcballButtonEvent(pos, down);
    }
    else if (button == GLFW_MOUSE_BUTTON_2 ||
            (button == GLFW_MOUSE_BUTTON_1 && modifiers == GLFW_MOD_SHIFT))
    {
        _userTranslationStart = _userTranslation;
        _userTranslationActive = true;
        _userTranslationPixel = pos;
    }
    if (button == GLFW_MOUSE_BUTTON_1 && !down)
    {
        _viewCamera->arcballButtonEvent(pos, false);
    }
    if (!down)
    {
        _userTranslationActive = false;
    }
    return true;
}

void Viewer::initCamera()
{
    _viewCamera->setViewportSize(mx::Vector2(static_cast<float>(m_size[0]), static_cast<float>(m_size[1])));

    // Disable user camera controls when non-centered views are requested.
    _userCameraEnabled = _cameraTarget == mx::Vector3(0.0) &&
                         _meshScale == 1.0f;

    if (!_userCameraEnabled || _geometryHandler->getMeshes().empty())
    {
        return;
    }

    const mx::Vector3& boxMax = _geometryHandler->getMaximumBounds();
    const mx::Vector3& boxMin = _geometryHandler->getMinimumBounds();
    mx::Vector3 sphereCenter = (boxMax + boxMin) * 0.5;

    float turntableRotation = fmod((360.0f / _turntableSteps) * _turntableStep, 360.0f);
    float yRotation = _meshRotation[1] + (_turntableEnabled ? turntableRotation : 0.0f);
    mx::Matrix44 meshRotation = mx::Matrix44::createRotationZ(_meshRotation[2] / 180.0f * PI) *
                                mx::Matrix44::createRotationY(yRotation / 180.0f * PI) *
                                mx::Matrix44::createRotationX(_meshRotation[0] / 180.0f * PI);
    _meshTranslation = -meshRotation.transformPoint(sphereCenter);
    _meshScale = IDEAL_MESH_SPHERE_RADIUS / (sphereCenter - boxMin).getMagnitude();
}

void Viewer::updateCameras()
{
#ifdef MATERIALXVIEW_METAL_BACKEND
    auto& createPerspectiveMatrix = mx::Camera::createPerspectiveMatrixZP;
    auto& createOrthographicMatrix = mx::Camera::createOrthographicMatrixZP;
#else
    auto& createPerspectiveMatrix = mx::Camera::createPerspectiveMatrix;
    auto& createOrthographicMatrix = mx::Camera::createOrthographicMatrix;
#endif
    mx::Matrix44 viewMatrix, projectionMatrix;
    float aspectRatio = (float) m_size.x() / (float) m_size.y();
    if (_cameraViewAngle != 0.0f)
    {
        viewMatrix = mx::Camera::createViewMatrix(_cameraPosition, _cameraTarget, _cameraUp);
        float fH = std::tan(_cameraViewAngle / 360.0f * PI) * _cameraNearDist;
        float fW = fH * aspectRatio;
        projectionMatrix = createPerspectiveMatrix(-fW, fW, -fH, fH, _cameraNearDist, _cameraFarDist);
    }
    else
    {
        viewMatrix = mx::Matrix44::createTranslation(mx::Vector3(0.0f, 0.0f, -ORTHO_VIEW_DISTANCE));
        float fH = ORTHO_PROJECTION_HEIGHT;
        float fW = fH * aspectRatio;
        projectionMatrix = createOrthographicMatrix(-fW, fW, -fH, fH, 0.0f, ORTHO_VIEW_DISTANCE + _cameraFarDist);
    }
#ifdef MATERIALXVIEW_METAL_BACKEND
    projectionMatrix[1][1] = -projectionMatrix[1][1];
    _colorTexture = metal_texture();
#endif
    float turntableRotation = fmod((360.0f / _turntableSteps) * _turntableStep, 360.0f);
    float yRotation = _meshRotation[1] + (_turntableEnabled ? turntableRotation : 0.0f);
    mx::Matrix44 meshRotation = mx::Matrix44::createRotationZ(_meshRotation[2] / 180.0f * PI) *
                                mx::Matrix44::createRotationY(yRotation / 180.0f * PI) *
                                mx::Matrix44::createRotationX(_meshRotation[0] / 180.0f * PI);

    mx::Matrix44 arcball = mx::Matrix44::IDENTITY;
    if (_userCameraEnabled)
    {
        arcball = _viewCamera->arcballMatrix();
    }

    _viewCamera->setWorldMatrix(meshRotation *
                                mx::Matrix44::createTranslation(_meshTranslation + _userTranslation) *
                                mx::Matrix44::createScale(mx::Vector3(_meshScale * _cameraZoom)));
    _viewCamera->setViewMatrix(arcball * viewMatrix);
    _viewCamera->setProjectionMatrix(projectionMatrix);

    _envCamera->setWorldMatrix(mx::Matrix44::createScale(mx::Vector3(300.0f)));
    _envCamera->setViewMatrix(_viewCamera->getViewMatrix());
    _envCamera->setProjectionMatrix(_viewCamera->getProjectionMatrix());

    mx::NodePtr dirLight = _lightHandler->getFirstLightOfCategory(DIR_LIGHT_NODE_CATEGORY);
    if (dirLight)
    {
        mx::Vector3 sphereCenter = (_geometryHandler->getMaximumBounds() + _geometryHandler->getMinimumBounds()) * 0.5;
        float r = (sphereCenter - _geometryHandler->getMinimumBounds()).getMagnitude();
        _shadowCamera->setWorldMatrix(meshRotation * mx::Matrix44::createTranslation(-sphereCenter));
        _shadowCamera->setProjectionMatrix(mx::Camera::createOrthographicMatrixZP(-r, r, -r, r, 0.0f, r * 2.0f));
        mx::ValuePtr value = dirLight->getInputValue("direction");
        if (value->isA<mx::Vector3>())
        {
            mx::Vector3 dir = mx::Matrix44::createRotationY(_lightRotation / 180.0f * PI).transformVector(value->asA<mx::Vector3>());
            _shadowCamera->setViewMatrix(mx::Camera::createViewMatrix(dir * -r, mx::Vector3(0.0f), _cameraUp));
        }
    }
}

void Viewer::updateDisplayedProperties()
{
    _propertyEditor.updateContents(this);
    createSaveMaterialsInterface(_propertyEditor.getWindow(), "Save Material");
    perform_layout();
}

mx::ImagePtr Viewer::getAmbientOcclusionImage(mx::MaterialPtr material)
{
    const mx::string AO_FILENAME_SUFFIX = "_ao";
    const mx::string AO_FILENAME_EXTENSION = "png";

    if (!material || !_genContext.getOptions().hwAmbientOcclusion)
    {
        return nullptr;
    }

    std::string aoSuffix = material->getUdim().empty() ? AO_FILENAME_SUFFIX : AO_FILENAME_SUFFIX + "_" + material->getUdim();
    mx::FilePath aoFilename = _meshFilename;
    aoFilename.removeExtension();
    aoFilename = aoFilename.asString() + aoSuffix;
    aoFilename.addExtension(AO_FILENAME_EXTENSION);
    return _imageHandler->acquireImage(aoFilename);
}

void Viewer::splitDirectLight(mx::ImagePtr envRadianceMap, mx::ImagePtr& indirectMap, mx::DocumentPtr& dirLightDoc)
{
    mx::Vector3 lightDir;
    mx::Color3 lightColor;
    mx::ImagePair imagePair = envRadianceMap->splitByLuminance(ENV_MAP_SPLIT_RADIANCE);

    mx::computeDominantLight(imagePair.second, lightDir, lightColor);
    float lightIntensity = std::max(std::max(lightColor[0], lightColor[1]), lightColor[2]);
    if (lightIntensity)
    {
        lightColor /= lightIntensity;
    }

    dirLightDoc = mx::createDocument();
    mx::NodePtr dirLightNode = dirLightDoc->addNode(DIR_LIGHT_NODE_CATEGORY, "dir_light", mx::LIGHT_SHADER_TYPE_STRING);
    dirLightNode->setInputValue("direction", lightDir);
    dirLightNode->setInputValue("color", lightColor);
    dirLightNode->setInputValue("intensity", lightIntensity);
    indirectMap = imagePair.first;
}

mx::MaterialPtr Viewer::getEnvironmentMaterial()
{
    if (!_envMaterial)
    {
        mx::FilePath envFilename = _searchPath.find(mx::FilePath("resources/Lights/environment_map.mtlx"));
        try
        {
            _envMaterial = _renderPipeline->createMaterial();
            _envMaterial->generateEnvironmentShader(_genContext, envFilename, _stdLib, _envRadianceFilename);
        }
        catch (std::exception& e)
        {
            std::cerr << "Failed to generate environment shader: " << e.what() << std::endl;
            _envMaterial = nullptr;
        }
    }

    return _envMaterial;
}

mx::MaterialPtr Viewer::getWireframeMaterial()
{
    if (!_wireMaterial)
    {
        try
        {
            mx::ShaderPtr hwShader = mx::createConstantShader(_genContext, _stdLib, "__WIRE_SHADER__", mx::Color3(1.0f));
            _wireMaterial = _renderPipeline->createMaterial();
            _wireMaterial->generateShader(hwShader);
        }
        catch (std::exception& e)
        {
            std::cerr << "Failed to generate wireframe shader: " << e.what() << std::endl;
            _wireMaterial = nullptr;
        }
    }

    return _wireMaterial;
}

void Viewer::invalidateShadowMap()
{
    if (_shadowMap)
    {
        _imageHandler->releaseRenderResources(_shadowMap);
        _shadowMap = nullptr;
    }
}

void Viewer::toggleTurntable(bool enable)
{
    _turntableEnabled = enable;

    if (enable)
    {
        _turntableTimer.startTimer();
    }
    else
    {
        float turntableRotation = fmod((360.0f / _turntableSteps) * _turntableStep, 360.0f);
        _meshRotation[1] = fmod(_meshRotation[1] + turntableRotation, 360.0f);
        _turntableTimer.endTimer();
    }
    invalidateShadowMap();
    _turntableStep = 0;
}

void Viewer::setShaderInterfaceType(mx::ShaderInterfaceType interfaceType)
{
    _genContext.getOptions().shaderInterfaceType = interfaceType;
#ifndef MATERIALXVIEW_METAL_BACKEND
    _genContextEssl.getOptions().shaderInterfaceType = interfaceType;
#endif
#if MATERIALX_BUILD_GEN_OSL
    _genContextOsl.getOptions().shaderInterfaceType = interfaceType;
#endif
#if MATERIALX_BUILD_GEN_MDL
    _genContextMdl.getOptions().shaderInterfaceType = interfaceType;
#endif
    reloadShaders();
    updateDisplayedProperties();
}
