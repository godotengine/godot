//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALXVIEW_VIEWER_H
#define MATERIALXVIEW_VIEWER_H

#include <MaterialXView/Editor.h>
#include <MaterialXView/RenderPipeline.h>

#include <MaterialXRender/ShaderMaterial.h>
#include <MaterialXRender/Camera.h>
#include <MaterialXRender/GeometryHandler.h>
#include <MaterialXRender/LightHandler.h>
#include <MaterialXRender/ImageHandler.h>
#include <MaterialXRender/Timer.h>


#include <MaterialXCore/Unit.h>

namespace mx = MaterialX;
namespace ng = nanogui;

class DocumentModifiers
{
  public:
    mx::StringMap remapElements;
    mx::StringSet skipElements;
    std::string filePrefixTerminator;
};

class Viewer : public ng::Screen
{
    friend class RenderPipeline;
    friend class GLRenderPipeline;
    friend class MetalRenderPipeline;
  public:
    Viewer(const std::string& materialFilename,
           const std::string& meshFilename,
           const std::string& envRadianceFilename,
           const mx::FileSearchPath& searchPath,
           const mx::FilePathVec& libraryFolders,
           int screenWidth,
           int screenHeight,
           const mx::Color3& screenColor);
    ~Viewer() { }

    // Initialize the viewer for rendering.
    void initialize();

    // Set the rotation of the current mesh as Euler angles.
    void setMeshRotation(const mx::Vector3& rotation)
    {
        _meshRotation = rotation;
    }

    // Set the scale of the current mesh.
    void setMeshScale(float scale)
    {
        _meshScale = scale;
    }

    // Set whether turntable rendering is enabled.
    void setTurntableEnabled(bool val)
    {
        _turntableEnabled = val;
    }

    // Set the total number of steps for one 360 degree rotation.
    void setTurntableSteps(int steps)
    {
        _turntableSteps = steps;
    }

    // Set the world-space position of the camera.
    void setCameraPosition(const mx::Vector3& position)
    {
        _cameraPosition = position;
    }

    // Set the world-space target of the camera.
    void setCameraTarget(const mx::Vector3& target)
    {
        _cameraTarget = target;
    }

    // Set the view angle of the camera.
    void setCameraViewAngle(float angle)
    {
        _cameraViewAngle = angle;
    }

    // Set the zoom scale of the camera.
    void setCameraZoom(float zoom)
    {
        _cameraZoom = zoom;
    }

    // Set the method for specular environment rendering.
    void setSpecularEnvironmentMethod(mx::HwSpecularEnvironmentMethod method)
    {
        _genContext.getOptions().hwSpecularEnvironmentMethod = method;
    }

    // Set the number of environment samples.
    void setEnvSampleCount(int count)
    {
        _lightHandler->setEnvSampleCount(count);
    }

    // Set the environment light intensity.
    void setEnvLightIntensity(float intensity)
    {
        _lightHandler->setEnvLightIntensity(intensity);
    }

    // Set the rotation of the lighting environment about the Y axis.
    void setLightRotation(float rotation)
    {
        _lightRotation = rotation;
    }

    // Enable or disable shadow maps.
    void setShadowMapEnable(bool enable)
    {
        _genContext.getOptions().hwShadowMap = enable;
    }

    // Enable or disable drawing environment as the background.
    void setDrawEnvironment(bool enable)
    {
        _drawEnvironment = enable;
    }

    // Set the modifiers to be applied to loaded documents.
    void setDocumentModifiers(const DocumentModifiers& modifiers)
    {
        _modifiers = modifiers;
    }

    // Set the target width for texture baking.
    void setBakeWidth(unsigned int bakeWidth)
    {
        _bakeWidth = bakeWidth;
    }

    // Set the target height for texture baking.
    void setBakeHeight(unsigned int bakeHeight)
    {
        _bakeHeight = bakeHeight;
    }

    // Set the output document filename for texture baking.
    void setBakeFilename(const mx::FilePath& bakeFilename)
    {
        _bakeFilename = bakeFilename;
    }

    // Return true if all inputs should be shown in the property editor.
    bool getShowAllInputs() const
    {
        return _showAllInputs;
    }

    // Return the underlying NanoGUI window.
    ng::Window* getWindow() const
    {
        return _window;
    }

    // Return the active image handler.
    mx::ImageHandlerPtr getImageHandler() const
    {
        return _imageHandler;
    }

    // Return the selected material.
    mx::MaterialPtr getSelectedMaterial() const
    {
        if (_selectedMaterial < _materials.size())
        {
            return _materials[_selectedMaterial];
        }
        return nullptr;
    }

    // Return the selected mesh partition.
    mx::MeshPartitionPtr getSelectedGeometry() const
    {
        if (_selectedGeom < _geometryList.size())
        {
            return _geometryList[_selectedGeom];
        }
        return nullptr;
    }

    // Request a capture of the current frame, writing it to the given filename.
    void requestFrameCapture(const mx::FilePath& filename)
    {
        _captureRequested = true;
        _captureFilename = filename;
    }

    // Request that the viewer be closed after the next frame is rendered.
    void requestExit()
    {
        _exitRequested = true;
    }

    // Bake textures to disk using the current render pipeline.
    void bakeTextures()
    {
        _renderPipeline->bakeTextures();
    }

  private:
    void draw_contents() override;
    bool keyboard_event(int key, int scancode, int action, int modifiers) override;
    bool scroll_event(const ng::Vector2i& p, const ng::Vector2f& rel) override;
    bool mouse_motion_event(const ng::Vector2i& p, const ng::Vector2i& rel, int button, int modifiers) override;
    bool mouse_button_event(const ng::Vector2i& p, int button, bool down, int modifiers) override;
    
    void initContext(mx::GenContext& context);
    void loadMesh(const mx::FilePath& filename);
    void loadEnvironmentLight();
    void applyDirectLights(mx::DocumentPtr doc);
    void loadDocument(const mx::FilePath& filename, mx::DocumentPtr libraries);
    void reloadShaders();
    void loadStandardLibraries();
    void saveShaderSource(mx::GenContext& context);
    void loadShaderSource();
    void saveDotFiles();

    // Compute the resolution for texture baking.
    mx::UnsignedIntPair computeBakingResolution(mx::ConstDocumentPtr doc);
    
    // Translate the current material to the target shading model.
    mx::DocumentPtr translateMaterial();

    // Assign the given material to the given geometry, or remove any
    // existing assignment if the given material is nullptr.
    void assignMaterial(mx::MeshPartitionPtr geometry, mx::MaterialPtr material);

    // Mark the given material as currently selected in the viewer.
    void setSelectedMaterial(mx::MaterialPtr material)
    {
        for (size_t i = 0; i < _materials.size(); i++)
        {
            if (material == _materials[i])
            {
                _selectedMaterial = i;
                break;
            }
        }
    }

    // Generate a base output filepath for data derived from the current material.
    mx::FilePath getBaseOutputPath();

    // Return an element predicate for documents written from the viewer.
    mx::ElementPredicate getElementPredicate();

    void initCamera();
    void updateCameras();
    void updateGeometrySelections();
    void updateMaterialSelections();
    void updateMaterialSelectionUI();
    void updateDisplayedProperties();

    void createLoadMeshInterface(Widget* parent, const std::string& label);
    void createLoadMaterialsInterface(Widget* parent, const std::string& label);
    void createLoadEnvironmentInterface(Widget* parent, const std::string& label);
    void createSaveMaterialsInterface(Widget* parent, const std::string& label);
    void createPropertyEditorInterface(Widget* parent, const std::string& label);
    void createAdvancedSettings(Widget* parent);

    // Return the ambient occlusion image, if any, associated with the given material.
    mx::ImagePtr getAmbientOcclusionImage(mx::MaterialPtr material);
    
    // Split the given radiance map into indirect and direct components,
    // returning a new indirect map and directional light document.
    void splitDirectLight(mx::ImagePtr envRadianceMap, mx::ImagePtr& indirectMap, mx::DocumentPtr& dirLightDoc);

    mx::MaterialPtr getEnvironmentMaterial();
    mx::MaterialPtr getWireframeMaterial();

    mx::ImagePtr getShadowMap();
    void invalidateShadowMap();

    mx::ImagePtr renderWedge();
    void renderTurnable();
    void renderScreenSpaceQuad(mx::MaterialPtr material);

    // Update the directional albedo table.
    void updateAlbedoTable();

    // Toggle turntable
    void toggleTurntable(bool enable);

    // Set shader interface type
    void setShaderInterfaceType(mx::ShaderInterfaceType interfaceType);

  private:
    ng::Window* _window;
    RenderPipelinePtr _renderPipeline;

    mx::FilePath _materialFilename;
    mx::FileSearchPath _materialSearchPath;
    mx::FilePath _meshFilename;
    mx::FilePath _envRadianceFilename;

    mx::FileSearchPath _searchPath;
    mx::FilePathVec _libraryFolders;

    mx::Vector3 _meshTranslation;
    mx::Vector3 _meshRotation;
    float _meshScale;

    bool _turntableEnabled;
    int _turntableSteps;
    int _turntableStep;
    mx::ScopedTimer _turntableTimer;

    mx::Vector3 _cameraPosition;
    mx::Vector3 _cameraTarget;
    mx::Vector3 _cameraUp;
    float _cameraViewAngle;
    float _cameraNearDist;
    float _cameraFarDist;
    float _cameraZoom;

    bool _userCameraEnabled;
    mx::Vector3 _userTranslation;
    mx::Vector3 _userTranslationStart;
    bool _userTranslationActive;
    mx::Vector2 _userTranslationPixel;

    // Document management
    mx::DocumentPtr _stdLib;
    DocumentModifiers _modifiers;
    mx::StringSet _xincludeFiles;

    // Lighting information
    mx::FilePath _lightRigFilename;
    mx::DocumentPtr _lightRigDoc;
    float _lightRotation;

    // Light processing options
    bool _normalizeEnvironment;
    bool _splitDirectLight;
    bool _generateReferenceIrradiance;
    bool _saveGeneratedLights;

    // Shadow mapping
    mx::MaterialPtr _shadowMaterial;
    mx::MaterialPtr _shadowBlurMaterial;
    mx::ImagePtr _shadowMap;
    unsigned int _shadowSoftness;

    // Ambient occlusion
    float _ambientOcclusionGain;

    // Geometry selections
    std::vector<mx::MeshPartitionPtr> _geometryList;
    size_t _selectedGeom;
    ng::Label* _geomLabel;
    ng::ComboBox* _geometrySelectionBox;

    // Material selections
    std::vector<mx::MaterialPtr> _materials;
    mx::MaterialPtr _wireMaterial;
    size_t _selectedMaterial;
    ng::Label* _materialLabel;
    ng::ComboBox* _materialSelectionBox;
    PropertyEditor _propertyEditor;

    // Material assignments
    std::map<mx::MeshPartitionPtr, mx::MaterialPtr> _materialAssignments;

    // Cameras
    mx::CameraPtr _identityCamera;
    mx::CameraPtr _viewCamera;
    mx::CameraPtr _envCamera;
    mx::CameraPtr _shadowCamera;

    // Resource handlers
    mx::GeometryHandlerPtr _geometryHandler;
    mx::ImageHandlerPtr _imageHandler;
    mx::LightHandlerPtr _lightHandler;

    // Supporting materials and geometry.
    mx::GeometryHandlerPtr _envGeometryHandler;
    mx::MaterialPtr _envMaterial;
    mx::MeshPtr _quadMesh;

    // Shader generator contexts
    mx::GenContext _genContext;
#ifndef MATERIALXVIEW_METAL_BACKEND
    mx::GenContext _genContextEssl;
#endif
#if MATERIALX_BUILD_GEN_OSL
    mx::GenContext _genContextOsl;
#endif
#if MATERIALX_BUILD_GEN_MDL
    mx::GenContext _genContextMdl;
#endif
    // Unit registry
    mx::UnitConverterRegistryPtr _unitRegistry;

    // Viewing options
    bool _drawEnvironment;
    bool _outlineSelection;

    // Render options
    bool _renderTransparency;
    bool _renderDoubleSided;
    
    // Framebuffer Color Texture
    void* _colorTexture;

    // Scene options
    mx::StringVec _distanceUnitOptions;
    mx::LinearUnitConverterPtr _distanceUnitConverter;

    // Mesh loading options
    bool _splitByUdims;

    // Material loading options
    bool _mergeMaterials;
    bool _showAllInputs;
    bool _flattenSubgraphs;

    // Shader translation
    std::string _targetShader;

    // Frame capture
    bool _captureRequested;
    mx::FilePath _captureFilename;
    bool _exitRequested;

    // Wedge rendering
    bool _wedgeRequested;
    mx::FilePath _wedgeFilename;
    std::string _wedgePropertyName;
    float _wedgePropertyMin;
    float _wedgePropertyMax;
    unsigned int _wedgeImageCount;

    // Texture baking
    bool _bakeHdr;
    bool _bakeAverage;
    bool _bakeOptimize;
    bool _bakeRequested;
    unsigned int _bakeWidth;
    unsigned int _bakeHeight;
    bool _bakeDocumentPerMaterial;
    mx::FilePath _bakeFilename;
};

extern const mx::Vector3 DEFAULT_CAMERA_POSITION;
extern const float DEFAULT_CAMERA_VIEW_ANGLE;
extern const float DEFAULT_CAMERA_ZOOM;

#endif // MATERIALXVIEW_VIEWER_H
