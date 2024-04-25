# MaterialX Viewer

The MaterialX Viewer leverages shader generation to build GLSL shaders from MaterialX graphs, rendering the results using the NanoGUI framework.  The standard set of pattern and physically based shading nodes is supported, and libraries of custom nodes can be included as additional library paths.

### Example Images

**Figure 1:** Procedural and uniform materials in the MaterialX viewer
<p float="left">
  <img src="/documents/Images/MaterialXView_Marble.png" width="248" />
  <img src="/documents/Images/MaterialXView_Copper.png" width="248" />
  <img src="/documents/Images/MaterialXView_Plastic.png" width="248" />
  <img src="/documents/Images/MaterialXView_Carpaint.png" width="248" />
</p>

**Figure 2:** Textured, color-space-managed materials in the MaterialX viewer
<p float="left">
  <img src="/documents/Images/MaterialXView_TiledBrass.png" width="500" />
  <img src="/documents/Images/MaterialXView_TiledWood.png" width="500" />
</p>

## Building The MaterialX Viewer
Select the `MATERIALX_BUILD_VIEWER` option in CMake to build the MaterialX Viewer.  Installation will copy the **MaterialXView** executable to a `/bin` directory within the selected install folder.

### Summary of Viewer Options

1.  **Load Mesh**: Load a new geometry in the OBJ or glTF format.
2.  **Load Material**: Load a material document in the MTLX format.
3.  **Load Environment**: Load a lat-long environment light in the HDR format.
4.  **Property Editor**: View or edit properties of the current material.
5.  **Advanced Settings** : Asset and rendering options.

### Geometry

The default display geometry for the MaterialX viewer is the Arnold Shader Ball, which was contributed to the MaterialX project by the Solid Angle team at Autodesk.  To change the display geometry, click `Load Mesh` and navigate to the [Geometry](../../resources/Geometry) folder for additional models in the OBJ format.

If a loaded geometry contains more than one geometric group, then a `Select Geometry` drop-down box will appear, allowing the user to select which group is active.  The active geometric group will be used for subsequent actions such as material assignment and rendering property changes.

### Materials

To change the displayed material, click `Load Material` and navigate to the [Materials/Examples/StandardSurface](../../resources/Materials/Examples/StandardSurface) or [Materials/Examples/UsdPreviewSurface](../../resources/Materials/Examples/UsdPreviewSurface) folders, which contain a selection of example materials in the MTLX format.

Once a material is loaded into the viewer, its parameters may be inspected and adjusted by clicking the `Property Editor` and scrolling through the list of parameters.  An edited material may be saved to the file system by clicking `Save Material`.

Multiple material documents can be combined in a single session by navigating to `Advanced Settings` and enabling `Merge Materials`.  Loading new materials with this setting enabled will add them to the current material list, where they can be assigned to geometry via the `Assigned Material` drop-down box.  Alternatively the `LEFT` and `RIGHT` arrows can be used to cycle through the list of available materials.

If a material document containing `look` elements is loaded into the viewer, then any material assignments within the look will be applied to geometric groups that match the specified geometry strings.  See [standard_surface_look_brass_tiled.mtlx](../../resources/Materials/Examples/StandardSurface/standard_surface_look_brass_tiled.mtlx) for an example of a material document containing look elements.

### Lighting

The default lighting environment for the viewer is the San Giuseppe Bridge environment from HDRI Haven.  To load another environment into the viewer, click `Load Environment` and navigate to the [Lights](../../resources/Lights) folder, or load any HDR environment in the latitude-longitude format.  If the HDR file on disk has a companion MaterialX document with a matching name, then this document will be loaded as the direct lighting rig for the environment; otherwise only indirect lighting will be rendered.  If the HDR file on disk has a companion image in an `irradiance` subfolder, then this image will be loaded as the diffuse convolution of the environment; otherwise, a diffuse convolution will be generated at load-time using spherical harmonics.

Shadow maps from the primary directional light may be enabled with the `Shadow Map` option under `Advanced Settings`.  Ambient occlusion, if available for the given geometry, may be enabled with the `Ambient Occlusion` option.  The fidelity of environment lighting may be improved by increasing the value of `Environment Samples`, though this requires additional GPU resources and can affect the interactivity of the viewer.

### Images

By default, the MaterialX viewer loads and saves image files using `stb_image`, which supports commmon 8-bit formats such as JPEG, PNG, TGA, and BMP, as well as the HDR format for high-dynamic-range images.  If you need access to additional image formats such as EXR and TIFF, then the MaterialX viewer can be built with support for `OpenImageIO`.  To build MaterialX with OpenImageIO, check the `MATERIALX_BUILD_OIIO` option in CMake, and specify the location of your OpenImageIO installation with the `MATERIALX_OIIO_DIR` option.

### Keyboard Shortcuts

- `R`: Reload the current material from file.  Hold `SHIFT` to reload all standard libraries as well.
- `G`: Save the current GLSL shader source to file.
- `O`: Save the current OSL shader source to file.
- `M`: Save the current MDL shader source to file.
- `L`: Load GLSL shader source from file.  Editing the source files before loading provides a way to debug and experiment with shader source code.
- `D`: Save each node graph in the current material as a DOT file.  See www.graphviz.org for more details on this format.
- `F`: Capture the current frame and save to file.
- `W`: Create a wedge rendering and save to file.  See `Advanced Settings` for additional controls.
- `T`: Translate the current material to a different shading model.  See `Advanced Settings` for additional controls.
- `B`: Bake the current material to textures.  See `Advanced Settings` for additional controls.
- `UP` : Select the previous geometry.
- `DOWN` : Select the next geometry.
- `RIGHT` : Switch to the next material.
- `LEFT` : Switch to the previous material.
- `+` : Zoom in with the camera.
- `-` : Zoom out with the camera.

### Command-Line Options

The following are common command-line options for MaterialXView, and a complete list can be displayed with the `--help` option.
- `--material [FILENAME]` : Specify the filename of the MTLX document to be displayed in the viewer
- `--mesh [FILENAME]` : Specify the filename of the OBJ or glTF mesh to be displayed in the viewer
- `--meshRotation [VECTOR3]` : Specify the rotation of the displayed mesh as three comma-separated floats, representing rotations in degrees about the X, Y, and Z axes (defaults to 0,0,0)
- `--meshScale [FLOAT]` : Specify the uniform scale of the displayed mesh
- `--cameraPosition [VECTOR3]` : Specify the position of the camera as three comma-separated floats (defaults to 0,0,5)
- `--cameraTarget [VECTOR3]` : Specify the position of the camera target as three comma-separated floats (defaults to 0,0,0)
- `--cameraViewAngle [FLOAT]` : Specify the view angle of the camera, or zero for an orthographic projection (defaults to 45)
- `--cameraZoom [FLOAT]` : Specify the zoom factor for the camera, implemented as a mesh scale multiplier (defaults to 1)
- `--envRad [FILENAME]` : Specify the filename of the environment light to display, stored as HDR environment radiance in the latitude-longitude format
- `--envMethod [INTEGER]` : Specify the environment lighting method (0 = filtered importance sampling, 1 = prefiltered environment maps, defaults to 0)
- `--envSampleCount [INTEGER]` :  Specify the environment sample count (defaults to 16)
- `--lightRotation [FLOAT]` : Specify the rotation in degrees of the lighting environment about the Y axis (defaults to 0)
- `--path [FILEPATH]` : Specify an additional data search path location (e.g. '/projects/MaterialX').  This absolute path will be queried when locating data libraries, XInclude references, and referenced images.
- `--library [FILEPATH]` : Specify an additional data library folder (e.g. 'vendorlib', 'studiolib').  This relative path will be appended to each location in the data search path when loading data libraries.
- `--screenWidth [INTEGER]` : Specify the width of the screen image in pixels (defaults to 1280)
- `--screenHeight [INTEGER]` : Specify the height of the screen image in pixels (defaults to 960)
- `--screenColor [VECTOR3]` : Specify the background color of the viewer as three comma-separated floats (defaults to 0.3,0.3,0.32)
- `--captureFilename [FILENAME]` : Specify the filename to which the first rendered frame should be written
- `--refresh [FLOAT]` : Specify the refresh period for the viewer in milliseconds (defaults to 50, set to -1 to disable)
- `--remap [TOKEN1:TOKEN2]` : Specify the remapping from one token to another when MaterialX document is loaded
- `--skip [NAME]` : Specify to skip elements matching the given name attribute
- `--terminator [STRING]` : Specify to enforce the given terminator string for file prefixes
- `--help` : Display the complete list of command-line options
