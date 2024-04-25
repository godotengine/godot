<!-----
README for MaterialX Specification v1.39
----->

**MaterialX** is an open standard for representing rich material and look-development content in computer graphics, enabling its platform-independent description and exchange across applications and renderers.  MaterialX addresses the need for a common, open standard to represent the data values and relationships required to describe the look of a computer graphics model, including shading networks, patterns and texturing, complex nested materials and geometric assignments. To further encourage interchangeable CG look setups, MaterialX also defines a large set of standard shading and processing nodes with a precise mechanism for functional extensibility.

The documents in this folder comprise the complete MaterialX Specification, version 1.39.

* [**MaterialX Specification**](./MaterialX.Specification.md) - the main Specification, describing definitions, core functionality and the standard node library
* [**MaterialX Physically Based Shading Nodes**](./MaterialX.PBRSpec.md) - describes BSDF and other shading function nodes useful in constructing complex layered rendering shaders using node graphs
* [**MaterialX Geometry Extensions**](./MaterialX.GeomExts.md) - additional MaterialX elements to define geometry-related information such as collections, properties and material assignments
* [**MaterialX Supplemental Notes**](./MaterialX.Supplement.md) - describes recommended naming and structuring conventions for libraries of custom node definitions

<p>

---


**MaterialX v1.39** provides the following enhancements over v1.38:


**MaterialX Geometry Extensions**

The parts of the main MaterialX Specification document dealing with various Geometry-related features has now been split into a separate [**MaterialX Geometry Extensions**](./MaterialX.GeomExts.md) document, describing Collections, Geometry Name Expressions, geometry-related data types, Geometry Info elements and the GeomProp and Token elements used within them, and Look, Property, Visibility and assignment elements.

With this split, applications can claim to be MaterialX Compatible if they support all the things described in the main Specification, e.g. the elements for nodegraph shading networks and materials as well as the standard set of nodes, while using an application's native mechanisms or something like USD to describe the assignment of these materials to geometry.  Applications may additionally support the MaterialX Geometry Extensions and thus use a single unified representation for complete CG objecct looks.


**New Support for Shader AOVs**

Previously, MaterialX used custom types with a structure of output variables to define shader AOVs.  But this approach was not very flexible and in fact had not been implemented.  In v1.39, nodegraph-based shader implementations can include new [&lt;aovoutput> elements](./MaterialX.Specification.md#aov-output-elements) to define AOVs which renderers can use to output additional channels of information in addition to the final shading result, while file-based &lt;implementation>s can similarly define AOVs using [&lt;aov> elements](./MaterialX.Specification.md#implementation-aov-elements).


**Array Types Now Uniform and Static Length**

Many shading languages do not support dynamic array types with a variable length, so MaterialX now only supports arrays with a fixed maximum length, and all array-type node inputs must be uniform; nodes are no longer permitted to output an array type.  Array-type inputs may be accompanied by a uniform integer input declaring the number of array elements actually used in the array (the &lt;curveadjust> node has been updated in this way).  Because of this change, the unimplemented &lt;arrayappend> node has been removed.


**Connectable Uniform Inputs and New Tokenvalue Node**

A uniform node input is now explicitly allowed to be connected to the output of a &lt;constant> node.  This makes it possible to define a uniform value and use it in multiple places in a nodegraph.

Similarly, &lt;token>s in materials and other node instances may now be connected to the output of a new &lt;tokenvalue> node: this is essentially a &lt;constant> node but which connects to &lt;token>s rather than &lt;input>s.


**Standardized Color Space Names**

The [standard colorspace names](./MaterialX.Specification.md#color-spaces-and-color-management-systems) in MaterialX have now been defined explicitly in the Specification, and are aligned to their definitions in the ACES 1.2 OCIO config file.  With this change, there is no need for a definition of "cms" or "cmsconfig" in MaterialX documents, so those two attributes have been deprecated.  Additionally, two new colorspaces, "srgb_displayp3" and "lin_displayp3" have been added as standard colorspaces.


**Disambiguated Nodedef and Nodegraph References**

Normally, the set of provided inputs to a node and their types in conjunction with the output type of the node itself is sufficient to disambiguate exactly which nodedef signature should be applied.  In the rare situations where this is not sufficient, it is now permissible for any node instantiation to specify the name of a nodedef to completely disambiguate the intended node signature.

Additionally, a &lt;nodegraph> could previously declare itself to be an implementation of a particular &lt;nodedef> by providing a "nodedef" attribute, which is still the preferred method for making this association.  Now, it is also permissible for an [&lt;implementation> element](39/MaterialX.Specification.md#custom-node-definition-using-implementation-elements) to provide a "nodegraph" attribute to declare that nodegraph to be the implementation for the nodedef specified in the &lt;implementation>.  This allows a single nodegraph to be the implementation of multiple nodedefs, e.g. two different node names with the same underlying implementation, or if the only difference between two versions of a nodedef is the default values.


**Generalized Swizzle Operator Removed**

The standard &lt;swizzle> node using a string of channel names and allowing arbitrary channel reordering is very inefficient (and in some shading languages virtually impossible) to implement as previously specified, and as such has been removed.  Nodegraphs should instead use combinations of &lt;extract> (which is now a standard node), &lt;separateN> and &lt;combineN> nodes to perform arbitrary channel reordering.  Additionally, the previous "channels" attribute for inputs which allowed arbitrary channel reordering and used string "swizzle" channel naming has been replaced with an integer "channel" attribute, allowing a float input to be connected to a specified channel number of a color<em>N</em> or vector<em>N</em> output.  This is both far more efficient to implement and more closely matches the conventions for connecting different input and output types available in modern DCCs.


**New Unlit Surface Shader and Standard Materials**

A new &lt;surface_unlit> node for unlit surfaces has been added to the standard library.

Additionally, the standard &lt;surfacematerial> material now supports both single- or double-sided surfaces with the addition of a separate `backsurface` input.


**Inheritance and Hints for Typedefs**

Typedefs may now inherit from other types, including built-in types, and may provide hints about their values such as floating-point precision.  These new "inherit" and "hint" attributes are themselves merely metadata hints about the types; applications and code generators are still expected to provide their own precise definitions for all custom types.


**New and Updated Standard Library Nodes**

In 1.39, we are removing the distinction between "standard nodes" and "supplemental nodes", and descriptions of both can now be found in the main Specification document.  Nodes that are implemented in the standard distribution using nodegraphs are annotated with "(NG)" in the Spec to differentiate them from nodes implemented in each rendering target's native shading language.

Additionally, the following new operator nodes have been added to the standard library:

* [Procedural nodes](./MaterialX.Specification.md#procedural-nodes): **tokenvalue**, **checkerboard**, **fractal2d**, **cellnoise1d**, **unifiednoise2d**, **unifiednoise3d**
* [Geometric nodes](./MaterialX.Specification.md#geometric-nodes): **bump**, **geompropvalueuniform**
* [Math nodes](./MaterialX.Specification.md#math-nodes): boolean **and**, **or**, **not**; **distance**, **transformcolor**, **creatematrix** and **triplanarblend**, as well as integer-output variants of **floor** and **ceil**
* [Adjustment nodes](./MaterialX.Specification.md#adjustment-nodes): **curveinversecubic**, **curveuniformlinear**, **curveuniformcubic** and **colorcorrect**
* [Conditional nodes](./MaterialX.Specification.md#conditional-nodes): boolean-output variants of **ifgreater**, **ifgreatereq** and **ifequal**; new **ifelse** node
* [Channel nodes](./MaterialX.Specification.md#channel-nodes): **extractrowvector** and **separatecolor4**


**New Physically Based Shading Nodes**

The following new standard physically based shading nodes have been added:

* [EDF nodes](./MaterialX.PBRSpec.md#edf-nodes): **generalized_schlick_edf**
* [Shader nodes](./MaterialX.PBRSpec.md#shader-nodes): **environment** (latlong environment light source)


**Other Changes**

* The &lt;member> element for &lt;typedef>s and the "member" attribute for inputs have been removed from the Specification, as they had never been implemented and it was not clear how they could be implemented generally.
* The "valuerange" and "valuecurve" attributes describing expressions and function curves have been removed, in favor of using the new &lt;curveinversecubic> / &lt;curveuniformcubic> / etc. nodes.
* The &lt;geomcolor>, &lt;geompropvalue> and &lt;geompropvalueuniform> nodes for color3/4-type values can now take a "colorspace" attribute to declare the colorspace of the property value.
* The &lt;cellnoise2d> and &lt;cellnoise3d> nodes now support vector<em>N</em> output types in addition to float output.
* The &lt;noise2d/3d>, &lt;fractal2d/3d>, &lt;cellnoise2d/3d> and &lt;worleynoise2d/3d> nodes now support a "period" input.
* The &lt;worleynoise2d> and &lt;worleynoise3d> nodes now support a number of different distance metrics.
* The &lt;time> node no longer has a "frames per second" input: the application is now always expected to generate the "current time in seconds" using an appropriate method.  The "fps" input was removed because variable-rate real-time applications have no static "fps", and it's generally not good to bake a situation-dependent value like fps into a shading network.
* A standard "tangent" space is now defined in addition to "model", "object" and "world" spaces, and the &lt;heighttonormal> node now accepts a uniform "space" input to define the space of the output normal vector.
* The &lt;switch> node now supports 10 inputs instead of just 5.
* The &lt;surface> and &lt;displacement> nodes are now part of the main Specification rather than being Physically Based Shading nodes.
* &lt;Token> elements are now explicitly allowed to be children of compound nodegraphs, and token values may now have defined enum/enumvalues.
* Inputs in &lt;nodedef>s may now supply "hints" to code generators as to their intended interpretation, e.g. "transparency" or "opacity".
* &lt;Attributedef> elements may now define enum/enumvalues to list acceptable values or labels/mapped values for an attribute.
* If a string input specifies an "enum" list, the list is now considered a "strict" list of allowable values; no values are allowed outside that list.  To make the input non-strict, one must omit the "enum" atribute from the input.


Suggestions for v1.39:

* Add a boolean “bound” output to the various geometry property nodes, so materials can be flexible if a given attribute doesn’t exist. Especially ones like &lt;texcoord> that don’t let users specify names.

