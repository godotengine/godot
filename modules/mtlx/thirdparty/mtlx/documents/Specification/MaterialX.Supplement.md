<!-----
MaterialX Supplemental Notes v1.39
----->


# MaterialX: Supplemental Notes

**Version 1.39**  
Doug Smythe - Industrial Light & Magic  
Jonathan Stone - Lucasfilm Advanced Development Group  
March 25, 2023  



# Introduction

This document details additional information about MaterialX and how it may be incorporated into studio pipelines.  The document describes recommended naming convention for node definition elements and a directory structure to define packages of node definitions and implementations from various sources.

Previous versions of the MaterialX Supplemental Notes document included descriptions of additional node types: these node descriptions have now been folded back into the [Main Specification Document](./MaterialX.Specification.md#nodes) alongside all the other standard node descriptions.


## Table of Contents

**[Introduction](#introduction)**  

**[Recommended Element Naming Conventions](#recommended-element-naming-conventions)**  

**[Material and Node Library File Structure](#material-and-node-library-file-structure)**  
 [Examples](#examples)  

**[Definitions, Assets, and Libraries](#definitions-assets-and-libraries)**  
 [Organization Using Node Graphs](#organization-using-node-graphs)  
 [Publishing Definitions](#publishing-definitions)  
 [Dependencies and Organization](#dependencies-and-organization)  
 [Deployment, Transmission, and Translation](#deployment-transmission-and-translation)  



# Recommended Element Naming Conventions

While MaterialX elements can be given any valid name as described in the MaterialX Names section of the main specification, adhering to the following recommended naming conventions will make it easier to predict the name of a nodedef for use in implementation and nodegraph elements as well as help reduce the possibility of elements from different sources having the same name.

**Nodedef**:  "ND\__nodename_\__outputtype_[\__target_][\__version_]", or for nodes with multiple input types for a given output type (e.g. &lt;convert>), "ND\__nodename_\__inputtype_\__outputtype_[\__target_][\__version_]".

**Implementation**: "IM\__nodename_[\__inputtype_]\__outputtype_[\__target_][\__version_]".

**Nodegraph**, as an implementation for a node: "NG\__nodename_[\__inputtype_]\__outputtype_[\__target_][\__version_]".



# Material and Node Library File Structure

As studios and vendors develop libraries of shared definitions and implementations of MaterialX materials and nodes for various targets, it becomes beneficial to have a consistent, logical organizational structure for the files on disk that make up these libraries.  In this section, we propose a structure for files defining libraries of material nodes, &lt;nodedef>s, nodegraph implementations and actual target-specific native source code, as well as a mechanism for applications and MaterialX content to find and reference files within these libraries.

Legend for various components within folder hierarchies:

| Term | Description |
| --- | --- |
| _libname_ | The name of the library; the MaterialX Standard nodes are the "stdlib" library.  Libraries may choose to declare themselves to be in the <em>libname</em> namespace, although this is not required. |
| _target_ | The target for an implementation, e.g. "glsl", "oslpattern", "osl" or "mdl". |
| _sourcefiles_ | Source files (including includes and makefiles) for the target, in whatever format and structure the applicable build system requires. |


Here is the suggested structure and naming for the various files making up a MaterialX material or node definition library setup.  Italicized terms should be replaced with appropriate values, while boldface terms should appear verbatim.  The optional "\_\*" component of the filename can be any useful descriptor of the file's contents, e.g. "\_ng" for nodegraphs or "\_mtls" for materials.


 _libname_/_libname_**\_defs.mtlx**  (1)  
 _libname_/_libname_\_\***.mtlx**  (2)  
 _libname_/_target_/_libname_\_target[\_\*]**\_impl.mtlx**  (3)  
 _libname_/_target_/_sourcefiles_  (4)  



1. Nodedefs and other definitions in library _libname_.
2. Additional elements (e.g. nodegraph implementations for nodes, materials, etc.) in library _libname_.
3. Implementation elements for _libname_ specific to target _target_.
4. Source code files for _libname_ implementations specific to target _target_.

Note that nodedef files and nodegraph-implementation files go at the top _libname_ level, while &lt;implementation> element files go under the corresponding _libname_/_target_ level, next to their source code files.  This is so that studios may easily install only the implementations that are relevant to them, and applications can easily locate the implementations of nodes for specific desired targets.  Libraries are free to add additional arbitrarily-named folders for related content, such as an "images" subfolder for material library textures.

The _libname_\_defs.mtlx file typically contains nodedefs for the library, but may also contain other node types such as implementation nodegraphs, materials, looks, and any other element types.  The use of additional _libname_\_\*.mtlx files is optional, but those files should be Xincluded by the _libname_\_defs.mtlx file.

A file referenced by a MaterialX document or tool (e.g. XInclude files, filenames in &lt;image> or other MaterialX nodes, or command-line arguments in MaterialX tools) can be specified using either a relative or a fully-qualified absolute filesystem path.  A relative path is interpreted to be relative to either the location of the referencing MaterialX document itself, or relative to a location found within the current MaterialX search path: this path may be specified via an application setting (e.g. the `--path` option in MaterialXView) or globally using the MATERIALX_SEARCH_PATH environment variable.  These search paths are used for both XIncluded definitions and filename input values (e.g. images for nodes or source code for &lt;implementation>s), and applications may choose to define different search paths for different contexts if desired, e.g. for document processing vs. rendering.

The standard libraries `stdlib` and `pbrlib` are typically included _automatically_ by MaterialX applications, rather than through explicit XInclude directives within .mtlx files.  Non-standard libraries are included into MaterialX documents by XIncluding the top-level _libname_/_libname_\_defs.mtlx file, which is expected to in turn XInclude any additional .mtlx files needed by the library.


### Examples

In the examples below, MXROOT is a placeholder for one of the root paths defined in the current MaterialX search path.

A library of studio-custom material shading networks and example library materials:

```
    MXROOT/mtllib/mtllib_defs.mtlx                (material nodedefs and nodegraphs)
    MXROOT/mtllib/mtllib_mtls.mtlx                (library of materials using mtllib_defs)
    MXROOT/mtllib/images/*.tif                    (texture files used by mtllib_mtls <image> nodes)
```

Documents may include the above library using

```xml
   <xi:include href="mtllib/mtllib_defs.mtlx"/>
```

and that file would XInclude `mtllib_mtls.mtlx`.  &lt;Image> nodes within `mtllib_mtls.mtlx` would use `file` input values such as "images/bronze_color.tif", e.g. relative to the path of the `mtllib_mtls.mtlx` file itself.

Standard node definitions and reference OSL implementation:

```
    MXROOT/stdlib/stdlib_defs.mtlx                    (standard library node definitions)
    MXROOT/stdlib/stdlib_ng.mtlx                      (supplemental library node nodegraphs)
    MXROOT/stdlib/osl/stdlib_osl_impl.mtlx            (stdlib OSL implementation elem file)
    MXROOT/stdlib/osl/*.{h,osl} (etc)                 (stdlib OSL source files)
```

Layout for "genglsl" and "genosl" implementations of "stdlib" for MaterialX's shadergen component, referencing the above standard `stdlib_defs.mtlx` file:

```
    # Generated-GLSL implementations
    MXROOT/stdlib/genglsl/stdlib_genglsl_impl.mtlx    (stdlib genGLSL implementation file)
    MXROOT/stdlib/genglsl/stdlib_genglsl_cm_impl.mtlx (stdlib genGLSL color-mgmt impl. file)
    MXROOT/stdlib/genglsl/*.{inline,glsl}             (stdlib common genGLSL code)

    # Generated-OSL implementations
    MXROOT/stdlib/genosl/stdlib_genosl_impl.mtlx      (stdlib genOSL implementation file)
    MXROOT/stdlib/genosl/stdlib_genosl_cm_impl.mtlx   (stdlib genOSL color-mgmt impl. file)
    MXROOT/stdlib/genosl/*.{inline,osl}               (stdlib common genOSL code)
```

Layout for the shadergen PBR shader library ("pbrlib") with implementations for "genglsl" and "genosl" (generated GLSL and OSL, respectively) targets:

```
    MXROOT/pbrlib/pbrlib_defs.mtlx                    (PBR library definitions)
    MXROOT/pbrlib/pbrlib_ng.mtlx                      (PBR library nodegraphs)
    MXROOT/pbrlib/genglsl/pbrlib_genglsl_impl.mtlx    (pbr impl file referencing genGLSL source)
    MXROOT/pbrlib/genglsl/*.{inline,glsl}             (pbr common genGLSL code)
    MXROOT/pbrlib/genosl/pbrlib_genosl_impl.mtlx      (pbr impl file referencing genOSL source)
    MXROOT/pbrlib/genosl/*.{inline,osl}               (pbr common genOSL code)
```



# Definitions, Assets, and Libraries

In this section we propose a set of guidelines for managing unique definitions or assets and organization into libraries, wherein:

* Definitions: Correspond directly to &lt;nodedefs> which may or be either source code implementations or based on existing node definitions.
* Assets: is a term which corresponds to a definition plus any additional metadata on a definition and /or related resources such as input images.  These can be organized in logical groupings based on a desired semantic.
* Libraries: are a collection of assets.


### Organization Using Node Graphs

While it is possible to just have a set of connected nodes in a document, it is not possible to have any formal unique interface. This can invariably lead to nodes which have duplicate names, the inability to control what interfaces are exposed and inability to maintain variations over time.

Thus the base requirement for a definition is to encapsulate the nodes into a &lt;nodegraph>. This provides for:

1. Hiding Complexity: Where all nodes are scoped with the graph. For user interaction point, it makes possible the ability to “drill down” into a graph as needed but otherwise a black box representation can be provided.
2. Identifier / Path Uniqueness : The nodegraph name decreases the chances of name clashes. For example two top level nodes both called “foo” would have unique paths “bar1/foo” and “bar2/foo” when placed into two nodegraphs “bar1” and “bar2”.  
3. Interface / node signature control where specific inputs may be exposed via “interfacename” connections and outputs as desired.  This differs from “hiding” inputs or outputs which do not change the signature. The former enforces what is exposed to the user while the latter are just interface hints.

For individual inputs it is recommended to add the following additional attributes as required:

1. Real World Units : If an input value depends on scene / geometry size then a unit attribute should always be added. For example if the graph represents floor tile, then to place it properly the size of the tile. A preset set of “distance” units is provided as part of the standard library.
2. Colorspace: If an input value is represented in a given color space then to support proper transformation into rendering color space this attribute should be set. A preset set of colorspace names conforming to the AcesCg 1.2 configuration is provided as part of the standard library.

Though not strictly required it is very useful to have appropriate default values as without these the defaults will be zero values. Thus for example a “scale” attribute for a texture node should not be left to default to zero.


### Publishing Definitions

From a &lt;nodegraph> a definition can be (new &lt;nodedef>) created or “published”. Publishing allows for the following important capabilities:

1. Reuse: The ability to reuse an unique node definition as opposed to duplicating graph implementations.
2. Variation: The ability to create or apply variations to an instance independent from the implementation.
3. Interoperability: Support definitions with common properties that are mutually understood by all consumers be exchanged.

In order to support these capabilities It is recommended that the following attributes always be specified:

1. A unique name identifier: This can follow the ND\_ and NG\_ convention described. It is recommended that the signature of the interface also be encoded to provide uniqueness, especially if the definition may be polymorphic.
2. A namespace identifier (\*): When checking uniqueness namespace is always used so it is not required to be part of the name identifier to provide uniqueness.
    * It should not be used as it will result in the namespace being prepended multiple times. E.g. a “foo” node with a namespace “myspace” has a unique identifier of “myspace:node”. If the node is named “myspace:node”, then the resulting identifier is “myspace:myspace:node”.
    * Note that import of a Document will prepend namespaces as required without namespace duplication.
3. A version identifier: While this can be a general string, it is recommended that this be a template with a specific format to allow for known increment numbering. E.g. The format may be “v#.#” to support minor and major versioning. This requires that only one out of all versions be tagged as the default version. Care should be taken to ensure this as the first one found will be used as the default.
4. A nodegroup identifier: This can be one mechanism used for definition organization, or for user interface presentation. It is also used for shader generation to some extent as it provides hints as to the type of node. For example &lt;image> nodes are in the “texture2d” node group.
5. A documentation string. Though not strictly required this provides some ability to tell what a node does, and can be used as a user interface helper. Currently there is no formatting associated with this but it is possible to embed a format.

Note that utilities which codify publishing logic are provided as part of the core distribution.

To support variation it is proposed that both &lt;token>s and &lt;variant>s be used.



1. Tokens: These allow for the sample “template” be used for filenames with having to create new definitions. This can reduce the number of required definitions but also reduce the number of presets required. For example tokens can be used to control the format, resolution of the desired filename identifier.
2. Variants and Variant Sets: There are no hard-and-fast “rules” for when to create a definition vs use a definition with variants but one possible recommendation is to use variants when there are no interface signature differences (_Discuss_?). Some advantages include the fact that variants may be packaged and deployed independent of definitions and/or new definitions do not need to be deployed per variation. Note that for now only value overrides are possible.


### Dependencies and Organization

The more definitions are added including those based on other definitions, the harder it can be to discover what definitions are required given documents with some set of node instances.

To support separability of dependents, the following logical high level semantic breakdown is proposed:



1. Static “Core” library definitions. These include stdlib, pbrlib and bxdf. The recommendation is to always load these in and assume that they exist. For separability, it is recommended that these all be stored in a single runtime Document.
2. Static custom library definitions. These are based on core libraries. The recommendation is to not directly reference any core libraries using the Xinclude mechanism. This can result in duplicate possibly conflicting definitions being loaded. The “upgrade” mechanism will ensure that all core and custom libraries are upgraded to the appropriate target version. For separability, it is recommended that these all be stored in a single runtime Document.
3. Dynamically created definitions.  If this capability is allowed then it can be useful to have a fixed set of locations that these definitions can update. This could be local to the user or to update an existing custom library.

Additional groupings can be added to provide semantic organization (such as by “nodegroup”) though the recommendation is that they live within a common library root or package.

For an asset with dependent resources there are many options. Two of which are:



1. Co-locate resources with the definition. This allows for easier “packaging” of a single asset such as for transmission purposes but this can require additional discovery logic to find resources associated with a definition and may result in duplication of resources.
2. Located in a parallel folder structure to definitions. The onus is to maintain this parallel structure but the search logic complexity is the same for resources as it is for definitions.

If a definition is a source code implementation, then additional path search logic is required for discoverability during generation.

The following search paths are available:



* MATERIALX_SEARCH_PATH: This environment variable is used as part of both definition and resource search (e.g. relative filename path support).
* Source code paths: This can be registered at time of code generation as part of the generation “context”. It is recommended to follow the source path locations which would be relative to any custom definitions, using the “language” identifier of the code generator to discover the appropriate source code files.

An example usage of pathing can be found in the sample Viewer. The logic is as follows:



* The module/binary path is set as the default “root” definition path. Additional definition roots are included via MATERIALX_SEARCH_PATH. 
* The set of roots are considered to be the parent of “resources” and “libraries” folders, for resource and definitions respectively.
* The search path root for resources would be the “`<rootpath>/resources`” by default. This allows for handling of resources which are part of an assets definition. For example a brick shader located at “`/myroot/shaders/brick.mtlx`” may have the brick textures referenced at location “`/myroot/textures/brick.exr`”. Setting a single search path to “`/myroot`” handles the “parallel” folder organization mentioned, with the relative reference being “`textures/brick.exr`”
* For any shader at a given path a path to that shader can be added when resolving dependent resources. This can be used to handle the co-located folder organization. For example the shader may reside in “`/myroot/shader/brick.mtlx`”, and the texture in “`/myroot/shader/textures/brick.exr`”. Setting a root to “`myroot/shader`” and a relative reference to “`textures/brick.exr`” will allow for proper discovery.

For runtime, it is recommended that instead of reading in documents that they be “imported”. This allows for mechanisms such as namespace handling as well as tagging of source location (“sourceURI”) to occur per document. At some point all dependent documents need to be merged into a single one as there is no concept of referenced in-memory documents. Tagging is a useful mechanism to allow filtering out / exclusion of definitions from the main document content. For example, the main document can be “cleared” out while retaining the library definitions.

As there may be definitions dependent on other definitions, it is never recommended to unload core libraries, and care be taken when unloading custom or dynamic libraries. It may be useful to re-load all definitions if there is a desire to unload any one library.

Note that code generation is context based. If the context is not cleared, then dependent source code implementations will be retained. It is recommended to clear implementations if definitions are unloaded.


### Deployment, Transmission, and Translation

Given a set of definitions it is useful to consider how it will be deployed.

Some deployments for which file access may be restricted or accessing many files is a performance issue, pre-package up definitions, source and associated resources may be required. For example, this is currently used for the sample Web viewer deployment.

Some deployments may not want to handle non-core definitions or may not be able to handle (complex) node graphs. Additionally the definition may be transmitted as a shader. Thus, when creating a new definition it is recommended to determine the level of support for:



1. Flattening: Can the definition be converted to a series of nodes which have source code implementations.
2. Baking: Can the definition be converted to an image.
3. Translation: Can the implementation be converted mapped / converted to another implementation which can be consumed.
4. Shader Reflection: Can the desired metadata be passed to the shader for introspection.

Additional metadata which is not a formal part of the specification may lead to the inability to be understood by consumers.

