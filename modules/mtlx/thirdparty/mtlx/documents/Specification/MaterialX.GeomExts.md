<!-----
MaterialX Geometry Extensions v1.39
----->


# MaterialX Geometry Extensions

**Version 1.39**  
Doug Smythe - Industrial Light & Magic  
Jonathan Stone - Lucasfilm Advanced Development Group  
October 21, 2022  


# Introduction

The core [**MaterialX Specification**](./MaterialX.Specification.md) defines a number of element types and specific functional node definitions which can be used to describe the structure of shading networks and materials, including the definitions and functionality of custom shading operations.

There are many formats that can be used to describe the associations between shading materials and renderable geometry as well as various data and metadata associated with geometry.  For performance and other reasons it is often desirable to use native application mechanisms or something like Pixar's USD[^1] to describe these associations.  However, there is significant value in being able to store a complete description of a CG object "look" within a single application-independent file format.  This document describes extensions to the core MaterialX Specification that can be used to define collections of geometries, geometric properties and their values per geometry, and assignment of materials, variants, visibility and rendering properties to specific geometries in named looks, either directly or via geometry name expressions or named collections.


## Table of Contents

**[Introduction](#introduction)**  

**[Geometry Representation](#geometry-representation)**  
 [Lights](#lights)  
 [Geometry Name Expressions](#geometry-name-expressions)  
 [Collections](#collections)  
 [Geometry Prefixes](#geometry-prefixes)  

**[Additional MaterialX Data Types](#additional-materialx-data-types)**  

**[Additional Filename Substitutions](#additional-filename-substitutions)**  

**[Geometry Info Elements](#geometry-info-elements)**  
 [GeomInfo Definition](#geominfo-definition)  
 [GeomProp Elements](#geomprop-elements)  
 [Geometry Token Elements](#geometry-token-elements)  
 [TokenDefault Elements](#tokendefault-elements)  
 [Reserved GeomProp Names](#reserved-geomprop-names)  

**[Look and Property Elements](#look-and-property-elements)**  
 [Property Definition](#property-definition)  
 [Look Definition](#look-definition)  
 [Assignment Elements](#assignment-elements)  
 [MaterialAssign Elements](#materialassign-elements)  
 [VariantAssign Elements](#variantassign-elements)  
 [Visibility Elements](#visibility-elements)  
 [PropertyAssign Elements](#propertyassign-elements)  
 [Look Examples](#look-examples)  

**[References](#references)**



# Geometry Representation

Geometry is referenced by but not specifically defined within MaterialX content.  The file in which geometry is defined can optionally be declared using `geomfile` attributes within any element; that `geomfile` declaration will then apply to any geometry name referenced within the scope of that element, e.g. any `geom` attributes, including those defining the contents of collections (but not when referencing the contents of a collection via a `collection` attribute).  If a geomfile is not defined for the scope of any particular `geom` attribute, it is presumed that the host application can resolve the location of the geometry definition.

The geometry naming conventions used in the MaterialX specification are designed to be compatible with those used in Alembic ([http://www.alembic.io/](http://www.alembic.io/)) and USD ([http://graphics.pixar.com/usd](http://graphics.pixar.com/usd)).  "Geometry" can be any particular geometric object that a host application may support, including but not limited to polygons, meshes, subdivision surfaces, NURBS, implicit surfaces, particle sets, volumes, lights, procedurally-defined objects, etc.  The only requirements for MaterialX are that geometries are named using the convention specified below, can be assigned to a material and can be rendered.

The naming of geometry should follow a syntax similar to UNIX full paths:

```
   /string1/string2/string3/...
```

E.g. an initial "/" followed by one or more hierarchy level strings separated by "/"s, ending with a final string and no "/".  The strings making up the path component for a level of hierarchy cannot contain spaces or "/"s or any of the characters reserved for geometry name expressions (see below).  Individual implementations may have further restrictions on what characters may be used for hierarchy level names, so for ultimate compatibility it is recommended to use names comprised only of upper- or lower-case letters, digits 0-9, and underscores ("_").

Geometry names (e.g. the full path name) must be unique within the entire set of geometries referenced in a setup.  Note that _there is no implied transformation hierarchy in the specified geometry paths_: the paths are simply the names of the geometry.  However, the path-like nature of geometry names can be used to benefit in geometry name expression pattern matching and assignments.

Note: if a geometry mesh is divided into partitions, the syntax for the parent mesh would be:

```
   /path/to/geom/meshname
```

and for the child partitions, the syntax would be:

```
   /path/to/geom/meshname/partitionname
```

Assignments to non-leaf locations apply hierarchically to all geometries below the specified location, unless they are the target of another assignment.  By extension, an assignment to "/" applies to _all _geometries within the MaterialX setup, unless they are the target of another assignment.


## Lights

Computer Graphics assets often include lights as part of the asset, such as the headlights of a car.  MaterialX does not define "light" objects per se, but instead allows referencing externally-defined light objects in the same manner as geometry, via a UNIX-like path.  MaterialX does not describe the position, view or shape of a light object: MaterialX presumes that these properties are stored within the external representation.

Light object geometries can be turned off (muted) in looks by making the light geometry invisible, assignment of "light"-context shader materials can be done using a &lt;materialassign> within a &lt;look>, and illumination and shadowing assignments can be handled using &lt;visibility> declarations for the light geometry.  See the [**Look Definition**](#look-definition) section below for details.



## Geometry Name Expressions

Certain elements in MaterialX files support geometry specification via expressions.  The syntax for geometry name expressions in MaterialX largely follows that of “glob” patterns for filenames in Unix environments, with a few extensions for the specific needs of geometry references.

Within a single hierarchy level (e.g. between "/"s):

* `*`  matches 0 or more characters
* `?`  matches exactly one character
* `[]` are used to match any individual character within the brackets, with "-" meaning match anything between the character preceding and the character following the "-"
* `{}` are used to match any of the comma-separated strings or expressions within the braces

Additionally, a `/` will match only exactly a single `/` in a geometry name, e.g. as a boundary for a hierarchy level, while a `//` will match a single `/`, or two `/`s any number of hierarchy levels apart; `//` can be used to specify a match at any hierarchy depth.  If a geometry name ends with `//*`, the final `*` will only match leaf geometries in the hierarchy.  A geometry name of `//*` by itself will match all leaf geometries in an entire scene, while the name `//*//` will match all geometries at any level, including nested geometries, and the name `/a/b/c//*//` will match all geometries at any level below `/a/b/c`.  It should be noted that for a mesh with partitions, it is the partitions and not the mesh which are treated as leaf geometry by MaterialX geometry names using `//*`.



## Collections

Collections are recipes for building a list of geometries (which can be any path within the geometry hierarchy), which can be used as a shorthand for assignments to a (potentially large) number of geometries at once.  Collections can be built up from lists of specific geometries, geometries matching defined geometry name expressions, other collections, or any combination of those.

A **&lt;collection>** element contains lists of geometry expressions and/or collections to be included, and an optional list of geometry expressions to be excluded:

```xml
  <collection name="collectionname" [includegeom="geomexpr1[,geomexpr2]..."]
             [includecollection="collectionname1[,collectionname2]..."]
             [excludegeom="geomexpr3[,geomexpr4]..."]/>
```

Either `includegeom` and/or `includecollection` must be specified.  The `includegeom` and `includecollection` lists are applied first, followed by the `excludegeom` list.  This can be used to build up the contents of a collection in pieces, or to add expression-matched geometry then remove specific unwanted matched geometries.  The contents of a collection can itself be used to define a portion of another collection.  The contents of each `includecollection` collection are effectively evaluated in whole before being added to the collection being built.

If the containing file is capable of defining MaterialX-compliant collections (e.g. an Alembic or USD file), its collections can be referred to in any situation where a <code>collection="<em>name</em>"</code> reference is allowed.



## Geometry Prefixes

As a shorthand convenience, MaterialX allows the specification of a `geomprefix` attribute that will be prepended to data values of type "geomname" or "geomnamearray" (e.g. `geom` attributes in `<geominfo>`, `<collection>`, `<materialassign>`, and `<visibility>` elements) specified within the scope of the element defining the `geomprefix`,  similar to how MaterialX allows the specification of a `fileprefix` attribute which is prepended to input values of type "filename".  For data values of type "geomnamearray", the `geomprefix` is prepended to each individual comma-separated geometry name.  Since the values of the prefix and the geometry are string-concatenated, the value of a `geomprefix` should generally end with a "/".  Geomprefix is commonly used to split off leading portions of geometry paths common to all geometry names, e.g. to define the "asset root" path.

So the following MTLX file snippets are equivalent:


```xml
  <materialx>
    <collection name="c_plastic" includegeom="/a/b/g1, /a/b/g2, /a/b/g5, /a/b/c/d/g6"/>
  </materialx>

  <materialx geomprefix="/a/b/">
    <collection name="c_plastic" includegeom="g1, g2, g5, c/d/g6"/>
  </materialx>
```



# Additional MaterialX Data Types

Systems supporting MaterialX Geometry Extensions support the following additional standard data types:

**GeomName** and **GeomNameArray**: attributes of type "geomname" are just strings within quotes, but specifically mean the name of a single geometry using the conventions described in the [**Geometry Representation**](#geometry-representation) and [**Geometry Name Expressions**](#geometry-name-expressions) sections.  A geomname is allowed to use a geometry name expression as long as it resolves to a single geometry.  Attributes of type "geomnamearray" are strings within quotes containing a comma-separated list of one or more geomname values with or without expressions, and may resolve to any number of geometries.


# Additional Filename Substitutions

Filename input values for various nodes can include one or more special strings which will be replaced by the application with values derived from the current geometry, from the MaterialX state, or from the host application environment.  Applications which support MaterialX Geometry Extensions also support the following filename substitution:


| Token | Description |
| ---- | ---- |
| &lt;<em>geometry token</em>> | The value of a specified token declared in a &lt;geominfo> element or as a uniform primvar value (generally of type string or integer) for the current geometry. |


Only applications fully supporting Geometry Extensions may allow using a &lt;_geometry token_> as part of a larger filename string.  All applications should allow the use of "&lt;_geometry token_>" as the full filename string, in which case the string primvar value stored with the geometry is used as the filename unchanged; the string primvar value itself might be allowed to contain another token such as &lt;UDIM> which the renderer may be able to parse and replace itself.



# Geometry Info Elements

Geometry Info ("geominfo") elements are used to define sets of named geometric properties with constant values, and to associate them with specific external geometries.

The most common use for geominfo elements is to define the filenames (or portions of filenames) of texture map images mapped onto the geometry.  Typically, there are several types of textures such as color, roughness, bump, opacity, etc. associated with each geometry: each texture name string would be a separate &lt;token> within the &lt;geominfo>.  These images could contain texture data for multiple geometries, which would either be listed in the `geom` attribute of the &lt;geominfo> element, or be assembled into a collection and the name of that collection would be specified in the `collection` attribute.


## GeomInfo Definition

A **&lt;geominfo>** element contains one or more geometry property and/or token definitions, and associates them and their values with all geometries listed in the `geom` or `collection` attribute of the &lt;geominfo>:

```xml
  <geominfo name="name" [geom="geomexpr1,geomexpr2,geomexpr3"] [collection="coll"]>
    ...geometry property and token value definitions...
  </geominfo>
```

Note that no two &lt;geominfo>s may define values for the same geometry property or token for the same geometry, whether the geometry is specified directly, matched via a geometry name expression, or contained within a specified collection.

Attributes for GeomInfo elements:

* `name` (string, required): the unique name of the GeomInfo element
* `geom` (geomnamearray, optional): the list of geometries and/or geometry name expressions that the GeomInfo is to apply to
* `collection` (string, optional): the name of a geometric collection

Either a `geom` or a `collection` may be specified, but not both.



### GeomProp Elements

The core MaterialX Specification defines a Geometric Property, or "geomprop", as an intrinsic or user-defined surface coordinate property of geometries referenced in a specific space and/or index, and provides several nodes to retrive the values of these properties within a shading network nodegraph, as well as a &lt;geompropdef> element used to define the name and output type of custom geometric properties beyond the standard ones: `position`, `normal`, `tangent`, `bitangent`, `texcoord` and `geomcolor`.

MaterialX Geometry Extensions expands upons this by allowing the use of &lt;geomprop> elements to define specific uniform values of a geometric property with specific geometries, as opposed to relying on those values being defined externally.  This could include application-specific metadata, attributes passed from a lighting package to a renderer, or other geometry-specific data.  A geomprop may also specify a `unittype` and `unit` if appropriate to indicate that the geometric property's value is in that unit; see the [**Units** section of the main MaterialX Specification](./MaterialX.Specification.md#units), although typically the &lt;geompropdef> would define the `unittype` and `unit`, and a geomprop would only provide an overriding `unit` if the unit for its value differed from the geompropdef's defined default unit.

```xml
    <geomprop name="propname" type="proptype" value="value"/>
```

GeomProp elements have the following attributes:

* `name` (string, required): the name of the geometric property to define
* `type` (string, required): the data type of the given property
* `value` (any MaterialX type, required): the value to assign to the given property.
* `unittype` (attribute, string, optional): the type of unit for this property, e.g. "distance", which must be defined by a &lt;unittypedef>.  Default is to not specify a unittype.
* `unit` (attribute, string, optional): the specific unit for this property.  Default is to not specify a unit.

Only float and vector<em>N</em> geometric properties may specify a `unittype` and a `unit`.

For example, one could specify a unique surface ID value associated with a geometry:

```xml
  <geompropdef name="surfid" type="integer"/>
  <geominfo name="gi1" geom="/a/g1">
    <geomprop name="surfid" type="integer" value="15"/>
  </geominfo>
```

GeomProp values can be accessed from a nodegraph using a `<geompropvalue>` node:

```xml
  <geompropvalue name="srfidval1" type="integer" geomprop="surfid" default="0">
```

A &lt;geomprop> can also be used to define a default value for an intrinsic varying geometric property such as "geomcolor" for the geometry specified by the enclosing &lt;geominfo>, which would be returned by the corresponding Geometric node (e.g. &lt;geomcolor>) if the current geometry did not itself define values for that property.

```xml
  <geominfo name="gi2" geom="/a/g2">
    <geomprop name="geomcolor" type="color3" value="0.5, 0, 0"/>
  </geominfo>
```



### Geometry Token Elements

Token elements may be used within &lt;geominfo> elements to define constant (typically string or integer) named values associated with specific geometries.  These geometry token values can be substituted into filenames within image nodes; see the [**Additional Filename Substitutions**](#additional-filename-substitutions) section above for details:

```xml
  <token name="tokenname" type="tokentype" value="value"/>
```

The "value" can be any MaterialX type, but since tokens are used in filename substitutions, string and integer values are recommended.

Token elements have the following attributes:

* `name` (string, required): the name of the geometry token to define
* `type` (string, required): the geometry token's type
* `value` (any MaterialX type, optional): the value to assign to that token name for this geometry.

For example, one could specify a texture identifier value associated with a geometry:

```xml
  <geominfo name="gi1" geom="/a/g1">
    <token name="txtid" type="string" value="Lengine"/>
  </geominfo>
```

and then reference that token's value in a filename:

```xml
  <image name="cc1" type="color3">
    <input name="file" type="filename"
        value="txt/color/asset.color.<txtid>.tif"/>
  </image>
```

The &lt;txtid> in the file name would be replaced by whatever value the txtid token had for each geometry.


### TokenDefault Elements

TokenDefault elements define the default value for a specified geometry token name; this default value will be used in a filename string substitution if an explicit token value is not defined for the current geometry.  Since TokenDefault does not apply to any geometry in particular, it must be used outside of a &lt;geominfo> element.

```xml
  <tokendefault name="diffmap" type="string" value="color1"/>
```


### Reserved GeomProp Names

Workflows involving textures with implicitly-computed filenames based on u,v coordinates (such as &lt;UDIM> and &lt;UVTILE>) can be made more efficient by explicitly listing the set of values that they resolve to for any given geometry.  The MaterialX specification reserves two geomprop names for this purpose, `udimset` and `uvtileset`, each of which is a stringarray containing a comma-separated list of UDIM or UVTILE values:

```xml
  <geominfo name="gi4" geom="/a/g1,/a/g2">
    <geomprop name="udimset" type="stringarray" value="1002,1003,1012,1013"/>
  </geominfo>

  <geominfo name="gi5" geom="/a/g4">
    <geomprop name="uvtileset" type="stringarray" value="u2_v1,u2_v2"/>
  </geominfo>
```



# Look and Property Elements

**Look** elements define the assignments of materials, visibility and other properties to geometries and geometry collections.  In MaterialX, a number of geometries are associated with each stated material, visibility type or property in a look, as opposed to defining the particular material or properties for each geometry.

**Property** elements define non-material properties that can be assigned to geometries or collections in Looks.  There are a number of standard MaterialX property types that can be applied universally for any rendering target, as well as a mechanism to define target-specific properties for geometries or collections.

A MaterialX document can contain multiple property and/or look elements.


## Property Definition

A **&lt;property>** element defines the name, type and value of a look-specific non-material property of geometry; &lt;**propertyset**> elements are used to group a number of &lt;property>s into a single named object.  The connection between properties or propertysets and specific geometries or collections is done in a &lt;look> element, so that these properties can be reused across different geometries, and enabled in some looks but not others.  &lt;Property> elements may only be used within &lt;propertyset>s; they may not be used independently, although a dedicated &lt;propertyassign> element may be used within a &lt;look> to declare a property name, type, value and assignment all at once.

```xml
  <propertyset name="set1">
    <property name="twosided" type="boolean" value="true"/>
    <property name="trace_maxdiffusedepth" target="rmanris" type="float" value="3"/>
  </propertyset>
```

The following properties are considered standard in MaterialX, and should be respected on all platforms that support these concepts:


| Property | Type | Default Value |
| --- | --- | --- |
| **`twosided`** | boolean | false |
| **`matte`** | boolean | false |

where `twosided` means the geometry should be rendered even if the surface normal faces away from camera, and `matte` means the geometry should hold out, or "matte" out anything behind it (including in the alpha channel).

In the example above, the "trace_maxdiffusedepth" property is target-specific, having been restricted to the context of Renderman RIS by setting its `target` attribute to “rmanris”.



## Look Definition

A **&lt;look>** element contains one or more material, variant, visibility and/or propertyset assignment declarations:

```xml
  <look name="lookname" [inherit="looktoinheritfrom"]>
    ...materialassign, variantassign, visibilityassign, property/propertysetassign declarations...
  </look>
```

Looks can inherit the assignments from another look by including an `inherit` attribute.  The look can then specify additional assignments that will apply on top of/in place of whatever came from the source look.  This is useful for defining a base look and then one or more "variation" looks.  It is permissible for an inherited-from look to itself inherit from another look, but a look can inherit from only one parent look.

A number of looks can be grouped together into a **LookGroup**, e.g. to indicate which looks are defined for a particular asset:

```xml
  <lookgroup name="lookgroupname" looks="look1[,look2[,look3...]]" [default="lookname"]/>
```

where `lookgroupname` is the name of the lookgroup being defined, `look1`/`look2`/etc. are the names of &lt;look> or &lt;lookgroup> elements to be contained in the lookgroup (a lookgroup name would resolve to the set of looks recursively contained in that lookgroup), and `default` (if specified) specifies the name of one of the looks defined in `looks` to be the default look to use.  A look can be contained in any number of lookgroups.

&lt;Look> and &lt;lookgroup> elements also support other attributes such as `xpos`, `ypos` and `uicolor` as described in the Standard UI Attributes section above.


## Assignment Elements

Various types of assignment elements are used within looks to assign materials, categorized visibility and properties to specific geometries, or variants to materials.

For elements which make assignments to geometries, the pathed names within `geom` attributes or stored within collections do not need to resolve strictly to "leaf" path locations or actual renderable geometry names: assignments can also be made to intermediate "branch" geometry path locations, which will then apply to any geometry at a deeper level in the path hierarchy which does not have another "closer to the leaf" level assignment.  E.g. an assignment to "/a/b/c" will effectively apply to "/a/b/c/d" and "/a/b/c/foo/bar" (and anything else whose full path name begins with "/a/b/c/") if no other assignment is made to "/a/b/c/d", "/a/b/c/foo", or "/a/b/c/foo/bar".  If a look inherits from another look, the child look can replace assignments made to any specific path location (e.g. a child assignment to "/a/b/c" would take precedence over a parent look's assignment to "/a/b/c"), but an assignment by the parent look to a more "leaf"-level path location would take precedence over a child look assignment to a higher "branch"-level location.


### MaterialAssign Elements

MaterialAssign elements are used within a &lt;look> to connect a specified material to one or more geometries or collections (either a `geom` or a `collection` may be specified, but not both).

```xml
  <materialassign name="maname" material="materialname"
                 [geom="geomexpr1[,geomexpr2...]"] [collection="collectionname"]
                 [exclusive=true|false]>
    ...optional variantassign elements...
  </materialassign>
```

Material assignments are generally assumed to be mutually-exclusive, that is, any individual geometry is assigned to only one material.  Therefore, assign declarations should be processed in the order they appear in the file, and if any geometry appears in multiple &lt;materialassign>s, the last &lt;materialassign> wins.  However, some applications allow multiple materials to be assigned to the same geometry as long as the shader node types don't overlap.  If the `exclusive` attribute is set to false (default is true), then earlier material assigns will still take effect for all shader node types not defined in the materials of later assigns: for each shader node type, the shader within the last assigned material referencing a matching shader node type wins.  If a particular application does not support multiple material assignments to the same geometry, the value of `exclusive` is ignored and only the last full material and its shaders are assigned to the geometry, and the parser should issue a warning.


### VariantAssign Elements

VariantAssign elements are used within a &lt;materialassign> or a &lt;look> to apply the values defined in one variant of a variantset to one assigned material, or to all applicable materials in a look.

```xml
  <look name="look1">
    <variantassign name="va1" variantset="varset1" variant="var1"/>
    <materialassign name="ma1" material="material1" geom="...">
      <variantassign name="va2" variantset="varset2" variant="var2"/>
    </materialassign>
    <materialassign name="ma2" material="material2" geom="..."/>
    ...
  </look>
```

VariantAssign elements have the following attributes:

* `name` (string, required): the unique name of the VariantAssign element
* `variantset` (string, required): the name of the variantset to apply the variant from
* `variant` (string, required): the name of the variant within `variantset` to use

In the above example, the input/token values defined within variant "var1" will be applied to and and all identically-named inputs/tokens found in either "material1" or "material2" unless restricted by a `node` or `nodedef` attribute defined in the &lt;variantset>, while values defined within variant "var2" will only be applied to matching-named bindings in "material1".  VariantAssigns are applied in the order specified within a scope, with those within a &lt;materialassign> taking precedence over those which are direct children of the &lt;look>.


### Visibility Elements

Visibility elements are used within a &lt;look> to define various types of generalized visibility between a "viewer" object and other geometries.  A "viewer object" is simply a geometry that has the ability to "see" other geometries in some rendering context and thus may need to have the list of geometries that it "sees" in different contexts be specified; the most common examples are light sources and a primary rendering camera.

```xml
  <visibility name="vname" [viewergeom="objectname"]
             [geom="geomexpr1[,geomexpr2...]"] [collection="collectionname"]
             [vistype="visibilitytype"] [visible="false"]/>
```

Visibility elements have the following attributes:

* `name` (string, required): the unique name of the Visibility element
* `viewergeom` (geomnamearray, optional): the list of viewer geometry objects that the &lt;visibility> assignment affects
* `viewercollection` (string, optional): the name of a collection containing viewer geometry objects that the &lt;visibility> assignment affects
* `geom` (geomnamearray, optional): the list of geometries and/or geometry name expressions that the `viewergeom` object should (or shouldn't) "see"
* `collection` (string, optional): the name of a defined collection of geometries that the `viewergeom` object should (or shouldn't) "see"
* `vistype` (string, optional): the type of visibility being defined; see table below
* `visible` (boolean, optional): if false, the geom/collection objects will be invisible to this particular type of visibility; defaults to "true".

The `viewergeom` attribute (and/or the contents of a collection referred to by the `viewercollection` attribute) typically refers to the name of a light (or list of lights) or other "geometry viewing" object(s).  If `viewergeom`/`viewercollection` are omitted, the visibility applies to all applicable viewers (camera, light, geometry) within the given render context; `viewergeom`/`viewercollection` are not typically specified for `vistype` "camera".  Either `geom` or `collection` must be defined but not both; similarly, one cannot define both a `viewergeom` and a `viewercollection`.

The `vistype` attribute refers to a specific type of visibility.  If a particular `vistype` is not assigned within a &lt;look>, then all geometry is visible by default to all `viewergeom`s for that `vistype`; this means that to have only a certain subset of geometries be visible (either overall or to a particular `vistype`), it is necessary to first assign &lt;visibility> with `visible="false"` to all geometry.  Additional &lt;visibility> assignments to the same `vistype` within a &lt;look> are applied on top of the current visibility state.  The following `vistype`s are predefined by MaterialX; applications are free to define additional `vistype`s:


| Vistype | Description |
| --- | --- |
| **`camera`** | camera or "primary" ray visibility |
| **`illumination`** | geom or collection is illuminated by the viewergeom light(s) |
| **`shadow`** | geom or collection casts shadows from the viewergeom light(s) |
| **`secondary`** | indirect/bounce ray visibility of geom or collection to viewergeom geometry |


If `vistype` is not specified, then the visibility assignment applies to _all_ visibility types, and in fact will take precedence over any specific `vistype` setting on the same geometry: geometry assigned a &lt;`visibility>` with no `vistype` and `visible="false"` will not be visible to camera, shadows, secondary rays, or any other ray or render type.  This mechanism can be used to cleanly hide geometry not needed in certain variations of an asset, e.g. different costume pieces or alternate damage shapes.

If the &lt;visibility> `geom` or `collection` refers to light geometry, then assigning `vistype="camera"` determines whether or not the light object itself is visible to the camera/viewer (e.g. "do you see the bulb"), while assigning `visible="false"` with no `vistype` will mute the light so it is neither visible to camera nor emitting any light.

For the "secondary" vistype, `viewergeom` should be renderable geometry rather than a light, to declare that certain other geometry is or is not visible to indirect bounce illumination or raytraced reflections in that `viewergeom`.  In this example, "/b" would not be seen in reflections nor contribute indirect bounce illumination to "/a", while geometry "/c" would not be visible to _any_ secondary rays:

```xml
  <visibility name="v2" viewergeom="/a" geom="/b" vistype="secondary" visible="false"/>
  <visibility name="v3" geom="/c" vistype="secondary" visible="false"/>
```


### PropertyAssign Elements

PropertyAssign and PropertySetAssign elements are used within a &lt;look> to connect a specified property value or propertyset to one or more geometries or collections.

```xml
  <propertyassign name="paname" property="propertyname" type="type" value="value"
                 [target="target"]
                 [geom="geomexpr1[,geomexpr2...]"] [collection="collectionname"]/>
  <propertysetassign name="psaname" propertyset="propertysetname"
                 [geom="geomexpr1[,geomexpr2...]"] [collection="collectionname"]/>
```

Either a `geom` or a `collection` may be specified, but not both.  Multiple property/propertyset assignments can be made to the same geometry or collection, as long as no conflicting assignment is made.  If there are any conflicting assignments, it is up to the host application to determine how such conflicts are to be resolved, but host applications should apply property assignments in the order they are listed in the look, so it should generally be safe to assume that if two property/propertyset assignments set different values for the same property to the same geometry, the later assignment will win.


## Look Examples

This example defines four collections, a light shader and material, and a propertyset, which are then used by two looks:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<materialx>
  <!-- assume <nodedef> and <surfacematerial> elements to define Mplastic1,2 and Mmetal1,2 are placed or included here -->
  <collection name="c_plastic" includegeom="/a/g1,/a/g2,/a/g5"/>
  <collection name="c_metal" includegeom="/a/g3,/a/g4"/>
  <collection name="c_lamphouse" includegeom="/a/lamp1/housing*Mesh"/>
  <collection name="c_setgeom" includegeom="/b"/>
  <nodedef name="ND_disklgt_lgt" node="disk_lgt">
    <input name="emissionmap" type="filename" value=""/>
    <input name="gain" type="float" value="1.0"/>
    <output name="out" type="lightshader"/>
  </nodedef>
  <disk_lgt name="LSheadlight">
    <input name="gain" type="float" value="500.0"/>
  </disk_lgt>
  <lightmaterial name="Mheadlight">
    <input name="lightshader" type="lightshader" nodename="LSheadlight"/>
  </lightmaterial>
  <propertyset name="standard">
    <property name="displacementbound_sphere" target="rmanris" type="float"
           value="0.05"/>
    <property name="trace_maxdiffusedepth" target="rmanris" type="float" value="5"/>
  </propertyset>
  <look name="lookA">
    <materialassign name="ma1" material="Mplastic1" collection="c_plastic"/>
    <materialassign name="ma2" material="Mmetal1" collection="c_metal"/>
    <materialassign name="ma3" material="Mheadlight" geom="/a/b/headlight"/>
    <visibility name="v1" viewergeom="/a/b/headlight" vistype="shadow" geom="/" visible="false"/>
    <visibility name="v2" viewergeom="/a/b/headlight" vistype="shadow" collection="c_lamphouse"/>
    <propertysetassign name="psa1" propertysetname="standard" geom="/"/>
  </look>
  <look name="lookB">
    <materialassign name="ma4" material="Mplastic2" collection="c_plastic"/>
    <materialassign name="ma5" material="Mmetal2" collection="c_metal"/>
    <propertysetassign name="psa2" propertysetname="standard" geom="/"/>
    <!-- make the setgeom invisible to camera but still visible to shadows and reflections -->
    <visibility name="v3" vistype="camera" collection="c_setgeom" visible="false"/>
  </look>
  <lookgroup name="assetlooks" looks="lookA,lookB" default="lookA"/>
</materialx>
```


# References

[^1]: <https://graphics.pixar.com/usd/release/index.html>

