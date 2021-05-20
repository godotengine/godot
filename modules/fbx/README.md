# Open Source FBX Specification for the Importer

The goal of this document is to make everything in FBX clearly stated, any errors will be corrected over time this
is a first draft.

## fbx parser - originally from assimp

- Folder: /modules/fbx/fbx_parser
- Upstream: assimp
- Original Version: git (308db73d0b3c2d1870cd3e465eaa283692a4cf23, 2019)
- License: BSD-3-Clause

This can never be updated from upstream, we have heavily modified the parser to provide memory safety and add some
functionality. If anything we should give this parser back to assimp at some point as it has a lot of new features.

# Updating assimp fbx parser

Don't. It's not possible the code is rewritten in many areas to remove thirdparty deps and various bugs are fixed.

Many days were put into rewriting the parser to use safe code and safe memory accessors.

# File Headers

FBX Binaries start with the header "Kaydara FBX Binary"

FBX ASCII documents contain a larger header, sometimes with copyright information for a file.

Detecting these is pretty simple.

# What is an OP link?
It's an object to property link. It lists the properties for that object in some cases. Source and destination based by
ID.

# What is a OO link?
Its an object to object link, it contains the ID source and destination ID.

# FBX Node connections

Nodes in FBX are connected using OO links, This means Object to Object.

FBX has a single other kind of link which is Object Property, this is used for Object to Property Links, this can be
 extra attributes, defaults, or even some simple settings.

# Bones / Joints / Locators

Bones in FBX are nodes, they initially have the Model:: Type, then have links to SubDeformer the sub deformer
is part of the skin there is also an explicit Skin link, which then links to the geometry using OO links in the
document.

# Rotation Order in FBX compared to Godot

**Godot uses the rotation order:** YXZ

**FBX has dynamic rotation order to prevent gimbal lock with complex animations**

```cpp
enum RotOrder {
	RotOrder_EulerXYZ = 0
	RotOrder_EulerXZY,
	RotOrder_EulerYZX,
	RotOrder_EulerYXZ,
	RotOrder_EulerZXY,
	RotOrder_EulerZYX,
	RotOrder_SphericXYZ // nobody uses this - as far as we can tell
};
```


# Pivot transforms

### Pivot description:
- Maya and 3DS max consider everything to be in node space (bones joints, skins, lights, cameras, etc)
- Everything is a node, this means essentially nodes are auto or variants
- They are local to the node in the tree.
- They are used to calculate where a node is in space
```c++
// For a better reference you can check editor_scene_importer_fbx.h
// references: GenFBXTransform / read the data in
// references: ComputePivotTransform / run the calculation
// This is the local pivot transform for the node, not the global transforms
Transform ComputePivotTransform(
		Transform chain[TransformationComp_MAXIMUM],
		Transform &geometric_transform) {
	// Maya pivots
	Transform T = chain[TransformationComp_Translation];
	Transform Roff = chain[TransformationComp_RotationOffset];
	Transform Rp = chain[TransformationComp_RotationPivot];
	Transform Rpre = chain[TransformationComp_PreRotation];
	Transform R = chain[TransformationComp_Rotation];
	Transform Rpost = chain[TransformationComp_PostRotation];
	Transform Soff = chain[TransformationComp_ScalingOffset];
	Transform Sp = chain[TransformationComp_ScalingPivot];
	Transform S = chain[TransformationComp_Scaling];

	// 3DS Max Pivots
	Transform OT = chain[TransformationComp_GeometricTranslation];
	Transform OR = chain[TransformationComp_GeometricRotation];
	Transform OS = chain[TransformationComp_GeometricScaling];

	// Calculate 3DS max pivot transform - use geometric space (e.g doesn't effect children nodes only the current node)
	geometric_transform = OT * OR * OS;
	// Calculate standard maya pivots
	return T * Roff * Rp * Rpre * R * Rpost.inverse() * Rp.inverse() * Soff * Sp * S * Sp.inverse();
}
```

# Transform inheritance for FBX Nodes

The goal of below is to explain why they implement this in the first place.

The use case is to make nodes have an option to override their local scaling or to make scaling influenced by orientation, which i would imagine would be useful for when you need to rotate a node and the child to scale based on the orientation rather than setting on the rotation matrix planes.
```cpp
// not modified the formatting here since this code must remain clear
enum TransformInheritance {
	Transform_RrSs = 0,
	// Parent Rotation * Local Rotation * Parent Scale * Local Scale  -- Parent Rotation Offset * Parent ScalingOffset (Local scaling is offset by rotation of parent node)
	Transform_RSrs = 1, // Parent Rotation * Parent Scale * Local Rotation * Local Scale -- Parent * Local (normal mode)
	Transform_Rrs = 2, // Parent Rotation * Local Rotation * Local Scale -- Node transform scale is the only relevant component
	TransformInheritance_MAX // end-of-enum sentinel
};

enum TransformInheritance {
	Transform_RrSs = 0,
	// Local scaling is offset by rotation of parent node
	Transform_RSrs = 1,
	// Parent * Local (normal mode)
	Transform_Rrs = 2,
	// Node transform scale is the only relevant component
	TransformInheritance_MAX // end-of-enum sentinel
};
```

# Axis in FBX

Godot has one format for the declared axis

AXIS X, AXIS Y, -AXIS Z

FBX supports any format you can think of. As it has to support Maya and 3DS Max.

#### FBX File Header
```json
GlobalSettings:  {
	Version: 1000
	Properties70:  {
		P: "UpAxis", "int", "Integer", "",1
		P: "UpAxisSign", "int", "Integer", "",1
		P: "FrontAxis", "int", "Integer", "",2
		P: "FrontAxisSign", "int", "Integer", "",1
		P: "CoordAxis", "int", "Integer", "",0
		P: "CoordAxisSign", "int", "Integer", "",1
		P: "OriginalUpAxis", "int", "Integer", "",1
		P: "OriginalUpAxisSign", "int", "Integer", "",1
		P: "UnitScaleFactor", "double", "Number", "",1
		P: "OriginalUnitScaleFactor", "double", "Number", "",1
		P: "AmbientColor", "ColorRGB", "Color", "",0,0,0
		P: "DefaultCamera", "KString", "", "", "Producer Perspective"
		P: "TimeMode", "enum", "", "",6
		P: "TimeProtocol", "enum", "", "",2
		P: "SnapOnFrameMode", "enum", "", "",0
		P: "TimeSpanStart", "KTime", "Time", "",0
		P: "TimeSpanStop", "KTime", "Time", "",92372316000
		P: "CustomFrameRate", "double", "Number", "",-1
		P: "TimeMarker", "Compound", "", ""
		P: "CurrentTimeMarker", "int", "Integer", "",-1
	}
}
```

#### FBX FILE declares axis dynamically using FBX header
Coord is X
Up is Y
Front is Z

#### GODOT - constant reference point
Coord is X positive,
Y is up positive,
Front is -Z negative

### Explaining MeshGeometry indexing

Reference type declared:
- Direct (directly related to the mapping information type)
- IndexToDirect (Map with key value, meaning depends on the MappingInformationType)

ControlPoint is a vertex
* None The mapping is undetermined.
* ByVertex There will be one mapping coordinate for each surface control point/vertex.
    * If you have direct reference type vertices [x]
    * If you have IndexToDirect reference type the UV
* ByPolygonVertex There will be one mapping coordinate for each vertex, for every polygon of which it is a part. This means that a vertex will have as many mapping coordinates as polygons of which it is a part. (Sorted by polygon, referencing vertex)
* ByPolygon There can be only one mapping coordinate for the whole polygon.
    * One mapping per polygon polygon x has this normal x
    * For each vertex of the polygon then set the normal to x
* ByEdge There will be one mapping coordinate for each unique edge in the mesh. This is meant to be used with smoothing layer elements. (Mapping is referencing the edge id)
* AllSame There can be only one mapping coordinate for the whole surface.
