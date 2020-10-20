/*************************************************************************/
/*  FBXDocument.h                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

/** @file  FBXDocument.h
 *  @brief FBX DOM
 */
#ifndef FBX_DOCUMENT_H
#define FBX_DOCUMENT_H

#include "FBXCommon.h"
#include "FBXParser.h"
#include "FBXProperties.h"
#include "core/math/transform.h"
#include "core/math/vector2.h"
#include "core/math/vector3.h"
#include "core/print_string.h"
#include <stdint.h>
#include <numeric>

#define _AI_CONCAT(a, b) a##b
#define AI_CONCAT(a, b) _AI_CONCAT(a, b)

namespace FBXDocParser {

class Parser;
class Object;
struct ImportSettings;
class Connection;

class PropertyTable;
class Document;
class Material;
class ShapeGeometry;
class LineGeometry;
class Geometry;

class Video;

class AnimationCurve;
class AnimationCurveNode;
class AnimationLayer;
class AnimationStack;

class BlendShapeChannel;
class BlendShape;
class Skin;
class Cluster;

typedef Object *ObjectPtr;
#define new_Object new Object

/** Represents a delay-parsed FBX objects. Many objects in the scene
 *  are not needed by assimp, so it makes no sense to parse them
 *  upfront. */
class LazyObject {
public:
	LazyObject(uint64_t id, const ElementPtr element, const Document &doc);
	~LazyObject();

	ObjectPtr LoadObject();

	/* Casting weak pointers to their templated type safely and preserving ref counting and safety
	 * with lock() keyword to prevent leaking memory
	 */
	template <typename T>
	const T *Get() {
		ObjectPtr ob = LoadObject();
		return dynamic_cast<const T *>(ob);
	}

	uint64_t ID() const {
		return id;
	}

	bool IsBeingConstructed() const {
		return (flags & BEING_CONSTRUCTED) != 0;
	}

	bool FailedToConstruct() const {
		return (flags & FAILED_TO_CONSTRUCT) != 0;
	}

	const ElementPtr GetElement() const {
		return element;
	}

	const Document &GetDocument() const {
		return doc;
	}

private:
	const Document &doc;
	ElementPtr element = nullptr;
	std::shared_ptr<Object> object = nullptr;
	const uint64_t id = 0;

	enum Flags {
		BEING_CONSTRUCTED = 0x1,
		FAILED_TO_CONSTRUCT = 0x2
	};

	unsigned int flags = 0;
};

/** Base class for in-memory (DOM) representations of FBX objects */
class Object {
public:
	Object(uint64_t id, const ElementPtr element, const std::string &name);

	virtual ~Object();

	const ElementPtr SourceElement() const {
		return element;
	}

	const std::string &Name() const {
		return name;
	}

	uint64_t ID() const {
		return id;
	}

protected:
	const ElementPtr element;
	const std::string name;
	const uint64_t id = 0;
};

/** DOM class for generic FBX NoteAttribute blocks. NoteAttribute's just hold a property table,
 *  fixed members are added by deriving classes. */
class NodeAttribute : public Object {
public:
	NodeAttribute(uint64_t id, const ElementPtr element, const Document &doc, const std::string &name);

	virtual ~NodeAttribute();

	const PropertyTable *Props() const {
		return props;
	}

private:
	const PropertyTable *props;
};

/** DOM base class for FBX camera settings attached to a node */
class CameraSwitcher : public NodeAttribute {
public:
	CameraSwitcher(uint64_t id, const ElementPtr element, const Document &doc, const std::string &name);

	virtual ~CameraSwitcher();

	int CameraID() const {
		return cameraId;
	}

	const std::string &CameraName() const {
		return cameraName;
	}

	const std::string &CameraIndexName() const {
		return cameraIndexName;
	}

private:
	int cameraId;
	std::string cameraName;
	std::string cameraIndexName;
};

#define fbx_stringize(a) #a

#define fbx_simple_property(name, type, default_value)                           \
	type name() const {                                                          \
		return PropertyGet<type>(Props(), fbx_stringize(name), (default_value)); \
	}

// XXX improve logging
#define fbx_simple_enum_property(name, type, default_value)                                               \
	type name() const {                                                                                   \
		const int ival = PropertyGet<int>(Props(), fbx_stringize(name), static_cast<int>(default_value)); \
		if (ival < 0 || ival >= AI_CONCAT(type, _MAX)) {                                                  \
			return static_cast<type>(default_value);                                                      \
		}                                                                                                 \
		return static_cast<type>(ival);                                                                   \
	}

class FbxPoseNode;
class FbxPose : public Object {
public:
	FbxPose(uint64_t id, const ElementPtr element, const Document &doc, const std::string &name);

	const std::vector<FbxPoseNode *> &GetBindPoses() const {
		return pose_nodes;
	}

	virtual ~FbxPose();

private:
	std::vector<FbxPoseNode *> pose_nodes;
};

class FbxPoseNode {
public:
	FbxPoseNode(const ElementPtr element, const Document &doc, const std::string &name) {
		const ScopePtr sc = GetRequiredScope(element);

		// get pose node transform
		const ElementPtr Transform = GetRequiredElement(sc, "Matrix", element);
		transform = ReadMatrix(Transform);

		// get node id this pose node is for
		const ElementPtr NodeId = sc->GetElement("Node");
		if (NodeId) {
			target_id = ParseTokenAsInt64(GetRequiredToken(NodeId, 0));
		}

		print_verbose("added posenode " + itos(target_id) + " transform: " + transform);
	}
	virtual ~FbxPoseNode() {
	}

	uint64_t GetNodeID() const {
		return target_id;
	}

	Transform GetBindPose() const {
		return transform;
	}

private:
	uint64_t target_id;
	Transform transform;
};

/** DOM base class for FBX cameras attached to a node */
class Camera : public NodeAttribute {
public:
	Camera(uint64_t id, const ElementPtr element, const Document &doc, const std::string &name);

	virtual ~Camera();

	fbx_simple_property(Position, Vector3, Vector3(0, 0, 0));
	fbx_simple_property(UpVector, Vector3, Vector3(0, 1, 0));
	fbx_simple_property(InterestPosition, Vector3, Vector3(0, 0, 0));

	fbx_simple_property(AspectWidth, float, 1.0f);
	fbx_simple_property(AspectHeight, float, 1.0f);
	fbx_simple_property(FilmWidth, float, 1.0f);
	fbx_simple_property(FilmHeight, float, 1.0f);

	fbx_simple_property(NearPlane, float, 0.1f);
	fbx_simple_property(FarPlane, float, 100.0f);

	fbx_simple_property(FilmAspectRatio, float, 1.0f);
	fbx_simple_property(ApertureMode, int, 0);

	fbx_simple_property(FieldOfView, float, 1.0f);
	fbx_simple_property(FocalLength, float, 1.0f);
};

/** DOM base class for FBX null markers attached to a node */
class Null : public NodeAttribute {
public:
	Null(uint64_t id, const ElementPtr element, const Document &doc, const std::string &name);
	virtual ~Null();
};

/** DOM base class for FBX limb node markers attached to a node */
class LimbNode : public NodeAttribute {
public:
	LimbNode(uint64_t id, const ElementPtr element, const Document &doc, const std::string &name);
	virtual ~LimbNode();
};

/** DOM base class for FBX lights attached to a node */
class Light : public NodeAttribute {
public:
	Light(uint64_t id, const ElementPtr element, const Document &doc, const std::string &name);
	virtual ~Light();

	enum Type {
		Type_Point,
		Type_Directional,
		Type_Spot,
		Type_Area,
		Type_Volume,

		Type_MAX // end-of-enum sentinel
	};

	enum Decay {
		Decay_None,
		Decay_Linear,
		Decay_Quadratic,
		Decay_Cubic,

		Decay_MAX // end-of-enum sentinel
	};

	fbx_simple_property(Color, Vector3, Vector3(1, 1, 1));
	fbx_simple_enum_property(LightType, Type, 0);
	fbx_simple_property(CastLightOnObject, bool, false);
	fbx_simple_property(DrawVolumetricLight, bool, true);
	fbx_simple_property(DrawGroundProjection, bool, true);
	fbx_simple_property(DrawFrontFacingVolumetricLight, bool, false);
	fbx_simple_property(Intensity, float, 100.0f);
	fbx_simple_property(InnerAngle, float, 0.0f);
	fbx_simple_property(OuterAngle, float, 45.0f);
	fbx_simple_property(Fog, int, 50);
	fbx_simple_enum_property(DecayType, Decay, 2);
	fbx_simple_property(DecayStart, float, 1.0f);
	fbx_simple_property(FileName, std::string, "");

	fbx_simple_property(EnableNearAttenuation, bool, false);
	fbx_simple_property(NearAttenuationStart, float, 0.0f);
	fbx_simple_property(NearAttenuationEnd, float, 0.0f);
	fbx_simple_property(EnableFarAttenuation, bool, false);
	fbx_simple_property(FarAttenuationStart, float, 0.0f);
	fbx_simple_property(FarAttenuationEnd, float, 0.0f);

	fbx_simple_property(CastShadows, bool, true);
	fbx_simple_property(ShadowColor, Vector3, Vector3(0, 0, 0));

	fbx_simple_property(AreaLightShape, int, 0);

	fbx_simple_property(LeftBarnDoor, float, 20.0f);
	fbx_simple_property(RightBarnDoor, float, 20.0f);
	fbx_simple_property(TopBarnDoor, float, 20.0f);
	fbx_simple_property(BottomBarnDoor, float, 20.0f);
	fbx_simple_property(EnableBarnDoor, bool, true);
};

class Model;

typedef Model *ModelPtr;
#define new_Model new Model

/** DOM base class for FBX models (even though its semantics are more "node" than "model" */
class Model : public Object {
public:
	enum RotOrder {
		RotOrder_EulerXYZ = 0,
		RotOrder_EulerXZY,
		RotOrder_EulerYZX,
		RotOrder_EulerYXZ,
		RotOrder_EulerZXY,
		RotOrder_EulerZYX,

		RotOrder_SphericXYZ,

		RotOrder_MAX // end-of-enum sentinel
	};

	Model(uint64_t id, const ElementPtr element, const Document &doc, const std::string &name);

	virtual ~Model();

	fbx_simple_property(QuaternionInterpolate, int, 0);

	fbx_simple_property(RotationOffset, Vector3, Vector3());
	fbx_simple_property(RotationPivot, Vector3, Vector3());
	fbx_simple_property(ScalingOffset, Vector3, Vector3());
	fbx_simple_property(ScalingPivot, Vector3, Vector3());
	fbx_simple_property(TranslationActive, bool, false);
	fbx_simple_property(TranslationMin, Vector3, Vector3());
	fbx_simple_property(TranslationMax, Vector3, Vector3());

	fbx_simple_property(TranslationMinX, bool, false);
	fbx_simple_property(TranslationMaxX, bool, false);
	fbx_simple_property(TranslationMinY, bool, false);
	fbx_simple_property(TranslationMaxY, bool, false);
	fbx_simple_property(TranslationMinZ, bool, false);
	fbx_simple_property(TranslationMaxZ, bool, false);

	fbx_simple_enum_property(RotationOrder, RotOrder, 0);
	fbx_simple_property(RotationSpaceForLimitOnly, bool, false);
	fbx_simple_property(RotationStiffnessX, float, 0.0f);
	fbx_simple_property(RotationStiffnessY, float, 0.0f);
	fbx_simple_property(RotationStiffnessZ, float, 0.0f);
	fbx_simple_property(AxisLen, float, 0.0f);

	fbx_simple_property(PreRotation, Vector3, Vector3());
	fbx_simple_property(PostRotation, Vector3, Vector3());
	fbx_simple_property(RotationActive, bool, false);

	fbx_simple_property(RotationMin, Vector3, Vector3());
	fbx_simple_property(RotationMax, Vector3, Vector3());

	fbx_simple_property(RotationMinX, bool, false);
	fbx_simple_property(RotationMaxX, bool, false);
	fbx_simple_property(RotationMinY, bool, false);
	fbx_simple_property(RotationMaxY, bool, false);
	fbx_simple_property(RotationMinZ, bool, false);
	fbx_simple_property(RotationMaxZ, bool, false);
	fbx_simple_enum_property(InheritType, TransformInheritance, 0);

	fbx_simple_property(ScalingActive, bool, false);
	fbx_simple_property(ScalingMin, Vector3, Vector3());
	fbx_simple_property(ScalingMax, Vector3, Vector3(1, 1, 1));
	fbx_simple_property(ScalingMinX, bool, false);
	fbx_simple_property(ScalingMaxX, bool, false);
	fbx_simple_property(ScalingMinY, bool, false);
	fbx_simple_property(ScalingMaxY, bool, false);
	fbx_simple_property(ScalingMinZ, bool, false);
	fbx_simple_property(ScalingMaxZ, bool, false);

	fbx_simple_property(GeometricTranslation, Vector3, Vector3());
	fbx_simple_property(GeometricRotation, Vector3, Vector3());
	fbx_simple_property(GeometricScaling, Vector3, Vector3(1, 1, 1));

	fbx_simple_property(MinDampRangeX, float, 0.0f);
	fbx_simple_property(MinDampRangeY, float, 0.0f);
	fbx_simple_property(MinDampRangeZ, float, 0.0f);
	fbx_simple_property(MaxDampRangeX, float, 0.0f);
	fbx_simple_property(MaxDampRangeY, float, 0.0f);
	fbx_simple_property(MaxDampRangeZ, float, 0.0f);

	fbx_simple_property(MinDampStrengthX, float, 0.0f);
	fbx_simple_property(MinDampStrengthY, float, 0.0f);
	fbx_simple_property(MinDampStrengthZ, float, 0.0f);
	fbx_simple_property(MaxDampStrengthX, float, 0.0f);
	fbx_simple_property(MaxDampStrengthY, float, 0.0f);
	fbx_simple_property(MaxDampStrengthZ, float, 0.0f);

	fbx_simple_property(PreferredAngleX, float, 0.0f);
	fbx_simple_property(PreferredAngleY, float, 0.0f);
	fbx_simple_property(PreferredAngleZ, float, 0.0f);

	fbx_simple_property(Show, bool, true);
	fbx_simple_property(LODBox, bool, false);
	fbx_simple_property(Freeze, bool, false);

	const std::string &Shading() const {
		return shading;
	}

	const std::string &Culling() const {
		return culling;
	}

	const PropertyTable *Props() const {
		return props;
	}

	/** Get material links */
	const std::vector<const Material *> &GetMaterials() const {
		return materials;
	}

	/** Get geometry links */
	const std::vector<const Geometry *> &GetGeometry() const {
		return geometry;
	}

	/** Get node attachments */
	const std::vector<const NodeAttribute *> &GetAttributes() const {
		return attributes;
	}

	/** convenience method to check if the node has a Null node marker */
	bool IsNull() const;

private:
	void ResolveLinks(const ElementPtr element, const Document &doc);

private:
	std::vector<const Material *> materials;
	std::vector<const Geometry *> geometry;
	std::vector<const NodeAttribute *> attributes;

	std::string shading;
	std::string culling;
	const PropertyTable *props = nullptr;
};

class ModelLimbNode : public Model {
public:
	ModelLimbNode(uint64_t id, const ElementPtr element, const Document &doc, const std::string &name);

	virtual ~ModelLimbNode();
};

/** DOM class for generic FBX textures */
class Texture : public Object {
public:
	Texture(uint64_t id, const ElementPtr element, const Document &doc, const std::string &name);

	virtual ~Texture();

	const std::string &Type() const {
		return type;
	}

	const std::string &FileName() const {
		return fileName;
	}

	const std::string &RelativeFilename() const {
		return relativeFileName;
	}

	const std::string &AlphaSource() const {
		return alphaSource;
	}

	const Vector2 &UVTranslation() const {
		return uvTrans;
	}

	const Vector2 &UVScaling() const {
		return uvScaling;
	}

	const PropertyTable *Props() const {
		return props;
	}

	// return a 4-tuple
	const unsigned int *Crop() const {
		return crop;
	}

	const Video *Media() const {
		return media;
	}

private:
	Vector2 uvTrans;
	Vector2 uvScaling;

	std::string type;
	std::string relativeFileName;
	std::string fileName;
	std::string alphaSource;
	const PropertyTable *props = nullptr;

	unsigned int crop[4] = { 0 };

	const Video *media = nullptr;
};

/** DOM class for layered FBX textures */
class LayeredTexture : public Object {
public:
	LayeredTexture(uint64_t id, const ElementPtr element, const Document &doc, const std::string &name);
	virtual ~LayeredTexture();

	// Can only be called after construction of the layered texture object due to construction flag.
	void fillTexture(const Document &doc);

	enum BlendMode {
		BlendMode_Translucent,
		BlendMode_Additive,
		BlendMode_Modulate,
		BlendMode_Modulate2,
		BlendMode_Over,
		BlendMode_Normal,
		BlendMode_Dissolve,
		BlendMode_Darken,
		BlendMode_ColorBurn,
		BlendMode_LinearBurn,
		BlendMode_DarkerColor,
		BlendMode_Lighten,
		BlendMode_Screen,
		BlendMode_ColorDodge,
		BlendMode_LinearDodge,
		BlendMode_LighterColor,
		BlendMode_SoftLight,
		BlendMode_HardLight,
		BlendMode_VividLight,
		BlendMode_LinearLight,
		BlendMode_PinLight,
		BlendMode_HardMix,
		BlendMode_Difference,
		BlendMode_Exclusion,
		BlendMode_Subtract,
		BlendMode_Divide,
		BlendMode_Hue,
		BlendMode_Saturation,
		BlendMode_Color,
		BlendMode_Luminosity,
		BlendMode_Overlay,
		BlendMode_BlendModeCount
	};

	const Texture *getTexture(int index = 0) const {
		return textures[index];
	}
	int textureCount() const {
		return static_cast<int>(textures.size());
	}
	BlendMode GetBlendMode() const {
		return blendMode;
	}
	float Alpha() {
		return alpha;
	}

private:
	std::vector<const Texture *> textures;
	BlendMode blendMode;
	float alpha;
};

typedef std::map<std::string, const Texture *> TextureMap;
typedef std::map<std::string, const LayeredTexture *> LayeredTextureMap;

/** DOM class for generic FBX videos */
class Video : public Object {
public:
	Video(uint64_t id, const ElementPtr element, const Document &doc, const std::string &name);

	virtual ~Video();

	const std::string &Type() const {
		return type;
	}

	const std::string &FileName() const {
		return fileName;
	}

	const std::string &RelativeFilename() const {
		return relativeFileName;
	}

	const PropertyTable *Props() const {
		return props;
	}

	const uint8_t *Content() const {
		return content;
	}

	uint64_t ContentLength() const {
		return contentLength;
	}

	uint8_t *RelinquishContent() {
		uint8_t *ptr = content;
		content = 0;
		return ptr;
	}

	bool operator==(const Video &other) const {
		return (
				type == other.type && relativeFileName == other.relativeFileName && fileName == other.fileName);
	}

	bool operator<(const Video &other) const {
		return std::tie(type, relativeFileName, fileName) < std::tie(other.type, other.relativeFileName, other.fileName);
	}

private:
	std::string type;
	std::string relativeFileName;
	std::string fileName;
	const PropertyTable *props = nullptr;

	uint64_t contentLength;
	uint8_t *content;
};

/** DOM class for generic FBX materials */
class Material : public Object {
public:
	Material(uint64_t id, const ElementPtr element, const Document &doc, const std::string &name);

	virtual ~Material();

	const std::string &GetShadingModel() const {
		return shading;
	}

	bool IsMultilayer() const {
		return multilayer;
	}

	const PropertyTable *Props() const {
		return props;
	}

	const TextureMap &Textures() const {
		return textures;
	}

	const LayeredTextureMap &LayeredTextures() const {
		return layeredTextures;
	}

private:
	std::string shading;
	bool multilayer;
	const PropertyTable *props;

	TextureMap textures;
	LayeredTextureMap layeredTextures;
};

// signed int keys (this can happen!)
typedef std::vector<int64_t> KeyTimeList;
typedef std::vector<float> KeyValueList;

/** Represents a FBX animation curve (i.e. a 1-dimensional set of keyframes and values therefor) */
class AnimationCurve : public Object {
public:
	AnimationCurve(uint64_t id, const ElementPtr element, const std::string &name, const Document &doc);
	virtual ~AnimationCurve();

	/** get list of keyframe positions (time).
     *  Invariant: |GetKeys()| > 0 */
	const KeyTimeList &GetKeys() const {
		return keys;
	}

	/** get list of keyframe values.
      * Invariant: |GetKeys()| == |GetValues()| && |GetKeys()| > 0*/
	const KeyValueList &GetValues() const {
		return values;
	}

	const std::map<int64_t, float> &GetValueTimeTrack() const {
		return keyvalues;
	}

	const std::vector<float> &GetAttributes() const {
		return attributes;
	}

	const std::vector<unsigned int> &GetFlags() const {
		return flags;
	}

private:
	KeyTimeList keys;
	KeyValueList values;
	std::vector<float> attributes;
	std::map<int64_t, float> keyvalues;
	std::vector<unsigned int> flags;
};

/* Typedef for pointers for the animation handler */
typedef std::shared_ptr<AnimationCurve> AnimationCurvePtr;
typedef std::weak_ptr<AnimationCurve> AnimationCurveWeakPtr;
typedef std::map<std::string, const AnimationCurve *> AnimationMap;

/* Animation Curve node ptr */
typedef std::shared_ptr<AnimationCurveNode> AnimationCurveNodePtr;
typedef std::weak_ptr<AnimationCurveNode> AnimationCurveNodeWeakPtr;

/** Represents a FBX animation curve (i.e. a mapping from single animation curves to nodes) */
class AnimationCurveNode : public Object {
public:
	/* the optional white list specifies a list of property names for which the caller
    wants animations for. If the curve node does not match one of these, std::range_error
    will be thrown. */
	AnimationCurveNode(uint64_t id, const ElementPtr element, const std::string &name, const Document &doc,
			const char *const *target_prop_whitelist = nullptr, size_t whitelist_size = 0);

	virtual ~AnimationCurveNode();

	const PropertyTable *Props() const {
		return props;
	}

	const AnimationMap &Curves() const;

	/** Object the curve is assigned to, this can be NULL if the
     *  target object has no DOM representation or could not
     *  be read for other reasons.*/
	Object *Target() const {
		return target;
	}

	Model *TargetAsModel() const {
		return dynamic_cast<Model *>(target);
	}

	NodeAttribute *TargetAsNodeAttribute() const {
		return dynamic_cast<NodeAttribute *>(target);
	}

	/** Property of Target() that is being animated*/
	const std::string &TargetProperty() const {
		return prop;
	}

private:
	Object *target = nullptr;
	const PropertyTable *props;
	mutable AnimationMap curves;
	std::string prop;
	const Document &doc;
};

typedef std::vector<const AnimationCurveNode *> AnimationCurveNodeList;

typedef std::shared_ptr<AnimationLayer> AnimationLayerPtr;
typedef std::weak_ptr<AnimationLayer> AnimationLayerWeakPtr;
typedef std::vector<const AnimationLayer *> AnimationLayerList;

/** Represents a FBX animation layer (i.e. a list of node animations) */
class AnimationLayer : public Object {
public:
	AnimationLayer(uint64_t id, const ElementPtr element, const std::string &name, const Document &doc);
	virtual ~AnimationLayer();

	const PropertyTable *Props() const {
		//ai_assert(props.get());
		return props;
	}

	/* the optional white list specifies a list of property names for which the caller
    wants animations for. Curves not matching this list will not be added to the
    animation layer. */
	const AnimationCurveNodeList Nodes(const char *const *target_prop_whitelist = nullptr, size_t whitelist_size = 0) const;

private:
	const PropertyTable *props;
	const Document &doc;
};

/** Represents a FBX animation stack (i.e. a list of animation layers) */
class AnimationStack : public Object {
public:
	AnimationStack(uint64_t id, const ElementPtr element, const std::string &name, const Document &doc);
	virtual ~AnimationStack();

	fbx_simple_property(LocalStart, int64_t, 0L);
	fbx_simple_property(LocalStop, int64_t, 0L);
	fbx_simple_property(ReferenceStart, int64_t, 0L);
	fbx_simple_property(ReferenceStop, int64_t, 0L);

	const PropertyTable *Props() const {
		return props;
	}

	const AnimationLayerList &Layers() const {
		return layers;
	}

private:
	const PropertyTable *props = nullptr;
	AnimationLayerList layers;
};

/** DOM class for deformers */
class Deformer : public Object {
public:
	Deformer(uint64_t id, const ElementPtr element, const Document &doc, const std::string &name);
	virtual ~Deformer();

	const PropertyTable *Props() const {
		//ai_assert(props.get());
		return props;
	}

private:
	const PropertyTable *props;
};

/** Constraints are from Maya they can help us with BoneAttachments :) **/
class Constraint : public Object {
public:
	Constraint(uint64_t id, const ElementPtr element, const Document &doc, const std::string &name);
	virtual ~Constraint();

private:
	const PropertyTable *props;
};

typedef std::vector<float> WeightArray;
typedef std::vector<unsigned int> WeightIndexArray;

/** DOM class for BlendShapeChannel deformers */
class BlendShapeChannel : public Deformer {
public:
	BlendShapeChannel(uint64_t id, const ElementPtr element, const Document &doc, const std::string &name);

	virtual ~BlendShapeChannel();

	float DeformPercent() const {
		return percent;
	}

	const WeightArray &GetFullWeights() const {
		return fullWeights;
	}

	const std::vector<const ShapeGeometry *> &GetShapeGeometries() const {
		return shapeGeometries;
	}

private:
	float percent;
	WeightArray fullWeights;
	std::vector<const ShapeGeometry *> shapeGeometries;
};

/** DOM class for BlendShape deformers */
class BlendShape : public Deformer {
public:
	BlendShape(uint64_t id, const ElementPtr element, const Document &doc, const std::string &name);

	virtual ~BlendShape();

	const std::vector<const BlendShapeChannel *> &BlendShapeChannels() const {
		return blendShapeChannels;
	}

private:
	std::vector<const BlendShapeChannel *> blendShapeChannels;
};

/** DOM class for skin deformer clusters (aka sub-deformers) */
class Cluster : public Deformer {
public:
	Cluster(uint64_t id, const ElementPtr element, const Document &doc, const std::string &name);

	virtual ~Cluster();

	/** get the list of deformer weights associated with this cluster.
     *  Use #GetIndices() to get the associated vertices. Both arrays
     *  have the same size (and may also be empty). */
	const std::vector<float> &GetWeights() const {
		return weights;
	}

	/** get indices into the vertex data of the geometry associated
     *  with this cluster. Use #GetWeights() to get the associated weights.
     *  Both arrays have the same size (and may also be empty). */
	const std::vector<unsigned int> &GetIndices() const {
		return indices;
	}

	/** */
	const Transform &GetTransform() const {
		return transform;
	}

	const Transform &TransformLink() const {
		return transformLink;
	}

	const Model *TargetNode() const {
		return node;
	}

	const Transform &TransformAssociateModel() const {
		return transformAssociateModel;
	}

	bool TransformAssociateModelValid() const {
		return valid_transformAssociateModel;
	}

	// property is not in the fbx file
	// if the cluster has an associate model
	// we then have an additive type
	enum SkinLinkMode {
		SkinLinkMode_Normalized = 0,
		SkinLinkMode_Additive = 1
	};

	SkinLinkMode GetLinkMode() {
		return link_mode;
	}

private:
	std::vector<float> weights;
	std::vector<unsigned int> indices;

	Transform transform;
	Transform transformLink;
	Transform transformAssociateModel;
	SkinLinkMode link_mode;
	bool valid_transformAssociateModel;
	const Model *node = nullptr;
};

/** DOM class for skin deformers */
class Skin : public Deformer {
public:
	Skin(uint64_t id, const ElementPtr element, const Document &doc, const std::string &name);

	virtual ~Skin();

	float DeformAccuracy() const {
		return accuracy;
	}

	const std::vector<const Cluster *> &Clusters() const {
		return clusters;
	}

	enum SkinType {
		Skin_Rigid = 0,
		Skin_Linear,
		Skin_DualQuaternion,
		Skin_Blend
	};

	const SkinType &GetSkinType() const {
		return skinType;
	}

private:
	float accuracy;
	SkinType skinType;
	std::vector<const Cluster *> clusters;
};

/** Represents a link between two FBX objects. */
class Connection {
public:
	Connection(uint64_t insertionOrder, uint64_t src, uint64_t dest, const std::string &prop, const Document &doc);
	~Connection();

	// note: a connection ensures that the source and dest objects exist, but
	// not that they have DOM representations, so the return value of one of
	// these functions can still be NULL.
	Object *SourceObject() const;
	Object *DestinationObject() const;

	// these, however, are always guaranteed to be valid
	LazyObject *LazySourceObject() const;
	LazyObject *LazyDestinationObject() const;

	/** return the name of the property the connection is attached to.
      * this is an empty string for object to object (OO) connections. */
	const std::string &PropertyName() const {
		return prop;
	}

	uint64_t InsertionOrder() const {
		return insertionOrder;
	}

	int CompareTo(const Connection *c) const {
		//ai_assert(nullptr != c);

		// note: can't subtract because this would overflow uint64_t
		if (InsertionOrder() > c->InsertionOrder()) {
			return 1;
		} else if (InsertionOrder() < c->InsertionOrder()) {
			return -1;
		}
		return 0;
	}

	bool Compare(const Connection *c) const {
		//ai_assert(nullptr != c);

		return InsertionOrder() < c->InsertionOrder();
	}

public:
	uint64_t insertionOrder;
	const std::string prop;

	uint64_t src, dest;
	const Document &doc;
};

// XXX again, unique_ptr would be useful. shared_ptr is too
// bloated since the objects have a well-defined single owner
// during their entire lifetime (Document). FBX files have
// up to many thousands of objects (most of which we never use),
// so the memory overhead for them should be kept at a minimum.
typedef std::map<uint64_t, LazyObject *> ObjectMap;
typedef std::map<std::string, const PropertyTable *> PropertyTemplateMap;
typedef std::multimap<uint64_t, const Connection *> ConnectionMap;

/** DOM class for global document settings, a single instance per document can
 *  be accessed via Document.Globals(). */
class FileGlobalSettings {
public:
	FileGlobalSettings(const Document &doc, const PropertyTable *props);

	~FileGlobalSettings();

	const PropertyTable *Props() const {
		return props;
	}

	const Document &GetDocument() const {
		return doc;
	}

	fbx_simple_property(UpAxis, int, 1);
	fbx_simple_property(UpAxisSign, int, 1);
	fbx_simple_property(FrontAxis, int, 2);
	fbx_simple_property(FrontAxisSign, int, 1);
	fbx_simple_property(CoordAxis, int, 0);
	fbx_simple_property(CoordAxisSign, int, 1);
	fbx_simple_property(OriginalUpAxis, int, 0);
	fbx_simple_property(OriginalUpAxisSign, int, 1);
	fbx_simple_property(UnitScaleFactor, float, 1);
	fbx_simple_property(OriginalUnitScaleFactor, float, 1);
	fbx_simple_property(AmbientColor, Vector3, Vector3(0, 0, 0));
	fbx_simple_property(DefaultCamera, std::string, "");

	enum FrameRate {
		FrameRate_DEFAULT = 0,
		FrameRate_120 = 1,
		FrameRate_100 = 2,
		FrameRate_60 = 3,
		FrameRate_50 = 4,
		FrameRate_48 = 5,
		FrameRate_30 = 6,
		FrameRate_30_DROP = 7,
		FrameRate_NTSC_DROP_FRAME = 8,
		FrameRate_NTSC_FULL_FRAME = 9,
		FrameRate_PAL = 10,
		FrameRate_CINEMA = 11,
		FrameRate_1000 = 12,
		FrameRate_CINEMA_ND = 13,
		FrameRate_CUSTOM = 14,

		FrameRate_MAX // end-of-enum sentinel
	};

	fbx_simple_enum_property(TimeMode, FrameRate, FrameRate_DEFAULT);
	fbx_simple_property(TimeSpanStart, uint64_t, 0L);
	fbx_simple_property(TimeSpanStop, uint64_t, 0L);
	fbx_simple_property(CustomFrameRate, float, -1.0f);

private:
	const PropertyTable *props = nullptr;
	const Document &doc;
};

/** DOM root for a FBX file */
class Document {
public:
	Document(const Parser &parser, const ImportSettings &settings);

	~Document();

	LazyObject *GetObject(uint64_t id) const;

	bool IsSafeToImport() const {
		return SafeToImport;
	}

	bool IsBinary() const {
		return parser.IsBinary();
	}

	unsigned int FBXVersion() const {
		return fbxVersion;
	}

	const std::string &Creator() const {
		return creator;
	}

	// elements (in this order): Year, Month, Day, Hour, Second, Millisecond
	const unsigned int *CreationTimeStamp() const {
		return creationTimeStamp;
	}

	const FileGlobalSettings *GlobalSettingsPtr() const {
		return globals.get();
	}

	const PropertyTemplateMap &Templates() const {
		return templates;
	}

	const ObjectMap &Objects() const {
		return objects;
	}

	const ImportSettings &Settings() const {
		return settings;
	}

	const ConnectionMap &ConnectionsBySource() const {
		return src_connections;
	}

	const ConnectionMap &ConnectionsByDestination() const {
		return dest_connections;
	}

	// note: the implicit rule in all DOM classes is to always resolve
	// from destination to source (since the FBX object hierarchy is,
	// with very few exceptions, a DAG, this avoids cycles). In all
	// cases that may involve back-facing edges in the object graph,
	// use LazyObject::IsBeingConstructed() to check.

	std::vector<const Connection *> GetConnectionsBySourceSequenced(uint64_t source) const;
	std::vector<const Connection *> GetConnectionsByDestinationSequenced(uint64_t dest) const;

	std::vector<const Connection *> GetConnectionsBySourceSequenced(uint64_t source, const char *classname) const;
	std::vector<const Connection *> GetConnectionsByDestinationSequenced(uint64_t dest, const char *classname) const;

	std::vector<const Connection *> GetConnectionsBySourceSequenced(uint64_t source,
			const char *const *classnames, size_t count) const;
	std::vector<const Connection *> GetConnectionsByDestinationSequenced(uint64_t dest,
			const char *const *classnames,
			size_t count) const;

	const std::vector<const AnimationStack *> &AnimationStacks() const;
	const std::vector<uint64_t> &GetAnimationStackIDs() const {
		return animationStacks;
	}

	const std::vector<uint64_t> &GetConstraintStackIDs() const {
		return constraints;
	}

	const std::vector<uint64_t> &GetBindPoseIDs() const {
		return bind_poses;
	};

	const std::vector<uint64_t> &GetMaterialIDs() const {
		return materials;
	};

	const std::vector<uint64_t> &GetSkinIDs() const {
		return skins;
	}

private:
	std::vector<const Connection *> GetConnectionsSequenced(uint64_t id, const ConnectionMap &) const;
	std::vector<const Connection *> GetConnectionsSequenced(uint64_t id, bool is_src,
			const ConnectionMap &,
			const char *const *classnames,
			size_t count) const;
	bool ReadHeader();
	void ReadObjects();
	void ReadPropertyTemplates();
	void ReadConnections();
	void ReadGlobalSettings();

private:
	const ImportSettings &settings;

	ObjectMap objects;
	const Parser &parser;
	bool SafeToImport = false;

	PropertyTemplateMap templates;
	ConnectionMap src_connections;
	ConnectionMap dest_connections;

	unsigned int fbxVersion = 0;
	std::string creator;
	unsigned int creationTimeStamp[7] = { 0 };

	std::vector<uint64_t> animationStacks;
	std::vector<uint64_t> bind_poses;
	// constraints aren't in the tree / at least they are not easy to access.
	std::vector<uint64_t> constraints;
	std::vector<uint64_t> materials;
	std::vector<uint64_t> skins;
	mutable std::vector<const AnimationStack *> animationStacksResolved;
	std::shared_ptr<FileGlobalSettings> globals = nullptr;
};

} // namespace FBXDocParser

namespace std {
template <>
struct hash<const FBXDocParser::Video> {
	std::size_t operator()(const FBXDocParser::Video &video) const {
		using std::hash;
		using std::size_t;
		using std::string;

		size_t res = 17;
		res = res * 31 + hash<string>()(video.Name());
		res = res * 31 + hash<string>()(video.RelativeFilename());
		res = res * 31 + hash<string>()(video.Type());

		return res;
	}
};
} // namespace std

#endif // FBX_DOCUMENT_H
