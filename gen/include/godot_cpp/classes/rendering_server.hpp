/**************************************************************************/
/*  rendering_server.hpp                                                  */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#pragma once

#include <godot_cpp/classes/image.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/rendering_device.hpp>
#include <godot_cpp/core/object.hpp>
#include <godot_cpp/variant/aabb.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/color.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>
#include <godot_cpp/variant/packed_color_array.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>
#include <godot_cpp/variant/packed_int64_array.hpp>
#include <godot_cpp/variant/packed_vector2_array.hpp>
#include <godot_cpp/variant/packed_vector3_array.hpp>
#include <godot_cpp/variant/rect2.hpp>
#include <godot_cpp/variant/rid.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/string_name.hpp>
#include <godot_cpp/variant/transform2d.hpp>
#include <godot_cpp/variant/transform3d.hpp>
#include <godot_cpp/variant/typed_array.hpp>
#include <godot_cpp/variant/variant.hpp>
#include <godot_cpp/variant/vector3i.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

struct Basis;
class Callable;
struct Plane;
struct Vector2;
struct Vector2i;
struct Vector3;

class RenderingServer : public Object {
	GDEXTENSION_CLASS(RenderingServer, Object)

	static RenderingServer *singleton;

public:
	enum TextureType {
		TEXTURE_TYPE_2D = 0,
		TEXTURE_TYPE_LAYERED = 1,
		TEXTURE_TYPE_3D = 2,
	};

	enum TextureLayeredType {
		TEXTURE_LAYERED_2D_ARRAY = 0,
		TEXTURE_LAYERED_CUBEMAP = 1,
		TEXTURE_LAYERED_CUBEMAP_ARRAY = 2,
	};

	enum CubeMapLayer {
		CUBEMAP_LAYER_LEFT = 0,
		CUBEMAP_LAYER_RIGHT = 1,
		CUBEMAP_LAYER_BOTTOM = 2,
		CUBEMAP_LAYER_TOP = 3,
		CUBEMAP_LAYER_FRONT = 4,
		CUBEMAP_LAYER_BACK = 5,
	};

	enum ShaderMode {
		SHADER_SPATIAL = 0,
		SHADER_CANVAS_ITEM = 1,
		SHADER_PARTICLES = 2,
		SHADER_SKY = 3,
		SHADER_FOG = 4,
		SHADER_MAX = 5,
	};

	enum ArrayType {
		ARRAY_VERTEX = 0,
		ARRAY_NORMAL = 1,
		ARRAY_TANGENT = 2,
		ARRAY_COLOR = 3,
		ARRAY_TEX_UV = 4,
		ARRAY_TEX_UV2 = 5,
		ARRAY_CUSTOM0 = 6,
		ARRAY_CUSTOM1 = 7,
		ARRAY_CUSTOM2 = 8,
		ARRAY_CUSTOM3 = 9,
		ARRAY_BONES = 10,
		ARRAY_WEIGHTS = 11,
		ARRAY_INDEX = 12,
		ARRAY_MAX = 13,
	};

	enum ArrayCustomFormat {
		ARRAY_CUSTOM_RGBA8_UNORM = 0,
		ARRAY_CUSTOM_RGBA8_SNORM = 1,
		ARRAY_CUSTOM_RG_HALF = 2,
		ARRAY_CUSTOM_RGBA_HALF = 3,
		ARRAY_CUSTOM_R_FLOAT = 4,
		ARRAY_CUSTOM_RG_FLOAT = 5,
		ARRAY_CUSTOM_RGB_FLOAT = 6,
		ARRAY_CUSTOM_RGBA_FLOAT = 7,
		ARRAY_CUSTOM_MAX = 8,
	};

	enum ArrayFormat : uint64_t {
		ARRAY_FORMAT_VERTEX = 1,
		ARRAY_FORMAT_NORMAL = 2,
		ARRAY_FORMAT_TANGENT = 4,
		ARRAY_FORMAT_COLOR = 8,
		ARRAY_FORMAT_TEX_UV = 16,
		ARRAY_FORMAT_TEX_UV2 = 32,
		ARRAY_FORMAT_CUSTOM0 = 64,
		ARRAY_FORMAT_CUSTOM1 = 128,
		ARRAY_FORMAT_CUSTOM2 = 256,
		ARRAY_FORMAT_CUSTOM3 = 512,
		ARRAY_FORMAT_BONES = 1024,
		ARRAY_FORMAT_WEIGHTS = 2048,
		ARRAY_FORMAT_INDEX = 4096,
		ARRAY_FORMAT_BLEND_SHAPE_MASK = 7,
		ARRAY_FORMAT_CUSTOM_BASE = 13,
		ARRAY_FORMAT_CUSTOM_BITS = 3,
		ARRAY_FORMAT_CUSTOM0_SHIFT = 13,
		ARRAY_FORMAT_CUSTOM1_SHIFT = 16,
		ARRAY_FORMAT_CUSTOM2_SHIFT = 19,
		ARRAY_FORMAT_CUSTOM3_SHIFT = 22,
		ARRAY_FORMAT_CUSTOM_MASK = 7,
		ARRAY_COMPRESS_FLAGS_BASE = 25,
		ARRAY_FLAG_USE_2D_VERTICES = 33554432,
		ARRAY_FLAG_USE_DYNAMIC_UPDATE = 67108864,
		ARRAY_FLAG_USE_8_BONE_WEIGHTS = 134217728,
		ARRAY_FLAG_USES_EMPTY_VERTEX_ARRAY = 268435456,
		ARRAY_FLAG_COMPRESS_ATTRIBUTES = 536870912,
		ARRAY_FLAG_FORMAT_VERSION_BASE = 35,
		ARRAY_FLAG_FORMAT_VERSION_SHIFT = 35,
		ARRAY_FLAG_FORMAT_VERSION_1 = 0,
		ARRAY_FLAG_FORMAT_VERSION_2 = 34359738368,
		ARRAY_FLAG_FORMAT_CURRENT_VERSION = 34359738368,
		ARRAY_FLAG_FORMAT_VERSION_MASK = 255,
	};

	enum PrimitiveType {
		PRIMITIVE_POINTS = 0,
		PRIMITIVE_LINES = 1,
		PRIMITIVE_LINE_STRIP = 2,
		PRIMITIVE_TRIANGLES = 3,
		PRIMITIVE_TRIANGLE_STRIP = 4,
		PRIMITIVE_MAX = 5,
	};

	enum BlendShapeMode {
		BLEND_SHAPE_MODE_NORMALIZED = 0,
		BLEND_SHAPE_MODE_RELATIVE = 1,
	};

	enum MultimeshTransformFormat {
		MULTIMESH_TRANSFORM_2D = 0,
		MULTIMESH_TRANSFORM_3D = 1,
	};

	enum MultimeshPhysicsInterpolationQuality {
		MULTIMESH_INTERP_QUALITY_FAST = 0,
		MULTIMESH_INTERP_QUALITY_HIGH = 1,
	};

	enum LightProjectorFilter {
		LIGHT_PROJECTOR_FILTER_NEAREST = 0,
		LIGHT_PROJECTOR_FILTER_LINEAR = 1,
		LIGHT_PROJECTOR_FILTER_NEAREST_MIPMAPS = 2,
		LIGHT_PROJECTOR_FILTER_LINEAR_MIPMAPS = 3,
		LIGHT_PROJECTOR_FILTER_NEAREST_MIPMAPS_ANISOTROPIC = 4,
		LIGHT_PROJECTOR_FILTER_LINEAR_MIPMAPS_ANISOTROPIC = 5,
	};

	enum LightType {
		LIGHT_DIRECTIONAL = 0,
		LIGHT_OMNI = 1,
		LIGHT_SPOT = 2,
	};

	enum LightParam {
		LIGHT_PARAM_ENERGY = 0,
		LIGHT_PARAM_INDIRECT_ENERGY = 1,
		LIGHT_PARAM_VOLUMETRIC_FOG_ENERGY = 2,
		LIGHT_PARAM_SPECULAR = 3,
		LIGHT_PARAM_RANGE = 4,
		LIGHT_PARAM_SIZE = 5,
		LIGHT_PARAM_ATTENUATION = 6,
		LIGHT_PARAM_SPOT_ANGLE = 7,
		LIGHT_PARAM_SPOT_ATTENUATION = 8,
		LIGHT_PARAM_SHADOW_MAX_DISTANCE = 9,
		LIGHT_PARAM_SHADOW_SPLIT_1_OFFSET = 10,
		LIGHT_PARAM_SHADOW_SPLIT_2_OFFSET = 11,
		LIGHT_PARAM_SHADOW_SPLIT_3_OFFSET = 12,
		LIGHT_PARAM_SHADOW_FADE_START = 13,
		LIGHT_PARAM_SHADOW_NORMAL_BIAS = 14,
		LIGHT_PARAM_SHADOW_BIAS = 15,
		LIGHT_PARAM_SHADOW_PANCAKE_SIZE = 16,
		LIGHT_PARAM_SHADOW_OPACITY = 17,
		LIGHT_PARAM_SHADOW_BLUR = 18,
		LIGHT_PARAM_TRANSMITTANCE_BIAS = 19,
		LIGHT_PARAM_INTENSITY = 20,
		LIGHT_PARAM_MAX = 21,
	};

	enum LightBakeMode {
		LIGHT_BAKE_DISABLED = 0,
		LIGHT_BAKE_STATIC = 1,
		LIGHT_BAKE_DYNAMIC = 2,
	};

	enum LightOmniShadowMode {
		LIGHT_OMNI_SHADOW_DUAL_PARABOLOID = 0,
		LIGHT_OMNI_SHADOW_CUBE = 1,
	};

	enum LightDirectionalShadowMode {
		LIGHT_DIRECTIONAL_SHADOW_ORTHOGONAL = 0,
		LIGHT_DIRECTIONAL_SHADOW_PARALLEL_2_SPLITS = 1,
		LIGHT_DIRECTIONAL_SHADOW_PARALLEL_4_SPLITS = 2,
	};

	enum LightDirectionalSkyMode {
		LIGHT_DIRECTIONAL_SKY_MODE_LIGHT_AND_SKY = 0,
		LIGHT_DIRECTIONAL_SKY_MODE_LIGHT_ONLY = 1,
		LIGHT_DIRECTIONAL_SKY_MODE_SKY_ONLY = 2,
	};

	enum ShadowQuality {
		SHADOW_QUALITY_HARD = 0,
		SHADOW_QUALITY_SOFT_VERY_LOW = 1,
		SHADOW_QUALITY_SOFT_LOW = 2,
		SHADOW_QUALITY_SOFT_MEDIUM = 3,
		SHADOW_QUALITY_SOFT_HIGH = 4,
		SHADOW_QUALITY_SOFT_ULTRA = 5,
		SHADOW_QUALITY_MAX = 6,
	};

	enum ReflectionProbeUpdateMode {
		REFLECTION_PROBE_UPDATE_ONCE = 0,
		REFLECTION_PROBE_UPDATE_ALWAYS = 1,
	};

	enum ReflectionProbeAmbientMode {
		REFLECTION_PROBE_AMBIENT_DISABLED = 0,
		REFLECTION_PROBE_AMBIENT_ENVIRONMENT = 1,
		REFLECTION_PROBE_AMBIENT_COLOR = 2,
	};

	enum DecalTexture {
		DECAL_TEXTURE_ALBEDO = 0,
		DECAL_TEXTURE_NORMAL = 1,
		DECAL_TEXTURE_ORM = 2,
		DECAL_TEXTURE_EMISSION = 3,
		DECAL_TEXTURE_MAX = 4,
	};

	enum DecalFilter {
		DECAL_FILTER_NEAREST = 0,
		DECAL_FILTER_LINEAR = 1,
		DECAL_FILTER_NEAREST_MIPMAPS = 2,
		DECAL_FILTER_LINEAR_MIPMAPS = 3,
		DECAL_FILTER_NEAREST_MIPMAPS_ANISOTROPIC = 4,
		DECAL_FILTER_LINEAR_MIPMAPS_ANISOTROPIC = 5,
	};

	enum VoxelGIQuality {
		VOXEL_GI_QUALITY_LOW = 0,
		VOXEL_GI_QUALITY_HIGH = 1,
	};

	enum ParticlesMode {
		PARTICLES_MODE_2D = 0,
		PARTICLES_MODE_3D = 1,
	};

	enum ParticlesTransformAlign {
		PARTICLES_TRANSFORM_ALIGN_DISABLED = 0,
		PARTICLES_TRANSFORM_ALIGN_Z_BILLBOARD = 1,
		PARTICLES_TRANSFORM_ALIGN_Y_TO_VELOCITY = 2,
		PARTICLES_TRANSFORM_ALIGN_Z_BILLBOARD_Y_TO_VELOCITY = 3,
	};

	enum ParticlesDrawOrder {
		PARTICLES_DRAW_ORDER_INDEX = 0,
		PARTICLES_DRAW_ORDER_LIFETIME = 1,
		PARTICLES_DRAW_ORDER_REVERSE_LIFETIME = 2,
		PARTICLES_DRAW_ORDER_VIEW_DEPTH = 3,
	};

	enum ParticlesCollisionType {
		PARTICLES_COLLISION_TYPE_SPHERE_ATTRACT = 0,
		PARTICLES_COLLISION_TYPE_BOX_ATTRACT = 1,
		PARTICLES_COLLISION_TYPE_VECTOR_FIELD_ATTRACT = 2,
		PARTICLES_COLLISION_TYPE_SPHERE_COLLIDE = 3,
		PARTICLES_COLLISION_TYPE_BOX_COLLIDE = 4,
		PARTICLES_COLLISION_TYPE_SDF_COLLIDE = 5,
		PARTICLES_COLLISION_TYPE_HEIGHTFIELD_COLLIDE = 6,
	};

	enum ParticlesCollisionHeightfieldResolution {
		PARTICLES_COLLISION_HEIGHTFIELD_RESOLUTION_256 = 0,
		PARTICLES_COLLISION_HEIGHTFIELD_RESOLUTION_512 = 1,
		PARTICLES_COLLISION_HEIGHTFIELD_RESOLUTION_1024 = 2,
		PARTICLES_COLLISION_HEIGHTFIELD_RESOLUTION_2048 = 3,
		PARTICLES_COLLISION_HEIGHTFIELD_RESOLUTION_4096 = 4,
		PARTICLES_COLLISION_HEIGHTFIELD_RESOLUTION_8192 = 5,
		PARTICLES_COLLISION_HEIGHTFIELD_RESOLUTION_MAX = 6,
	};

	enum FogVolumeShape {
		FOG_VOLUME_SHAPE_ELLIPSOID = 0,
		FOG_VOLUME_SHAPE_CONE = 1,
		FOG_VOLUME_SHAPE_CYLINDER = 2,
		FOG_VOLUME_SHAPE_BOX = 3,
		FOG_VOLUME_SHAPE_WORLD = 4,
		FOG_VOLUME_SHAPE_MAX = 5,
	};

	enum ViewportScaling3DMode {
		VIEWPORT_SCALING_3D_MODE_BILINEAR = 0,
		VIEWPORT_SCALING_3D_MODE_FSR = 1,
		VIEWPORT_SCALING_3D_MODE_FSR2 = 2,
		VIEWPORT_SCALING_3D_MODE_METALFX_SPATIAL = 3,
		VIEWPORT_SCALING_3D_MODE_METALFX_TEMPORAL = 4,
		VIEWPORT_SCALING_3D_MODE_MAX = 5,
	};

	enum ViewportUpdateMode {
		VIEWPORT_UPDATE_DISABLED = 0,
		VIEWPORT_UPDATE_ONCE = 1,
		VIEWPORT_UPDATE_WHEN_VISIBLE = 2,
		VIEWPORT_UPDATE_WHEN_PARENT_VISIBLE = 3,
		VIEWPORT_UPDATE_ALWAYS = 4,
	};

	enum ViewportClearMode {
		VIEWPORT_CLEAR_ALWAYS = 0,
		VIEWPORT_CLEAR_NEVER = 1,
		VIEWPORT_CLEAR_ONLY_NEXT_FRAME = 2,
	};

	enum ViewportEnvironmentMode {
		VIEWPORT_ENVIRONMENT_DISABLED = 0,
		VIEWPORT_ENVIRONMENT_ENABLED = 1,
		VIEWPORT_ENVIRONMENT_INHERIT = 2,
		VIEWPORT_ENVIRONMENT_MAX = 3,
	};

	enum ViewportSDFOversize {
		VIEWPORT_SDF_OVERSIZE_100_PERCENT = 0,
		VIEWPORT_SDF_OVERSIZE_120_PERCENT = 1,
		VIEWPORT_SDF_OVERSIZE_150_PERCENT = 2,
		VIEWPORT_SDF_OVERSIZE_200_PERCENT = 3,
		VIEWPORT_SDF_OVERSIZE_MAX = 4,
	};

	enum ViewportSDFScale {
		VIEWPORT_SDF_SCALE_100_PERCENT = 0,
		VIEWPORT_SDF_SCALE_50_PERCENT = 1,
		VIEWPORT_SDF_SCALE_25_PERCENT = 2,
		VIEWPORT_SDF_SCALE_MAX = 3,
	};

	enum ViewportMSAA {
		VIEWPORT_MSAA_DISABLED = 0,
		VIEWPORT_MSAA_2X = 1,
		VIEWPORT_MSAA_4X = 2,
		VIEWPORT_MSAA_8X = 3,
		VIEWPORT_MSAA_MAX = 4,
	};

	enum ViewportAnisotropicFiltering {
		VIEWPORT_ANISOTROPY_DISABLED = 0,
		VIEWPORT_ANISOTROPY_2X = 1,
		VIEWPORT_ANISOTROPY_4X = 2,
		VIEWPORT_ANISOTROPY_8X = 3,
		VIEWPORT_ANISOTROPY_16X = 4,
		VIEWPORT_ANISOTROPY_MAX = 5,
	};

	enum ViewportScreenSpaceAA {
		VIEWPORT_SCREEN_SPACE_AA_DISABLED = 0,
		VIEWPORT_SCREEN_SPACE_AA_FXAA = 1,
		VIEWPORT_SCREEN_SPACE_AA_SMAA = 2,
		VIEWPORT_SCREEN_SPACE_AA_MAX = 3,
	};

	enum ViewportOcclusionCullingBuildQuality {
		VIEWPORT_OCCLUSION_BUILD_QUALITY_LOW = 0,
		VIEWPORT_OCCLUSION_BUILD_QUALITY_MEDIUM = 1,
		VIEWPORT_OCCLUSION_BUILD_QUALITY_HIGH = 2,
	};

	enum ViewportRenderInfo {
		VIEWPORT_RENDER_INFO_OBJECTS_IN_FRAME = 0,
		VIEWPORT_RENDER_INFO_PRIMITIVES_IN_FRAME = 1,
		VIEWPORT_RENDER_INFO_DRAW_CALLS_IN_FRAME = 2,
		VIEWPORT_RENDER_INFO_MAX = 3,
	};

	enum ViewportRenderInfoType {
		VIEWPORT_RENDER_INFO_TYPE_VISIBLE = 0,
		VIEWPORT_RENDER_INFO_TYPE_SHADOW = 1,
		VIEWPORT_RENDER_INFO_TYPE_CANVAS = 2,
		VIEWPORT_RENDER_INFO_TYPE_MAX = 3,
	};

	enum ViewportDebugDraw {
		VIEWPORT_DEBUG_DRAW_DISABLED = 0,
		VIEWPORT_DEBUG_DRAW_UNSHADED = 1,
		VIEWPORT_DEBUG_DRAW_LIGHTING = 2,
		VIEWPORT_DEBUG_DRAW_OVERDRAW = 3,
		VIEWPORT_DEBUG_DRAW_WIREFRAME = 4,
		VIEWPORT_DEBUG_DRAW_NORMAL_BUFFER = 5,
		VIEWPORT_DEBUG_DRAW_VOXEL_GI_ALBEDO = 6,
		VIEWPORT_DEBUG_DRAW_VOXEL_GI_LIGHTING = 7,
		VIEWPORT_DEBUG_DRAW_VOXEL_GI_EMISSION = 8,
		VIEWPORT_DEBUG_DRAW_SHADOW_ATLAS = 9,
		VIEWPORT_DEBUG_DRAW_DIRECTIONAL_SHADOW_ATLAS = 10,
		VIEWPORT_DEBUG_DRAW_SCENE_LUMINANCE = 11,
		VIEWPORT_DEBUG_DRAW_SSAO = 12,
		VIEWPORT_DEBUG_DRAW_SSIL = 13,
		VIEWPORT_DEBUG_DRAW_PSSM_SPLITS = 14,
		VIEWPORT_DEBUG_DRAW_DECAL_ATLAS = 15,
		VIEWPORT_DEBUG_DRAW_SDFGI = 16,
		VIEWPORT_DEBUG_DRAW_SDFGI_PROBES = 17,
		VIEWPORT_DEBUG_DRAW_GI_BUFFER = 18,
		VIEWPORT_DEBUG_DRAW_DISABLE_LOD = 19,
		VIEWPORT_DEBUG_DRAW_CLUSTER_OMNI_LIGHTS = 20,
		VIEWPORT_DEBUG_DRAW_CLUSTER_SPOT_LIGHTS = 21,
		VIEWPORT_DEBUG_DRAW_CLUSTER_DECALS = 22,
		VIEWPORT_DEBUG_DRAW_CLUSTER_REFLECTION_PROBES = 23,
		VIEWPORT_DEBUG_DRAW_OCCLUDERS = 24,
		VIEWPORT_DEBUG_DRAW_MOTION_VECTORS = 25,
		VIEWPORT_DEBUG_DRAW_INTERNAL_BUFFER = 26,
	};

	enum ViewportVRSMode {
		VIEWPORT_VRS_DISABLED = 0,
		VIEWPORT_VRS_TEXTURE = 1,
		VIEWPORT_VRS_XR = 2,
		VIEWPORT_VRS_MAX = 3,
	};

	enum ViewportVRSUpdateMode {
		VIEWPORT_VRS_UPDATE_DISABLED = 0,
		VIEWPORT_VRS_UPDATE_ONCE = 1,
		VIEWPORT_VRS_UPDATE_ALWAYS = 2,
		VIEWPORT_VRS_UPDATE_MAX = 3,
	};

	enum SkyMode {
		SKY_MODE_AUTOMATIC = 0,
		SKY_MODE_QUALITY = 1,
		SKY_MODE_INCREMENTAL = 2,
		SKY_MODE_REALTIME = 3,
	};

	enum CompositorEffectFlags {
		COMPOSITOR_EFFECT_FLAG_ACCESS_RESOLVED_COLOR = 1,
		COMPOSITOR_EFFECT_FLAG_ACCESS_RESOLVED_DEPTH = 2,
		COMPOSITOR_EFFECT_FLAG_NEEDS_MOTION_VECTORS = 4,
		COMPOSITOR_EFFECT_FLAG_NEEDS_ROUGHNESS = 8,
		COMPOSITOR_EFFECT_FLAG_NEEDS_SEPARATE_SPECULAR = 16,
	};

	enum CompositorEffectCallbackType {
		COMPOSITOR_EFFECT_CALLBACK_TYPE_PRE_OPAQUE = 0,
		COMPOSITOR_EFFECT_CALLBACK_TYPE_POST_OPAQUE = 1,
		COMPOSITOR_EFFECT_CALLBACK_TYPE_POST_SKY = 2,
		COMPOSITOR_EFFECT_CALLBACK_TYPE_PRE_TRANSPARENT = 3,
		COMPOSITOR_EFFECT_CALLBACK_TYPE_POST_TRANSPARENT = 4,
		COMPOSITOR_EFFECT_CALLBACK_TYPE_ANY = -1,
	};

	enum EnvironmentBG {
		ENV_BG_CLEAR_COLOR = 0,
		ENV_BG_COLOR = 1,
		ENV_BG_SKY = 2,
		ENV_BG_CANVAS = 3,
		ENV_BG_KEEP = 4,
		ENV_BG_CAMERA_FEED = 5,
		ENV_BG_MAX = 6,
	};

	enum EnvironmentAmbientSource {
		ENV_AMBIENT_SOURCE_BG = 0,
		ENV_AMBIENT_SOURCE_DISABLED = 1,
		ENV_AMBIENT_SOURCE_COLOR = 2,
		ENV_AMBIENT_SOURCE_SKY = 3,
	};

	enum EnvironmentReflectionSource {
		ENV_REFLECTION_SOURCE_BG = 0,
		ENV_REFLECTION_SOURCE_DISABLED = 1,
		ENV_REFLECTION_SOURCE_SKY = 2,
	};

	enum EnvironmentGlowBlendMode {
		ENV_GLOW_BLEND_MODE_ADDITIVE = 0,
		ENV_GLOW_BLEND_MODE_SCREEN = 1,
		ENV_GLOW_BLEND_MODE_SOFTLIGHT = 2,
		ENV_GLOW_BLEND_MODE_REPLACE = 3,
		ENV_GLOW_BLEND_MODE_MIX = 4,
	};

	enum EnvironmentFogMode {
		ENV_FOG_MODE_EXPONENTIAL = 0,
		ENV_FOG_MODE_DEPTH = 1,
	};

	enum EnvironmentToneMapper {
		ENV_TONE_MAPPER_LINEAR = 0,
		ENV_TONE_MAPPER_REINHARD = 1,
		ENV_TONE_MAPPER_FILMIC = 2,
		ENV_TONE_MAPPER_ACES = 3,
		ENV_TONE_MAPPER_AGX = 4,
	};

	enum EnvironmentSSRRoughnessQuality {
		ENV_SSR_ROUGHNESS_QUALITY_DISABLED = 0,
		ENV_SSR_ROUGHNESS_QUALITY_LOW = 1,
		ENV_SSR_ROUGHNESS_QUALITY_MEDIUM = 2,
		ENV_SSR_ROUGHNESS_QUALITY_HIGH = 3,
	};

	enum EnvironmentSSAOQuality {
		ENV_SSAO_QUALITY_VERY_LOW = 0,
		ENV_SSAO_QUALITY_LOW = 1,
		ENV_SSAO_QUALITY_MEDIUM = 2,
		ENV_SSAO_QUALITY_HIGH = 3,
		ENV_SSAO_QUALITY_ULTRA = 4,
	};

	enum EnvironmentSSILQuality {
		ENV_SSIL_QUALITY_VERY_LOW = 0,
		ENV_SSIL_QUALITY_LOW = 1,
		ENV_SSIL_QUALITY_MEDIUM = 2,
		ENV_SSIL_QUALITY_HIGH = 3,
		ENV_SSIL_QUALITY_ULTRA = 4,
	};

	enum EnvironmentSDFGIYScale {
		ENV_SDFGI_Y_SCALE_50_PERCENT = 0,
		ENV_SDFGI_Y_SCALE_75_PERCENT = 1,
		ENV_SDFGI_Y_SCALE_100_PERCENT = 2,
	};

	enum EnvironmentSDFGIRayCount {
		ENV_SDFGI_RAY_COUNT_4 = 0,
		ENV_SDFGI_RAY_COUNT_8 = 1,
		ENV_SDFGI_RAY_COUNT_16 = 2,
		ENV_SDFGI_RAY_COUNT_32 = 3,
		ENV_SDFGI_RAY_COUNT_64 = 4,
		ENV_SDFGI_RAY_COUNT_96 = 5,
		ENV_SDFGI_RAY_COUNT_128 = 6,
		ENV_SDFGI_RAY_COUNT_MAX = 7,
	};

	enum EnvironmentSDFGIFramesToConverge {
		ENV_SDFGI_CONVERGE_IN_5_FRAMES = 0,
		ENV_SDFGI_CONVERGE_IN_10_FRAMES = 1,
		ENV_SDFGI_CONVERGE_IN_15_FRAMES = 2,
		ENV_SDFGI_CONVERGE_IN_20_FRAMES = 3,
		ENV_SDFGI_CONVERGE_IN_25_FRAMES = 4,
		ENV_SDFGI_CONVERGE_IN_30_FRAMES = 5,
		ENV_SDFGI_CONVERGE_MAX = 6,
	};

	enum EnvironmentSDFGIFramesToUpdateLight {
		ENV_SDFGI_UPDATE_LIGHT_IN_1_FRAME = 0,
		ENV_SDFGI_UPDATE_LIGHT_IN_2_FRAMES = 1,
		ENV_SDFGI_UPDATE_LIGHT_IN_4_FRAMES = 2,
		ENV_SDFGI_UPDATE_LIGHT_IN_8_FRAMES = 3,
		ENV_SDFGI_UPDATE_LIGHT_IN_16_FRAMES = 4,
		ENV_SDFGI_UPDATE_LIGHT_MAX = 5,
	};

	enum SubSurfaceScatteringQuality {
		SUB_SURFACE_SCATTERING_QUALITY_DISABLED = 0,
		SUB_SURFACE_SCATTERING_QUALITY_LOW = 1,
		SUB_SURFACE_SCATTERING_QUALITY_MEDIUM = 2,
		SUB_SURFACE_SCATTERING_QUALITY_HIGH = 3,
	};

	enum DOFBokehShape {
		DOF_BOKEH_BOX = 0,
		DOF_BOKEH_HEXAGON = 1,
		DOF_BOKEH_CIRCLE = 2,
	};

	enum DOFBlurQuality {
		DOF_BLUR_QUALITY_VERY_LOW = 0,
		DOF_BLUR_QUALITY_LOW = 1,
		DOF_BLUR_QUALITY_MEDIUM = 2,
		DOF_BLUR_QUALITY_HIGH = 3,
	};

	enum InstanceType {
		INSTANCE_NONE = 0,
		INSTANCE_MESH = 1,
		INSTANCE_MULTIMESH = 2,
		INSTANCE_PARTICLES = 3,
		INSTANCE_PARTICLES_COLLISION = 4,
		INSTANCE_LIGHT = 5,
		INSTANCE_REFLECTION_PROBE = 6,
		INSTANCE_DECAL = 7,
		INSTANCE_VOXEL_GI = 8,
		INSTANCE_LIGHTMAP = 9,
		INSTANCE_OCCLUDER = 10,
		INSTANCE_VISIBLITY_NOTIFIER = 11,
		INSTANCE_FOG_VOLUME = 12,
		INSTANCE_MAX = 13,
		INSTANCE_GEOMETRY_MASK = 14,
	};

	enum InstanceFlags {
		INSTANCE_FLAG_USE_BAKED_LIGHT = 0,
		INSTANCE_FLAG_USE_DYNAMIC_GI = 1,
		INSTANCE_FLAG_DRAW_NEXT_FRAME_IF_VISIBLE = 2,
		INSTANCE_FLAG_IGNORE_OCCLUSION_CULLING = 3,
		INSTANCE_FLAG_MAX = 4,
	};

	enum ShadowCastingSetting {
		SHADOW_CASTING_SETTING_OFF = 0,
		SHADOW_CASTING_SETTING_ON = 1,
		SHADOW_CASTING_SETTING_DOUBLE_SIDED = 2,
		SHADOW_CASTING_SETTING_SHADOWS_ONLY = 3,
	};

	enum VisibilityRangeFadeMode {
		VISIBILITY_RANGE_FADE_DISABLED = 0,
		VISIBILITY_RANGE_FADE_SELF = 1,
		VISIBILITY_RANGE_FADE_DEPENDENCIES = 2,
	};

	enum BakeChannels {
		BAKE_CHANNEL_ALBEDO_ALPHA = 0,
		BAKE_CHANNEL_NORMAL = 1,
		BAKE_CHANNEL_ORM = 2,
		BAKE_CHANNEL_EMISSION = 3,
	};

	enum CanvasTextureChannel {
		CANVAS_TEXTURE_CHANNEL_DIFFUSE = 0,
		CANVAS_TEXTURE_CHANNEL_NORMAL = 1,
		CANVAS_TEXTURE_CHANNEL_SPECULAR = 2,
	};

	enum NinePatchAxisMode {
		NINE_PATCH_STRETCH = 0,
		NINE_PATCH_TILE = 1,
		NINE_PATCH_TILE_FIT = 2,
	};

	enum CanvasItemTextureFilter {
		CANVAS_ITEM_TEXTURE_FILTER_DEFAULT = 0,
		CANVAS_ITEM_TEXTURE_FILTER_NEAREST = 1,
		CANVAS_ITEM_TEXTURE_FILTER_LINEAR = 2,
		CANVAS_ITEM_TEXTURE_FILTER_NEAREST_WITH_MIPMAPS = 3,
		CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS = 4,
		CANVAS_ITEM_TEXTURE_FILTER_NEAREST_WITH_MIPMAPS_ANISOTROPIC = 5,
		CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS_ANISOTROPIC = 6,
		CANVAS_ITEM_TEXTURE_FILTER_MAX = 7,
	};

	enum CanvasItemTextureRepeat {
		CANVAS_ITEM_TEXTURE_REPEAT_DEFAULT = 0,
		CANVAS_ITEM_TEXTURE_REPEAT_DISABLED = 1,
		CANVAS_ITEM_TEXTURE_REPEAT_ENABLED = 2,
		CANVAS_ITEM_TEXTURE_REPEAT_MIRROR = 3,
		CANVAS_ITEM_TEXTURE_REPEAT_MAX = 4,
	};

	enum CanvasGroupMode {
		CANVAS_GROUP_MODE_DISABLED = 0,
		CANVAS_GROUP_MODE_CLIP_ONLY = 1,
		CANVAS_GROUP_MODE_CLIP_AND_DRAW = 2,
		CANVAS_GROUP_MODE_TRANSPARENT = 3,
	};

	enum CanvasLightMode {
		CANVAS_LIGHT_MODE_POINT = 0,
		CANVAS_LIGHT_MODE_DIRECTIONAL = 1,
	};

	enum CanvasLightBlendMode {
		CANVAS_LIGHT_BLEND_MODE_ADD = 0,
		CANVAS_LIGHT_BLEND_MODE_SUB = 1,
		CANVAS_LIGHT_BLEND_MODE_MIX = 2,
	};

	enum CanvasLightShadowFilter {
		CANVAS_LIGHT_FILTER_NONE = 0,
		CANVAS_LIGHT_FILTER_PCF5 = 1,
		CANVAS_LIGHT_FILTER_PCF13 = 2,
		CANVAS_LIGHT_FILTER_MAX = 3,
	};

	enum CanvasOccluderPolygonCullMode {
		CANVAS_OCCLUDER_POLYGON_CULL_DISABLED = 0,
		CANVAS_OCCLUDER_POLYGON_CULL_CLOCKWISE = 1,
		CANVAS_OCCLUDER_POLYGON_CULL_COUNTER_CLOCKWISE = 2,
	};

	enum GlobalShaderParameterType {
		GLOBAL_VAR_TYPE_BOOL = 0,
		GLOBAL_VAR_TYPE_BVEC2 = 1,
		GLOBAL_VAR_TYPE_BVEC3 = 2,
		GLOBAL_VAR_TYPE_BVEC4 = 3,
		GLOBAL_VAR_TYPE_INT = 4,
		GLOBAL_VAR_TYPE_IVEC2 = 5,
		GLOBAL_VAR_TYPE_IVEC3 = 6,
		GLOBAL_VAR_TYPE_IVEC4 = 7,
		GLOBAL_VAR_TYPE_RECT2I = 8,
		GLOBAL_VAR_TYPE_UINT = 9,
		GLOBAL_VAR_TYPE_UVEC2 = 10,
		GLOBAL_VAR_TYPE_UVEC3 = 11,
		GLOBAL_VAR_TYPE_UVEC4 = 12,
		GLOBAL_VAR_TYPE_FLOAT = 13,
		GLOBAL_VAR_TYPE_VEC2 = 14,
		GLOBAL_VAR_TYPE_VEC3 = 15,
		GLOBAL_VAR_TYPE_VEC4 = 16,
		GLOBAL_VAR_TYPE_COLOR = 17,
		GLOBAL_VAR_TYPE_RECT2 = 18,
		GLOBAL_VAR_TYPE_MAT2 = 19,
		GLOBAL_VAR_TYPE_MAT3 = 20,
		GLOBAL_VAR_TYPE_MAT4 = 21,
		GLOBAL_VAR_TYPE_TRANSFORM_2D = 22,
		GLOBAL_VAR_TYPE_TRANSFORM = 23,
		GLOBAL_VAR_TYPE_SAMPLER2D = 24,
		GLOBAL_VAR_TYPE_SAMPLER2DARRAY = 25,
		GLOBAL_VAR_TYPE_SAMPLER3D = 26,
		GLOBAL_VAR_TYPE_SAMPLERCUBE = 27,
		GLOBAL_VAR_TYPE_SAMPLEREXT = 28,
		GLOBAL_VAR_TYPE_MAX = 29,
	};

	enum RenderingInfo {
		RENDERING_INFO_TOTAL_OBJECTS_IN_FRAME = 0,
		RENDERING_INFO_TOTAL_PRIMITIVES_IN_FRAME = 1,
		RENDERING_INFO_TOTAL_DRAW_CALLS_IN_FRAME = 2,
		RENDERING_INFO_TEXTURE_MEM_USED = 3,
		RENDERING_INFO_BUFFER_MEM_USED = 4,
		RENDERING_INFO_VIDEO_MEM_USED = 5,
		RENDERING_INFO_PIPELINE_COMPILATIONS_CANVAS = 6,
		RENDERING_INFO_PIPELINE_COMPILATIONS_MESH = 7,
		RENDERING_INFO_PIPELINE_COMPILATIONS_SURFACE = 8,
		RENDERING_INFO_PIPELINE_COMPILATIONS_DRAW = 9,
		RENDERING_INFO_PIPELINE_COMPILATIONS_SPECIALIZATION = 10,
	};

	enum PipelineSource {
		PIPELINE_SOURCE_CANVAS = 0,
		PIPELINE_SOURCE_MESH = 1,
		PIPELINE_SOURCE_SURFACE = 2,
		PIPELINE_SOURCE_DRAW = 3,
		PIPELINE_SOURCE_SPECIALIZATION = 4,
		PIPELINE_SOURCE_MAX = 5,
	};

	enum SplashStretchMode {
		SPLASH_STRETCH_MODE_DISABLED = 0,
		SPLASH_STRETCH_MODE_KEEP = 1,
		SPLASH_STRETCH_MODE_KEEP_WIDTH = 2,
		SPLASH_STRETCH_MODE_KEEP_HEIGHT = 3,
		SPLASH_STRETCH_MODE_COVER = 4,
		SPLASH_STRETCH_MODE_IGNORE = 5,
	};

	enum Features {
		FEATURE_SHADERS = 0,
		FEATURE_MULTITHREADED = 1,
	};

	static const int NO_INDEX_ARRAY = -1;
	static const int ARRAY_WEIGHTS_SIZE = 4;
	static const int CANVAS_ITEM_Z_MIN = -4096;
	static const int CANVAS_ITEM_Z_MAX = 4096;
	static const int CANVAS_LAYER_MIN = -2147483647 - 1;
	static const int CANVAS_LAYER_MAX = 2147483647;
	static const int MAX_GLOW_LEVELS = 7;
	static const int MAX_CURSORS = 8;
	static const int MAX_2D_DIRECTIONAL_LIGHTS = 8;
	static const int MAX_MESH_SURFACES = 256;
	static const int MATERIAL_RENDER_PRIORITY_MIN = -128;
	static const int MATERIAL_RENDER_PRIORITY_MAX = 127;
	static const int ARRAY_CUSTOM_COUNT = 4;
	static const int PARTICLES_EMIT_FLAG_POSITION = 1;
	static const int PARTICLES_EMIT_FLAG_ROTATION_SCALE = 2;
	static const int PARTICLES_EMIT_FLAG_VELOCITY = 4;
	static const int PARTICLES_EMIT_FLAG_COLOR = 8;
	static const int PARTICLES_EMIT_FLAG_CUSTOM = 16;

	static RenderingServer *get_singleton();

	RID texture_2d_create(const Ref<Image> &p_image);
	RID texture_2d_layered_create(const TypedArray<Ref<Image>> &p_layers, RenderingServer::TextureLayeredType p_layered_type);
	RID texture_3d_create(Image::Format p_format, int32_t p_width, int32_t p_height, int32_t p_depth, bool p_mipmaps, const TypedArray<Ref<Image>> &p_data);
	RID texture_proxy_create(const RID &p_base);
	RID texture_create_from_native_handle(RenderingServer::TextureType p_type, Image::Format p_format, uint64_t p_native_handle, int32_t p_width, int32_t p_height, int32_t p_depth, int32_t p_layers = 1, RenderingServer::TextureLayeredType p_layered_type = (RenderingServer::TextureLayeredType)0);
	void texture_2d_update(const RID &p_texture, const Ref<Image> &p_image, int32_t p_layer);
	void texture_3d_update(const RID &p_texture, const TypedArray<Ref<Image>> &p_data);
	void texture_proxy_update(const RID &p_texture, const RID &p_proxy_to);
	RID texture_2d_placeholder_create();
	RID texture_2d_layered_placeholder_create(RenderingServer::TextureLayeredType p_layered_type);
	RID texture_3d_placeholder_create();
	Ref<Image> texture_2d_get(const RID &p_texture) const;
	Ref<Image> texture_2d_layer_get(const RID &p_texture, int32_t p_layer) const;
	TypedArray<Ref<Image>> texture_3d_get(const RID &p_texture) const;
	void texture_replace(const RID &p_texture, const RID &p_by_texture);
	void texture_set_size_override(const RID &p_texture, int32_t p_width, int32_t p_height);
	void texture_set_path(const RID &p_texture, const String &p_path);
	String texture_get_path(const RID &p_texture) const;
	Image::Format texture_get_format(const RID &p_texture) const;
	void texture_set_force_redraw_if_visible(const RID &p_texture, bool p_enable);
	RID texture_rd_create(const RID &p_rd_texture, RenderingServer::TextureLayeredType p_layer_type = (RenderingServer::TextureLayeredType)0);
	RID texture_get_rd_texture(const RID &p_texture, bool p_srgb = false) const;
	uint64_t texture_get_native_handle(const RID &p_texture, bool p_srgb = false) const;
	RID shader_create();
	void shader_set_code(const RID &p_shader, const String &p_code);
	void shader_set_path_hint(const RID &p_shader, const String &p_path);
	String shader_get_code(const RID &p_shader) const;
	TypedArray<Dictionary> get_shader_parameter_list(const RID &p_shader) const;
	Variant shader_get_parameter_default(const RID &p_shader, const StringName &p_name) const;
	void shader_set_default_texture_parameter(const RID &p_shader, const StringName &p_name, const RID &p_texture, int32_t p_index = 0);
	RID shader_get_default_texture_parameter(const RID &p_shader, const StringName &p_name, int32_t p_index = 0) const;
	RID material_create();
	void material_set_shader(const RID &p_shader_material, const RID &p_shader);
	void material_set_param(const RID &p_material, const StringName &p_parameter, const Variant &p_value);
	Variant material_get_param(const RID &p_material, const StringName &p_parameter) const;
	void material_set_render_priority(const RID &p_material, int32_t p_priority);
	void material_set_next_pass(const RID &p_material, const RID &p_next_material);
	void material_set_use_debanding(bool p_enable);
	RID mesh_create_from_surfaces(const TypedArray<Dictionary> &p_surfaces, int32_t p_blend_shape_count = 0);
	RID mesh_create();
	uint32_t mesh_surface_get_format_offset(BitField<RenderingServer::ArrayFormat> p_format, int32_t p_vertex_count, int32_t p_array_index) const;
	uint32_t mesh_surface_get_format_vertex_stride(BitField<RenderingServer::ArrayFormat> p_format, int32_t p_vertex_count) const;
	uint32_t mesh_surface_get_format_normal_tangent_stride(BitField<RenderingServer::ArrayFormat> p_format, int32_t p_vertex_count) const;
	uint32_t mesh_surface_get_format_attribute_stride(BitField<RenderingServer::ArrayFormat> p_format, int32_t p_vertex_count) const;
	uint32_t mesh_surface_get_format_skin_stride(BitField<RenderingServer::ArrayFormat> p_format, int32_t p_vertex_count) const;
	uint32_t mesh_surface_get_format_index_stride(BitField<RenderingServer::ArrayFormat> p_format, int32_t p_vertex_count) const;
	void mesh_add_surface(const RID &p_mesh, const Dictionary &p_surface);
	void mesh_add_surface_from_arrays(const RID &p_mesh, RenderingServer::PrimitiveType p_primitive, const Array &p_arrays, const Array &p_blend_shapes = Array(), const Dictionary &p_lods = Dictionary(), BitField<RenderingServer::ArrayFormat> p_compress_format = (BitField<RenderingServer::ArrayFormat>)0);
	int32_t mesh_get_blend_shape_count(const RID &p_mesh) const;
	void mesh_set_blend_shape_mode(const RID &p_mesh, RenderingServer::BlendShapeMode p_mode);
	RenderingServer::BlendShapeMode mesh_get_blend_shape_mode(const RID &p_mesh) const;
	void mesh_surface_set_material(const RID &p_mesh, int32_t p_surface, const RID &p_material);
	RID mesh_surface_get_material(const RID &p_mesh, int32_t p_surface) const;
	Dictionary mesh_get_surface(const RID &p_mesh, int32_t p_surface);
	Array mesh_surface_get_arrays(const RID &p_mesh, int32_t p_surface) const;
	TypedArray<Array> mesh_surface_get_blend_shape_arrays(const RID &p_mesh, int32_t p_surface) const;
	int32_t mesh_get_surface_count(const RID &p_mesh) const;
	void mesh_set_custom_aabb(const RID &p_mesh, const AABB &p_aabb);
	AABB mesh_get_custom_aabb(const RID &p_mesh) const;
	void mesh_surface_remove(const RID &p_mesh, int32_t p_surface);
	void mesh_clear(const RID &p_mesh);
	void mesh_surface_update_vertex_region(const RID &p_mesh, int32_t p_surface, int32_t p_offset, const PackedByteArray &p_data);
	void mesh_surface_update_attribute_region(const RID &p_mesh, int32_t p_surface, int32_t p_offset, const PackedByteArray &p_data);
	void mesh_surface_update_skin_region(const RID &p_mesh, int32_t p_surface, int32_t p_offset, const PackedByteArray &p_data);
	void mesh_surface_update_index_region(const RID &p_mesh, int32_t p_surface, int32_t p_offset, const PackedByteArray &p_data);
	void mesh_set_shadow_mesh(const RID &p_mesh, const RID &p_shadow_mesh);
	RID multimesh_create();
	void multimesh_allocate_data(const RID &p_multimesh, int32_t p_instances, RenderingServer::MultimeshTransformFormat p_transform_format, bool p_color_format = false, bool p_custom_data_format = false, bool p_use_indirect = false);
	int32_t multimesh_get_instance_count(const RID &p_multimesh) const;
	void multimesh_set_mesh(const RID &p_multimesh, const RID &p_mesh);
	void multimesh_instance_set_transform(const RID &p_multimesh, int32_t p_index, const Transform3D &p_transform);
	void multimesh_instance_set_transform_2d(const RID &p_multimesh, int32_t p_index, const Transform2D &p_transform);
	void multimesh_instance_set_color(const RID &p_multimesh, int32_t p_index, const Color &p_color);
	void multimesh_instance_set_custom_data(const RID &p_multimesh, int32_t p_index, const Color &p_custom_data);
	RID multimesh_get_mesh(const RID &p_multimesh) const;
	AABB multimesh_get_aabb(const RID &p_multimesh) const;
	void multimesh_set_custom_aabb(const RID &p_multimesh, const AABB &p_aabb);
	AABB multimesh_get_custom_aabb(const RID &p_multimesh) const;
	Transform3D multimesh_instance_get_transform(const RID &p_multimesh, int32_t p_index) const;
	Transform2D multimesh_instance_get_transform_2d(const RID &p_multimesh, int32_t p_index) const;
	Color multimesh_instance_get_color(const RID &p_multimesh, int32_t p_index) const;
	Color multimesh_instance_get_custom_data(const RID &p_multimesh, int32_t p_index) const;
	void multimesh_set_visible_instances(const RID &p_multimesh, int32_t p_visible);
	int32_t multimesh_get_visible_instances(const RID &p_multimesh) const;
	void multimesh_set_buffer(const RID &p_multimesh, const PackedFloat32Array &p_buffer);
	RID multimesh_get_command_buffer_rd_rid(const RID &p_multimesh) const;
	RID multimesh_get_buffer_rd_rid(const RID &p_multimesh) const;
	PackedFloat32Array multimesh_get_buffer(const RID &p_multimesh) const;
	void multimesh_set_buffer_interpolated(const RID &p_multimesh, const PackedFloat32Array &p_buffer, const PackedFloat32Array &p_buffer_previous);
	void multimesh_set_physics_interpolated(const RID &p_multimesh, bool p_interpolated);
	void multimesh_set_physics_interpolation_quality(const RID &p_multimesh, RenderingServer::MultimeshPhysicsInterpolationQuality p_quality);
	void multimesh_instance_reset_physics_interpolation(const RID &p_multimesh, int32_t p_index);
	void multimesh_instances_reset_physics_interpolation(const RID &p_multimesh);
	RID skeleton_create();
	void skeleton_allocate_data(const RID &p_skeleton, int32_t p_bones, bool p_is_2d_skeleton = false);
	int32_t skeleton_get_bone_count(const RID &p_skeleton) const;
	void skeleton_bone_set_transform(const RID &p_skeleton, int32_t p_bone, const Transform3D &p_transform);
	Transform3D skeleton_bone_get_transform(const RID &p_skeleton, int32_t p_bone) const;
	void skeleton_bone_set_transform_2d(const RID &p_skeleton, int32_t p_bone, const Transform2D &p_transform);
	Transform2D skeleton_bone_get_transform_2d(const RID &p_skeleton, int32_t p_bone) const;
	void skeleton_set_base_transform_2d(const RID &p_skeleton, const Transform2D &p_base_transform);
	RID directional_light_create();
	RID omni_light_create();
	RID spot_light_create();
	void light_set_color(const RID &p_light, const Color &p_color);
	void light_set_param(const RID &p_light, RenderingServer::LightParam p_param, float p_value);
	void light_set_shadow(const RID &p_light, bool p_enabled);
	void light_set_projector(const RID &p_light, const RID &p_texture);
	void light_set_negative(const RID &p_light, bool p_enable);
	void light_set_cull_mask(const RID &p_light, uint32_t p_mask);
	void light_set_distance_fade(const RID &p_decal, bool p_enabled, float p_begin, float p_shadow, float p_length);
	void light_set_reverse_cull_face_mode(const RID &p_light, bool p_enabled);
	void light_set_shadow_caster_mask(const RID &p_light, uint32_t p_mask);
	void light_set_bake_mode(const RID &p_light, RenderingServer::LightBakeMode p_bake_mode);
	void light_set_max_sdfgi_cascade(const RID &p_light, uint32_t p_cascade);
	void light_omni_set_shadow_mode(const RID &p_light, RenderingServer::LightOmniShadowMode p_mode);
	void light_directional_set_shadow_mode(const RID &p_light, RenderingServer::LightDirectionalShadowMode p_mode);
	void light_directional_set_blend_splits(const RID &p_light, bool p_enable);
	void light_directional_set_sky_mode(const RID &p_light, RenderingServer::LightDirectionalSkyMode p_mode);
	void light_projectors_set_filter(RenderingServer::LightProjectorFilter p_filter);
	void lightmaps_set_bicubic_filter(bool p_enable);
	void positional_soft_shadow_filter_set_quality(RenderingServer::ShadowQuality p_quality);
	void directional_soft_shadow_filter_set_quality(RenderingServer::ShadowQuality p_quality);
	void directional_shadow_atlas_set_size(int32_t p_size, bool p_is_16bits);
	RID reflection_probe_create();
	void reflection_probe_set_update_mode(const RID &p_probe, RenderingServer::ReflectionProbeUpdateMode p_mode);
	void reflection_probe_set_intensity(const RID &p_probe, float p_intensity);
	void reflection_probe_set_blend_distance(const RID &p_probe, float p_blend_distance);
	void reflection_probe_set_ambient_mode(const RID &p_probe, RenderingServer::ReflectionProbeAmbientMode p_mode);
	void reflection_probe_set_ambient_color(const RID &p_probe, const Color &p_color);
	void reflection_probe_set_ambient_energy(const RID &p_probe, float p_energy);
	void reflection_probe_set_max_distance(const RID &p_probe, float p_distance);
	void reflection_probe_set_size(const RID &p_probe, const Vector3 &p_size);
	void reflection_probe_set_origin_offset(const RID &p_probe, const Vector3 &p_offset);
	void reflection_probe_set_as_interior(const RID &p_probe, bool p_enable);
	void reflection_probe_set_enable_box_projection(const RID &p_probe, bool p_enable);
	void reflection_probe_set_enable_shadows(const RID &p_probe, bool p_enable);
	void reflection_probe_set_cull_mask(const RID &p_probe, uint32_t p_layers);
	void reflection_probe_set_reflection_mask(const RID &p_probe, uint32_t p_layers);
	void reflection_probe_set_resolution(const RID &p_probe, int32_t p_resolution);
	void reflection_probe_set_mesh_lod_threshold(const RID &p_probe, float p_pixels);
	RID decal_create();
	void decal_set_size(const RID &p_decal, const Vector3 &p_size);
	void decal_set_texture(const RID &p_decal, RenderingServer::DecalTexture p_type, const RID &p_texture);
	void decal_set_emission_energy(const RID &p_decal, float p_energy);
	void decal_set_albedo_mix(const RID &p_decal, float p_albedo_mix);
	void decal_set_modulate(const RID &p_decal, const Color &p_color);
	void decal_set_cull_mask(const RID &p_decal, uint32_t p_mask);
	void decal_set_distance_fade(const RID &p_decal, bool p_enabled, float p_begin, float p_length);
	void decal_set_fade(const RID &p_decal, float p_above, float p_below);
	void decal_set_normal_fade(const RID &p_decal, float p_fade);
	void decals_set_filter(RenderingServer::DecalFilter p_filter);
	void gi_set_use_half_resolution(bool p_half_resolution);
	RID voxel_gi_create();
	void voxel_gi_allocate_data(const RID &p_voxel_gi, const Transform3D &p_to_cell_xform, const AABB &p_aabb, const Vector3i &p_octree_size, const PackedByteArray &p_octree_cells, const PackedByteArray &p_data_cells, const PackedByteArray &p_distance_field, const PackedInt32Array &p_level_counts);
	Vector3i voxel_gi_get_octree_size(const RID &p_voxel_gi) const;
	PackedByteArray voxel_gi_get_octree_cells(const RID &p_voxel_gi) const;
	PackedByteArray voxel_gi_get_data_cells(const RID &p_voxel_gi) const;
	PackedByteArray voxel_gi_get_distance_field(const RID &p_voxel_gi) const;
	PackedInt32Array voxel_gi_get_level_counts(const RID &p_voxel_gi) const;
	Transform3D voxel_gi_get_to_cell_xform(const RID &p_voxel_gi) const;
	void voxel_gi_set_dynamic_range(const RID &p_voxel_gi, float p_range);
	void voxel_gi_set_propagation(const RID &p_voxel_gi, float p_amount);
	void voxel_gi_set_energy(const RID &p_voxel_gi, float p_energy);
	void voxel_gi_set_baked_exposure_normalization(const RID &p_voxel_gi, float p_baked_exposure);
	void voxel_gi_set_bias(const RID &p_voxel_gi, float p_bias);
	void voxel_gi_set_normal_bias(const RID &p_voxel_gi, float p_bias);
	void voxel_gi_set_interior(const RID &p_voxel_gi, bool p_enable);
	void voxel_gi_set_use_two_bounces(const RID &p_voxel_gi, bool p_enable);
	void voxel_gi_set_quality(RenderingServer::VoxelGIQuality p_quality);
	RID lightmap_create();
	void lightmap_set_textures(const RID &p_lightmap, const RID &p_light, bool p_uses_sh);
	void lightmap_set_probe_bounds(const RID &p_lightmap, const AABB &p_bounds);
	void lightmap_set_probe_interior(const RID &p_lightmap, bool p_interior);
	void lightmap_set_probe_capture_data(const RID &p_lightmap, const PackedVector3Array &p_points, const PackedColorArray &p_point_sh, const PackedInt32Array &p_tetrahedra, const PackedInt32Array &p_bsp_tree);
	PackedVector3Array lightmap_get_probe_capture_points(const RID &p_lightmap) const;
	PackedColorArray lightmap_get_probe_capture_sh(const RID &p_lightmap) const;
	PackedInt32Array lightmap_get_probe_capture_tetrahedra(const RID &p_lightmap) const;
	PackedInt32Array lightmap_get_probe_capture_bsp_tree(const RID &p_lightmap) const;
	void lightmap_set_baked_exposure_normalization(const RID &p_lightmap, float p_baked_exposure);
	void lightmap_set_probe_capture_update_speed(float p_speed);
	RID particles_create();
	void particles_set_mode(const RID &p_particles, RenderingServer::ParticlesMode p_mode);
	void particles_set_emitting(const RID &p_particles, bool p_emitting);
	bool particles_get_emitting(const RID &p_particles);
	void particles_set_amount(const RID &p_particles, int32_t p_amount);
	void particles_set_amount_ratio(const RID &p_particles, float p_ratio);
	void particles_set_lifetime(const RID &p_particles, double p_lifetime);
	void particles_set_one_shot(const RID &p_particles, bool p_one_shot);
	void particles_set_pre_process_time(const RID &p_particles, double p_time);
	void particles_request_process_time(const RID &p_particles, float p_time);
	void particles_set_explosiveness_ratio(const RID &p_particles, float p_ratio);
	void particles_set_randomness_ratio(const RID &p_particles, float p_ratio);
	void particles_set_interp_to_end(const RID &p_particles, float p_factor);
	void particles_set_emitter_velocity(const RID &p_particles, const Vector3 &p_velocity);
	void particles_set_custom_aabb(const RID &p_particles, const AABB &p_aabb);
	void particles_set_speed_scale(const RID &p_particles, double p_scale);
	void particles_set_use_local_coordinates(const RID &p_particles, bool p_enable);
	void particles_set_process_material(const RID &p_particles, const RID &p_material);
	void particles_set_fixed_fps(const RID &p_particles, int32_t p_fps);
	void particles_set_interpolate(const RID &p_particles, bool p_enable);
	void particles_set_fractional_delta(const RID &p_particles, bool p_enable);
	void particles_set_collision_base_size(const RID &p_particles, float p_size);
	void particles_set_transform_align(const RID &p_particles, RenderingServer::ParticlesTransformAlign p_align);
	void particles_set_trails(const RID &p_particles, bool p_enable, float p_length_sec);
	void particles_set_trail_bind_poses(const RID &p_particles, const TypedArray<Transform3D> &p_bind_poses);
	bool particles_is_inactive(const RID &p_particles);
	void particles_request_process(const RID &p_particles);
	void particles_restart(const RID &p_particles);
	void particles_set_subemitter(const RID &p_particles, const RID &p_subemitter_particles);
	void particles_emit(const RID &p_particles, const Transform3D &p_transform, const Vector3 &p_velocity, const Color &p_color, const Color &p_custom, uint32_t p_emit_flags);
	void particles_set_draw_order(const RID &p_particles, RenderingServer::ParticlesDrawOrder p_order);
	void particles_set_draw_passes(const RID &p_particles, int32_t p_count);
	void particles_set_draw_pass_mesh(const RID &p_particles, int32_t p_pass, const RID &p_mesh);
	AABB particles_get_current_aabb(const RID &p_particles);
	void particles_set_emission_transform(const RID &p_particles, const Transform3D &p_transform);
	RID particles_collision_create();
	void particles_collision_set_collision_type(const RID &p_particles_collision, RenderingServer::ParticlesCollisionType p_type);
	void particles_collision_set_cull_mask(const RID &p_particles_collision, uint32_t p_mask);
	void particles_collision_set_sphere_radius(const RID &p_particles_collision, float p_radius);
	void particles_collision_set_box_extents(const RID &p_particles_collision, const Vector3 &p_extents);
	void particles_collision_set_attractor_strength(const RID &p_particles_collision, float p_strength);
	void particles_collision_set_attractor_directionality(const RID &p_particles_collision, float p_amount);
	void particles_collision_set_attractor_attenuation(const RID &p_particles_collision, float p_curve);
	void particles_collision_set_field_texture(const RID &p_particles_collision, const RID &p_texture);
	void particles_collision_height_field_update(const RID &p_particles_collision);
	void particles_collision_set_height_field_resolution(const RID &p_particles_collision, RenderingServer::ParticlesCollisionHeightfieldResolution p_resolution);
	void particles_collision_set_height_field_mask(const RID &p_particles_collision, uint32_t p_mask);
	RID fog_volume_create();
	void fog_volume_set_shape(const RID &p_fog_volume, RenderingServer::FogVolumeShape p_shape);
	void fog_volume_set_size(const RID &p_fog_volume, const Vector3 &p_size);
	void fog_volume_set_material(const RID &p_fog_volume, const RID &p_material);
	RID visibility_notifier_create();
	void visibility_notifier_set_aabb(const RID &p_notifier, const AABB &p_aabb);
	void visibility_notifier_set_callbacks(const RID &p_notifier, const Callable &p_enter_callable, const Callable &p_exit_callable);
	RID occluder_create();
	void occluder_set_mesh(const RID &p_occluder, const PackedVector3Array &p_vertices, const PackedInt32Array &p_indices);
	RID camera_create();
	void camera_set_perspective(const RID &p_camera, float p_fovy_degrees, float p_z_near, float p_z_far);
	void camera_set_orthogonal(const RID &p_camera, float p_size, float p_z_near, float p_z_far);
	void camera_set_frustum(const RID &p_camera, float p_size, const Vector2 &p_offset, float p_z_near, float p_z_far);
	void camera_set_transform(const RID &p_camera, const Transform3D &p_transform);
	void camera_set_cull_mask(const RID &p_camera, uint32_t p_layers);
	void camera_set_environment(const RID &p_camera, const RID &p_env);
	void camera_set_camera_attributes(const RID &p_camera, const RID &p_effects);
	void camera_set_compositor(const RID &p_camera, const RID &p_compositor);
	void camera_set_use_vertical_aspect(const RID &p_camera, bool p_enable);
	RID viewport_create();
	void viewport_set_use_xr(const RID &p_viewport, bool p_use_xr);
	void viewport_set_size(const RID &p_viewport, int32_t p_width, int32_t p_height);
	void viewport_set_active(const RID &p_viewport, bool p_active);
	void viewport_set_parent_viewport(const RID &p_viewport, const RID &p_parent_viewport);
	void viewport_attach_to_screen(const RID &p_viewport, const Rect2 &p_rect = Rect2(0, 0, 0, 0), int32_t p_screen = 0);
	void viewport_set_render_direct_to_screen(const RID &p_viewport, bool p_enabled);
	void viewport_set_canvas_cull_mask(const RID &p_viewport, uint32_t p_canvas_cull_mask);
	void viewport_set_scaling_3d_mode(const RID &p_viewport, RenderingServer::ViewportScaling3DMode p_scaling_3d_mode);
	void viewport_set_scaling_3d_scale(const RID &p_viewport, float p_scale);
	void viewport_set_fsr_sharpness(const RID &p_viewport, float p_sharpness);
	void viewport_set_texture_mipmap_bias(const RID &p_viewport, float p_mipmap_bias);
	void viewport_set_anisotropic_filtering_level(const RID &p_viewport, RenderingServer::ViewportAnisotropicFiltering p_anisotropic_filtering_level);
	void viewport_set_update_mode(const RID &p_viewport, RenderingServer::ViewportUpdateMode p_update_mode);
	RenderingServer::ViewportUpdateMode viewport_get_update_mode(const RID &p_viewport) const;
	void viewport_set_clear_mode(const RID &p_viewport, RenderingServer::ViewportClearMode p_clear_mode);
	RID viewport_get_render_target(const RID &p_viewport) const;
	RID viewport_get_texture(const RID &p_viewport) const;
	void viewport_set_disable_3d(const RID &p_viewport, bool p_disable);
	void viewport_set_disable_2d(const RID &p_viewport, bool p_disable);
	void viewport_set_environment_mode(const RID &p_viewport, RenderingServer::ViewportEnvironmentMode p_mode);
	void viewport_attach_camera(const RID &p_viewport, const RID &p_camera);
	void viewport_set_scenario(const RID &p_viewport, const RID &p_scenario);
	void viewport_attach_canvas(const RID &p_viewport, const RID &p_canvas);
	void viewport_remove_canvas(const RID &p_viewport, const RID &p_canvas);
	void viewport_set_snap_2d_transforms_to_pixel(const RID &p_viewport, bool p_enabled);
	void viewport_set_snap_2d_vertices_to_pixel(const RID &p_viewport, bool p_enabled);
	void viewport_set_default_canvas_item_texture_filter(const RID &p_viewport, RenderingServer::CanvasItemTextureFilter p_filter);
	void viewport_set_default_canvas_item_texture_repeat(const RID &p_viewport, RenderingServer::CanvasItemTextureRepeat p_repeat);
	void viewport_set_canvas_transform(const RID &p_viewport, const RID &p_canvas, const Transform2D &p_offset);
	void viewport_set_canvas_stacking(const RID &p_viewport, const RID &p_canvas, int32_t p_layer, int32_t p_sublayer);
	void viewport_set_transparent_background(const RID &p_viewport, bool p_enabled);
	void viewport_set_global_canvas_transform(const RID &p_viewport, const Transform2D &p_transform);
	void viewport_set_sdf_oversize_and_scale(const RID &p_viewport, RenderingServer::ViewportSDFOversize p_oversize, RenderingServer::ViewportSDFScale p_scale);
	void viewport_set_positional_shadow_atlas_size(const RID &p_viewport, int32_t p_size, bool p_use_16_bits = false);
	void viewport_set_positional_shadow_atlas_quadrant_subdivision(const RID &p_viewport, int32_t p_quadrant, int32_t p_subdivision);
	void viewport_set_msaa_3d(const RID &p_viewport, RenderingServer::ViewportMSAA p_msaa);
	void viewport_set_msaa_2d(const RID &p_viewport, RenderingServer::ViewportMSAA p_msaa);
	void viewport_set_use_hdr_2d(const RID &p_viewport, bool p_enabled);
	void viewport_set_screen_space_aa(const RID &p_viewport, RenderingServer::ViewportScreenSpaceAA p_mode);
	void viewport_set_use_taa(const RID &p_viewport, bool p_enable);
	void viewport_set_use_debanding(const RID &p_viewport, bool p_enable);
	void viewport_set_use_occlusion_culling(const RID &p_viewport, bool p_enable);
	void viewport_set_occlusion_rays_per_thread(int32_t p_rays_per_thread);
	void viewport_set_occlusion_culling_build_quality(RenderingServer::ViewportOcclusionCullingBuildQuality p_quality);
	int32_t viewport_get_render_info(const RID &p_viewport, RenderingServer::ViewportRenderInfoType p_type, RenderingServer::ViewportRenderInfo p_info);
	void viewport_set_debug_draw(const RID &p_viewport, RenderingServer::ViewportDebugDraw p_draw);
	void viewport_set_measure_render_time(const RID &p_viewport, bool p_enable);
	double viewport_get_measured_render_time_cpu(const RID &p_viewport) const;
	double viewport_get_measured_render_time_gpu(const RID &p_viewport) const;
	void viewport_set_vrs_mode(const RID &p_viewport, RenderingServer::ViewportVRSMode p_mode);
	void viewport_set_vrs_update_mode(const RID &p_viewport, RenderingServer::ViewportVRSUpdateMode p_mode);
	void viewport_set_vrs_texture(const RID &p_viewport, const RID &p_texture);
	RID sky_create();
	void sky_set_radiance_size(const RID &p_sky, int32_t p_radiance_size);
	void sky_set_mode(const RID &p_sky, RenderingServer::SkyMode p_mode);
	void sky_set_material(const RID &p_sky, const RID &p_material);
	Ref<Image> sky_bake_panorama(const RID &p_sky, float p_energy, bool p_bake_irradiance, const Vector2i &p_size);
	RID compositor_effect_create();
	void compositor_effect_set_enabled(const RID &p_effect, bool p_enabled);
	void compositor_effect_set_callback(const RID &p_effect, RenderingServer::CompositorEffectCallbackType p_callback_type, const Callable &p_callback);
	void compositor_effect_set_flag(const RID &p_effect, RenderingServer::CompositorEffectFlags p_flag, bool p_set);
	RID compositor_create();
	void compositor_set_compositor_effects(const RID &p_compositor, const TypedArray<RID> &p_effects);
	RID environment_create();
	void environment_set_background(const RID &p_env, RenderingServer::EnvironmentBG p_bg);
	void environment_set_camera_id(const RID &p_env, int32_t p_id);
	void environment_set_sky(const RID &p_env, const RID &p_sky);
	void environment_set_sky_custom_fov(const RID &p_env, float p_scale);
	void environment_set_sky_orientation(const RID &p_env, const Basis &p_orientation);
	void environment_set_bg_color(const RID &p_env, const Color &p_color);
	void environment_set_bg_energy(const RID &p_env, float p_multiplier, float p_exposure_value);
	void environment_set_canvas_max_layer(const RID &p_env, int32_t p_max_layer);
	void environment_set_ambient_light(const RID &p_env, const Color &p_color, RenderingServer::EnvironmentAmbientSource p_ambient = (RenderingServer::EnvironmentAmbientSource)0, float p_energy = 1.0, float p_sky_contribution = 0.0, RenderingServer::EnvironmentReflectionSource p_reflection_source = (RenderingServer::EnvironmentReflectionSource)0);
	void environment_set_glow(const RID &p_env, bool p_enable, const PackedFloat32Array &p_levels, float p_intensity, float p_strength, float p_mix, float p_bloom_threshold, RenderingServer::EnvironmentGlowBlendMode p_blend_mode, float p_hdr_bleed_threshold, float p_hdr_bleed_scale, float p_hdr_luminance_cap, float p_glow_map_strength, const RID &p_glow_map);
	void environment_set_tonemap(const RID &p_env, RenderingServer::EnvironmentToneMapper p_tone_mapper, float p_exposure, float p_white);
	void environment_set_tonemap_agx_contrast(const RID &p_env, float p_agx_contrast);
	void environment_set_adjustment(const RID &p_env, bool p_enable, float p_brightness, float p_contrast, float p_saturation, bool p_use_1d_color_correction, const RID &p_color_correction);
	void environment_set_ssr(const RID &p_env, bool p_enable, int32_t p_max_steps, float p_fade_in, float p_fade_out, float p_depth_tolerance);
	void environment_set_ssao(const RID &p_env, bool p_enable, float p_radius, float p_intensity, float p_power, float p_detail, float p_horizon, float p_sharpness, float p_light_affect, float p_ao_channel_affect);
	void environment_set_fog(const RID &p_env, bool p_enable, const Color &p_light_color, float p_light_energy, float p_sun_scatter, float p_density, float p_height, float p_height_density, float p_aerial_perspective, float p_sky_affect, RenderingServer::EnvironmentFogMode p_fog_mode = (RenderingServer::EnvironmentFogMode)0);
	void environment_set_fog_depth(const RID &p_env, float p_curve, float p_begin, float p_end);
	void environment_set_sdfgi(const RID &p_env, bool p_enable, int32_t p_cascades, float p_min_cell_size, RenderingServer::EnvironmentSDFGIYScale p_y_scale, bool p_use_occlusion, float p_bounce_feedback, bool p_read_sky, float p_energy, float p_normal_bias, float p_probe_bias);
	void environment_set_volumetric_fog(const RID &p_env, bool p_enable, float p_density, const Color &p_albedo, const Color &p_emission, float p_emission_energy, float p_anisotropy, float p_length, float p_detail_spread, float p_gi_inject, bool p_temporal_reprojection, float p_temporal_reprojection_amount, float p_ambient_inject, float p_sky_affect);
	void environment_glow_set_use_bicubic_upscale(bool p_enable);
	void environment_set_ssr_half_size(bool p_half_size);
	void environment_set_ssr_roughness_quality(RenderingServer::EnvironmentSSRRoughnessQuality p_quality);
	void environment_set_ssao_quality(RenderingServer::EnvironmentSSAOQuality p_quality, bool p_half_size, float p_adaptive_target, int32_t p_blur_passes, float p_fadeout_from, float p_fadeout_to);
	void environment_set_ssil_quality(RenderingServer::EnvironmentSSILQuality p_quality, bool p_half_size, float p_adaptive_target, int32_t p_blur_passes, float p_fadeout_from, float p_fadeout_to);
	void environment_set_sdfgi_ray_count(RenderingServer::EnvironmentSDFGIRayCount p_ray_count);
	void environment_set_sdfgi_frames_to_converge(RenderingServer::EnvironmentSDFGIFramesToConverge p_frames);
	void environment_set_sdfgi_frames_to_update_light(RenderingServer::EnvironmentSDFGIFramesToUpdateLight p_frames);
	void environment_set_volumetric_fog_volume_size(int32_t p_size, int32_t p_depth);
	void environment_set_volumetric_fog_filter_active(bool p_active);
	Ref<Image> environment_bake_panorama(const RID &p_environment, bool p_bake_irradiance, const Vector2i &p_size);
	void screen_space_roughness_limiter_set_active(bool p_enable, float p_amount, float p_limit);
	void sub_surface_scattering_set_quality(RenderingServer::SubSurfaceScatteringQuality p_quality);
	void sub_surface_scattering_set_scale(float p_scale, float p_depth_scale);
	RID camera_attributes_create();
	void camera_attributes_set_dof_blur_quality(RenderingServer::DOFBlurQuality p_quality, bool p_use_jitter);
	void camera_attributes_set_dof_blur_bokeh_shape(RenderingServer::DOFBokehShape p_shape);
	void camera_attributes_set_dof_blur(const RID &p_camera_attributes, bool p_far_enable, float p_far_distance, float p_far_transition, bool p_near_enable, float p_near_distance, float p_near_transition, float p_amount);
	void camera_attributes_set_exposure(const RID &p_camera_attributes, float p_multiplier, float p_normalization);
	void camera_attributes_set_auto_exposure(const RID &p_camera_attributes, bool p_enable, float p_min_sensitivity, float p_max_sensitivity, float p_speed, float p_scale);
	RID scenario_create();
	void scenario_set_environment(const RID &p_scenario, const RID &p_environment);
	void scenario_set_fallback_environment(const RID &p_scenario, const RID &p_environment);
	void scenario_set_camera_attributes(const RID &p_scenario, const RID &p_effects);
	void scenario_set_compositor(const RID &p_scenario, const RID &p_compositor);
	RID instance_create2(const RID &p_base, const RID &p_scenario);
	RID instance_create();
	void instance_set_base(const RID &p_instance, const RID &p_base);
	void instance_set_scenario(const RID &p_instance, const RID &p_scenario);
	void instance_set_layer_mask(const RID &p_instance, uint32_t p_mask);
	void instance_set_pivot_data(const RID &p_instance, float p_sorting_offset, bool p_use_aabb_center);
	void instance_set_transform(const RID &p_instance, const Transform3D &p_transform);
	void instance_attach_object_instance_id(const RID &p_instance, uint64_t p_id);
	void instance_set_blend_shape_weight(const RID &p_instance, int32_t p_shape, float p_weight);
	void instance_set_surface_override_material(const RID &p_instance, int32_t p_surface, const RID &p_material);
	void instance_set_visible(const RID &p_instance, bool p_visible);
	void instance_geometry_set_transparency(const RID &p_instance, float p_transparency);
	void instance_teleport(const RID &p_instance);
	void instance_set_custom_aabb(const RID &p_instance, const AABB &p_aabb);
	void instance_attach_skeleton(const RID &p_instance, const RID &p_skeleton);
	void instance_set_extra_visibility_margin(const RID &p_instance, float p_margin);
	void instance_set_visibility_parent(const RID &p_instance, const RID &p_parent);
	void instance_set_ignore_culling(const RID &p_instance, bool p_enabled);
	void instance_geometry_set_flag(const RID &p_instance, RenderingServer::InstanceFlags p_flag, bool p_enabled);
	void instance_geometry_set_cast_shadows_setting(const RID &p_instance, RenderingServer::ShadowCastingSetting p_shadow_casting_setting);
	void instance_geometry_set_material_override(const RID &p_instance, const RID &p_material);
	void instance_geometry_set_material_overlay(const RID &p_instance, const RID &p_material);
	void instance_geometry_set_visibility_range(const RID &p_instance, float p_min, float p_max, float p_min_margin, float p_max_margin, RenderingServer::VisibilityRangeFadeMode p_fade_mode);
	void instance_geometry_set_lightmap(const RID &p_instance, const RID &p_lightmap, const Rect2 &p_lightmap_uv_scale, int32_t p_lightmap_slice);
	void instance_geometry_set_lod_bias(const RID &p_instance, float p_lod_bias);
	void instance_geometry_set_shader_parameter(const RID &p_instance, const StringName &p_parameter, const Variant &p_value);
	Variant instance_geometry_get_shader_parameter(const RID &p_instance, const StringName &p_parameter) const;
	Variant instance_geometry_get_shader_parameter_default_value(const RID &p_instance, const StringName &p_parameter) const;
	TypedArray<Dictionary> instance_geometry_get_shader_parameter_list(const RID &p_instance) const;
	PackedInt64Array instances_cull_aabb(const AABB &p_aabb, const RID &p_scenario = RID()) const;
	PackedInt64Array instances_cull_ray(const Vector3 &p_from, const Vector3 &p_to, const RID &p_scenario = RID()) const;
	PackedInt64Array instances_cull_convex(const TypedArray<Plane> &p_convex, const RID &p_scenario = RID()) const;
	TypedArray<Ref<Image>> bake_render_uv2(const RID &p_base, const TypedArray<RID> &p_material_overrides, const Vector2i &p_image_size);
	RID canvas_create();
	void canvas_set_item_mirroring(const RID &p_canvas, const RID &p_item, const Vector2 &p_mirroring);
	void canvas_set_item_repeat(const RID &p_item, const Vector2 &p_repeat_size, int32_t p_repeat_times);
	void canvas_set_modulate(const RID &p_canvas, const Color &p_color);
	void canvas_set_disable_scale(bool p_disable);
	RID canvas_texture_create();
	void canvas_texture_set_channel(const RID &p_canvas_texture, RenderingServer::CanvasTextureChannel p_channel, const RID &p_texture);
	void canvas_texture_set_shading_parameters(const RID &p_canvas_texture, const Color &p_base_color, float p_shininess);
	void canvas_texture_set_texture_filter(const RID &p_canvas_texture, RenderingServer::CanvasItemTextureFilter p_filter);
	void canvas_texture_set_texture_repeat(const RID &p_canvas_texture, RenderingServer::CanvasItemTextureRepeat p_repeat);
	RID canvas_item_create();
	void canvas_item_set_parent(const RID &p_item, const RID &p_parent);
	void canvas_item_set_default_texture_filter(const RID &p_item, RenderingServer::CanvasItemTextureFilter p_filter);
	void canvas_item_set_default_texture_repeat(const RID &p_item, RenderingServer::CanvasItemTextureRepeat p_repeat);
	void canvas_item_set_visible(const RID &p_item, bool p_visible);
	void canvas_item_set_light_mask(const RID &p_item, int32_t p_mask);
	void canvas_item_set_visibility_layer(const RID &p_item, uint32_t p_visibility_layer);
	void canvas_item_set_transform(const RID &p_item, const Transform2D &p_transform);
	void canvas_item_set_clip(const RID &p_item, bool p_clip);
	void canvas_item_set_distance_field_mode(const RID &p_item, bool p_enabled);
	void canvas_item_set_custom_rect(const RID &p_item, bool p_use_custom_rect, const Rect2 &p_rect = Rect2(0, 0, 0, 0));
	void canvas_item_set_modulate(const RID &p_item, const Color &p_color);
	void canvas_item_set_self_modulate(const RID &p_item, const Color &p_color);
	void canvas_item_set_draw_behind_parent(const RID &p_item, bool p_enabled);
	void canvas_item_set_interpolated(const RID &p_item, bool p_interpolated);
	void canvas_item_reset_physics_interpolation(const RID &p_item);
	void canvas_item_transform_physics_interpolation(const RID &p_item, const Transform2D &p_transform);
	void canvas_item_add_line(const RID &p_item, const Vector2 &p_from, const Vector2 &p_to, const Color &p_color, float p_width = -1.0, bool p_antialiased = false);
	void canvas_item_add_polyline(const RID &p_item, const PackedVector2Array &p_points, const PackedColorArray &p_colors, float p_width = -1.0, bool p_antialiased = false);
	void canvas_item_add_multiline(const RID &p_item, const PackedVector2Array &p_points, const PackedColorArray &p_colors, float p_width = -1.0, bool p_antialiased = false);
	void canvas_item_add_rect(const RID &p_item, const Rect2 &p_rect, const Color &p_color, bool p_antialiased = false);
	void canvas_item_add_circle(const RID &p_item, const Vector2 &p_pos, float p_radius, const Color &p_color, bool p_antialiased = false);
	void canvas_item_add_ellipse(const RID &p_item, const Vector2 &p_pos, float p_major, float p_minor, const Color &p_color, bool p_antialiased = false);
	void canvas_item_add_texture_rect(const RID &p_item, const Rect2 &p_rect, const RID &p_texture, bool p_tile = false, const Color &p_modulate = Color(1, 1, 1, 1), bool p_transpose = false);
	void canvas_item_add_msdf_texture_rect_region(const RID &p_item, const Rect2 &p_rect, const RID &p_texture, const Rect2 &p_src_rect, const Color &p_modulate = Color(1, 1, 1, 1), int32_t p_outline_size = 0, float p_px_range = 1.0, float p_scale = 1.0);
	void canvas_item_add_lcd_texture_rect_region(const RID &p_item, const Rect2 &p_rect, const RID &p_texture, const Rect2 &p_src_rect, const Color &p_modulate);
	void canvas_item_add_texture_rect_region(const RID &p_item, const Rect2 &p_rect, const RID &p_texture, const Rect2 &p_src_rect, const Color &p_modulate = Color(1, 1, 1, 1), bool p_transpose = false, bool p_clip_uv = true);
	void canvas_item_add_nine_patch(const RID &p_item, const Rect2 &p_rect, const Rect2 &p_source, const RID &p_texture, const Vector2 &p_topleft, const Vector2 &p_bottomright, RenderingServer::NinePatchAxisMode p_x_axis_mode = (RenderingServer::NinePatchAxisMode)0, RenderingServer::NinePatchAxisMode p_y_axis_mode = (RenderingServer::NinePatchAxisMode)0, bool p_draw_center = true, const Color &p_modulate = Color(1, 1, 1, 1));
	void canvas_item_add_primitive(const RID &p_item, const PackedVector2Array &p_points, const PackedColorArray &p_colors, const PackedVector2Array &p_uvs, const RID &p_texture);
	void canvas_item_add_polygon(const RID &p_item, const PackedVector2Array &p_points, const PackedColorArray &p_colors, const PackedVector2Array &p_uvs = PackedVector2Array(), const RID &p_texture = RID());
	void canvas_item_add_triangle_array(const RID &p_item, const PackedInt32Array &p_indices, const PackedVector2Array &p_points, const PackedColorArray &p_colors, const PackedVector2Array &p_uvs = PackedVector2Array(), const PackedInt32Array &p_bones = PackedInt32Array(), const PackedFloat32Array &p_weights = PackedFloat32Array(), const RID &p_texture = RID(), int32_t p_count = -1);
	void canvas_item_add_mesh(const RID &p_item, const RID &p_mesh, const Transform2D &p_transform = Transform2D(), const Color &p_modulate = Color(1, 1, 1, 1), const RID &p_texture = RID());
	void canvas_item_add_multimesh(const RID &p_item, const RID &p_mesh, const RID &p_texture = RID());
	void canvas_item_add_particles(const RID &p_item, const RID &p_particles, const RID &p_texture);
	void canvas_item_add_set_transform(const RID &p_item, const Transform2D &p_transform);
	void canvas_item_add_clip_ignore(const RID &p_item, bool p_ignore);
	void canvas_item_add_animation_slice(const RID &p_item, double p_animation_length, double p_slice_begin, double p_slice_end, double p_offset = 0.0);
	void canvas_item_set_sort_children_by_y(const RID &p_item, bool p_enabled);
	void canvas_item_set_z_index(const RID &p_item, int32_t p_z_index);
	void canvas_item_set_z_as_relative_to_parent(const RID &p_item, bool p_enabled);
	void canvas_item_set_copy_to_backbuffer(const RID &p_item, bool p_enabled, const Rect2 &p_rect);
	void canvas_item_attach_skeleton(const RID &p_item, const RID &p_skeleton);
	void canvas_item_clear(const RID &p_item);
	void canvas_item_set_draw_index(const RID &p_item, int32_t p_index);
	void canvas_item_set_material(const RID &p_item, const RID &p_material);
	void canvas_item_set_use_parent_material(const RID &p_item, bool p_enabled);
	void canvas_item_set_instance_shader_parameter(const RID &p_instance, const StringName &p_parameter, const Variant &p_value);
	Variant canvas_item_get_instance_shader_parameter(const RID &p_instance, const StringName &p_parameter) const;
	Variant canvas_item_get_instance_shader_parameter_default_value(const RID &p_instance, const StringName &p_parameter) const;
	TypedArray<Dictionary> canvas_item_get_instance_shader_parameter_list(const RID &p_instance) const;
	void canvas_item_set_visibility_notifier(const RID &p_item, bool p_enable, const Rect2 &p_area, const Callable &p_enter_callable, const Callable &p_exit_callable);
	void canvas_item_set_canvas_group_mode(const RID &p_item, RenderingServer::CanvasGroupMode p_mode, float p_clear_margin = 5.0, bool p_fit_empty = false, float p_fit_margin = 0.0, bool p_blur_mipmaps = false);
	Rect2 debug_canvas_item_get_rect(const RID &p_item);
	RID canvas_light_create();
	void canvas_light_attach_to_canvas(const RID &p_light, const RID &p_canvas);
	void canvas_light_set_enabled(const RID &p_light, bool p_enabled);
	void canvas_light_set_texture_scale(const RID &p_light, float p_scale);
	void canvas_light_set_transform(const RID &p_light, const Transform2D &p_transform);
	void canvas_light_set_texture(const RID &p_light, const RID &p_texture);
	void canvas_light_set_texture_offset(const RID &p_light, const Vector2 &p_offset);
	void canvas_light_set_color(const RID &p_light, const Color &p_color);
	void canvas_light_set_height(const RID &p_light, float p_height);
	void canvas_light_set_energy(const RID &p_light, float p_energy);
	void canvas_light_set_z_range(const RID &p_light, int32_t p_min_z, int32_t p_max_z);
	void canvas_light_set_layer_range(const RID &p_light, int32_t p_min_layer, int32_t p_max_layer);
	void canvas_light_set_item_cull_mask(const RID &p_light, int32_t p_mask);
	void canvas_light_set_item_shadow_cull_mask(const RID &p_light, int32_t p_mask);
	void canvas_light_set_mode(const RID &p_light, RenderingServer::CanvasLightMode p_mode);
	void canvas_light_set_shadow_enabled(const RID &p_light, bool p_enabled);
	void canvas_light_set_shadow_filter(const RID &p_light, RenderingServer::CanvasLightShadowFilter p_filter);
	void canvas_light_set_shadow_color(const RID &p_light, const Color &p_color);
	void canvas_light_set_shadow_smooth(const RID &p_light, float p_smooth);
	void canvas_light_set_blend_mode(const RID &p_light, RenderingServer::CanvasLightBlendMode p_mode);
	void canvas_light_set_interpolated(const RID &p_light, bool p_interpolated);
	void canvas_light_reset_physics_interpolation(const RID &p_light);
	void canvas_light_transform_physics_interpolation(const RID &p_light, const Transform2D &p_transform);
	RID canvas_light_occluder_create();
	void canvas_light_occluder_attach_to_canvas(const RID &p_occluder, const RID &p_canvas);
	void canvas_light_occluder_set_enabled(const RID &p_occluder, bool p_enabled);
	void canvas_light_occluder_set_polygon(const RID &p_occluder, const RID &p_polygon);
	void canvas_light_occluder_set_as_sdf_collision(const RID &p_occluder, bool p_enable);
	void canvas_light_occluder_set_transform(const RID &p_occluder, const Transform2D &p_transform);
	void canvas_light_occluder_set_light_mask(const RID &p_occluder, int32_t p_mask);
	void canvas_light_occluder_set_interpolated(const RID &p_occluder, bool p_interpolated);
	void canvas_light_occluder_reset_physics_interpolation(const RID &p_occluder);
	void canvas_light_occluder_transform_physics_interpolation(const RID &p_occluder, const Transform2D &p_transform);
	RID canvas_occluder_polygon_create();
	void canvas_occluder_polygon_set_shape(const RID &p_occluder_polygon, const PackedVector2Array &p_shape, bool p_closed);
	void canvas_occluder_polygon_set_cull_mode(const RID &p_occluder_polygon, RenderingServer::CanvasOccluderPolygonCullMode p_mode);
	void canvas_set_shadow_texture_size(int32_t p_size);
	void global_shader_parameter_add(const StringName &p_name, RenderingServer::GlobalShaderParameterType p_type, const Variant &p_default_value);
	void global_shader_parameter_remove(const StringName &p_name);
	TypedArray<StringName> global_shader_parameter_get_list() const;
	void global_shader_parameter_set(const StringName &p_name, const Variant &p_value);
	void global_shader_parameter_set_override(const StringName &p_name, const Variant &p_value);
	Variant global_shader_parameter_get(const StringName &p_name) const;
	RenderingServer::GlobalShaderParameterType global_shader_parameter_get_type(const StringName &p_name) const;
	void free_rid(const RID &p_rid);
	void request_frame_drawn_callback(const Callable &p_callable);
	bool has_changed() const;
	uint64_t get_rendering_info(RenderingServer::RenderingInfo p_info);
	String get_video_adapter_name() const;
	String get_video_adapter_vendor() const;
	RenderingDevice::DeviceType get_video_adapter_type() const;
	String get_video_adapter_api_version() const;
	String get_current_rendering_driver_name() const;
	String get_current_rendering_method() const;
	RID make_sphere_mesh(int32_t p_latitudes, int32_t p_longitudes, float p_radius);
	RID get_test_cube();
	RID get_test_texture();
	RID get_white_texture();
	void set_boot_image_with_stretch(const Ref<Image> &p_image, const Color &p_color, RenderingServer::SplashStretchMode p_stretch_mode, bool p_use_filter = true);
	void set_boot_image(const Ref<Image> &p_image, const Color &p_color, bool p_scale, bool p_use_filter = true);
	Color get_default_clear_color();
	void set_default_clear_color(const Color &p_color);
	bool has_os_feature(const String &p_feature) const;
	void set_debug_generate_wireframes(bool p_generate);
	bool is_render_loop_enabled() const;
	void set_render_loop_enabled(bool p_enabled);
	double get_frame_setup_time_cpu() const;
	void force_sync();
	void force_draw(bool p_swap_buffers = true, double p_frame_step = 0.0);
	RenderingDevice *get_rendering_device() const;
	RenderingDevice *create_local_rendering_device() const;
	bool is_on_render_thread();
	void call_on_render_thread(const Callable &p_callable);
	bool has_feature(RenderingServer::Features p_feature) const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Object::register_virtuals<T, B>();
	}

	~RenderingServer();

public:
};

} // namespace godot

VARIANT_ENUM_CAST(RenderingServer::TextureType);
VARIANT_ENUM_CAST(RenderingServer::TextureLayeredType);
VARIANT_ENUM_CAST(RenderingServer::CubeMapLayer);
VARIANT_ENUM_CAST(RenderingServer::ShaderMode);
VARIANT_ENUM_CAST(RenderingServer::ArrayType);
VARIANT_ENUM_CAST(RenderingServer::ArrayCustomFormat);
VARIANT_BITFIELD_CAST(RenderingServer::ArrayFormat);
VARIANT_ENUM_CAST(RenderingServer::PrimitiveType);
VARIANT_ENUM_CAST(RenderingServer::BlendShapeMode);
VARIANT_ENUM_CAST(RenderingServer::MultimeshTransformFormat);
VARIANT_ENUM_CAST(RenderingServer::MultimeshPhysicsInterpolationQuality);
VARIANT_ENUM_CAST(RenderingServer::LightProjectorFilter);
VARIANT_ENUM_CAST(RenderingServer::LightType);
VARIANT_ENUM_CAST(RenderingServer::LightParam);
VARIANT_ENUM_CAST(RenderingServer::LightBakeMode);
VARIANT_ENUM_CAST(RenderingServer::LightOmniShadowMode);
VARIANT_ENUM_CAST(RenderingServer::LightDirectionalShadowMode);
VARIANT_ENUM_CAST(RenderingServer::LightDirectionalSkyMode);
VARIANT_ENUM_CAST(RenderingServer::ShadowQuality);
VARIANT_ENUM_CAST(RenderingServer::ReflectionProbeUpdateMode);
VARIANT_ENUM_CAST(RenderingServer::ReflectionProbeAmbientMode);
VARIANT_ENUM_CAST(RenderingServer::DecalTexture);
VARIANT_ENUM_CAST(RenderingServer::DecalFilter);
VARIANT_ENUM_CAST(RenderingServer::VoxelGIQuality);
VARIANT_ENUM_CAST(RenderingServer::ParticlesMode);
VARIANT_ENUM_CAST(RenderingServer::ParticlesTransformAlign);
VARIANT_ENUM_CAST(RenderingServer::ParticlesDrawOrder);
VARIANT_ENUM_CAST(RenderingServer::ParticlesCollisionType);
VARIANT_ENUM_CAST(RenderingServer::ParticlesCollisionHeightfieldResolution);
VARIANT_ENUM_CAST(RenderingServer::FogVolumeShape);
VARIANT_ENUM_CAST(RenderingServer::ViewportScaling3DMode);
VARIANT_ENUM_CAST(RenderingServer::ViewportUpdateMode);
VARIANT_ENUM_CAST(RenderingServer::ViewportClearMode);
VARIANT_ENUM_CAST(RenderingServer::ViewportEnvironmentMode);
VARIANT_ENUM_CAST(RenderingServer::ViewportSDFOversize);
VARIANT_ENUM_CAST(RenderingServer::ViewportSDFScale);
VARIANT_ENUM_CAST(RenderingServer::ViewportMSAA);
VARIANT_ENUM_CAST(RenderingServer::ViewportAnisotropicFiltering);
VARIANT_ENUM_CAST(RenderingServer::ViewportScreenSpaceAA);
VARIANT_ENUM_CAST(RenderingServer::ViewportOcclusionCullingBuildQuality);
VARIANT_ENUM_CAST(RenderingServer::ViewportRenderInfo);
VARIANT_ENUM_CAST(RenderingServer::ViewportRenderInfoType);
VARIANT_ENUM_CAST(RenderingServer::ViewportDebugDraw);
VARIANT_ENUM_CAST(RenderingServer::ViewportVRSMode);
VARIANT_ENUM_CAST(RenderingServer::ViewportVRSUpdateMode);
VARIANT_ENUM_CAST(RenderingServer::SkyMode);
VARIANT_ENUM_CAST(RenderingServer::CompositorEffectFlags);
VARIANT_ENUM_CAST(RenderingServer::CompositorEffectCallbackType);
VARIANT_ENUM_CAST(RenderingServer::EnvironmentBG);
VARIANT_ENUM_CAST(RenderingServer::EnvironmentAmbientSource);
VARIANT_ENUM_CAST(RenderingServer::EnvironmentReflectionSource);
VARIANT_ENUM_CAST(RenderingServer::EnvironmentGlowBlendMode);
VARIANT_ENUM_CAST(RenderingServer::EnvironmentFogMode);
VARIANT_ENUM_CAST(RenderingServer::EnvironmentToneMapper);
VARIANT_ENUM_CAST(RenderingServer::EnvironmentSSRRoughnessQuality);
VARIANT_ENUM_CAST(RenderingServer::EnvironmentSSAOQuality);
VARIANT_ENUM_CAST(RenderingServer::EnvironmentSSILQuality);
VARIANT_ENUM_CAST(RenderingServer::EnvironmentSDFGIYScale);
VARIANT_ENUM_CAST(RenderingServer::EnvironmentSDFGIRayCount);
VARIANT_ENUM_CAST(RenderingServer::EnvironmentSDFGIFramesToConverge);
VARIANT_ENUM_CAST(RenderingServer::EnvironmentSDFGIFramesToUpdateLight);
VARIANT_ENUM_CAST(RenderingServer::SubSurfaceScatteringQuality);
VARIANT_ENUM_CAST(RenderingServer::DOFBokehShape);
VARIANT_ENUM_CAST(RenderingServer::DOFBlurQuality);
VARIANT_ENUM_CAST(RenderingServer::InstanceType);
VARIANT_ENUM_CAST(RenderingServer::InstanceFlags);
VARIANT_ENUM_CAST(RenderingServer::ShadowCastingSetting);
VARIANT_ENUM_CAST(RenderingServer::VisibilityRangeFadeMode);
VARIANT_ENUM_CAST(RenderingServer::BakeChannels);
VARIANT_ENUM_CAST(RenderingServer::CanvasTextureChannel);
VARIANT_ENUM_CAST(RenderingServer::NinePatchAxisMode);
VARIANT_ENUM_CAST(RenderingServer::CanvasItemTextureFilter);
VARIANT_ENUM_CAST(RenderingServer::CanvasItemTextureRepeat);
VARIANT_ENUM_CAST(RenderingServer::CanvasGroupMode);
VARIANT_ENUM_CAST(RenderingServer::CanvasLightMode);
VARIANT_ENUM_CAST(RenderingServer::CanvasLightBlendMode);
VARIANT_ENUM_CAST(RenderingServer::CanvasLightShadowFilter);
VARIANT_ENUM_CAST(RenderingServer::CanvasOccluderPolygonCullMode);
VARIANT_ENUM_CAST(RenderingServer::GlobalShaderParameterType);
VARIANT_ENUM_CAST(RenderingServer::RenderingInfo);
VARIANT_ENUM_CAST(RenderingServer::PipelineSource);
VARIANT_ENUM_CAST(RenderingServer::SplashStretchMode);
VARIANT_ENUM_CAST(RenderingServer::Features);

