/*************************************************************************/
/*  register_scene_types.cpp                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "register_scene_types.h"

#include "core/config/project_settings.h"
#include "core/object/class_db.h"
#include "core/os/os.h"
#include "scene/2d/animated_sprite_2d.h"
#include "scene/2d/area_2d.h"
#include "scene/2d/audio_stream_player_2d.h"
#include "scene/2d/back_buffer_copy.h"
#include "scene/2d/camera_2d.h"
#include "scene/2d/canvas_group.h"
#include "scene/2d/canvas_modulate.h"
#include "scene/2d/collision_polygon_2d.h"
#include "scene/2d/collision_shape_2d.h"
#include "scene/2d/cpu_particles_2d.h"
#include "scene/2d/gpu_particles_2d.h"
#include "scene/2d/joints_2d.h"
#include "scene/2d/light_2d.h"
#include "scene/2d/light_occluder_2d.h"
#include "scene/2d/line_2d.h"
#include "scene/2d/mesh_instance_2d.h"
#include "scene/2d/multimesh_instance_2d.h"
#include "scene/2d/navigation_agent_2d.h"
#include "scene/2d/navigation_obstacle_2d.h"
#include "scene/2d/parallax_background.h"
#include "scene/2d/parallax_layer.h"
#include "scene/2d/path_2d.h"
#include "scene/2d/physics_body_2d.h"
#include "scene/2d/polygon_2d.h"
#include "scene/2d/position_2d.h"
#include "scene/2d/ray_cast_2d.h"
#include "scene/2d/remote_transform_2d.h"
#include "scene/2d/skeleton_2d.h"
#include "scene/2d/sprite_2d.h"
#include "scene/2d/tile_map.h"
#include "scene/2d/touch_screen_button.h"
#include "scene/2d/visibility_notifier_2d.h"
#include "scene/2d/y_sort.h"
#include "scene/animation/animation_blend_space_1d.h"
#include "scene/animation/animation_blend_space_2d.h"
#include "scene/animation/animation_blend_tree.h"
#include "scene/animation/animation_node_state_machine.h"
#include "scene/animation/animation_player.h"
#include "scene/animation/animation_tree.h"
#include "scene/animation/root_motion_view.h"
#include "scene/animation/tween.h"
#include "scene/audio/audio_stream_player.h"
#include "scene/debugger/scene_debugger.h"
#include "scene/gui/aspect_ratio_container.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/center_container.h"
#include "scene/gui/check_box.h"
#include "scene/gui/check_button.h"
#include "scene/gui/code_edit.h"
#include "scene/gui/color_picker.h"
#include "scene/gui/color_rect.h"
#include "scene/gui/control.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/file_dialog.h"
#include "scene/gui/graph_edit.h"
#include "scene/gui/graph_node.h"
#include "scene/gui/grid_container.h"
#include "scene/gui/item_list.h"
#include "scene/gui/label.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/link_button.h"
#include "scene/gui/margin_container.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/nine_patch_rect.h"
#include "scene/gui/option_button.h"
#include "scene/gui/panel.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/popup_menu.h"
#include "scene/gui/progress_bar.h"
#include "scene/gui/reference_rect.h"
#include "scene/gui/rich_text_effect.h"
#include "scene/gui/rich_text_label.h"
#include "scene/gui/scroll_bar.h"
#include "scene/gui/scroll_container.h"
#include "scene/gui/separator.h"
#include "scene/gui/slider.h"
#include "scene/gui/spin_box.h"
#include "scene/gui/split_container.h"
#include "scene/gui/subviewport_container.h"
#include "scene/gui/tab_container.h"
#include "scene/gui/tabs.h"
#include "scene/gui/text_edit.h"
#include "scene/gui/texture_button.h"
#include "scene/gui/texture_progress_bar.h"
#include "scene/gui/texture_rect.h"
#include "scene/gui/tree.h"
#include "scene/gui/video_player.h"
#include "scene/main/canvas_item.h"
#include "scene/main/canvas_layer.h"
#include "scene/main/http_request.h"
#include "scene/main/instance_placeholder.h"
#include "scene/main/resource_preloader.h"
#include "scene/main/scene_tree.h"
#include "scene/main/timer.h"
#include "scene/main/viewport.h"
#include "scene/main/window.h"
#include "scene/resources/audio_stream_sample.h"
#include "scene/resources/bit_map.h"
#include "scene/resources/box_shape_3d.h"
#include "scene/resources/camera_effects.h"
#include "scene/resources/capsule_shape_2d.h"
#include "scene/resources/capsule_shape_3d.h"
#include "scene/resources/circle_shape_2d.h"
#include "scene/resources/concave_polygon_shape_2d.h"
#include "scene/resources/concave_polygon_shape_3d.h"
#include "scene/resources/convex_polygon_shape_2d.h"
#include "scene/resources/convex_polygon_shape_3d.h"
#include "scene/resources/cylinder_shape_3d.h"
#include "scene/resources/default_theme/default_theme.h"
#include "scene/resources/font.h"
#include "scene/resources/gradient.h"
#include "scene/resources/height_map_shape_3d.h"
#include "scene/resources/line_shape_2d.h"
#include "scene/resources/material.h"
#include "scene/resources/mesh.h"
#include "scene/resources/mesh_data_tool.h"
#include "scene/resources/navigation_mesh.h"
#include "scene/resources/packed_scene.h"
#include "scene/resources/particles_material.h"
#include "scene/resources/physics_material.h"
#include "scene/resources/polygon_path_finder.h"
#include "scene/resources/primitive_meshes.h"
#include "scene/resources/ray_shape_2d.h"
#include "scene/resources/ray_shape_3d.h"
#include "scene/resources/rectangle_shape_2d.h"
#include "scene/resources/resource_format_text.h"
#include "scene/resources/segment_shape_2d.h"
#include "scene/resources/sky.h"
#include "scene/resources/sky_material.h"
#include "scene/resources/sphere_shape_3d.h"
#include "scene/resources/surface_tool.h"
#include "scene/resources/syntax_highlighter.h"
#include "scene/resources/text_file.h"
#include "scene/resources/text_line.h"
#include "scene/resources/text_paragraph.h"
#include "scene/resources/texture.h"
#include "scene/resources/tile_set.h"
#include "scene/resources/video_stream.h"
#include "scene/resources/visual_shader.h"
#include "scene/resources/visual_shader_nodes.h"
#include "scene/resources/visual_shader_sdf_nodes.h"
#include "scene/resources/world_2d.h"
#include "scene/resources/world_3d.h"
#include "scene/resources/world_margin_shape_3d.h"
#include "scene/scene_string_names.h"

// Needed by animation code, so keep when 3D disabled.
#include "scene/3d/node_3d.h"
#include "scene/3d/skeleton_3d.h"

#include "scene/main/shader_globals_override.h"

#ifndef _3D_DISABLED
#include "scene/3d/area_3d.h"
#include "scene/3d/audio_stream_player_3d.h"
#include "scene/3d/baked_lightmap.h"
#include "scene/3d/bone_attachment_3d.h"
#include "scene/3d/camera_3d.h"
#include "scene/3d/collision_polygon_3d.h"
#include "scene/3d/collision_shape_3d.h"
#include "scene/3d/cpu_particles_3d.h"
#include "scene/3d/decal.h"
#include "scene/3d/gi_probe.h"
#include "scene/3d/gpu_particles_3d.h"
#include "scene/3d/gpu_particles_collision_3d.h"
#include "scene/3d/immediate_geometry_3d.h"
#include "scene/3d/light_3d.h"
#include "scene/3d/lightmap_probe.h"
#include "scene/3d/listener_3d.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/3d/multimesh_instance_3d.h"
#include "scene/3d/navigation_agent_3d.h"
#include "scene/3d/navigation_obstacle_3d.h"
#include "scene/3d/navigation_region_3d.h"
#include "scene/3d/occluder_instance_3d.h"
#include "scene/3d/path_3d.h"
#include "scene/3d/physics_body_3d.h"
#include "scene/3d/physics_joint_3d.h"
#include "scene/3d/position_3d.h"
#include "scene/3d/proximity_group_3d.h"
#include "scene/3d/ray_cast_3d.h"
#include "scene/3d/reflection_probe.h"
#include "scene/3d/remote_transform_3d.h"
#include "scene/3d/skeleton_ik_3d.h"
#include "scene/3d/soft_body_3d.h"
#include "scene/3d/spring_arm_3d.h"
#include "scene/3d/sprite_3d.h"
#include "scene/3d/vehicle_body_3d.h"
#include "scene/3d/visibility_notifier_3d.h"
#include "scene/3d/world_environment.h"
#include "scene/3d/xr_nodes.h"
#include "scene/resources/environment.h"
#include "scene/resources/mesh_library.h"
#endif

static Ref<ResourceFormatSaverText> resource_saver_text;
static Ref<ResourceFormatLoaderText> resource_loader_text;

static Ref<ResourceFormatLoaderFont> resource_loader_font;

#ifndef DISABLE_DEPRECATED
static Ref<ResourceFormatLoaderCompatFont> resource_loader_compat_font;
#endif /* DISABLE_DEPRECATED */

static Ref<ResourceFormatLoaderStreamTexture2D> resource_loader_stream_texture;
static Ref<ResourceFormatLoaderStreamTextureLayered> resource_loader_texture_layered;
static Ref<ResourceFormatLoaderStreamTexture3D> resource_loader_texture_3d;

static Ref<ResourceFormatSaverShader> resource_saver_shader;
static Ref<ResourceFormatLoaderShader> resource_loader_shader;

void register_scene_types() {
	SceneStringNames::create();

	OS::get_singleton()->yield(); //may take time to init

	Node::init_node_hrcr();

	resource_loader_font.instance();
	ResourceLoader::add_resource_format_loader(resource_loader_font);

#ifndef DISABLE_DEPRECATED
	resource_loader_compat_font.instance();
	ResourceLoader::add_resource_format_loader(resource_loader_compat_font);
#endif /* DISABLE_DEPRECATED */

	resource_loader_stream_texture.instance();
	ResourceLoader::add_resource_format_loader(resource_loader_stream_texture);

	resource_loader_texture_layered.instance();
	ResourceLoader::add_resource_format_loader(resource_loader_texture_layered);

	resource_loader_texture_3d.instance();
	ResourceLoader::add_resource_format_loader(resource_loader_texture_3d);

	resource_saver_text.instance();
	ResourceSaver::add_resource_format_saver(resource_saver_text, true);

	resource_loader_text.instance();
	ResourceLoader::add_resource_format_loader(resource_loader_text, true);

	resource_saver_shader.instance();
	ResourceSaver::add_resource_format_saver(resource_saver_shader, true);

	resource_loader_shader.instance();
	ResourceLoader::add_resource_format_loader(resource_loader_shader, true);

	OS::get_singleton()->yield(); //may take time to init

	ClassDB::register_class<Object>();

	ClassDB::register_class<Node>();
	ClassDB::register_virtual_class<InstancePlaceholder>();

	ClassDB::register_virtual_class<Viewport>();
	ClassDB::register_class<SubViewport>();
	ClassDB::register_class<ViewportTexture>();
	ClassDB::register_class<HTTPRequest>();
	ClassDB::register_class<Timer>();
	ClassDB::register_class<CanvasLayer>();
	ClassDB::register_class<CanvasModulate>();
	ClassDB::register_class<ResourcePreloader>();
	ClassDB::register_class<Window>();

	/* REGISTER GUI */

	ClassDB::register_class<ButtonGroup>();
	ClassDB::register_virtual_class<BaseButton>();

	OS::get_singleton()->yield(); //may take time to init

	ClassDB::register_class<Shortcut>();
	ClassDB::register_class<Control>();
	ClassDB::register_class<Button>();
	ClassDB::register_class<Label>();
	ClassDB::register_virtual_class<ScrollBar>();
	ClassDB::register_class<HScrollBar>();
	ClassDB::register_class<VScrollBar>();
	ClassDB::register_class<ProgressBar>();
	ClassDB::register_virtual_class<Slider>();
	ClassDB::register_class<HSlider>();
	ClassDB::register_class<VSlider>();
	ClassDB::register_class<Popup>();
	ClassDB::register_class<PopupPanel>();
	ClassDB::register_class<MenuButton>();
	ClassDB::register_class<CheckBox>();
	ClassDB::register_class<CheckButton>();
	ClassDB::register_class<LinkButton>();
	ClassDB::register_class<Panel>();
	ClassDB::register_virtual_class<Range>();

	OS::get_singleton()->yield(); //may take time to init

	ClassDB::register_class<TextureRect>();
	ClassDB::register_class<ColorRect>();
	ClassDB::register_class<NinePatchRect>();
	ClassDB::register_class<ReferenceRect>();
	ClassDB::register_class<AspectRatioContainer>();
	ClassDB::register_class<TabContainer>();
	ClassDB::register_class<Tabs>();
	ClassDB::register_virtual_class<Separator>();
	ClassDB::register_class<HSeparator>();
	ClassDB::register_class<VSeparator>();
	ClassDB::register_class<TextureButton>();
	ClassDB::register_class<Container>();
	ClassDB::register_virtual_class<BoxContainer>();
	ClassDB::register_class<HBoxContainer>();
	ClassDB::register_class<VBoxContainer>();
	ClassDB::register_class<GridContainer>();
	ClassDB::register_class<CenterContainer>();
	ClassDB::register_class<ScrollContainer>();
	ClassDB::register_class<PanelContainer>();

	OS::get_singleton()->yield(); //may take time to init

	ClassDB::register_class<TextureProgressBar>();
	ClassDB::register_class<ItemList>();

	ClassDB::register_class<LineEdit>();
	ClassDB::register_class<VideoPlayer>();

#ifndef ADVANCED_GUI_DISABLED
	ClassDB::register_class<FileDialog>();

	ClassDB::register_class<PopupMenu>();
	ClassDB::register_class<Tree>();

	ClassDB::register_class<TextEdit>();
	ClassDB::register_class<CodeEdit>();
	ClassDB::register_class<SyntaxHighlighter>();
	ClassDB::register_class<CodeHighlighter>();

	ClassDB::register_virtual_class<TreeItem>();
	ClassDB::register_class<OptionButton>();
	ClassDB::register_class<SpinBox>();
	ClassDB::register_class<ColorPicker>();
	ClassDB::register_class<ColorPickerButton>();
	ClassDB::register_class<RichTextLabel>();
	ClassDB::register_class<RichTextEffect>();
	ClassDB::register_class<CharFXTransform>();

	ClassDB::register_class<AcceptDialog>();
	ClassDB::register_class<ConfirmationDialog>();

	ClassDB::register_class<MarginContainer>();
	ClassDB::register_class<SubViewportContainer>();
	ClassDB::register_virtual_class<SplitContainer>();
	ClassDB::register_class<HSplitContainer>();
	ClassDB::register_class<VSplitContainer>();
	ClassDB::register_class<GraphNode>();
	ClassDB::register_class<GraphEdit>();

	OS::get_singleton()->yield(); //may take time to init

	bool swap_cancel_ok = false;
	if (DisplayServer::get_singleton()) {
		swap_cancel_ok = GLOBAL_DEF_NOVAL("gui/common/swap_cancel_ok", bool(DisplayServer::get_singleton()->get_swap_cancel_ok()));
	}
	AcceptDialog::set_swap_cancel_ok(swap_cancel_ok);
#endif

	/* REGISTER 3D */

	// Needed even with _3D_DISABLED as used in animation code.
	ClassDB::register_class<Node3D>();
	ClassDB::register_virtual_class<Node3DGizmo>();
	ClassDB::register_class<Skin>();
	ClassDB::register_virtual_class<SkinReference>();
	ClassDB::register_class<Skeleton3D>();

	ClassDB::register_class<AnimationPlayer>();
	ClassDB::register_class<Tween>();

	ClassDB::register_class<AnimationTree>();
	ClassDB::register_class<AnimationNode>();
	ClassDB::register_class<AnimationRootNode>();
	ClassDB::register_class<AnimationNodeBlendTree>();
	ClassDB::register_class<AnimationNodeBlendSpace1D>();
	ClassDB::register_class<AnimationNodeBlendSpace2D>();
	ClassDB::register_class<AnimationNodeStateMachine>();
	ClassDB::register_class<AnimationNodeStateMachinePlayback>();

	ClassDB::register_class<AnimationNodeStateMachineTransition>();
	ClassDB::register_class<AnimationNodeOutput>();
	ClassDB::register_class<AnimationNodeOneShot>();
	ClassDB::register_class<AnimationNodeAnimation>();
	ClassDB::register_class<AnimationNodeAdd2>();
	ClassDB::register_class<AnimationNodeAdd3>();
	ClassDB::register_class<AnimationNodeBlend2>();
	ClassDB::register_class<AnimationNodeBlend3>();
	ClassDB::register_class<AnimationNodeTimeScale>();
	ClassDB::register_class<AnimationNodeTimeSeek>();
	ClassDB::register_class<AnimationNodeTransition>();

	ClassDB::register_class<ShaderGlobalsOverride>(); //can be used in any shader

	OS::get_singleton()->yield(); //may take time to init

#ifndef _3D_DISABLED
	ClassDB::register_virtual_class<VisualInstance3D>();
	ClassDB::register_virtual_class<GeometryInstance3D>();
	ClassDB::register_class<Camera3D>();
	ClassDB::register_class<ClippedCamera3D>();
	ClassDB::register_class<Listener3D>();
	ClassDB::register_class<XRCamera3D>();
	ClassDB::register_class<XRController3D>();
	ClassDB::register_class<XRAnchor3D>();
	ClassDB::register_class<XROrigin3D>();
	ClassDB::register_class<MeshInstance3D>();
	ClassDB::register_class<OccluderInstance3D>();
	ClassDB::register_class<Occluder3D>();
	ClassDB::register_class<ImmediateGeometry3D>();
	ClassDB::register_virtual_class<SpriteBase3D>();
	ClassDB::register_class<Sprite3D>();
	ClassDB::register_class<AnimatedSprite3D>();
	ClassDB::register_virtual_class<Light3D>();
	ClassDB::register_class<DirectionalLight3D>();
	ClassDB::register_class<OmniLight3D>();
	ClassDB::register_class<SpotLight3D>();
	ClassDB::register_class<ReflectionProbe>();
	ClassDB::register_class<Decal>();
	ClassDB::register_class<GIProbe>();
	ClassDB::register_class<GIProbeData>();
	ClassDB::register_class<BakedLightmap>();
	ClassDB::register_class<BakedLightmapData>();
	ClassDB::register_class<LightmapProbe>();
	ClassDB::register_virtual_class<Lightmapper>();
	ClassDB::register_class<GPUParticles3D>();
	ClassDB::register_virtual_class<GPUParticlesCollision3D>();
	ClassDB::register_class<GPUParticlesCollisionBox>();
	ClassDB::register_class<GPUParticlesCollisionSphere>();
	ClassDB::register_class<GPUParticlesCollisionSDF>();
	ClassDB::register_class<GPUParticlesCollisionHeightField>();
	ClassDB::register_virtual_class<GPUParticlesAttractor3D>();
	ClassDB::register_class<GPUParticlesAttractorBox>();
	ClassDB::register_class<GPUParticlesAttractorSphere>();
	ClassDB::register_class<GPUParticlesAttractorVectorField>();
	ClassDB::register_class<CPUParticles3D>();
	ClassDB::register_class<Position3D>();

	ClassDB::register_class<RootMotionView>();
	ClassDB::set_class_enabled("RootMotionView", false); //disabled by default, enabled by editor

	OS::get_singleton()->yield(); //may take time to init

	ClassDB::register_virtual_class<CollisionObject3D>();
	ClassDB::register_virtual_class<PhysicsBody3D>();
	ClassDB::register_class<StaticBody3D>();
	ClassDB::register_class<RigidBody3D>();
	ClassDB::register_class<KinematicCollision3D>();
	ClassDB::register_class<KinematicBody3D>();
	ClassDB::register_class<SpringArm3D>();

	ClassDB::register_class<PhysicalBone3D>();
	ClassDB::register_class<SoftBody3D>();

	ClassDB::register_class<SkeletonIK3D>();
	ClassDB::register_class<BoneAttachment3D>();

	ClassDB::register_class<VehicleBody3D>();
	ClassDB::register_class<VehicleWheel3D>();
	ClassDB::register_class<Area3D>();
	ClassDB::register_class<ProximityGroup3D>();
	ClassDB::register_class<CollisionShape3D>();
	ClassDB::register_class<CollisionPolygon3D>();
	ClassDB::register_class<RayCast3D>();
	ClassDB::register_class<MultiMeshInstance3D>();

	ClassDB::register_class<Curve3D>();
	ClassDB::register_class<Path3D>();
	ClassDB::register_class<PathFollow3D>();
	ClassDB::register_class<VisibilityNotifier3D>();
	ClassDB::register_class<VisibilityEnabler3D>();
	ClassDB::register_class<WorldEnvironment>();
	ClassDB::register_class<RemoteTransform3D>();

	ClassDB::register_virtual_class<Joint3D>();
	ClassDB::register_class<PinJoint3D>();
	ClassDB::register_class<HingeJoint3D>();
	ClassDB::register_class<SliderJoint3D>();
	ClassDB::register_class<ConeTwistJoint3D>();
	ClassDB::register_class<Generic6DOFJoint3D>();

	ClassDB::register_class<NavigationRegion3D>();
	ClassDB::register_class<NavigationAgent3D>();
	ClassDB::register_class<NavigationObstacle3D>();

	OS::get_singleton()->yield(); //may take time to init
#endif

	/* REGISTER SHADER */

	ClassDB::register_class<Shader>();
	ClassDB::register_class<VisualShader>();
	ClassDB::register_virtual_class<VisualShaderNode>();
	ClassDB::register_class<VisualShaderNodeCustom>();
	ClassDB::register_class<VisualShaderNodeInput>();
	ClassDB::register_virtual_class<VisualShaderNodeOutput>();
	ClassDB::register_virtual_class<VisualShaderNodeResizableBase>();
	ClassDB::register_virtual_class<VisualShaderNodeGroupBase>();
	ClassDB::register_virtual_class<VisualShaderNodeConstant>();
	ClassDB::register_class<VisualShaderNodeComment>();
	ClassDB::register_class<VisualShaderNodeFloatConstant>();
	ClassDB::register_class<VisualShaderNodeIntConstant>();
	ClassDB::register_class<VisualShaderNodeBooleanConstant>();
	ClassDB::register_class<VisualShaderNodeColorConstant>();
	ClassDB::register_class<VisualShaderNodeVec3Constant>();
	ClassDB::register_class<VisualShaderNodeTransformConstant>();
	ClassDB::register_class<VisualShaderNodeFloatOp>();
	ClassDB::register_class<VisualShaderNodeIntOp>();
	ClassDB::register_class<VisualShaderNodeVectorOp>();
	ClassDB::register_class<VisualShaderNodeColorOp>();
	ClassDB::register_class<VisualShaderNodeTransformMult>();
	ClassDB::register_class<VisualShaderNodeTransformVecMult>();
	ClassDB::register_class<VisualShaderNodeFloatFunc>();
	ClassDB::register_class<VisualShaderNodeIntFunc>();
	ClassDB::register_class<VisualShaderNodeVectorFunc>();
	ClassDB::register_class<VisualShaderNodeColorFunc>();
	ClassDB::register_class<VisualShaderNodeTransformFunc>();
	ClassDB::register_class<VisualShaderNodeDotProduct>();
	ClassDB::register_class<VisualShaderNodeVectorLen>();
	ClassDB::register_class<VisualShaderNodeDeterminant>();
	ClassDB::register_class<VisualShaderNodeScalarDerivativeFunc>();
	ClassDB::register_class<VisualShaderNodeVectorDerivativeFunc>();
	ClassDB::register_class<VisualShaderNodeClamp>();
	ClassDB::register_class<VisualShaderNodeFaceForward>();
	ClassDB::register_class<VisualShaderNodeOuterProduct>();
	ClassDB::register_class<VisualShaderNodeSmoothStep>();
	ClassDB::register_class<VisualShaderNodeStep>();
	ClassDB::register_class<VisualShaderNodeVectorDistance>();
	ClassDB::register_class<VisualShaderNodeVectorRefract>();
	ClassDB::register_class<VisualShaderNodeMix>();
	ClassDB::register_class<VisualShaderNodeVectorCompose>();
	ClassDB::register_class<VisualShaderNodeTransformCompose>();
	ClassDB::register_class<VisualShaderNodeVectorDecompose>();
	ClassDB::register_class<VisualShaderNodeTransformDecompose>();
	ClassDB::register_class<VisualShaderNodeTexture>();
	ClassDB::register_class<VisualShaderNodeCurveTexture>();
	ClassDB::register_virtual_class<VisualShaderNodeSample3D>();
	ClassDB::register_class<VisualShaderNodeTexture2DArray>();
	ClassDB::register_class<VisualShaderNodeTexture3D>();
	ClassDB::register_class<VisualShaderNodeCubemap>();
	ClassDB::register_virtual_class<VisualShaderNodeUniform>();
	ClassDB::register_class<VisualShaderNodeUniformRef>();
	ClassDB::register_class<VisualShaderNodeFloatUniform>();
	ClassDB::register_class<VisualShaderNodeIntUniform>();
	ClassDB::register_class<VisualShaderNodeBooleanUniform>();
	ClassDB::register_class<VisualShaderNodeColorUniform>();
	ClassDB::register_class<VisualShaderNodeVec3Uniform>();
	ClassDB::register_class<VisualShaderNodeTransformUniform>();
	ClassDB::register_class<VisualShaderNodeTextureUniform>();
	ClassDB::register_class<VisualShaderNodeTextureUniformTriplanar>();
	ClassDB::register_class<VisualShaderNodeTexture2DArrayUniform>();
	ClassDB::register_class<VisualShaderNodeTexture3DUniform>();
	ClassDB::register_class<VisualShaderNodeCubemapUniform>();
	ClassDB::register_class<VisualShaderNodeIf>();
	ClassDB::register_class<VisualShaderNodeSwitch>();
	ClassDB::register_class<VisualShaderNodeFresnel>();
	ClassDB::register_class<VisualShaderNodeExpression>();
	ClassDB::register_class<VisualShaderNodeGlobalExpression>();
	ClassDB::register_class<VisualShaderNodeIs>();
	ClassDB::register_class<VisualShaderNodeCompare>();
	ClassDB::register_class<VisualShaderNodeMultiplyAdd>();

	ClassDB::register_class<VisualShaderNodeSDFToScreenUV>();
	ClassDB::register_class<VisualShaderNodeScreenUVToSDF>();
	ClassDB::register_class<VisualShaderNodeTextureSDF>();
	ClassDB::register_class<VisualShaderNodeTextureSDFNormal>();
	ClassDB::register_class<VisualShaderNodeSDFRaymarch>();

	ClassDB::register_class<ShaderMaterial>();
	ClassDB::register_virtual_class<CanvasItem>();
	ClassDB::register_class<CanvasTexture>();
	ClassDB::register_class<CanvasItemMaterial>();
	SceneTree::add_idle_callback(CanvasItemMaterial::flush_changes);
	CanvasItemMaterial::init_shaders();

	/* REGISTER 2D */

	ClassDB::register_class<Node2D>();
	ClassDB::register_class<CanvasGroup>();
	ClassDB::register_class<CPUParticles2D>();
	ClassDB::register_class<GPUParticles2D>();
	ClassDB::register_class<Sprite2D>();
	ClassDB::register_class<SpriteFrames>();
	ClassDB::register_class<AnimatedSprite2D>();
	ClassDB::register_class<Position2D>();
	ClassDB::register_class<Line2D>();
	ClassDB::register_class<MeshInstance2D>();
	ClassDB::register_class<MultiMeshInstance2D>();
	ClassDB::register_virtual_class<CollisionObject2D>();
	ClassDB::register_virtual_class<PhysicsBody2D>();
	ClassDB::register_class<StaticBody2D>();
	ClassDB::register_class<RigidBody2D>();
	ClassDB::register_class<KinematicBody2D>();
	ClassDB::register_class<KinematicCollision2D>();
	ClassDB::register_class<Area2D>();
	ClassDB::register_class<CollisionShape2D>();
	ClassDB::register_class<CollisionPolygon2D>();
	ClassDB::register_class<RayCast2D>();
	ClassDB::register_class<VisibilityNotifier2D>();
	ClassDB::register_class<VisibilityEnabler2D>();
	ClassDB::register_class<Polygon2D>();
	ClassDB::register_class<Skeleton2D>();
	ClassDB::register_class<Bone2D>();
	ClassDB::register_virtual_class<Light2D>();
	ClassDB::register_class<PointLight2D>();
	ClassDB::register_class<DirectionalLight2D>();
	ClassDB::register_class<LightOccluder2D>();
	ClassDB::register_class<OccluderPolygon2D>();
	ClassDB::register_class<YSort>();
	ClassDB::register_class<BackBufferCopy>();

	OS::get_singleton()->yield(); //may take time to init

	ClassDB::register_class<Camera2D>();
	ClassDB::register_virtual_class<Joint2D>();
	ClassDB::register_class<PinJoint2D>();
	ClassDB::register_class<GrooveJoint2D>();
	ClassDB::register_class<DampedSpringJoint2D>();
	ClassDB::register_class<TileSet>();
	ClassDB::register_virtual_class<TileSetSource>();
	ClassDB::register_class<TileSetAtlasSource>();
	ClassDB::register_class<TileSetScenesCollectionSource>();
	ClassDB::register_class<TileData>();
	ClassDB::register_class<TileMap>();
	ClassDB::register_class<ParallaxBackground>();
	ClassDB::register_class<ParallaxLayer>();
	ClassDB::register_class<TouchScreenButton>();
	ClassDB::register_class<RemoteTransform2D>();

	OS::get_singleton()->yield(); //may take time to init

	/* REGISTER RESOURCES */

	ClassDB::register_virtual_class<Shader>();
	ClassDB::register_class<ParticlesMaterial>();
	SceneTree::add_idle_callback(ParticlesMaterial::flush_changes);
	ParticlesMaterial::init_shaders();

	ClassDB::register_class<ProceduralSkyMaterial>();
	ClassDB::register_class<PanoramaSkyMaterial>();
	ClassDB::register_class<PhysicalSkyMaterial>();

	ClassDB::register_virtual_class<Mesh>();
	ClassDB::register_class<ArrayMesh>();
	ClassDB::register_class<MultiMesh>();
	ClassDB::register_class<SurfaceTool>();
	ClassDB::register_class<MeshDataTool>();

#ifndef _3D_DISABLED
	ClassDB::register_virtual_class<PrimitiveMesh>();
	ClassDB::register_class<BoxMesh>();
	ClassDB::register_class<CapsuleMesh>();
	ClassDB::register_class<CylinderMesh>();
	ClassDB::register_class<PlaneMesh>();
	ClassDB::register_class<PrismMesh>();
	ClassDB::register_class<QuadMesh>();
	ClassDB::register_class<SphereMesh>();
	ClassDB::register_class<TubeTrailMesh>();
	ClassDB::register_class<RibbonTrailMesh>();
	ClassDB::register_class<PointMesh>();
	ClassDB::register_virtual_class<Material>();
	ClassDB::register_virtual_class<BaseMaterial3D>();
	ClassDB::register_class<StandardMaterial3D>();
	ClassDB::register_class<ORMMaterial3D>();
	SceneTree::add_idle_callback(BaseMaterial3D::flush_changes);
	BaseMaterial3D::init_shaders();

	ClassDB::register_class<MeshLibrary>();

	OS::get_singleton()->yield(); //may take time to init

	ClassDB::register_virtual_class<Shape3D>();
	ClassDB::register_class<RayShape3D>();
	ClassDB::register_class<SphereShape3D>();
	ClassDB::register_class<BoxShape3D>();
	ClassDB::register_class<CapsuleShape3D>();
	ClassDB::register_class<CylinderShape3D>();
	ClassDB::register_class<HeightMapShape3D>();
	ClassDB::register_class<WorldMarginShape3D>();
	ClassDB::register_class<ConvexPolygonShape3D>();
	ClassDB::register_class<ConcavePolygonShape3D>();

	OS::get_singleton()->yield(); //may take time to init

	ClassDB::register_class<VelocityTracker3D>();
#endif

	ClassDB::register_class<PhysicsMaterial>();
	ClassDB::register_class<World3D>();
	ClassDB::register_class<Environment>();
	ClassDB::register_class<CameraEffects>();
	ClassDB::register_class<World2D>();
	ClassDB::register_virtual_class<Texture>();
	ClassDB::register_virtual_class<Texture2D>();
	ClassDB::register_class<Sky>();
	ClassDB::register_class<StreamTexture2D>();
	ClassDB::register_class<ImageTexture>();
	ClassDB::register_class<AtlasTexture>();
	ClassDB::register_class<MeshTexture>();
	ClassDB::register_class<CurveTexture>();
	ClassDB::register_class<GradientTexture>();
	ClassDB::register_class<ProxyTexture>();
	ClassDB::register_class<AnimatedTexture>();
	ClassDB::register_class<CameraTexture>();
	ClassDB::register_virtual_class<TextureLayered>();
	ClassDB::register_virtual_class<ImageTextureLayered>();
	ClassDB::register_virtual_class<Texture3D>();
	ClassDB::register_class<ImageTexture3D>();
	ClassDB::register_class<StreamTexture3D>();
	ClassDB::register_class<Cubemap>();
	ClassDB::register_class<CubemapArray>();
	ClassDB::register_class<Texture2DArray>();
	ClassDB::register_virtual_class<StreamTextureLayered>();
	ClassDB::register_class<StreamCubemap>();
	ClassDB::register_class<StreamCubemapArray>();
	ClassDB::register_class<StreamTexture2DArray>();

	ClassDB::register_class<Animation>();
	ClassDB::register_class<FontData>();
	ClassDB::register_class<Font>();
	ClassDB::register_class<Curve>();

	ClassDB::register_class<TextFile>();
	ClassDB::register_class<TextLine>();
	ClassDB::register_class<TextParagraph>();

	ClassDB::register_virtual_class<StyleBox>();
	ClassDB::register_class<StyleBoxEmpty>();
	ClassDB::register_class<StyleBoxTexture>();
	ClassDB::register_class<StyleBoxFlat>();
	ClassDB::register_class<StyleBoxLine>();
	ClassDB::register_class<Theme>();

	ClassDB::register_class<PolygonPathFinder>();
	ClassDB::register_class<BitMap>();
	ClassDB::register_class<Gradient>();

	OS::get_singleton()->yield(); //may take time to init

	ClassDB::register_class<AudioStreamPlayer>();
	ClassDB::register_class<AudioStreamPlayer2D>();
#ifndef _3D_DISABLED
	ClassDB::register_class<AudioStreamPlayer3D>();
#endif
	ClassDB::register_virtual_class<VideoStream>();
	ClassDB::register_class<AudioStreamSample>();

	OS::get_singleton()->yield(); //may take time to init

	ClassDB::register_virtual_class<Shape2D>();
	ClassDB::register_class<LineShape2D>();
	ClassDB::register_class<SegmentShape2D>();
	ClassDB::register_class<RayShape2D>();
	ClassDB::register_class<CircleShape2D>();
	ClassDB::register_class<RectangleShape2D>();
	ClassDB::register_class<CapsuleShape2D>();
	ClassDB::register_class<ConvexPolygonShape2D>();
	ClassDB::register_class<ConcavePolygonShape2D>();
	ClassDB::register_class<Curve2D>();
	ClassDB::register_class<Path2D>();
	ClassDB::register_class<PathFollow2D>();

	ClassDB::register_class<NavigationMesh>();
	ClassDB::register_class<NavigationPolygon>();
	ClassDB::register_class<NavigationRegion2D>();
	ClassDB::register_class<NavigationAgent2D>();
	ClassDB::register_class<NavigationObstacle2D>();

	OS::get_singleton()->yield(); //may take time to init

	ClassDB::register_virtual_class<SceneState>();
	ClassDB::register_class<PackedScene>();

	ClassDB::register_class<SceneTree>();
	ClassDB::register_virtual_class<SceneTreeTimer>(); //sorry, you can't create it

#ifndef DISABLE_DEPRECATED
	// Dropped in 4.0, near approximation.
	ClassDB::add_compatibility_class("AnimationTreePlayer", "AnimationTree");
	ClassDB::add_compatibility_class("BitmapFont", "Font");
	ClassDB::add_compatibility_class("DynamicFont", "Font");
	ClassDB::add_compatibility_class("DynamicFontData", "FontData");
	ClassDB::add_compatibility_class("ToolButton", "Button");
	ClassDB::add_compatibility_class("Navigation3D", "Node3D");
	ClassDB::add_compatibility_class("Navigation2D", "Node2D");

	// Renamed in 4.0.
	// Keep alphabetical ordering to easily locate classes and avoid duplicates.
	ClassDB::add_compatibility_class("AnimatedSprite", "AnimatedSprite2D");
	ClassDB::add_compatibility_class("Area", "Area3D");
	ClassDB::add_compatibility_class("ARVRCamera", "XRCamera3D");
	ClassDB::add_compatibility_class("ARVRController", "XRController3D");
	ClassDB::add_compatibility_class("ARVRAnchor", "XRAnchor3D");
	ClassDB::add_compatibility_class("ARVRInterface", "XRInterface");
	ClassDB::add_compatibility_class("ARVROrigin", "XROrigin3D");
	ClassDB::add_compatibility_class("ARVRPositionalTracker", "XRPositionalTracker");
	ClassDB::add_compatibility_class("ARVRServer", "XRServer");
	ClassDB::add_compatibility_class("BoneAttachment", "BoneAttachment3D");
	ClassDB::add_compatibility_class("BoxShape", "BoxShape3D");
	ClassDB::add_compatibility_class("BulletPhysicsDirectBodyState", "BulletPhysicsDirectBodyState3D");
	ClassDB::add_compatibility_class("BulletPhysicsServer", "BulletPhysicsServer3D");
	ClassDB::add_compatibility_class("Camera", "Camera3D");
	ClassDB::add_compatibility_class("CapsuleShape", "CapsuleShape3D");
	ClassDB::add_compatibility_class("ClippedCamera", "ClippedCamera3D");
	ClassDB::add_compatibility_class("CollisionObject", "CollisionObject3D");
	ClassDB::add_compatibility_class("CollisionPolygon", "CollisionPolygon3D");
	ClassDB::add_compatibility_class("CollisionShape", "CollisionShape3D");
	ClassDB::add_compatibility_class("ConcavePolygonShape", "ConcavePolygonShape3D");
	ClassDB::add_compatibility_class("ConeTwistJoint", "ConeTwistJoint3D");
	ClassDB::add_compatibility_class("ConvexPolygonShape", "ConvexPolygonShape3D");
	ClassDB::add_compatibility_class("CPUParticles", "CPUParticles3D");
	ClassDB::add_compatibility_class("CSGBox", "CSGBox3D");
	ClassDB::add_compatibility_class("CSGCombiner", "CSGCombiner3D");
	ClassDB::add_compatibility_class("CSGCylinder", "CSGCylinder3D");
	ClassDB::add_compatibility_class("CSGMesh", "CSGMesh3D");
	ClassDB::add_compatibility_class("CSGPolygon", "CSGPolygon3D");
	ClassDB::add_compatibility_class("CSGPrimitive", "CSGPrimitive3D");
	ClassDB::add_compatibility_class("CSGShape", "CSGShape3D");
	ClassDB::add_compatibility_class("CSGSphere", "CSGSphere3D");
	ClassDB::add_compatibility_class("CSGTorus", "CSGTorus3D");
	ClassDB::add_compatibility_class("CubeMesh", "BoxMesh");
	ClassDB::add_compatibility_class("CylinderShape", "CylinderShape3D");
	ClassDB::add_compatibility_class("DirectionalLight", "DirectionalLight3D");
	ClassDB::add_compatibility_class("EditorSpatialGizmo", "EditorNode3DGizmo");
	ClassDB::add_compatibility_class("EditorSpatialGizmoPlugin", "EditorNode3DGizmoPlugin");
	ClassDB::add_compatibility_class("Generic6DOFJoint", "Generic6DOFJoint3D");
	ClassDB::add_compatibility_class("HeightMapShape", "HeightMapShape3D");
	ClassDB::add_compatibility_class("HingeJoint", "HingeJoint3D");
	ClassDB::add_compatibility_class("ImmediateGeometry", "ImmediateGeometry3D");
	ClassDB::add_compatibility_class("Joint", "Joint3D");
	ClassDB::add_compatibility_class("KinematicBody", "KinematicBody3D");
	ClassDB::add_compatibility_class("KinematicCollision", "KinematicCollision3D");
	ClassDB::add_compatibility_class("Light", "Light3D");
	ClassDB::add_compatibility_class("Listener", "Listener3D");
	ClassDB::add_compatibility_class("MeshInstance", "MeshInstance3D");
	ClassDB::add_compatibility_class("MultiMeshInstance", "MultiMeshInstance3D");
	ClassDB::add_compatibility_class("NavigationAgent", "NavigationAgent3D");
	ClassDB::add_compatibility_class("NavigationMeshInstance", "NavigationRegion3D");
	ClassDB::add_compatibility_class("NavigationObstacle", "NavigationObstacle3D");
	ClassDB::add_compatibility_class("NavigationPolygonInstance", "NavigationRegion2D");
	ClassDB::add_compatibility_class("NavigationRegion", "NavigationRegion3D");
	ClassDB::add_compatibility_class("Navigation2DServer", "NavigationServer2D");
	ClassDB::add_compatibility_class("NavigationServer", "NavigationServer3D");
	ClassDB::add_compatibility_class("OmniLight", "OmniLight3D");
	ClassDB::add_compatibility_class("PanoramaSky", "Sky");
	ClassDB::add_compatibility_class("Particles", "GPUParticles3D");
	ClassDB::add_compatibility_class("Particles2D", "GPUParticles2D");
	ClassDB::add_compatibility_class("Path", "Path3D");
	ClassDB::add_compatibility_class("PathFollow", "PathFollow3D");
	ClassDB::add_compatibility_class("PhysicalBone", "PhysicalBone3D");
	ClassDB::add_compatibility_class("Physics2DDirectBodyStateSW", "PhysicsDirectBodyState2DSW");
	ClassDB::add_compatibility_class("Physics2DDirectBodyState", "PhysicsDirectBodyState2D");
	ClassDB::add_compatibility_class("Physics2DDirectSpaceState", "PhysicsDirectSpaceState2D");
	ClassDB::add_compatibility_class("Physics2DServerSW", "PhysicsServer2DSW");
	ClassDB::add_compatibility_class("Physics2DServer", "PhysicsServer2D");
	ClassDB::add_compatibility_class("Physics2DShapeQueryParameters", "PhysicsShapeQueryParameters2D");
	ClassDB::add_compatibility_class("Physics2DShapeQueryResult", "PhysicsShapeQueryResult2D");
	ClassDB::add_compatibility_class("Physics2DTestMotionResult", "PhysicsTestMotionResult2D");
	ClassDB::add_compatibility_class("PhysicsBody", "PhysicsBody3D");
	ClassDB::add_compatibility_class("PhysicsDirectBodyState", "PhysicsDirectBodyState3D");
	ClassDB::add_compatibility_class("PhysicsDirectSpaceState", "PhysicsDirectSpaceState3D");
	ClassDB::add_compatibility_class("PhysicsServer", "PhysicsServer3D");
	ClassDB::add_compatibility_class("PhysicsShapeQueryParameters", "PhysicsShapeQueryParameters3D");
	ClassDB::add_compatibility_class("PhysicsShapeQueryResult", "PhysicsShapeQueryResult3D");
	ClassDB::add_compatibility_class("PinJoint", "PinJoint3D");
	ClassDB::add_compatibility_class("PlaneShape", "WorldMarginShape3D");
	ClassDB::add_compatibility_class("ProceduralSky", "Sky");
	ClassDB::add_compatibility_class("ProximityGroup", "ProximityGroup3D");
	ClassDB::add_compatibility_class("RayCast", "RayCast3D");
	ClassDB::add_compatibility_class("RayShape", "RayShape3D");
	ClassDB::add_compatibility_class("RemoteTransform", "RemoteTransform3D");
	ClassDB::add_compatibility_class("RigidBody", "RigidBody3D");
	ClassDB::add_compatibility_class("Shape", "Shape3D");
	ClassDB::add_compatibility_class("ShortCut", "Shortcut");
	ClassDB::add_compatibility_class("Skeleton", "Skeleton3D");
	ClassDB::add_compatibility_class("SkeletonIK", "SkeletonIK3D");
	ClassDB::add_compatibility_class("SliderJoint", "SliderJoint3D");
	ClassDB::add_compatibility_class("SoftBody", "SoftBody3D");
	ClassDB::add_compatibility_class("Spatial", "Node3D");
	ClassDB::add_compatibility_class("SpatialGizmo", "Node3DGizmo");
	ClassDB::add_compatibility_class("SpatialMaterial", "StandardMaterial3D");
	ClassDB::add_compatibility_class("SpatialVelocityTracker", "VelocityTracker3D");
	ClassDB::add_compatibility_class("SphereShape", "SphereShape3D");
	ClassDB::add_compatibility_class("SpotLight", "SpotLight3D");
	ClassDB::add_compatibility_class("SpringArm", "SpringArm3D");
	ClassDB::add_compatibility_class("Sprite", "Sprite2D");
	ClassDB::add_compatibility_class("StaticBody", "StaticBody3D");
	ClassDB::add_compatibility_class("TextureProgress", "TextureProgressBar");
	ClassDB::add_compatibility_class("VehicleBody", "VehicleBody3D");
	ClassDB::add_compatibility_class("VehicleWheel", "VehicleWheel3D");
	ClassDB::add_compatibility_class("ViewportContainer", "SubViewportContainer");
	ClassDB::add_compatibility_class("Viewport", "SubViewport");
	ClassDB::add_compatibility_class("VisibilityEnabler", "VisibilityEnabler3D");
	ClassDB::add_compatibility_class("VisibilityNotifier", "VisibilityNotifier3D");
	ClassDB::add_compatibility_class("VisualServer", "RenderingServer");
	ClassDB::add_compatibility_class("VisualShaderNodeScalarConstant", "VisualShaderNodeFloatConstant");
	ClassDB::add_compatibility_class("VisualShaderNodeScalarFunc", "VisualShaderNodeFloatFunc");
	ClassDB::add_compatibility_class("VisualShaderNodeScalarOp", "VisualShaderNodeFloatOp");
	ClassDB::add_compatibility_class("VisualShaderNodeScalarUniform", "VisualShaderNodeFloatUniform");
	ClassDB::add_compatibility_class("VisualShaderNodeScalarClamp", "VisualShaderNodeClamp");
	ClassDB::add_compatibility_class("VisualShaderNodeVectorClamp", "VisualShaderNodeClamp");
	ClassDB::add_compatibility_class("VisualShaderNodeScalarInterp", "VisualShaderNodeMix");
	ClassDB::add_compatibility_class("VisualShaderNodeVectorInterp", "VisualShaderNodeMix");
	ClassDB::add_compatibility_class("VisualShaderNodeVectorScalarMix", "VisualShaderNodeMix");
	ClassDB::add_compatibility_class("VisualShaderNodeScalarSmoothStep", "VisualShaderNodeSmoothStep");
	ClassDB::add_compatibility_class("VisualShaderNodeVectorSmoothStep", "VisualShaderNodeSmoothStep");
	ClassDB::add_compatibility_class("VisualShaderNodeVectorScalarSmoothStep", "VisualShaderNodeSmoothStep");
	ClassDB::add_compatibility_class("VisualShaderNodeVectorScalarStep", "VisualShaderNodeStep");
	ClassDB::add_compatibility_class("VisualShaderNodeScalarSwitch", "VisualShaderNodeSwitch");
	ClassDB::add_compatibility_class("World", "World3D");
	ClassDB::add_compatibility_class("StreamTexture", "StreamTexture2D");
	ClassDB::add_compatibility_class("Light2D", "PointLight2D");

#endif /* DISABLE_DEPRECATED */

	OS::get_singleton()->yield(); //may take time to init

	for (int i = 0; i < 20; i++) {
		GLOBAL_DEF_BASIC(vformat("layer_names/2d_render/layer_%d", i), "");
		GLOBAL_DEF_BASIC(vformat("layer_names/2d_physics/layer_%d", i), "");
		GLOBAL_DEF_BASIC(vformat("layer_names/2d_navigation/layer_%d", i), "");
		GLOBAL_DEF_BASIC(vformat("layer_names/3d_render/layer_%d", i), "");
		GLOBAL_DEF_BASIC(vformat("layer_names/3d_physics/layer_%d", i), "");
		GLOBAL_DEF_BASIC(vformat("layer_names/3d_navigation/layer_%d", i), "");
	}

	bool default_theme_hidpi = GLOBAL_DEF("gui/theme/use_hidpi", false);
	ProjectSettings::get_singleton()->set_custom_property_info("gui/theme/use_hidpi", PropertyInfo(Variant::BOOL, "gui/theme/use_hidpi", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED));
	String theme_path = GLOBAL_DEF_RST("gui/theme/custom", "");
	ProjectSettings::get_singleton()->set_custom_property_info("gui/theme/custom", PropertyInfo(Variant::STRING, "gui/theme/custom", PROPERTY_HINT_FILE, "*.tres,*.res,*.theme", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED));
	String font_path = GLOBAL_DEF_RST("gui/theme/custom_font", "");
	ProjectSettings::get_singleton()->set_custom_property_info("gui/theme/custom_font", PropertyInfo(Variant::STRING, "gui/theme/custom_font", PROPERTY_HINT_FILE, "*.tres,*.res,*.font", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED));

	Ref<Font> font;
	if (font_path != String()) {
		font = ResourceLoader::load(font_path);
		if (!font.is_valid()) {
			ERR_PRINT("Error loading custom font '" + font_path + "'");
		}
	}

	// Always make the default theme to avoid invalid default font/icon/style in the given theme.
	if (RenderingServer::get_singleton()) {
		make_default_theme(default_theme_hidpi, font);
		ColorPicker::init_shaders(); // RenderingServer needs to exist for this to succeed.
	}

	if (theme_path != String()) {
		Ref<Theme> theme = ResourceLoader::load(theme_path);
		if (theme.is_valid()) {
			Theme::set_project_default(theme);
			if (font.is_valid()) {
				Theme::set_default_font(font);
			}
		} else {
			ERR_PRINT("Error loading custom theme '" + theme_path + "'");
		}
	}
	SceneDebugger::initialize();
}

void unregister_scene_types() {
	SceneDebugger::deinitialize();
	clear_default_theme();

	ResourceLoader::remove_resource_format_loader(resource_loader_font);
	resource_loader_font.unref();

#ifndef DISABLE_DEPRECATED
	ResourceLoader::remove_resource_format_loader(resource_loader_compat_font);
	resource_loader_compat_font.unref();
#endif /* DISABLE_DEPRECATED */

	ResourceLoader::remove_resource_format_loader(resource_loader_texture_layered);
	resource_loader_texture_layered.unref();

	ResourceLoader::remove_resource_format_loader(resource_loader_texture_3d);
	resource_loader_texture_3d.unref();

	ResourceLoader::remove_resource_format_loader(resource_loader_stream_texture);
	resource_loader_stream_texture.unref();

	ResourceSaver::remove_resource_format_saver(resource_saver_text);
	resource_saver_text.unref();

	ResourceLoader::remove_resource_format_loader(resource_loader_text);
	resource_loader_text.unref();

	ResourceSaver::remove_resource_format_saver(resource_saver_shader);
	resource_saver_shader.unref();

	ResourceLoader::remove_resource_format_loader(resource_loader_shader);
	resource_loader_shader.unref();

	//StandardMaterial3D is not initialised when 3D is disabled, so it shouldn't be cleaned up either
#ifndef _3D_DISABLED
	BaseMaterial3D::finish_shaders();
#endif // _3D_DISABLED

	ParticlesMaterial::finish_shaders();
	CanvasItemMaterial::finish_shaders();
	ColorPicker::finish_shaders();
	SceneStringNames::free();
}
