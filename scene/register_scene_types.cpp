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
#include "core/extension/native_extension_manager.h"
#include "core/object/class_db.h"
#include "core/os/os.h"
#include "scene/2d/animated_sprite_2d.h"
#include "scene/2d/area_2d.h"
#include "scene/2d/audio_listener_2d.h"
#include "scene/2d/audio_stream_player_2d.h"
#include "scene/2d/back_buffer_copy.h"
#include "scene/2d/camera_2d.h"
#include "scene/2d/canvas_group.h"
#include "scene/2d/canvas_modulate.h"
#include "scene/2d/collision_polygon_2d.h"
#include "scene/2d/collision_shape_2d.h"
#include "scene/2d/cpu_particles_2d.h"
#include "scene/2d/gpu_particles_2d.h"
#include "scene/2d/joint_2d.h"
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
#include "scene/2d/physical_bone_2d.h"
#include "scene/2d/physics_body_2d.h"
#include "scene/2d/polygon_2d.h"
#include "scene/2d/position_2d.h"
#include "scene/2d/ray_cast_2d.h"
#include "scene/2d/remote_transform_2d.h"
#include "scene/2d/shape_cast_2d.h"
#include "scene/2d/skeleton_2d.h"
#include "scene/2d/sprite_2d.h"
#include "scene/2d/tile_map.h"
#include "scene/2d/touch_screen_button.h"
#include "scene/2d/visible_on_screen_notifier_2d.h"
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
#include "scene/gui/tab_bar.h"
#include "scene/gui/tab_container.h"
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
#include "scene/resources/immediate_mesh.h"
#include "scene/resources/material.h"
#include "scene/resources/mesh.h"
#include "scene/resources/mesh_data_tool.h"
#include "scene/resources/navigation_mesh.h"
#include "scene/resources/packed_scene.h"
#include "scene/resources/particles_material.h"
#include "scene/resources/physics_material.h"
#include "scene/resources/polygon_path_finder.h"
#include "scene/resources/primitive_meshes.h"
#include "scene/resources/rectangle_shape_2d.h"
#include "scene/resources/resource_format_text.h"
#include "scene/resources/segment_shape_2d.h"
#include "scene/resources/separation_ray_shape_2d.h"
#include "scene/resources/separation_ray_shape_3d.h"
#include "scene/resources/skeleton_modification_2d.h"
#include "scene/resources/skeleton_modification_2d_ccdik.h"
#include "scene/resources/skeleton_modification_2d_fabrik.h"
#include "scene/resources/skeleton_modification_2d_jiggle.h"
#include "scene/resources/skeleton_modification_2d_lookat.h"
#include "scene/resources/skeleton_modification_2d_physicalbones.h"
#include "scene/resources/skeleton_modification_2d_stackholder.h"
#include "scene/resources/skeleton_modification_2d_twoboneik.h"
#include "scene/resources/skeleton_modification_3d.h"
#include "scene/resources/skeleton_modification_3d_ccdik.h"
#include "scene/resources/skeleton_modification_3d_fabrik.h"
#include "scene/resources/skeleton_modification_3d_jiggle.h"
#include "scene/resources/skeleton_modification_3d_lookat.h"
#include "scene/resources/skeleton_modification_3d_stackholder.h"
#include "scene/resources/skeleton_modification_3d_twoboneik.h"
#include "scene/resources/skeleton_modification_stack_2d.h"
#include "scene/resources/skeleton_modification_stack_3d.h"
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
#include "scene/resources/visual_shader_particle_nodes.h"
#include "scene/resources/visual_shader_sdf_nodes.h"
#include "scene/resources/world_2d.h"
#include "scene/resources/world_3d.h"
#include "scene/resources/world_boundary_shape_2d.h"
#include "scene/resources/world_boundary_shape_3d.h"
#include "scene/scene_string_names.h"

#include "scene/main/shader_globals_override.h"

#ifndef _3D_DISABLED
#include "scene/3d/area_3d.h"
#include "scene/3d/audio_listener_3d.h"
#include "scene/3d/audio_stream_player_3d.h"
#include "scene/3d/bone_attachment_3d.h"
#include "scene/3d/camera_3d.h"
#include "scene/3d/collision_polygon_3d.h"
#include "scene/3d/collision_shape_3d.h"
#include "scene/3d/cpu_particles_3d.h"
#include "scene/3d/decal.h"
#include "scene/3d/fog_volume.h"
#include "scene/3d/gpu_particles_3d.h"
#include "scene/3d/gpu_particles_collision_3d.h"
#include "scene/3d/importer_mesh_instance_3d.h"
#include "scene/3d/joint_3d.h"
#include "scene/3d/light_3d.h"
#include "scene/3d/lightmap_gi.h"
#include "scene/3d/lightmap_probe.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/3d/multimesh_instance_3d.h"
#include "scene/3d/navigation_agent_3d.h"
#include "scene/3d/navigation_obstacle_3d.h"
#include "scene/3d/navigation_region_3d.h"
#include "scene/3d/node_3d.h"
#include "scene/3d/occluder_instance_3d.h"
#include "scene/3d/path_3d.h"
#include "scene/3d/physics_body_3d.h"
#include "scene/3d/position_3d.h"
#include "scene/3d/proximity_group_3d.h"
#include "scene/3d/ray_cast_3d.h"
#include "scene/3d/reflection_probe.h"
#include "scene/3d/remote_transform_3d.h"
#include "scene/3d/skeleton_3d.h"
#include "scene/3d/skeleton_ik_3d.h"
#include "scene/3d/soft_dynamic_body_3d.h"
#include "scene/3d/spring_arm_3d.h"
#include "scene/3d/sprite_3d.h"
#include "scene/3d/vehicle_body_3d.h"
#include "scene/3d/visible_on_screen_notifier_3d.h"
#include "scene/3d/voxel_gi.h"
#include "scene/3d/world_environment.h"
#include "scene/3d/xr_nodes.h"
#include "scene/resources/environment.h"
#include "scene/resources/fog_material.h"
#include "scene/resources/importer_mesh.h"
#include "scene/resources/mesh_library.h"
#endif

static Ref<ResourceFormatSaverText> resource_saver_text;
static Ref<ResourceFormatLoaderText> resource_loader_text;

static Ref<ResourceFormatLoaderStreamTexture2D> resource_loader_stream_texture;
static Ref<ResourceFormatLoaderStreamTextureLayered> resource_loader_texture_layered;
static Ref<ResourceFormatLoaderStreamTexture3D> resource_loader_texture_3d;

static Ref<ResourceFormatSaverShader> resource_saver_shader;
static Ref<ResourceFormatLoaderShader> resource_loader_shader;

void register_scene_types() {
	SceneStringNames::create();

	OS::get_singleton()->yield(); // may take time to init

	Node::init_node_hrcr();

	resource_loader_stream_texture.instantiate();
	ResourceLoader::add_resource_format_loader(resource_loader_stream_texture);

	resource_loader_texture_layered.instantiate();
	ResourceLoader::add_resource_format_loader(resource_loader_texture_layered);

	resource_loader_texture_3d.instantiate();
	ResourceLoader::add_resource_format_loader(resource_loader_texture_3d);

	resource_saver_text.instantiate();
	ResourceSaver::add_resource_format_saver(resource_saver_text, true);

	resource_loader_text.instantiate();
	ResourceLoader::add_resource_format_loader(resource_loader_text, true);

	resource_saver_shader.instantiate();
	ResourceSaver::add_resource_format_saver(resource_saver_shader, true);

	resource_loader_shader.instantiate();
	ResourceLoader::add_resource_format_loader(resource_loader_shader, true);

	OS::get_singleton()->yield(); // may take time to init

	GDREGISTER_CLASS(Object);

	GDREGISTER_CLASS(Node);
	GDREGISTER_VIRTUAL_CLASS(InstancePlaceholder);

	GDREGISTER_VIRTUAL_CLASS(Viewport);
	GDREGISTER_CLASS(SubViewport);
	GDREGISTER_CLASS(ViewportTexture);
	GDREGISTER_CLASS(HTTPRequest);
	GDREGISTER_CLASS(Timer);
	GDREGISTER_CLASS(CanvasLayer);
	GDREGISTER_CLASS(CanvasModulate);
	GDREGISTER_CLASS(ResourcePreloader);
	GDREGISTER_CLASS(Window);

	/* REGISTER GUI */

	GDREGISTER_CLASS(ButtonGroup);
	GDREGISTER_VIRTUAL_CLASS(BaseButton);

	OS::get_singleton()->yield(); // may take time to init

	GDREGISTER_CLASS(Control);
	GDREGISTER_CLASS(Button);
	GDREGISTER_CLASS(Label);
	GDREGISTER_VIRTUAL_CLASS(ScrollBar);
	GDREGISTER_CLASS(HScrollBar);
	GDREGISTER_CLASS(VScrollBar);
	GDREGISTER_CLASS(ProgressBar);
	GDREGISTER_VIRTUAL_CLASS(Slider);
	GDREGISTER_CLASS(HSlider);
	GDREGISTER_CLASS(VSlider);
	GDREGISTER_CLASS(Popup);
	GDREGISTER_CLASS(PopupPanel);
	GDREGISTER_CLASS(MenuButton);
	GDREGISTER_CLASS(CheckBox);
	GDREGISTER_CLASS(CheckButton);
	GDREGISTER_CLASS(LinkButton);
	GDREGISTER_CLASS(Panel);
	GDREGISTER_VIRTUAL_CLASS(Range);

	OS::get_singleton()->yield(); // may take time to init

	GDREGISTER_CLASS(TextureRect);
	GDREGISTER_CLASS(ColorRect);
	GDREGISTER_CLASS(NinePatchRect);
	GDREGISTER_CLASS(ReferenceRect);
	GDREGISTER_CLASS(AspectRatioContainer);
	GDREGISTER_CLASS(TabContainer);
	GDREGISTER_CLASS(TabBar);
	GDREGISTER_VIRTUAL_CLASS(Separator);
	GDREGISTER_CLASS(HSeparator);
	GDREGISTER_CLASS(VSeparator);
	GDREGISTER_CLASS(TextureButton);
	GDREGISTER_CLASS(Container);
	GDREGISTER_VIRTUAL_CLASS(BoxContainer);
	GDREGISTER_CLASS(HBoxContainer);
	GDREGISTER_CLASS(VBoxContainer);
	GDREGISTER_CLASS(GridContainer);
	GDREGISTER_CLASS(CenterContainer);
	GDREGISTER_CLASS(ScrollContainer);
	GDREGISTER_CLASS(PanelContainer);

	OS::get_singleton()->yield(); // may take time to init

	GDREGISTER_CLASS(TextureProgressBar);
	GDREGISTER_CLASS(ItemList);

	GDREGISTER_CLASS(LineEdit);
	GDREGISTER_CLASS(VideoPlayer);

#ifndef ADVANCED_GUI_DISABLED
	GDREGISTER_CLASS(FileDialog);

	GDREGISTER_CLASS(PopupMenu);
	GDREGISTER_CLASS(Tree);

	GDREGISTER_CLASS(TextEdit);
	GDREGISTER_CLASS(CodeEdit);
	GDREGISTER_CLASS(SyntaxHighlighter);
	GDREGISTER_CLASS(CodeHighlighter);

	GDREGISTER_VIRTUAL_CLASS(TreeItem);
	GDREGISTER_CLASS(OptionButton);
	GDREGISTER_CLASS(SpinBox);
	GDREGISTER_CLASS(ColorPicker);
	GDREGISTER_CLASS(ColorPickerButton);
	GDREGISTER_CLASS(RichTextLabel);
	GDREGISTER_CLASS(RichTextEffect);
	GDREGISTER_CLASS(CharFXTransform);

	GDREGISTER_CLASS(AcceptDialog);
	GDREGISTER_CLASS(ConfirmationDialog);

	GDREGISTER_CLASS(MarginContainer);
	GDREGISTER_CLASS(SubViewportContainer);
	GDREGISTER_VIRTUAL_CLASS(SplitContainer);
	GDREGISTER_CLASS(HSplitContainer);
	GDREGISTER_CLASS(VSplitContainer);
	GDREGISTER_CLASS(GraphNode);
	GDREGISTER_CLASS(GraphEdit);

	OS::get_singleton()->yield(); // may take time to init

	bool swap_cancel_ok = false;
	if (DisplayServer::get_singleton()) {
		swap_cancel_ok = GLOBAL_DEF_NOVAL("gui/common/swap_cancel_ok", bool(DisplayServer::get_singleton()->get_swap_cancel_ok()));
	}
	AcceptDialog::set_swap_cancel_ok(swap_cancel_ok);
#endif

	/* REGISTER ANIMATION */

	GDREGISTER_CLASS(AnimationPlayer);
	GDREGISTER_CLASS(Tween);
	GDREGISTER_VIRTUAL_CLASS(Tweener);
	GDREGISTER_CLASS(PropertyTweener);
	GDREGISTER_CLASS(IntervalTweener);
	GDREGISTER_CLASS(CallbackTweener);
	GDREGISTER_CLASS(MethodTweener);

	GDREGISTER_CLASS(AnimationTree);
	GDREGISTER_CLASS(AnimationNode);
	GDREGISTER_CLASS(AnimationRootNode);
	GDREGISTER_CLASS(AnimationNodeBlendTree);
	GDREGISTER_CLASS(AnimationNodeBlendSpace1D);
	GDREGISTER_CLASS(AnimationNodeBlendSpace2D);
	GDREGISTER_CLASS(AnimationNodeStateMachine);
	GDREGISTER_CLASS(AnimationNodeStateMachinePlayback);

	GDREGISTER_CLASS(AnimationNodeStateMachineTransition);
	GDREGISTER_CLASS(AnimationNodeOutput);
	GDREGISTER_CLASS(AnimationNodeOneShot);
	GDREGISTER_CLASS(AnimationNodeAnimation);
	GDREGISTER_CLASS(AnimationNodeAdd2);
	GDREGISTER_CLASS(AnimationNodeAdd3);
	GDREGISTER_CLASS(AnimationNodeBlend2);
	GDREGISTER_CLASS(AnimationNodeBlend3);
	GDREGISTER_CLASS(AnimationNodeTimeScale);
	GDREGISTER_CLASS(AnimationNodeTimeSeek);
	GDREGISTER_CLASS(AnimationNodeTransition);

	GDREGISTER_CLASS(ShaderGlobalsOverride); // can be used in any shader

	OS::get_singleton()->yield(); // may take time to init

	/* REGISTER 3D */

#ifndef _3D_DISABLED
	GDREGISTER_CLASS(Node3D);
	GDREGISTER_VIRTUAL_CLASS(Node3DGizmo);
	GDREGISTER_CLASS(Skin);
	GDREGISTER_VIRTUAL_CLASS(SkinReference);
	GDREGISTER_CLASS(Skeleton3D);
	GDREGISTER_CLASS(ImporterMesh);
	GDREGISTER_CLASS(ImporterMeshInstance3D);
	GDREGISTER_VIRTUAL_CLASS(VisualInstance3D);
	GDREGISTER_VIRTUAL_CLASS(GeometryInstance3D);
	GDREGISTER_CLASS(Camera3D);
	GDREGISTER_CLASS(AudioListener3D);
	GDREGISTER_CLASS(XRCamera3D);
	GDREGISTER_VIRTUAL_CLASS(XRNode3D);
	GDREGISTER_CLASS(XRController3D);
	GDREGISTER_CLASS(XRAnchor3D);
	GDREGISTER_CLASS(XROrigin3D);
	GDREGISTER_CLASS(MeshInstance3D);
	GDREGISTER_CLASS(OccluderInstance3D);
	GDREGISTER_CLASS(Occluder3D);
	GDREGISTER_VIRTUAL_CLASS(SpriteBase3D);
	GDREGISTER_CLASS(Sprite3D);
	GDREGISTER_CLASS(AnimatedSprite3D);
	GDREGISTER_VIRTUAL_CLASS(Light3D);
	GDREGISTER_CLASS(DirectionalLight3D);
	GDREGISTER_CLASS(OmniLight3D);
	GDREGISTER_CLASS(SpotLight3D);
	GDREGISTER_CLASS(ReflectionProbe);
	GDREGISTER_CLASS(Decal);
	GDREGISTER_CLASS(VoxelGI);
	GDREGISTER_CLASS(VoxelGIData);
	GDREGISTER_CLASS(LightmapGI);
	GDREGISTER_CLASS(LightmapGIData);
	GDREGISTER_CLASS(LightmapProbe);
	GDREGISTER_VIRTUAL_CLASS(Lightmapper);
	GDREGISTER_CLASS(GPUParticles3D);
	GDREGISTER_VIRTUAL_CLASS(GPUParticlesCollision3D);
	GDREGISTER_CLASS(GPUParticlesCollisionBox);
	GDREGISTER_CLASS(GPUParticlesCollisionSphere);
	GDREGISTER_CLASS(GPUParticlesCollisionSDF);
	GDREGISTER_CLASS(GPUParticlesCollisionHeightField);
	GDREGISTER_VIRTUAL_CLASS(GPUParticlesAttractor3D);
	GDREGISTER_CLASS(GPUParticlesAttractorBox);
	GDREGISTER_CLASS(GPUParticlesAttractorSphere);
	GDREGISTER_CLASS(GPUParticlesAttractorVectorField);
	GDREGISTER_CLASS(CPUParticles3D);
	GDREGISTER_CLASS(Position3D);

	GDREGISTER_CLASS(RootMotionView);
	ClassDB::set_class_enabled("RootMotionView", false); // disabled by default, enabled by editor

	OS::get_singleton()->yield(); // may take time to init

	GDREGISTER_VIRTUAL_CLASS(CollisionObject3D);
	GDREGISTER_VIRTUAL_CLASS(PhysicsBody3D);
	GDREGISTER_CLASS(StaticBody3D);
	GDREGISTER_CLASS(AnimatableBody3D);
	GDREGISTER_CLASS(RigidDynamicBody3D);
	GDREGISTER_CLASS(KinematicCollision3D);
	GDREGISTER_CLASS(CharacterBody3D);
	GDREGISTER_CLASS(SpringArm3D);

	GDREGISTER_CLASS(PhysicalBone3D);
	GDREGISTER_CLASS(SoftDynamicBody3D);

	GDREGISTER_CLASS(SkeletonIK3D);
	GDREGISTER_CLASS(BoneAttachment3D);

	GDREGISTER_CLASS(VehicleBody3D);
	GDREGISTER_CLASS(VehicleWheel3D);
	GDREGISTER_CLASS(Area3D);
	GDREGISTER_CLASS(ProximityGroup3D);
	GDREGISTER_CLASS(CollisionShape3D);
	GDREGISTER_CLASS(CollisionPolygon3D);
	GDREGISTER_CLASS(RayCast3D);
	GDREGISTER_CLASS(MultiMeshInstance3D);

	GDREGISTER_CLASS(Curve3D);
	GDREGISTER_CLASS(Path3D);
	GDREGISTER_CLASS(PathFollow3D);
	GDREGISTER_CLASS(VisibleOnScreenNotifier3D);
	GDREGISTER_CLASS(VisibleOnScreenEnabler3D);
	GDREGISTER_CLASS(WorldEnvironment);
	GDREGISTER_CLASS(FogVolume);
	GDREGISTER_CLASS(FogMaterial);
	GDREGISTER_CLASS(RemoteTransform3D);

	GDREGISTER_VIRTUAL_CLASS(Joint3D);
	GDREGISTER_CLASS(PinJoint3D);
	GDREGISTER_CLASS(HingeJoint3D);
	GDREGISTER_CLASS(SliderJoint3D);
	GDREGISTER_CLASS(ConeTwistJoint3D);
	GDREGISTER_CLASS(Generic6DOFJoint3D);

	GDREGISTER_CLASS(NavigationRegion3D);
	GDREGISTER_CLASS(NavigationAgent3D);
	GDREGISTER_CLASS(NavigationObstacle3D);

	OS::get_singleton()->yield(); // may take time to init
#endif

	/* REGISTER SHADER */

	GDREGISTER_CLASS(Shader);
	GDREGISTER_CLASS(VisualShader);
	GDREGISTER_VIRTUAL_CLASS(VisualShaderNode);
	GDREGISTER_CLASS(VisualShaderNodeCustom);
	GDREGISTER_CLASS(VisualShaderNodeInput);
	GDREGISTER_VIRTUAL_CLASS(VisualShaderNodeOutput);
	GDREGISTER_VIRTUAL_CLASS(VisualShaderNodeResizableBase);
	GDREGISTER_VIRTUAL_CLASS(VisualShaderNodeGroupBase);
	GDREGISTER_VIRTUAL_CLASS(VisualShaderNodeConstant);
	GDREGISTER_CLASS(VisualShaderNodeComment);
	GDREGISTER_CLASS(VisualShaderNodeFloatConstant);
	GDREGISTER_CLASS(VisualShaderNodeIntConstant);
	GDREGISTER_CLASS(VisualShaderNodeBooleanConstant);
	GDREGISTER_CLASS(VisualShaderNodeColorConstant);
	GDREGISTER_CLASS(VisualShaderNodeVec3Constant);
	GDREGISTER_CLASS(VisualShaderNodeTransformConstant);
	GDREGISTER_CLASS(VisualShaderNodeFloatOp);
	GDREGISTER_CLASS(VisualShaderNodeIntOp);
	GDREGISTER_CLASS(VisualShaderNodeVectorOp);
	GDREGISTER_CLASS(VisualShaderNodeColorOp);
	GDREGISTER_CLASS(VisualShaderNodeTransformOp);
	GDREGISTER_CLASS(VisualShaderNodeTransformVecMult);
	GDREGISTER_CLASS(VisualShaderNodeFloatFunc);
	GDREGISTER_CLASS(VisualShaderNodeIntFunc);
	GDREGISTER_CLASS(VisualShaderNodeVectorFunc);
	GDREGISTER_CLASS(VisualShaderNodeColorFunc);
	GDREGISTER_CLASS(VisualShaderNodeTransformFunc);
	GDREGISTER_CLASS(VisualShaderNodeUVFunc);
	GDREGISTER_CLASS(VisualShaderNodeDotProduct);
	GDREGISTER_CLASS(VisualShaderNodeVectorLen);
	GDREGISTER_CLASS(VisualShaderNodeDeterminant);
	GDREGISTER_CLASS(VisualShaderNodeScalarDerivativeFunc);
	GDREGISTER_CLASS(VisualShaderNodeVectorDerivativeFunc);
	GDREGISTER_CLASS(VisualShaderNodeClamp);
	GDREGISTER_CLASS(VisualShaderNodeFaceForward);
	GDREGISTER_CLASS(VisualShaderNodeOuterProduct);
	GDREGISTER_CLASS(VisualShaderNodeSmoothStep);
	GDREGISTER_CLASS(VisualShaderNodeStep);
	GDREGISTER_CLASS(VisualShaderNodeVectorDistance);
	GDREGISTER_CLASS(VisualShaderNodeVectorRefract);
	GDREGISTER_CLASS(VisualShaderNodeMix);
	GDREGISTER_CLASS(VisualShaderNodeVectorCompose);
	GDREGISTER_CLASS(VisualShaderNodeTransformCompose);
	GDREGISTER_CLASS(VisualShaderNodeVectorDecompose);
	GDREGISTER_CLASS(VisualShaderNodeTransformDecompose);
	GDREGISTER_CLASS(VisualShaderNodeTexture);
	GDREGISTER_CLASS(VisualShaderNodeCurveTexture);
	GDREGISTER_CLASS(VisualShaderNodeCurveXYZTexture);
	GDREGISTER_VIRTUAL_CLASS(VisualShaderNodeSample3D);
	GDREGISTER_CLASS(VisualShaderNodeTexture2DArray);
	GDREGISTER_CLASS(VisualShaderNodeTexture3D);
	GDREGISTER_CLASS(VisualShaderNodeCubemap);
	GDREGISTER_VIRTUAL_CLASS(VisualShaderNodeUniform);
	GDREGISTER_CLASS(VisualShaderNodeUniformRef);
	GDREGISTER_CLASS(VisualShaderNodeFloatUniform);
	GDREGISTER_CLASS(VisualShaderNodeIntUniform);
	GDREGISTER_CLASS(VisualShaderNodeBooleanUniform);
	GDREGISTER_CLASS(VisualShaderNodeColorUniform);
	GDREGISTER_CLASS(VisualShaderNodeVec3Uniform);
	GDREGISTER_CLASS(VisualShaderNodeTransformUniform);
	GDREGISTER_CLASS(VisualShaderNodeTextureUniform);
	GDREGISTER_CLASS(VisualShaderNodeTextureUniformTriplanar);
	GDREGISTER_CLASS(VisualShaderNodeTexture2DArrayUniform);
	GDREGISTER_CLASS(VisualShaderNodeTexture3DUniform);
	GDREGISTER_CLASS(VisualShaderNodeCubemapUniform);
	GDREGISTER_CLASS(VisualShaderNodeIf);
	GDREGISTER_CLASS(VisualShaderNodeSwitch);
	GDREGISTER_CLASS(VisualShaderNodeFresnel);
	GDREGISTER_CLASS(VisualShaderNodeExpression);
	GDREGISTER_CLASS(VisualShaderNodeGlobalExpression);
	GDREGISTER_CLASS(VisualShaderNodeIs);
	GDREGISTER_CLASS(VisualShaderNodeCompare);
	GDREGISTER_CLASS(VisualShaderNodeMultiplyAdd);
	GDREGISTER_CLASS(VisualShaderNodeBillboard);

	GDREGISTER_CLASS(VisualShaderNodeSDFToScreenUV);
	GDREGISTER_CLASS(VisualShaderNodeScreenUVToSDF);
	GDREGISTER_CLASS(VisualShaderNodeTextureSDF);
	GDREGISTER_CLASS(VisualShaderNodeTextureSDFNormal);
	GDREGISTER_CLASS(VisualShaderNodeSDFRaymarch);

	GDREGISTER_CLASS(VisualShaderNodeParticleOutput);
	GDREGISTER_VIRTUAL_CLASS(VisualShaderNodeParticleEmitter);
	GDREGISTER_CLASS(VisualShaderNodeParticleSphereEmitter);
	GDREGISTER_CLASS(VisualShaderNodeParticleBoxEmitter);
	GDREGISTER_CLASS(VisualShaderNodeParticleRingEmitter);
	GDREGISTER_CLASS(VisualShaderNodeParticleMeshEmitter);
	GDREGISTER_CLASS(VisualShaderNodeParticleMultiplyByAxisAngle);
	GDREGISTER_CLASS(VisualShaderNodeParticleConeVelocity);
	GDREGISTER_CLASS(VisualShaderNodeParticleRandomness);
	GDREGISTER_CLASS(VisualShaderNodeParticleAccelerator);
	GDREGISTER_CLASS(VisualShaderNodeParticleEmit);

	GDREGISTER_CLASS(ShaderMaterial);
	GDREGISTER_VIRTUAL_CLASS(CanvasItem);
	GDREGISTER_CLASS(CanvasTexture);
	GDREGISTER_CLASS(CanvasItemMaterial);
	SceneTree::add_idle_callback(CanvasItemMaterial::flush_changes);
	CanvasItemMaterial::init_shaders();

	/* REGISTER 2D */

	GDREGISTER_CLASS(Node2D);
	GDREGISTER_CLASS(CanvasGroup);
	GDREGISTER_CLASS(CPUParticles2D);
	GDREGISTER_CLASS(GPUParticles2D);
	GDREGISTER_CLASS(Sprite2D);
	GDREGISTER_CLASS(SpriteFrames);
	GDREGISTER_CLASS(AnimatedSprite2D);
	GDREGISTER_CLASS(Position2D);
	GDREGISTER_CLASS(Line2D);
	GDREGISTER_CLASS(MeshInstance2D);
	GDREGISTER_CLASS(MultiMeshInstance2D);
	GDREGISTER_VIRTUAL_CLASS(CollisionObject2D);
	GDREGISTER_VIRTUAL_CLASS(PhysicsBody2D);
	GDREGISTER_CLASS(StaticBody2D);
	GDREGISTER_CLASS(AnimatableBody2D);
	GDREGISTER_CLASS(RigidDynamicBody2D);
	GDREGISTER_CLASS(CharacterBody2D);
	GDREGISTER_CLASS(KinematicCollision2D);
	GDREGISTER_CLASS(Area2D);
	GDREGISTER_CLASS(CollisionShape2D);
	GDREGISTER_CLASS(CollisionPolygon2D);
	GDREGISTER_CLASS(RayCast2D);
	GDREGISTER_CLASS(ShapeCast2D);
	GDREGISTER_CLASS(VisibleOnScreenNotifier2D);
	GDREGISTER_CLASS(VisibleOnScreenEnabler2D);
	GDREGISTER_CLASS(Polygon2D);
	GDREGISTER_CLASS(Skeleton2D);
	GDREGISTER_CLASS(Bone2D);
	GDREGISTER_VIRTUAL_CLASS(Light2D);
	GDREGISTER_CLASS(PointLight2D);
	GDREGISTER_CLASS(DirectionalLight2D);
	GDREGISTER_CLASS(LightOccluder2D);
	GDREGISTER_CLASS(OccluderPolygon2D);
	GDREGISTER_CLASS(BackBufferCopy);

	OS::get_singleton()->yield(); // may take time to init

	GDREGISTER_CLASS(Camera2D);
	GDREGISTER_CLASS(AudioListener2D);
	GDREGISTER_VIRTUAL_CLASS(Joint2D);
	GDREGISTER_CLASS(PinJoint2D);
	GDREGISTER_CLASS(GrooveJoint2D);
	GDREGISTER_CLASS(DampedSpringJoint2D);
	GDREGISTER_CLASS(TileSet);
	GDREGISTER_VIRTUAL_CLASS(TileSetSource);
	GDREGISTER_CLASS(TileSetAtlasSource);
	GDREGISTER_CLASS(TileSetScenesCollectionSource);
	GDREGISTER_CLASS(TileMapPattern);
	GDREGISTER_CLASS(TileData);
	GDREGISTER_CLASS(TileMap);
	GDREGISTER_CLASS(ParallaxBackground);
	GDREGISTER_CLASS(ParallaxLayer);
	GDREGISTER_CLASS(TouchScreenButton);
	GDREGISTER_CLASS(RemoteTransform2D);

	GDREGISTER_CLASS(SkeletonModificationStack2D);
	GDREGISTER_CLASS(SkeletonModification2D);
	GDREGISTER_CLASS(SkeletonModification2DLookAt);
	GDREGISTER_CLASS(SkeletonModification2DCCDIK);
	GDREGISTER_CLASS(SkeletonModification2DFABRIK);
	GDREGISTER_CLASS(SkeletonModification2DJiggle);
	GDREGISTER_CLASS(SkeletonModification2DTwoBoneIK);
	GDREGISTER_CLASS(SkeletonModification2DStackHolder);

	GDREGISTER_CLASS(PhysicalBone2D);
	GDREGISTER_CLASS(SkeletonModification2DPhysicalBones);

	OS::get_singleton()->yield(); // may take time to init

	/* REGISTER RESOURCES */

	GDREGISTER_VIRTUAL_CLASS(Shader);
	GDREGISTER_CLASS(ParticlesMaterial);
	SceneTree::add_idle_callback(ParticlesMaterial::flush_changes);
	ParticlesMaterial::init_shaders();

	GDREGISTER_CLASS(ProceduralSkyMaterial);
	GDREGISTER_CLASS(PanoramaSkyMaterial);
	GDREGISTER_CLASS(PhysicalSkyMaterial);

	GDREGISTER_VIRTUAL_CLASS(Mesh);
	GDREGISTER_CLASS(ArrayMesh);
	GDREGISTER_CLASS(ImmediateMesh);
	GDREGISTER_CLASS(MultiMesh);
	GDREGISTER_CLASS(SurfaceTool);
	GDREGISTER_CLASS(MeshDataTool);

#ifndef _3D_DISABLED
	GDREGISTER_VIRTUAL_CLASS(PrimitiveMesh);
	GDREGISTER_CLASS(BoxMesh);
	GDREGISTER_CLASS(CapsuleMesh);
	GDREGISTER_CLASS(CylinderMesh);
	GDREGISTER_CLASS(PlaneMesh);
	GDREGISTER_CLASS(PrismMesh);
	GDREGISTER_CLASS(QuadMesh);
	GDREGISTER_CLASS(SphereMesh);
	GDREGISTER_CLASS(TubeTrailMesh);
	GDREGISTER_CLASS(RibbonTrailMesh);
	GDREGISTER_CLASS(PointMesh);
	GDREGISTER_VIRTUAL_CLASS(Material);
	GDREGISTER_VIRTUAL_CLASS(BaseMaterial3D);
	GDREGISTER_CLASS(StandardMaterial3D);
	GDREGISTER_CLASS(ORMMaterial3D);
	SceneTree::add_idle_callback(BaseMaterial3D::flush_changes);
	BaseMaterial3D::init_shaders();

	GDREGISTER_CLASS(MeshLibrary);

	OS::get_singleton()->yield(); // may take time to init

	GDREGISTER_VIRTUAL_CLASS(Shape3D);
	GDREGISTER_CLASS(SeparationRayShape3D);
	GDREGISTER_CLASS(SphereShape3D);
	GDREGISTER_CLASS(BoxShape3D);
	GDREGISTER_CLASS(CapsuleShape3D);
	GDREGISTER_CLASS(CylinderShape3D);
	GDREGISTER_CLASS(HeightMapShape3D);
	GDREGISTER_CLASS(WorldBoundaryShape3D);
	GDREGISTER_CLASS(ConvexPolygonShape3D);
	GDREGISTER_CLASS(ConcavePolygonShape3D);

	ClassDB::register_class<SkeletonModificationStack3D>();
	ClassDB::register_class<SkeletonModification3D>();
	ClassDB::register_class<SkeletonModification3DLookAt>();
	ClassDB::register_class<SkeletonModification3DCCDIK>();
	ClassDB::register_class<SkeletonModification3DFABRIK>();
	ClassDB::register_class<SkeletonModification3DJiggle>();
	ClassDB::register_class<SkeletonModification3DTwoBoneIK>();
	ClassDB::register_class<SkeletonModification3DStackHolder>();

	OS::get_singleton()->yield(); // may take time to init

	GDREGISTER_CLASS(VelocityTracker3D);
#endif

	GDREGISTER_CLASS(PhysicsMaterial);
	GDREGISTER_CLASS(World3D);
	GDREGISTER_CLASS(Environment);
	GDREGISTER_CLASS(CameraEffects);
	GDREGISTER_CLASS(World2D);
	GDREGISTER_VIRTUAL_CLASS(Texture);
	GDREGISTER_VIRTUAL_CLASS(Texture2D);
	GDREGISTER_CLASS(Sky);
	GDREGISTER_CLASS(StreamTexture2D);
	GDREGISTER_CLASS(ImageTexture);
	GDREGISTER_CLASS(AtlasTexture);
	GDREGISTER_CLASS(MeshTexture);
	GDREGISTER_CLASS(CurveTexture);
	GDREGISTER_CLASS(CurveXYZTexture);
	GDREGISTER_CLASS(GradientTexture1D);
	GDREGISTER_CLASS(GradientTexture2D);
	GDREGISTER_CLASS(ProxyTexture);
	GDREGISTER_CLASS(AnimatedTexture);
	GDREGISTER_CLASS(CameraTexture);
	GDREGISTER_VIRTUAL_CLASS(TextureLayered);
	GDREGISTER_VIRTUAL_CLASS(ImageTextureLayered);
	GDREGISTER_VIRTUAL_CLASS(Texture3D);
	GDREGISTER_CLASS(ImageTexture3D);
	GDREGISTER_CLASS(StreamTexture3D);
	GDREGISTER_CLASS(Cubemap);
	GDREGISTER_CLASS(CubemapArray);
	GDREGISTER_CLASS(Texture2DArray);
	GDREGISTER_VIRTUAL_CLASS(StreamTextureLayered);
	GDREGISTER_CLASS(StreamCubemap);
	GDREGISTER_CLASS(StreamCubemapArray);
	GDREGISTER_CLASS(StreamTexture2DArray);

	GDREGISTER_CLASS(Animation);
	GDREGISTER_CLASS(FontData);
	GDREGISTER_CLASS(Font);
	GDREGISTER_CLASS(Curve);

	GDREGISTER_CLASS(TextLine);
	GDREGISTER_CLASS(TextParagraph);

	GDREGISTER_VIRTUAL_CLASS(StyleBox);
	GDREGISTER_CLASS(StyleBoxEmpty);
	GDREGISTER_CLASS(StyleBoxTexture);
	GDREGISTER_CLASS(StyleBoxFlat);
	GDREGISTER_CLASS(StyleBoxLine);
	GDREGISTER_CLASS(Theme);

	GDREGISTER_CLASS(PolygonPathFinder);
	GDREGISTER_CLASS(BitMap);
	GDREGISTER_CLASS(Gradient);

	OS::get_singleton()->yield(); // may take time to init

	GDREGISTER_CLASS(AudioStreamPlayer);
	GDREGISTER_CLASS(AudioStreamPlayer2D);
#ifndef _3D_DISABLED
	GDREGISTER_CLASS(AudioStreamPlayer3D);
#endif
	GDREGISTER_VIRTUAL_CLASS(VideoStream);
	GDREGISTER_CLASS(AudioStreamSample);

	OS::get_singleton()->yield(); // may take time to init

	GDREGISTER_VIRTUAL_CLASS(Shape2D);
	GDREGISTER_CLASS(WorldBoundaryShape2D);
	GDREGISTER_CLASS(SegmentShape2D);
	GDREGISTER_CLASS(SeparationRayShape2D);
	GDREGISTER_CLASS(CircleShape2D);
	GDREGISTER_CLASS(RectangleShape2D);
	GDREGISTER_CLASS(CapsuleShape2D);
	GDREGISTER_CLASS(ConvexPolygonShape2D);
	GDREGISTER_CLASS(ConcavePolygonShape2D);
	GDREGISTER_CLASS(Curve2D);
	GDREGISTER_CLASS(Path2D);
	GDREGISTER_CLASS(PathFollow2D);

	GDREGISTER_CLASS(NavigationMesh);
	GDREGISTER_CLASS(NavigationPolygon);
	GDREGISTER_CLASS(NavigationRegion2D);
	GDREGISTER_CLASS(NavigationAgent2D);
	GDREGISTER_CLASS(NavigationObstacle2D);

	OS::get_singleton()->yield(); // may take time to init

	GDREGISTER_VIRTUAL_CLASS(SceneState);
	GDREGISTER_CLASS(PackedScene);

	GDREGISTER_CLASS(SceneTree);
	GDREGISTER_VIRTUAL_CLASS(SceneTreeTimer); // sorry, you can't create it

#ifndef DISABLE_DEPRECATED
	// Dropped in 4.0, near approximation.
	ClassDB::add_compatibility_class("AnimationTreePlayer", "AnimationTree");
	ClassDB::add_compatibility_class("BitmapFont", "Font");
	ClassDB::add_compatibility_class("DynamicFont", "Font");
	ClassDB::add_compatibility_class("DynamicFontData", "FontData");
	ClassDB::add_compatibility_class("ToolButton", "Button");
	ClassDB::add_compatibility_class("Navigation3D", "Node3D");
	ClassDB::add_compatibility_class("Navigation2D", "Node2D");
	ClassDB::add_compatibility_class("YSort", "Node2D");
	ClassDB::add_compatibility_class("GIProbe", "VoxelGI");
	ClassDB::add_compatibility_class("GIProbeData", "VoxelGIData");
	ClassDB::add_compatibility_class("BakedLightmap", "LightmapGI");
	ClassDB::add_compatibility_class("BakedLightmapData", "LightmapGIData");

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
	ClassDB::add_compatibility_class("GradientTexture", "GradientTexture1D");
	ClassDB::add_compatibility_class("HeightMapShape", "HeightMapShape3D");
	ClassDB::add_compatibility_class("HingeJoint", "HingeJoint3D");
	ClassDB::add_compatibility_class("Joint", "Joint3D");
	ClassDB::add_compatibility_class("KinematicBody", "CharacterBody3D");
	ClassDB::add_compatibility_class("KinematicBody2D", "CharacterBody2D");
	ClassDB::add_compatibility_class("KinematicCollision", "KinematicCollision3D");
	ClassDB::add_compatibility_class("Light", "Light3D");
	ClassDB::add_compatibility_class("LineShape2D", "WorldBoundaryShape2D");
	ClassDB::add_compatibility_class("Listener", "AudioListener3D");
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
	ClassDB::add_compatibility_class("Physics2DDirectBodyState", "PhysicsDirectBodyState2D");
	ClassDB::add_compatibility_class("Physics2DDirectSpaceState", "PhysicsDirectSpaceState2D");
	ClassDB::add_compatibility_class("Physics2DServer", "PhysicsServer2D");
	ClassDB::add_compatibility_class("Physics2DShapeQueryParameters", "PhysicsShapeQueryParameters2D");
	ClassDB::add_compatibility_class("Physics2DTestMotionResult", "PhysicsTestMotionResult2D");
	ClassDB::add_compatibility_class("PhysicsBody", "PhysicsBody3D");
	ClassDB::add_compatibility_class("PhysicsDirectBodyState", "PhysicsDirectBodyState3D");
	ClassDB::add_compatibility_class("PhysicsDirectSpaceState", "PhysicsDirectSpaceState3D");
	ClassDB::add_compatibility_class("PhysicsServer", "PhysicsServer3D");
	ClassDB::add_compatibility_class("PhysicsShapeQueryParameters", "PhysicsShapeQueryParameters3D");
	ClassDB::add_compatibility_class("PinJoint", "PinJoint3D");
	ClassDB::add_compatibility_class("PlaneShape", "WorldBoundaryShape3D");
	ClassDB::add_compatibility_class("ProceduralSky", "Sky");
	ClassDB::add_compatibility_class("ProximityGroup", "ProximityGroup3D");
	ClassDB::add_compatibility_class("RayCast", "RayCast3D");
	ClassDB::add_compatibility_class("RayShape", "SeparationRayShape3D");
	ClassDB::add_compatibility_class("RayShape2D", "SeparationRayShape2D");
	ClassDB::add_compatibility_class("RemoteTransform", "RemoteTransform3D");
	ClassDB::add_compatibility_class("RigidBody", "RigidDynamicBody3D");
	ClassDB::add_compatibility_class("RigidBody2D", "RigidDynamicBody2D");
	ClassDB::add_compatibility_class("Shape", "Shape3D");
	ClassDB::add_compatibility_class("ShortCut", "Shortcut");
	ClassDB::add_compatibility_class("Skeleton", "Skeleton3D");
	ClassDB::add_compatibility_class("SkeletonIK", "SkeletonIK3D");
	ClassDB::add_compatibility_class("SliderJoint", "SliderJoint3D");
	ClassDB::add_compatibility_class("SoftBody", "SoftDynamicBody3D");
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
	ClassDB::add_compatibility_class("VisibilityEnabler", "VisibleOnScreenEnabler3D");
	ClassDB::add_compatibility_class("VisibilityNotifier", "VisibleOnScreenNotifier3D");
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
	ClassDB::add_compatibility_class("VisualShaderNodeScalarTransformMult", "VisualShaderNodeTransformOp");
	ClassDB::add_compatibility_class("World", "World3D");
	ClassDB::add_compatibility_class("StreamTexture", "StreamTexture2D");
	ClassDB::add_compatibility_class("Light2D", "PointLight2D");
	ClassDB::add_compatibility_class("VisibilityNotifier2D", "VisibleOnScreenNotifier2D");
	ClassDB::add_compatibility_class("VisibilityNotifier3D", "VisibleOnScreenNotifier3D");

#endif /* DISABLE_DEPRECATED */

	OS::get_singleton()->yield(); // may take time to init

	for (int i = 0; i < 20; i++) {
		GLOBAL_DEF_BASIC(vformat("layer_names/2d_render/layer_%d", i + 1), "");
		GLOBAL_DEF_BASIC(vformat("layer_names/3d_render/layer_%d", i + 1), "");
	}

	for (int i = 0; i < 32; i++) {
		GLOBAL_DEF_BASIC(vformat("layer_names/2d_physics/layer_%d", i + 1), "");
		GLOBAL_DEF_BASIC(vformat("layer_names/2d_navigation/layer_%d", i + 1), "");
		GLOBAL_DEF_BASIC(vformat("layer_names/3d_physics/layer_%d", i + 1), "");
		GLOBAL_DEF_BASIC(vformat("layer_names/3d_navigation/layer_%d", i + 1), "");
	}

	if (RenderingServer::get_singleton()) {
		ColorPicker::init_shaders(); // RenderingServer needs to exist for this to succeed.
	}

	SceneDebugger::initialize();

	NativeExtensionManager::get_singleton()->initialize_extensions(NativeExtension::INITIALIZATION_LEVEL_SCENE);
}

void initialize_theme() {
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
}

void unregister_scene_types() {
	NativeExtensionManager::get_singleton()->deinitialize_extensions(NativeExtension::INITIALIZATION_LEVEL_SCENE);

	SceneDebugger::deinitialize();
	clear_default_theme();

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

	// StandardMaterial3D is not initialised when 3D is disabled, so it shouldn't be cleaned up either
#ifndef _3D_DISABLED
	BaseMaterial3D::finish_shaders();
#endif // _3D_DISABLED

	PhysicalSkyMaterial::cleanup_shader();
	PanoramaSkyMaterial::cleanup_shader();
	ProceduralSkyMaterial::cleanup_shader();

	ParticlesMaterial::finish_shaders();
	CanvasItemMaterial::finish_shaders();
	ColorPicker::finish_shaders();
	SceneStringNames::free();
}
