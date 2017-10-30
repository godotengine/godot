/*************************************************************************/
/*  register_scene_types.cpp                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "core/class_db.h"
#include "core/os/os.h"
#include "core/project_settings.h"
#include "scene/2d/animated_sprite.h"
#include "scene/2d/area_2d.h"
#include "scene/2d/audio_stream_player_2d.h"
#include "scene/2d/back_buffer_copy.h"
#include "scene/2d/camera_2d.h"
#include "scene/2d/canvas_item.h"
#include "scene/2d/canvas_modulate.h"
#include "scene/2d/collision_polygon_2d.h"
#include "scene/2d/collision_shape_2d.h"
#include "scene/2d/joints_2d.h"
#include "scene/2d/light_2d.h"
#include "scene/2d/light_occluder_2d.h"
#include "scene/2d/line_2d.h"
#include "scene/2d/navigation2d.h"
#include "scene/2d/parallax_background.h"
#include "scene/2d/parallax_layer.h"
#include "scene/2d/particles_2d.h"
#include "scene/2d/path_2d.h"
#include "scene/2d/physics_body_2d.h"
#include "scene/2d/polygon_2d.h"
#include "scene/2d/position_2d.h"
#include "scene/2d/ray_cast_2d.h"
#include "scene/2d/remote_transform_2d.h"
#include "scene/2d/screen_button.h"
#include "scene/2d/sprite.h"
#include "scene/2d/tile_map.h"
#include "scene/2d/visibility_notifier_2d.h"
#include "scene/2d/y_sort.h"
#include "scene/animation/animation_player.h"
#include "scene/animation/animation_tree_player.h"
#include "scene/animation/tween.h"
#include "scene/audio/audio_player.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/center_container.h"
#include "scene/gui/check_box.h"
#include "scene/gui/check_button.h"
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
#include "scene/gui/option_button.h"
#include "scene/gui/panel.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/nine_patch_rect.h"
#include "scene/gui/popup_menu.h"
#include "scene/gui/progress_bar.h"
#include "scene/gui/reference_rect.h"
#include "scene/gui/rich_text_label.h"
#include "scene/gui/scroll_bar.h"
#include "scene/gui/scroll_container.h"
#include "scene/gui/separator.h"
#include "scene/gui/slider.h"
#include "scene/gui/spin_box.h"
#include "scene/gui/split_container.h"
#include "scene/gui/tab_container.h"
#include "scene/gui/tabs.h"
#include "scene/gui/text_edit.h"
#include "scene/gui/texture_button.h"
#include "scene/gui/texture_progress.h"
#include "scene/gui/texture_rect.h"
#include "scene/gui/tool_button.h"
#include "scene/gui/tree.h"
#include "scene/gui/video_player.h"
#include "scene/gui/viewport_container.h"
#include "scene/main/canvas_layer.h"
#include "scene/main/http_request.h"
#include "scene/main/instance_placeholder.h"
#include "scene/main/resource_preloader.h"
#include "scene/main/scene_tree.h"
#include "scene/main/timer.h"
#include "scene/main/viewport.h"
#include "scene/resources/audio_stream_sample.h"
#include "scene/resources/bit_mask.h"
#include "scene/resources/box_shape.h"
#include "scene/resources/capsule_shape.h"
#include "scene/resources/capsule_shape_2d.h"
#include "scene/resources/circle_shape_2d.h"
#include "scene/resources/color_ramp.h"
#include "scene/resources/concave_polygon_shape.h"
#include "scene/resources/concave_polygon_shape_2d.h"
#include "scene/resources/convex_polygon_shape.h"
#include "scene/resources/convex_polygon_shape_2d.h"
#include "scene/resources/default_theme/default_theme.h"
#include "scene/resources/dynamic_font.h"
#include "scene/resources/dynamic_font_stb.h"
#include "scene/resources/material.h"
#include "scene/resources/mesh.h"
#include "scene/resources/mesh_data_tool.h"
#include "scene/resources/mesh_library.h"
#include "scene/resources/packed_scene.h"
#include "scene/resources/plane_shape.h"
#include "scene/resources/polygon_path_finder.h"
#include "scene/resources/primitive_meshes.h"
#include "scene/resources/ray_shape.h"
#include "scene/resources/rectangle_shape_2d.h"
#include "scene/resources/scene_format_text.h"
#include "scene/resources/segment_shape_2d.h"
#include "scene/resources/shader_graph.h"
#include "scene/resources/shape_line_2d.h"
#include "scene/resources/sky_box.h"
#include "scene/resources/sphere_shape.h"
#include "scene/resources/surface_tool.h"
#include "scene/resources/texture.h"
#include "scene/resources/tile_set.h"
#include "scene/resources/video_stream.h"
#include "scene/resources/world.h"
#include "scene/resources/world_2d.h"
#include "scene/scene_string_names.h"

#include "scene/3d/particles.h"
#include "scene/3d/scenario_fx.h"
#include "scene/3d/spatial.h"

#ifndef _3D_DISABLED
#include "scene/3d/area.h"
#include "scene/3d/arvr_nodes.h"
#include "scene/3d/audio_stream_player_3d.h"
#include "scene/3d/bone_attachment.h"
#include "scene/3d/camera.h"
#include "scene/3d/collision_polygon.h"
#include "scene/3d/collision_shape.h"
#include "scene/3d/gi_probe.h"
#include "scene/3d/immediate_geometry.h"
#include "scene/3d/interpolated_camera.h"
#include "scene/3d/light.h"
#include "scene/3d/listener.h"
#include "scene/3d/mesh_instance.h"
#include "scene/3d/multimesh_instance.h"
#include "scene/3d/navigation.h"
#include "scene/3d/navigation_mesh.h"
#include "scene/3d/path.h"
#include "scene/3d/physics_body.h"
#include "scene/3d/physics_joint.h"
#include "scene/3d/portal.h"
#include "scene/3d/position_3d.h"
#include "scene/3d/proximity_group.h"
#include "scene/3d/ray_cast.h"
#include "scene/3d/reflection_probe.h"
#include "scene/3d/remote_transform.h"
#include "scene/3d/room_instance.h"
#include "scene/3d/skeleton.h"
#include "scene/3d/sprite_3d.h"
#include "scene/3d/vehicle_body.h"
#include "scene/3d/visibility_notifier.h"
#include "scene/resources/environment.h"
#endif

static ResourceFormatLoaderTheme *resource_loader_theme = NULL;

static ResourceFormatSaverText *resource_saver_text = NULL;
static ResourceFormatLoaderText *resource_loader_text = NULL;

static ResourceFormatLoaderDynamicFont *resource_loader_dynamic_font = NULL;

static ResourceFormatLoaderStreamTexture *resource_loader_stream_texture = NULL;

void register_scene_types() {

	SceneStringNames::create();

	OS::get_singleton()->yield(); //may take time to init

	Node::init_node_hrcr();

	resource_loader_dynamic_font = memnew(ResourceFormatLoaderDynamicFont);
	ResourceLoader::add_resource_format_loader(resource_loader_dynamic_font);

	resource_loader_stream_texture = memnew(ResourceFormatLoaderStreamTexture);
	ResourceLoader::add_resource_format_loader(resource_loader_stream_texture);

	resource_loader_theme = memnew(ResourceFormatLoaderTheme);
	ResourceLoader::add_resource_format_loader(resource_loader_theme);

	bool default_theme_hidpi = GLOBAL_DEF("gui/theme/use_hidpi", false);
	ProjectSettings::get_singleton()->set_custom_property_info("gui/theme/use_hidpi", PropertyInfo(Variant::BOOL, "gui/theme/use_hidpi", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED));
	String theme_path = GLOBAL_DEF("gui/theme/custom", "");
	ProjectSettings::get_singleton()->set_custom_property_info("gui/theme/custom", PropertyInfo(Variant::STRING, "gui/theme/custom", PROPERTY_HINT_FILE, "*.tres,*.res", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED));
	String font_path = GLOBAL_DEF("gui/theme/custom_font", "");
	ProjectSettings::get_singleton()->set_custom_property_info("gui/theme/custom_font", PropertyInfo(Variant::STRING, "gui/theme/custom_font", PROPERTY_HINT_FILE, "*.tres,*.res,*.font", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED));

	bool has_theme = false;
	if (theme_path != String()) {
		Ref<Theme> theme = ResourceLoader::load(theme_path);
		if (theme.is_valid()) {
			Theme::set_default(theme);
			has_theme = true;
		} else {
			ERR_PRINTS("Error loading custom theme '" + theme_path + "'");
		}
	}

	if (!has_theme) {
		Ref<Font> font;
		if (font_path != String()) {
			font = ResourceLoader::load(font_path);
		}
		make_default_theme(default_theme_hidpi, font);
	}

	OS::get_singleton()->yield(); //may take time to init

	ClassDB::register_class<Object>();

	ClassDB::register_class<Node>();
	ClassDB::register_virtual_class<InstancePlaceholder>();

	ClassDB::register_class<Viewport>();
	ClassDB::register_class<ViewportTexture>();
	ClassDB::register_class<HTTPRequest>();
	ClassDB::register_class<Timer>();
	ClassDB::register_class<CanvasLayer>();
	ClassDB::register_class<CanvasModulate>();
	ClassDB::register_class<ResourcePreloader>();

	/* REGISTER GUI */
	ClassDB::register_class<ButtonGroup>();
	ClassDB::register_virtual_class<BaseButton>();

	OS::get_singleton()->yield(); //may take time to init

	ClassDB::register_class<ShortCut>();
	ClassDB::register_class<Control>();
	ClassDB::register_class<Button>();
	ClassDB::register_class<Label>();
	ClassDB::register_class<ScrollBar>();
	ClassDB::register_class<HScrollBar>();
	ClassDB::register_class<VScrollBar>();
	ClassDB::register_class<ProgressBar>();
	ClassDB::register_class<Slider>();
	ClassDB::register_class<HSlider>();
	ClassDB::register_class<VSlider>();
	ClassDB::register_class<Popup>();
	ClassDB::register_class<PopupPanel>();
	ClassDB::register_class<MenuButton>();
	ClassDB::register_class<CheckBox>();
	ClassDB::register_class<CheckButton>();
	ClassDB::register_class<ToolButton>();
	ClassDB::register_class<LinkButton>();
	ClassDB::register_class<Panel>();
	ClassDB::register_class<Range>();

	OS::get_singleton()->yield(); //may take time to init

	ClassDB::register_class<TextureRect>();
	ClassDB::register_class<ColorRect>();
	ClassDB::register_class<NinePatchRect>();
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
	ClassDB::register_virtual_class<SplitContainer>();
	ClassDB::register_class<HSplitContainer>();
	ClassDB::register_class<VSplitContainer>();
	ClassDB::register_class<GraphNode>();
	ClassDB::register_class<GraphEdit>();

	OS::get_singleton()->yield(); //may take time to init

	ClassDB::register_class<TextureProgress>();
	ClassDB::register_class<ItemList>();

#ifndef ADVANCED_GUI_DISABLED

	ClassDB::register_class<FileDialog>();
	ClassDB::register_class<LineEdit>();
	ClassDB::register_class<PopupMenu>();
	ClassDB::register_class<Tree>();

	ClassDB::register_class<TextEdit>();

	ClassDB::register_virtual_class<TreeItem>();
	ClassDB::register_class<OptionButton>();
	ClassDB::register_class<SpinBox>();
	ClassDB::register_class<ReferenceRect>();
	ClassDB::register_class<ColorPicker>();
	ClassDB::register_class<ColorPickerButton>();
	ClassDB::register_class<RichTextLabel>();
	ClassDB::register_class<PopupDialog>();
	ClassDB::register_class<WindowDialog>();
	ClassDB::register_class<AcceptDialog>();
	ClassDB::register_class<ConfirmationDialog>();
	ClassDB::register_class<VideoPlayer>();
	ClassDB::register_class<MarginContainer>();
	ClassDB::register_class<ViewportContainer>();

	OS::get_singleton()->yield(); //may take time to init

#endif

	/* REGISTER 3D */

	ClassDB::register_class<Spatial>();
	ClassDB::register_virtual_class<SpatialGizmo>();
	ClassDB::register_class<Skeleton>();
	ClassDB::register_class<AnimationPlayer>();
	ClassDB::register_class<Tween>();

	OS::get_singleton()->yield(); //may take time to init

#ifndef _3D_DISABLED
	ClassDB::register_class<BoneAttachment>();
	ClassDB::register_virtual_class<VisualInstance>();
	ClassDB::register_virtual_class<GeometryInstance>();
	ClassDB::register_class<Camera>();
	ClassDB::register_class<Listener>();
	ClassDB::register_class<ARVRCamera>();
	ClassDB::register_class<ARVRController>();
	ClassDB::register_class<ARVRAnchor>();
	ClassDB::register_class<ARVROrigin>();
	ClassDB::register_class<InterpolatedCamera>();
	ClassDB::register_class<MeshInstance>();
	ClassDB::register_class<ImmediateGeometry>();
	ClassDB::register_virtual_class<SpriteBase3D>();
	ClassDB::register_class<Sprite3D>();
	ClassDB::register_class<AnimatedSprite3D>();
	ClassDB::register_virtual_class<Light>();
	ClassDB::register_class<DirectionalLight>();
	ClassDB::register_class<OmniLight>();
	ClassDB::register_class<SpotLight>();
	ClassDB::register_class<ReflectionProbe>();
	ClassDB::register_class<GIProbe>();
	ClassDB::register_class<GIProbeData>();
	ClassDB::register_class<AnimationTreePlayer>();
	ClassDB::register_class<Particles>();
	ClassDB::register_class<Position3D>();
	ClassDB::register_class<NavigationMeshInstance>();
	ClassDB::register_class<NavigationMesh>();
	ClassDB::register_class<Navigation>();

	OS::get_singleton()->yield(); //may take time to init

	ClassDB::register_virtual_class<CollisionObject>();
	ClassDB::register_virtual_class<PhysicsBody>();
	ClassDB::register_class<StaticBody>();
	ClassDB::register_class<RigidBody>();
	ClassDB::register_class<KinematicCollision>();
	ClassDB::register_class<KinematicBody>();

	ClassDB::register_class<VehicleBody>();
	ClassDB::register_class<VehicleWheel>();
	ClassDB::register_class<Area>();
	ClassDB::register_class<ProximityGroup>();
	ClassDB::register_class<CollisionShape>();
	ClassDB::register_class<CollisionPolygon>();
	ClassDB::register_class<RayCast>();
	ClassDB::register_class<MultiMeshInstance>();

	ClassDB::register_class<Curve3D>();
	ClassDB::register_class<Path>();
	ClassDB::register_class<PathFollow>();
	ClassDB::register_class<VisibilityNotifier>();
	ClassDB::register_class<VisibilityEnabler>();
	ClassDB::register_class<WorldEnvironment>();
	ClassDB::register_class<RemoteTransform>();

	ClassDB::register_virtual_class<Joint>();
	ClassDB::register_class<PinJoint>();
	ClassDB::register_class<HingeJoint>();
	ClassDB::register_class<SliderJoint>();
	ClassDB::register_class<ConeTwistJoint>();
	ClassDB::register_class<Generic6DOFJoint>();

	OS::get_singleton()->yield(); //may take time to init

#endif
	ClassDB::register_class<MeshLibrary>();
	AcceptDialog::set_swap_ok_cancel(GLOBAL_DEF("gui/common/swap_ok_cancel", bool(OS::get_singleton()->get_swap_ok_cancel())));

	ClassDB::register_class<Shader>();
	ClassDB::register_class<ShaderMaterial>();
	ClassDB::register_virtual_class<CanvasItem>();
	ClassDB::register_class<CanvasItemMaterial>();
	SceneTree::add_idle_callback(CanvasItemMaterial::flush_changes);
	CanvasItemMaterial::init_shaders();
	ClassDB::register_class<Node2D>();
	ClassDB::register_class<Particles2D>();
	//ClassDB::register_class<ParticleAttractor2D>();
	ClassDB::register_class<Sprite>();
	//ClassDB::register_type<ViewportSprite>();
	ClassDB::register_class<SpriteFrames>();
	ClassDB::register_class<AnimatedSprite>();
	ClassDB::register_class<Position2D>();
	ClassDB::register_class<Line2D>();
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
	ClassDB::register_class<Light2D>();
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
	ClassDB::register_class<TileMap>();
	ClassDB::register_class<ParallaxBackground>();
	ClassDB::register_class<ParallaxLayer>();
	ClassDB::register_class<TouchScreenButton>();
	ClassDB::register_class<RemoteTransform2D>();

	OS::get_singleton()->yield(); //may take time to init

	/* REGISTER RESOURCES */

	ClassDB::register_virtual_class<Shader>();

#ifndef _3D_DISABLED
	ClassDB::register_virtual_class<Mesh>();
	ClassDB::register_class<ArrayMesh>();
	ClassDB::register_virtual_class<PrimitiveMesh>();
	ClassDB::register_class<CapsuleMesh>();
	ClassDB::register_class<CubeMesh>();
	ClassDB::register_class<CylinderMesh>();
	ClassDB::register_class<PlaneMesh>();
	ClassDB::register_class<PrismMesh>();
	ClassDB::register_class<QuadMesh>();
	ClassDB::register_class<SphereMesh>();
	ClassDB::register_virtual_class<Material>();
	ClassDB::register_class<SpatialMaterial>();
	SceneTree::add_idle_callback(SpatialMaterial::flush_changes);
	SpatialMaterial::init_shaders();

	ClassDB::register_class<ParticlesMaterial>();
	SceneTree::add_idle_callback(ParticlesMaterial::flush_changes);
	ParticlesMaterial::init_shaders();

	ClassDB::register_class<MultiMesh>();
	ClassDB::register_class<MeshLibrary>();

	OS::get_singleton()->yield(); //may take time to init

	ClassDB::register_virtual_class<Shape>();
	ClassDB::register_class<RayShape>();
	ClassDB::register_class<SphereShape>();
	ClassDB::register_class<BoxShape>();
	ClassDB::register_class<CapsuleShape>();
	ClassDB::register_class<PlaneShape>();
	ClassDB::register_class<ConvexPolygonShape>();
	ClassDB::register_class<ConcavePolygonShape>();

	ClassDB::register_class<SurfaceTool>();
	ClassDB::register_class<MeshDataTool>();

	OS::get_singleton()->yield(); //may take time to init

	ClassDB::register_class<SpatialVelocityTracker>();
#endif
	ClassDB::register_class<World>();
	ClassDB::register_class<Environment>();
	ClassDB::register_class<World2D>();
	ClassDB::register_virtual_class<Texture>();
	ClassDB::register_virtual_class<Sky>();
	ClassDB::register_class<PanoramaSky>();
	ClassDB::register_class<ProceduralSky>();
	ClassDB::register_class<StreamTexture>();
	ClassDB::register_class<ImageTexture>();
	ClassDB::register_class<AtlasTexture>();
	ClassDB::register_class<LargeTexture>();
	ClassDB::register_class<CurveTexture>();
	ClassDB::register_class<GradientTexture>();
	ClassDB::register_class<CubeMap>();
	ClassDB::register_class<Animation>();
	ClassDB::register_virtual_class<Font>();
	ClassDB::register_class<BitmapFont>();
	ClassDB::register_class<Curve>();

	ClassDB::register_class<DynamicFontData>();
	ClassDB::register_class<DynamicFont>();

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

	ClassDB::register_class<Navigation2D>();
	ClassDB::register_class<NavigationPolygon>();
	ClassDB::register_class<NavigationPolygonInstance>();

	OS::get_singleton()->yield(); //may take time to init

	ClassDB::register_virtual_class<SceneState>();
	ClassDB::register_class<PackedScene>();

	ClassDB::register_class<SceneTree>();
	ClassDB::register_virtual_class<SceneTreeTimer>(); //sorry, you can't create it

#ifndef DISABLE_DEPRECATED
	ClassDB::add_compatibility_class("ImageSkyBox", "PanoramaSky");
	ClassDB::add_compatibility_class("FixedSpatialMaterial", "SpatialMaterial");
	ClassDB::add_compatibility_class("Mesh", "ArrayMesh");

#endif

	OS::get_singleton()->yield(); //may take time to init

	resource_saver_text = memnew(ResourceFormatSaverText);
	ResourceSaver::add_resource_format_saver(resource_saver_text, true);

	resource_loader_text = memnew(ResourceFormatLoaderText);
	ResourceLoader::add_resource_format_loader(resource_loader_text, true);

	for (int i = 0; i < 20; i++) {
		GLOBAL_DEF("layer_names/2d_render/layer_" + itos(i + 1), "");
		GLOBAL_DEF("layer_names/2d_physics/layer_" + itos(i + 1), "");
		GLOBAL_DEF("layer_names/3d_render/layer_" + itos(i + 1), "");
		GLOBAL_DEF("layer_names/3d_physics/layer_" + itos(i + 1), "");
	}
}

void unregister_scene_types() {

	clear_default_theme();

	memdelete(resource_loader_dynamic_font);
	memdelete(resource_loader_stream_texture);
	memdelete(resource_loader_theme);

	if (resource_saver_text) {
		memdelete(resource_saver_text);
	}
	if (resource_loader_text) {
		memdelete(resource_loader_text);
	}

	SpatialMaterial::finish_shaders();
	ParticlesMaterial::finish_shaders();
	CanvasItemMaterial::finish_shaders();
	SceneStringNames::free();
}
