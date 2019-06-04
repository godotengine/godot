/*************************************************************************/
/*  register_scene_types.cpp                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "globals.h"
#include "os/os.h"
#include "scene/io/resource_format_image.h"
#include "scene/io/resource_format_wav.h"

//#include "scene/io/scene_format_script.h"
#include "object_type_db.h"
#include "resources/default_theme/default_theme.h"
#include "scene/2d/light_2d.h"
#include "scene/2d/light_occluder_2d.h"
#include "scene/2d/particles_2d.h"
#include "scene/2d/path_2d.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/button_array.h"
#include "scene/gui/button_group.h"
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
#include "scene/gui/patch_9_frame.h"
#include "scene/gui/popup_menu.h"
#include "scene/gui/progress_bar.h"
#include "scene/gui/reference_frame.h"
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
#include "scene/gui/texture_frame.h"
#include "scene/gui/texture_progress.h"
#include "scene/gui/tool_button.h"
#include "scene/gui/tree.h"
#include "scene/gui/video_player.h"
#include "scene/main/canvas_layer.h"
#include "scene/main/http_request.h"
#include "scene/main/instance_placeholder.h"
#include "scene/main/viewport.h"
#include "scene/resources/video_stream.h"

#include "scene/2d/animated_sprite.h"
#include "scene/2d/back_buffer_copy.h"
#include "scene/2d/canvas_item.h"
#include "scene/2d/polygon_2d.h"
#include "scene/2d/sprite.h"

#include "scene/2d/visibility_notifier_2d.h"

#include "scene/2d/area_2d.h"
#include "scene/2d/camera_2d.h"
#include "scene/2d/canvas_modulate.h"
#include "scene/2d/collision_polygon_2d.h"
#include "scene/2d/collision_shape_2d.h"
#include "scene/2d/joints_2d.h"
#include "scene/2d/navigation2d.h"
#include "scene/2d/parallax_background.h"
#include "scene/2d/parallax_layer.h"
#include "scene/2d/physics_body_2d.h"
#include "scene/2d/ray_cast_2d.h"
#include "scene/2d/remote_transform_2d.h"
#include "scene/2d/sample_player_2d.h"
#include "scene/2d/screen_button.h"
#include "scene/2d/sound_player_2d.h"
#include "scene/2d/y_sort.h"

#include "scene/2d/position_2d.h"
#include "scene/2d/tile_map.h"
//#include "scene/2d/tile_map.h"
#include "scene/resources/tile_set.h"

#include "scene/animation/animation_player.h"
#include "scene/animation/animation_tree_player.h"
#include "scene/animation/tween.h"
#include "scene/main/resource_preloader.h"
#include "scene/main/scene_main_loop.h"
#include "scene/resources/packed_scene.h"

#include "scene/resources/dynamic_font.h"
#include "scene/resources/dynamic_font_stb.h"
#include "scene/resources/mesh_data_tool.h"
#include "scene/resources/scene_preloader.h"
#include "scene/resources/surface_tool.h"

#include "scene/main/timer.h"

#include "scene/audio/event_player.h"
#include "scene/audio/sound_room_params.h"
#include "scene/audio/stream_player.h"
#include "scene/resources/box_shape.h"
#include "scene/resources/capsule_shape.h"
#include "scene/resources/concave_polygon_shape.h"
#include "scene/resources/convex_polygon_shape.h"
#include "scene/resources/plane_shape.h"
#include "scene/resources/ray_shape.h"
#include "scene/resources/sphere_shape.h"

#include "scene/resources/capsule_shape_2d.h"
#include "scene/resources/circle_shape_2d.h"
#include "scene/resources/concave_polygon_shape_2d.h"
#include "scene/resources/convex_polygon_shape_2d.h"
#include "scene/resources/rectangle_shape_2d.h"
#include "scene/resources/segment_shape_2d.h"
#include "scene/resources/shape_line_2d.h"

#include "scene/resources/mesh_library.h"

#include "scene/resources/polygon_path_finder.h"

#include "scene/audio/sample_player.h"
#include "scene/resources/material.h"
#include "scene/resources/mesh.h"
#include "scene/resources/room.h"
#include "scene/resources/sample.h"
#include "scene/resources/texture.h"

#include "scene/resources/shader_graph.h"

#include "scene/resources/world.h"
#include "scene/resources/world_2d.h"

#include "scene/resources/audio_stream.h"
#include "scene/resources/bit_mask.h"
#include "scene/resources/color_ramp.h"
#include "scene/resources/gibberish_stream.h"
#include "scene/resources/sample_library.h"
#include "scene/scene_string_names.h"

#include "scene/3d/bone_attachment.h"
#include "scene/3d/path.h"
#include "scene/3d/room_instance.h"
#include "scene/3d/scenario_fx.h"
#include "scene/3d/skeleton.h"
#include "scene/3d/spatial.h"

#include "scene/3d/visibility_notifier.h"

#ifndef _3D_DISABLED
#include "scene/3d/camera.h"
#include "scene/3d/listener.h"

#include "scene/3d/interpolated_camera.h"
#include "scene/3d/light.h"
#include "scene/3d/mesh_instance.h"
#include "scene/3d/particles.h"
#include "scene/3d/physics_body.h"
#include "scene/3d/portal.h"
#include "scene/3d/position_3d.h"
#include "scene/3d/quad.h"
#include "scene/3d/test_cube.h"
#include "scene/resources/environment.h"

#include "scene/3d/area.h"
#include "scene/3d/baked_light_instance.h"
#include "scene/3d/body_shape.h"
#include "scene/3d/collision_polygon.h"
#include "scene/3d/immediate_geometry.h"
#include "scene/3d/multimesh_instance.h"
#include "scene/3d/navigation.h"
#include "scene/3d/navigation_mesh.h"
#include "scene/3d/physics_joint.h"
#include "scene/3d/proximity_group.h"
#include "scene/3d/ray_cast.h"
#include "scene/3d/spatial_sample_player.h"
#include "scene/3d/spatial_stream_player.h"
#include "scene/3d/sprite_3d.h"
#include "scene/3d/vehicle_body.h"
#endif

#include "scene/resources/scene_format_text.h"

static ResourceFormatLoaderImage *resource_loader_image = NULL;
static ResourceFormatLoaderWAV *resource_loader_wav = NULL;

#ifdef TOOLS_ENABLED

#endif
static ResourceFormatLoaderTheme *resource_loader_theme = NULL;
static ResourceFormatLoaderShader *resource_loader_shader = NULL;

static ResourceFormatSaverText *resource_saver_text = NULL;
static ResourceFormatLoaderText *resource_loader_text = NULL;

static ResourceFormatLoaderDynamicFont *resource_loader_dynamic_font = NULL;

//static SceneStringNames *string_names;

void register_scene_types() {

	SceneStringNames::create();

	OS::get_singleton()->yield(); //may take time to init

	Node::init_node_hrcr();

	resource_loader_image = memnew(ResourceFormatLoaderImage);
	ResourceLoader::add_resource_format_loader(resource_loader_image);

	resource_loader_wav = memnew(ResourceFormatLoaderWAV);
	ResourceLoader::add_resource_format_loader(resource_loader_wav);
	resource_loader_dynamic_font = memnew(ResourceFormatLoaderDynamicFont);
	ResourceLoader::add_resource_format_loader(resource_loader_dynamic_font);

#ifdef TOOLS_ENABLED

	//scene first!

#endif

	resource_loader_theme = memnew(ResourceFormatLoaderTheme);
	ResourceLoader::add_resource_format_loader(resource_loader_theme);

	resource_loader_shader = memnew(ResourceFormatLoaderShader);
	ResourceLoader::add_resource_format_loader(resource_loader_shader);

	bool default_theme_hidpi = GLOBAL_DEF("display/use_hidpi_theme", false);
	Globals::get_singleton()->set_custom_property_info("display/use_hidpi_theme", PropertyInfo(Variant::BOOL, "display/use_hidpi_theme", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED));
	String theme_path = GLOBAL_DEF("display/custom_theme", "");
	Globals::get_singleton()->set_custom_property_info("display/custom_theme", PropertyInfo(Variant::STRING, "display/custom_theme", PROPERTY_HINT_FILE, "*.tres,*.res", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED));
	String font_path = GLOBAL_DEF("display/custom_theme_font", "");
	Globals::get_singleton()->set_custom_property_info("display/custom_theme_font", PropertyInfo(Variant::STRING, "display/custom_theme_font", PROPERTY_HINT_FILE, "*.tres,*.res,*.fnt", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED));

	if (theme_path != String()) {
		Ref<Theme> theme = ResourceLoader::load(theme_path);
		if (theme.is_valid()) {
			Theme::set_default(theme);
		}
	} else {

		Ref<Font> font;
		if (font_path != String()) {
			font = ResourceLoader::load(font_path);
		}
		make_default_theme(default_theme_hidpi, font);
	}

	OS::get_singleton()->yield(); //may take time to init

	ObjectTypeDB::register_type<Object>();

	ObjectTypeDB::register_type<Node>();
	ObjectTypeDB::register_virtual_type<InstancePlaceholder>();

	ObjectTypeDB::register_type<Viewport>();
	ObjectTypeDB::register_virtual_type<RenderTargetTexture>();
	ObjectTypeDB::register_type<HTTPRequest>();
	ObjectTypeDB::register_type<Timer>();
	ObjectTypeDB::register_type<CanvasLayer>();
	ObjectTypeDB::register_type<CanvasModulate>();
	ObjectTypeDB::register_type<ResourcePreloader>();

	/* REGISTER GUI */
	ObjectTypeDB::register_type<ButtonGroup>();
	ObjectTypeDB::register_virtual_type<BaseButton>();

	OS::get_singleton()->yield(); //may take time to init

	ObjectTypeDB::register_type<ShortCut>();
	ObjectTypeDB::register_type<Control>();
	//	ObjectTypeDB::register_type<EmptyControl>();
	ObjectTypeDB::add_compatibility_type("EmptyControl", "Control");
	ObjectTypeDB::register_type<Button>();
	ObjectTypeDB::register_type<Label>();
	ObjectTypeDB::register_type<HScrollBar>();
	ObjectTypeDB::register_type<VScrollBar>();
	ObjectTypeDB::register_type<ProgressBar>();
	ObjectTypeDB::register_type<HSlider>();
	ObjectTypeDB::register_type<VSlider>();
	ObjectTypeDB::register_type<Popup>();
	ObjectTypeDB::register_type<PopupPanel>();
	ObjectTypeDB::register_type<MenuButton>();
	ObjectTypeDB::register_type<CheckBox>();
	ObjectTypeDB::register_type<CheckButton>();
	ObjectTypeDB::register_type<ToolButton>();
	ObjectTypeDB::register_type<LinkButton>();
	ObjectTypeDB::register_type<Panel>();
	ObjectTypeDB::register_type<Range>();

	OS::get_singleton()->yield(); //may take time to init

	ObjectTypeDB::register_type<TextureFrame>();
	ObjectTypeDB::register_type<ColorFrame>();
	ObjectTypeDB::register_type<Patch9Frame>();
	ObjectTypeDB::register_type<TabContainer>();
	ObjectTypeDB::register_type<Tabs>();
	ObjectTypeDB::register_virtual_type<Separator>();
	ObjectTypeDB::register_type<HSeparator>();
	ObjectTypeDB::register_type<VSeparator>();
	ObjectTypeDB::register_type<TextureButton>();
	ObjectTypeDB::register_type<Container>();
	ObjectTypeDB::register_virtual_type<BoxContainer>();
	ObjectTypeDB::register_type<HBoxContainer>();
	ObjectTypeDB::register_type<VBoxContainer>();
	ObjectTypeDB::register_type<GridContainer>();
	ObjectTypeDB::register_type<CenterContainer>();
	ObjectTypeDB::register_type<ScrollContainer>();
	ObjectTypeDB::register_type<PanelContainer>();
	ObjectTypeDB::register_virtual_type<SplitContainer>();
	ObjectTypeDB::register_type<HSplitContainer>();
	ObjectTypeDB::register_type<VSplitContainer>();
	ObjectTypeDB::register_type<GraphNode>();
	ObjectTypeDB::register_type<GraphEdit>();

	OS::get_singleton()->yield(); //may take time to init

	ObjectTypeDB::register_virtual_type<ButtonArray>();
	ObjectTypeDB::register_type<HButtonArray>();
	ObjectTypeDB::register_type<VButtonArray>();
	ObjectTypeDB::register_type<TextureProgress>();
	ObjectTypeDB::register_type<ItemList>();

#ifndef ADVANCED_GUI_DISABLED

	ObjectTypeDB::register_type<FileDialog>();
	ObjectTypeDB::register_type<LineEdit>();
	ObjectTypeDB::register_type<PopupMenu>();
	ObjectTypeDB::register_type<Tree>();

	ObjectTypeDB::register_type<TextEdit>();

	ObjectTypeDB::register_virtual_type<TreeItem>();
	ObjectTypeDB::register_type<OptionButton>();
	ObjectTypeDB::register_type<SpinBox>();
	ObjectTypeDB::register_type<ReferenceFrame>();
	ObjectTypeDB::register_type<ColorPicker>();
	ObjectTypeDB::register_type<ColorPickerButton>();
	ObjectTypeDB::register_type<RichTextLabel>();
	ObjectTypeDB::register_type<PopupDialog>();
	ObjectTypeDB::register_type<WindowDialog>();
	ObjectTypeDB::register_type<AcceptDialog>();
	ObjectTypeDB::register_type<ConfirmationDialog>();
	ObjectTypeDB::register_type<VideoPlayer>();
	ObjectTypeDB::register_type<MarginContainer>();

	OS::get_singleton()->yield(); //may take time to init

#endif

	/* REGISTER 3D */

	ObjectTypeDB::register_type<Spatial>();
	ObjectTypeDB::register_virtual_type<SpatialGizmo>();
	ObjectTypeDB::register_type<Skeleton>();
	ObjectTypeDB::register_type<AnimationPlayer>();
	ObjectTypeDB::register_type<Tween>();

	OS::get_singleton()->yield(); //may take time to init

#ifndef _3D_DISABLED
	ObjectTypeDB::register_type<BoneAttachment>();
	ObjectTypeDB::register_virtual_type<VisualInstance>();
	ObjectTypeDB::register_type<Camera>();
	ObjectTypeDB::register_type<Listener>();
	ObjectTypeDB::register_type<InterpolatedCamera>();
	ObjectTypeDB::register_type<TestCube>();
	ObjectTypeDB::register_type<MeshInstance>();
	ObjectTypeDB::register_type<ImmediateGeometry>();
	ObjectTypeDB::register_type<Sprite3D>();
	ObjectTypeDB::register_type<AnimatedSprite3D>();
	ObjectTypeDB::register_virtual_type<Light>();
	ObjectTypeDB::register_type<DirectionalLight>();
	ObjectTypeDB::register_type<OmniLight>();
	ObjectTypeDB::register_type<SpotLight>();
	ObjectTypeDB::register_type<AnimationTreePlayer>();
	ObjectTypeDB::register_type<Portal>();
	ObjectTypeDB::register_type<Particles>();
	ObjectTypeDB::register_type<Position3D>();
	ObjectTypeDB::register_type<Quad>();
	ObjectTypeDB::register_type<NavigationMeshInstance>();
	ObjectTypeDB::register_type<NavigationMesh>();
	ObjectTypeDB::register_type<Navigation>();

	OS::get_singleton()->yield(); //may take time to init

	ObjectTypeDB::register_virtual_type<CollisionObject>();
	ObjectTypeDB::register_type<StaticBody>();
	ObjectTypeDB::register_type<RigidBody>();
	ObjectTypeDB::register_type<KinematicBody>();

	ObjectTypeDB::register_type<VehicleBody>();
	ObjectTypeDB::register_type<VehicleWheel>();
	ObjectTypeDB::register_type<Area>();
	ObjectTypeDB::register_type<ProximityGroup>();
	ObjectTypeDB::register_type<CollisionShape>();
	ObjectTypeDB::register_type<CollisionPolygon>();
	ObjectTypeDB::register_type<RayCast>();
	ObjectTypeDB::register_type<MultiMeshInstance>();
	ObjectTypeDB::register_type<Room>();
	ObjectTypeDB::register_type<Curve3D>();
	ObjectTypeDB::register_type<Path>();
	ObjectTypeDB::register_type<PathFollow>();
	ObjectTypeDB::register_type<VisibilityNotifier>();
	ObjectTypeDB::register_type<VisibilityEnabler>();
	ObjectTypeDB::register_type<BakedLightInstance>();
	ObjectTypeDB::register_type<BakedLightSampler>();
	ObjectTypeDB::register_type<WorldEnvironment>();

	ObjectTypeDB::register_virtual_type<Joint>();
	ObjectTypeDB::register_type<PinJoint>();
	ObjectTypeDB::register_type<HingeJoint>();
	ObjectTypeDB::register_type<SliderJoint>();
	ObjectTypeDB::register_type<ConeTwistJoint>();
	ObjectTypeDB::register_type<Generic6DOFJoint>();

	//scenariofx

	OS::get_singleton()->yield(); //may take time to init

	ObjectTypeDB::register_type<SpatialSamplePlayer>();
	ObjectTypeDB::register_type<SpatialStreamPlayer>();
	ObjectTypeDB::register_type<SoundRoomParams>();

#endif
	ObjectTypeDB::register_type<MeshLibrary>();
	AcceptDialog::set_swap_ok_cancel(GLOBAL_DEF("display/swap_ok_cancel", bool(OS::get_singleton()->get_swap_ok_cancel())));

	ObjectTypeDB::register_type<SamplePlayer>();
	ObjectTypeDB::register_type<StreamPlayer>();
	ObjectTypeDB::register_type<EventPlayer>();

	ObjectTypeDB::register_type<CanvasItemMaterial>();
	ObjectTypeDB::register_virtual_type<CanvasItem>();
	ObjectTypeDB::register_type<Node2D>();
	ObjectTypeDB::register_type<Particles2D>();
	ObjectTypeDB::register_type<ParticleAttractor2D>();
	ObjectTypeDB::register_type<Sprite>();
	ObjectTypeDB::register_type<ViewportSprite>();
	ObjectTypeDB::register_type<SpriteFrames>();
	ObjectTypeDB::register_type<AnimatedSprite>();
	ObjectTypeDB::register_type<Position2D>();
	ObjectTypeDB::register_virtual_type<CollisionObject2D>();
	ObjectTypeDB::register_virtual_type<PhysicsBody2D>();
	ObjectTypeDB::register_type<StaticBody2D>();
	ObjectTypeDB::register_type<RigidBody2D>();
	ObjectTypeDB::register_type<KinematicBody2D>();
	KinematicBody2D::motion_fix_enabled = bool(GLOBAL_DEF("physics_2d/motion_fix_enabled", false));
	ObjectTypeDB::register_type<Area2D>();
	ObjectTypeDB::register_type<CollisionShape2D>();
	ObjectTypeDB::register_type<CollisionPolygon2D>();
	ObjectTypeDB::register_type<RayCast2D>();
	ObjectTypeDB::register_type<VisibilityNotifier2D>();
	ObjectTypeDB::register_type<VisibilityEnabler2D>();
	ObjectTypeDB::register_type<Polygon2D>();
	ObjectTypeDB::register_type<Light2D>();
	ObjectTypeDB::register_type<LightOccluder2D>();
	ObjectTypeDB::register_type<OccluderPolygon2D>();
	ObjectTypeDB::register_type<YSort>();
	ObjectTypeDB::register_type<BackBufferCopy>();
	if (bool(GLOBAL_DEF("physics/remove_collision_helpers_at_runtime", false))) {
		ObjectTypeDB::set_type_enabled("CollisionShape2D", false);
		ObjectTypeDB::set_type_enabled("CollisionPolygon2D", false);
		ObjectTypeDB::set_type_enabled("CollisionShape", false);
		ObjectTypeDB::set_type_enabled("CollisionPolygon", false);
	}

	OS::get_singleton()->yield(); //may take time to init

	ObjectTypeDB::register_type<Camera2D>();
	ObjectTypeDB::register_virtual_type<Joint2D>();
	ObjectTypeDB::register_type<PinJoint2D>();
	ObjectTypeDB::register_type<GrooveJoint2D>();
	ObjectTypeDB::register_type<DampedSpringJoint2D>();
	ObjectTypeDB::register_type<TileSet>();
	ObjectTypeDB::register_type<TileMap>();
	ObjectTypeDB::register_type<ParallaxBackground>();
	ObjectTypeDB::register_type<ParallaxLayer>();
	ObjectTypeDB::register_virtual_type<SoundPlayer2D>();
	ObjectTypeDB::register_type<SamplePlayer2D>();
	ObjectTypeDB::register_type<TouchScreenButton>();
	ObjectTypeDB::register_type<RemoteTransform2D>();

	OS::get_singleton()->yield(); //may take time to init

	/* REGISTER RESOURCES */

	ObjectTypeDB::register_virtual_type<Shader>();
	ObjectTypeDB::register_virtual_type<ShaderGraph>();
	ObjectTypeDB::register_type<CanvasItemShader>();
	ObjectTypeDB::register_type<CanvasItemShaderGraph>();

#ifndef _3D_DISABLED
	ObjectTypeDB::register_type<Mesh>();
	ObjectTypeDB::register_virtual_type<Material>();
	ObjectTypeDB::register_type<FixedMaterial>();
	ObjectTypeDB::register_type<ShaderMaterial>();
	ObjectTypeDB::register_type<RoomBounds>();
	ObjectTypeDB::register_type<MaterialShaderGraph>();
	ObjectTypeDB::register_type<MaterialShader>();
	ObjectTypeDB::add_compatibility_type("Shader", "MaterialShader");
	ObjectTypeDB::add_compatibility_type("ParticleSystemMaterial", "FixedMaterial");
	ObjectTypeDB::add_compatibility_type("UnshadedMaterial", "FixedMaterial");
	ObjectTypeDB::register_type<MultiMesh>();
	ObjectTypeDB::register_type<MeshLibrary>();

	OS::get_singleton()->yield(); //may take time to init

	ObjectTypeDB::register_type<RayShape>();
	ObjectTypeDB::register_type<SphereShape>();
	ObjectTypeDB::register_type<BoxShape>();
	ObjectTypeDB::register_type<CapsuleShape>();
	ObjectTypeDB::register_type<PlaneShape>();
	ObjectTypeDB::register_type<ConvexPolygonShape>();
	ObjectTypeDB::register_type<ConcavePolygonShape>();

	ObjectTypeDB::register_type<SurfaceTool>();
	ObjectTypeDB::register_type<MeshDataTool>();
	ObjectTypeDB::register_type<BakedLight>();

	OS::get_singleton()->yield(); //may take time to init

#endif
	ObjectTypeDB::register_type<World>();
	ObjectTypeDB::register_type<Environment>();
	ObjectTypeDB::register_type<World2D>();
	ObjectTypeDB::register_virtual_type<Texture>();
	ObjectTypeDB::register_type<ImageTexture>();
	ObjectTypeDB::register_type<AtlasTexture>();
	ObjectTypeDB::register_type<LargeTexture>();
	ObjectTypeDB::register_type<CubeMap>();
	ObjectTypeDB::register_type<Animation>();
	ObjectTypeDB::register_virtual_type<Font>();
	ObjectTypeDB::register_type<BitmapFont>();

	ObjectTypeDB::register_type<DynamicFontData>();
	ObjectTypeDB::register_type<DynamicFont>();

	ObjectTypeDB::register_type<StyleBoxEmpty>();
	ObjectTypeDB::register_type<StyleBoxTexture>();
	ObjectTypeDB::register_type<StyleBoxFlat>();
	ObjectTypeDB::register_type<StyleBoxImageMask>();
	ObjectTypeDB::register_type<Theme>();

	ObjectTypeDB::add_compatibility_type("Font", "BitmapFont");

	ObjectTypeDB::register_type<PolygonPathFinder>();
	ObjectTypeDB::register_type<BitMap>();
	ObjectTypeDB::register_type<ColorRamp>();

	OS::get_singleton()->yield(); //may take time to init

	ObjectTypeDB::register_type<Sample>();
	ObjectTypeDB::register_type<SampleLibrary>();
	ObjectTypeDB::register_virtual_type<AudioStream>();
	ObjectTypeDB::register_virtual_type<AudioStreamPlayback>();
	//TODO: Adapt to the new AudioStream API or drop (GH-3307)
	//	ObjectTypeDB::register_type<AudioStreamGibberish>();
	ObjectTypeDB::register_virtual_type<VideoStream>();

	OS::get_singleton()->yield(); //may take time to init

	ObjectTypeDB::register_virtual_type<Shape2D>();
	ObjectTypeDB::register_type<LineShape2D>();
	ObjectTypeDB::register_type<SegmentShape2D>();
	ObjectTypeDB::register_type<RayShape2D>();
	ObjectTypeDB::register_type<CircleShape2D>();
	ObjectTypeDB::register_type<RectangleShape2D>();
	ObjectTypeDB::register_type<CapsuleShape2D>();
	ObjectTypeDB::register_type<ConvexPolygonShape2D>();
	ObjectTypeDB::register_type<ConcavePolygonShape2D>();
	ObjectTypeDB::register_type<Curve2D>();
	ObjectTypeDB::register_type<Path2D>();
	ObjectTypeDB::register_type<PathFollow2D>();

	ObjectTypeDB::register_type<Navigation2D>();
	ObjectTypeDB::register_type<NavigationPolygon>();
	ObjectTypeDB::register_type<NavigationPolygonInstance>();

	OS::get_singleton()->yield(); //may take time to init

	ObjectTypeDB::register_virtual_type<SceneState>();
	ObjectTypeDB::register_type<PackedScene>();

	ObjectTypeDB::register_type<SceneTree>();

	OS::get_singleton()->yield(); //may take time to init

	resource_saver_text = memnew(ResourceFormatSaverText);
	ResourceSaver::add_resource_format_saver(resource_saver_text, true);

	resource_loader_text = memnew(ResourceFormatLoaderText);
	ResourceLoader::add_resource_format_loader(resource_loader_text, true);
}

void unregister_scene_types() {

	clear_default_theme();

	memdelete(resource_loader_image);
	memdelete(resource_loader_wav);
	memdelete(resource_loader_dynamic_font);

#ifdef TOOLS_ENABLED

#endif

	memdelete(resource_loader_theme);
	memdelete(resource_loader_shader);

	if (resource_saver_text) {
		memdelete(resource_saver_text);
	}
	if (resource_loader_text) {
		memdelete(resource_loader_text);
	}
	SceneStringNames::free();
}
