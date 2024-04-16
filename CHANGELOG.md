# Changelog

This file lists all changes made between the current feature branch and the
previous feature release. It is equivalent to the listings on our
[interactive changelog](https://godotengine.github.io/godot-interactive-changelog/).

Changelogs for earlier feature releases are available in their respective Git
branches, and linked at the [end of this file](#Past-releases).

## 4.2.2 - TBA

- [Interactive changelog](https://godotengine.github.io/godot-interactive-changelog/#4.2.2)

#### 2D

- Add Texture2D and Texture3D icons ([GH-81169](https://github.com/godotengine/godot/pull/81169)).
- Change suffix of SkeletonModification2DTwoBoneIK from m to px ([GH-86056](https://github.com/godotengine/godot/pull/86056)).
- Reset TileMap editor `drag_type` when the toolbar mode is not selected ([GH-86066](https://github.com/godotengine/godot/pull/86066)).
- Fix potential infinite loop when calculating tile editor zoom level ([GH-86568](https://github.com/godotengine/godot/pull/86568)).
- Editor: Fix threading problems with `TileMap` preview ([GH-87470](https://github.com/godotengine/godot/pull/87470)).
- TileSet: Fix crash when deleting dragged polygon point ([GH-88912](https://github.com/godotengine/godot/pull/88912)).
- Prevent threading problems in `TileMap` ([GH-88916](https://github.com/godotengine/godot/pull/88916)).

#### 3D

- Prompt require editor restart to user when gizmo color changed ([GH-82872](https://github.com/godotengine/godot/pull/82872)).
- Improve Curve3D debug drawing ([GH-83698](https://github.com/godotengine/godot/pull/83698)).
- Use screen-aligned quads for origin lines to avoid issues on NVidia ([GH-83895](https://github.com/godotengine/godot/pull/83895)).
- Prevent division by 0 when creating vertices of a PrismMesh ([GH-86931](https://github.com/godotengine/godot/pull/86931)).
- Make viewport message dependent on framerate not physics step ([GH-87631](https://github.com/godotengine/godot/pull/87631)).
- Show modifier key functionality on all the editor tools ([GH-87989](https://github.com/godotengine/godot/pull/87989)).
- Don't access Node3D/Node2D/Control global transform in `reparent` unless needed ([GH-89003](https://github.com/godotengine/godot/pull/89003)).
- Fix Path3D picking working incorrectly when viewport is in half resolution ([GH-89901](https://github.com/godotengine/godot/pull/89901)).
- Fix OpenGL `_shadow_atlas_find_shadow` error when light instance is freed ([GH-90233](https://github.com/godotengine/godot/pull/90233)).
- Fix `RenderingServer.instance_set_transform` docs saying it's not global ([GH-90396](https://github.com/godotengine/godot/pull/90396)).

#### Animation

- Fix setting bezier track handle mode from inspector ([GH-83533](https://github.com/godotengine/godot/pull/83533)).
- Make AnimationTree reference AnimationPlayer instead of AnimationMixer ([GH-84583](https://github.com/godotengine/godot/pull/84583)).
- Replace memory allocation point of ValueTrack correctly in AnimationMixer ([GH-85001](https://github.com/godotengine/godot/pull/85001)).
- Copy track update mode when adding reset key ([GH-85540](https://github.com/godotengine/godot/pull/85540)).
- Make unstore AnimationLibrary if AnimationTree is assigned AnimationPlayer ([GH-85575](https://github.com/godotengine/godot/pull/85575)).
- Fix linear interpolation not working with mixed (int/float) keyframes ([GH-86046](https://github.com/godotengine/godot/pull/86046)).
- Fix animated tile random start time not scaled to animation duration ([GH-86210](https://github.com/godotengine/godot/pull/86210)).
- Make default `blend_left` consider current blend amount ([GH-86221](https://github.com/godotengine/godot/pull/86221)).
- Fix discrete key retrieval method after start ([GH-86227](https://github.com/godotengine/godot/pull/86227)).
- Fix invalid 3-to-4 renames of `add_animation` to `add_animation_library` ([GH-86647](https://github.com/godotengine/godot/pull/86647)).
- Fix Audio track crossfade in AnimationPlayer does not work ([GH-86718](https://github.com/godotengine/godot/pull/86718)).
- Add autocompletion options for AnimatedSprite's other play methods ([GH-86733](https://github.com/godotengine/godot/pull/86733)).
- Fix passing int to tween's `from` with float property will be forced to interpolate as int ([GH-87459](https://github.com/godotengine/godot/pull/87459)).
- Move the line of infinity loop checking in AnimationStateMachine ([GH-89575](https://github.com/godotengine/godot/pull/89575)).
- Fix Bezier Editor throwing error when adding key with Ctrl+Click ([GH-89638](https://github.com/godotengine/godot/pull/89638)).
- Fix AnimationPlaybackTrack seeking behavior overall ([GH-89794](https://github.com/godotengine/godot/pull/89794)).
- Fix setting animation save paths on import breaking on Windows ([GH-90003](https://github.com/godotengine/godot/pull/90003)).
- Fix loop condition in bone mapping ([GH-90019](https://github.com/godotengine/godot/pull/90019)).

#### Assetlib

- Fix broken layout of asset library page ([GH-88761](https://github.com/godotengine/godot/pull/88761)).

#### Audio

- Fix permanently selected audio bus effects ([GH-85879](https://github.com/godotengine/godot/pull/85879)).
- Fix `OggPacketSequencePlayback::next_ogg_packet()` never returning false ([GH-85996](https://github.com/godotengine/godot/pull/85996)).
- Fix `AudioServer::start_playback_stream` does not iterate through given bus volumes ([GH-86584](https://github.com/godotengine/godot/pull/86584)).
- Fix playback position label update in Audio Stream Importer ([GH-86824](https://github.com/godotengine/godot/pull/86824)).
- Fix Dummy audio driver initialization issue on WASAPI output device initialization failure ([GH-87010](https://github.com/godotengine/godot/pull/87010)).
- Fix a possible crash when importing an OGG file with zero-length packets ([GH-87246](https://github.com/godotengine/godot/pull/87246)).
- Fix audio crackling issues due to incorrect WASAPI buffer size ([GH-89283](https://github.com/godotengine/godot/pull/89283)).

#### Buildsystem

- SCons: Add `stack_size` and `default_pthread_stack_size` options to Web target ([GH-75166](https://github.com/godotengine/godot/pull/75166)).
- Remove lgtm.yml since LGTM is now replaced by Github Code Scanning ([GH-81874](https://github.com/godotengine/godot/pull/81874)).
- Fix some build errors with `disable_3d=yes` ([GH-86874](https://github.com/godotengine/godot/pull/86874)).
- makerst: Suggest using `[code skip-lint]` to skip warnings if intended ([GH-87218](https://github.com/godotengine/godot/pull/87218)).
- macOS: Fix MoltenVK SDK detection after file location changes in 1.3.275.0 ([GH-87305](https://github.com/godotengine/godot/pull/87305)).
- Update Android dependencies for the project ([GH-87346](https://github.com/godotengine/godot/pull/87346)).
- iOS: Update linker flags for Xcode 15.2 ([GH-87358](https://github.com/godotengine/godot/pull/87358)).
- CI: Update `mymindstorm/setup-emsdk` to v14, should fix cache folder conflicts ([GH-87575](https://github.com/godotengine/godot/pull/87575)).
- Undefine yet another macro from Windows headers ([GH-87627](https://github.com/godotengine/godot/pull/87627)).
- CI scripts: Fix `printf` for format checks ([GH-87864](https://github.com/godotengine/godot/pull/87864)).
- Add basic Emacs `.gitignore` entries ([GH-87878](https://github.com/godotengine/godot/pull/87878)).
- macOS: Check Vulkan SDK version when looking for MoltenVK libs ([GH-87960](https://github.com/godotengine/godot/pull/87960)).
- Fix emscripten 3.1.51 breaking change about `*glGetProcAddress()` ([GH-87981](https://github.com/godotengine/godot/pull/87981)).
- Web: Bump closure compiler spec to `ECMASCRIPT_2021` ([GH-88010](https://github.com/godotengine/godot/pull/88010)).
- SCons: Fix incremental builds breaking when querying the dependency tree from a SCsub ([GH-88025](https://github.com/godotengine/godot/pull/88025)).
- Fix gradle build errors when the build path contains non-ASCII characters ([GH-88304](https://github.com/godotengine/godot/pull/88304)).
- OS_LinuxBSD: Add missing RenderServer include for `get_video_adapter_driver_info` ([GH-88525](https://github.com/godotengine/godot/pull/88525)).
- Add `WASM_BIGINT` linker flag to the web build ([GH-88594](https://github.com/godotengine/godot/pull/88594)).
- CI: Update actions to latest versions, use default runner .NET version ([GH-88671](https://github.com/godotengine/godot/pull/88671)).
- Fix running tests in template builds ([GH-88759](https://github.com/godotengine/godot/pull/88759)).
- SCons: Disable verbose output for SCU build ([GH-89483](https://github.com/godotengine/godot/pull/89483)).
- CI: Improve fetching of changed files ([GH-89980](https://github.com/godotengine/godot/pull/89980)).

#### C#

- Use `get_instance_binding` instead of set ([GH-84947](https://github.com/godotengine/godot/pull/84947)).
- Bump the `JetBrains.Rider.PathLocator` PackageReference to 1.0.8 ([GH-85460](https://github.com/godotengine/godot/pull/85460)).
- Fix typo in GD0103 error link ([GH-86592](https://github.com/godotengine/godot/pull/86592)).
- Fix return type hint for methods ([GH-86972](https://github.com/godotengine/godot/pull/86972)).
- [C#] Fix `Encloses` failing on shared upper bound for `AABB` and `Rect2(I)` ([GH-87264](https://github.com/godotengine/godot/pull/87264)).
- Fix not assigning `runtime_initialized` when initializing with AOT ([GH-87597](https://github.com/godotengine/godot/pull/87597)).
- Fix possible deadlock when creating scripts during a background garbage collection ([GH-87669](https://github.com/godotengine/godot/pull/87669)).
- Fix issues for StringName reference in `CSharpInstanceBridge.Get` ([GH-87682](https://github.com/godotengine/godot/pull/87682)).
- Fix incorrect condition for error filtering ([GH-87790](https://github.com/godotengine/godot/pull/87790)).
- Fix duplicate key issue on reload ([GH-87838](https://github.com/godotengine/godot/pull/87838)).
- Match Core implementation of `BinToInt` & `HexToInt` ([GH-88453](https://github.com/godotengine/godot/pull/88453)).
- Bump `Rider.PathLocator` nuget version, which provides a fix for detecting Rider installations ([GH-88544](https://github.com/godotengine/godot/pull/88544)).
- Hint fallback property as node when it is a node ([GH-89175](https://github.com/godotengine/godot/pull/89175)).
- [C#] Fix `Transform3D.InterpolateWith` applying rotation before scale ([GH-89843](https://github.com/godotengine/godot/pull/89843)).

#### Codestyle

- Style: Mark clang-format 16 as supported for pre-commit hook ([GH-85837](https://github.com/godotengine/godot/pull/85837)).

#### Core

- Fix crash caused by stale owner ([GH-78997](https://github.com/godotengine/godot/pull/78997)).
- Prevent crash on conversion of invalid data in `Image` ([GH-84782](https://github.com/godotengine/godot/pull/84782)).
- Fix duplicating sub-scene may get two copies of internal node ([GH-84824](https://github.com/godotengine/godot/pull/84824)).
- Fix RegEx `search_all` for zero length matches/lookahead ([GH-85783](https://github.com/godotengine/godot/pull/85783)).
- Fix `FileAccessPack::get_buffer` updating position past the length of file ([GH-85991](https://github.com/godotengine/godot/pull/85991)).
- Fix `RegEx.sub` truncating string when `end` is used ([GH-86052](https://github.com/godotengine/godot/pull/86052)).
- Include `name` field in MethodInfo operator == ([GH-86259](https://github.com/godotengine/godot/pull/86259)).
- Add `PackedRealArray` as an alias for `Vector<real_t>` ([GH-86324](https://github.com/godotengine/godot/pull/86324)).
- Fix data race in PagedArray ([GH-86412](https://github.com/godotengine/godot/pull/86412)).
- Ensure special characters are escaped in TSCN connections and editable hint ([GH-86417](https://github.com/godotengine/godot/pull/86417)).
- Initialize pointers in `a_star.cpp` ([GH-86590](https://github.com/godotengine/godot/pull/86590)).
- Pseudolocalization: Prevent out of bounds reads ([GH-86827](https://github.com/godotengine/godot/pull/86827)).
- Avoid regressing in progress reporting in resource load ([GH-86845](https://github.com/godotengine/godot/pull/86845)).
- Fix wrong fail condition in compressed texture's `_set_data` ([GH-86950](https://github.com/godotengine/godot/pull/86950)).
- Fix ZIPPacker storing file permissions unexpectedly ([GH-86985](https://github.com/godotengine/godot/pull/86985)).
- Fix `AABB.encloses` failing on shared upper bound ([GH-87118](https://github.com/godotengine/godot/pull/87118)).
- Fix inaccuracies in the report of leaked objects ([GH-87222](https://github.com/godotengine/godot/pull/87222)).
- Explicitly initialize all of `FileAccess::create_func[ACCESS_MAX]` ([GH-87389](https://github.com/godotengine/godot/pull/87389)).
- Add check for negative `buffer_size` inside `StreamPeerGZIP::start_(de)compression` ([GH-87448](https://github.com/godotengine/godot/pull/87448)).
- Fix `FileAccessMemory` off by one error in `eof_reached` ([GH-87571](https://github.com/godotengine/godot/pull/87571)).
- Update visuals immediately after resizing `Placeholder*` textures ([GH-87854](https://github.com/godotengine/godot/pull/87854)).
- Fix owner when importing meshes ([GH-88497](https://github.com/godotengine/godot/pull/88497)).
- Fix inefficient list iteration in Node ([GH-88785](https://github.com/godotengine/godot/pull/88785)).
- Fix `String::begins_with` when both strings are empty ([GH-89194](https://github.com/godotengine/godot/pull/89194)).
- Prevent further infinite recursion when printing errors ([GH-89490](https://github.com/godotengine/godot/pull/89490)).
- Fix `ResourceLoader.load` cache with relative paths ([GH-90038](https://github.com/godotengine/godot/pull/90038)).
- Fix `ZIPReader.get_files()` error on empty zip files ([GH-90404](https://github.com/godotengine/godot/pull/90404)).

#### Documentation

- Overhaul Node documentation ([GH-68560](https://github.com/godotengine/godot/pull/68560)).
- Update typed arrays documentation ([GH-79075](https://github.com/godotengine/godot/pull/79075)).
- Clarify `Tween.set_parallel()` ([GH-79758](https://github.com/godotengine/godot/pull/79758)).
- Clarify that `Callable` will not be encoded with `var_to_bytes` ([GH-79813](https://github.com/godotengine/godot/pull/79813)).
- Explain which nodes receive `NOTIFICATION_WM_SIZE_CHANGED` ([GH-80094](https://github.com/godotengine/godot/pull/80094)).
- Improve docs on how ParallaxLayer mirroring works ([GH-80896](https://github.com/godotengine/godot/pull/80896)).
- Clarify `PackedByteArray.decompress*` limitations with external data ([GH-81689](https://github.com/godotengine/godot/pull/81689)).
- Document Bounce = 1.0 not being sufficient for infinite energy conservation ([GH-82968](https://github.com/godotengine/godot/pull/82968)).
- Add instructions to get shape of `RayCast2D/3D` ([GH-83735](https://github.com/godotengine/godot/pull/83735)).
- Clarify behavior of RayCast when `get_collision_point()` is used inside a collision shape ([GH-84085](https://github.com/godotengine/godot/pull/84085)).
- Complete the docs for Quaternion ([GH-84140](https://github.com/godotengine/godot/pull/84140)).
- Clarify that `@GlobalScope.clamp` does not do component-wise clamping ([GH-84656](https://github.com/godotengine/godot/pull/84656)).
- Add performance note to `Array.resize()` ([GH-84666](https://github.com/godotengine/godot/pull/84666)).
- Document changing the window's resizable status at runtime ([GH-84886](https://github.com/godotengine/godot/pull/84886)).
- Fix the documentation of `Bone2D::apply_rest` ([GH-85503](https://github.com/godotengine/godot/pull/85503)).
- Update the description of the method `get_connection_list` in GraphEdit ([GH-86161](https://github.com/godotengine/godot/pull/86161)).
- Add description for rendering/limits/spatial_indexer/threaded_cull_minimum_instances ([GH-86246](https://github.com/godotengine/godot/pull/86246)).
- Remove pointer to deprecated class page from Skeleton3D ([GH-86326](https://github.com/godotengine/godot/pull/86326)).
- Improve RichTextLabel `install_effect()` documentation ([GH-86331](https://github.com/godotengine/godot/pull/86331)).
- Add note that a large value for `Label.outline_size` is not recommended ([GH-86345](https://github.com/godotengine/godot/pull/86345)).
- Clarify doc: `Node.get_child` returns null for invalid index ([GH-86349](https://github.com/godotengine/godot/pull/86349)).
- Fix various typos in documentation ([GH-86549](https://github.com/godotengine/godot/pull/86549)).
- Fix several reported issues in String's documentation ([GH-86639](https://github.com/godotengine/godot/pull/86639)).
- Correct C# syntax in `_validate_property` example for the Object class ([GH-86649](https://github.com/godotengine/godot/pull/86649)).
- Link to mentioned articles in the docs instead of just naming them ([GH-86666](https://github.com/godotengine/godot/pull/86666)).
- Fix incorrect VehicleWheel3D Roll Influence description ([GH-86672](https://github.com/godotengine/godot/pull/86672)).
- Update `get_image` doc to mention that it will return an empty image with invalid texture ([GH-86685](https://github.com/godotengine/godot/pull/86685)).
- Improve all documentation about MIDI support ([GH-86693](https://github.com/godotengine/godot/pull/86693)).
- Fix various typos in documentation ([GH-86820](https://github.com/godotengine/godot/pull/86820)).
- Specify how CanvasTexture does not work in 3D ([GH-86848](https://github.com/godotengine/godot/pull/86848)).
- Add a few notes to Sprite3D's documentation ([GH-86872](https://github.com/godotengine/godot/pull/86872)).
- Add missing descriptions to TextServer's constants ([GH-86895](https://github.com/godotengine/godot/pull/86895)).
- Add missing documentation for AudioStream & AudioStreamPlayback ([GH-86958](https://github.com/godotengine/godot/pull/86958)).
- Add documentation to EditorExportPlatformWeb ([GH-86977](https://github.com/godotengine/godot/pull/86977)).
- Add missing descriptions for Image's documentation ([GH-86997](https://github.com/godotengine/godot/pull/86997)).
- Add missing descriptions to PrimitiveMesh and SoftBody3D ([GH-87011](https://github.com/godotengine/godot/pull/87011)).
- Add documentation to PhysicsServer2DExtension ([GH-87018](https://github.com/godotengine/godot/pull/87018)).
- Mention `CollisionPolygon2D.polygon` is local to the given CollisionPolygon2D ([GH-87024](https://github.com/godotengine/godot/pull/87024)).
- Add documentation to PhysicsDirectBodyState2DExtension ([GH-87030](https://github.com/godotengine/godot/pull/87030)).
- Add missing descriptions to remaining signals ([GH-87047](https://github.com/godotengine/godot/pull/87047)).
- Overhaul AABB's documentation ([GH-87114](https://github.com/godotengine/godot/pull/87114)).
- Add descriptions to the remaining properties of RDPipelineDepthStencilState ([GH-87156](https://github.com/godotengine/godot/pull/87156)).
- Overhaul Basis' documentation ([GH-87175](https://github.com/godotengine/godot/pull/87175)).
- Include `animation.length` in Animation example ([GH-87180](https://github.com/godotengine/godot/pull/87180)).
- Overhaul Quaternion documentation ([GH-87181](https://github.com/godotengine/godot/pull/87181)).
- Replace some "uncommon" words in class reference ([GH-87223](https://github.com/godotengine/godot/pull/87223)).
- Overhaul Transform3D documentation ([GH-87334](https://github.com/godotengine/godot/pull/87334)).
- Tweak XROrigin3D documentation ([GH-87444](https://github.com/godotengine/godot/pull/87444)).
- Mention and deprecate InputEventJoypadButton's pressure ([GH-87676](https://github.com/godotengine/godot/pull/87676)).
- Minor fix in DisplayServer docs to include Linux & Windows in `FEATURE_NATIVE_DIALOG` ([GH-87913](https://github.com/godotengine/godot/pull/87913)).
- Fix inverted link in docs ([GH-87951](https://github.com/godotengine/godot/pull/87951)).
- Remove duplicate `ARRAY_CUSTOM_RGBA8_UNORM` ref in Mesh docs ([GH-87973](https://github.com/godotengine/godot/pull/87973)).
- Document `OS.execute()` limitations on Android ([GH-87983](https://github.com/godotengine/godot/pull/87983)).
- Fix docs for Color class regarding bits per component ([GH-87985](https://github.com/godotengine/godot/pull/87985)).
- Improve documentation on Android package unique name ([GH-88024](https://github.com/godotengine/godot/pull/88024)).
- Document initial position project settings not affecting run from editor ([GH-88040](https://github.com/godotengine/godot/pull/88040)).
- Document using RichTextLabel's `meta_clicked` to handle clickable URLs ([GH-88255](https://github.com/godotengine/godot/pull/88255)).
- Clarify that LightmapGI is not supported in compatibility renderer ([GH-88431](https://github.com/godotengine/godot/pull/88431)).
- Fix function description for `Font.get_char_size()` ([GH-88444](https://github.com/godotengine/godot/pull/88444)).
- Fix some leftover references to `hint_albedo` in docs ([GH-88503](https://github.com/godotengine/godot/pull/88503)).
- Fix "dimensionnal" typo ([GH-88559](https://github.com/godotengine/godot/pull/88559)).
- C#: Document that MainLoop needs to be registered in the global class ([GH-88635](https://github.com/godotengine/godot/pull/88635)).
- Clarify behavior of opening a new file in `FileAccess` ([GH-88758](https://github.com/godotengine/godot/pull/88758)).
- Doc: Fix GDScript casing of `String.num_scientific` ([GH-88767](https://github.com/godotengine/godot/pull/88767)).
- Document that `parse_input_event` doesn't influence the OS ([GH-88810](https://github.com/godotengine/godot/pull/88810)).
- Add necessary elaboration to documentation for `Node3D::get_parent_node_3d` ([GH-88841](https://github.com/godotengine/godot/pull/88841)).
- Doc: Fix some incorrect uses of "children" ([GH-88920](https://github.com/godotengine/godot/pull/88920)).
- RTL: Clarify that line, paragraph, and character numbers are zero-indexed ([GH-88956](https://github.com/godotengine/godot/pull/88956)).
- Doc: Clarify some details about deferred calls ([GH-88961](https://github.com/godotengine/godot/pull/88961)).
- [C#] Fix typo in `Color` documentation ([GH-89092](https://github.com/godotengine/godot/pull/89092)).
- Address a few issues in Transform3D documentation ([GH-89147](https://github.com/godotengine/godot/pull/89147)).
- Docs: [C#] Use `PropertyName` constants in more places ([GH-89246](https://github.com/godotengine/godot/pull/89246)).
- Doc: Clarify `bsearch(_custom)` behavior ([GH-89280](https://github.com/godotengine/godot/pull/89280)).
- Doc: Clarify description for `get_unix_time_from_system` on UTC ([GH-89454](https://github.com/godotengine/godot/pull/89454)).
- Doc: Clarify behavior of `String.format` with keys in replacements ([GH-89608](https://github.com/godotengine/godot/pull/89608)).
- Fix documentation for receiving light from light probes ([GH-89699](https://github.com/godotengine/godot/pull/89699)).
- Doc: Fix casing of some C# names ([GH-89710](https://github.com/godotengine/godot/pull/89710)).
- Add reference to InputEventJoypadButton in `_shortcut_input` doc ([GH-89838](https://github.com/godotengine/godot/pull/89838)).
- Fix wrong return type mention in `AStarGrid2D` docs ([GH-89862](https://github.com/godotengine/godot/pull/89862)).
- Doc: Document loading behavior with relative paths ([GH-90039](https://github.com/godotengine/godot/pull/90039)).
- Doc: Clarify the behavior of `Vector2/3.cross` and mention parallel vectors ([GH-90072](https://github.com/godotengine/godot/pull/90072)).
- Fix small error in Variant doc ([GH-90088](https://github.com/godotengine/godot/pull/90088)).

#### Editor

- Ignore directory entries in TPZ ([GH-79374](https://github.com/godotengine/godot/pull/79374)).
- Load project metadata file only when needed ([GH-79785](https://github.com/godotengine/godot/pull/79785)).
- Prevent race condition on initial breakpoints from DAP ([GH-84895](https://github.com/godotengine/godot/pull/84895)).
- Do not bother with line colors if `line_number_gutter` is not yet calculated ([GH-84907](https://github.com/godotengine/godot/pull/84907)).
- Don't abort loading when `ext_resource` is missing ([GH-85159](https://github.com/godotengine/godot/pull/85159)).
- Hide Node dock successfully on undo/redo and deletion ([GH-85686](https://github.com/godotengine/godot/pull/85686)).
- Fix "Class name cannot be empty" error when sorting no import files sort by type ([GH-86064](https://github.com/godotengine/godot/pull/86064)).
- Properly select the newly duplicated file ([GH-86164](https://github.com/godotengine/godot/pull/86164)).
- Fix file disappearing when renaming dependencies ([GH-86177](https://github.com/godotengine/godot/pull/86177)).
- Fix duplicating multiple nodes at different depths in `SceneTreeDock` ([GH-86211](https://github.com/godotengine/godot/pull/86211)).
- Optimize scanning routines in the project manager ([GH-86271](https://github.com/godotengine/godot/pull/86271)).
- Clear sub-resources list when no sub-resource exists ([GH-86388](https://github.com/godotengine/godot/pull/86388)).
- Stop escaping `'` on POT generation ([GH-86669](https://github.com/godotengine/godot/pull/86669)).
- Fix missing autocompletion for inheriting classes ([GH-86729](https://github.com/godotengine/godot/pull/86729)).
- Display functions that were previously forgotten in Profiler ([GH-86772](https://github.com/godotengine/godot/pull/86772)).
- Fix editor profiler script function sort order ([GH-87661](https://github.com/godotengine/godot/pull/87661)).
- Fix frame number underflow in visual profiler ([GH-87876](https://github.com/godotengine/godot/pull/87876)).
- Fix skipping normal category followed by custom one ([GH-88318](https://github.com/godotengine/godot/pull/88318)).
- Fix leak of scene used for customization during export ([GH-88726](https://github.com/godotengine/godot/pull/88726)).
- Add missing `variablesReference` field to DAP `evaluate` request ([GH-89110](https://github.com/godotengine/godot/pull/89110)).
- Fix same-name (sub)groups interfering in Inspector ([GH-89631](https://github.com/godotengine/godot/pull/89631)).
- Fix wrong extension filter for dependency editor ([GH-89912](https://github.com/godotengine/godot/pull/89912)).
- Ignore `ERR_FILE_CANT_OPEN` error when loading ([GH-90269](https://github.com/godotengine/godot/pull/90269)).
- Fix duplicated folder reference in Godot Editor after changing filename case ([GH-90280](https://github.com/godotengine/godot/pull/90280)).

#### Export

- Update the validation logic for the package name ([GH-84676](https://github.com/godotengine/godot/pull/84676)).
- [iOS one-click] Add support for Xcode 15 devicectl ([GH-85546](https://github.com/godotengine/godot/pull/85546)).
- Set an appropriate minimum size for labels in windows that display incorrectly ([GH-86145](https://github.com/godotengine/godot/pull/86145)).
- Specify the path to the Java SDK used for the Android gradle build ([GH-86383](https://github.com/godotengine/godot/pull/86383)).
- Add DummyShader handling to Dummy RenderingServer to ensure shader parameters are saved in headless export ([GH-87392](https://github.com/godotengine/godot/pull/87392)).
- Remove workaround in GLTF exporter that double converts `ra` textures to `rg` ([GH-87775](https://github.com/godotengine/godot/pull/87775)).
- Linux: Handle export preset forward compat with 4.3+ platform name ([GH-89047](https://github.com/godotengine/godot/pull/89047)).
- Fix reporting exit code when command line export fails ([GH-89234](https://github.com/godotengine/godot/pull/89234)).
- iOS: Enable Storyboard launch screen by default ([GH-89336](https://github.com/godotengine/godot/pull/89336)).
- Windows: Fix exporting as ZIP when console wrapper and/or embedded PCK is enabled ([GH-89511](https://github.com/godotengine/godot/pull/89511)).
- Android: Add `POST_NOTIFICATIONS` permission to the list of permissions available in the Export dialog ([GH-90377](https://github.com/godotengine/godot/pull/90377)).
- [iOS Export] Fix adding static libs to the Xcode project ([GH-90379](https://github.com/godotengine/godot/pull/90379)).

#### GDExtension

- Fix overriding `CollisionObject3D::_mouse_enter()` and `_mouse_exit()` from GDExtension ([GH-85870](https://github.com/godotengine/godot/pull/85870)).
- Fix operator documentation in GDExtension API dump with docs ([GH-86087](https://github.com/godotengine/godot/pull/86087)).
- Replace `GDVIRTUAL_CALL` with `GDVIRTUAL_REQUIRED_CALL` where applicable ([GH-86169](https://github.com/godotengine/godot/pull/86169)).
- Correctly register editor-only module classes with the API ([GH-86209](https://github.com/godotengine/godot/pull/86209)).
- Editor: Add missing virtual bind to `EditorNode3DGizmo(Plugin)` ([GH-86881](https://github.com/godotengine/godot/pull/86881)).
- Fix virtual calls for GDExtension in `CollisionObject2D` ([GH-86908](https://github.com/godotengine/godot/pull/86908)).
- Expose `NOTIFICATION_EXTENSION_RELOADED` to `ClassDB` ([GH-87863](https://github.com/godotengine/godot/pull/87863)).
- Add null check for GDExtension deinitialization ([GH-87938](https://github.com/godotengine/godot/pull/87938)).
- Fix `gdextension_compat_hashes.cpp` for double precision builds ([GH-88188](https://github.com/godotengine/godot/pull/88188)).
- Fix `Resource::get_rid override` not working in GDExtension ([GH-90273](https://github.com/godotengine/godot/pull/90273)).

#### GDScript

- Fix DAP breakpoints being cleared on closed scripts ([GH-84898](https://github.com/godotengine/godot/pull/84898)).
- Speed up `GDScript::get_must_clear_dependencies()` ([GH-85603](https://github.com/godotengine/godot/pull/85603)).
- Make GDScriptAnalyzer aware of properties from other languages ([GH-85703](https://github.com/godotengine/godot/pull/85703)).
- Fix accessing static function as `Callable` in static context ([GH-86088](https://github.com/godotengine/godot/pull/86088)).
- Fix POT generator skips some nodes ([GH-86091](https://github.com/godotengine/godot/pull/86091)).
- Fix regression when autocompleting subscript on get node ([GH-86111](https://github.com/godotengine/godot/pull/86111)).
- Fix the autocomplete function for the `self` keyword ([GH-86341](https://github.com/godotengine/godot/pull/86341)).
- Fix POT generator skips some nodes (part 2) ([GH-86471](https://github.com/godotengine/godot/pull/86471)).
- Improve sorting of enum autocompletion ([GH-86667](https://github.com/godotengine/godot/pull/86667)).
- Lambda hotswap fixes ([GH-86860](https://github.com/godotengine/godot/pull/86860)).
- Prevent running `String` number functions on invalid literal ([GH-87941](https://github.com/godotengine/godot/pull/87941)).
- Allow LSP to process multiple messages per poll ([GH-89284](https://github.com/godotengine/godot/pull/89284)).
- Update `@GDScript` documentation ([GH-89484](https://github.com/godotengine/godot/pull/89484)).

#### GUI

- Trigger zoom from pan gestures when pressing ctrl ([GH-80994](https://github.com/godotengine/godot/pull/80994)).
- Fix opening docs writing extra navigation history ([GH-82498](https://github.com/godotengine/godot/pull/82498)).
- Fix dock visibility issues ([GH-84122](https://github.com/godotengine/godot/pull/84122)).
- RichTextLabel: Fix `remove_paragraph` crash by popping current ([GH-84312](https://github.com/godotengine/godot/pull/84312)).
- Fix crash on hiding grandparent Control on mouse exit ([GH-85313](https://github.com/godotengine/godot/pull/85313)).
- Ensure slider grabs focus only when it can ([GH-85652](https://github.com/godotengine/godot/pull/85652)).
- Fix unnecessarily quantizing current color in color picker ([GH-85749](https://github.com/godotengine/godot/pull/85749)).
- Correctly enforce minimum window size in editor ([GH-85887](https://github.com/godotengine/godot/pull/85887)).
- Fix theme access in the Groups editor ([GH-86031](https://github.com/godotengine/godot/pull/86031)).
- Fix `ColorPicker`'s alpha slider arrow offset ([GH-86034](https://github.com/godotengine/godot/pull/86034)).
- Fix `MenuBar` and `MenuButton` hover position scaling properly with the scale factor multiplier ([GH-86304](https://github.com/godotengine/godot/pull/86304)).
- Fix double `text_changed` signal when overwriting selection in LineEdit ([GH-86460](https://github.com/godotengine/godot/pull/86460)).
- Fix D&D viewport position calculation ([GH-86511](https://github.com/godotengine/godot/pull/86511)).
- Redraw `TreeItem` on more changes ([GH-87415](https://github.com/godotengine/godot/pull/87415)).
- macOS: Fix changing main menu item names ([GH-87912](https://github.com/godotengine/godot/pull/87912)).
- Only recurse depth wise in `Tree::_count_selected_items` ([GH-87943](https://github.com/godotengine/godot/pull/87943)).
- Fix `Slider`'s mouse drag position when grabber is centered ([GH-88068](https://github.com/godotengine/godot/pull/88068)).
- Fix `TabBar` size when theme changes ([GH-88293](https://github.com/godotengine/godot/pull/88293)).
- Editor: Improve clarity and style of `ResourcePicker` menu ([GH-88435](https://github.com/godotengine/godot/pull/88435)).
- Editor: Add missing ellipses to menu options that open dialogs ([GH-88436](https://github.com/godotengine/godot/pull/88436)).
- Fix crash when selecting re-added `TreeItem::Cell` ([GH-88917](https://github.com/godotengine/godot/pull/88917)).
- RTL: Fix meta hover area detection ([GH-89158](https://github.com/godotengine/godot/pull/89158)).
- Fix a pixel misalignment in the blue robot logo ([GH-89711](https://github.com/godotengine/godot/pull/89711)).
- Correct FileDialog Theme overrides ([GH-89845](https://github.com/godotengine/godot/pull/89845)).

#### Import

- Fix Scene Importer crashing when animation or mesh save paths are invalid ([GH-83856](https://github.com/godotengine/godot/pull/83856)).
- Replace `//` with `\\` before sending path to Blender ([GH-85335](https://github.com/godotengine/godot/pull/85335)).
- Added proper timeout for blender rpc connection ([GH-85519](https://github.com/godotengine/godot/pull/85519)).
- Fix squish RGTC_R decompression corruption ([GH-85863](https://github.com/godotengine/godot/pull/85863)).
- Prevent overriding file info of another file when reimport creates extra files ([GH-85922](https://github.com/godotengine/godot/pull/85922)).
- Fix `squish` DXT5 RA-As-RG channel swapping ([GH-85967](https://github.com/godotengine/godot/pull/85967)).
- Support unspecified linear size in DDS files ([GH-86336](https://github.com/godotengine/godot/pull/86336)).
- Add obj importer changes to use ImporterMesh ([GH-86365](https://github.com/godotengine/godot/pull/86365)).
- GLTF: Fix three bugs which prevented extracted textures from being refreshed ([GH-86504](https://github.com/godotengine/godot/pull/86504)).
- Fix BasisUniversal ETC RA as RG transcoding ([GH-86916](https://github.com/godotengine/godot/pull/86916)).
- Allow configuring the maximum width for atlas import ([GH-87145](https://github.com/godotengine/godot/pull/87145)).
- Fix crash when previewing a scene with a mesh as the root node ([GH-87782](https://github.com/godotengine/godot/pull/87782)).
- Fix crash when importing a GLTF file with a skeleton as the root ([GH-87933](https://github.com/godotengine/godot/pull/87933)).
- Fix GLTF exporting invalid meshes and attempting to export gizmo meshes ([GH-87934](https://github.com/godotengine/godot/pull/87934)).
- Properly calculate binormal when creating SurfaceTool from arrays ([GH-88725](https://github.com/godotengine/godot/pull/88725)).
- Multiple fixes for compressed meshes ([GH-88738](https://github.com/godotengine/godot/pull/88738)).
- Fix wrong indexing when generating dummy tangents in GLTF import ([GH-88931](https://github.com/godotengine/godot/pull/88931)).
- Add `--import` command-line flag ([GH-90431](https://github.com/godotengine/godot/pull/90431)).

#### Input

- Ensure the active window gains the keyboard focus ([GH-80548](https://github.com/godotengine/godot/pull/80548)).
- Prevent escape key from closing Editor Settings window when filtering for shortcuts ([GH-86654](https://github.com/godotengine/godot/pull/86654)).
- Fix global position for `InputEventMouse` in `viewport::push_input` ([GH-88473](https://github.com/godotengine/godot/pull/88473)).
- Fix description of touch input position ([GH-89509](https://github.com/godotengine/godot/pull/89509)).

#### Multiplayer

- Fix `complete_auth` notifying the wrong peer ([GH-86257](https://github.com/godotengine/godot/pull/86257)).
- Fix auth not waiting for confirmation in some cases ([GH-86260](https://github.com/godotengine/godot/pull/86260)).
- Fix spawned nodes not working after reset ([GH-87185](https://github.com/godotengine/godot/pull/87185)).
- Fix remote net ID cleanup ([GH-87186](https://github.com/godotengine/godot/pull/87186)).
- Handle cleanup of "scene cache" nodes ([GH-87190](https://github.com/godotengine/godot/pull/87190)).
- Networking scene multiplayer: Fix removing connected peer during disconnection ([GH-88826](https://github.com/godotengine/godot/pull/88826)).
- Fix node config warning not updating for `Multiplayer{Spawner,Synchronizer}` ([GH-89839](https://github.com/godotengine/godot/pull/89839)).
- Fix dead code doing unnecessary allocation ([GH-90315](https://github.com/godotengine/godot/pull/90315)).

#### Navigation

- Fix property hints for parsed collision mask ([GH-88156](https://github.com/godotengine/godot/pull/88156)).
- Fix `NavigationServer.set_debug_enabled()` doing nothing ([GH-90200](https://github.com/godotengine/godot/pull/90200)).

#### Network

- enet: Sync with upstream commit c44b7d0 ([GH-90244](https://github.com/godotengine/godot/pull/90244)).
- Fix missing return in `StreamPeerTCP::poll` when connection is `STATUS_CONNECTED` ([GH-90471](https://github.com/godotengine/godot/pull/90741)).

#### Particles

- Only update particle velocity when it changes ([GH-86474](https://github.com/godotengine/godot/pull/86474)).
- Fix early activation of particle trail sections ([GH-89042](https://github.com/godotengine/godot/pull/89042)).

#### Physics

- Fix body leaving area gravity influence ([GH-82961](https://github.com/godotengine/godot/pull/82961)).
- Fix CollisionObject3D Gizmo not updated after calling `shape_owner_*` functions ([GH-84610](https://github.com/godotengine/godot/pull/84610)).
- Fix `SoftBody3D` for double-precision builds ([GH-88402](https://github.com/godotengine/godot/pull/88402)).
- Allow for 32 max collisions in `test_body_motion` ([GH-89517](https://github.com/godotengine/godot/pull/89517)).
- Fix separating axes for 3D cylinder-face collisions ([GH-89960](https://github.com/godotengine/godot/pull/89960)).

#### Plugin

- Fix creating and updating plugin with dot in folder name ([GH-83329](https://github.com/godotengine/godot/pull/83329)).
- Editor: Fix `_parse_category()` is not called for custom categories ([GH-87915](https://github.com/godotengine/godot/pull/87915)).

#### Porting

- Fix NetBSD executable path ([GH-84469](https://github.com/godotengine/godot/pull/84469)).
- Make screen_get_refresh_rate() respect iOS Low Power Mode ([GH-85026](https://github.com/godotengine/godot/pull/85026)).
- Fix key mapping for `XK_KP_Delete` key ([GH-86160](https://github.com/godotengine/godot/pull/86160)).
- X11: Fix Godot stealing focus on alternative window managers ([GH-86441](https://github.com/godotengine/godot/pull/86441)).
- Fix `OS.get_system_font_path` and `OS.get_system_font_path_for_text` to return correct slashes ([GH-86552](https://github.com/godotengine/godot/pull/86552)).
- Fix virtual keyboard for decimal values on Android ([GH-86619](https://github.com/godotengine/godot/pull/86619)).
- X11: Don't re-set input focus if the given window already has it (fixes Godot stealing input focus on i3) ([GH-86671](https://github.com/godotengine/godot/pull/86671)).
- iOS: Set provisioning style for both `iPhone Developer` and `iPhone Distribution` to automatic ([GH-86748](https://github.com/godotengine/godot/pull/86748)).
- Fix `get_window_safe_area` on Android ([GH-86761](https://github.com/godotengine/godot/pull/86761)).
- Disable automatic permissions request ([GH-87080](https://github.com/godotengine/godot/pull/87080)).
- macOS: Update window visible state on deminiaturize ([GH-87465](https://github.com/godotengine/godot/pull/87465)).
- Make dark mode Title Bar work on Windows 10 1909 (build:18363) and above ([GH-87549](https://github.com/godotengine/godot/pull/87549)).
- Add workaround for emscripten >= 3.1.47 LTO build ([GH-87956](https://github.com/godotengine/godot/pull/87956)).
- macOS: Enabled secure restorable state ([GH-88050](https://github.com/godotengine/godot/pull/88050)).
- macOS: Allow `open_shell` to handle filenames without `file://` ([GH-88126](https://github.com/godotengine/godot/pull/88126)).
- Windows: Fix windows `is_path_invalid`, and apply it to directory creation ([GH-88129](https://github.com/godotengine/godot/pull/88129)).
- Fix the fetching of images in `CF_DIB` format in `DisplayServerWindows::clipboard_get_image` ([GH-88220](https://github.com/godotengine/godot/pull/88220)).
- macOS: Fix color picker on HDR screens ([GH-88274](https://github.com/godotengine/godot/pull/88274)).
- [Android 14] Fix GodotEditText white box showing during editor load ([GH-88351](https://github.com/godotengine/godot/pull/88351)).
- Windows: Disable fallback to ANGLE logic when compiled w/o ANGLE support ([GH-89351](https://github.com/godotengine/godot/pull/89351)).
- Fix platform name in the message about unsupported CPU architecture ([GH-89598](https://github.com/godotengine/godot/pull/89598)).
- Fix issue with moving maximized window in macOS ([GH-90101](https://github.com/godotengine/godot/pull/90101)).
- Fix macOS menu bar & dock stop appearing after closing sub-window ([GH-90131](https://github.com/godotengine/godot/pull/90131)).
- Make sysctl calls on FreeBSD ([GH-90295](https://github.com/godotengine/godot/pull/90295)).

#### Rendering

- Make the rendering method dropdown also affect mobile if compatible ([GH-72461](https://github.com/godotengine/godot/pull/72461)).
- Add thread guard for `force_draw` and update related documentation ([GH-82953](https://github.com/godotengine/godot/pull/82953)).
- Transform mesh's AABB to skeleton's space when calculating mesh's bounds ([GH-84451](https://github.com/godotengine/godot/pull/84451)).
- Fix Camera2D frame delay (port from 3.x) ([GH-84465](https://github.com/godotengine/godot/pull/84465)).
- Only copy the relevant portion of the screen when copying to backbuffer in Compatibility backend ([GH-84733](https://github.com/godotengine/godot/pull/84733)).
- Store ArrayMesh path in RenderingServer for use in error messages ([GH-84894](https://github.com/godotengine/godot/pull/84894)).
- Remove GI methods in parentheses from light baking options ([GH-85219](https://github.com/godotengine/godot/pull/85219)).
- Force ANGLE on all pre GCN 4th gen. AMD/ATI GPUs ([GH-85273](https://github.com/godotengine/godot/pull/85273)).
- Fix invalid `frame` index when Sprite2D's `hframes` or `vframes` change ([GH-85317](https://github.com/godotengine/godot/pull/85317)).
- Use render method from OS instead of project settings in compositor RD ([GH-85387](https://github.com/godotengine/godot/pull/85387)).
- Avoid crashes when engine leaks canvas items and friends ([GH-85520](https://github.com/godotengine/godot/pull/85520)).
- Apply some low-hanging fruit optimizations to Vulkan RD ([GH-85532](https://github.com/godotengine/godot/pull/85532)).
- Add wireframe for compatibility mode ([GH-85621](https://github.com/godotengine/godot/pull/85621)).
- Expose `copy_effects` compute shader in Mobile backend ([GH-85793](https://github.com/godotengine/godot/pull/85793)).
- Fix CanvasOcclusionShaderRD format error with double precision build ([GH-85822](https://github.com/godotengine/godot/pull/85822)).
- Windows: Always use ANGLE in ARM builds ([GH-86001](https://github.com/godotengine/godot/pull/86001)).
- Fix radiance for sky in GLES stereo rendering ([GH-86018](https://github.com/godotengine/godot/pull/86018)).
- Fix Volumetric Fog VoxelGI updates ([GH-86023](https://github.com/godotengine/godot/pull/86023)).
- Fix LightmapperRD division warning in MSVC ([GH-86555](https://github.com/godotengine/godot/pull/86555)).
- Fix Polygon2D to Skeleton2D transform calculation ([GH-86557](https://github.com/godotengine/godot/pull/86557)).
- Implement overdraw, lighting, and unshaded debug draw modes for OpenGL ([GH-86677](https://github.com/godotengine/godot/pull/86677)).
- Fix global transform being wrong on entering tree ([GH-86841](https://github.com/godotengine/godot/pull/86841)).
- Fix SSR not working properly in stereo ([GH-86996](https://github.com/godotengine/godot/pull/86996)).
- Add `shader_cache_dir_valid` check to `_save_to_cache` ([GH-87096](https://github.com/godotengine/godot/pull/87096)).
- Fix 2D normals for transposed texture ([GH-87225](https://github.com/godotengine/godot/pull/87225)).
- Disable scissor test after rendering batches in compatibility renderer ([GH-87489](https://github.com/godotengine/godot/pull/87489)).
- Significantly improve the speed of shader compilation in compatibility backend ([GH-87553](https://github.com/godotengine/godot/pull/87553)).
- Free dummy renderer objects ([GH-87710](https://github.com/godotengine/godot/pull/87710)).
- Do not reflect the origin lines in a mirror ([GH-87757](https://github.com/godotengine/godot/pull/87757)).
- Fix missing instance type in dummy renderer ([GH-88097](https://github.com/godotengine/godot/pull/88097)).
- Make `RID_Owner<Texture>` threadsafe in `TextureStorage` for GLES3 ([GH-88205](https://github.com/godotengine/godot/pull/88205)).
- Disable ReShade in the editor and project manager (if run via Vulkan) ([GH-88316](https://github.com/godotengine/godot/pull/88316)).
- Make dummy rendering server appear as a high end platform to fix vulkan shader compile error when exporting ([GH-88409](https://github.com/godotengine/godot/pull/88409)).
- Fix shader cache with transform feedback on some Android devices ([GH-88573](https://github.com/godotengine/godot/pull/88573)).
- Fail early if shader mode is invalid in dummy renderer ([GH-88581](https://github.com/godotengine/godot/pull/88581)).
- Add fix for TAA passes rendering black meshes on XR ([GH-88830](https://github.com/godotengine/godot/pull/88830)).
- Make Overdraw, Lighting and Shadow Splits debug draw modes ignore decals ([GH-89253](https://github.com/godotengine/godot/pull/89253)).
- Fix missed light clusters when inside clipped lights ([GH-89450](https://github.com/godotengine/godot/pull/89450)).
- Fix mobile renderer RID leaks ([GH-89531](https://github.com/godotengine/godot/pull/89531)).
- TileMap: Fix forcing cleanup on exiting tree/canvas ([GH-90012](https://github.com/godotengine/godot/pull/90012)).
- Allow Decal Emission Energy values above 128 in the inspector ([GH-90217](https://github.com/godotengine/godot/pull/90217)).

#### Shaders

- Fix visual shader's `screen_uv` input preview uses position of node rather than a sample area like uv ([GH-84348](https://github.com/godotengine/godot/pull/84348)).
- Check if the ref shader is valid in visual shader's `_update_option_menu` ([GH-87356](https://github.com/godotengine/godot/pull/87356)).
- Fully initialize all members of structs `IdentifierActions`, `GeneratedCode` and `DefaultIdentifierActions` ([GH-88021](https://github.com/godotengine/godot/pull/88021)).
- Change shader compiler default setting to avoid doctool error ([GH-88996](https://github.com/godotengine/godot/pull/88996)).

#### XR

- Fix crash when using OpenXR extension wrappers from GDExtension ([GH-88689](https://github.com/godotengine/godot/pull/88689)).
- Improve warning when XR shaders are not enabled ([GH-89397](https://github.com/godotengine/godot/pull/89397)).

#### Thirdparty

- Sync controller mappings DB with SDL2 community repo ([GH-90406](https://github.com/godotengine/godot/pull/90406)).
- basis_universal: Unbundle jpgd to fix symbol conflict, use our newer copy with SSE2 support ([GH-88508](https://github.com/godotengine/godot/pull/88508)).
- certs: Sync with Mozilla bundle as of Mar 11, 2024 ([GH-90211](https://github.com/godotengine/godot/pull/90211)).
- libpng: Update to 1.6.43 ([GH-89314](https://github.com/godotengine/godot/pull/89314)).
- mbedtls: Update to upstream version 2.28.8 ([GH-90209](https://github.com/godotengine/godot/pull/90209)).
- miniupnpc: Update to version 2.2.6 ([GH-88285](https://github.com/godotengine/godot/pull/88285)).
- ThorVG: Update to 0.12.10 ([GH-90243](https://github.com/godotengine/godot/pull/90243)).
- tinyexr: Update to 1.0.8 ([GH-88702](https://github.com/godotengine/godot/pull/88702)).
- zlib/minizip: Update to version 1.3.1 ([GH-87527](https://github.com/godotengine/godot/pull/87527)).


## 4.2.1 - 2023-12-12

- [Release announcement](https://godotengine.org/article/maintenance-release-godot-4-2-1)
- [Interactive changelog](https://godotengine.github.io/godot-interactive-changelog/#4.2.1)

#### 2D

- Fix UV editor not using texture transform ([GH-84076](https://github.com/godotengine/godot/pull/84076)).
- Fix generating terrain icon with certain image formats ([GH-84507](https://github.com/godotengine/godot/pull/84507)).
- Keep scene tiles even if the TileMap is invisible ([GH-85753](https://github.com/godotengine/godot/pull/85753)).
- Fix TileMap occluders ([GH-85893](https://github.com/godotengine/godot/pull/85893)).

#### 3D

- Only allow MeshInstance3D-inherited nodes in MultiMesh Populate Surface dialog ([GH-84933](https://github.com/godotengine/godot/pull/84933)).

#### Animation

- Fix imported track flag on sliced animations ([GH-85061](https://github.com/godotengine/godot/pull/85061)).
- Prevent a crash when calling `AnimationMixer::restore` with an invalid resource ([GH-85428](https://github.com/godotengine/godot/pull/85428)).
- Fix AnimationPlayer seeking for Discrete keys ([GH-85569](https://github.com/godotengine/godot/pull/85569)).
- Fix Tween loop initial value ([GH-85681](https://github.com/godotengine/godot/pull/85681)).

#### Audio

- Fix importing WAV files with odd chunk sizes ([GH-85556](https://github.com/godotengine/godot/pull/85556)).

#### Buildsystem

- Use Python venv if detected when building VS project ([GH-84593](https://github.com/godotengine/godot/pull/84593)).
- Fix the Web platform team's codeowners link ([GH-85746](https://github.com/godotengine/godot/pull/85746)).
- Fix invalid Python escape sequences ([GH-85818](https://github.com/godotengine/godot/pull/85818)).

#### Core

- Set language encoding flag when using `ZIPPacker` ([GH-78732](https://github.com/godotengine/godot/pull/78732)).
- Fix crash when hashing empty `CharString` ([GH-85389](https://github.com/godotengine/godot/pull/85389)).
- Prevent infinite recursion when printing errors ([GH-85397](https://github.com/godotengine/godot/pull/85397)).
- Fix property groups overriding real properties ([GH-85486](https://github.com/godotengine/godot/pull/85486)).
- Do not reload resources and send notification if locale is not changed ([GH-85787](https://github.com/godotengine/godot/pull/85787)).

#### Documentation

- Improve and clarify texture filtering documentation ([GH-83907](https://github.com/godotengine/godot/pull/83907)).
- Fix documentation for `icon_and_font_color` editor setting ([GH-85491](https://github.com/godotengine/godot/pull/85491)).
- Improve documentation for `CameraAttributesPhysical.exposure_shutter_speed` ([GH-85599](https://github.com/godotengine/godot/pull/85599)).
- Fix missing heading in translated online class reference ([GH-85877](https://github.com/godotengine/godot/pull/85877)).

#### Editor

- Remove exp hint of a few properties ([GH-80326](https://github.com/godotengine/godot/pull/80326)).
- Fix UV editor not showing polygon correctly ([GH-84116](https://github.com/godotengine/godot/pull/84116)).
- Inspector: Fix clearing array/dictionary element with `<Object#null>` ([GH-84237](https://github.com/godotengine/godot/pull/84237)).
- Allow dragging editable children ([GH-84310](https://github.com/godotengine/godot/pull/84310)).
- Fix errors on file rename or move in the Filesystem Dock ([GH-84520](https://github.com/godotengine/godot/pull/84520)).
- Fix issue with 3D scene drag and drop preview node ([GH-85087](https://github.com/godotengine/godot/pull/85087)).
- Fix SnapGrid is almost invisible in light theme ([GH-85585](https://github.com/godotengine/godot/pull/85585)).
- Fix theme application in various editor dialogs ([GH-85745](https://github.com/godotengine/godot/pull/85745)).

#### Export

- Fix order of operations for macOS template check ([GH-84990](https://github.com/godotengine/godot/pull/84990)).
- iOS: Use `mdfind` to check if Xcode is installed in one-click deploy code ([GH-85774](https://github.com/godotengine/godot/pull/85774)).

#### GDExtension

- Fix updating cached singletons when reloading GDScripts ([GH-85373](https://github.com/godotengine/godot/pull/85373)).
- Fix crash when using incompatible versions of Godot Jolt ([GH-85779](https://github.com/godotengine/godot/pull/85779)).

#### GDScript

- Improve autocompletion with `get_node` ([GH-79386](https://github.com/godotengine/godot/pull/79386)).
- Filter groups and categories from autocompletion ([GH-85196](https://github.com/godotengine/godot/pull/85196)).

#### GUI

- Enable scrolling of output with UI scale changes ([GH-82079](https://github.com/godotengine/godot/pull/82079)).
- VideoPlayer: Fix reloading translation remapped stream ([GH-84794](https://github.com/godotengine/godot/pull/84794)).
- Restored Control properties when you undo a parenting of a Control to a Container ([GH-85181](https://github.com/godotengine/godot/pull/85181)).
- Make sure `Window`'s title is respected before we compute the size ([GH-85312](https://github.com/godotengine/godot/pull/85312)).
- RTL: Fix CharFX character offset calculation ([GH-85363](https://github.com/godotengine/godot/pull/85363)).
- Limit window size updates on title change ([GH-85542](https://github.com/godotengine/godot/pull/85542)).
- Fix size and visuals of the `InputEventConfigurationDialog` ([GH-85790](https://github.com/godotengine/godot/pull/85790)).
- Limit window size updates on title translation change ([GH-85828](https://github.com/godotengine/godot/pull/85828)).

#### Import

- Fix memory leak on error paths in tinyexr loader ([GH-85002](https://github.com/godotengine/godot/pull/85002)).
- Fix memory corruption and assert failures in convex decomposition ([GH-85631](https://github.com/godotengine/godot/pull/85631)).

#### Input

- X11: Send IME update notification deferred ([GH-85306](https://github.com/godotengine/godot/pull/85306)).
- Fix IME key event being erased in macOS ([GH-85458](https://github.com/godotengine/godot/pull/85458)).
- Fix SubViewport physics picking ([GH-85665](https://github.com/godotengine/godot/pull/85665)).

#### Navigation

- Fix missing NavigationLink property updates in constructor ([GH-83802](https://github.com/godotengine/godot/pull/83802)).
- Fix missing NavigationRegion property updates in constructor ([GH-83812](https://github.com/godotengine/godot/pull/83812)).
- Fix missing NavigationAgent property updates in constructor ([GH-83814](https://github.com/godotengine/godot/pull/83814)).
- Fix missing NavigationObstacle property updates in constructor ([GH-83816](https://github.com/godotengine/godot/pull/83816)).
- Fix memory leak in 'NavigationServer3D' involving static obstacles ([GH-84816](https://github.com/godotengine/godot/pull/84816)).
- Fix NavigationRegion2D transform update ([GH-85258](https://github.com/godotengine/godot/pull/85258)).

#### Particles

- Only allow MeshInstance3D-based nodes in particles emission shape node selector ([GH-84891](https://github.com/godotengine/godot/pull/84891)).

#### Plugin

- Correctly check scripts that must inherit `EditorPlugin` ([GH-85271](https://github.com/godotengine/godot/pull/85271)).

#### Porting

- Do not consume mouse messages in windows with `no_focus` on Windows OS ([GH-85484](https://github.com/godotengine/godot/pull/85484)).
- Set what were default values for Web platform linker flags `-sSTACK_SIZE` and `-sDEFAULT_PTHREAD_STACK_SIZE` ([GH-86036](https://github.com/godotengine/godot/pull/86036)).

#### Rendering

- Fix buffer updates going to the wrong cmd buffer if barriers were 0 ([GH-83736](https://github.com/godotengine/godot/pull/83736)).
- Fix bad parameter for `rendering_method` crashes Godot ([GH-84241](https://github.com/godotengine/godot/pull/84241)).
- Add `shadows_disabled` macro in Compatibility renderer ([GH-84416](https://github.com/godotengine/godot/pull/84416)).
- Vulkan: Fix incorrect access to the buffers on Android ([GH-84852](https://github.com/godotengine/godot/pull/84852)).
- Use vertex input mask for creating vertex arrays ([GH-85092](https://github.com/godotengine/godot/pull/85092)).
- Fix typo in BaseMaterial3D conversion from 3.x SpatialMaterial ([GH-85269](https://github.com/godotengine/godot/pull/85269)).
- Set ReflectionProbe frame before mapping id in mobile renderer ([GH-85635](https://github.com/godotengine/godot/pull/85635)).
- Add a descriptive error message when creating a mesh surface from the wrong array type ([GH-85646](https://github.com/godotengine/godot/pull/85646)).
- GLES3: Skip batches with zero instance count while rendering canvas ([GH-85778](https://github.com/godotengine/godot/pull/85778)).
- macOS: Switch ANGLE backend to ANGLE over OpenGL, switch default compatibility renderer back to native ([GH-85785](https://github.com/godotengine/godot/pull/85785)).
- Ensure that 2D meshes use a proper input mask ([GH-85972](https://github.com/godotengine/godot/pull/85972)).

#### Shaders

- Automatically ensure correct normals in Compatibility renderer ([GH-82804](https://github.com/godotengine/godot/pull/82804)).
- Comment the shader template light function by default ([GH-84594](https://github.com/godotengine/godot/pull/84594)).

#### XR

- Remove unused grip touch action from default OpenXR action map ([GH-85048](https://github.com/godotengine/godot/pull/85048)).


## 4.2 - 2023-11-30

- [Release announcement](https://godotengine.org/article/godot-4-2-arrives-in-style)
- [Migration guide](https://docs.godotengine.org/en/latest/tutorials/migrating/upgrading_to_godot_4.2.html)
- [Interactive changelog](https://godotengine.github.io/godot-interactive-changelog/#4.2)
- [Breaking changes](https://github.com/godotengine/godot/pulls?q=is%3Apr+is%3Amerged+label%3A%22breaks+compat%22+milestone%3A4.2)

#### 2D

- Greatly improve Y-sort performance on TileMaps ([GH-73813](https://github.com/godotengine/godot/pull/73813)).
- Add separate editor plugin for TileMap and TileSet ([GH-74717](https://github.com/godotengine/godot/pull/74717)).
- Cleanup tiles outside the texture ([GH-77986](https://github.com/godotengine/godot/pull/77986)).
- Move TileMap layers to their own class ([GH-78328](https://github.com/godotengine/godot/pull/78328)).
- Add option to swap default Alt+scroll zooming behavior in 2D editor ([GH-78451](https://github.com/godotengine/godot/pull/78451)).
- Add white rect to TileMap selection tool ([GH-78519](https://github.com/godotengine/godot/pull/78519)).
- Improve string drawing in the tiledata editor ([GH-78522](https://github.com/godotengine/godot/pull/78522)).
- Make sure the shortcut key respects the context in `TileSetAtlasSourceEditor` ([GH-78920](https://github.com/godotengine/godot/pull/78920)).
- Fix `Camera2D.rotating` not being converted and reversed properly ([GH-79264](https://github.com/godotengine/godot/pull/79264)).
- Streamline creating tile atlas sources ([GH-79285](https://github.com/godotengine/godot/pull/79285)).
- Rework modifying tile source ID ([GH-79419](https://github.com/godotengine/godot/pull/79419)).
- Allow using floating-point bone sizes and outline widths in the 2D editor ([GH-79434](https://github.com/godotengine/godot/pull/79434)).
- Add option to expand tile polygon editors ([GH-79512](https://github.com/godotengine/godot/pull/79512)).
- Add `is_conformal` method to Basis and Transform2D ([GH-79523](https://github.com/godotengine/godot/pull/79523)).
- Improve message when no tile is selected to edit ([GH-79562](https://github.com/godotengine/godot/pull/79562)).
- Fix crash when deleting tileset terrains ([GH-79618](https://github.com/godotengine/godot/pull/79618)).
- Fix Camera2D crash when edited scene root is null ([GH-79645](https://github.com/godotengine/godot/pull/79645)).
- Auto create tile for multiple atlases ([GH-79678](https://github.com/godotengine/godot/pull/79678)).
- Fix `CanvasModulate` logic for modulating the canvas ([GH-79747](https://github.com/godotengine/godot/pull/79747)).
- Fix `get_cursor_shape()` in tile atlas editor ([GH-79837](https://github.com/godotengine/godot/pull/79837)).
- Fix crash when executing `TileMap.fix_invalid_tiles` ([GH-79851](https://github.com/godotengine/godot/pull/79851)).
- Improve atlas tile size dragging ([GH-79899](https://github.com/godotengine/godot/pull/79899)).
- Add help label about creating multiple/big tiles ([GH-79904](https://github.com/godotengine/godot/pull/79904)).
- Properly clear scene tiles ([GH-79941](https://github.com/godotengine/godot/pull/79941)).
- Edit TileSet source on double click ([GH-80037](https://github.com/godotengine/godot/pull/80037)).
- Fix "a number is required" error when printing RID ([GH-80122](https://github.com/godotengine/godot/pull/80122)).
- Ignore null "id" in tile source proxy ([GH-80135](https://github.com/godotengine/godot/pull/80135)).
- Add per-tile flipping and transposing ([GH-80144](https://github.com/godotengine/godot/pull/80144)).
- Fix multiple usability issues in the texture region editor ([GH-80435](https://github.com/godotengine/godot/pull/80435)).
- Fix TileSet with TileMap handling ([GH-80462](https://github.com/godotengine/godot/pull/80462)).
- Fix TileSet not disappearing on deselecting TileMap ([GH-80529](https://github.com/godotengine/godot/pull/80529)).
- TileMap: Check for possible scenes to be erased ([GH-80658](https://github.com/godotengine/godot/pull/80658)).
- Pass missing arguments to `TileMap::get_used_cells_by_id` ([GH-80729](https://github.com/godotengine/godot/pull/80729)).
- Improve scene tiles workflow ([GH-80754](https://github.com/godotengine/godot/pull/80754)).
- Simplify making texture nodes in 2D editor ([GH-80771](https://github.com/godotengine/godot/pull/80771)).
- Add `px` suffix for TileSet `separation` property ([GH-80934](https://github.com/godotengine/godot/pull/80934)).
- Convert TileSet Atlas Merge input images to RGBA8 to match output, if needed ([GH-80943](https://github.com/godotengine/godot/pull/80943)).
- Call `add_child` after `set_rect` to fix size bug ([GH-80968](https://github.com/godotengine/godot/pull/80968)).
- Added checks to remove meta arrays when creating and undoing guides ([GH-81011](https://github.com/godotengine/godot/pull/81011)).
- Improve TileMap performances by using quadrants only for rendering ([GH-81070](https://github.com/godotengine/godot/pull/81070)).
- Allow configuring primary line X/Ys separately ([GH-81255](https://github.com/godotengine/godot/pull/81255)).
- Fix `TileMap::get_used_rect` incorrectly handling empty layers ([GH-81423](https://github.com/godotengine/godot/pull/81423)).
- Fix rotated 2D movement gizmo ([GH-81735](https://github.com/godotengine/godot/pull/81735)).
- Incorporate min and max zoom limits into the EditorZoomWidget ([GH-81812](https://github.com/godotengine/godot/pull/81812)).
- Fix TileMap editor so that pressing control deselects cells correctly ([GH-81925](https://github.com/godotengine/godot/pull/81925)).
- Don't allow transforming scene tiles ([GH-81971](https://github.com/godotengine/godot/pull/81971)).
- Fix animated tile time-slice calculation accumulating float errors ([GH-82360](https://github.com/godotengine/godot/pull/82360)).
- Fix transform calculations for drag-moving CanvasItems in editor ([GH-82667](https://github.com/godotengine/godot/pull/82667)).
- Prioritize points in polygon editor hover ([GH-82853](https://github.com/godotengine/godot/pull/82853)).
- Fixes undo/redo in tileset polygon editor ([GH-83093](https://github.com/godotengine/godot/pull/83093)).
- Warn users when TileMap is set as Y-sorted but no layer is ([GH-83144](https://github.com/godotengine/godot/pull/83144)).
- Fix tilemap live editing while game is running ([GH-83146](https://github.com/godotengine/godot/pull/83146)).
- Update `TileMap` layer draw index when it's dirty ([GH-83151](https://github.com/godotengine/godot/pull/83151)).
- Swap TileMap and TileSet buttons ([GH-83244](https://github.com/godotengine/godot/pull/83244)).
- Allow disabling the built-in tilemap navigation ([GH-83273](https://github.com/godotengine/godot/pull/83273)).
- Fix cannot update remote after disabling `use_global_coordinates` in `RemoteTransform2D` ([GH-83323](https://github.com/godotengine/godot/pull/83323)).
- Fix screen center position returned for rotated Camera2D ([GH-83427](https://github.com/godotengine/godot/pull/83427)).
- Fix bug where TileMap wouldn't update material correctly on assignment ([GH-83475](https://github.com/godotengine/godot/pull/83475)).
- Allow normal maps on TileMaps that use texture padding ([GH-83489](https://github.com/godotengine/godot/pull/83489)).
- Fix Polygon2D undo on transforming vertices ([GH-83659](https://github.com/godotengine/godot/pull/83659)).
- Fix TileSet painting options appear out of screen ([GH-83790](https://github.com/godotengine/godot/pull/83790)).
- Fix normals in TileSet when using CanvasTextures ([GH-83887](https://github.com/godotengine/godot/pull/83887)).
- Fix TileMap layer reverts and defaults ([GH-83888](https://github.com/godotengine/godot/pull/83888)).
- Fix `get_used_rect`, `get_used_cells` and `get_used_cells_by_id` in TileMap after a call to `clear()` ([GH-83890](https://github.com/godotengine/godot/pull/83890)).
- Fix Y-sort origin not working when set in TileMap runtime updates ([GH-84004](https://github.com/godotengine/godot/pull/84004)).
- Fix 2D bone weight editor not accounting for offset ([GH-84070](https://github.com/godotengine/godot/pull/84070)).
- Prevent crash and error spam related to Sprite2D with a region ([GH-84361](https://github.com/godotengine/godot/pull/84361)).
- TileMap: Fix compatibility code for old `cell_quadrant_size` property name ([GH-85463](https://github.com/godotengine/godot/pull/85463)).

#### 3D

- Re-add a Camera3D icon gizmo to the 3D editor ([GH-53104](https://github.com/godotengine/godot/pull/53104)).
- Implement numeric blender-style transforms ([GH-58389](https://github.com/godotengine/godot/pull/58389)).
- Wrap mouse for blender-style transforms ([GH-59467](https://github.com/godotengine/godot/pull/59467)).
- Improve editing of box collision shapes ([GH-71092](https://github.com/godotengine/godot/pull/71092)).
- Show visual-oriented 3D node gizmos only when selected ([GH-75303](https://github.com/godotengine/godot/pull/75303)).
- Fix Camera3D `project_*` methods not accounting for frustum offset ([GH-75806](https://github.com/godotengine/godot/pull/75806)).
- Avoid reimporting lightmap textures every getter call ([GH-77788](https://github.com/godotengine/godot/pull/77788)).
- Fix 3D viewport grid disappearing on scene tab changes ([GH-78694](https://github.com/godotengine/godot/pull/78694)).
- Fix VoxelGI saving VoxelGIData as a built-in file, despite being prompted to save it to an external file ([GH-78772](https://github.com/godotengine/godot/pull/78772)).
- Expose `compute_convex_mesh_points` function to GDScript ([GH-78871](https://github.com/godotengine/godot/pull/78871)).
- Change property hint range for camera attributes exposure multiplier ([GH-79138](https://github.com/godotengine/godot/pull/79138)).
- Make CSGShape follow curve's tilt in Path mode ([GH-79355](https://github.com/godotengine/godot/pull/79355)).
- Convert some Callables to `callable_mp()` ([GH-79373](https://github.com/godotengine/godot/pull/79373)).
- Initialize View Frame Time estimates to match 120 FPS ([GH-80124](https://github.com/godotengine/godot/pull/80124)).
- Add helper for 3D gizmos and unify box ([GH-80278](https://github.com/godotengine/godot/pull/80278)).
- Add handles to control Curve3D tilt ([GH-80329](https://github.com/godotengine/godot/pull/80329)).
- Allow setting values greater than the maximum in TorusMesh inspector ([GH-80441](https://github.com/godotengine/godot/pull/80441)).
- Add `global_basis` property to `Node3D` ([GH-80512](https://github.com/godotengine/godot/pull/80512)).
- Cleanup MeshLibrary changed signals ([GH-80782](https://github.com/godotengine/godot/pull/80782)).
- Improve Path3D gizmo usability ([GH-80802](https://github.com/godotengine/godot/pull/80802)).
- GridMap: Ensure the visibility is updated when entering the tree ([GH-81106](https://github.com/godotengine/godot/pull/81106)).
- Fix some keys triggering their actions twice in GridMap ([GH-81531](https://github.com/godotengine/godot/pull/81531)).
- Add 3D editor gizmo icons for Decal, LightmapProbe and FogVolume ([GH-81554](https://github.com/godotengine/godot/pull/81554)).
- Fix local 3D translation editing ([GH-81609](https://github.com/godotengine/godot/pull/81609)).
- Fix Curve3D baking up vectors for nontrivial curves ([GH-81885](https://github.com/godotengine/godot/pull/81885)).
- Update mesh list UI immediately after setting mesh library in gridmap ([GH-81914](https://github.com/godotengine/godot/pull/81914)).
- Optimize and tweak some SVGs, improve consistency between icons, and fix broken masks in light mode ([GH-82133](https://github.com/godotengine/godot/pull/82133)).
- Fix grid snapping for box shape gizmos ([GH-82381](https://github.com/godotengine/godot/pull/82381)).
- Make 3D editor gizmos and debug shapes ignore fog ([GH-82413](https://github.com/godotengine/godot/pull/82413)).
- Tweak Camera3D `size` property hint to make dragging more useful ([GH-82604](https://github.com/godotengine/godot/pull/82604)).
- Make gizmo plugin handle `SpriteBase3D` instead of `Sprite3D` ([GH-82901](https://github.com/godotengine/godot/pull/82901)).
- Enable UV2 on primitive meshes when using the MeshInstance3D context menu ([GH-82937](https://github.com/godotengine/godot/pull/82937)).
- Add an editor tool to automatically upgrade and re-save meshes ([GH-83613](https://github.com/godotengine/godot/pull/83613)).
- Fix some `Node3DEditor` snapping issues ([GH-84049](https://github.com/godotengine/godot/pull/84049)).
- Fix PlaneMesh tangents for 'Face X' orientation ([GH-84097](https://github.com/godotengine/godot/pull/84097)).
- Hide CSGShape's `debug_collision_shape` when it is invisible ([GH-84174](https://github.com/godotengine/godot/pull/84174)).

#### Animation

- Skip keyframe creation dialog when holding Shift in the animation editor ([GH-54524](https://github.com/godotengine/godot/pull/54524)).
- Allow changing imported AnimationLibrary names in AnimationPlayer in the editor ([GH-67965](https://github.com/godotengine/godot/pull/67965)).
- Add animation playback preview to scene import settings ([GH-76367](https://github.com/godotengine/godot/pull/76367)).
- Additional cleanup of bone editors ([GH-77096](https://github.com/godotengine/godot/pull/77096)).
- Add `TileSetAtlasSource::TileAnimationMode` options and allow to shuffle tile animations ([GH-77257](https://github.com/godotengine/godot/pull/77257)).
- Include animation frames in tile atlas merge ([GH-77316](https://github.com/godotengine/godot/pull/77316)).
- Fix infinite loop state check in `AnimationStateMachine` ([GH-79141](https://github.com/godotengine/godot/pull/79141)).
- Add 3.x compatibility for animation loop mode ([GH-79155](https://github.com/godotengine/godot/pull/79155)).
- Fix `Animation::subtract_variant` for affine transforms ([GH-79279](https://github.com/godotengine/godot/pull/79279)).
- Fix `AnimationNodeTransition` with negative time scale ([GH-79403](https://github.com/godotengine/godot/pull/79403)).
- Fix `tween_property` on `Basis` to properly update its value ([GH-79426](https://github.com/godotengine/godot/pull/79426)).
- Fix the error when clicking AnimationTree in the editor ([GH-79588](https://github.com/godotengine/godot/pull/79588)).
- Make `AnimationNodeBlendTree` use `RBMap` instead `HashMap` ([GH-79595](https://github.com/godotengine/godot/pull/79595)).
- Fix rename animation in SpriteFramesEditor ([GH-79600](https://github.com/godotengine/godot/pull/79600)).
- SpriteFrames Editor: Fix FPS applied to two animations when switching animation ([GH-79692](https://github.com/godotengine/godot/pull/79692)).
- Make animation name list scroll to new animation in `SpriteEditor` ([GH-79743](https://github.com/godotengine/godot/pull/79743)).
- SpriteFrames Editor: Fix Frame Duration applied to wrong frame when switching frame ([GH-79872](https://github.com/godotengine/godot/pull/79872)).
- Improve and clarify paused Tweens ([GH-79879](https://github.com/godotengine/godot/pull/79879)).
- Avoid emitting signals if the animation is not ready to be processed ([GH-80367](https://github.com/godotengine/godot/pull/80367)).
- Fix initial value with delay in PropertyTweener ([GH-80702](https://github.com/godotengine/godot/pull/80702)).
- Ensure methods skipped by `AnimationPlayer::seek` are not called ([GH-80708](https://github.com/godotengine/godot/pull/80708)).
- Implement `AnimationMixer` as a base class of `AnimationPlayer` and `AnimationTree` ([GH-80813](https://github.com/godotengine/godot/pull/80813)).
- Revive onion skinning ([GH-80939](https://github.com/godotengine/godot/pull/80939)).
- Prevent errors if Tween callback's object is freed ([GH-81127](https://github.com/godotengine/godot/pull/81127)).
- Select node when clicked in AnimationPlayer timeline ([GH-81188](https://github.com/godotengine/godot/pull/81188)).
- Fix incorrect cast when animating `int` ([GH-81296](https://github.com/godotengine/godot/pull/81296)).
- Fix animation keyframes being skipped when played backwards ([GH-81452](https://github.com/godotengine/godot/pull/81452)).
- Check if property exists before tweening ([GH-81525](https://github.com/godotengine/godot/pull/81525)).
- Ignore method track when drawing line between keys ([GH-81563](https://github.com/godotengine/godot/pull/81563)).
- Hide animation toolbar above the viewport correctly when switching scenes ([GH-81606](https://github.com/godotengine/godot/pull/81606)).
- Defer updating the animations Tree in SpriteFramesEditor to avoid crashes ([GH-81643](https://github.com/godotengine/godot/pull/81643)).
- SceneTreeDock: Remove animation tracks with correct indices ([GH-81651](https://github.com/godotengine/godot/pull/81651)).
- Fix BoneAttachment3D signal connection ([GH-81695](https://github.com/godotengine/godot/pull/81695)).
- Fix crash when clicking on "Interpolation Mode" with nonexistent node path ([GH-81779](https://github.com/godotengine/godot/pull/81779)).
- Improve retarget auto-mapping algorithm ([GH-81843](https://github.com/godotengine/godot/pull/81843)).
- Fix theme access and improve UX in AnimationTree editor ([GH-82210](https://github.com/godotengine/godot/pull/82210)).
- Fix `SkeletonIK3D` editor preview when changing active node ([GH-82391](https://github.com/godotengine/godot/pull/82391)).
- Reimport bone attachment fixes ([GH-82471](https://github.com/godotengine/godot/pull/82471)).
- Fix "Some nodes are referenced by animation tracks" when deleting instance ([GH-82486](https://github.com/godotengine/godot/pull/82486)).
- Fix GroupedStateMachine reset ([GH-82563](https://github.com/godotengine/godot/pull/82563)).
- Fix crash when deleting the player in `AnimationPlayerEditorPlugin` ([GH-82573](https://github.com/godotengine/godot/pull/82573)).
- Limit animation audio clip inspector offset sliders to clip length ([GH-82627](https://github.com/godotengine/godot/pull/82627)).
- Tweak AnimationPlayer speed scale property hint to make dragging more useful ([GH-82641](https://github.com/godotengine/godot/pull/82641)).
- Fix `AnimationPlayer::play()` process unwanted start between the same animations ([GH-82898](https://github.com/godotengine/godot/pull/82898)).
- AnimationMixer: Fix non-numeric misc type (`Resource`, `Dictionary` & etc.) values cannot be blended with `UpdateMode.UPDATE_CONTINUOUS` ([GH-83030](https://github.com/godotengine/godot/pull/83030)).
- Move animation slice processing to `_post_fix_animations` ([GH-83036](https://github.com/godotengine/godot/pull/83036)).
- Set new SkeletonRestFixer tracks as imported ([GH-83076](https://github.com/godotengine/godot/pull/83076)).
- Fix editor crash when re-importing GLTF while animation is playing ([GH-83104](https://github.com/godotengine/godot/pull/83104)).
- Show AnimationMixer warning for non-numeric types only when relevant ([GH-83417](https://github.com/godotengine/godot/pull/83417)).
- Fix onion skinning internals activating audio/method/animation tracks ([GH-83430](https://github.com/godotengine/godot/pull/83430)).
- Remove AnimationMixer bindings only bound in the editor ([GH-83440](https://github.com/godotengine/godot/pull/83440)).
- Re-add close button for nodes in `AnimationNodeBlendTree` editor ([GH-83507](https://github.com/godotengine/godot/pull/83507)).
- Automatic reconnection of nodes in blend tree ([GH-83534](https://github.com/godotengine/godot/pull/83534)).
- Add vertical scrolling to bézier track editor ([GH-83776](https://github.com/godotengine/godot/pull/83776)).
- Ensure AnimationPlayer evaluate animations when autoplay is enabled and node becomes ready ([GH-83781](https://github.com/godotengine/godot/pull/83781)).
- Fix animation track paths updated by scene dock ([GH-83934](https://github.com/godotengine/godot/pull/83934)).
- Unpress buttons in AnimationTree when switching to read-only mode ([GH-84052](https://github.com/godotengine/godot/pull/84052)).
- Fix invalid return from some more `_get/_set` ([GH-84060](https://github.com/godotengine/godot/pull/84060)).
- Add descriptive warning for animation track hint fails ([GH-84129](https://github.com/godotengine/godot/pull/84129)).
- Fix AnimationTimeline time not updating when dragged ([GH-84170](https://github.com/godotengine/godot/pull/84170)).
- Avoid popping up dialogs excessively in the Animation editor ([GH-84208](https://github.com/godotengine/godot/pull/84208)).
- Fix perform_node_renames handling of AnimationMixers track paths ([GH-84282](https://github.com/godotengine/godot/pull/84282)).
- Unexpose internal data property of `AnimationLibrary` ([GH-84376](https://github.com/godotengine/godot/pull/84376)).
- Add `PackedArray` to the list of enforcing `Discrete` for `AnimationMixer` ([GH-84390](https://github.com/godotengine/godot/pull/84390)).
- Fix RESET not effective when saving inactive scene ([GH-84405](https://github.com/godotengine/godot/pull/84405)).
- Change AnimationLibEditor's mixer to actual Mixer ([GH-84551](https://github.com/godotengine/godot/pull/84551)).
- Remove AnimatedSprite pointer when clearing editor ([GH-84625](https://github.com/godotengine/godot/pull/84625)).
- Rework blending method in `Variant` animation for `Int`/`Array`/`String` ([GH-84815](https://github.com/godotengine/godot/pull/84815)).
- Fix ValueTrack with Resource is leaking ([GH-84942](https://github.com/godotengine/godot/pull/84942)).
- Fix seeking bug in AnimationPlayerEditor ([GH-85193](https://github.com/godotengine/godot/pull/85193)).
- Clear seeked/started flag after seeking/advancing in AnimationPlayer ([GH-85221](https://github.com/godotengine/godot/pull/85221)).
- Bind `_reset`/`_restore` in AnimationMixer ([GH-85254](https://github.com/godotengine/godot/pull/85254)).
- Fix TrackCache memory crash ([GH-85266](https://github.com/godotengine/godot/pull/85266)).
- Perform safe copies in `AnimatedValuesBackup::get_cache_copy()` ([GH-85302](https://github.com/godotengine/godot/pull/85302)).
- Fix a crash when trying to restore uncopyable animation tracks ([GH-85308](https://github.com/godotengine/godot/pull/85308)).
- Check the seeking if it is processed immediately after playback as a special case ([GH-85366](https://github.com/godotengine/godot/pull/85366)).
- Make AnimationPlaybackTrack keep state when stopping ([GH-85411](https://github.com/godotengine/godot/pull/85411)).
- AnimationMixer: Validate ObjectID before blend in case the object was freed ([GH-85461](https://github.com/godotengine/godot/pull/85461)).

#### Assetlib

- Fix long plugin names breaking the UI ([GH-80555](https://github.com/godotengine/godot/pull/80555)).
- Improve handling of archives when installing assets ([GH-81358](https://github.com/godotengine/godot/pull/81358)).
- Allow to specify target folder when installing assets ([GH-81620](https://github.com/godotengine/godot/pull/81620)).

#### Audio

- PulseAudio: Remove `get_latency()` caching ([GH-45152](https://github.com/godotengine/godot/pull/45152)).
- Context aware MIDI event printing ([GH-68820](https://github.com/godotengine/godot/pull/68820)).
- Add build option to enable MP1 and MP2 support in minimp3 ([GH-72729](https://github.com/godotengine/godot/pull/72729)).
- Add a `--audio-output-latency` command-line argument ([GH-78013](https://github.com/godotengine/godot/pull/78013)).
- Implement loading OGG files from buffer and file path ([GH-78084](https://github.com/godotengine/godot/pull/78084)).
- Simpler default values for AudioStreamRandomizer ([GH-80171](https://github.com/godotengine/godot/pull/80171)).
- Fix OGG audio loop offset pop ([GH-80452](https://github.com/godotengine/godot/pull/80452)).
- Add project settings for AVAudioSessionCategory on iOS ([GH-81196](https://github.com/godotengine/godot/pull/81196)).
- Remove some dead declarations in `audio_stream_player.h` ([GH-81485](https://github.com/godotengine/godot/pull/81485)).
- Fix audio stream generators getting freed accidentally ([GH-81508](https://github.com/godotengine/godot/pull/81508)).
- Add a `bus_renamed` AudioServer signal ([GH-81641](https://github.com/godotengine/godot/pull/81641)).
- Only warn once about OGG seeking issues ([GH-81704](https://github.com/godotengine/godot/pull/81704)).
- Tweak property order in the inspector for AudioStreamRandomizer ([GH-82411](https://github.com/godotengine/godot/pull/82411)).
- Fix `AudioStreamRandomizer.random_volume_offset_db` not working ([GH-82478](https://github.com/godotengine/godot/pull/82478)).
- Fix pausing stream on entering tree ([GH-83779](https://github.com/godotengine/godot/pull/83779)).
- Fix OGG Vorbis infinite error spam with corrupt file ([GH-84723](https://github.com/godotengine/godot/pull/84723)).

#### Buildsystem

- SCons: Add `object_prefix` option ([GH-62652](https://github.com/godotengine/godot/pull/62652)).
- Allow unbundling OpenXR (for Linux distros) ([GH-73443](https://github.com/godotengine/godot/pull/73443)).
- Add a Linux ThreadSanitizer job to CI ([GH-73777](https://github.com/godotengine/godot/pull/73777)).
- libpng: Enable intrinsics on x86/SSE2, ppc64/VSX, and all arm/NEON ([GH-78325](https://github.com/godotengine/godot/pull/78325)).
- Add static check for overzealous .gitignores and fix an example of such ([GH-78908](https://github.com/godotengine/godot/pull/78908)).
- SCons : Add `scu_limit` argument ([GH-78959](https://github.com/godotengine/godot/pull/78959)).
- Linux: Fix build with `use_sowrap=no` and various warnings/errors ([GH-79097](https://github.com/godotengine/godot/pull/79097)).
- Linux: Allow unbundling brotli to use system library ([GH-79101](https://github.com/godotengine/godot/pull/79101)).
- Linux: Link libsquish directly when unbundling, .pc file unreliable ([GH-79105](https://github.com/godotengine/godot/pull/79105)).
- Fix build options configuration for Visual Studio projects ([GH-79238](https://github.com/godotengine/godot/pull/79238)).
- CI: Allow skipping our GHA workflows with `DISABLE_GODOT_CI` variable ([GH-79321](https://github.com/godotengine/godot/pull/79321)).
- Fix `doc_status.py` trying to get removed `version` tag from XML ([GH-79406](https://github.com/godotengine/godot/pull/79406)).
- Have `core_bind.h` Thread type syntax match `core_bind.cpp` ([GH-79466](https://github.com/godotengine/godot/pull/79466)).
- Web: Use `fvisibility=hidden` for side module when `dlink_enabled` ([GH-79578](https://github.com/godotengine/godot/pull/79578)).
- Header formatting commits to `.git-blame-ignore-revs` ([GH-79615](https://github.com/godotengine/godot/pull/79615)).
- Fix gcc builds failing on Windows ([GH-79724](https://github.com/godotengine/godot/pull/79724)).
- Windows: Try using objcopy and strip with prefix and without prefix ([GH-79871](https://github.com/godotengine/godot/pull/79871)).
- CI: Extract godot-cpp testing into its own job ([GH-80091](https://github.com/godotengine/godot/pull/80091)).
- CI: Free disk space on Linux runners ([GH-80115](https://github.com/godotengine/godot/pull/80115)).
- CI: Compare API compatibility against both 4.0 and 4.1 ([GH-80354](https://github.com/godotengine/godot/pull/80354)).
- Suppress NavigationServer race conditions ([GH-80392](https://github.com/godotengine/godot/pull/80392)).
- Fix API validation script on macOS ([GH-80471](https://github.com/godotengine/godot/pull/80471)).
- SCons: Disable misbehaving MSVC incremental linking ([GH-80482](https://github.com/godotengine/godot/pull/80482)).
- SCons: Carry over the `windows_subsystem` setting to the generated vsproj ([GH-80547](https://github.com/godotengine/godot/pull/80547)).
- SCons: Disable C++ exception handling ([GH-80612](https://github.com/godotengine/godot/pull/80612)).
- Fix GCC `-Wmaybe-uninitialized` warnings ([GH-80615](https://github.com/godotengine/godot/pull/80615)).
- SCons: Enable `/WX` on LINKFLAGS for MSVC with `werror=yes` ([GH-80711](https://github.com/godotengine/godot/pull/80711)).
- SCons: Make ktx module require basis_universal ([GH-80789](https://github.com/godotengine/godot/pull/80789)).
- Windows: Add main executable to the console wrapper dependencies to prevent simultaneous linking ([GH-80918](https://github.com/godotengine/godot/pull/80918)).
- Assign extension validation files to the GDExtension and .NET teams in CODEOWNERS ([GH-81116](https://github.com/godotengine/godot/pull/81116)).
- Remove too greedy gcov/lcov ignores from `.gitignore` ([GH-81120](https://github.com/godotengine/godot/pull/81120)).
- SCons: Add option for MSVC incremental linking ([GH-81144](https://github.com/godotengine/godot/pull/81144)).
- CI: Bump version for `actions/checkout@v4` and `actions/setup-dotnet@v3` ([GH-81302](https://github.com/godotengine/godot/pull/81302)).
- CI: Compat checks: Make fetching the reference API more robust ([GH-81337](https://github.com/godotengine/godot/pull/81337)).
- Web: Workaround Emscripten 3.1.42+ LTO regression ([GH-81340](https://github.com/godotengine/godot/pull/81340)).
- UWP: Remove platform port, needs to be redone from scratch for 4.x ([GH-81416](https://github.com/godotengine/godot/pull/81416)).
- Haiku: Remove remnants of past WIP platform port ([GH-81420](https://github.com/godotengine/godot/pull/81420)).
- Web: Fix version check for missing scalbnf LTO workaround ([GH-81869](https://github.com/godotengine/godot/pull/81869)).
- macOS: Workaround Xcode 15 linker bug ([GH-81968](https://github.com/godotengine/godot/pull/81968)).
- SCons: Fix Python 3.12 SyntaxError with regex escape sequences ([GH-82290](https://github.com/godotengine/godot/pull/82290)).
- Windows: Remove `MSVC` define redundant with `_MSC_VER` ([GH-82304](https://github.com/godotengine/godot/pull/82304)).
- Updated compiler version detection ([GH-82325](https://github.com/godotengine/godot/pull/82325)).
- Fix compiler detection ([GH-82352](https://github.com/godotengine/godot/pull/82352)).
- Fix build on MSVC 2017 ([GH-82450](https://github.com/godotengine/godot/pull/82450)).
- iOS: Fix build with Xcode 15 ([GH-82458](https://github.com/godotengine/godot/pull/82458)).
- Web: Fix `dlink_enabled` build ([GH-82633](https://github.com/godotengine/godot/pull/82633)).
- SCons: Change `check_c_headers` from tuple array to dictionary ([GH-82638](https://github.com/godotengine/godot/pull/82638)).
- Validate `code` tags for class and member references ([GH-82691](https://github.com/godotengine/godot/pull/82691)).
- Fix building without GDScript ([GH-82985](https://github.com/godotengine/godot/pull/82985)).
- CI: Switch mesa PPA from kisak-mesa to turtle ([GH-83147](https://github.com/godotengine/godot/pull/83147)).
- CI: Workaround recently broken add-apt-repository on GHA ([GH-83214](https://github.com/godotengine/godot/pull/83214)).
- X11: Fix unused variables warning when `touch` is disabled ([GH-83265](https://github.com/godotengine/godot/pull/83265)).
- Bump the java version to version 17 ([GH-83515](https://github.com/godotengine/godot/pull/83515)).
- SCons: Use CXXFLAGS to disable exceptions, it's only for C++ ([GH-83618](https://github.com/godotengine/godot/pull/83618)).
- Web: Fix closure compiler builds using BIGINT ([GH-83720](https://github.com/godotengine/godot/pull/83720)).
- SCons: Reduce and cleanup verbose output for SCU builds ([GH-83996](https://github.com/godotengine/godot/pull/83996)).
- Use colored output on CI for Doctest ([GH-84100](https://github.com/godotengine/godot/pull/84100)).
- Linux: Remove hardcoded lib path for x86 cross-compilation ([GH-84307](https://github.com/godotengine/godot/pull/84307)).
- Replace the use of the `ANDROID_SDK_ROOT` env variable with `ANDROID_HOME` ([GH-84316](https://github.com/godotengine/godot/pull/84316)).
- CI: Pin Emscripten to 3.1.39 ([GH-84717](https://github.com/godotengine/godot/pull/84717)).
- Donors: Change tiers to match Dev Fund, sync latest data ([GH-84884](https://github.com/godotengine/godot/pull/84884)).
- makerst: Disallow user-contributed notes on the class index page ([GH-85006](https://github.com/godotengine/godot/pull/85006)).
- Use mingw-std-threads in MinGW builds ([GH-85039](https://github.com/godotengine/godot/pull/85039)).
- Avoid conflict between mingw-std-threads and Clang's own ([GH-85208](https://github.com/godotengine/godot/pull/85208)).
- Fix generating vsproj with SCons 4.6.0+ ([GH-85357](https://github.com/godotengine/godot/pull/85357)).
- Add unsigned char cast ifdef ([GH-85500](https://github.com/godotengine/godot/pull/85500)).

#### C#

- Allow readonly and writeonly C# properties to be accessed from GDScript ([GH-67304](https://github.com/godotengine/godot/pull/67304)).
- Support exporting for Android ([GH-73257](https://github.com/godotengine/godot/pull/73257)).
- Fix crash with `DisposablesTracker_OnGodotShuttingDown` ([GH-78157](https://github.com/godotengine/godot/pull/78157)).
- Add `PropertyHint.Enum` support to `Array<StringName>` ([GH-78264](https://github.com/godotengine/godot/pull/78264)).
- Check if JetBrains Rider editor path is empty ([GH-78516](https://github.com/godotengine/godot/pull/78516)).
- Automatically generate version defines ([GH-78722](https://github.com/godotengine/godot/pull/78722)).
- Update the RiderPathLocator to support the JetBrains Toolbox 2.0 ([GH-78832](https://github.com/godotengine/godot/pull/78832)).
- Add platform name to the exported data directory ([GH-78846](https://github.com/godotengine/godot/pull/78846)).
- Fix deserialization of delegates that are 0-parameter overloads ([GH-78877](https://github.com/godotengine/godot/pull/78877)).
- Add a Roslyn analyzer for global classes ([GH-79007](https://github.com/godotengine/godot/pull/79007)).
- Add missing `useModelFront` parameter to GodotSharp Basis and Transform ([GH-79082](https://github.com/godotengine/godot/pull/79082)).
- Compare symbol names without null flow state ([GH-79094](https://github.com/godotengine/godot/pull/79094)).
- Add null check before calling `UnregisterGodotObject` ([GH-79151](https://github.com/godotengine/godot/pull/79151)).
- Fix command line exporting ([GH-79173](https://github.com/godotengine/godot/pull/79173)).
- Add a warning about C# differences to the class reference ([GH-79206](https://github.com/godotengine/godot/pull/79206)).
- Document generated members ([GH-79239](https://github.com/godotengine/godot/pull/79239)).
- Print error when MethodBind/Callable call fails ([GH-79249](https://github.com/godotengine/godot/pull/79249)).
- Improve `GD.PushError` and `GD.PushWarning` ([GH-79280](https://github.com/godotengine/godot/pull/79280)).
- Fix MSVC dotnet builds failing if running `dev_mode` ([GH-79351](https://github.com/godotengine/godot/pull/79351)).
- Move build button to EditorRunBar ([GH-79357](https://github.com/godotengine/godot/pull/79357)).
- Fix line in OpenInExternalEditor ([GH-79404](https://github.com/godotengine/godot/pull/79404)).
- Generate instance types for singletons ([GH-79470](https://github.com/godotengine/godot/pull/79470)).
- Suppress NU5128 warning ([GH-79501](https://github.com/godotengine/godot/pull/79501)).
- Fix issues in C# documentation comments ([GH-79748](https://github.com/godotengine/godot/pull/79748)).
- Delegate opening files for Rider to the RiderPathLocator NuGet package ([GH-79958](https://github.com/godotengine/godot/pull/79958)).
- Mono: Avoid dictionary lookup for common colors ([GH-80047](https://github.com/godotengine/godot/pull/80047)).
- Show alert if .NET assemblies dir does not exist ([GH-80212](https://github.com/godotengine/godot/pull/80212)).
- Redesign MSBuild panel ([GH-80260](https://github.com/godotengine/godot/pull/80260)).
- Fix typo in parameter name in documentation ([GH-80303](https://github.com/godotengine/godot/pull/80303)).
- Improve diagnostic messages and add help link ([GH-80489](https://github.com/godotengine/godot/pull/80489)).
- Fix exporting for Android ([GH-80521](https://github.com/godotengine/godot/pull/80521)).
- Generate and use compat methods ([GH-80527](https://github.com/godotengine/godot/pull/80527)).
- Implement `proxy_name` for EnumInterface ([GH-80628](https://github.com/godotengine/godot/pull/80628)).
- Include argument types in generated methods ([GH-80629](https://github.com/godotengine/godot/pull/80629)).
- paramref now properly tagged in documentation ([GH-80630](https://github.com/godotengine/godot/pull/80630)).
- Fixed delegate docstring logic ([GH-80631](https://github.com/godotengine/godot/pull/80631)).
- Dereference editor types in core documentation ([GH-80632](https://github.com/godotengine/godot/pull/80632)).
- CI: Propagate error code when glue generation fails ([GH-80846](https://github.com/godotengine/godot/pull/80846)).
- Updated C# example of AddPropertyInfo ([GH-80851](https://github.com/godotengine/godot/pull/80851)).
- Replace `StringNameCache` with `SNAME` ([GH-81073](https://github.com/godotengine/godot/pull/81073)).
- Add abstract class support ([GH-81101](https://github.com/godotengine/godot/pull/81101)).
- Replace usage of deprecated `project_settings_changed` signal ([GH-81175](https://github.com/godotengine/godot/pull/81175)).
- Fix int's C# documentation ([GH-81227](https://github.com/godotengine/godot/pull/81227)).
- Expose `asinh`, `acosh` and `atanh` in Mathf ([GH-81229](https://github.com/godotengine/godot/pull/81229)).
- Fix double unregistration on dispose of Array ([GH-81230](https://github.com/godotengine/godot/pull/81230)).
- Hide hostfxr not found error ([GH-81690](https://github.com/godotengine/godot/pull/81690)).
- Move `bindings_generator` warnings to `.editorconfig` ([GH-81703](https://github.com/godotengine/godot/pull/81703)).
- Make C# static methods accessible ([GH-81783](https://github.com/godotengine/godot/pull/81783)).
- Fixed VS 2022 Mac compatibility ([GH-81802](https://github.com/godotengine/godot/pull/81802)).
- Add Vector2/3/4i.MAX and MIN ([GH-81819](https://github.com/godotengine/godot/pull/81819)).
- Implemented `{project}` placeholder for external dotnet editor ([GH-81847](https://github.com/godotengine/godot/pull/81847)).
- CS1591 from `NoWarn` to `suggestion` ([GH-81934](https://github.com/godotengine/godot/pull/81934)).
- Use `HashCode.Combine()` for basic composite types instead of xor ([GH-82240](https://github.com/godotengine/godot/pull/82240)).
- Remove compat method that is now generated ([GH-82375](https://github.com/godotengine/godot/pull/82375)).
- Fixed an error in `Vector3.BezierDerivative` in mono module ([GH-82664](https://github.com/godotengine/godot/pull/82664)).
- Fix C# editor dialogs ([GH-82683](https://github.com/godotengine/godot/pull/82683)).
- Add C# iOS support ([GH-82729](https://github.com/godotengine/godot/pull/82729)).
- Add C# compat members for 4.2 changes ([GH-82740](https://github.com/godotengine/godot/pull/82740)).
- Add checks to Android export ([GH-82762](https://github.com/godotengine/godot/pull/82762)).
- Report diagnostic for Node exports in a type that doesn't derive from Node ([GH-82918](https://github.com/godotengine/godot/pull/82918)).
- Fix unresolved `inheritdoc` links in `Compat.cs` ([GH-83001](https://github.com/godotengine/godot/pull/83001)).
- Untyped to typed `ArgumentNullException` ([GH-83051](https://github.com/godotengine/godot/pull/83051)).
- Fix MSVC dotnet `dev_mode` regression ([GH-83114](https://github.com/godotengine/godot/pull/83114)).
- Fix lookup for singleton instance types ([GH-83249](https://github.com/godotengine/godot/pull/83249)).
- Fallback to the latest SDK ([GH-83325](https://github.com/godotengine/godot/pull/83325)).
- Fix converting default Callables to native ([GH-83357](https://github.com/godotengine/godot/pull/83357)).
- Allow exporting games without C# ([GH-83422](https://github.com/godotengine/godot/pull/83422)).
- Clarify C# docs for operators performing `xform_inv` ([GH-83514](https://github.com/godotengine/godot/pull/83514)).
- Fix C# docs to use the proper XML ([GH-83529](https://github.com/godotengine/godot/pull/83529)).
- Fix generated nested class order ([GH-83532](https://github.com/godotengine/godot/pull/83532)).
- Add `NOTIFICATION_PREDELETE_CLEANUP` notification to fix C# `Dispose()` ([GH-83670](https://github.com/godotengine/godot/pull/83670)).
- Free dialogs when exiting the editor ([GH-83809](https://github.com/godotengine/godot/pull/83809)).
- Fix node names of submenu items across the editor ([GH-84617](https://github.com/godotengine/godot/pull/84617)).
- Rename `Vector2/3/4I.Min/Max` to `MinValue/MaxValue` ([GH-84663](https://github.com/godotengine/godot/pull/84663)).
- Fail `callp` silently if script is not valid ([GH-84897](https://github.com/godotengine/godot/pull/84897)).
- iOS: Fix dotnet export ([GH-84945](https://github.com/godotengine/godot/pull/84945)).

#### Codestyle

- Made hidden ProjectSettings groups more explicit ([GH-61818](https://github.com/godotengine/godot/pull/61818)).
- Extract StyleBoxFlat, StyleBoxTexture and StyleBoxLine in their own file ([GH-68396](https://github.com/godotengine/godot/pull/68396)).
- Extract and reorganize texture resource classes ([GH-68460](https://github.com/godotengine/godot/pull/68460)).
- Add command-line option to run a `MainLoop` by its global class name ([GH-78045](https://github.com/godotengine/godot/pull/78045)).
- Remove unnecessary value assignments throughout the codebase ([GH-78538](https://github.com/godotengine/godot/pull/78538)).
- Remove uses of `vformat()` with no placeholders ([GH-78797](https://github.com/godotengine/godot/pull/78797)).
- Unify and streamline connecting to Resource changes ([GH-78993](https://github.com/godotengine/godot/pull/78993)).
- Remove unused struct from NavigationMesh ([GH-79713](https://github.com/godotengine/godot/pull/79713)).
- Update NavObstacle creation to new Mutex style ([GH-79916](https://github.com/godotengine/godot/pull/79916)).
- Use compile-time Unicode string conversion ([GH-80362](https://github.com/godotengine/godot/pull/80362)).
- Undefine `typed_array.h` templates after use ([GH-80792](https://github.com/godotengine/godot/pull/80792)).
- Remove debug print ([GH-81129](https://github.com/godotengine/godot/pull/81129)).
- Extract `ScriptInstance` into its own file to simplify includes ([GH-81388](https://github.com/godotengine/godot/pull/81388)).
- Don't use auto where not warranted ([GH-81414](https://github.com/godotengine/godot/pull/81414)).
- Core: Replace `ERR_FAIL_COND` with `ERR_FAIL_NULL` where applicable ([GH-81487](https://github.com/godotengine/godot/pull/81487)).
- [Scene,Main] Replace `ERR_FAIL_COND` with `ERR_FAIL_NULL` where applicable ([GH-81568](https://github.com/godotengine/godot/pull/81568)).
- [Drivers,Platform] Replace `ERR_FAIL_COND` with `ERR_FAIL_NULL` where applicable ([GH-81583](https://github.com/godotengine/godot/pull/81583)).
- Editor: Replace `ERR_FAIL_COND` with `ERR_FAIL_NULL` where applicable ([GH-81705](https://github.com/godotengine/godot/pull/81705)).
- Modules: Replace `ERR_FAIL_COND` with `ERR_FAIL_NULL` where applicable ([GH-81760](https://github.com/godotengine/godot/pull/81760)).
- Fix typo in `heuristic_euclidian` helper in AStarGrid2D ([GH-82297](https://github.com/godotengine/godot/pull/82297)).
- Servers: Replace `ERR_FAIL_COND` with `ERR_FAIL_NULL` where applicable ([GH-82313](https://github.com/godotengine/godot/pull/82313)).
- Fix some typos in source ([GH-82779](https://github.com/godotengine/godot/pull/82779)).
- Replace `sanity` with `safety` for checks ([GH-83002](https://github.com/godotengine/godot/pull/83002)).
- Replace `ERR_FAIL_COND` with `ERR_FAIL_NULL` where applicable ([GH-83003](https://github.com/godotengine/godot/pull/83003)).
- Fix incorrect null check ([GH-83068](https://github.com/godotengine/godot/pull/83068)).
- Clean up some Editor, OpenXR, VideoStream code ([GH-83683](https://github.com/godotengine/godot/pull/83683)).
- Solve race condition between AThousandShips and Akien ([GH-83740](https://github.com/godotengine/godot/pull/83740)).
- Don't use TTR/RTR for ERR/WARN prints ([GH-84774](https://github.com/godotengine/godot/pull/84774)).

#### Core

- Check parameter validity in `Object::set_script` ([GH-46125](https://github.com/godotengine/godot/pull/46125)).
- Add `settings_changed` signal to ProjectSettings ([GH-62038](https://github.com/godotengine/godot/pull/62038)).
- Fix the behavior of the resource property of the sub-scene root node on instantiation ([GH-65011](https://github.com/godotengine/godot/pull/65011)).
- Disallow invalid escape sequences in `JSON.parse` ([GH-66170](https://github.com/godotengine/godot/pull/66170)).
- Reimplement Resource.`_setup_local_to_scene` & deprecate signal ([GH-67080](https://github.com/godotengine/godot/pull/67080)).
- Deprecate `Resource.setup_local_to_scene` ([GH-67082](https://github.com/godotengine/godot/pull/67082)).
- Fix crash when saving resources with circular references ([GH-68281](https://github.com/godotengine/godot/pull/68281)).
- Add `type_string()` utility ([GH-69624](https://github.com/godotengine/godot/pull/69624)).
- Add a type conversion method to Variant Utility and expose to scripting ([GH-70080](https://github.com/godotengine/godot/pull/70080)).
- Ensure `MainLoop` and its custom script is set right after it's resolved ([GH-70771](https://github.com/godotengine/godot/pull/70771)).
- Ensure that SceneTree is initialized and finalized at correct time ([GH-72248](https://github.com/godotengine/godot/pull/72248)).
- Avoid sorting CallableCustomMethodPointers by their actual address values ([GH-72346](https://github.com/godotengine/godot/pull/72346)).
- Remove unused `threaded_array_processor.h` ([GH-74012](https://github.com/godotengine/godot/pull/74012)).
- Expose and document `Image.get_mipmap_count()` ([GH-74142](https://github.com/godotengine/godot/pull/74142)).
- Fix `Image.convert()` overwriting custom mipmaps ([GH-74238](https://github.com/godotengine/godot/pull/74238)).
- Support numeric/binary hash comparison for floats derived from Variants (as well as existing semantic comparison) ([GH-74588](https://github.com/godotengine/godot/pull/74588)).
- Expose `_validate_property()` for scripting ([GH-75778](https://github.com/godotengine/godot/pull/75778)).
- Add function `ZIPReader::file_exists` ([GH-76860](https://github.com/godotengine/godot/pull/76860)).
- Add `Node.get_tree_string` and `Node.get_tree_string_pretty` ([GH-77072](https://github.com/godotengine/godot/pull/77072)).
- Add a `--max-fps` command-line argument to set a FPS limit ([GH-78012](https://github.com/godotengine/godot/pull/78012)).
- Make a header for VariantUtilityFunctions ([GH-78108](https://github.com/godotengine/godot/pull/78108)).
- Added `Image::load_svg_from_(buffer|string)` ([GH-78248](https://github.com/godotengine/godot/pull/78248)).
- Add inverse hyperbolic functions `asinh()`, `acosh()` & `atanh()` ([GH-78404](https://github.com/godotengine/godot/pull/78404)).
- Add `String.reverse` method ([GH-78529](https://github.com/godotengine/godot/pull/78529)).
- Refactor CallQueue flushing for clarity ([GH-78612](https://github.com/godotengine/godot/pull/78612)).
- Fix `Object::notification` order ([GH-78634](https://github.com/godotengine/godot/pull/78634)).
- Allow renaming child nodes in `_ready` ([GH-78706](https://github.com/godotengine/godot/pull/78706)).
- Support loading of translations on threads ([GH-78747](https://github.com/godotengine/godot/pull/78747)).
- Fix zero-sized WorkerThreadPool not processing group tasks ([GH-78845](https://github.com/godotengine/godot/pull/78845)).
- Fix `Node::add_sibling` parent check ([GH-78847](https://github.com/godotengine/godot/pull/78847)).
- Fix error when non-ASCII characters in resource pack path ([GH-78935](https://github.com/godotengine/godot/pull/78935)).
- Reimplement scene change ([GH-78988](https://github.com/godotengine/godot/pull/78988)).
- Improve error message for `Node.set_owner` ([GH-79000](https://github.com/godotengine/godot/pull/79000)).
- Fix range error for `Array.slice` ([GH-79103](https://github.com/godotengine/godot/pull/79103)).
- TextServer: Remove excessive Dictionary checks ([GH-79166](https://github.com/godotengine/godot/pull/79166)).
- Fix erroneous `pad_zeros()` warning ([GH-79202](https://github.com/godotengine/godot/pull/79202)).
- Fix `PackedScene::get_last_modified_time()` always returns `0` ([GH-79237](https://github.com/godotengine/godot/pull/79237)).
- Add vararg `call()` method to C++ Callable ([GH-79341](https://github.com/godotengine/godot/pull/79341)).
- Fix byte to float color conversion in `DisplayServerWindows::screen_get_pixel` ([GH-79350](https://github.com/godotengine/godot/pull/79350)).
- Fix recursion level check for array stringification ([GH-79370](https://github.com/godotengine/godot/pull/79370)).
- Fix script name of Hant and Hans ([GH-79654](https://github.com/godotengine/godot/pull/79654)).
- Mention expected resource type in ResourceLoader load error ([GH-79737](https://github.com/godotengine/godot/pull/79737)).
- Add determinant check for minimized windows ([GH-79766](https://github.com/godotengine/godot/pull/79766)).
- FastNoiseLite: Fix cellular jitter using incorrect default value ([GH-79922](https://github.com/godotengine/godot/pull/79922)).
- Disable error condition for accessing safe rect ([GH-79937](https://github.com/godotengine/godot/pull/79937)).
- Fix life cycle of ResourceImporterTexture not tracked properly ([GH-79954](https://github.com/godotengine/godot/pull/79954)).
- String: Fix Unicode parsing error message encoding and related JSON tests ([GH-79980](https://github.com/godotengine/godot/pull/79980)).
- Fix life cycle of ResourceImporterTexture better ([GH-79981](https://github.com/godotengine/godot/pull/79981)).
- Fix global transform validity for `Node2D` and `Control` ([GH-80105](https://github.com/godotengine/godot/pull/80105)).
- Fix typo in modulo error message ([GH-80114](https://github.com/godotengine/godot/pull/80114)).
- Actually store safe-rect in embedder ([GH-80117](https://github.com/godotengine/godot/pull/80117)).
- Usage notes for DEV_ASSERT macro ([GH-80156](https://github.com/godotengine/godot/pull/80156)).
- Avoid crash on exiting due to late prints ([GH-80161](https://github.com/godotengine/godot/pull/80161)).
- Add `rotate_toward` and `angle_difference` methods ([GH-80225](https://github.com/godotengine/godot/pull/80225)).
- Avoid retrieving the object ID of a stack variable if it is nil ([GH-80256](https://github.com/godotengine/godot/pull/80256)).
- Remove `CanvasItem::_invalidate_global_transform` ([GH-80320](https://github.com/godotengine/godot/pull/80320)).
- Use StringName consistently to refer to the Master audio bus name ([GH-80381](https://github.com/godotengine/godot/pull/80381)).
- Deprecate `project_settings_changed` signal ([GH-80450](https://github.com/godotengine/godot/pull/80450)).
- Remove `DDS_INDEXED` from DDSFormat ([GH-80669](https://github.com/godotengine/godot/pull/80669)).
- Allow to get a list of visible embedded `Window`s ([GH-80673](https://github.com/godotengine/godot/pull/80673)).
- Fix CanvasItem notification thread guard ([GH-80752](https://github.com/godotengine/godot/pull/80752)).
- Optimize `String.left()` and `String.right()` ([GH-80824](https://github.com/godotengine/godot/pull/80824)).
- Implement center window function ([GH-81012](https://github.com/godotengine/godot/pull/81012)).
- Fix `JavaScriptBridge.eval()` never returning PackedByteArray ([GH-81015](https://github.com/godotengine/godot/pull/81015)).
- Add check to ensure registered classes are declared ([GH-81020](https://github.com/godotengine/godot/pull/81020)).
- Fix recursion level check for `VariantWriter::write()` with objects ([GH-81123](https://github.com/godotengine/godot/pull/81123)).
- Fix comparison of `Callable`s with binds ([GH-81131](https://github.com/godotengine/godot/pull/81131)).
- Fix consistency of GradientTexture changes ([GH-81137](https://github.com/godotengine/godot/pull/81137)).
- Fix for non-deterministic behavior in PCKPacker ([GH-81280](https://github.com/godotengine/godot/pull/81280)).
- String: Fix string conversion for -0.0 float values ([GH-81328](https://github.com/godotengine/godot/pull/81328)).
- Fix `SIGN(NAN)` returning 1 ([GH-81464](https://github.com/godotengine/godot/pull/81464)).
- Use pass by reference in ZIPPacker & ZIPReader signatures ([GH-81501](https://github.com/godotengine/godot/pull/81501)).
- Add `Vector2/3/4i.MAX` and `MIN` ([GH-81741](https://github.com/godotengine/godot/pull/81741)).
- Crypto: Fix `generate_random_bytes` for large chunks ([GH-81884](https://github.com/godotengine/godot/pull/81884)).
- Fix allocation size overflow check in `CowData` ([GH-81917](https://github.com/godotengine/godot/pull/81917)).
- Remove unnecessary line from `Projection::get_z_far` ([GH-81986](https://github.com/godotengine/godot/pull/81986)).
- Make all render driver project settings require restart ([GH-82008](https://github.com/godotengine/godot/pull/82008)).
- Add support for ImageTexture3D serialization ([GH-82055](https://github.com/godotengine/godot/pull/82055)).
- Made signal handling more uniform during crashes ([GH-82163](https://github.com/godotengine/godot/pull/82163)).
- Replace `radians` range hint with `radians_as_degrees` ([GH-82195](https://github.com/godotengine/godot/pull/82195)).
- Fix not being able to set Node process priority in certain cases ([GH-82358](https://github.com/godotengine/godot/pull/82358)).
- Fix axis getting mixed up when split leaf ([GH-82436](https://github.com/godotengine/godot/pull/82436)).
- Error handling for `FileAccess.get_file_as_*` ([GH-82595](https://github.com/godotengine/godot/pull/82595)).
- Do not replace starting digit with underscore when making identifier ([GH-82786](https://github.com/godotengine/godot/pull/82786)).
- Fix `RBMap`'s, iterator-based, `remove()` ([GH-82797](https://github.com/godotengine/godot/pull/82797)).
- Add missing double-precision flag for Vector4 & Projection in `encode_variant` ([GH-83202](https://github.com/godotengine/godot/pull/83202)).
- Update `triangulate_delaunay()` to avoid needless reallocations ([GH-83434](https://github.com/godotengine/godot/pull/83434)).
- Fix heap-use-after-free when resource loaded with `load_threaded_request` is never fetched ([GH-83782](https://github.com/godotengine/godot/pull/83782)).
- Fix `FastNoiseLite.get_seamless_image` function crash with bad param ([GH-83978](https://github.com/godotengine/godot/pull/83978)).
- Fix invalid return from some `_get/_set` ([GH-84054](https://github.com/godotengine/godot/pull/84054)).
- Prevent `encode_variant` doing `memcpy` from `nullptr` ([GH-84155](https://github.com/godotengine/godot/pull/84155)).
- Fix uninitialized variable in `Image::fix_alpha_edges()` ([GH-84173](https://github.com/godotengine/godot/pull/84173)).
- Fix `FastNoiseLite.get_image` crashes with bad param ([GH-84181](https://github.com/godotengine/godot/pull/84181)).
- Add comment why off-by-one error is hard to fix ([GH-84297](https://github.com/godotengine/godot/pull/84297)).
- Remove unused `NOTIFICATION_NODE_RECACHE_REQUESTED` notification ([GH-84419](https://github.com/godotengine/godot/pull/84419)).
- Keep Variant type after `zero()` ([GH-84597](https://github.com/godotengine/godot/pull/84597)).
- Make languages bookkeeping thread-safe ([GH-84657](https://github.com/godotengine/godot/pull/84657)).
- Fix crash when saving compressed image as JPG & WebP ([GH-84758](https://github.com/godotengine/godot/pull/84758)).
- Fix translation remapping check for imported resources ([GH-84791](https://github.com/godotengine/godot/pull/84791)).
- Let languages init & finish run without locks held ([GH-84847](https://github.com/godotengine/godot/pull/84847)).
- Fix `sizeof` usage for Variant pointers in `alloca` ([GH-84925](https://github.com/godotengine/godot/pull/84925)).
- Let scene replacement benefit from certain late pieces of frame logic ([GH-85184](https://github.com/godotengine/godot/pull/85184)).
- Prevent read-after-free in the queued CallableCustomStaticMethodPointer, fixes `slot >= slot_max` errors in release templates ([GH-85280](https://github.com/godotengine/godot/pull/85280)).

#### Documentation

- Document when to use `_unhandled_key_input` over `_unhandled_input` ([GH-42100](https://github.com/godotengine/godot/pull/42100)).
- Overhaul Rect2 & Rect2i Documentation ([GH-69816](https://github.com/godotengine/godot/pull/69816)).
- Improve documentation of `nearest_po2()` ([GH-72091](https://github.com/godotengine/godot/pull/72091)).
- Improve the top docs sections of VFX classes ([GH-78865](https://github.com/godotengine/godot/pull/78865)).
- Fix a typo in the `String.to_float` description ([GH-78989](https://github.com/godotengine/godot/pull/78989)).
- Fix a typo in the TLSOptions documentation ([GH-79048](https://github.com/godotengine/godot/pull/79048)).
- Doctool: Remove version attribute from XML header ([GH-79092](https://github.com/godotengine/godot/pull/79092)).
- Fix incorrect documentation for `Engine.get_architecture_name()` ([GH-79174](https://github.com/godotengine/godot/pull/79174)).
- Fix various typos in documentation ([GH-79223](https://github.com/godotengine/godot/pull/79223)).
- Fix rigid body `contact_monitor` property description ([GH-79250](https://github.com/godotengine/godot/pull/79250)).
- Add performance note for parsing source geometry ([GH-79252](https://github.com/godotengine/godot/pull/79252)).
- Clarify return value of `get_dependencies()` ([GH-79306](https://github.com/godotengine/godot/pull/79306)).
- Clarify `EditorExportPlugin::add_file` only remaps in `_export_file` ([GH-79310](https://github.com/godotengine/godot/pull/79310)).
- Fix documentation for consistency ([GH-79353](https://github.com/godotengine/godot/pull/79353)).
- Add detail to NavigationAgent signal descriptions ([GH-79401](https://github.com/godotengine/godot/pull/79401)).
- Fill in descriptions for import options in the class reference ([GH-79405](https://github.com/godotengine/godot/pull/79405)).
- Clarify the purpose of RichTextLabel text highlight padding ([GH-79571](https://github.com/godotengine/godot/pull/79571)).
- Clarify when the `changed` signal is emitted for `Resource` and `Material` ([GH-79656](https://github.com/godotengine/godot/pull/79656)).
- Fix many typos in documentation ([GH-79661](https://github.com/godotengine/godot/pull/79661)).
- Add detail to emitting docs for particles ([GH-79720](https://github.com/godotengine/godot/pull/79720)).
- Clarify `set_multiplayer_authority` documentation regarding propagation ([GH-79764](https://github.com/godotengine/godot/pull/79764)).
- Fix some mixups between 2D/3D in documentation ([GH-79781](https://github.com/godotengine/godot/pull/79781)).
- Update outdated C# code sample in `AStarGrid2D` documentation ([GH-79794](https://github.com/godotengine/godot/pull/79794)).
- Document `linear_stiffness` in SoftBody3D ([GH-79933](https://github.com/godotengine/godot/pull/79933)).
- Add note about mouse movement input events in `MouseFilter` constants ([GH-79934](https://github.com/godotengine/godot/pull/79934)).
- Update C# example of `tween_method` with a parameter to the lambda method ([GH-79962](https://github.com/godotengine/godot/pull/79962)).
- Avoid punning `[param enabled]` in documentation ([GH-80066](https://github.com/godotengine/godot/pull/80066)).
- Fix invalid use of markdown syntax in classref ([GH-80109](https://github.com/godotengine/godot/pull/80109)).
- Overhaul bool documentation ([GH-80141](https://github.com/godotengine/godot/pull/80141)).
- Fix packet details of ENetConnection `EventType` `EVENT_RECEIVE` documentation ([GH-80182](https://github.com/godotengine/godot/pull/80182)).
- Document that `Input.is_action_*` should not be used during input-handling ([GH-80185](https://github.com/godotengine/godot/pull/80185)).
- Revert incorrect `Rect2.expand` description ([GH-80217](https://github.com/godotengine/godot/pull/80217)).
- Fix various typos in classref ([GH-80243](https://github.com/godotengine/godot/pull/80243)).
- Fix wrong example output of `float*Color` in classref ([GH-80245](https://github.com/godotengine/godot/pull/80245)).
- Clarify relationship between `basis` and `transform` properties of `Node3D` ([GH-80254](https://github.com/godotengine/godot/pull/80254)).
- Clarify `SceneTree.current_scene` functionality ([GH-80264](https://github.com/godotengine/godot/pull/80264)).
- Consistency for NodePath doc code examples ([GH-80443](https://github.com/godotengine/godot/pull/80443)).
- Document `RenderingServer.get_video_adapter_name()` may report a fixed name ([GH-80445](https://github.com/godotengine/godot/pull/80445)).
- Fixed tiny spelling error in NavigationAgent2D docs ([GH-80474](https://github.com/godotengine/godot/pull/80474)).
- Fix example for `Object._set` documentation ([GH-80475](https://github.com/godotengine/godot/pull/80475)).
- Document behavior of the `application/config/version` project setting ([GH-80511](https://github.com/godotengine/godot/pull/80511)).
- Clarify the behavior of CSGMesh using ArrayMesh ([GH-80543](https://github.com/godotengine/godot/pull/80543)).
- Change documentation spelling of macOS key 'Command' to match guidelines ([GH-80567](https://github.com/godotengine/godot/pull/80567)).
- docs: Fix link to two's complement wiki page ([GH-80608](https://github.com/godotengine/godot/pull/80608)).
- Add missing tutorials to documentation classes ([GH-80653](https://github.com/godotengine/godot/pull/80653)).
- Clarify existence of groups ([GH-80677](https://github.com/godotengine/godot/pull/80677)).
- Document `pick_random` for empty arrays ([GH-80694](https://github.com/godotengine/godot/pull/80694)).
- Fix empty XML tag doc in XMLParser.xml ([GH-80698](https://github.com/godotengine/godot/pull/80698)).
- Add missing RenderingDevice method descriptions ([GH-80716](https://github.com/godotengine/godot/pull/80716)).
- Document mouse-picking limit of 64 objects ([GH-80875](https://github.com/godotengine/godot/pull/80875)).
- Fix various typos in classref ([GH-80884](https://github.com/godotengine/godot/pull/80884)).
- Clarify Array class methods that return error ([GH-80936](https://github.com/godotengine/godot/pull/80936)).
- Rephrase ConfigFile class methods error description ([GH-80970](https://github.com/godotengine/godot/pull/80970)).
- Improve XMLParser's documentation ([GH-80997](https://github.com/godotengine/godot/pull/80997)).
- Improve Line2D documentation ([GH-81084](https://github.com/godotengine/godot/pull/81084)).
- Add a note about `SceneTree.create_tween()` method ([GH-81087](https://github.com/godotengine/godot/pull/81087)).
- Improve `MeshDataTool.get_face_vertex()` method description ([GH-81088](https://github.com/godotengine/godot/pull/81088)).
- Improve `Object.get_property_list()` method description ([GH-81093](https://github.com/godotengine/godot/pull/81093)).
- Some grammar and punctuation enhancements in the class reference ([GH-81097](https://github.com/godotengine/godot/pull/81097)).
- Grammatical improvements for the RayCast 2D and 3D class references ([GH-81132](https://github.com/godotengine/godot/pull/81132)).
- Fix typo in WebRTCPeerConnection documentation ([GH-81157](https://github.com/godotengine/godot/pull/81157)).
- Document integer scaling functionality and limitation ([GH-81176](https://github.com/godotengine/godot/pull/81176)).
- Fix typos in NavigationAgent3D documentation ([GH-81190](https://github.com/godotengine/godot/pull/81190)).
- Fix misleading description of `MeshDataTool.get_vertex()` method ([GH-81212](https://github.com/godotengine/godot/pull/81212)).
- Use `[constant]` instead of `[code]` when possible ([GH-81228](https://github.com/godotengine/godot/pull/81228)).
- Fix typos in LineEdit documentation ([GH-81232](https://github.com/godotengine/godot/pull/81232)).
- docs: Update AABB `get_support` description ([GH-81249](https://github.com/godotengine/godot/pull/81249)).
- Improve canvas layer index documentation ([GH-81270](https://github.com/godotengine/godot/pull/81270)).
- Fix unmatched brackets in the documentation ([GH-81330](https://github.com/godotengine/godot/pull/81330)).
- Fix description of dock slot usage in the documentation ([GH-81445](https://github.com/godotengine/godot/pull/81445)).
- Document ScrollContainer signals being emitted for touch events only ([GH-81517](https://github.com/godotengine/godot/pull/81517)).
- Doc: Reference String <-> PackedByteArray conversions from each other ([GH-81564](https://github.com/godotengine/godot/pull/81564)).
- Fix typos in EditorDebuggerPlugin and RDShaderSPIRV classref ([GH-81565](https://github.com/godotengine/godot/pull/81565)).
- Add an example for `Dictionary.merge()`, mention lack of recursion ([GH-81622](https://github.com/godotengine/godot/pull/81622)).
- Add missing `is_deprecated` flag on the `SurfaceTool.generate_lod` function ([GH-81634](https://github.com/godotengine/godot/pull/81634)).
- Add note about format to splash image description ([GH-81672](https://github.com/godotengine/godot/pull/81672)).
- Add missing documentation for `Skeleton3D` methods ([GH-81697](https://github.com/godotengine/godot/pull/81697)).
- Improve VisibleOnScreen classes' docs ([GH-81774](https://github.com/godotengine/godot/pull/81774)).
- Fix required parameter values for 2D textures in `RenderingDevice.texture_clear()` ([GH-81936](https://github.com/godotengine/godot/pull/81936)).
- Fix example in gravity project settings doc ([GH-81967](https://github.com/godotengine/godot/pull/81967)).
- docs: Fix incorrect GL format code for 16 bit float formats ([GH-82050](https://github.com/godotengine/godot/pull/82050)).
- Fix documentation on how to get the keycode string from a `physical_keycode` ([GH-82092](https://github.com/godotengine/godot/pull/82092)).
- Docs: Update and sync Window and DisplayServer window mode descriptions ([GH-82179](https://github.com/godotengine/godot/pull/82179)).
- Document that `resource_name` is not always supported ([GH-82406](https://github.com/godotengine/godot/pull/82406)).
- Clarify difference between surface material and surface override material ([GH-82499](https://github.com/godotengine/godot/pull/82499)).
- Fix metadata name in MovieWriter.xml ([GH-82541](https://github.com/godotengine/godot/pull/82541)).
- Improve SeparationRayShape docs ([GH-82544](https://github.com/godotengine/godot/pull/82544)).
- Fix `RefCounted.unreference()` documentation providing wrong info ([GH-82557](https://github.com/godotengine/godot/pull/82557)).
- Document `get_time_zone_from_system` will return a localized timezone name ([GH-82609](https://github.com/godotengine/godot/pull/82609)).
- Improve `NavigationAgent3D.target_position` documentation readability ([GH-82671](https://github.com/godotengine/godot/pull/82671)).
- Add docs for Node3DGizmo to clarify its link to EditorNode3DGizmo ([GH-82681](https://github.com/godotengine/godot/pull/82681)).
- Clarify `AStarGrid2D.is_in_bounds` functionality ([GH-82724](https://github.com/godotengine/godot/pull/82724)).
- Fix typos in documentation: `than/then` and `loose/lose` ([GH-82748](https://github.com/godotengine/godot/pull/82748)).
- Add a recommendation to turn on type hints with untyped declaration warning ([GH-82801](https://github.com/godotengine/godot/pull/82801)).
- Clarify `change_dir()` and access scopes ([GH-82849](https://github.com/godotengine/godot/pull/82849)).
- Specify the behavior of `get_tree()` when the node is not in the scene tree ([GH-82863](https://github.com/godotengine/godot/pull/82863)).
- Added docs for DRAW_ORDER_REVERSE_LIFETIME constant and minor XR log improvement ([GH-82866](https://github.com/godotengine/godot/pull/82866)).
- Fixed a missing word ([GH-82883](https://github.com/godotengine/godot/pull/82883)).
- Add `sdf_collision` property description to LightOccluder2D ([GH-82906](https://github.com/godotengine/godot/pull/82906)).
- Explain circular references and how to break them ([GH-82942](https://github.com/godotengine/godot/pull/82942)).
- Update `draw_polyline` documentation to clarify negative width behavior ([GH-82991](https://github.com/godotengine/godot/pull/82991)).
- Add documentation on which buttons JOY_BUTTON_START corresponds to ([GH-83013](https://github.com/godotengine/godot/pull/83013)).
- Update SpinBox documentation to include resetting to min/max behavior ([GH-83038](https://github.com/godotengine/godot/pull/83038)).
- Add semicolon to OS documentation case statement ([GH-83066](https://github.com/godotengine/godot/pull/83066)).
- Cleanup various repository documentation files ([GH-83095](https://github.com/godotengine/godot/pull/83095)).
- Make error suggestion less ambiguous ([GH-83327](https://github.com/godotengine/godot/pull/83327)).
- Document UID behavior in ResourceSaver's save function ([GH-83388](https://github.com/godotengine/godot/pull/83388)).
- Docs: Fix link to Android Gradle build tutorial ([GH-83433](https://github.com/godotengine/godot/pull/83433)).
- Document `AudioStreamGeneratorPlayback.get_skips()` ([GH-83435](https://github.com/godotengine/godot/pull/83435)).
- Fix description of `Animation::copy_track` ([GH-83441](https://github.com/godotengine/godot/pull/83441)).
- Clarify docs for operators performing `xform_inv` ([GH-83461](https://github.com/godotengine/godot/pull/83461)).
- Doc: Change return type of `_Set` method from `void` to `bool` in C# code example ([GH-83602](https://github.com/godotengine/godot/pull/83602)).
- Fix Object class C# syntax error ([GH-83609](https://github.com/godotengine/godot/pull/83609)).
- Clarify `NOTIFICATION_SCROLL_BEGIN/END` behavior ([GH-83636](https://github.com/godotengine/godot/pull/83636)).
- Fill remaining global scope constant descriptions ([GH-83652](https://github.com/godotengine/godot/pull/83652)).
- ProjectSettings: Fix description of physics jitter ([GH-83768](https://github.com/godotengine/godot/pull/83768)).
- Add C# Example to ImmediateMesh.xml ([GH-83839](https://github.com/godotengine/godot/pull/83839)).
- Improve documentation related for particle subemitters, collision and attractors ([GH-83916](https://github.com/godotengine/godot/pull/83916)).
- Fill out Material documentation and clarify `render_priority` and `next_pass` sorting ([GH-83931](https://github.com/godotengine/godot/pull/83931)).
- Fixed `window_width_override` description ([GH-84101](https://github.com/godotengine/godot/pull/84101)).
- Fix typo in ConcavePolygonShape2D/3D description ([GH-84111](https://github.com/godotengine/godot/pull/84111)).
- Add missing word in `NOTIFICATION_POST_ENTER_TREE` documentation ([GH-84224](https://github.com/godotengine/godot/pull/84224)).
- Fix documentation in MultiplayerAPIExtension ([GH-84226](https://github.com/godotengine/godot/pull/84226)).
- Add a description for the `velocity_pivot` parameter ([GH-84276](https://github.com/godotengine/godot/pull/84276)).
- Update `add_submenu_item` doc to mention that submenu should already exist ([GH-84283](https://github.com/godotengine/godot/pull/84283)).
- Clarify that `get_time_zone_from_system` will return a localized timezone name ([GH-84301](https://github.com/godotengine/godot/pull/84301)).
- Fix sentence in RandomNumberGenerator.xml ([GH-84322](https://github.com/godotengine/godot/pull/84322)).
- Update the description for the `InputEventMagnifyGesture` and `InputEventPanGesture` gestures ([GH-84408](https://github.com/godotengine/godot/pull/84408)).
- Sync changes between ShapeCast and RayCast class references ([GH-84567](https://github.com/godotengine/godot/pull/84567)).
- Resolve collisions in reference anchors added for methods ([GH-84618](https://github.com/godotengine/godot/pull/84618)).
- Add C# example for the AudioStreamGenerator code snippet ([GH-84648](https://github.com/godotengine/godot/pull/84648)).
- Remove a redundant semicolon from `max_fps` documentation ([GH-84667](https://github.com/godotengine/godot/pull/84667)).
- Clarify that `DisplayServer.window_set_*_callback` aren't supported on Window nodes ([GH-84669](https://github.com/godotengine/godot/pull/84669)).
- Fix link in the docs about ResourceImporterTextureAtlas ([GH-84698](https://github.com/godotengine/godot/pull/84698)).
- Fix a property reference in `EditorSpinSlider` documentation ([GH-84709](https://github.com/godotengine/godot/pull/84709)).
- Fix typo in `TextureServer.font_get_face_index()` description ([GH-84784](https://github.com/godotengine/godot/pull/84784)).
- Link to runtime loading/saving tutorial and improve Image documentation ([GH-84844](https://github.com/godotengine/godot/pull/84844)).
- Mark `SubViewportContainer::_propagate_input_event` experimental ([GH-84911](https://github.com/godotengine/godot/pull/84911)).
- Fix translation po file not found when `make rst LANGARG=zh_CN` ([GH-85073](https://github.com/godotengine/godot/pull/85073)).
- Enhance `SceneTree.change_scene*()` methods' docs ([GH-85279](https://github.com/godotengine/godot/pull/85279)).
- Add changelog for Godot 4.2 ([GH-85510](https://github.com/godotengine/godot/pull/85509)).

#### Editor

- Replace all flags with one value when holding Ctrl/Cmd in the layers editor ([GH-39364](https://github.com/godotengine/godot/pull/39364)).
- Improve `CodeEdit`'s toggle comments behavior ([GH-44557](https://github.com/godotengine/godot/pull/44557)).
- Document editor import options in the class reference ([GH-49524](https://github.com/godotengine/godot/pull/49524)).
- Reorganize buttons in the project manager ([GH-50674](https://github.com/godotengine/godot/pull/50674)).
- Streamline the project import workflow ([GH-51478](https://github.com/godotengine/godot/pull/51478)).
- Focus current node after connecting ([GH-54071](https://github.com/godotengine/godot/pull/54071)).
- Allow enter key to add properties to replication editor list ([GH-65558](https://github.com/godotengine/godot/pull/65558)).
- Add editor setting to toggle automatic code completion ([GH-68140](https://github.com/godotengine/godot/pull/68140)).
- Replace Ctrl in editor shortcuts with Cmd or Ctrl depending on platform ([GH-71905](https://github.com/godotengine/godot/pull/71905)).
- Overhaul the Gradient Editor ([GH-71915](https://github.com/godotengine/godot/pull/71915)).
- Don't save scripts when exiting editor ([GH-73641](https://github.com/godotengine/godot/pull/73641)).
- Fix Filter Files shortcut input is not properly handled ([GH-73981](https://github.com/godotengine/godot/pull/73981)).
- Fix conversion of hex color strings in project converter ([GH-74026](https://github.com/godotengine/godot/pull/74026)).
- Add coloring for completion of vector components ([GH-74809](https://github.com/godotengine/godot/pull/74809)).
- Expose 'Reimport' on right-click context menu in the FileSystem panel ([GH-75137](https://github.com/godotengine/godot/pull/75137)).
- Added `--gpu-index` to `forwardable_cli_arguments` ([GH-75198](https://github.com/godotengine/godot/pull/75198)).
- Enhance NodePath property editing ([GH-75274](https://github.com/godotengine/godot/pull/75274)).
- Ensure binds are duplicated with `Node` signals ([GH-75382](https://github.com/godotengine/godot/pull/75382)).
- Make `EditorInterface` accessible as a singleton ([GH-75694](https://github.com/godotengine/godot/pull/75694)).
- Apply new input validation method for Create Plugin dialog ([GH-76778](https://github.com/godotengine/godot/pull/76778)).
- Expose `save_all_scenes` method to EditorInterface ([GH-77537](https://github.com/godotengine/godot/pull/77537)).
- Increase vertical size of `CurveEdit` when `Inspector` widens ([GH-77625](https://github.com/godotengine/godot/pull/77625)).
- Allow to pick which Resources will be made unique ([GH-77855](https://github.com/godotengine/godot/pull/77855)).
- Fix batch rename for unique name and empty name ([GH-78292](https://github.com/godotengine/godot/pull/78292)).
- Change light themes default contrast from -0.08 to -0.06 ([GH-78297](https://github.com/godotengine/godot/pull/78297)).
- Auto-update properties when replacing a node ([GH-78300](https://github.com/godotengine/godot/pull/78300)).
- Only display 15 nodes in the Recent section of the Create New Node dialog ([GH-78309](https://github.com/godotengine/godot/pull/78309)).
- Fix tooltip of enum value without description ([GH-78524](https://github.com/godotengine/godot/pull/78524)).
- Speed up closing multiple scripts ([GH-78604](https://github.com/godotengine/godot/pull/78604)).
- Re-enable docs cache with fixes ([GH-78615](https://github.com/godotengine/godot/pull/78615)).
- Use bullet points in shader editor creation dialog ([GH-78631](https://github.com/godotengine/godot/pull/78631)).
- Tweak documentation to use bold font when a class is referencing itself ([GH-78649](https://github.com/godotengine/godot/pull/78649)).
- Fix indentation in script templates ([GH-78675](https://github.com/godotengine/godot/pull/78675)).
- Standardize dialog input validation as a new class ([GH-78744](https://github.com/godotengine/godot/pull/78744)).
- Sort project tags before saving ([GH-78775](https://github.com/godotengine/godot/pull/78775)).
- Project converter: Use same rendering driver as Project Manager ([GH-78795](https://github.com/godotengine/godot/pull/78795)).
- Fix drag-dropping nodes to parent with internal nodes ([GH-78816](https://github.com/godotengine/godot/pull/78816)).
- Fix history mismatch ([GH-78827](https://github.com/godotengine/godot/pull/78827)).
- Improve material and mesh preview buttons ([GH-78858](https://github.com/godotengine/godot/pull/78858)).
- Add icons for 3D texture classes ([GH-78903](https://github.com/godotengine/godot/pull/78903)).
- Fix dropping files from `res://` to `res://` ([GH-78914](https://github.com/godotengine/godot/pull/78914)).
- Do not change a node unique name to the same name ([GH-78925](https://github.com/godotengine/godot/pull/78925)).
- Translate "No match" message in FindReplaceBar ([GH-78938](https://github.com/godotengine/godot/pull/78938)).
- Windows: Always double-quote path when launching explorer.exe to browse ([GH-78963](https://github.com/godotengine/godot/pull/78963)).
- [Terminal Output] Reset text properties after `print_rich` ([GH-79017](https://github.com/godotengine/godot/pull/79017)).
- Fix missing arrows in integer vector properties ([GH-79021](https://github.com/godotengine/godot/pull/79021)).
- Optimize SVG icons and remove unused Transpose icon ([GH-79062](https://github.com/godotengine/godot/pull/79062)).
- Collapse bottom panel if there is no active tab ([GH-79078](https://github.com/godotengine/godot/pull/79078)).
- Fix `ui_cancel` action not closing `FindReplaceBar` ([GH-79079](https://github.com/godotengine/godot/pull/79079)).
- Add tooltip description wrapping in scene tree and plugin settings ([GH-79090](https://github.com/godotengine/godot/pull/79090)).
- Improve user-friendliness of project version mismatch message ([GH-79118](https://github.com/godotengine/godot/pull/79118)).
- Optimize Variant icons and a few others ([GH-79161](https://github.com/godotengine/godot/pull/79161)).
- Don't grab theme icons for scripts ([GH-79203](https://github.com/godotengine/godot/pull/79203)).
- Show only compatible nodes in 'Select a node' window ([GH-79213](https://github.com/godotengine/godot/pull/79213)).
- Assume root when dropping node to unassigned script ([GH-79258](https://github.com/godotengine/godot/pull/79258)).
- Keep `GraphNode` port icons crisp at high zoom levels and remove artifacts ([GH-79262](https://github.com/godotengine/godot/pull/79262)).
- Hide/show `AcceptDialog`'s button spacer on button visibility changed ([GH-79274](https://github.com/godotengine/godot/pull/79274)).
- Change explicit 'Godot 4.0' references to 'Godot 4' ([GH-79277](https://github.com/godotengine/godot/pull/79277)).
- Fix dragged nodes icon size ([GH-79283](https://github.com/godotengine/godot/pull/79283)).
- Improve text in popup warning, remove "upgrade or downgrade" text ([GH-79299](https://github.com/godotengine/godot/pull/79299)).
- Allow adding a custom side menu to EditorFileDialog ([GH-79313](https://github.com/godotengine/godot/pull/79313)).
- Make indentation indicators translatable ([GH-79358](https://github.com/godotengine/godot/pull/79358)).
- Improve signal callback generation ([GH-79366](https://github.com/godotengine/godot/pull/79366)).
- Add missing word to text of the alert dialog ([GH-79381](https://github.com/godotengine/godot/pull/79381)).
- Disable irrelevant scene tab context menu items ([GH-79382](https://github.com/godotengine/godot/pull/79382)).
- Don't use splash minimum display time in editor ([GH-79388](https://github.com/godotengine/godot/pull/79388)).
- Include display server type in "Copy System Info" ([GH-79396](https://github.com/godotengine/godot/pull/79396)).
- Fix rendering driver in Copy System Info for the Compatibility rendering method ([GH-79416](https://github.com/godotengine/godot/pull/79416)).
- Add icons to some placeholder classes ([GH-79431](https://github.com/godotengine/godot/pull/79431)).
- Hide explicitly specified flag value in Inspector ([GH-79457](https://github.com/godotengine/godot/pull/79457)).
- Add a shortcut to paste nodes as sibling of the selected node ([GH-79467](https://github.com/godotengine/godot/pull/79467)).
- Emit `history_changed` on merged UndoRedo actions ([GH-79484](https://github.com/godotengine/godot/pull/79484)).
- Show valid types in SceneTreeDialog ([GH-79593](https://github.com/godotengine/godot/pull/79593)).
- Fix wrong Curve connection ([GH-79609](https://github.com/godotengine/godot/pull/79609)).
- Add Ctrl+/ as a shortcut to toggle comment in addition to Ctrl+K ([GH-79610](https://github.com/godotengine/godot/pull/79610)).
- Make Help.svg not look disabled ([GH-79613](https://github.com/godotengine/godot/pull/79613)).
- Avoid duplicating the "Filters" section ([GH-79650](https://github.com/godotengine/godot/pull/79650)).
- Fix arg count checks in `SceneDebugger` ([GH-79655](https://github.com/godotengine/godot/pull/79655)).
- Add placeholder items to TileSet layer list ([GH-79676](https://github.com/godotengine/godot/pull/79676)).
- Change the text for the flat button preview to follow pattern ([GH-79734](https://github.com/godotengine/godot/pull/79734)).
- Fix typo in ResourceImporterImageFont ([GH-79736](https://github.com/godotengine/godot/pull/79736)).
- In Create New Scene dialog derive the default root node name based on `editor/naming/node_name_casing` ([GH-79756](https://github.com/godotengine/godot/pull/79756)).
- Make the single window mode check more strict ([GH-79793](https://github.com/godotengine/godot/pull/79793)).
- Make blend file importer warnings translatable ([GH-79807](https://github.com/godotengine/godot/pull/79807)).
- Fix undo methods for DELETE in EditorAutoloadSettings ([GH-79832](https://github.com/godotengine/godot/pull/79832)).
- Fix usability issues with scene tabs ([GH-79852](https://github.com/godotengine/godot/pull/79852)).
- Add tooltips to the plugin editor creation dialog ([GH-79891](https://github.com/godotengine/godot/pull/79891)).
- Fix spacing between icon and "Output" button ([GH-79908](https://github.com/godotengine/godot/pull/79908)).
- Fix crash when using "Close All Tabs" ([GH-79917](https://github.com/godotengine/godot/pull/79917)).
- Automatically add path to built-in scripts ([GH-79920](https://github.com/godotengine/godot/pull/79920)).
- Sort system font menu in Inspector ([GH-79928](https://github.com/godotengine/godot/pull/79928)).
- Fix out of bounds access when updating current scene ([GH-79945](https://github.com/godotengine/godot/pull/79945)).
- Uncollapse favorites by default in the editor FileSystem dock ([GH-79971](https://github.com/godotengine/godot/pull/79971)).
- Reverse condition for skipping directories ([GH-79984](https://github.com/godotengine/godot/pull/79984)).
- Fix escaping issues with POT generator ([GH-80058](https://github.com/godotengine/godot/pull/80058)).
- Fix API hash related crash in `EditorSettings` ([GH-80089](https://github.com/godotengine/godot/pull/80089)).
- Add UndoRedo icon ([GH-80102](https://github.com/godotengine/godot/pull/80102)).
- Add FileAccess and DirAccess icons ([GH-80103](https://github.com/godotengine/godot/pull/80103)).
- Add path to missing import texture metadata to error message ([GH-80107](https://github.com/godotengine/godot/pull/80107)).
- Add an icon to the Performance object ([GH-80113](https://github.com/godotengine/godot/pull/80113)).
- Optimize and fix up some SVGs ([GH-80119](https://github.com/godotengine/godot/pull/80119)).
- Add ShaderInclude class icon ([GH-80129](https://github.com/godotengine/godot/pull/80129)).
- Use the gray color for all abstract classes ([GH-80184](https://github.com/godotengine/godot/pull/80184)).
- Horizontal split view for Filesystem Dock ([GH-80241](https://github.com/godotengine/godot/pull/80241)).
- Fix menu items that trigger secondary interface missing ellipsis ([GH-80355](https://github.com/godotengine/godot/pull/80355)).
- Improve Signal Dock for script classes ([GH-80411](https://github.com/godotengine/godot/pull/80411)).
- Add custom color support to project folders ([GH-80440](https://github.com/godotengine/godot/pull/80440)).
- [Editor Log] Clear rich print tags only after the last line ([GH-80476](https://github.com/godotengine/godot/pull/80476)).
- Extract editor scene tabs into their own component ([GH-80490](https://github.com/godotengine/godot/pull/80490)).
- Fixes Scene corruption when child scene is renamed in another directory ([GH-80503](https://github.com/godotengine/godot/pull/80503)).
- Avoid unnecessary inspector updates when loading or switching scenes ([GH-80517](https://github.com/godotengine/godot/pull/80517)).
- Add EditorStringNames singleton ([GH-80573](https://github.com/godotengine/godot/pull/80573)).
- Add CurveXYZTexture icon ([GH-80598](https://github.com/godotengine/godot/pull/80598)).
- Fix crash on exit where `TileSet` calls destroyed `TileSetAtlasSourceEditor` ([GH-80607](https://github.com/godotengine/godot/pull/80607)).
- Fix `TileMapEditorPlugin` crash by storing tilemap ID instead of pointer ([GH-80610](https://github.com/godotengine/godot/pull/80610)).
- Add PortableCompressedTexture2D icon ([GH-80659](https://github.com/godotengine/godot/pull/80659)).
- Make the NavigationAgent3D icon more readable ([GH-80661](https://github.com/godotengine/godot/pull/80661)).
- Recurse into resources to check for changed node paths ([GH-80721](https://github.com/godotengine/godot/pull/80721)).
- Add a RegEx icon ([GH-80724](https://github.com/godotengine/godot/pull/80724)).
- Don't cache script signal descriptions ([GH-80726](https://github.com/godotengine/godot/pull/80726)).
- Disable translation of root name on scene creation ([GH-80811](https://github.com/godotengine/godot/pull/80811)).
- Avoid creating any useless undo action when dragging nodes in place ([GH-80817](https://github.com/godotengine/godot/pull/80817)).
- Unedit nodes early when closing scene tab ([GH-80849](https://github.com/godotengine/godot/pull/80849)).
- Save "Show Built-In Actions" state to project metadata ([GH-80879](https://github.com/godotengine/godot/pull/80879)).
- Differentiate between core and editor-only singletons ([GH-80962](https://github.com/godotengine/godot/pull/80962)).
- Cleanup some `GLOBAL_DEF`s ([GH-80972](https://github.com/godotengine/godot/pull/80972)).
- Add a property hint range to Auto Refresh Interval editor setting ([GH-80975](https://github.com/godotengine/godot/pull/80975)).
- Display time of last save in the unsaved changes confirmation editor dialog ([GH-80976](https://github.com/godotengine/godot/pull/80976)).
- Fix paste value emptying an array on some right click location ([GH-80977](https://github.com/godotengine/godot/pull/80977)).
- Move the new RegEx icons into their respective module ([GH-80998](https://github.com/godotengine/godot/pull/80998)).
- FileSystemDock: Don't update current path on rename when file list has focus ([GH-81007](https://github.com/godotengine/godot/pull/81007)).
- Improve warnings when running scripts in the editor ([GH-81022](https://github.com/godotengine/godot/pull/81022)).
- Properly remember custom text color in scene tree ([GH-81061](https://github.com/godotengine/godot/pull/81061)).
- Fix Quick Open not opening binary resources ([GH-81068](https://github.com/godotengine/godot/pull/81068)).
- Refactor disabling scene tab context menu options ([GH-81072](https://github.com/godotengine/godot/pull/81072)).
- Prevent creating any type of file with a leading dot ([GH-81075](https://github.com/godotengine/godot/pull/81075)).
- Signal Connection Dock improvements ([GH-81092](https://github.com/godotengine/godot/pull/81092)).
- Fix a crash when built-in script is not saved and have syntax error ([GH-81156](https://github.com/godotengine/godot/pull/81156)).
- Use `ui_text_submit` instead of `ui_accept` to confirm and close text prompts ([GH-81189](https://github.com/godotengine/godot/pull/81189)).
- Inspector and Signal docks improvements ([GH-81221](https://github.com/godotengine/godot/pull/81221)).
- Fix `EditorFileDialog` clears the file name on changing directory ([GH-81226](https://github.com/godotengine/godot/pull/81226)).
- Fix clamping logic in `EditorSpinSlider` ([GH-81278](https://github.com/godotengine/godot/pull/81278)).
- Show doc tooltips when hovering properties in the theme editor ([GH-81284](https://github.com/godotengine/godot/pull/81284)).
- Change precedence in rules to make location after proper casing ([GH-81304](https://github.com/godotengine/godot/pull/81304)).
- Fix TextFile not reloading when changed from external editors ([GH-81319](https://github.com/godotengine/godot/pull/81319)).
- Check the native base of scripts when resolving icons ([GH-81336](https://github.com/godotengine/godot/pull/81336)).
- Fix saving editor folder colors ([GH-81344](https://github.com/godotengine/godot/pull/81344)).
- Avoid text substitution in EditorHelp messages ([GH-81346](https://github.com/godotengine/godot/pull/81346)).
- Update folder colors when moving or renaming ([GH-81380](https://github.com/godotengine/godot/pull/81380)).
- Rearrange "Main Menu > Help" items ([GH-81399](https://github.com/godotengine/godot/pull/81399)).
- Remove leftover debug print in `FileSystemDock` ([GH-81407](https://github.com/godotengine/godot/pull/81407)).
- Fix property array tooltip shows wrong ID on later pages ([GH-81408](https://github.com/godotengine/godot/pull/81408)).
- Fix bugs of copying scene root node or pasting node as scene root ([GH-81415](https://github.com/godotengine/godot/pull/81415)).
- Expose `EditorInspector::get_edited_object` to GDScript ([GH-81425](https://github.com/godotengine/godot/pull/81425)).
- Fix unexpected behaviors of using Duplicate To on folders ([GH-81437](https://github.com/godotengine/godot/pull/81437)).
- Fix FindReplaceBar losing focus too early ([GH-81450](https://github.com/godotengine/godot/pull/81450)).
- Ignore empty lines when uncommenting code ([GH-81486](https://github.com/godotengine/godot/pull/81486)).
- SceneTreeDock: Avoid changing the currently edited object when attaching a script ([GH-81510](https://github.com/godotengine/godot/pull/81510)).
- Allow contextual plugins to persist temporarily ([GH-81523](https://github.com/godotengine/godot/pull/81523)).
- Improve undo action names ([GH-81569](https://github.com/godotengine/godot/pull/81569)).
- Make editor support `--fullscreen` command-line argument ([GH-81608](https://github.com/godotengine/godot/pull/81608)).
- Add XML files to default TextFile extensions in the editor ([GH-81625](https://github.com/godotengine/godot/pull/81625)).
- Avoid resetting the code completion popup excessively ([GH-81633](https://github.com/godotengine/godot/pull/81633)).
- Fix dependency handling on move or rename in the filesystem dock ([GH-81657](https://github.com/godotengine/godot/pull/81657)).
- Don't paste nodes as sibling of scene root ([GH-81673](https://github.com/godotengine/godot/pull/81673)).
- Clarify filtering by node type and group in the Scene tree dock ([GH-81675](https://github.com/godotengine/godot/pull/81675)).
- Create a field when Ctrl-dropping a resource into the code editor ([GH-81708](https://github.com/godotengine/godot/pull/81708)).
- Make LineEdit secret character easier to change and enter ([GH-81724](https://github.com/godotengine/godot/pull/81724)).
- Fix folder moving in file system dock ([GH-81725](https://github.com/godotengine/godot/pull/81725)).
- Fix internal `CONNECT_INHERITED` being saved in PackedScene & Make Local ([GH-81737](https://github.com/godotengine/godot/pull/81737)).
- Fix Connection dock's popups always allowing disconnect ([GH-81750](https://github.com/godotengine/godot/pull/81750)).
- Change icon for position key ([GH-81751](https://github.com/godotengine/godot/pull/81751)).
- Add Ctrl+P as shortcut to quick open files in addition to Shift+Alt+O ([GH-81770](https://github.com/godotengine/godot/pull/81770)).
- Make editor camera speed indicator use `m/s` and `m` ([GH-81810](https://github.com/godotengine/godot/pull/81810)).
- Fix grayed out paint icons ([GH-81813](https://github.com/godotengine/godot/pull/81813)).
- Add CanvasTexture icon ([GH-81834](https://github.com/godotengine/godot/pull/81834)).
- Make UIDs clickable in the script editor ([GH-81927](https://github.com/godotengine/godot/pull/81927)).
- Improve the Torus icons ([GH-81978](https://github.com/godotengine/godot/pull/81978)).
- While dragging files don't move not selected cursor item in filesystem-dock ([GH-82045](https://github.com/godotengine/godot/pull/82045)).
- Revamp how documentation tooltips work ([GH-82051](https://github.com/godotengine/godot/pull/82051)).
- Fix several issues with renaming in FileSystem dock ([GH-82075](https://github.com/godotengine/godot/pull/82075)).
- Fix skeleton 3d editor's toolbar ui deleted from wrong container ([GH-82131](https://github.com/godotengine/godot/pull/82131)).
- Fix leak when calling `remove_control_from_menu_panel` ([GH-82171](https://github.com/godotengine/godot/pull/82171)).
- Fix CurveEdit crash when dragging the curve if it is null ([GH-82181](https://github.com/godotengine/godot/pull/82181)).
- Add call validation to CommandPalette ([GH-82194](https://github.com/godotengine/godot/pull/82194)).
- Remove the separator from ItemList's thumbnails mode ([GH-82236](https://github.com/godotengine/godot/pull/82236)).
- Fix missing dependency warning popup ([GH-82244](https://github.com/godotengine/godot/pull/82244)).
- Fix can't unset exported typed array element when the type is set to Node ([GH-82287](https://github.com/godotengine/godot/pull/82287)).
- Fix ScriptCreateDialog not accepting on submit ([GH-82328](https://github.com/godotengine/godot/pull/82328)).
- Add error checks for DirAccess creation ([GH-82347](https://github.com/godotengine/godot/pull/82347)).
- Color match editor log toggles and flat pressed buttons ([GH-82365](https://github.com/godotengine/godot/pull/82365)).
- Fix submenus deleted accidentally ([GH-82371](https://github.com/godotengine/godot/pull/82371)).
- Fix leak when closing theme editor preview tabs ([GH-82442](https://github.com/godotengine/godot/pull/82442)).
- Make terrains peering bit property names translatable ([GH-82509](https://github.com/godotengine/godot/pull/82509)).
- Don't remove favorite files in EditorFileDialog ([GH-82537](https://github.com/godotengine/godot/pull/82537)).
- Use theme icon size when calculating category minimum size ([GH-82540](https://github.com/godotengine/godot/pull/82540)).
- Add more context to some `Window` errors ([GH-82590](https://github.com/godotengine/godot/pull/82590)).
- "Whole Words" search can detect word boundaries inside the search term ([GH-82694](https://github.com/godotengine/godot/pull/82694)).
- Search terms are now highlighted when the bar opens with a selection ([GH-82707](https://github.com/godotengine/godot/pull/82707)).
- Fix node icons appearing too big in some cases ([GH-82728](https://github.com/godotengine/godot/pull/82728)).
- Fix loading floating dock layout ([GH-82742](https://github.com/godotengine/godot/pull/82742)).
- Removes extents to size conversion ([GH-82754](https://github.com/godotengine/godot/pull/82754)).
- Fix checking the visibility condition of selected file in the Godot editor's dock ([GH-82806](https://github.com/godotengine/godot/pull/82806)).
- Fix unsaved changes not getting discarded ([GH-82847](https://github.com/godotengine/godot/pull/82847)).
- Provide translation strings for folder colors ([GH-82858](https://github.com/godotengine/godot/pull/82858)).
- Fix debugger behavior with multi-session debugging ([GH-82868](https://github.com/godotengine/godot/pull/82868)).
- Disable disconnect button for inherited signals ([GH-82875](https://github.com/godotengine/godot/pull/82875)).
- Fix garbled text in editor toasters ([GH-82913](https://github.com/godotengine/godot/pull/82913)).
- Don't apply frame delay project setting to the editor ([GH-82929](https://github.com/godotengine/godot/pull/82929)).
- Tweak metadata property tooltip to avoid being misleading ([GH-82940](https://github.com/godotengine/godot/pull/82940)).
- Fix dependency menu not showing up if scene failed to load ([GH-83024](https://github.com/godotengine/godot/pull/83024)).
- Fix `EditorFileSystemDirectory::get_file_deps()` may return wrong result ([GH-83081](https://github.com/godotengine/godot/pull/83081)).
- Fix some issues with `EditorHelpTooltip` ([GH-83094](https://github.com/godotengine/godot/pull/83094)).
- Fix highlight rect in "Whole search" being slightly offset ([GH-83101](https://github.com/godotengine/godot/pull/83101)).
- Don't auto translate theme type list ([GH-83177](https://github.com/godotengine/godot/pull/83177)).
- Project Manager: Open project when "Enter" is pressed when the search box is focused ([GH-83210](https://github.com/godotengine/godot/pull/83210)).
- Disable port name auto translation in Visual Shader editor ([GH-83233](https://github.com/godotengine/godot/pull/83233)).
- Fix saving wrong edited scene state when switching scene tabs ([GH-83251](https://github.com/godotengine/godot/pull/83251)).
- Don't try updating wrong NodePaths in resources ([GH-83263](https://github.com/godotengine/godot/pull/83263)).
- Keep focus on floating window when showing ProgressDialog ([GH-83290](https://github.com/godotengine/godot/pull/83290)).
- Fix FindReplaceBar focus problems ([GH-83335](https://github.com/godotengine/godot/pull/83335)).
- Remove toggling of unique names in subscenes ([GH-83370](https://github.com/godotengine/godot/pull/83370)).
- Fix multiple comment delimiter break toggle comment shortcut ([GH-83382](https://github.com/godotengine/godot/pull/83382)).
- Disallow 'Make Local' command on inherited nodes ([GH-83386](https://github.com/godotengine/godot/pull/83386)).
- Disable "Edit Transitions..." item if no animations are present ([GH-83402](https://github.com/godotengine/godot/pull/83402)).
- Set `icon_max_width` in the ConnectionsDock tree ([GH-83447](https://github.com/godotengine/godot/pull/83447)).
- Fix close button in FindReplaceBar ([GH-83459](https://github.com/godotengine/godot/pull/83459)).
- Prevent crash when creating custom file tooltip ([GH-83487](https://github.com/godotengine/godot/pull/83487)).
- Mesh instance UV2 unwrapping improvements ([GH-83498](https://github.com/godotengine/godot/pull/83498)).
- Fix StringName leaks in GDExtension, core, and editor themes ([GH-83562](https://github.com/godotengine/godot/pull/83562)).
- Enable new addon after hiding ProjectSettings ([GH-83576](https://github.com/godotengine/godot/pull/83576)).
- Fix ownership bugs in node copy and pasting ([GH-83596](https://github.com/godotengine/godot/pull/83596)).
- Support duplication of foreign nodes ([GH-83597](https://github.com/godotengine/godot/pull/83597)).
- Fix crash on recovered orphaned nodes ([GH-83604](https://github.com/godotengine/godot/pull/83604)).
- Fix StringName leaks in VariantParser ([GH-83619](https://github.com/godotengine/godot/pull/83619)).
- Improve threading in ClassDB and EditorHelp ([GH-83695](https://github.com/godotengine/godot/pull/83695)).
- Fix wrong shader rename in 3-to-4 project converter ([GH-83708](https://github.com/godotengine/godot/pull/83708)).
- Clamp the height of description text for property selectors ([GH-83745](https://github.com/godotengine/godot/pull/83745)).
- Fix "as" capitalization in editor strings ([GH-83815](https://github.com/godotengine/godot/pull/83815)).
- Remove margins from editor scrollbars ([GH-83868](https://github.com/godotengine/godot/pull/83868)).
- Fix potential crash on failed move ([GH-83937](https://github.com/godotengine/godot/pull/83937)).
- Use Hashset for dependency list when moving ([GH-83941](https://github.com/godotengine/godot/pull/83941)).
- Limit custom icons size in various editor widgets ([GH-84011](https://github.com/godotengine/godot/pull/84011)).
- Add read-only info to resource embedded in other scenes ([GH-84048](https://github.com/godotengine/godot/pull/84048)).
- Ignore path error for built-in scripts/shaders ([GH-84077](https://github.com/godotengine/godot/pull/84077)).
- Change dropdown type filter from Texture to Texture2D in certain nodes ([GH-84113](https://github.com/godotengine/godot/pull/84113)).
- Fix file rename crash after toggling split mode ([GH-84217](https://github.com/godotengine/godot/pull/84217)).
- Fix crash on rename collision in thumbnail grid ([GH-84218](https://github.com/godotengine/godot/pull/84218)).
- Make remote inspector groups not foldable ([GH-84257](https://github.com/godotengine/godot/pull/84257)).
- Automatically pick the Android SDK path using environment variables ([GH-84285](https://github.com/godotengine/godot/pull/84285)).
- Fix pressing save in Import Defaults not working ([GH-84291](https://github.com/godotengine/godot/pull/84291)).
- Disconnect `EditorNode` from file dialogs on destruction ([GH-84302](https://github.com/godotengine/godot/pull/84302)).
- Fix CSGShape debug_collision_shape crash ([GH-84338](https://github.com/godotengine/godot/pull/84338)).
- Polish & fix editor help cache generation ([GH-84354](https://github.com/godotengine/godot/pull/84354)).
- Fix inverted condition when unwrapping lightmap ([GH-84374](https://github.com/godotengine/godot/pull/84374)).
- Fix engine configuration icons using old convention ([GH-84404](https://github.com/godotengine/godot/pull/84404)).
- Tweak FastNoiseLite property hints for better slider usability ([GH-84494](https://github.com/godotengine/godot/pull/84494)).
- Fix pressing Enter being ignored in "Create Shader" dialog ([GH-84539](https://github.com/godotengine/godot/pull/84539)).
- Fix for stopping the Undo History being desynchronized from actual Undo queue ([GH-84557](https://github.com/godotengine/godot/pull/84557)).
- Correctly set up shortcut context in the shader editor ([GH-84614](https://github.com/godotengine/godot/pull/84614)).
- Save scene when saving built-in resource ([GH-84630](https://github.com/godotengine/godot/pull/84630)).
- Abort threaded preview generators on exit ([GH-84716](https://github.com/godotengine/godot/pull/84716)).
- Fix texture region editor not selecting restored snap mode ([GH-84762](https://github.com/godotengine/godot/pull/84762)).
- Reduced output spam from rapid property changes ([GH-84795](https://github.com/godotengine/godot/pull/84795)).
- Remove EditorFileDialog warning when skipping project directories ([GH-84797](https://github.com/godotengine/godot/pull/84797)).
- macOS: Cleanup default GL driver setting ([GH-84929](https://github.com/godotengine/godot/pull/84929)).
- Make script/shader editor save shortcuts unique again ([GH-84931](https://github.com/godotengine/godot/pull/84931)).
- Provide more context when scene fails to load ([GH-85083](https://github.com/godotengine/godot/pull/85083)).
- Add Save As... option to EditorResourcePicker ([GH-85150](https://github.com/godotengine/godot/pull/85150)).
- Avoid saving scene while already saving the scene ([GH-85154](https://github.com/godotengine/godot/pull/85154)).
- Fix project name being overwritten every time `show_dialog` is called ([GH-85169](https://github.com/godotengine/godot/pull/85169)).
- Rework the surface upgrade tool to inform users without blocking ([GH-85222](https://github.com/godotengine/godot/pull/85222)).
- Fix crash caused by conflicting menu option IDs ([GH-85227](https://github.com/godotengine/godot/pull/85227)).
- Suppress surface upgrade warnings when showing SurfaceUpgradeTool warning ([GH-85249](https://github.com/godotengine/godot/pull/85249)).
- Save and restore previous window mode when toggling full-screen ([GH-85427](https://github.com/godotengine/godot/pull/85427)).
- Disable a prohibitively slow code branch when reparenting nodes ([GH-85517](https://github.com/godotengine/godot/pull/85517)).

#### Export

- Add a "version" project setting and use it in new export presets ([GH-35555](https://github.com/godotengine/godot/pull/35555)).
- Implement iOS one-click deploy ([GH-70662](https://github.com/godotengine/godot/pull/70662)).
- Add options to show icon in Android TV and run app as Android launcher ([GH-78164](https://github.com/godotengine/godot/pull/78164)).
- Add a button in the export dialog to fix missing texture formats ([GH-78457](https://github.com/godotengine/godot/pull/78457)).
- iOS: Add `export_project_only` flag ([GH-78641](https://github.com/godotengine/godot/pull/78641)).
- Re-architect how Android plugins are packaged and handled at export time ([GH-78958](https://github.com/godotengine/godot/pull/78958)).
- Fix export options of scripted `EditorExportPlugin`s ([GH-79025](https://github.com/godotengine/godot/pull/79025)).
- Android: Add option to always use WiFi to connect to remote debug ([GH-79504](https://github.com/godotengine/godot/pull/79504)).
- Improve headings for the export mode in the Export dialog ([GH-79725](https://github.com/godotengine/godot/pull/79725)).
- [macOS Export] Disable unpacked .app bundle export on Windows ([GH-79950](https://github.com/godotengine/godot/pull/79950)).
- Fix Windows console wrapper and icon being swapped ([GH-80357](https://github.com/godotengine/godot/pull/80357)).
- Add export setting to control whether to show the Godot app in the app library ([GH-80569](https://github.com/godotengine/godot/pull/80569)).
- Fix redundant enter tree notification in project export texture format ([GH-80967](https://github.com/godotengine/godot/pull/80967)).
- [iOS export] Switch export target extension based on export type ([GH-81365](https://github.com/godotengine/godot/pull/81365)).
- Expose `EditorExportPlatform::get_os_name()` ([GH-81430](https://github.com/godotengine/godot/pull/81430)).
- Fix `SubViewport` with `UPDATE_WHEN_VISIBLE` not working properly in exported project ([GH-81607](https://github.com/godotengine/godot/pull/81607)).
- [macOS export] Fix GDExtension framework `+x` flag errors, allow recursive signing on non macOS platform ([GH-81969](https://github.com/godotengine/godot/pull/81969)).
- Fix TextServer data export ([GH-82103](https://github.com/godotengine/godot/pull/82103)).
- iOS: Fix build on Xcode 14 and older ([GH-83088](https://github.com/godotengine/godot/pull/83088)).
- macOS: Remove deprecated altool notarization support, disable rcodesign for C# version ([GH-83482](https://github.com/godotengine/godot/pull/83482)).
- Use "version" project setting as macOS/iOS "short_version" fallback ([GH-83686](https://github.com/godotengine/godot/pull/83686)).
- Improve app / file version validation ([GH-84296](https://github.com/godotengine/godot/pull/84296)).
- [macOS export] Improve icon generation ([GH-84521](https://github.com/godotengine/godot/pull/84521)).
- Preserve the output from the gradle build command ([GH-84779](https://github.com/godotengine/godot/pull/84779)).
- Prevent the surface upgrade tool from running during export ([GH-85136](https://github.com/godotengine/godot/pull/85136)).
- iOS: Check if Xcode is installed in one-click deploy code ([GH-85168](https://github.com/godotengine/godot/pull/85168)).

#### GDExtension

- Fix GDExtension classes derived from abstract GDExtension classes always being registered as abstract ([GH-67512](https://github.com/godotengine/godot/pull/67512)).
- Add GDExtension support for OpenXR extension wrappers ([GH-68259](https://github.com/godotengine/godot/pull/68259)).
- Allow GDExtension to register unexposed classes ([GH-70329](https://github.com/godotengine/godot/pull/70329)).
- Set vararg methods' ptrcall of builtin classes, and let them can be called without arguments ([GH-76047](https://github.com/godotengine/godot/pull/76047)).
- Add GDExtension function to construct StringName directly from `char*` ([GH-78580](https://github.com/godotengine/godot/pull/78580)).
- Allow implementing `get_class_category` in GDExtension ([GH-78995](https://github.com/godotengine/godot/pull/78995)).
- Allow CallableCustom objects to be created from GDExtensions ([GH-79005](https://github.com/godotengine/godot/pull/79005)).
- Allow resizing Strings from GDExtension ([GH-79156](https://github.com/godotengine/godot/pull/79156)).
- Prevent GDExtensions from trying to remove editor plugins at shutdown ([GH-79492](https://github.com/godotengine/godot/pull/79492)).
- Fix `_get_property_list` not working correctly in parent classes ([GH-79683](https://github.com/godotengine/godot/pull/79683)).
- Add `_bind_compatibility_methods` to Object ([GH-79702](https://github.com/godotengine/godot/pull/79702)).
- Fix incorrect virtual function in `VideoStream.set_paused` ([GH-79710](https://github.com/godotengine/godot/pull/79710)).
- Add support for indexed properties in GDExtension ([GH-79763](https://github.com/godotengine/godot/pull/79763)).
- Add `get_script_instance` to GDExtension ([GH-80040](https://github.com/godotengine/godot/pull/80040)).
- `PtrToArg::convert()` uses const-reference where possible ([GH-80075](https://github.com/godotengine/godot/pull/80075)).
- Fix or workaround recent extension API compatibility issues ([GH-80168](https://github.com/godotengine/godot/pull/80168)).
- Copy DLL to a temp file before opening ([GH-80188](https://github.com/godotengine/godot/pull/80188)).
- CI: Make extension API compatibility check mandatory ([GH-80220](https://github.com/godotengine/godot/pull/80220)).
- Implement reloading of GDExtensions ([GH-80284](https://github.com/godotengine/godot/pull/80284)).
- Add compatibility notice after #78266 ([GH-80374](https://github.com/godotengine/godot/pull/80374)).
- Expose PlaceHolderScriptInstance to GDExtension ([GH-80394](https://github.com/godotengine/godot/pull/80394)).
- Fix version check for GDExtension ([GH-80591](https://github.com/godotengine/godot/pull/80591)).
- Use `String::resize()` and `CharString` in `text_server_adv` again ([GH-80642](https://github.com/godotengine/godot/pull/80642)).
- Add functions for non-ptr style virtual calls in GDExtension ([GH-80671](https://github.com/godotengine/godot/pull/80671)).
- SCons: Fix ThorVG build option in TextServers with #80095 ([GH-80713](https://github.com/godotengine/godot/pull/80713)).
- Remove DLL copy if it fails to load ([GH-80720](https://github.com/godotengine/godot/pull/80720)).
- Godot Android plugin re-architecture ([GH-80740](https://github.com/godotengine/godot/pull/80740)).
- Exclude unexposed classes from the `extension_api.json` ([GH-80852](https://github.com/godotengine/godot/pull/80852)).
- Fix overriding `_export_begin`, `_export_file` and `_export_end` from GDExtension ([GH-80999](https://github.com/godotengine/godot/pull/80999)).
- Allocate `GDExtensionScriptInstanceInfo2` for compatibility on the heap to prevent crash ([GH-81206](https://github.com/godotengine/godot/pull/81206)).
- Use godot-cpp 4.1 for the "Godot CPP" CI workflow to prevent circular dependency ([GH-81238](https://github.com/godotengine/godot/pull/81238)).
- fix `bool` unknown in C ([GH-81247](https://github.com/godotengine/godot/pull/81247)).
- Allow implementing `ScriptInstance::validate_property()` from GDExtension ([GH-81261](https://github.com/godotengine/godot/pull/81261)).
- Fix bindings of `PhysicsServer3DRenderingServerHandler` ([GH-81298](https://github.com/godotengine/godot/pull/81298)).
- Add compatibility methods for RenderingDevice BarrierMask ([GH-81356](https://github.com/godotengine/godot/pull/81356)).
- Allow implementing `Object::_validate_property()` from GDExtension ([GH-81515](https://github.com/godotengine/godot/pull/81515)).
- Fix method hashes with default arguments ([GH-81521](https://github.com/godotengine/godot/pull/81521)).
- Delete left-over DLL copy before making a new copy ([GH-81576](https://github.com/godotengine/godot/pull/81576)).
- Expose `texture_create_from_extension` to GDExtension ([GH-82168](https://github.com/godotengine/godot/pull/82168)).
- Remove redundant method bind hash check ([GH-82191](https://github.com/godotengine/godot/pull/82191)).
- Optionally include documentation in GDExtension API dump ([GH-82331](https://github.com/godotengine/godot/pull/82331)).
- Fix type of `notification_func` ([GH-82332](https://github.com/godotengine/godot/pull/82332)).
- Moved `face_index` field in 3D `RayResult` to end of struct ([GH-82403](https://github.com/godotengine/godot/pull/82403)).
- Fix inconsistent `last_modified_time` handling in GDExtension ([GH-82603](https://github.com/godotengine/godot/pull/82603)).
- Don't deprecate old method of getting script category ([GH-82682](https://github.com/godotengine/godot/pull/82682)).
- Fixes to allow object-less callables throughout Godot ([GH-82695](https://github.com/godotengine/godot/pull/82695)).
- Web: Catch using GDExtensions in a non-dlink build ([GH-82790](https://github.com/godotengine/godot/pull/82790)).
- Convert `validated_call()` to `ptrcall()` (rather than `call()`) ([GH-82794](https://github.com/godotengine/godot/pull/82794)).
- Expose `Object::free_instance_binding()` to GDExtension ([GH-82799](https://github.com/godotengine/godot/pull/82799)).
- Resolve relative icon paths for GDExtensions ([GH-82842](https://github.com/godotengine/godot/pull/82842)).
- Fix extensions loading/initializing even when entry point fails ([GH-82861](https://github.com/godotengine/godot/pull/82861)).
- Remove I/O error popup when failing to load/unload extension ([GH-82907](https://github.com/godotengine/godot/pull/82907)).
- On Linux, favor local symbols when loading a shared library ([GH-82973](https://github.com/godotengine/godot/pull/82973)).
- Use correct return pointer for validated calls that return `Variant` ([GH-83054](https://github.com/godotengine/godot/pull/83054)).
- Fix incorrect error message about vararg methods ([GH-83107](https://github.com/godotengine/godot/pull/83107)).
- Fix missing editor singletons when dumping extension api ([GH-83239](https://github.com/godotengine/godot/pull/83239)).
- Prevent issues with the editor trying to reload GDExtensions through its usual mechanism ([GH-83285](https://github.com/godotengine/godot/pull/83285)).
- Add brief description in GDExtension API dump with docs ([GH-83318](https://github.com/godotengine/godot/pull/83318)).
- Fix comment in `gdextension_interface.h` ([GH-83415](https://github.com/godotengine/godot/pull/83415)).
- Allow coexistence of GDScript and GDExtension virtual methods in the same object ([GH-83583](https://github.com/godotengine/godot/pull/83583)).
- Add `path` option to `ScriptLanguageExtension::_validate` ([GH-83588](https://github.com/godotengine/godot/pull/83588)).
- Fix `variant_iter_get()` actually calling `iter_next()` ([GH-83681](https://github.com/godotengine/godot/pull/83681)).
- Fixed error on loading extensions ([GH-83734](https://github.com/godotengine/godot/pull/83734)).
- Use `ObjectID` when creating custom callable ([GH-83800](https://github.com/godotengine/godot/pull/83800)).
- Linux: Disable `RTLD_DEEPBIND` mode for `dlopen()` in sanitizer builds ([GH-84210](https://github.com/godotengine/godot/pull/84210)).
- Save and compare modification times separately for reload ([GH-84315](https://github.com/godotengine/godot/pull/84315)).
- [iOS, GDExtension] Fix loading and exporting static libraries and xcframeworks ([GH-84493](https://github.com/godotengine/godot/pull/84493)).
- Change `GDExtension`'s `library_path` back to an absolute path ([GH-84620](https://github.com/godotengine/godot/pull/84620)).
- Remove Android specific abis from the export preset feature list ([GH-84720](https://github.com/godotengine/godot/pull/84720)).
- Check that `GDExtensionCompatHashes` are valid when generating `extension_api.json` ([GH-84973](https://github.com/godotengine/godot/pull/84973)).
- iOS: Fix GDExtension init callback array reallocation ([GH-85216](https://github.com/godotengine/godot/pull/85216)).

#### GDScript

- Highlight doc comments in a different color ([GH-72751](https://github.com/godotengine/godot/pull/72751)).
- Fix jumping to function definition using `Ctrl+LMB` or the "Lookup Symbol" button ([GH-73196](https://github.com/godotengine/godot/pull/73196)).
- Improve GDScript identifier tokenization ([GH-73226](https://github.com/godotengine/godot/pull/73226)).
- Add code region folding to CodeEdit ([GH-74843](https://github.com/godotengine/godot/pull/74843)).
- Add raw string literals (r-strings) ([GH-74995](https://github.com/godotengine/godot/pull/74995)).
- Show script errors from depended scripts ([GH-75216](https://github.com/godotengine/godot/pull/75216)).
- Fix for not being able to ignore shadowing warnings on class scope ([GH-75620](https://github.com/godotengine/godot/pull/75620)).
- Add a script method to get its class icon ([GH-75656](https://github.com/godotengine/godot/pull/75656)).
- Improve call analysis ([GH-75988](https://github.com/godotengine/godot/pull/75988)).
- Support threads in the script debugger ([GH-76582](https://github.com/godotengine/godot/pull/76582)).
- Fix conflict between property and group names ([GH-78254](https://github.com/godotengine/godot/pull/78254)).
- Add error message when a GDScript resource fails to load ([GH-78540](https://github.com/godotengine/godot/pull/78540)).
- Check `get_node()` shorthand in static functions ([GH-78552](https://github.com/godotengine/godot/pull/78552)).
- Editor: Remove unused Class Name field from Create Script dialog ([GH-78573](https://github.com/godotengine/godot/pull/78573)).
- Fix incorrect error message for utility functions ([GH-78882](https://github.com/godotengine/godot/pull/78882)).
- Add `@deprecated` and `@experimental` doc comment tags ([GH-78941](https://github.com/godotengine/godot/pull/78941)).
- Fix regression with GDScript enum descriptions now showing up in documentation ([GH-78953](https://github.com/godotengine/godot/pull/78953)).
- Add static analysis error reporting in `GDScriptCache::get_full_script()` ([GH-79163](https://github.com/godotengine/godot/pull/79163)).
- Make onready variables created from dropping nodes include custom types ([GH-79198](https://github.com/godotengine/godot/pull/79198)).
- Solve `_populate_class_members()` cyclic dependency problem ([GH-79205](https://github.com/godotengine/godot/pull/79205)).
- Properly track extents of constants ([GH-79301](https://github.com/godotengine/godot/pull/79301)).
- Load global classes when running debug tests ([GH-79425](https://github.com/godotengine/godot/pull/79425)).
- Fix subscript resolution for constant non-metatypes ([GH-79510](https://github.com/godotengine/godot/pull/79510)).
- Change GDScript tests to use InstancePlaceholder as the example abstract class ([GH-79524](https://github.com/godotengine/godot/pull/79524)).
- Highlight comment markers (`TODO`, `FIXME`, etc.) ([GH-79761](https://github.com/godotengine/godot/pull/79761)).
- Fix bug with identifier shadowed below in current scope ([GH-79880](https://github.com/godotengine/godot/pull/79880)).
- Replace ptrcalls on MethodBind to validated calls ([GH-79893](https://github.com/godotengine/godot/pull/79893)).
- Add validation for `@export_node_path` annotation arguments ([GH-79935](https://github.com/godotengine/godot/pull/79935)).
- Optimize operators by assuming the types ([GH-79990](https://github.com/godotengine/godot/pull/79990)).
- Add constant string support for POT generator ([GH-80020](https://github.com/godotengine/godot/pull/80020)).
- Implement pattern guards for match statement ([GH-80085](https://github.com/godotengine/godot/pull/80085)).
- Fix regression with native signal not found ([GH-80165](https://github.com/godotengine/godot/pull/80165)).
- Add static typing for `for` loop variable ([GH-80247](https://github.com/godotengine/godot/pull/80247)).
- Assign temporary path to preloaded resources ([GH-80281](https://github.com/godotengine/godot/pull/80281)).
- Fix completion option location not found ([GH-80283](https://github.com/godotengine/godot/pull/80283)).
- Allow mixed indentation on blank lines ([GH-80365](https://github.com/godotengine/godot/pull/80365)).
- Fix `get_method` from named lambda ([GH-80506](https://github.com/godotengine/godot/pull/80506)).
- Fix "Identifier not found" error when accessing inner class from inside ([GH-80510](https://github.com/godotengine/godot/pull/80510)).
- Fix superfluous `"` in error message ([GH-80568](https://github.com/godotengine/godot/pull/80568)).
- Check if any global script class is shadowed by a variable ([GH-80587](https://github.com/godotengine/godot/pull/80587)).
- Fixes LSP connection error when launched in a separate thread ([GH-80686](https://github.com/godotengine/godot/pull/80686)).
- Improve DocGen ([GH-80745](https://github.com/godotengine/godot/pull/80745)).
- Fix expected argument count for `Callable` call errors ([GH-80844](https://github.com/godotengine/godot/pull/80844)).
- Fix lambda resolution with cyclic references ([GH-80923](https://github.com/godotengine/godot/pull/80923)).
- Allow using local constants as types ([GH-80964](https://github.com/godotengine/godot/pull/80964)).
- Language Server: Improve hovered symbol resolution, fix renaming bugs, implement reference lookup ([GH-80973](https://github.com/godotengine/godot/pull/80973)).
- Fix `_get_debug_tooltip` crash if tooltip string is too large ([GH-81018](https://github.com/godotengine/godot/pull/81018)).
- Fix highlighting of hex numbers with separators ([GH-81039](https://github.com/godotengine/godot/pull/81039)).
- Fix `get_*_list()` methods return incorrect info ([GH-81079](https://github.com/godotengine/godot/pull/81079)).
- Optimize GDScript VM codegen for MSVC ([GH-81200](https://github.com/godotengine/godot/pull/81200)).
- Fix subclass methods not inheriting RPC info ([GH-81201](https://github.com/godotengine/godot/pull/81201)).
- Fix an error when dragging nodes into built-in scripts because script does not inherit Node ([GH-81299](https://github.com/godotengine/godot/pull/81299)).
- Don't make array literal typed in weak type context ([GH-81332](https://github.com/godotengine/godot/pull/81332)).
- Add an optional `untyped_declaration` warning ([GH-81355](https://github.com/godotengine/godot/pull/81355)).
- Remove `REDUNDANT_FOR_VARIABLE_TYPE` warning ([GH-81440](https://github.com/godotengine/godot/pull/81440)).
- Fix compilation of expressions compiling other classes ([GH-81577](https://github.com/godotengine/godot/pull/81577)).
- Fix dumping of signal API parameters ([GH-81599](https://github.com/godotengine/godot/pull/81599)).
- Fix some lambda bugs ([GH-81605](https://github.com/godotengine/godot/pull/81605)).
- Fix lambda hot reloading ([GH-81628](https://github.com/godotengine/godot/pull/81628)).
- Fix POT generator crash on assignee with index ([GH-81653](https://github.com/godotengine/godot/pull/81653)).
- Fix and improve doc comment parsing ([GH-81699](https://github.com/godotengine/godot/pull/81699)).
- Add check for `super()` methods not being implemented ([GH-81808](https://github.com/godotengine/godot/pull/81808)).
- LSP: Fix autocomplete quote handling ([GH-81833](https://github.com/godotengine/godot/pull/81833)).
- LSP: Add `--lsp-port` as a command line argument ([GH-81844](https://github.com/godotengine/godot/pull/81844)).
- Rewrite a small comment in GDScript tokenizer code ([GH-81881](https://github.com/godotengine/godot/pull/81881)).
- Make array literal typed if `for` loop variable type is specified ([GH-82030](https://github.com/godotengine/godot/pull/82030)).
- GDScript DocGen: Fix and improve appearance of metatypes and values ([GH-82067](https://github.com/godotengine/godot/pull/82067)).
- Prevent constructing and inheriting engine singletons ([GH-82098](https://github.com/godotengine/godot/pull/82098)).
- Fix `--gdscript-docs` tool failing when autoloads are used in the project ([GH-82116](https://github.com/godotengine/godot/pull/82116)).
- Add `INFERRED_DECLARATION` warning ([GH-82139](https://github.com/godotengine/godot/pull/82139)).
- Fix duplication of inherited script properties ([GH-82186](https://github.com/godotengine/godot/pull/82186)).
- Fix crash with `GDScriptNativeClass` ([GH-82294](https://github.com/godotengine/godot/pull/82294)).
- Fix for GDScriptHighlighter dictionaries as function arguments ([GH-82326](https://github.com/godotengine/godot/pull/82326)).
- Add return type covariance and parameter type contravariance ([GH-82477](https://github.com/godotengine/godot/pull/82477)).
- Improve highlighting of types ([GH-82516](https://github.com/godotengine/godot/pull/82516)).
- Fix `UNSAFE_CALL_ARGUMENT` warning for `Variant` constructors ([GH-82547](https://github.com/godotengine/godot/pull/82547)).
- Core: Fix `Object::has_method()` for script static methods ([GH-82767](https://github.com/godotengine/godot/pull/82767)).
- Fix `native_type` is empty for autoload without script ([GH-82784](https://github.com/godotengine/godot/pull/82784)).
- Fix unresolved datatype for incomplete binary operator ([GH-82789](https://github.com/godotengine/godot/pull/82789)).
- Add error when exporting node in non `Node`-derived classes ([GH-82843](https://github.com/godotengine/godot/pull/82843)).
- Fixes internal Script Editor crash with External Editor active ([GH-82956](https://github.com/godotengine/godot/pull/82956)).
- Fix external editor hot reload for GDScript ([GH-82986](https://github.com/godotengine/godot/pull/82986)).
- Fix GDScript cache assigning UID as scene path ([GH-83039](https://github.com/godotengine/godot/pull/83039)).
- GDScript DocGen: Fix regression with return metatypes ([GH-83049](https://github.com/godotengine/godot/pull/83049)).
- Deleting unnecessary include in GDScriptParser ([GH-83050](https://github.com/godotengine/godot/pull/83050)).
- Fix modifying base script exports not propagating to derived scripts ([GH-83123](https://github.com/godotengine/godot/pull/83123)).
- Add autocompletion for static variables accessed via class ([GH-83150](https://github.com/godotengine/godot/pull/83150)).
- Code Editor: Fix regression with using doc comments for code regions ([GH-83216](https://github.com/godotengine/godot/pull/83216)).
- Fix unresolved datatype for incomplete expressions ([GH-83257](https://github.com/godotengine/godot/pull/83257)).
- Fix grammar typo in GDScript error message ([GH-83455](https://github.com/godotengine/godot/pull/83455)).
- Fix non-static call is allowed in static var lambda body ([GH-83486](https://github.com/godotengine/godot/pull/83486)).
- Fix `GDScriptCache::get_full_script` eating parsing errors because of early exit ([GH-83540](https://github.com/godotengine/godot/pull/83540)).
- Don't optimize division and modulo on debug ([GH-83569](https://github.com/godotengine/godot/pull/83569)).
- Fix comment typo in `gdscript_parser.h` ([GH-83792](https://github.com/godotengine/godot/pull/83792)).
- SCons: Fix build with GDScript LSP disabled ([GH-84191](https://github.com/godotengine/godot/pull/84191)).
- Fix lambda cross-thread dynamics (take 2) ([GH-85248](https://github.com/godotengine/godot/pull/85248)).
- Fix GDScript thread-exit routine assuming thread-enter was called ([GH-85432](https://github.com/godotengine/godot/pull/85432)).

#### GUI

- Add option to allow echo events in menu shortcuts ([GH-36493](https://github.com/godotengine/godot/pull/36493)).
- Expose and rename ItemList's `_check_shape_changed` to `force_update_list_size` ([GH-63634](https://github.com/godotengine/godot/pull/63634)).
- Add Duplicate Lines shortcut to CodeTextEditor ([GH-66553](https://github.com/godotengine/godot/pull/66553)).
- Refactor `mouse_entered` and `mouse_exited` signals ([GH-67791](https://github.com/godotengine/godot/pull/67791)).
- Fix cursor behavior for multiselect in Tree while holding CTRL ([GH-71024](https://github.com/godotengine/godot/pull/71024)).
- Fix code completion override of home and end keys ([GH-71519](https://github.com/godotengine/godot/pull/71519)).
- ItemList: Clarify distinction between disabled and selected in sending signals ([GH-74250](https://github.com/godotengine/godot/pull/74250)).
- Add `inner_item_margin_*` Theme constants to the Tree control ([GH-75460](https://github.com/godotengine/godot/pull/75460)).
- Expose finding valid focus neighbors of a `Control` by side ([GH-76027](https://github.com/godotengine/godot/pull/76027)).
- Fix RichTextLabel character line and paragraph index getters ([GH-76759](https://github.com/godotengine/godot/pull/76759)).
- Add a `[pulse]` built-in effect to RichTextLabel ([GH-77117](https://github.com/godotengine/godot/pull/77117)).
- Fix unnecessary break when calculating the height of visible lines ([GH-77280](https://github.com/godotengine/godot/pull/77280)).
- Prevent disappearance of mouse when SpinBox is hidden while dragging ([GH-77804](https://github.com/godotengine/godot/pull/77804)).
- Make it possible to change character transform in RichTextEffect ([GH-77819](https://github.com/godotengine/godot/pull/77819)).
- Add `loop` property to VideoStreamPlayer ([GH-77857](https://github.com/godotengine/godot/pull/77857)).
- Expose VideoStreamPlayer video length ([GH-77858](https://github.com/godotengine/godot/pull/77858)).
- Ensure that `_drop_physics_mouseover` only happens when necessary ([GH-78078](https://github.com/godotengine/godot/pull/78078)).
- Use S, V in hue bar of ColorPicker ([GH-78100](https://github.com/godotengine/godot/pull/78100)).
- Move registration of `fallbacks` property in the base Font class ([GH-78266](https://github.com/godotengine/godot/pull/78266)).
- Add ability to set the tooltip text of a `TreeItem` button ([GH-78393](https://github.com/godotengine/godot/pull/78393)).
- Make GraphEdit's cpp virtuals equal to gdscript ([GH-78426](https://github.com/godotengine/godot/pull/78426)).
- Fix ColorPicker margin theme property ([GH-78468](https://github.com/godotengine/godot/pull/78468)).
- Embedded Popups store their safe_rect in their embedder ([GH-78476](https://github.com/godotengine/godot/pull/78476)).
- Fix text overlapping icon in `Tree` ([GH-78756](https://github.com/godotengine/godot/pull/78756)).
- Enable `InputEvent`-filtering in `SubViewportContainer` ([GH-78762](https://github.com/godotengine/godot/pull/78762)).
- Fix disabled slider highlighting ([GH-78776](https://github.com/godotengine/godot/pull/78776)).
- Fix delay on tab resizing when (un)hovering tabs ([GH-78777](https://github.com/godotengine/godot/pull/78777)).
- Fix invalid minimum size for translated messages in option button ([GH-78835](https://github.com/godotengine/godot/pull/78835)).
- Fix incorrect property names in `FontFile::_get_property_list()` ([GH-78907](https://github.com/godotengine/godot/pull/78907)).
- Add compatibility properties to `TouchScreenButton` ([GH-78940](https://github.com/godotengine/godot/pull/78940)).
- RTL: Add `pop_all`, `push_context` and `pop_context` methods, and use it for `print_rich` to avoid unclosed tags ([GH-79011](https://github.com/godotengine/godot/pull/79011)).
- Move cached values into `color_mode.cpp` and apply fixes to OKHSL ([GH-79037](https://github.com/godotengine/godot/pull/79037)).
- Bind missing default value for `RichTextLabel` methods ([GH-79053](https://github.com/godotengine/godot/pull/79053)).
- Rename `button_pressed` default signal binding to avoid shadowing ([GH-79064](https://github.com/godotengine/godot/pull/79064)).
- Revert "Fix focusloss of non-exclusive `AcceptDialog` with `close_on_escape`" ([GH-79084](https://github.com/godotengine/godot/pull/79084)).
- Allow to focus individual tabs in `TabBar`/`TabContainer` ([GH-79104](https://github.com/godotengine/godot/pull/79104)).
- Enabled missing Tree title button font and font size theme settings ([GH-79165](https://github.com/godotengine/godot/pull/79165)).
- Debug CanvasItem redraw ([GH-79169](https://github.com/godotengine/godot/pull/79169)).
- Deselect curve point with RMB on the empty space ([GH-79175](https://github.com/godotengine/godot/pull/79175)).
- Add `closed` property to Line2D ([GH-79182](https://github.com/godotengine/godot/pull/79182)).
- Update FileDialog button activity when `file_mode` is changed ([GH-79211](https://github.com/godotengine/godot/pull/79211)).
- Make `SubViewportContainer` event propagation aware of focused Control ([GH-79248](https://github.com/godotengine/godot/pull/79248)).
- HarfBuzz: Update to version 8.0.0 ([GH-79260](https://github.com/godotengine/godot/pull/79260)).
- ICU4C: Update to version 73.2 ([GH-79272](https://github.com/godotengine/godot/pull/79272)).
- FreeType: Update to version 2.13.1 ([GH-79273](https://github.com/godotengine/godot/pull/79273)).
- Check `FLAG_POPUP` to close an AcceptDialog when parent is focused ([GH-79293](https://github.com/godotengine/godot/pull/79293)).
- Remove GraphNode's comment property and related functionality ([GH-79307](https://github.com/godotengine/godot/pull/79307)).
- Clean up/refactor GraphEdit ([GH-79308](https://github.com/godotengine/godot/pull/79308)).
- Clean up/refactor GraphNode and make it more flexible ([GH-79311](https://github.com/godotengine/godot/pull/79311)).
- Fix `Tree` performance regression by using cache ([GH-79325](https://github.com/godotengine/godot/pull/79325)).
- macOS: Add `about_to_open` and `popup_hide` callback for the global menus ([GH-79361](https://github.com/godotengine/godot/pull/79361)).
- Add a default theme for unfocused Windows ([GH-79393](https://github.com/godotengine/godot/pull/79393)).
- Fix Button clipping when internal margins exist ([GH-79455](https://github.com/godotengine/godot/pull/79455)).
- Fix native popups auto-closing when interacting with non-client area ([GH-79456](https://github.com/godotengine/godot/pull/79456)).
- Make `OptionButton` resize when disabling "Fit to Longest Item" ([GH-79494](https://github.com/godotengine/godot/pull/79494)).
- Add drag'n'drop text option for `LineEdit` and `RichTextLabel` ([GH-79563](https://github.com/godotengine/godot/pull/79563)).
- macOS: Fix uncapped frame rate for windows in the non-active workspaces ([GH-79572](https://github.com/godotengine/godot/pull/79572)).
- Fix `root_node_layout_direction` project setting being incorrectly exposed as a range ([GH-79611](https://github.com/godotengine/godot/pull/79611)).
- Fix corner radius not scaling with theme scale in the default theme ([GH-79640](https://github.com/godotengine/godot/pull/79640)).
- Snap CharFX offset to nearest pixel when setting is enabled ([GH-79705](https://github.com/godotengine/godot/pull/79705)).
- Remove spaces from input of HTML color in color picker ([GH-79782](https://github.com/godotengine/godot/pull/79782)).
- Correctly display tooltips for buttons in Tree when they overlap cell content ([GH-79792](https://github.com/godotengine/godot/pull/79792)).
- Prevent SubViewportContainer overriding Subviewport's cursor with its own cursor ([GH-79805](https://github.com/godotengine/godot/pull/79805)).
- RichTextLabel: Ensure the `select_all` function selects all items ([GH-79818](https://github.com/godotengine/godot/pull/79818)).
- [Text Server] Fix ellipsis outline drawing ([GH-79844](https://github.com/godotengine/godot/pull/79844)).
- Label: Remove extra line spacing from Label minimum size calculations ([GH-79913](https://github.com/godotengine/godot/pull/79913)).
- Fix Tree check propagation not unchecking parent items ([GH-79946](https://github.com/godotengine/godot/pull/79946)).
- Free submenu children when clearing PopupMenu ([GH-79965](https://github.com/godotengine/godot/pull/79965)).
- Expose `Window`'s `_get_contents_minimum_size()` to scripting ([GH-80178](https://github.com/godotengine/godot/pull/80178)).
- Handle potential platform-specific `Window` mouse-enter/exit bugs gracefully ([GH-80187](https://github.com/godotengine/godot/pull/80187)).
- Add shortcut handling to `OptionButton` ([GH-80203](https://github.com/godotengine/godot/pull/80203)).
- Improve `Window._get_contents_minimum_size()`'s code ([GH-80219](https://github.com/godotengine/godot/pull/80219)).
- Expose the `TabBar` of a `TabContainer` ([GH-80227](https://github.com/godotengine/godot/pull/80227)).
- Fix scrolling `PopupMenu` on keyboard/controller input ([GH-80271](https://github.com/godotengine/godot/pull/80271)).
- Further separate icon from text of buttons in both editor and default themes ([GH-80285](https://github.com/godotengine/godot/pull/80285)).
- Dismiss currently visible or upcoming tooltips when pressing Escape ([GH-80364](https://github.com/godotengine/godot/pull/80364)).
- Fix `OptionButton` minimum size when "Fit Longest Item" is enabled ([GH-80366](https://github.com/godotengine/godot/pull/80366)).
- Fix `Button` text when the overrun behavior is other than "No Trimming" ([GH-80402](https://github.com/godotengine/godot/pull/80402)).
- RTL: Add support for image dynamic updating, padding, tooltips and size in percent ([GH-80410](https://github.com/godotengine/godot/pull/80410)).
- Fix CodeEdit completion being very slow in certain cases ([GH-80472](https://github.com/godotengine/godot/pull/80472)).
- Support other input methods on Popup/Dialogs' `_input_from_window` ([GH-80594](https://github.com/godotengine/godot/pull/80594)).
- [Bitmap fonts] Add support for scaling ([GH-80605](https://github.com/godotengine/godot/pull/80605)).
- RTL: Improve scroll bar responsiveness during updates ([GH-80606](https://github.com/godotengine/godot/pull/80606)).
- Add buttons to reorder inspector array items without dragging ([GH-80617](https://github.com/godotengine/godot/pull/80617)).
- Fix 2D/3D viewport context switching issues when script editor is floating ([GH-80647](https://github.com/godotengine/godot/pull/80647)).
- TextServer: Fix system font fallback and caret/selection behavior for composite characters ([GH-80650](https://github.com/godotengine/godot/pull/80650)).
- Allow comma as a decimal separator for SpinBox ([GH-80699](https://github.com/godotengine/godot/pull/80699)).
- TextServer: Fix issues with character breaks, add more tests ([GH-80777](https://github.com/godotengine/godot/pull/80777)).
- Fix crash when hiding subwindow during popup of new subwindow ([GH-80780](https://github.com/godotengine/godot/pull/80780)).
- Exit early in `TextEdit::_get_column_pos_of_word` to improve highlight performance ([GH-80809](https://github.com/godotengine/godot/pull/80809)).
- Fix "Go to parent folder" in `EditorFileDialog` ([GH-80821](https://github.com/godotengine/godot/pull/80821)).
- RTL: Fix `remove_paragraph` crashes ([GH-80847](https://github.com/godotengine/godot/pull/80847)).
- RTL: Adds "lang" tag to allow overriding language specific text rendering without starting a new paragraph ([GH-80848](https://github.com/godotengine/godot/pull/80848)).
- RTL: Improve performance by using list iterators for item/paragraph removal ([GH-80857](https://github.com/godotengine/godot/pull/80857)).
- Fix ColorPicker deferred mode not working for sliders ([GH-80916](https://github.com/godotengine/godot/pull/80916)).
- TextServer: Store extra spacing of individual font variations ([GH-80954](https://github.com/godotengine/godot/pull/80954)).
- Deselect multi caret when alt clicking on it ([GH-80956](https://github.com/godotengine/godot/pull/80956)).
- FileDialog: Avoid selecting the first item automatically in Open Folder Mode ([GH-81034](https://github.com/godotengine/godot/pull/81034)).
- Fix setting TabContainer's `font_hovered_color` theme property ([GH-81040](https://github.com/godotengine/godot/pull/81040)).
- RTL: Fix character line index for non-visual characters and characters on the line edge ([GH-81064](https://github.com/godotengine/godot/pull/81064)).
- Move default theme files to `scene/theme` ([GH-81065](https://github.com/godotengine/godot/pull/81065)).
- Only allow finite numbers in `Range.value` ([GH-81076](https://github.com/godotengine/godot/pull/81076)).
- Fix SpinBox not clearing text on improper input ([GH-81094](https://github.com/godotengine/godot/pull/81094)).
- TextServer: Fix SVG emoji placement ([GH-81103](https://github.com/godotengine/godot/pull/81103)).
- Fix a crash when plugin tries to call `make_mesh_previews` on enable ([GH-81121](https://github.com/godotengine/godot/pull/81121)).
- Unfocus LineEdit when pressing Escape ([GH-81128](https://github.com/godotengine/godot/pull/81128)).
- Implement a system to contextualize global themes ([GH-81130](https://github.com/godotengine/godot/pull/81130)).
- ItemList: Draw separators before selected style boxes ([GH-81155](https://github.com/godotengine/godot/pull/81155)).
- Fix TreeItem range slider not working properly ([GH-81174](https://github.com/godotengine/godot/pull/81174)).
- Fix ItemList not updating when icon scale changes ([GH-81268](https://github.com/godotengine/godot/pull/81268)).
- Fix ThemeDB initialization in tests ([GH-81305](https://github.com/godotengine/godot/pull/81305)).
- Register theme properties with ThemeDB ([GH-81312](https://github.com/godotengine/godot/pull/81312)).
- Update and properly list versions of the built-in fonts ([GH-81326](https://github.com/godotengine/godot/pull/81326)).
- Fix `TextEdit.get_rect_at_line_column returning` negative pos even though cursor is in viewable area of the control ([GH-81354](https://github.com/godotengine/godot/pull/81354)).
- TextServer: Use locale or first span language to select preferred direction for neutral text ([GH-81361](https://github.com/godotengine/godot/pull/81361)).
- Remove unnecessary validity checks from `Button` and `TextureRect` ([GH-81383](https://github.com/godotengine/godot/pull/81383)).
- Fix TextEdit placeholder with Inherited text direction ([GH-81396](https://github.com/godotengine/godot/pull/81396)).
- TextServer: Pass Dictionary properties by value and check property values instead of references ([GH-81406](https://github.com/godotengine/godot/pull/81406)).
- Fix subpixel layouts in text rendering ([GH-81438](https://github.com/godotengine/godot/pull/81438)).
- LineEdit: Update line edit offset on text delete ([GH-81443](https://github.com/godotengine/godot/pull/81443)).
- Correctly setup tooltip's style as theme variation ([GH-81463](https://github.com/godotengine/godot/pull/81463)).
- Fix submenu alignment with parent menu item ([GH-81477](https://github.com/godotengine/godot/pull/81477)).
- Fix accessing editor theme items throughout the UI ([GH-81516](https://github.com/godotengine/godot/pull/81516)).
- Hide the `dialog_text` property from `FileDialog` ([GH-81546](https://github.com/godotengine/godot/pull/81546)).
- Bind remaining theme properties to their respective classes ([GH-81551](https://github.com/godotengine/godot/pull/81551)).
- Improve the looks of 2D/3D viewport contextual toolbars ([GH-81557](https://github.com/godotengine/godot/pull/81557)).
- Use bound theme properties for documentation ([GH-81573](https://github.com/godotengine/godot/pull/81573)).
- Make `GraphEdit` toolbar more customizable ([GH-81582](https://github.com/godotengine/godot/pull/81582)).
- Fix GraphEdit port valid connections incorrectly checking sides ([GH-81600](https://github.com/godotengine/godot/pull/81600)).
- Expose `PopupMenu` `activate_item_by_event` method ([GH-81621](https://github.com/godotengine/godot/pull/81621)).
- Fix SpinBox will reset unsubmitted text when redrawing ([GH-81638](https://github.com/godotengine/godot/pull/81638)).
- Remove the equality check for `TabBar.set_tab_metadata` ([GH-81648](https://github.com/godotengine/godot/pull/81648)).
- Enable transparent background for GUI tooltips ([GH-81669](https://github.com/godotengine/godot/pull/81669)).
- Connect `CodeHighlighter` with `TextEdit` without friend-access ([GH-81921](https://github.com/godotengine/godot/pull/81921)).
- Replace flat buttons with flat-styled buttons with a visible pressed state ([GH-81939](https://github.com/godotengine/godot/pull/81939)).
- Check for type variations in inherited themes ([GH-82218](https://github.com/godotengine/godot/pull/82218)).
- Fix tooltips behaving incorrectly on `Tree` nodes ([GH-82226](https://github.com/godotengine/godot/pull/82226)).
- Add Font and Mesh icons that aren't grayed out ([GH-82302](https://github.com/godotengine/godot/pull/82302)).
- Rename close requests to delete requests in `GraphEdit` ([GH-82370](https://github.com/godotengine/godot/pull/82370)).
- Make hovered tabs be drawn with the unselected's width at minimum ([GH-82384](https://github.com/godotengine/godot/pull/82384)).
- Document, cleanup and fix some theme properties ([GH-82409](https://github.com/godotengine/godot/pull/82409)).
- TextServer: Store font extra spacing variations without making a full copy of font ([GH-82475](https://github.com/godotengine/godot/pull/82475)).
- FileDialog: Make `set_visible` compatible with native dialogs ([GH-82552](https://github.com/godotengine/godot/pull/82552)).
- Tweak the region folding icons ([GH-82653](https://github.com/godotengine/godot/pull/82653)).
- Fix storing invalid item height values in `ItemList` ([GH-82660](https://github.com/godotengine/godot/pull/82660)).
- SystemFont: Check name when selecting the best matching face from a collection ([GH-82712](https://github.com/godotengine/godot/pull/82712)).
- [File Dialog] Do not open native file dialogs in the edited scene ([GH-82743](https://github.com/godotengine/godot/pull/82743)).
- Organize TextEdit's inspector ([GH-82776](https://github.com/godotengine/godot/pull/82776)).
- Place LineEdit secret in its own section ([GH-82811](https://github.com/godotengine/godot/pull/82811)).
- Accept cancel event when unfocusing LineEdit ([GH-82914](https://github.com/godotengine/godot/pull/82914)).
- Fix right-click menu position for the debugger breakpoint tree ([GH-82924](https://github.com/godotengine/godot/pull/82924)).
- RTL: Remove unnecessary glyph position rounding ([GH-82970](https://github.com/godotengine/godot/pull/82970)).
- Do not apply extra spacing twice ([GH-83062](https://github.com/godotengine/godot/pull/83062)).
- Allow clicking buttons of non-selectable TreeItems ([GH-83065](https://github.com/godotengine/godot/pull/83065)).
- Remove vertical scrollbar padding from line width calc ([GH-83286](https://github.com/godotengine/godot/pull/83286)).
- Fix phantom tab right button ([GH-83296](https://github.com/godotengine/godot/pull/83296)).
- Fix incorrect offset of `PopupMenu` separator icons ([GH-83517](https://github.com/godotengine/godot/pull/83517)).
- Add bulk change guards to successive theme overrides in Editor and GUI ([GH-83626](https://github.com/godotengine/godot/pull/83626)).
- Fix `TabBar` and `TabContainer` dragging issues ([GH-83637](https://github.com/godotengine/godot/pull/83637)).
- Fix missing initial position modes for the main window ([GH-83824](https://github.com/godotengine/godot/pull/83824)).
- TextServerAdvanced: Keep dynamically loaded ICU data in memory ([GH-83827](https://github.com/godotengine/godot/pull/83827)).
- Increase precision of RAW mode in ColorPicker ([GH-83851](https://github.com/godotengine/godot/pull/83851)).
- Fix GraphNode slot index inconsistency ([GH-83892](https://github.com/godotengine/godot/pull/83892)).
- Save current tab in `TabBar` and `TabContainer` ([GH-83893](https://github.com/godotengine/godot/pull/83893)).
- Fix BaseButton `shortcut_feedback`'s timer will raise error when the button is removed from the scene tree ([GH-83925](https://github.com/godotengine/godot/pull/83925)).
- Translate TextEdit placeholder ([GH-83946](https://github.com/godotengine/godot/pull/83946)).
- Ensure input event is valid in `PopupMenu::activate_item_by_event` ([GH-83952](https://github.com/godotengine/godot/pull/83952)).
- [Menu Bar] Update min. size when items are added/removed/changed ([GH-83961](https://github.com/godotengine/godot/pull/83961)).
- Fix disabled tabs being selected when removing the current one ([GH-83963](https://github.com/godotengine/godot/pull/83963)).
- Fix `TabContainer` drag to rearrange issue ([GH-83966](https://github.com/godotengine/godot/pull/83966)).
- Fix TreeItem truncating node names too much when using a custom icon ([GH-84001](https://github.com/godotengine/godot/pull/84001)).
- Add foreign validation warning for rename actions ([GH-84022](https://github.com/godotengine/godot/pull/84022)).
- Include empty type variations in `Theme::get_type_list` ([GH-84127](https://github.com/godotengine/godot/pull/84127)).
- [Text Mesh] Fix incorrectly cached glyph offsets ([GH-84180](https://github.com/godotengine/godot/pull/84180)).
- Fix `activate_item_by_event` infinite recursion crash ([GH-84183](https://github.com/godotengine/godot/pull/84183)).
- TextServer: Fix glyph comparator ambiguous output ([GH-84232](https://github.com/godotengine/godot/pull/84232)).
- RTL: Fix underline/strikethrough line color changes ([GH-84233](https://github.com/godotengine/godot/pull/84233)).
- TextServer: Fix line breaks for dropcap and resizing embedded objects ([GH-84287](https://github.com/godotengine/godot/pull/84287)).
- Fix `ColorPicker` shape icon is invisible until shape is changed ([GH-84535](https://github.com/godotengine/godot/pull/84535)).
- Make mouse enter/exit notifications match mouse events ([GH-84547](https://github.com/godotengine/godot/pull/84547)).
- RTL: Fix list bullet alignment ([GH-84605](https://github.com/godotengine/godot/pull/84605)).
- Warn about autowrapped labels in containers ([GH-84662](https://github.com/godotengine/godot/pull/84662)).
- Allow auto-generated node names in `PopupMenu::add_submenu_item` ([GH-84668](https://github.com/godotengine/godot/pull/84668)).
- Add protection in `RichTextLabel.update_image` to prevent crash ([GH-84833](https://github.com/godotengine/godot/pull/84833)).
- Make Tree's `set_selected` check if the TreeItem belongs to the tree ([GH-84870](https://github.com/godotengine/godot/pull/84870)).
- Fix remapped font reloading on locale change ([GH-84873](https://github.com/godotengine/godot/pull/84873)).
- RTL: Fix excessive underline and table border draw calls ([GH-84874](https://github.com/godotengine/godot/pull/84874)).
- Add GraphEdit connection layer child as internal ([GH-85009](https://github.com/godotengine/godot/pull/85009)).
- Fix crash when hiding a Control during mouse-entering ([GH-85284](https://github.com/godotengine/godot/pull/85284)).
- Fix crash on late mouse enter/exit event arrival ([GH-85418](https://github.com/godotengine/godot/pull/85418)).

#### Import

- Fix ImageTextureLayered serialization issues ([GH-71394](https://github.com/godotengine/godot/pull/71394)).
- Add support for KTX image format so that we can use Basis Universal for GLTF ([GH-76572](https://github.com/godotengine/godot/pull/76572)).
- Add more physics options to the Scene importer ([GH-77533](https://github.com/godotengine/godot/pull/77533)).
- Fix reimporting files with non lowercase name extension ([GH-78567](https://github.com/godotengine/godot/pull/78567)).
- Add support for GLTF extension KHR_materials_emissive_strength ([GH-78621](https://github.com/godotengine/godot/pull/78621)).
- GLTF: Internal renames in material parsing code ([GH-78622](https://github.com/godotengine/godot/pull/78622)).
- Add layer, shadow and visibility range options to the Scene importer ([GH-78803](https://github.com/godotengine/godot/pull/78803)).
- Allow change import type without restarting editor ([GH-78890](https://github.com/godotengine/godot/pull/78890)).
- Fix property hint class name type string restriction and replace mode ([GH-79139](https://github.com/godotengine/godot/pull/79139)).
- Lossy WebP: Enable sharp RGB to YUV conversion ([GH-79257](https://github.com/godotengine/godot/pull/79257)).
- Add copyright to GLTFState ([GH-79267](https://github.com/godotengine/godot/pull/79267)).
- GLTF: Allow specifying export image format including from extensions ([GH-79314](https://github.com/godotengine/godot/pull/79314)).
- Add `KHR_materials_emissive_strength` extension support for exporting GLTFs ([GH-79421](https://github.com/godotengine/godot/pull/79421)).
- GLTF: Preserve the original bytes when extracting a texture while importing ([GH-79533](https://github.com/godotengine/godot/pull/79533)).
- Add `export_preserialize` to the GLTF export process ([GH-79623](https://github.com/godotengine/godot/pull/79623)).
- Set `base_path` and `filename` during GLTF export when writing to a file ([GH-79636](https://github.com/godotengine/godot/pull/79636)).
- Improve overriding the root type or root name in the scene importer ([GH-79774](https://github.com/godotengine/godot/pull/79774)).
- Cosmetic changes in GLTF node generation code ([GH-79775](https://github.com/godotengine/godot/pull/79775)).
- Improve GLTF export logic for scene root nodes ([GH-79801](https://github.com/godotengine/godot/pull/79801)).
- Fix reimporting scene with default values selected ([GH-79907](https://github.com/godotengine/godot/pull/79907)).
- Update ThorVG to v0.10.0 ([GH-80095](https://github.com/godotengine/godot/pull/80095)).
- Fix error message when reimporting resources with an empty scene open ([GH-80149](https://github.com/godotengine/godot/pull/80149)).
- More cosmetic improvements in the GLTF code ([GH-80205](https://github.com/godotengine/godot/pull/80205)).
- Fix doubly-reserved unique names in GLTF scene name assignment ([GH-80270](https://github.com/godotengine/godot/pull/80270)).
- GLTF: Improve logic for keeping track of the real root node ([GH-80272](https://github.com/godotengine/godot/pull/80272)).
- Use image index instead of texture index for `source_images` ([GH-80314](https://github.com/godotengine/godot/pull/80314)).
- Register and cleanup resource importer singletons in a predictable way ([GH-80377](https://github.com/godotengine/godot/pull/80377)).
- GLTF: Add center of mass property ([GH-80463](https://github.com/godotengine/godot/pull/80463)).
- Limit mesh complexity in LOD generation to prevent crashing ([GH-80467](https://github.com/godotengine/godot/pull/80467)).
- Fixed editor filesystem/import properties not being caught by the doctool ([GH-80576](https://github.com/godotengine/godot/pull/80576)).
- GLTF: Add a comment for skinned mesh tree placement ([GH-80807](https://github.com/godotengine/godot/pull/80807)).
- Fix skeletons when generating multiple Godot scenes from one GLTF ([GH-80831](https://github.com/godotengine/godot/pull/80831)).
- Make DDS loading code only check for R channel bitmask when loading grayscale images ([GH-80862](https://github.com/godotengine/godot/pull/80862)).
- Update Importing 3D scenes links to match splitting PR ([GH-80872](https://github.com/godotengine/godot/pull/80872)).
- Fix "Import Defaults" selector not being initialized incorrectly ([GH-80914](https://github.com/godotengine/godot/pull/80914)).
- Fix grayscale DDS loading ([GH-81134](https://github.com/godotengine/godot/pull/81134)).
- Update Blender export flags for 3.6 ([GH-81194](https://github.com/godotengine/godot/pull/81194)).
- GLTF: Change "Camera3D" generated node name to "Camera" ([GH-81264](https://github.com/godotengine/godot/pull/81264)).
- GLTF: Add root node export options and `GODOT_single_root` extension ([GH-81851](https://github.com/godotengine/godot/pull/81851)).
- Fix ImporterMesh bone weight handling during lightmap unwrap ([GH-81854](https://github.com/godotengine/godot/pull/81854)).
- Disable bounding box shadows for advanced scene importer ([GH-82190](https://github.com/godotengine/godot/pull/82190)).
- Fix GLTF importer forcing vertex colors on all materials ([GH-82272](https://github.com/godotengine/godot/pull/82272)).
- Avoid crash when generating LODs on meshes with non-finite vertices ([GH-82285](https://github.com/godotengine/godot/pull/82285)).
- Fix Image import crash ([GH-82408](https://github.com/godotengine/godot/pull/82408)).
- Avoid import dock cleanup for non-loadable assets ([GH-82490](https://github.com/godotengine/godot/pull/82490)).
- Fix the Advanced Import Settings window's 3D camera ([GH-82591](https://github.com/godotengine/godot/pull/82591)).
- ThorVG: update to v0.11.1 ([GH-83281](https://github.com/godotengine/godot/pull/83281)).
- Make translation importer skip not-supported lang tag, make it more robust ([GH-83600](https://github.com/godotengine/godot/pull/83600)).
- Prevent crash from importing a certain kind of invalid GLTF ([GH-83663](https://github.com/godotengine/godot/pull/83663)).
- Fix infinite loop when importing 3D object named "-colonly" ([GH-83764](https://github.com/godotengine/godot/pull/83764)).
- Fix crash when reimporting with Skeleton3D selected ([GH-83964](https://github.com/godotengine/godot/pull/83964)).
- Add method check for `_notify_skeleton_bones_renamed` ([GH-83986](https://github.com/godotengine/godot/pull/83986)).
- Enhance checks and user experience around tangent arrays in meshes ([GH-84252](https://github.com/godotengine/godot/pull/84252)).
- Implement glTF compatibility system for files imported in older Godot versions ([GH-84271](https://github.com/godotengine/godot/pull/84271)).
- Scan the filesystem in the first frame when using headless mode ([GH-84570](https://github.com/godotengine/godot/pull/84570)).
- Use the Blender file name instead of the generated GLTF file name ([GH-84678](https://github.com/godotengine/godot/pull/84678)).
- Fix Resource Importer use after free ([GH-84872](https://github.com/godotengine/godot/pull/84872)).

#### Input

- Check if input marked handled before processing additional CollisionObjects ([GH-48800](https://github.com/godotengine/godot/pull/48800)).
- Add Unit tests for viewport.cpp Physics 2D Picking ([GH-73477](https://github.com/godotengine/godot/pull/73477)).
- Fix code editor scrolling experience on track pads ([GH-73502](https://github.com/godotengine/godot/pull/73502)).
- Prevent double input events on gamepad when running through steam input ([GH-76045](https://github.com/godotengine/godot/pull/76045)).
- Implement `DisplayServer.keyboard_get_label_from_physical` method ([GH-77993](https://github.com/godotengine/godot/pull/77993)).
- Fix Physics Picking captured Object initialization ([GH-78383](https://github.com/godotengine/godot/pull/78383)).
- Add the ability to get per-platform information for joypads ([GH-78539](https://github.com/godotengine/godot/pull/78539)).
- Mention Xbox menu button by name in Start button description ([GH-78701](https://github.com/godotengine/godot/pull/78701)).
- Android: Set `echo` property for the physical keyboard events ([GH-79089](https://github.com/godotengine/godot/pull/79089)).
- Fix physics passive hovering with `MOUSE_FILTER_IGNORE` ([GH-79443](https://github.com/godotengine/godot/pull/79443)).
- Make GridMap shortcuts editable and not conflict with other plugins ([GH-79529](https://github.com/godotengine/godot/pull/79529)).
- Separate input-handled-state for different events during physics-picking ([GH-79546](https://github.com/godotengine/godot/pull/79546)).
- Fix crash on Windows when closing `Window` ([GH-80142](https://github.com/godotengine/godot/pull/80142)).
- Ensure TileMap editor shortcuts are handled ([GH-80317](https://github.com/godotengine/godot/pull/80317)).
- Fix nodes receiving mouse events in black bars of `Window` ([GH-80334](https://github.com/godotengine/godot/pull/80334)).
- Properly load multiple action sets in XR ([GH-80419](https://github.com/godotengine/godot/pull/80419)).
- Ensure `joy_connection_changed` is emitted on the main thread ([GH-80432](https://github.com/godotengine/godot/pull/80432)).
- Android Stylus pressure and tilt support ([GH-80644](https://github.com/godotengine/godot/pull/80644)).
- Fix GridMap shortcuts that should not be physical ([GH-80774](https://github.com/godotengine/godot/pull/80774)).
- Fix action state when multiple events are assigned ([GH-80859](https://github.com/godotengine/godot/pull/80859)).
- Fix Android input routing logic when using a hardware keyboard ([GH-80932](https://github.com/godotengine/godot/pull/80932)).
- Add missing YEN, SECTION and OPENURL names to keycode mappings ([GH-81054](https://github.com/godotengine/godot/pull/81054)).
- Prevent axis-based actions from getting stuck ([GH-81170](https://github.com/godotengine/godot/pull/81170)).
- Android: Fix joypad trigger value range ([GH-81322](https://github.com/godotengine/godot/pull/81322)).
- Fix `Input.is_action_just_pressed` flicker on joypad axes ([GH-82056](https://github.com/godotengine/godot/pull/82056)).
- Make InputEventShortcut always pressed ([GH-82203](https://github.com/godotengine/godot/pull/82203)).
- Sync controller mappings DB with SDL2 community repo ([GH-82245](https://github.com/godotengine/godot/pull/82245)).
- Add XInput device ID for wireless Series 2 Elite controller ([GH-82508](https://github.com/godotengine/godot/pull/82508)).
- Fix the timeframe when the Android gestures properties are retrieved ([GH-83173](https://github.com/godotengine/godot/pull/83173)).
- Fix Android logic for deferred window input events being inverted ([GH-83301](https://github.com/godotengine/godot/pull/83301)).
- Fix shortcut input for `EditorSceneTabs` ([GH-83501](https://github.com/godotengine/godot/pull/83501)).
- Sync controller mappings DB with SDL2 community repo ([GH-83845](https://github.com/godotengine/godot/pull/83845)).
- Add save shortcut for text shader editor to prevent it propagating to scene ([GH-84064](https://github.com/godotengine/godot/pull/84064)).
- Fix stuck cursor in Advanced Scene Importer ([GH-84661](https://github.com/godotengine/godot/pull/84661)).
- Rework input actions to be reliable ([GH-84685](https://github.com/godotengine/godot/pull/84685)).

#### Multiplayer

- Disallow nested custom multiplayers in `SceneTree` ([GH-77829](https://github.com/godotengine/godot/pull/77829)).
- Prevent crash when accessing `Node` Multiplayer from thread ([GH-79332](https://github.com/godotengine/godot/pull/79332)).
- Use `get/set_indexed` in MultiplayerSynchronizer ([GH-79479](https://github.com/godotengine/godot/pull/79479)).
- [Net/ENet] Better handle truncated socket messages ([GH-79699](https://github.com/godotengine/godot/pull/79699)).
- ENet: Properly set transfer flags when using custom channels ([GH-80293](https://github.com/godotengine/godot/pull/80293)).
- Fix watch properties not being correctly removed ([GH-81033](https://github.com/godotengine/godot/pull/81033)).
- Improve SceneReplicationConfig editor UX + optimizations ([GH-81136](https://github.com/godotengine/godot/pull/81136)).
- Various performance optimizations ([GH-82777](https://github.com/godotengine/godot/pull/82777)).
- Copy network authority when instancing placeholders ([GH-82846](https://github.com/godotengine/godot/pull/82846)).
- Fix synchronizer init and reset ([GH-83264](https://github.com/godotengine/godot/pull/83264)).
- Fix "on change" indexed properties ([GH-83279](https://github.com/godotengine/godot/pull/83279)).
- Display multiplayer authority ID in remote debugger ([GH-83437](https://github.com/godotengine/godot/pull/83437)).

#### Navigation

- Add NavigationRegion function to change navigation map ([GH-77191](https://github.com/godotengine/godot/pull/77191)).
- Add ProjectSettings navigation map default up ([GH-78365](https://github.com/godotengine/godot/pull/78365)).
- Add more basic tests for `NavigationServer3D` ([GH-78480](https://github.com/godotengine/godot/pull/78480)).
- Add advanced `NavigationServer3D` tests ([GH-78667](https://github.com/godotengine/godot/pull/78667)).
- Fix closest possible navigation path position ([GH-79004](https://github.com/godotengine/godot/pull/79004)).
- Add NavigationServer API to enable regions and links ([GH-79129](https://github.com/godotengine/godot/pull/79129)).
- Mark NavigationServer3D.region_bake_navigation_mesh() as deprecated ([GH-79137](https://github.com/godotengine/godot/pull/79137)).
- Add `clear` function to NavigationMesh / NavigationPolygon ([GH-79157](https://github.com/godotengine/godot/pull/79157)).
- Fix pathfinding funnel adding unwanted point ([GH-79228](https://github.com/godotengine/godot/pull/79228)).
- Fix NavigationObstacle2D debug position ([GH-79392](https://github.com/godotengine/godot/pull/79392)).
- Make NavigationRegion3D baking NavMesh on the main thread not finish deferred ([GH-79465](https://github.com/godotengine/godot/pull/79465)).
- Change 2D navigation ProjectSettings from integers to floats ([GH-79483](https://github.com/godotengine/godot/pull/79483)).
- Set default `cell_size` on new TileMap Layer navigation layer maps ([GH-79485](https://github.com/godotengine/godot/pull/79485)).
- Add more hints to navigation map cell size errors ([GH-79489](https://github.com/godotengine/godot/pull/79489)).
- Add a `fill_region` method to the `AStarGrid2D` ([GH-79495](https://github.com/godotengine/godot/pull/79495)).
- Move navigation mesh baking to NavigationServer ([GH-79643](https://github.com/godotengine/godot/pull/79643)).
- Disable NavigationMesh `edge_max_length` property by default ([GH-79786](https://github.com/godotengine/godot/pull/79786)).
- Add multi-threaded NavMesh baking to NavigationServer ([GH-79972](https://github.com/godotengine/godot/pull/79972)).
- Fix NavMesh `map_update_id` returning 0 results in errors ([GH-80189](https://github.com/godotengine/godot/pull/80189)).
- Fix missing include for `NavigationMesh` ([GH-80408](https://github.com/godotengine/godot/pull/80408)).
- Add 2D navigation mesh baking ([GH-80796](https://github.com/godotengine/godot/pull/80796)).
- Suppress expected errors in navigation-related unit tests ([GH-80833](https://github.com/godotengine/godot/pull/80833)).
- Fix compiling with 3D disabled due to unused navigation variable ([GH-81295](https://github.com/godotengine/godot/pull/81295)).
- Core: Some code style improvements to `AStarGrid2D` ([GH-81900](https://github.com/godotengine/godot/pull/81900)).
- Fix typo in dev assert in NavMeshGenerator2D ([GH-82368](https://github.com/godotengine/godot/pull/82368)).
- Update TileMap to use new navigation polygon baking ([GH-82465](https://github.com/godotengine/godot/pull/82465)).
- Fix NavigationObstacle3D debug being affected by rotation and scale ([GH-82593](https://github.com/godotengine/godot/pull/82593)).
- Fix enabling NavigationRegion3D saved disabled ([GH-83365](https://github.com/godotengine/godot/pull/83365)).
- Fix "Navigation map synchronization error" when using NavigationRegion2D ([GH-83568](https://github.com/godotengine/godot/pull/83568)).
- Fix NavRegion sync error messages ([GH-83574](https://github.com/godotengine/godot/pull/83574)).
- Fix NavigationObstacle3D height ([GH-83701](https://github.com/godotengine/godot/pull/83701)).
- Fix NavigationAgent3D stored y-axis velocity and make it optional ([GH-83705](https://github.com/godotengine/godot/pull/83705)).
- Fix NavigationLink enabled toggle ([GH-83709](https://github.com/godotengine/godot/pull/83709)).
- Fix hole in heightmap navigation mesh baking ([GH-83783](https://github.com/godotengine/godot/pull/83783)).
- Fix potential crashes with TileMap navmesh baking ([GH-83891](https://github.com/godotengine/godot/pull/83891)).
- Fix NavigationObstacle3DEditor parenting error ([GH-84055](https://github.com/godotengine/godot/pull/84055)).
- Fix NavigationObstacle elevation ([GH-84830](https://github.com/godotengine/godot/pull/84830)).
- Fix NavigationObstacle height ([GH-84857](https://github.com/godotengine/godot/pull/84857)).

#### Network

- Fix `rpc` calls with binds ([GH-78551](https://github.com/godotengine/godot/pull/78551)).
- Web: Fix WebSocket returning empty close-reason ([GH-79407](https://github.com/godotengine/godot/pull/79407)).
- Web: Always return -1 as body length in HTTPClientWeb ([GH-79846](https://github.com/godotengine/godot/pull/79846)).

#### Particles

- Add `finished` signal to CPUParticles ([GH-76853](https://github.com/godotengine/godot/pull/76853)).
- Add `finished` signal to GPUParticles ([GH-76859](https://github.com/godotengine/godot/pull/76859)).
- Initialize particles instance buffer in case it is used before being updated ([GH-78852](https://github.com/godotengine/godot/pull/78852)).
- Add option to center image when loading particle emission mask ([GH-78944](https://github.com/godotengine/godot/pull/78944)).
- Unify error condition for particles trail lifetime ([GH-79270](https://github.com/godotengine/godot/pull/79270)).
- Particle internal refactor and additions for more artistic control ([GH-79527](https://github.com/godotengine/godot/pull/79527)).
- Fix particle shader deterministic random values ([GH-80638](https://github.com/godotengine/godot/pull/80638)).
- Add motion vector support for GPU 3D Particles ([GH-80688](https://github.com/godotengine/godot/pull/80688)).
- Implement conversion from `CPUParticles` to `GPUParticles` (3D/2D) ([GH-80779](https://github.com/godotengine/godot/pull/80779)).
- Fix GPUParticles2D offset stutter ([GH-80984](https://github.com/godotengine/godot/pull/80984)).
- Fix z-billboard + y to velocity transform alignment to correctly respect non-uniform scale ([GH-81315](https://github.com/godotengine/godot/pull/81315)).
- Fix errors when freeing GPUParticles ([GH-82431](https://github.com/godotengine/godot/pull/82431)).
- Fixed multiple particle issues: division by zero, color ramp override, scale dependent on amount ratio ([GH-83488](https://github.com/godotengine/godot/pull/83488)).
- Fix typo in particles process material when using emission color texture ([GH-83620](https://github.com/godotengine/godot/pull/83620)).
- Fix massive performance hit due to enabling collision ([GH-83749](https://github.com/godotengine/godot/pull/83749)).
- Fix directed points not working, and fix friction formula ([GH-83831](https://github.com/godotengine/godot/pull/83831)).
- Fix `noise_direction` variable used before initialized in particle shader when using turbulence with collisions ([GH-83881](https://github.com/godotengine/godot/pull/83881)).
- Fix invalid parameter ranges ([GH-84006](https://github.com/godotengine/godot/pull/84006)).
- Fix friction being in the correct if/else branch ([GH-84028](https://github.com/godotengine/godot/pull/84028)).
- Fix damp as friction not updating shader code ([GH-84029](https://github.com/godotengine/godot/pull/84029)).
- Fix wrong rotation matrix for orbit z velocity ([GH-84056](https://github.com/godotengine/godot/pull/84056)).
- Fix turbulence post rework ([GH-84103](https://github.com/godotengine/godot/pull/84103)).
- OpenGL: Fix uninitialized memory usage for GPUParticles `interp_to_end` ([GH-84189](https://github.com/godotengine/godot/pull/84189)).
- Fix several Material texture parameter updates ([GH-84303](https://github.com/godotengine/godot/pull/84303)).
- Fix several ParticleProcessMaterial texture names ([GH-84829](https://github.com/godotengine/godot/pull/84829)).
- Fix radial inwards velocity clamping incorrectly (regression from #83488) ([GH-85252](https://github.com/godotengine/godot/pull/85252)).

#### Physics

- Add ability to get face index and barycentric coordinates from raycast ([GH-71233](https://github.com/godotengine/godot/pull/71233)).
- Add Mass Distribution, Deactivation, Solver inspector property groups ([GH-77943](https://github.com/godotengine/godot/pull/77943)).
- Correctly set mass for a rigid body with custom inertia and center of mass ([GH-78757](https://github.com/godotengine/godot/pull/78757)).
- Add `hit_back_faces` property to `RayCast3D` ([GH-79330](https://github.com/godotengine/godot/pull/79330)).
- Add state sync after call to `_integrate_forces` in `_body_state_changed` ([GH-79977](https://github.com/godotengine/godot/pull/79977)).
- Fix unit suffix for `HingeJoint3D`'s target velocity ([GH-80523](https://github.com/godotengine/godot/pull/80523)).
- Fix gizmo for `BoxShape3D` ([GH-80689](https://github.com/godotengine/godot/pull/80689)).
- Expose the `get_rid` method of Joint2D and Joint3D ([GH-80736](https://github.com/godotengine/godot/pull/80736)).
- Fix possible crash when Control overrides mouse input on Area2D ([GH-81006](https://github.com/godotengine/godot/pull/81006)).
- Update PinJoint2D API with angle limits and motor speed ([GH-81610](https://github.com/godotengine/godot/pull/81610)).
- Fix missing clear for some `set_exclude*` query parameter methods ([GH-82043](https://github.com/godotengine/godot/pull/82043)).
- Fix performance regression in RigidBody2D/3D and PhysicalBone3D ([GH-82393](https://github.com/godotengine/godot/pull/82393)).
- Fix not refitting upward from leaf nodes ([GH-82482](https://github.com/godotengine/godot/pull/82482)).
- Tweak Gravity Scale property hints to make dragging more useful ([GH-82634](https://github.com/godotengine/godot/pull/82634)).
- Allow TileMap physics/navigation to still work when hidden ([GH-83560](https://github.com/godotengine/godot/pull/83560)).
- Fix unit suffixes for `Generic6DOFJoint` ([GH-83672](https://github.com/godotengine/godot/pull/83672)).
- Ensure SoftBody3D does not use compressed mesh format ([GH-84165](https://github.com/godotengine/godot/pull/84165)).
- Fix rotated tile collision not working at runtime ([GH-84261](https://github.com/godotengine/godot/pull/84261)).
- Fix transform changes in `_integrate_forces` being overwritten ([GH-84799](https://github.com/godotengine/godot/pull/84799)).
- Fix transform sync in `RigidBody*D::_body_state_changed` ([GH-84924](https://github.com/godotengine/godot/pull/84924)).
- Update tilemap physics' World2D on reparenting ([GH-84968](https://github.com/godotengine/godot/pull/84968)).

#### Plugin

- Add `_get_unsaved_status()` method to EditorPlugin and implement it for script and shader editors ([GH-67503](https://github.com/godotengine/godot/pull/67503)).
- Expose editor viewports in EditorInterface ([GH-68696](https://github.com/godotengine/godot/pull/68696)).
- Allow changing feature profile via `EditorInterface` ([GH-74382](https://github.com/godotengine/godot/pull/74382)).
- Fix Camera2D is not working inside a MainScreenEditorPlugin ([GH-79867](https://github.com/godotengine/godot/pull/79867)).
- Keep `_export_begin()`'s `path` argument always consistent ([GH-81016](https://github.com/godotengine/godot/pull/81016)).
- Relax restriction on loading v1 Android plugins on Godot 4.2+ ([GH-81368](https://github.com/godotengine/godot/pull/81368)).
- Cleanups and improvements to the Godot Android library api ([GH-82893](https://github.com/godotengine/godot/pull/82893)).
- Editor: Fix `remove_control_from_dock` fails when dock is floating ([GH-83512](https://github.com/godotengine/godot/pull/83512)).

#### Porting

- [macOS, sandbox] Implement optional native file selection dialog support for sandboxed apps ([GH-47499](https://github.com/godotengine/godot/pull/47499)).
- Add `clipboard_has/get_image` methods to DisplayServer ([GH-63826](https://github.com/godotengine/godot/pull/63826)).
- Refactor Godot Android architecture ([GH-76821](https://github.com/godotengine/godot/pull/76821)).
- Windows: Flash both the window caption and taskbar button on `request_attention` ([GH-78263](https://github.com/godotengine/godot/pull/78263)).
- Add error checks and harmonize behavior of the `set_icon` method ([GH-78437](https://github.com/godotengine/godot/pull/78437)).
- Fix formatting of dlopen error message on Windows ([GH-78802](https://github.com/godotengine/godot/pull/78802)).
- macOS: Fix `set_native_icon` crash with empty or invalid ICNS file ([GH-79010](https://github.com/godotengine/godot/pull/79010)).
- Windows: Fix setting initial non-exclusive window mode ([GH-79016](https://github.com/godotengine/godot/pull/79016)).
- [macOS/iOS] Set MoltenVK logging level based on `--verbose` flag ([GH-79061](https://github.com/godotengine/godot/pull/79061)).
- Fix the fallback logic of `OS::shell_show_in_file_manager` ([GH-79087](https://github.com/godotengine/godot/pull/79087)).
- Avoid freeze when interacting with menus on Wayland by re-aquiring next swapchain image after updating swapchain ([GH-79143](https://github.com/godotengine/godot/pull/79143)).
- Fix Linux `move_to_trash` wrongly reporting files as not found ([GH-79284](https://github.com/godotengine/godot/pull/79284)).
- Fix `ProjectSettings::localize_path` for Windows paths ([GH-79342](https://github.com/godotengine/godot/pull/79342)).
- Windows: Implement native file selection dialog support ([GH-79574](https://github.com/godotengine/godot/pull/79574)).
- Fix NullPointerException when registering the sensors ([GH-79681](https://github.com/godotengine/godot/pull/79681)).
- Windows: Initialize COM as apartment-threaded ([GH-79693](https://github.com/godotengine/godot/pull/79693)).
- Add `proxy_to_pthread` option to `platform=web` ([GH-79711](https://github.com/godotengine/godot/pull/79711)).
- Fix file permissions for the web platform (affects every Unix-like platform) ([GH-79866](https://github.com/godotengine/godot/pull/79866)).
- Use EWMH for `DisplayServerX11::_window_minimize_check()` implementation ([GH-80036](https://github.com/godotengine/godot/pull/80036)).
- Web: Update npm packages ([GH-80092](https://github.com/godotengine/godot/pull/80092)).
- [Linux/Freedesktop] Implement native file selection dialog support ([GH-80104](https://github.com/godotengine/godot/pull/80104)).
- Windows: Do not force redraw window background on mouse pass-through region change ([GH-80153](https://github.com/godotengine/godot/pull/80153)).
- X11: Do not fail DisplayServer init if non-essential extensions are missing ([GH-80240](https://github.com/godotengine/godot/pull/80240)).
- Track hovered `Window` in `DisplayServerX11` ([GH-80279](https://github.com/godotengine/godot/pull/80279)).
- FileAccess: Add methods to get/set "hidden" and "read-only" attributes on macOS/BSD and Windows ([GH-80404](https://github.com/godotengine/godot/pull/80404)).
- DisplayServer: Add method to estimate window title bar size ([GH-80409](https://github.com/godotengine/godot/pull/80409)).
- macOS: Fix missing mouse exit events on window close ([GH-80439](https://github.com/godotengine/godot/pull/80439)).
- Android: Change the default "org.godotengine" package name to "com.example" ([GH-80761](https://github.com/godotengine/godot/pull/80761)).
- [Native File Dialogs] Refocus last focused window on close ([GH-80952](https://github.com/godotengine/godot/pull/80952)).
- Make Windows' safe save more resilient ([GH-81001](https://github.com/godotengine/godot/pull/81001)).
- Fix JavaScript callback memory leak issue ([GH-81105](https://github.com/godotengine/godot/pull/81105)).
- [Native File Dialogs] Improve filter list handling, add selected filter to the callback ([GH-81218](https://github.com/godotengine/godot/pull/81218)).
- macOS: Fix live resize with the latest MoltenVK version ([GH-81339](https://github.com/godotengine/godot/pull/81339)).
- Implement `clipboard_get`/`has_image` for X11 ([GH-81439](https://github.com/godotengine/godot/pull/81439)).
- Web: Disable raycast module by default (no occlusion culling) ([GH-81716](https://github.com/godotengine/godot/pull/81716)).
- Windows: Use clear color for non exclusive fullscreen border, fix maximize for borderless window switching to exclusive fs ([GH-82031](https://github.com/godotengine/godot/pull/82031)).
- macOS: Enforce non-zero window size ([GH-82037](https://github.com/godotengine/godot/pull/82037)).
- X11: Add support for using EGL/GLES instead of GLX ([GH-82101](https://github.com/godotengine/godot/pull/82101)).
- Support dark mode on Android and iOS ([GH-82230](https://github.com/godotengine/godot/pull/82230)).
- macOS: Fix borderless mode on macOS 13.6+ ([GH-82357](https://github.com/godotengine/godot/pull/82357)).
- macOS: Check all exclusive fullscreen windows before setting presentation mode ([GH-82423](https://github.com/godotengine/godot/pull/82423)).
- Web: Clarify that `OS.get_unique_id` is not supported ([GH-82441](https://github.com/godotengine/godot/pull/82441)).
- Fix `godot_js_wrapper_create_cb` regression ([GH-82736](https://github.com/godotengine/godot/pull/82736)).
- macOS: Fix ambiguous method call with older SDKs ([GH-82876](https://github.com/godotengine/godot/pull/82876)).
- Add method to check if filesystem is case sensitive ([GH-82957](https://github.com/godotengine/godot/pull/82957)).
- macOS: Use occlusionState instead of isOnActiveSpace to determine when window is drawable ([GH-83096](https://github.com/godotengine/godot/pull/83096)).
- Web: Improve Emscripten `locateFile` glue ([GH-83165](https://github.com/godotengine/godot/pull/83165)).
- Add error messages to the native menu and file dialogs callback ([GH-83181](https://github.com/godotengine/godot/pull/83181)).
- macOS: Fix crash when using system default menu shortcuts ([GH-83243](https://github.com/godotengine/godot/pull/83243)).
- Linux: Implement `DirAccess.is_case_sensitive` for EXT4 and F2FS ([GH-83266](https://github.com/godotengine/godot/pull/83266)).
- Improve X11 `screen_get_refresh_rate` performance ([GH-83902](https://github.com/godotengine/godot/pull/83902)).
- Add support for EGL 1.4 ([GH-83930](https://github.com/godotengine/godot/pull/83930)).
- Update the `launchMode` for the `GodotApp` activity ([GH-83954](https://github.com/godotengine/godot/pull/83954)).
- Fix freeze when requesting clipboard image from our own window ([GH-83970](https://github.com/godotengine/godot/pull/83970)).
- Fix macOS and Windows build with statically linked ANGLE/EGL ([GH-83988](https://github.com/godotengine/godot/pull/83988)).
- TTS_Linux: Fix size_t template issue on OpenBSD by using int consistently ([GH-84017](https://github.com/godotengine/godot/pull/84017)).
- Fix retrieving command line flags in Android ([GH-84102](https://github.com/godotengine/godot/pull/84102)).
- Fix Android editor crash issue when pressing Back ([GH-84414](https://github.com/godotengine/godot/pull/84414)).
- Fix Android disabling splash screen Show Image ([GH-84491](https://github.com/godotengine/godot/pull/84491)).
- Fix bug where maximized->fullscreen->windowed mode stays maximized ([GH-84504](https://github.com/godotengine/godot/pull/84504)).
- X11: Add fallback from desktop GL to GLES, suppress PRIME detector error spam ([GH-84513](https://github.com/godotengine/godot/pull/84513)).
- macOS: Fix fullscreen <-> exclusive fullscreen transition ([GH-84649](https://github.com/godotengine/godot/pull/84649)).
- macOS: Fix transparent and borderless flags interaction with full-screen mode ([GH-84683](https://github.com/godotengine/godot/pull/84683)).
- macOS: Process events before changing title style to update window frame ([GH-84927](https://github.com/godotengine/godot/pull/84927)).
- Fix issue causing Godot Android apps / games to freeze on close ([GH-85454](https://github.com/godotengine/godot/pull/85454)).
- Fix an issue causing the running project window to loop-restart when closed ([GH-85457](https://github.com/godotengine/godot/pull/85457)).

#### Rendering

- Fix directional LightmapGI being too dark with static lights ([GH-61910](https://github.com/godotengine/godot/pull/61910)).
- [macOS/Windows] Add optional ANGLE backed OpenGL renderer support (runtime backend selection) ([GH-72831](https://github.com/godotengine/godot/pull/72831)).
- Abort on startup with a visible alert if required Vulkan features are missing ([GH-73999](https://github.com/godotengine/godot/pull/73999)).
- Add content scale stretch modes, implement integer scaling ([GH-75784](https://github.com/godotengine/godot/pull/75784)).
- Draw frustum splices on top of direction shadow atlas for debug purposes ([GH-77085](https://github.com/godotengine/godot/pull/77085)).
- Split raster barrier into vertex and fragment barrier ([GH-77420](https://github.com/godotengine/godot/pull/77420)).
- Implement 3D shadows in the GL Compatibility renderer ([GH-77496](https://github.com/godotengine/godot/pull/77496)).
- Replace sampler arrays with constant sampler elements, simplify and reuse code for all shaders ([GH-77740](https://github.com/godotengine/godot/pull/77740)).
- Add support for GLSL source-level debugging with RenderDoc ([GH-77975](https://github.com/godotengine/godot/pull/77975)).
- Use Gaussian approximation for backbuffer mipmaps in GL Compatibility renderer ([GH-78168](https://github.com/godotengine/godot/pull/78168)).
- Clear specular buffer if sky mode is canvas and screen space effects are used ([GH-78624](https://github.com/godotengine/godot/pull/78624)).
- Fix threading bug in Vulkan rendering device ([GH-78794](https://github.com/godotengine/godot/pull/78794)).
- Fix sanitizers reports about octahedral tangents in RenderingServer ([GH-78902](https://github.com/godotengine/godot/pull/78902)).
- Take eye offset into account for depth in StandardMaterial3D ([GH-79049](https://github.com/godotengine/godot/pull/79049)).
- Expose RenderSceneBuffers(RD) through ClassDB ([GH-79142](https://github.com/godotengine/godot/pull/79142)).
- Clear the previously set state when configuring for a new scene root node ([GH-79201](https://github.com/godotengine/godot/pull/79201)).
- Add custom texture create function ([GH-79288](https://github.com/godotengine/godot/pull/79288)).
- Fix missing `_THREAD_SAFE_METHOD_` missing from `RenderingDeviceVulkan` `submit` and `sync` ([GH-79526](https://github.com/godotengine/godot/pull/79526)).
- Fix crash when calling `get_video_adapter_*` in a thread ([GH-79528](https://github.com/godotengine/godot/pull/79528)).
- GLES3: Reset anisotropic filtering when changing texture filtering mode ([GH-79568](https://github.com/godotengine/godot/pull/79568)).
- Fix bad LOD selection when Camera in Mesh AABB ([GH-79590](https://github.com/godotengine/godot/pull/79590)).
- Fix instance uniform data buffer update delay ([GH-79603](https://github.com/godotengine/godot/pull/79603)).
- ShaderRD compilation groups ([GH-79606](https://github.com/godotengine/godot/pull/79606)).
- Revert the change of the limit for interpolation of F0 for dielectrics and metals for Screen Space Reflections ([GH-79624](https://github.com/godotengine/godot/pull/79624)).
- Fix GLES3 multimesh rendering when using colors or custom data ([GH-79660](https://github.com/godotengine/godot/pull/79660)).
- GLES3: Don't call `glTexParameter*` for invalid filter and repeat modes ([GH-79685](https://github.com/godotengine/godot/pull/79685)).
- Add ability to call code on rendering thread ([GH-79696](https://github.com/godotengine/godot/pull/79696)).
- Unbind the framebuffer when updating meshes ([GH-79772](https://github.com/godotengine/godot/pull/79772)).
- Mobile: Uncomment code required for fog rendering on clear color ([GH-79776](https://github.com/godotengine/godot/pull/79776)).
- Use defaults to initialize sky data in case of no sky ([GH-79812](https://github.com/godotengine/godot/pull/79812)).
- Fix Vulkan multithreaded compute list and GPU particle processing ([GH-79849](https://github.com/godotengine/godot/pull/79849)).
- Fix use of discard in shaders ([GH-79865](https://github.com/godotengine/godot/pull/79865)).
- Lazily allocate RIDs for PlaceholderTextures to avoid allocating GPU resources unless used ([GH-79874](https://github.com/godotengine/godot/pull/79874)).
- Fix transparent viewport backgrounds with custom clear color ([GH-79876](https://github.com/godotengine/godot/pull/79876)).
- Check if shader cache directory is available before using cache ([GH-79883](https://github.com/godotengine/godot/pull/79883)).
- Vulkan: Fix dangling pointers in `_clean_up_swap_chain` ([GH-79884](https://github.com/godotengine/godot/pull/79884)).
- Add exceptions for breakage introduced in RD barriers ([GH-79911](https://github.com/godotengine/godot/pull/79911)).
- Make Fresnel darken SSR instead of blending with specular ([GH-79921](https://github.com/godotengine/godot/pull/79921)).
- Initialize MSDF parameters in BaseMaterial3D with default ([GH-79983](https://github.com/godotengine/godot/pull/79983)).
- Fix uninitialized variable ending up sent to Vulkan ([GH-80034](https://github.com/godotengine/godot/pull/80034)).
- Enable depth writes during shadow pass and depth pass. Disable during color pass ([GH-80070](https://github.com/godotengine/godot/pull/80070)).
- Fix validation layer warnings ([GH-80071](https://github.com/godotengine/godot/pull/80071)).
- Fix GLES3 changing 2D shadow atlas size is broken ([GH-80151](https://github.com/godotengine/godot/pull/80151)).
- Add option to enable HDR rendering in 2D ([GH-80215](https://github.com/godotengine/godot/pull/80215)).
- Initialize shader placeholders up front ([GH-80222](https://github.com/godotengine/godot/pull/80222)).
- Fix motion vectors being corrupted when using `precision=double` ([GH-80257](https://github.com/godotengine/godot/pull/80257)).
- Vulkan: Fix sanitizers error with empty shader name ([GH-80288](https://github.com/godotengine/godot/pull/80288)).
- Enhance Vulkan PSO caching ([GH-80296](https://github.com/godotengine/godot/pull/80296)).
- Use fullscreen tri instead of quad ([GH-80311](https://github.com/godotengine/godot/pull/80311)).
- Fix validation error when enabling SSIL alone ([GH-80315](https://github.com/godotengine/godot/pull/80315)).
- Ensure `POINT_SIZE` takes effect in the canvas item shader ([GH-80323](https://github.com/godotengine/godot/pull/80323)).
- Fix integer underflow when rounding up in VoxelGI ([GH-80356](https://github.com/godotengine/godot/pull/80356)).
- Fix issue with four subpasses always been requested in mobile renderer ([GH-80368](https://github.com/godotengine/godot/pull/80368)).
- Remove GPU readback from `NoiseTexture3D.get_format()` ([GH-80407](https://github.com/godotengine/godot/pull/80407)).
- Improve handling of motion vectors for multimesh instances ([GH-80414](https://github.com/godotengine/godot/pull/80414)).
- Add `buffer_copy` method to RenderingDevice ([GH-80424](https://github.com/godotengine/godot/pull/80424)).
- Clamp Volumetric Fog Length property to prevent rendering issues ([GH-80485](https://github.com/godotengine/godot/pull/80485)).
- Fix tonemapper, incorrect vertex count was specified ([GH-80502](https://github.com/godotengine/godot/pull/80502)).
- Fix critical regressions introduced in PR #80414 ([GH-80552](https://github.com/godotengine/godot/pull/80552)).
- Fix validation error when resizing window ([GH-80571](https://github.com/godotengine/godot/pull/80571)).
- Add motion vector support for animated surfaces ([GH-80618](https://github.com/godotengine/godot/pull/80618)).
- Fallback to linear color texture when using 2D HDR and MSDF font ([GH-80651](https://github.com/godotengine/godot/pull/80651)).
- Fix global shader uniform texture loading ([GH-80654](https://github.com/godotengine/godot/pull/80654)).
- Improve visual feedback when using the motion vectors debug view option ([GH-80723](https://github.com/godotengine/godot/pull/80723)).
- Fix Vulkan texture update ([GH-80781](https://github.com/godotengine/godot/pull/80781)).
- Fix memory access error for `MultiMesh` with GLES3 ([GH-80788](https://github.com/godotengine/godot/pull/80788)).
- Fix Vulkan crash with many Omni/SpotLights, Decals or ReflectionProbes ([GH-80845](https://github.com/godotengine/godot/pull/80845)).
- Implement OpenXR Foveated rendering support ([GH-80881](https://github.com/godotengine/godot/pull/80881)).
- Clear SDFGI textures when created ([GH-80889](https://github.com/godotengine/godot/pull/80889)).
- Fix integer value for `GL_MAX_UNIFORM_BLOCK_SIZE` overflowing ([GH-80909](https://github.com/godotengine/godot/pull/80909)).
- Fix missing decal mask in mobile renderer ([GH-80911](https://github.com/godotengine/godot/pull/80911)).
- Fix clear color on mobile renderer ([GH-80933](https://github.com/godotengine/godot/pull/80933)).
- Fix volumetric fog NaN values in textures from starting at a zero Vector2 ([GH-80992](https://github.com/godotengine/godot/pull/80992)).
- GLES3: Fix `glMapBufferRange` return null when `r_index + last_item_index > max_instance` ([GH-81036](https://github.com/godotengine/godot/pull/81036)).
- Fix missing `EARLY_FRAGMENT_TESTS_BIT` barrier flags ([GH-81059](https://github.com/godotengine/godot/pull/81059)).
- Fix VoxelGI CameraAttributes exposure normalization handling ([GH-81067](https://github.com/godotengine/godot/pull/81067)).
- Flip convention of motion vectors ([GH-81074](https://github.com/godotengine/godot/pull/81074)).
- Fixup special case of cluster render ([GH-81081](https://github.com/godotengine/godot/pull/81081)).
- Fix VoxelGI static light pairing ([GH-81124](https://github.com/godotengine/godot/pull/81124)).
- Vertex and attribute compression ([GH-81138](https://github.com/godotengine/godot/pull/81138)).
- Add render mode to use world coordinates in canvas item shader ([GH-81160](https://github.com/godotengine/godot/pull/81160)).
- Reset SDFGI when changing editor scene tabs ([GH-81167](https://github.com/godotengine/godot/pull/81167)).
- Add FidelityFX Super Resolution 2.2 (FSR 2.2.1) support ([GH-81197](https://github.com/godotengine/godot/pull/81197)).
- Add placeholder RID to GradientTexture1D ([GH-81198](https://github.com/godotengine/godot/pull/81198)).
- vulkan: Update all components to Vulkan SDK 1.3.261.1 ([GH-81219](https://github.com/godotengine/godot/pull/81219)).
- Windows: Fix not applying NVIDIA profile to new executables ([GH-81251](https://github.com/godotengine/godot/pull/81251)).
- Implement render mode `fog_disabled` and BaseMaterial3D setting Disable Fog ([GH-81286](https://github.com/godotengine/godot/pull/81286)).
- Use 16-bit index buffers instead of 32 when unnecessary ([GH-81288](https://github.com/godotengine/godot/pull/81288)).
- Fix `RDPipelineColorBlendState.attachments` setter ([GH-81333](https://github.com/godotengine/godot/pull/81333)).
- Fix mipmap bias behavior by refactoring how samplers are created by Material Storage ([GH-81350](https://github.com/godotengine/godot/pull/81350)).
- Fix clear color's alpha value will affects 2D editor in Compatibility mode ([GH-81395](https://github.com/godotengine/godot/pull/81395)).
- Propagate error correctly when max texture size for lightmaps is too small ([GH-81543](https://github.com/godotengine/godot/pull/81543)).
- Fix LightmapGI baking with GridMap ([GH-81545](https://github.com/godotengine/godot/pull/81545)).
- Fix GLES3 instanced rendering color and custom data defaults ([GH-81575](https://github.com/godotengine/godot/pull/81575)).
- Fix VoxelGI MultiMesh and CSG mesh baking ([GH-81616](https://github.com/godotengine/godot/pull/81616)).
- Improve GLES3 scene renderer compatibility with older devices ([GH-81650](https://github.com/godotengine/godot/pull/81650)).
- Replace OIDN denoiser in Lightmapper with a JNLM denoiser compute shader ([GH-81659](https://github.com/godotengine/godot/pull/81659)).
- Fix validation error when using pipeline cache control ([GH-81771](https://github.com/godotengine/godot/pull/81771)).
- Fix massive validation errors when enabling TAA + MSAA ([GH-81775](https://github.com/godotengine/godot/pull/81775)).
- Opt-in to Vulkan features we actually use ([GH-81827](https://github.com/godotengine/godot/pull/81827)).
- Add half-pixel offset to lightmapper rasterization ([GH-81872](https://github.com/godotengine/godot/pull/81872)).
- Polish a few things in Vulkan RD ([GH-81912](https://github.com/godotengine/godot/pull/81912)).
- Fix LightmapGI shading sometimes being unlit or black ([GH-81951](https://github.com/godotengine/godot/pull/81951)).
- Rewrite the GPU Lightmapper's indirect logic to match Godot 3.5's CPU Lightmapper ([GH-82068](https://github.com/godotengine/godot/pull/82068)).
- Fix ShaderGlobalsOverride property handling ([GH-82100](https://github.com/godotengine/godot/pull/82100)).
- Linux/OpenGL: Don't force vsync in the editor ([GH-82221](https://github.com/godotengine/godot/pull/82221)).
- Fix RaycastOcclusionCull World3D scenario memory leak ([GH-82291](https://github.com/godotengine/godot/pull/82291)).
- Optimizing glow behavior ([GH-82353](https://github.com/godotengine/godot/pull/82353)).
- Add device info to GLES3 shader cache key hash ([GH-82359](https://github.com/godotengine/godot/pull/82359)).
- ANGLE: Add fallback control options and defaults ([GH-82364](https://github.com/godotengine/godot/pull/82364)).
- Clamp ReflectionProbe Max Distance to 262,144 to fix rendering issues ([GH-82415](https://github.com/godotengine/godot/pull/82415)).
- Fix Decal clamping to positive values not being applied to RenderingServer ([GH-82416](https://github.com/godotengine/godot/pull/82416)).
- GLES3: Avoid freeing proxy textures clearing owner's data ([GH-82430](https://github.com/godotengine/godot/pull/82430)).
- Avoid trying to free null RIDs in FSR2 teardown ([GH-82445](https://github.com/godotengine/godot/pull/82445)).
- Fix mismatch between surface vertex array generation inside the function and the caller ([GH-82451](https://github.com/godotengine/godot/pull/82451)).
- Disable `lightmapper_rd` module in non-editor builds (and in Android editor) ([GH-82521](https://github.com/godotengine/godot/pull/82521)).
- Make the lightmapper not dilate before denoising ([GH-82533](https://github.com/godotengine/godot/pull/82533)).
- Use internal texture at internal resolution for calculating luminance (FSR2) ([GH-82534](https://github.com/godotengine/godot/pull/82534)).
- Fix cluster artifacts and negative light ([GH-82546](https://github.com/godotengine/godot/pull/82546)).
- Workaround crash due to null shader when running XR project with `--xr-mode` off ([GH-82679](https://github.com/godotengine/godot/pull/82679)).
- OpenXR: Properly skip frame render when the XR runtime is not yet ready ([GH-82752](https://github.com/godotengine/godot/pull/82752)).
- Forgot to add debanding to config object ([GH-82766](https://github.com/godotengine/godot/pull/82766)).
- Re-add optional OIDN denoise as an external executable ([GH-82832](https://github.com/godotengine/godot/pull/82832)).
- Fog shader: Fix undeclared identifier `global_variables` ([GH-82877](https://github.com/godotengine/godot/pull/82877)).
- Avoid default fallback material when using `world_vertex_coords` ([GH-82886](https://github.com/godotengine/godot/pull/82886)).
- Only perform modelview transform on tangent and binormal when vertex shader is in local space ([GH-82892](https://github.com/godotengine/godot/pull/82892)).
- Directional 2 Split Shadow stabilization fix ([GH-82974](https://github.com/godotengine/godot/pull/82974)).
- Fix VoxelGI bake memory leak ([GH-83035](https://github.com/godotengine/godot/pull/83035)).
- Fix `trace_ray()` function in the lightmapper missing hits with large triangles ([GH-83040](https://github.com/godotengine/godot/pull/83040)).
- Fix incorrect check in `_dict_to_surf` ([GH-83056](https://github.com/godotengine/godot/pull/83056)).
- Fix incorrect vertex data size calculation in `ImmediateMesh` ([GH-83100](https://github.com/godotengine/godot/pull/83100)).
- Fix compatibility shadow size not being initialized ([GH-83141](https://github.com/godotengine/godot/pull/83141)).
- Disable update spinner when debug redraw is active ([GH-83143](https://github.com/godotengine/godot/pull/83143)).
- Fix BaseMaterial3D update with certain material settings ([GH-83145](https://github.com/godotengine/godot/pull/83145)).
- Fix a few cases where surface format is still 32 bits ([GH-83169](https://github.com/godotengine/godot/pull/83169)).
- Sanitize tangents when creating mesh surfaces to avoid triggering the compressed mesh path in the shader ([GH-83179](https://github.com/godotengine/godot/pull/83179)).
- Add an extra backbuffer color texture that can be used when an upscaler is in use ([GH-83192](https://github.com/godotengine/godot/pull/83192)).
- Fix `TextureStorage` not assigning default scale ([GH-83199](https://github.com/godotengine/godot/pull/83199)).
- Cleanup instances of using uint32_t for mesh formats ([GH-83211](https://github.com/godotengine/godot/pull/83211)).
- Fix OpenGL directional shadow last split fading ([GH-83252](https://github.com/godotengine/godot/pull/83252)).
- Optimize lightmapper using triangle clusters on the acceleration structure ([GH-83284](https://github.com/godotengine/godot/pull/83284)).
- Fix disabling depth prepass break opaque materials ([GH-83371](https://github.com/godotengine/godot/pull/83371)).
- Fix Mobile renderer shader instance uniform access ([GH-83400](https://github.com/godotengine/godot/pull/83400)).
- Pass viewport size to shadow pass instead of using Vector2i(1,1) ([GH-83491](https://github.com/godotengine/godot/pull/83491)).
- Ensure that only visible paired lights are used ([GH-83493](https://github.com/godotengine/godot/pull/83493)).
- Bump version of Vulkan RD binary shader format ([GH-83563](https://github.com/godotengine/godot/pull/83563)).
- Fix shadow map debug visualization camera frustum index buffer size ([GH-83639](https://github.com/godotengine/godot/pull/83639)).
- Fixing incorrect normal map when using triplanar world mapping and mesh rotation ([GH-83658](https://github.com/godotengine/godot/pull/83658)).
- Some more fixes for compressed meshes ([GH-83704](https://github.com/godotengine/godot/pull/83704)).
- macOS: Fallback to native OpenGL renderer if ANGLE initialization failed ([GH-83753](https://github.com/godotengine/godot/pull/83753)).
- Fix `GPUParticles3D` on the Meta Quest 2 with OpenGL renderer ([GH-83756](https://github.com/godotengine/godot/pull/83756)).
- Add property hint for 2D shadow size project setting ([GH-83760](https://github.com/godotengine/godot/pull/83760)).
- Ensure `r_aabb` is always used when creating surfaces through the RenderingServer ([GH-83840](https://github.com/godotengine/godot/pull/83840)).
- Fix LightmapGI taking editor-only and sky-only lights into account ([GH-83861](https://github.com/godotengine/godot/pull/83861)).
- Add padding to normal attribute in Compatibility renderer to match the RD renderers ([GH-83906](https://github.com/godotengine/godot/pull/83906)).
- Fix reading shadow filter quality from project settings in compatibility ([GH-83998](https://github.com/godotengine/godot/pull/83998)).
- Fix crash when upgrading meshes from 3.x format ([GH-84047](https://github.com/godotengine/godot/pull/84047)).
- Fix multiple issues with UV compression ([GH-84159](https://github.com/godotengine/godot/pull/84159)).
- Parse OpenGL and Vulkan strings as UTF-8 ([GH-84197](https://github.com/godotengine/godot/pull/84197)).
- Overhaul the SurfaceUpgradeTool ([GH-84200](https://github.com/godotengine/godot/pull/84200)).
- Fix bug with alpha to coverage by enabling depth discard when using alpha to coverage ([GH-84211](https://github.com/godotengine/godot/pull/84211)).
- Fix cubemap downsampler logic ([GH-84223](https://github.com/godotengine/godot/pull/84223)).
- Fix WebXR on desktop emulator by resetting active texture unit ([GH-84267](https://github.com/godotengine/godot/pull/84267)).
- macOS: Improve ANGLE support detection ([GH-84288](https://github.com/godotengine/godot/pull/84288)).
- Use default samplers in base uniform set when rendering to reflection probes ([GH-84317](https://github.com/godotengine/godot/pull/84317)).
- Windows: Add some AMD GPUs to the OpenGL blocklist ([GH-84568](https://github.com/godotengine/godot/pull/84568)).
- Create tangent array if mesh created without tangents ([GH-84576](https://github.com/godotengine/godot/pull/84576)).
- Fix FogMaterial memory leak ([GH-84702](https://github.com/godotengine/godot/pull/84702)).
- GLES3: Protect against bogus `glGetShaderInfoLog` return values ([GH-84741](https://github.com/godotengine/godot/pull/84741)).
- GLES3: Ensure all ShaderData is properly initialized in `set_code` ([GH-84752](https://github.com/godotengine/godot/pull/84752)).
- Ensure optional CopyEffects variants are loaded last ([GH-84883](https://github.com/godotengine/godot/pull/84883)).
- Renderer Viewport correct `sizeof` usage ([GH-84952](https://github.com/godotengine/godot/pull/84952)).
- GLES3: Fix iOS Simulator by removing incorrect `system_fbo` overwrite ([GH-84955](https://github.com/godotengine/godot/pull/84955)).
- Ensure 2D MSAA resolve is performed when 3D content but no 2D content in scene ([GH-84957](https://github.com/godotengine/godot/pull/84957)).
- Prevent crash in `_nvapi_disable_threaded_optimization` when attached to renderdoc ([GH-85121](https://github.com/godotengine/godot/pull/85121)).
- Avoid division by zero in the fix surface compatibility routine ([GH-85138](https://github.com/godotengine/godot/pull/85138)).
- Fix potential double-close of draw command label ([GH-85147](https://github.com/godotengine/godot/pull/85147)).
- Enable non-multiview advanced shader group whenever advanced shaders are requested ([GH-85194](https://github.com/godotengine/godot/pull/85194)).

#### Shaders

- Improve shader editor templates to be more descriptive ([GH-51863](https://github.com/godotengine/godot/pull/51863)).
- Add more useful Visual Shader nodes ([GH-72664](https://github.com/godotengine/godot/pull/72664)).
- Add DEPTH to the visual shader output (for spatial mode) ([GH-73691](https://github.com/godotengine/godot/pull/73691)).
- Support shader preprocessor concatenation symbol ([GH-74737](https://github.com/godotengine/godot/pull/74737)).
- Make the dragging connections more user-friendly in visual shaders ([GH-78547](https://github.com/godotengine/godot/pull/78547)).
- Fix invalid shader compilation when using `hint_normal_roughness_texture` in mobile backend ([GH-78839](https://github.com/godotengine/godot/pull/78839)).
- Fix using uint suffix at the hex number declaration in shaders ([GH-78906](https://github.com/godotengine/godot/pull/78906)).
- Fix shader language float literal precision truncation ([GH-78972](https://github.com/godotengine/godot/pull/78972)).
- Fix "Create Shader Node" window position when visual shader editor is floating ([GH-78996](https://github.com/godotengine/godot/pull/78996)).
- Allow more hint types for uniform arrays ([GH-79100](https://github.com/godotengine/godot/pull/79100)).
- Make shader preprocessor keyword colors consistent ([GH-79112](https://github.com/godotengine/godot/pull/79112)).
- Fix comments and indentation in `.gdshaderinc` files ([GH-79158](https://github.com/godotengine/godot/pull/79158)).
- Fix shader type detection ([GH-79287](https://github.com/godotengine/godot/pull/79287)).
- Add autocomplete for filter/repeat hints on uniform arrays ([GH-79402](https://github.com/godotengine/godot/pull/79402)).
- Add error for undefined function in shader ([GH-79459](https://github.com/godotengine/godot/pull/79459)).
- Remove debugging print from shader cache ([GH-80125](https://github.com/godotengine/godot/pull/80125)).
- Fix Shader and ShaderInclude resource loading ([GH-80705](https://github.com/godotengine/godot/pull/80705)).
- Fix empty shader resource loading ([GH-81300](https://github.com/godotengine/godot/pull/81300)).
- Fix shader language preprocessor include marker handling ([GH-81381](https://github.com/godotengine/godot/pull/81381)).
- Fix shader text editor include file reloading ([GH-81410](https://github.com/godotengine/godot/pull/81410)).
- Fix int to uint implicit cast error when use mat3 uniform in compatibility renderer ([GH-81494](https://github.com/godotengine/godot/pull/81494)).
- Re-allows constants in global space to be initialized with function call ([GH-81619](https://github.com/godotengine/godot/pull/81619)).
- Implement drop-down list properties to the custom visual shader nodes ([GH-81688](https://github.com/godotengine/godot/pull/81688)).
- Visual Shaders: Make output-ports for vector types expandable by default ([GH-82088](https://github.com/godotengine/godot/pull/82088)).
- Add preprocessor pass on visual shader when showing generated code ([GH-82570](https://github.com/godotengine/godot/pull/82570)).
- Fix typo in `shader_language.cpp` ([GH-83004](https://github.com/godotengine/godot/pull/83004)).
- Close shader in Shader Editor tab when deleting shader file in FileSystem panel ([GH-83137](https://github.com/godotengine/godot/pull/83137)).
- Fix parameter shader node not declared when only connected to a VaryingSetter ([GH-83189](https://github.com/godotengine/godot/pull/83189)).
- Fix bool varying's generated code will be modified with flat ([GH-83194](https://github.com/godotengine/godot/pull/83194)).
- Fix visual shader crash when arranging ([GH-83678](https://github.com/godotengine/godot/pull/83678)).
- Prevent `_allocate_placeholders` crash if `p_version->variants` is null ([GH-83780](https://github.com/godotengine/godot/pull/83780)).
- Fix inability to uncomment code in text shader editor ([GH-83822](https://github.com/godotengine/godot/pull/83822)).
- Fix assign with swizzle in shader not doing varying validation check ([GH-83830](https://github.com/godotengine/godot/pull/83830)).
- Properly rename `INSTANCE_ID` and `VERTEX_ID` in canvas item shaders in the compatibility backend ([GH-84585](https://github.com/godotengine/godot/pull/84585)).
- Don't store shader edit path in metadata ([GH-84628](https://github.com/godotengine/godot/pull/84628)).
- Fix VisualShader Texture2DParameter node filter bug ([GH-84768](https://github.com/godotengine/godot/pull/84768)).
- Fix VisualShader connection use after free ([GH-84832](https://github.com/godotengine/godot/pull/84832)).
- Make `AMOUNT_RATIO` constant in the shader language specification ([GH-85086](https://github.com/godotengine/godot/pull/85086)).
- Set some dialogs in the VisualShader editor to be exclusive ([GH-85205](https://github.com/godotengine/godot/pull/85205)).

#### Tests

- Add unit tests for Variant for operator overloading ([GH-76244](https://github.com/godotengine/godot/pull/76244)).
- Add a test suite for TranslationServer ([GH-79331](https://github.com/godotengine/godot/pull/79331)).
- Add unit tests for PackedScene ([GH-79440](https://github.com/godotengine/godot/pull/79440)).
- Add a test suite to InputEvent ([GH-79444](https://github.com/godotengine/godot/pull/79444)).
- Remove use of `std::string` in test ([GH-80422](https://github.com/godotengine/godot/pull/80422)).
- Improve PackedScene unit test for complex scene ([GH-80423](https://github.com/godotengine/godot/pull/80423)).
- Fix errors when testing `Resource` ([GH-81456](https://github.com/godotengine/godot/pull/81456)).
- Adds additional tests for RegEx class ([GH-82225](https://github.com/godotengine/godot/pull/82225)).
- Simplify Geometry3D tests ([GH-82288](https://github.com/godotengine/godot/pull/82288)).
- Fix Variant assignment to Vec2 tests ([GH-83959](https://github.com/godotengine/godot/pull/83959)).

#### Thirdparty

- brotli: Sync with upstream 1.1.0 ([GH-82580](https://github.com/godotengine/godot/pull/82580)).
- freetype: Update to version 2.13.2 ([GH-81110](https://github.com/godotengine/godot/pull/81110)).
- harfbuzz: Update to version 8.2.2 ([GH-84080](https://github.com/godotengine/godot/pull/84080)).
- libpng: Update to upstream 1.6.40 ([GH-80262](https://github.com/godotengine/godot/pull/80262)).
- libwebp: Sync with upstream 1.3.2 ([GH-81663](https://github.com/godotengine/godot/pull/81663)).
- mbedtls: Fix MSVC ARM build after 2.28.3 enabled AES-NI intrinsics ([GH-81405](https://github.com/godotengine/godot/pull/81405)).
- mbedtls: Update to version 2.28.5 ([GH-83721](https://github.com/godotengine/godot/pull/83721)).
- mbedtls: Backport Windows fix to use bcrypt for entropy ([GH-84042](https://github.com/godotengine/godot/pull/84042)).
- miniupnpc: Update to version 2.2.5 ([GH-80382](https://github.com/godotengine/godot/pull/80382)).
- minizip: Fix `ZIPReader` failing to open empty zip files ([GH-73310](https://github.com/godotengine/godot/pull/73310)).
- minizip: Backport patch to fix CVE-2023-45853 ([GH-85509](https://github.com/godotengine/godot/pull/85509)).
- openxr: Sync with upstream 1.0.31, don't build obsolete dispatch table ([GH-83984](https://github.com/godotengine/godot/pull/83984)).
- r128: Update to include latest fix for intrinsics being incorrect included ([GH-84537](https://github.com/godotengine/godot/pull/84537)).
- thorvg: Update to 0.11.2 ([GH-83656](https://github.com/godotengine/godot/pull/83656)).
- tinyexr: Sync with upstream 1.0.7 ([GH-80384](https://github.com/godotengine/godot/pull/80384)).
- zlib/minizip: Update to version 1.3 ([GH-81111](https://github.com/godotengine/godot/pull/81111)).

#### XR

- Expose OpenXR raw hand tracking data ([GH-78032](https://github.com/godotengine/godot/pull/78032)).
- Fix issue with accessing hand tracking without timing info ([GH-78817](https://github.com/godotengine/godot/pull/78817)).
- Change to new PICO interaction profiles ([GH-79570](https://github.com/godotengine/godot/pull/79570)).
- Compile OpenXR into MacOS build ([GH-79614](https://github.com/godotengine/godot/pull/79614)).
- Optimized the XRTracker by reusing XRPose objects to minimize garbage collection overhead in C# ([GH-80198](https://github.com/godotengine/godot/pull/80198)).
- Fix casts of XR handles in OpenXRExtensionWrapperExtension ([GH-80656](https://github.com/godotengine/godot/pull/80656)).
- Remove error print from `XRServer.find_interface` ([GH-80730](https://github.com/godotengine/godot/pull/80730)).
- Ensure OpenXR classes are declared properly ([GH-81037](https://github.com/godotengine/godot/pull/81037)).
- Add XR tracking state-change signals ([GH-81239](https://github.com/godotengine/godot/pull/81239)).
- OpenXR: Fix missing add profile for Huawei ([GH-81534](https://github.com/godotengine/godot/pull/81534)).
- OpenXR: Fix error spam if session hasn't started yet ([GH-81536](https://github.com/godotengine/godot/pull/81536)).
- Fix issue with OpenXR environment blend mode not being applied properly ([GH-81561](https://github.com/godotengine/godot/pull/81561)).
- Add support for the OpenXR Eye gaze interaction extension ([GH-82614](https://github.com/godotengine/godot/pull/82614)).
- OpenXR - add access to hand joint validity flags ([GH-82715](https://github.com/godotengine/godot/pull/82715)).
- OpenXR: Fix small hand tracking issues ([GH-82722](https://github.com/godotengine/godot/pull/82722)).
- Skip 2D rendering if stereo enabled and fix couple of MSAA issues ([GH-83649](https://github.com/godotengine/godot/pull/83649)).
- Revert to `proxy_to_pthread=no` as default ([GH-83837](https://github.com/godotengine/godot/pull/83837)).
- Fix OpenXR sample count ([GH-84099](https://github.com/godotengine/godot/pull/84099)).

## Past releases

- [4.1](https://github.com/godotengine/godot/blob/4.1-stable/CHANGELOG.md)
- [4.0](https://github.com/godotengine/godot/blob/4.0-stable/CHANGELOG.md)
- [3.5](https://github.com/godotengine/godot/blob/3.5-stable/CHANGELOG.md)
- [3.4](https://github.com/godotengine/godot/blob/3.4-stable/CHANGELOG.md)
- [3.3](https://github.com/godotengine/godot/blob/3.3-stable/CHANGELOG.md)
- [3.2](https://github.com/godotengine/godot/blob/3.2-stable/CHANGELOG.md)
- [3.1](https://github.com/godotengine/godot/blob/3.1-stable/CHANGELOG.md)
- [3.0](https://github.com/godotengine/godot/blob/f2e19a26f556c42b7202072296dc072aaac2007c/CHANGELOG.md)
