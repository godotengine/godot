# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [3.4] - 2021-11-05

See the [release announcement](https://godotengine.org/article/godot-3-4-is-released) for details.

### Added

#### 2D

- Add `Listener2D` node ([GH-53429](https://github.com/godotengine/godot/pull/53429)).
- Add a 2D Viewport scale factor property ([GH-52137](https://github.com/godotengine/godot/pull/52137)).

#### 3D

- Implement octahedral map normal/tangent attribute compression ([GH-46800](https://github.com/godotengine/godot/pull/46800)).
- Add a `center_offset` property to both plane primitive and quad primitive ([GH-48763](https://github.com/godotengine/godot/pull/48763)).
- Options to clean/simplify convex hull generated from mesh ([GH-50328](https://github.com/godotengine/godot/pull/50328)).
- Allow unclamped colors in `Sprite3D` ([GH-51462](https://github.com/godotengine/godot/pull/51462)).

#### Animation

- Add animation "reset" track feature ([GH-44558](https://github.com/godotengine/godot/pull/44558)).
- Allow renaming bones and blend shapes ([GH-42827](https://github.com/godotengine/godot/pull/42827)).

#### Core

- Add frame delta smoothing option ([GH-48390](https://github.com/godotengine/godot/pull/48390)).
  * This option is enabled by default (`application/run/delta_smoothing`).
- Add option to sync frame delta after draw ([GH-48555](https://github.com/godotengine/godot/pull/48555)).
  * This option is experimental and disabled by default (`application/run/delta_sync_after_draw`).
- Expose OS data directory getter methods ([GH-49732](https://github.com/godotengine/godot/pull/49732)).
- Provide a getter for the project data directory ([GH-52714](https://github.com/godotengine/godot/pull/52714)).
- Add an option to make the project data directory non-hidden ([GH-52556](https://github.com/godotengine/godot/pull/52556), [GH-53779](https://github.com/godotengine/godot/pull/53779)).
- Add support for numeric XML entities to `XMLParser` ([GH-47978](https://github.com/godotengine/godot/pull/47978)).
- Add option for BVH thread safety ([GH-48892](https://github.com/godotengine/godot/pull/48892)).
- Add `Engine.print_error_messages` property to disable printing errors ([GH-50640](https://github.com/godotengine/godot/pull/50640)).
- Implement `OS.get_locale_language()` helper method ([GH-52740](https://github.com/godotengine/godot/pull/52740)).
- Allow using global classes as project `MainLoop` implementation ([GH-52438](https://github.com/godotengine/godot/pull/52438)).
- Add an `Array.pop_at()` method to pop an element at an arbitrary index ([GH-52143](https://github.com/godotengine/godot/pull/52143)).
- Expose enum related methods in ClassDB ([GH-52572](https://github.com/godotengine/godot/pull/52572)).
- Add `Thread.is_alive()` method to check if the thread is still doing work ([GH-53490](https://github.com/godotengine/godot/pull/53490)).
- Allow for platform `Thread` implementation override ([GH-52734](https://github.com/godotengine/godot/pull/52734)).
- Add support for generating `OpenSimplexNoise` noise images with an offset ([GH-48805](https://github.com/godotengine/godot/pull/48805)).

#### Crypto

- Add `AESContext`, RSA public keys, encryption, decryption, sign, and verify ([GH-48144](https://github.com/godotengine/godot/pull/48144)).
- Add `HMACContext` ([GH-48869](https://github.com/godotengine/godot/pull/48869)).

#### Editor

- Automatic remote debugger port assignment ([GH-37067](https://github.com/godotengine/godot/pull/37067)).
- Auto-reload scripts with external editor ([GH-51828](https://github.com/godotengine/godot/pull/51828)).
- Use QuickOpen to load resources in the inspector ([GH-37228](https://github.com/godotengine/godot/pull/37228)).
- Allow to create a node at specific position ([GH-50242](https://github.com/godotengine/godot/pull/50242)).
- Add the ability to reorder array elements from the inspector ([GH-50651](https://github.com/godotengine/godot/pull/50651)).
- Assign value to property by dropping to scene tree ([GH-50700](https://github.com/godotengine/godot/pull/50700)).
- Allow dropping property path into script editor ([GH-51629](https://github.com/godotengine/godot/pull/51629)).
- Save branch as scene by dropping to filesystem ([GH-52503](https://github.com/godotengine/godot/pull/52503)).
- Allow creating nodes in Animation Blend Tree by dragging from in/out ports ([GH-52966](https://github.com/godotengine/godot/pull/52966)).
- Allow dragging multiple resources onto exported array variable at once ([GH-50718](https://github.com/godotengine/godot/pull/50718)).
- Add zoom support to `SpriteFrames` editor plugin ([GH-48977](https://github.com/godotengine/godot/pull/48977)).
- Add `EditorResourcePicker` and `EditorScriptPicker` classes for plugins (and internal editor use) ([GH-49491](https://github.com/godotengine/godot/pull/49491)).
- Add up/down keys to increment/decrement value in editor spin slider ([GH-53090](https://github.com/godotengine/godot/pull/53090)).
- Implement camera orbiting shortcuts ([GH-51984](https://github.com/godotengine/godot/pull/51984)).
- Add ability to copy group name ([GH-53162](https://github.com/godotengine/godot/pull/53162)).
- Implement a `%command%` placeholder in the Main Run Args setting ([GH-35992](https://github.com/godotengine/godot/pull/35992)).
- Add keyboard shortcuts to the project manager ([GH-47894](https://github.com/godotengine/godot/pull/47894)).
- Add history navigation in the script editor using extra mouse buttons ([GH-53067](https://github.com/godotengine/godot/pull/53067)).

#### GDScript

- Allow `warning-ignore` in the same line as the respective warning ([GH-47863](https://github.com/godotengine/godot/pull/47863)).
- LSP: Implement `didSave` notify and rename request ([GH-48616](https://github.com/godotengine/godot/pull/48616)).
- LSP: Add support for custom host setting ([GH-52330](https://github.com/godotengine/godot/pull/52330)).
- LSP: Implement `applyEdit` for signal connecting ([GH-53068](https://github.com/godotengine/godot/pull/53068)).

#### GUI

- Button: Add focus font color to `Button` and derivatives ([GH-54264](https://github.com/godotengine/godot/pull/54264)).
- ButtonGroup: Add a `pressed `signal ([GH-48500](https://github.com/godotengine/godot/pull/48500)).
- CheckBox: Add disabled theme icons ([GH-37755](https://github.com/godotengine/godot/pull/37755)).
- ColorPicker: Display previous color and allow selecting it back ([GH-48611](https://github.com/godotengine/godot/pull/48611), [GH-48623](https://github.com/godotengine/godot/pull/48623)).
- DynamicFont: Allow using WOFF fonts ([GH-52052](https://github.com/godotengine/godot/pull/52052)).
- GraphEdit: Make zoom limits and step adjustable ([GH-50526](https://github.com/godotengine/godot/pull/50526)).
- ScrollBar: Add `increment_pressed` and `decrement_pressed` icons ([GH-51805](https://github.com/godotengine/godot/pull/51805)).
- TextureButton: Add `flip_h` and `flip_v` properties ([GH-30424](https://github.com/godotengine/godot/pull/30424)).
- TextureProgress: Add offset for progress texture ([GH-38722](https://github.com/godotengine/godot/pull/38722)).
- Theme: Various improvements to the Theme API ([GH-49487](https://github.com/godotengine/godot/pull/49487)).
- Theme: Add support for partial custom editor themes ([GH-51648](https://github.com/godotengine/godot/pull/51648)).
- Theme: Add API to retrieve the default font, and optimize property change notification ([GH-53397](https://github.com/godotengine/godot/pull/53397)).

#### Import

- Backport improved glTF module with scene export support ([GH-49120](https://github.com/godotengine/godot/pull/49120)).
- Implement lossless WebP encoding ([GH-47854](https://github.com/godotengine/godot/pull/47854)).
- Add anisotropic filter option for `TextureArray`s ([GH-51402](https://github.com/godotengine/godot/pull/51402)).
- Add "Normal Map Invert Y" import option for normal maps ([GH-48693](https://github.com/godotengine/godot/pull/48693)).
- Add optional region cropping for `TextureAtlas` importer ([GH-52652](https://github.com/godotengine/godot/pull/52652)).

#### Input

- Add support for physical scancodes, fixes non-latin layout scancodes on Linux ([GH-46764](https://github.com/godotengine/godot/pull/46764)).
- Add `action_get_deadzone()` method to InputMap ([GH-50065](https://github.com/godotengine/godot/pull/50065)).
- Allow getting axis/vector values from multiple actions ([GH-50788](https://github.com/godotengine/godot/pull/50788)).
- Allow checking for exact matches with action events ([GH-50874](https://github.com/godotengine/godot/pull/50874)).
- Exposed setters for sensor values ([GH-53742](https://github.com/godotengine/godot/pull/53742)).
- Expose `Input::flush_buffered_events()` ([GH-53812](https://github.com/godotengine/godot/pull/53812)).
- Allow input echo when changing UI focus ([GH-44456](https://github.com/godotengine/godot/pull/44456)).

#### Localization

- Add support for translating the class reference ([GH-53511](https://github.com/godotengine/godot/pull/53511)).
  * Includes Chinese (Simplified) and Spanish translations with high completion ratio, and initial translations for French, Japanese, and German.
- Allow overriding `get_message` with virtual method ([GH-53207](https://github.com/godotengine/godot/pull/53207)).

#### Mono (C#)

- iOS: Cache AOT compiler output ([GH-51191](https://github.com/godotengine/godot/pull/51191)).
- Add editor keyboard shortcut (<kbd>Alt+B</kbd>) for Mono Build solution button ([GH-52595](https://github.com/godotengine/godot/pull/52595)).
- Add support to export enum strings for `Array<string>` ([GH-52763](https://github.com/godotengine/godot/pull/52763)).
- Support arrays of `NodePath` and `RID` ([GH-53577](https://github.com/godotengine/godot/pull/53577)).
- Support marshaling generic `Godot.Object` ([GH-53582](https://github.com/godotengine/godot/pull/53582)).

#### Networking

- Add support for multiple address resolution in DNS requests ([GH-49020](https://github.com/godotengine/godot/pull/49020)).
- Implement `String::parse_url()` for parsing URLs ([GH-48205](https://github.com/godotengine/godot/pull/48205)).
- Add `get_buffered_amount()` to `WebRTCDataChannel` ([GH-50659](https://github.com/godotengine/godot/pull/50659)).
- Add `dtls_hostname` property to ENet ([GH-51434](https://github.com/godotengine/godot/pull/51434)).

#### Physics

- Enable setting the number of physics solver iterations ([GH-38387](https://github.com/godotengine/godot/pull/38387), [GH-50257](https://github.com/godotengine/godot/pull/50257)).
- Heightmap collision shape support in Godot Physics 3D ([GH-47349](https://github.com/godotengine/godot/pull/47349)).
- Add support for Dynamic BVH as 2D physics broadphase ([GH-48314](https://github.com/godotengine/godot/pull/48314)).
- Expose `body_test_motion` in 3D physics server ([GH-50103](https://github.com/godotengine/godot/pull/50103)).
- Add option to sync motion to physics in 3D `KinematicBody` ([GH-49446](https://github.com/godotengine/godot/pull/49446)).
- Expose collider RID in 2D/3D kinematic collision ([GH-49476](https://github.com/godotengine/godot/pull/49476)).
- Support for disabling physics on `SoftBody` ([GH-49835](https://github.com/godotengine/godot/pull/49835)).
- Backport new methods for `KinematicBody` and `KinematicCollision` ([GH-52116](https://github.com/godotengine/godot/pull/52116)).
- Expose `SoftBody` pin methods for scripting ([GH-52369](https://github.com/godotengine/godot/pull/52369)).

#### Porting

- Android: Add partial support for Android scoped storage ([GH-50359](https://github.com/godotengine/godot/pull/50359)).
- Android: Add initial support for Play Asset Delivery ([GH-52526](https://github.com/godotengine/godot/pull/52526)).
- Android: Implement per-pixel transparency ([GH-51935](https://github.com/godotengine/godot/pull/51935)).
- Android: Add basic user data backup option ([GH-49070](https://github.com/godotengine/godot/pull/49070)).
- Android: Add support for prompting the user to retain app data on uninstall ([GH-51605](https://github.com/godotengine/godot/pull/51605)).
- HTML5: Export as Progressive Web App (PWA) ([GH-48250](https://github.com/godotengine/godot/pull/48250)).
- HTML5: Implement Godot <-> JavaScript interface ([GH-48691](https://github.com/godotengine/godot/pull/48691)).
- HTML5: Implement AudioWorklet without threads ([GH-52650](https://github.com/godotengine/godot/pull/52650)).
- HTML5: Implement video driver selection for Web editor ([GH-53991](https://github.com/godotengine/godot/pull/53991)).
- HTML5: Add easy to use download API ([GH-48929](https://github.com/godotengine/godot/pull/48929)).
- iOS: Add pen pressure support for Apple Pencil ([GH-47469](https://github.com/godotengine/godot/pull/47469)).
- iOS: Add option to automatically generate icons and launch screens ([GH-49464](https://github.com/godotengine/godot/pull/49464)).
- iOS: Support multiple `plist` types in plugin ([GH-49802](https://github.com/godotengine/godot/pull/49802)).
- iOS: Implement missing OS `set`/`get_clipboard()` methods ([GH-52540](https://github.com/godotengine/godot/pull/52540)).
- Linux: Add initial support for the RISC-V architecture ([GH-53509](https://github.com/godotengine/godot/pull/53509)).
- macOS: Add GDNative Framework support, and minimal support for handling Unix symlinks ([GH-46860](https://github.com/godotengine/godot/pull/46860)).
- macOS: Add notarization support when exporting for macOS on a macOS host ([GH-49276](https://github.com/godotengine/godot/pull/49276)).
- Add support for ARM64 architecture for the Embree raycaster (Apple M1, Linux aarch64) ([GH-48455](https://github.com/godotengine/godot/pull/48455)).
  * Note that the OIDN denoiser is still not available on this architecture.

#### Rendering

- GLES2: Add basic support for CPU blendshapes ([GH-48480](https://github.com/godotengine/godot/pull/48480), [GH-51363](https://github.com/godotengine/godot/pull/51363)).
- GLES2: Allow using clearcoat, anisotropy and refraction in SpatialMaterial ([GH-51967](https://github.com/godotengine/godot/pull/51967)).
- GLES2: Implement `Viewport.keep_3d_linear` for VR applications to convert output to linear color space ([GH-51780](https://github.com/godotengine/godot/pull/51780)).
- GLES3: Allow repeat flag in viewport textures ([GH-34008](https://github.com/godotengine/godot/pull/34008)).
- GLES3: Add support for contrast-adaptive sharpening in 3D ([GH-47416](https://github.com/godotengine/godot/pull/47416)).
- Add an editor setting to configure number of threads for lightmap baking ([GH-52952](https://github.com/godotengine/godot/pull/52952)).
- Add ring emitter for 3D particles ([GH-47801](https://github.com/godotengine/godot/pull/47801)).
- Add rooms and portals-based occlusion culling ([GH-46130](https://github.com/godotengine/godot/pull/46130)).
- Add a new high quality tonemapper: ACES Fitted ([GH-52477](https://github.com/godotengine/godot/pull/52477)).
- Add soft shadows to the CPU lightmapper ([GH-50184](https://github.com/godotengine/godot/pull/50184)).
- Add high quality glow mode ([GH-51491](https://github.com/godotengine/godot/pull/51491)).
- Add new 3D point light attenuation as an option ([GH-52918](https://github.com/godotengine/godot/pull/52918)).
- Import option to split vertex buffer stream in positions and attributes ([GH-46574](https://github.com/godotengine/godot/pull/46574)).
- Add horizon specular occlusion ([GH-51416](https://github.com/godotengine/godot/pull/51416)).

#### Shaders

- Add support for structs and fragment-to-light varyings ([GH-48075](https://github.com/godotengine/godot/pull/48075)).
- Add support for global const arrays ([GH-50889](https://github.com/godotengine/godot/pull/50889)).
- Make `TIME` available in custom functions by default ([GH-49509](https://github.com/godotengine/godot/pull/49509)).

#### VisualScript

- Allow dropping custom node scripts in VisualScript editor ([GH-50696](https://github.com/godotengine/godot/pull/50696)).
- Expose visual script custom node type hints ([GH-50705](https://github.com/godotengine/godot/pull/50705)).
- Improve and streamline `VisualScriptFuncNode`s `Call` `Set` `Get` ([GH-50709](https://github.com/godotengine/godot/pull/50709)).

#### XR

- Add `VIEW_INDEX` variable in shader to know which eye/view we're rendering for ([GH-48011](https://github.com/godotengine/godot/pull/48011)).

### Changed

#### 2D

- Make the most recently added current `Camera2D` take precedence ([GH-50112](https://github.com/godotengine/godot/pull/50112)).

#### 3D

- Implement individual mesh transform for `MeshLibrary` items ([GH-52298](https://github.com/godotengine/godot/pull/52298)).

#### Buildsystem

- Refactor module defines into a generated header ([GH-50466](https://github.com/godotengine/godot/pull/50466)).

#### Core

- Promote object validity checks to release builds ([GH-51796](https://github.com/godotengine/godot/pull/51796)).
- Add detailed error messages to release builds (used to be debug-only) ([GH-53405](https://github.com/godotengine/godot/pull/53405)).
- Add Node name to `print()` of all nodes, makes `Object::to_string()` virtual ([GH-38819](https://github.com/godotengine/godot/pull/38819)).
- Thread callbacks can now take optional parameters ([GH-38078](https://github.com/godotengine/godot/pull/38078), [GH-51093](https://github.com/godotengine/godot/pull/51093)).
- Harmonize output of `OS.get_locale()` between platforms ([GH-40708](https://github.com/godotengine/godot/pull/40708)).
- Optimize hash comparison for integer and string keys in Dictionary ([GH-53557](https://github.com/godotengine/godot/pull/53557)).
- Improve the console error logging appearance: ([GH-49577](https://github.com/godotengine/godot/pull/49577)).
- Implement missing stringification for `PoolByteArray` and `PoolColorArray` ([GH-53655](https://github.com/godotengine/godot/pull/53655)).
- `AStar.get_available_point_id()` returns 0 instead of 1 when empty ([GH-48958](https://github.com/godotengine/godot/pull/48958)).
- Compare connections by object ID, making `.tscn` order deterministic ([GH-52493](https://github.com/godotengine/godot/pull/52493)).

#### Editor

- Refactor `Theme` item management in the theme editor ([GH-49512](https://github.com/godotengine/godot/pull/49512)).
- Overhaul the theme editor and improve user experience ([GH-49774](https://github.com/godotengine/godot/pull/49774)).
- Improve 2D editor zoom logic ([GH-50490](https://github.com/godotengine/godot/pull/50490), [GH-50499](https://github.com/godotengine/godot/pull/50499)).
- Make several actions in the Inspector dock more obvious ([GH-50528](https://github.com/godotengine/godot/pull/50528)).
- Improve the editor feature profiles UX ([GH-49643](https://github.com/godotengine/godot/pull/49643)).
- Improve the UI/UX of the Export Template Manager dialog ([GH-50531](https://github.com/godotengine/godot/pull/50531)).
- Improve FileSystem dock sorting ([GH-50565](https://github.com/godotengine/godot/pull/50565)).
- Improve the 3D editor manipulation gizmo ([GH-50597](https://github.com/godotengine/godot/pull/50597)).
- Increase object snapping distances in the 3D editor ([GH-53727](https://github.com/godotengine/godot/pull/53727)).
- Refactor layer property editor grid ([GH-51040](https://github.com/godotengine/godot/pull/51040)).
- Improve the animation bezier editor ([GH-48572](https://github.com/godotengine/godot/pull/48572)).
- Fix logic for showing `TileMap` debug collision shapes ([GH-49075](https://github.com/godotengine/godot/pull/49075)).
- Add custom debug shape thickness and color options to `RayCast` ([GH-49726](https://github.com/godotengine/godot/pull/49726)).
- Handle portrait mode monitors in the automatic editor scale detection ([GH-48597](https://github.com/godotengine/godot/pull/48597)).
- Remove high radiance sizes from the editor due to issues on specific GPUs ([GH-48906](https://github.com/godotengine/godot/pull/48906)).

#### GUI

- GraphEdit: Enable zooming with Ctrl + Scroll wheel and related fixes to zoom handling ([GH-47173](https://github.com/godotengine/godot/pull/47173)).
- Button: Don't change hovering during focus events ([GH-47280](https://github.com/godotengine/godot/pull/47280)).
- DynamicFont: Re-add support for kerning ([GH-49377](https://github.com/godotengine/godot/pull/49377)).
- LineEdit: Double click selects words, triple click selects all the content ([GH-46527](https://github.com/godotengine/godot/pull/46527)).
- LinkButton: Button text is now automatically translated like other Controls ([GH-52138](https://github.com/godotengine/godot/pull/52138)).
- Theme: StyleBox fake anti-aliasing improvements ([GH-51589](https://github.com/godotengine/godot/pull/51589)).

#### Import

- Optimize image channel detection ([GH-47396](https://github.com/godotengine/godot/pull/47396)).

#### Localization

- Only include editor translations above a threshold to save on binary size ([GH-54020](https://github.com/godotengine/godot/pull/54020)).

#### Mono (C#)

- macOS: Mono builds are now universal builds with support for both `x86_64` and `arm64` architectures ([GH-49248](https://github.com/godotengine/godot/pull/49248)).
- Improve C# method listing ([GH-52607](https://github.com/godotengine/godot/pull/52607)).
- Avoid modifying `csproj` globbing includes ([GH-54262](https://github.com/godotengine/godot/pull/54262)).
- Deprecate `Xform` methods removed in 4.0, the `*` operator is preferred ([GH-52762](https://github.com/godotengine/godot/pull/52762)).

#### Networking

- Enable range coder compression by default in `NetworkedMultiplayerENet` ([GH-51525](https://github.com/godotengine/godot/pull/51525)).

#### Physics

- Port Bullet's convex hull computer to replace `QuickHull` ([GH-48533](https://github.com/godotengine/godot/pull/48533)).
- Return RID instead of Object ID in `area`-/`body_shape_entered`/-`exited` signals ([GH-42743](https://github.com/godotengine/godot/pull/42743)).
- Optimize area detection and `intersect_shape` queries with concave shapes ([GH-48551](https://github.com/godotengine/godot/pull/48551)).
- Optimize raycast with large Heightmap shape data ([GH-48709](https://github.com/godotengine/godot/pull/48709)).
- Reload kinematic shapes when changing `PhysicsBody` mode to Kinematic ([GH-53118](https://github.com/godotengine/godot/pull/53118)).
- Wake up 2D and 3D bodies in impulse and force functions ([GH-53113](https://github.com/godotengine/godot/pull/53113)).
- Compile Bullet with threadsafe switch on ([GH-53183](https://github.com/godotengine/godot/pull/53183)).

#### Porting

- Android: Target API level 30, raise min API level to 19 ([GH-50359](https://github.com/godotengine/godot/pull/50359)).
- Android: Improve input responsiveness on underpowered Android devices ([GH-42220](https://github.com/godotengine/godot/pull/42220)).
- Android: Upgrade Android Gradle to version 7.2, now requires Java 11 ([GH-53610](https://github.com/godotengine/godot/pull/53610)).
- HTML5: Debug HTTP server refactor with SSL support ([GH-48250](https://github.com/godotengine/godot/pull/48250)).
- HTML5: Use 64KiB chunk size in JS `HTTPClient`, for consistency with native platforms ([GH-48501](https://github.com/godotengine/godot/pull/48501)).
- HTML5: Raise default initial memory to 64 MiB ([GH-50422](https://github.com/godotengine/godot/pull/50422)).
- HTML5: Use browser mix rate by default on the Web ([GH-52723](https://github.com/godotengine/godot/pull/52723)).
- HTML5: Refactor event handlers, drop most Emscripten HTML5 dependencies ([GH-52812](https://github.com/godotengine/godot/pull/52812)).
- iOS: Disable half float on GLES2 via platform override ([GH-54229](https://github.com/godotengine/godot/pull/54229)).
- macOS: Prefer .app bundle icon over the default one ([GH-48686](https://github.com/godotengine/godot/pull/48686)).

#### Rendering

- `VisualServer` now sorts based on AABB position ([GH-43506](https://github.com/godotengine/godot/pull/43506)).
- Make Blinn and Phong specular consider albedo and specular amount ([GH-51410](https://github.com/godotengine/godot/pull/51410)).

#### Shaders

- Default shader specular render mode to `SCHLICK_GGX` ([GH-51401](https://github.com/godotengine/godot/pull/51401)).

#### Thirdparty

- Bullet uppdated to version 3.17.
- Embree updated to version 3.13.0.
- MbedTLS updated to version 2.16.11 (security update).
- NanoSVG updated to 2021-09-03 git snapshot (security update).
- CA root certificates updated to 2021-07-05 bundle from Mozilla.
- SDL GameControllerDB updated to 2021-10-29 git snapshot.

#### XR

- Only update render target directly if ARVR mode is off ([GH-54258](https://github.com/godotengine/godot/pull/54258)).

### Removed

#### Porting

- Android: Remove non-functional native video OS methods ([GH-48537](https://github.com/godotengine/godot/pull/48537)).
- iOS: Remove duplicate orientation setting in the export preset ([GH-48943](https://github.com/godotengine/godot/pull/48943)).

### Fixed

#### 2D

- TileSet: Fix selection of spaced atlas tile when using priority ([GH-50886](https://github.com/godotengine/godot/pull/50886)).

#### 3D

- CSGSphere: Fix UV mapping ([GH-49195](https://github.com/godotengine/godot/pull/49195)).
- CSGPolygon: Fix multiple issues ([GH-49314](https://github.com/godotengine/godot/pull/49314)).
- CSGPolygon: Fixes and features: Angle simplification, UV tiling distance, interval type ([GH-52509](https://github.com/godotengine/godot/pull/52509)).
- GridMap: Fix erasing octants in the wrong order ([GH-50052](https://github.com/godotengine/godot/pull/50052)).
- PathFollow: Fix forward calculation for the position at the end of the curve ([GH-50986](https://github.com/godotengine/godot/pull/50986)).
- SphereMesh: Fix the normals when the sphere/hemisphere is oblong ([GH-51995](https://github.com/godotengine/godot/pull/51995)).
- Update mesh AABB when software skinning is used ([GH-53144](https://github.com/godotengine/godot/pull/53144)).

#### Animation

- Fix Tween active state and repeat after `stop()` and then `start()` ([GH-47142](https://github.com/godotengine/godot/pull/47142)).
- Fix SkeletonIK root bones being twisted incorrectly when rotated ([GH-48251](https://github.com/godotengine/godot/pull/48251)).
- Fix issues with `BlendSpace2D` `BLEND_MODE_DISCRETE_CARRY` ([GH-48375](https://github.com/godotengine/godot/pull/48375)).
- Fixed issue where bones become detached if multiple SkeletonIK nodes are used ([GH-49031](https://github.com/godotengine/godot/pull/49031)).
- Fix non functional 3D onion skinning ([GH-52664](https://github.com/godotengine/godot/pull/52664)).
- Fix Animation Playback Track not seeking properly ([GH-38107](https://github.com/godotengine/godot/pull/38107)).
- Fix bugs in `AnimationNodeTransition`'s behavior ([GH-52543](https://github.com/godotengine/godot/pull/52543), [GH-52555](https://github.com/godotengine/godot/pull/52555)).
- Fix rendering centered odd-size texture for `AnimatedSprite`/`AnimatedSprite3D` ([GH-53052](https://github.com/godotengine/godot/pull/53052)).

#### Audio

- Fix cubic resampling algorithm ([GH-51082](https://github.com/godotengine/godot/pull/51082)).

#### Core

- Make all file access 64-bit (`uint64_t`) ([GH-47254](https://github.com/godotengine/godot/pull/47254)).
  * This adds support for handling files bigger than 2.1 GiB, including on 32-bit OSes.
- Fix negative delta arguments ([GH-52947](https://github.com/godotengine/godot/pull/52947)).
- Complain if casting a freed object in a debug session ([GH-51095](https://github.com/godotengine/godot/pull/51095)).
- Fix read/write issues with `NaN` and `INF` in VariantParser ([GH-47500](https://github.com/godotengine/godot/pull/47500)).
- Fix sub-resource storing the wrong index in cache ([GH-49625](https://github.com/godotengine/godot/pull/49625)).
- Save binary `ProjectSettings` key length properly ([GH-49649](https://github.com/godotengine/godot/pull/49649)).
- Fix ZIP files being opened with two file descriptors ([GH-42337](https://github.com/godotengine/godot/pull/42337)).
- Fix `Transform::xform(Plane)` functions to handle non-uniform scaling ([GH-50637](https://github.com/godotengine/godot/pull/50637)).
- Fix renaming directories with `Directory.rename()` ([GH-51793](https://github.com/godotengine/godot/pull/51793)).
- Fix path with multiple slashes not being corrected on templates ([GH-52513](https://github.com/godotengine/godot/pull/52513)).
- Fix `String.get_base_dir()` handling of Windows top-level directories ([GH-52744](https://github.com/godotengine/godot/pull/52744)).
- Fix potential crash when creating thread with an invalid target instance ([GH-53060](https://github.com/godotengine/godot/pull/53060)).
- Fix behavior of `CONNECT_REFERENCE_COUNTED` option for signal connections ([GH-47442](https://github.com/godotengine/godot/pull/47442)).
- Fix swapped axes in `OpenSimplexNoise.get_image()` ([GH-30424](https://github.com/godotengine/godot/pull/30424)).
  * Breaks compat. If you need to preserve the 3.2 behavior, swap your first and second arguments in `get_image()`.
- Fix loading packed scene with editable children at runtime ([GH-49664](https://github.com/godotengine/godot/pull/49664)).
- Quote and escape `ConfigFile` keys when necessary ([GH-52180](https://github.com/godotengine/godot/pull/52180)).
- Write node groups on a single line when saving a `.tscn` file ([GH-52284](https://github.com/godotengine/godot/pull/52284)).

#### Editor

- Rationalize property reversion ([GH-51166](https://github.com/godotengine/godot/pull/51166)).
- Fix Marshalls infinite recursion crash in debugger ([GH-51068](https://github.com/godotengine/godot/pull/51068)).
- Fix slow load/save of scenes with many instances of the same script ([GH-49570](https://github.com/godotengine/godot/pull/49570)).
- Properly update `NodePath`s in the editor in more cases when nodes are moved or renamed ([GH-49812](https://github.com/godotengine/godot/pull/49812)).
- Fix scale sensitivity for 3D objects ([GH-52665](https://github.com/godotengine/godot/pull/52665)).
- Fix preview grid in `SpriteFrames` editor's "Select Frames" dialog ([GH-52461](https://github.com/godotengine/godot/pull/52461)).
- Fix `MeshInstance2D` edit rect ([GH-54070](https://github.com/godotengine/godot/pull/54070)).

#### GDScript

- Ignore property groups and categories in GDScript code completion ([GH-54272](https://github.com/godotengine/godot/pull/54272)).
- Fix parsing multi-line `preload` statement ([GH-52521](https://github.com/godotengine/godot/pull/52521)).
- Speedup running very big GDScript files ([GH-53507](https://github.com/godotengine/godot/pull/53507)).
- LSP: Fix `SymbolKind` reporting wrong types and `get_node()` parsing ([GH-50914](https://github.com/godotengine/godot/pull/50914), [GH-51283](https://github.com/godotengine/godot/pull/51283)).
- LSP: Report `new()` as `_init` & fix docstrings on multiline functions ([GH-53094](https://github.com/godotengine/godot/pull/53094)).

#### GUI

- GraphNode: Properly handle children with "Expand" flag ([GH-39810](https://github.com/godotengine/godot/pull/39810)).
- Label: Fix valign with stylebox borders ([GH-50478](https://github.com/godotengine/godot/pull/50478)).
- RichTextLabel: Fix auto-wrapping on CJK texts ([GH-49280](https://github.com/godotengine/godot/pull/49280)).
- RichTextLabel: Fix character horizontal offset calculation ([GH-52752](https://github.com/godotengine/godot/pull/52752)).
- RichTextLabel: Honor content marging when drawing font shadow ([GH-54054](https://github.com/godotengine/godot/pull/54054)).
- RichTextLabel: Fix meta link detection when used inside a fill tag ([GH-54114](https://github.com/godotengine/godot/pull/54114)).
- TabContainer: Fix moving dropped tab to incorrect child index ([GH-51177](https://github.com/godotengine/godot/pull/51177)).
- Tabs: Fix invisible tabs not being ignored ([GH-53551](https://github.com/godotengine/godot/pull/53551)).
- TextureProgress: Improve behavior with nine patch ([GH-45815](https://github.com/godotengine/godot/pull/45815)).
- Theme: Fix potential crash with custom themes using BitMap fonts ([GH-53410](https://github.com/godotengine/godot/pull/53410)).

#### Import

- Fix loading RLE compressed TGA files ([GH-49603](https://github.com/godotengine/godot/pull/49603)).
- Fix issue in `TextureAtlas` import of images with wrong size ([GH-42103](https://github.com/godotengine/godot/pull/42103)).
- Fix potential crash importing invalid BMP files ([GH-46555](https://github.com/godotengine/godot/pull/46555)).
- glTF: Improved error handling around invalid images and invalid meshes ([GH-48904](https://github.com/godotengine/godot/pull/48904), [GH-48912](https://github.com/godotengine/godot/pull/48912)).
- glTF: Fix incorrect skin deduplication when using named binds ([GH-48913](https://github.com/godotengine/godot/pull/48913)).

#### Input

- Fix game controllers ignoring the last listed button ([GH-48934](https://github.com/godotengine/godot/pull/48934)).

#### Mono (C#)

- iOS: Fix `P/Invoke` symbols being stripped by the linker, resulting in `EntryPointNotFoundException` crash at runtime ([GH-49248](https://github.com/godotengine/godot/pull/49248)).
- macOS: Automatically enable JIT entitlements for the Mono exports ([GH-50317](https://github.com/godotengine/godot/pull/50317)).
- Fix reloading `tool` scripts in the editor ([GH-52883](https://github.com/godotengine/godot/pull/52883)).
- Fix C# bindings generator for default value types ([GH-49702](https://github.com/godotengine/godot/pull/49702)).
- Ignore paths with invalid chars in `PathWhich` ([GH-50918](https://github.com/godotengine/godot/pull/50918)).
- Fix `List<T>` marshalling ([GH-53628](https://github.com/godotengine/godot/pull/53628)).
- Fix `hint_string` for enum arrays ([GH-53638](https://github.com/godotengine/godot/pull/53638)).
- Keep order for C# exported members ([GH-54199](https://github.com/godotengine/godot/pull/54199)).

#### Networking

- Fix parsing some IPv6 URLs for WebSocket ([GH-48205](https://github.com/godotengine/godot/pull/48205)).
- `WebsocketPeer` outbound buffer fixes and buffer size query ([GH-51037](https://github.com/godotengine/godot/pull/51037)).
- Fix IP address resolution incorrectly locking the main thread ([GH-51199](https://github.com/godotengine/godot/pull/51199)).

#### Physics

- Fix 2D and 3D moving platform logic ([GH-50166](https://github.com/godotengine/godot/pull/50166), [GH-51458](https://github.com/godotengine/godot/pull/51458)).
- Various fixes to 2D and 3D `KinematicBody` `move_and_slide` and `move_and_collide` ([GH-50495](https://github.com/godotengine/godot/pull/50495)).
- Improved logic for `KinematicBody` collision recovery depth ([GH-53451](https://github.com/godotengine/godot/pull/53451)).
- Fix `RayShape` recovery in `test_body_ray_separation` ([GH-53453](https://github.com/godotengine/godot/pull/53453)).
- Apply infinite inertia checks to Godot Physics 3D ([GH-42637](https://github.com/godotengine/godot/pull/42637)).
- Fix and clean disabled shapes handling in Godot physics servers ([GH-49845](https://github.com/godotengine/godot/pull/49845)).
- Fix `KinematicBody` axis lock ([GH-45176](https://github.com/godotengine/godot/pull/45176)).
- Don't override `KinematicCollision` reference when still in use in script ([GH-52955](https://github.com/godotengine/godot/pull/52955)).
- Fix ragdoll simulation when parent was readded to scene ([GH-48823](https://github.com/godotengine/godot/pull/48823)).
- Ignore disabled shapes for mass property calculations ([GH-49699](https://github.com/godotengine/godot/pull/49699)).

#### Porting

- Android: Add GDNative libraries to Android custom Gradle builds ([GH-49912](https://github.com/godotengine/godot/pull/49912)).
- Android: Resolve issue where the Godot app remains stuck when resuming ([GH-51584](https://github.com/godotengine/godot/pull/51584)).
- Android: Properly validate `godot_project_name_string` for Android special chars ([GH-54255](https://github.com/godotengine/godot/pull/54255)).
- HTML5: Fix bug in AudioWorklet when reading output buffer ([GH-52696](https://github.com/godotengine/godot/pull/52696)).
- HTML5: Release pressed events when the window is blurred on HTML5 platform ([GH-52973](https://github.com/godotengine/godot/pull/52973)).
- Linux: Fix input events random delay on X11 ([GH-54313](https://github.com/godotengine/godot/pull/54313)).
- Linux: Fix implementation of `move_to_trash` ([GH-44021](https://github.com/godotengine/godot/pull/44021)).
- Linux: Fix crash when using ALSA MIDI with PulseAudio ([GH-48350](https://github.com/godotengine/godot/pull/48350)).
- Linux: Fix `Directory::get_space_left()` result ([GH-49222](https://github.com/godotengine/godot/pull/49222)).
- macOS: Allow "on top" windows to enter fullscreen mode ([GH-49017](https://github.com/godotengine/godot/pull/49017)).
- macOS: Fix editor window missing events on macOS Monterey ([GH-54474](https://github.com/godotengine/godot/pull/54474)).
- macOS: Fix custom mouse cursor not set after mouse mode change ([GH-49848](https://github.com/godotengine/godot/pull/49848)).
- macOS: Fix `Directory::get_space_left()` result ([GH-49222](https://github.com/godotengine/godot/pull/49222)).
- macOS: Fix Xbox controllers in Bluetooth mode on macOS ([GH-51117](https://github.com/godotengine/godot/pull/51117)).
- macOS: Fix incorrect mouse position in fullscreen ([GH-52374](https://github.com/godotengine/godot/pull/52374)).
- macOS: Add entitlements required by OIDN JIT for the editor build ([GH-54067](https://github.com/godotengine/godot/pull/54067)).
- Windows: Fix platform file access to allow file sharing with external programs ([GH-51430](https://github.com/godotengine/godot/pull/51430)).
- Windows: Fix code signing with `osslsigncode` from Linux/macOS ([GH-49985](https://github.com/godotengine/godot/pull/49985)).
- Windows: Send error logs to `stderr` instead of `stdout`, like done on other OSes ([GH-39139](https://github.com/godotengine/godot/pull/39139)).
- Windows: Fix `OS.shell_open()` not returning errors ([GH-52842](https://github.com/godotengine/godot/pull/52842)).
- Windows: Allow renaming to change the case of Windows directories ([GH-43068](https://github.com/godotengine/godot/pull/43068)).
- Windows: Disable WebM SIMD optimization with YASM which triggers crashes ([GH-53959](https://github.com/godotengine/godot/pull/53959)).

#### Rendering

- GLES2: Fix ambient light flickering with multiple refprobes ([GH-53740](https://github.com/godotengine/godot/pull/53740)).
- GLES3: Fix draw order of transparent materials with multiple directional lights ([GH-47129](https://github.com/godotengine/godot/pull/47129)).
- GLES3: Fix multimesh being colored by other nodes ([GH-47582](https://github.com/godotengine/godot/pull/47582)).
- GLES3: Only add emission on base pass ([GH-53938](https://github.com/godotengine/godot/pull/53938)).
- GLES3: Fudge irradiance map lookup to avoid precision issues ([GH-54197](https://github.com/godotengine/godot/pull/54197)).
- Fixed `rotate_y` property of particle shaders ([GH-46687](https://github.com/godotengine/godot/pull/46687)).
- Fixed behavior of velocity spread in particles ([GH-47310](https://github.com/godotengine/godot/pull/47310)).
- Fixes depth sorting of meshes with transparent textures ([GH-50721](https://github.com/godotengine/godot/pull/50721)).
- Fix flipped binormal in `SpatialMaterial` triplanar mapping ([GH-49950](https://github.com/godotengine/godot/pull/49950)).
- Fix `CanvasItem` bounding rect calculation in some cases ([GH-49160](https://github.com/godotengine/godot/pull/49160)).
- Clamp negative colors regardless of the tonemapper to avoid artifacts ([GH-51439](https://github.com/godotengine/godot/pull/51439)).
- Fix Y billboard shear when rotating camera ([GH-52151](https://github.com/godotengine/godot/pull/52151)).
- Add half frame to `floor()` for animated particles UV to compensate precision errors ([GH-53233](https://github.com/godotengine/godot/pull/53233)).
- Prevent shaders from generating code before the constructor finishes ([GH-52475](https://github.com/godotengine/godot/pull/52475)).

## [3.3] - 2021-04-21

See the [release announcement](https://godotengine.org/article/godot-3-3-has-arrived) for details.

### Added

#### Audio

- [MP3 loading and playback support](https://github.com/godotengine/godot/pull/43007).
- [Add AudioEffectCapture to access the microphone in real-time](https://github.com/godotengine/godot/pull/45593).

#### Build system

- [Add `production=yes` option to set optimal options for production builds](https://github.com/godotengine/godot/pull/45593).
  - Users making custom builds should use this option which is equivalent to `use_lto=yes debug_symbols=no use_static_cpp=yes`.
  - **Note for Linux builds:** `use_static_cpp=yes` and `udev=yes` are now the default values, so you need `libudev` and `libstdc++-static` development packages to build in optimal conditions.
- [Add `optimize=none` to disable C/C++ compiler optimizations for release builds](https://github.com/godotengine/godot/pull/46966).
  - This can be used to speed up compile times when working on the engine itself and when debug checks/features aren't desired.

#### Core

- [New dynamic BVH for rendering and the GodotPhysics backends](https://github.com/godotengine/godot/pull/44901).
  - If you experience a regression in either physics or rendering, you can try [these Project Settings](https://github.com/godotengine/godot/pull/44901#issuecomment-758618531) to revert back to the previous Octree-based approach and possibly fix the issue. In either case, be sure to report the problem on GitHub.
- [Ability to restore RandomNumberGenerator state](https://github.com/godotengine/godot/pull/45019).
- [TileMap `show_collision` property to show/hide collision shapes in the editor and at run-time](https://github.com/godotengine/godot/pull/46623).
- [`Array.append_array()` method to append an array at the end of another array](https://github.com/godotengine/godot/pull/43398).
- [`OS.get_thread_caller_id()` method to print the ID of the thread the code is currently running on](https://github.com/godotengine/godot/pull/44732).
- [`Image.load_bmp_from_buffer()` method to load BMP images at run-time](https://github.com/godotengine/godot/pull/42947).
- [`Image.resize_to_po2()` now accepts an optional "interpolation" parameter, defaulting to bilinear filtering](https://github.com/godotengine/godot/pull/44460).
  - Nearest-neighbor filtering can be used for pixel art textures, and will be used automatically when resizing non-power-of-two textures for non-filtered textures in GLES2.
- [`OS.set_environment()` method to set environment variables](https://github.com/godotengine/godot/pull/46413).
- [`String.to_wchar()` method to get a PoolByteArray corresponding to a string's `wchar` data](https://github.com/godotengine/godot/pull/46343).
- [`String.naturalnocasecmp_to()` method to perform *natural* case-insensitive string comparison for sorting purposes](https://github.com/godotengine/godot/pull/45957).
- [`application/run/flush_stdout_on_print` project setting to enable per-line flushing of the standard output stream in release builds](https://github.com/godotengine/godot/pull/44393).

#### Editor

- [Infinite 3D editor grid](https://github.com/godotengine/godot/pull/43206) ([further improvements](https://github.com/godotengine/godot/pull/45594)).
- [New 3D rotation gizmo](https://github.com/godotengine/godot/pull/43016).
- [Support for copy-pasting nodes](https://github.com/godotengine/godot/pull/34892).
- [Detect external modification of scenes and prompt for reloading](https://github.com/godotengine/godot/pull/31747).
- [New editor to configure default import presets](https://github.com/godotengine/godot/pull/46354).
- [The 3D viewport's View Information pane now displays the viewport resolution](https://github.com/godotengine/godot/pull/45596).
- [`EditorInterface.get_editor_scale()` method to retrieve the editor scale factor](https://github.com/godotengine/godot/pull/47622).
  - This can be used for hiDPI support in editor plugins.
- [The `EditorInterface.edit_node()` method is now exposed](https://github.com/godotengine/godot/pull/47709).
- **HTML5:** [New web editor](https://godotengine.org/article/godot-web-progress-report-3), available as a [Progressive Web App](https://github.com/godotengine/godot/pull/46796).

#### GUI

- [Minimap support in GraphEdit](https://github.com/godotengine/godot/pull/43416).
- [New AspectRatioContainer Control node](https://github.com/godotengine/godot/pull/45129).
- [TextEdit's bookmark, breakpoint and "mark safe line" functions are now exposed to scripting](https://github.com/godotengine/godot/pull/40629).

#### Import

- [Rewritten and greatly improved FBX importer](https://godotengine.org/article/fbx-importer-rewritten-for-godot-3-2-4).
- ["Keep" mode to keep files as-is and export them](https://github.com/godotengine/godot/pull/47268).

#### Input

- [Support for buttons and D-pads mapped to half axes](https://github.com/godotengine/godot/pull/42800).
- [Support for new SDL game controller keywords (used by PS5 controller support)](https://github.com/godotengine/godot/pull/45798).
- [Add driving joystick type to Windows joystick handling](https://github.com/godotengine/godot/pull/44082).
- [Mouse event pass-through support for the window](https://github.com/godotengine/godot/pull/40205).

#### Mono (C#)

- [Reworked solution build output panel](https://github.com/godotengine/godot/pull/42547).

#### Physics

- [Support for pause-aware picking](https://github.com/godotengine/godot/pull/39421).
  - This breaks compatibility, but is not enabled by default for existing projects. See the project setting `physics/common/enable_pause_aware_picking`. It will be enabled by default for new projects created with 3.3.
- [CollisionObject can now display collision shape meshes](https://github.com/godotengine/godot/pull/45783).
- **Bullet:** [CollisionPolygon `margin` property](https://github.com/godotengine/godot/pull/45855).
- **GodotPhysics:** [Cylinder collision shape support](https://github.com/godotengine/godot/pull/45854).
  - This is experimental in 3.3, and will likely receive fixes in future Godot releases.

#### Porting

- **Android:** [Support for exporting as Android App Bundle](https://github.com/godotengine/godot-proposals/issues/342).
- **Android:** [Support for subview embedding](https://github.com/godotengine/godot-proposals/issues/1064).
- **Android:** [Support for notch cutouts on Android 9.0 and later](https://github.com/godotengine/godot/pull/43104).
- **Android:** [Support for mouse events](https://github.com/godotengine/godot/pull/42360).
- **Android:** [Support for keyboard modifiers and arrow keys](https://github.com/godotengine/godot/pull/40398).
- **Android:** [Implement `OS.get_screen_orientation()`](https://github.com/godotengine/godot/pull/43022).
- **HTML5:** [AudioWorklet support for multithreaded HTML5 builds](https://github.com/godotengine/godot/pull/43454).
- **HTML5:** [Optional GDNative support](https://github.com/godotengine/godot/pull/44076).
- **HTML5:** [Resizable canvas option to disable viewport resizing](https://github.com/godotengine/godot/pull/42266).
- **HTML5:** [Optional automatic WebGL 2.0 -> 1.0 fallback when WebGL 2.0 support is not available](https://github.com/godotengine/godot/pull/47659).
  - To benefit from this feature, enable the **Rendering > Quality > Driver > Fallback To Gles2** project setting as you would on other platforms.
- **HTML5:** [loDPI fallback support for improved performance on hiDPI displays (at the cost of visuals)](https://github.com/godotengine/godot/pull/46802).
  - To keep the previous behavior, **Allow Hidpi** must be enabled in the Project Settings before exporting the project.
- **iOS:** [Plugin support](https://github.com/godotengine/godot/pull/41340), with a similar interface to Android plugins.
- **iOS:** [Add a touch delay project setting](https://github.com/godotengine/godot/pull/42457).
- **iOS:** [Implemented native loading screen](https://github.com/godotengine/godot/pull/45693).
- **Linux:** [PulseAudio and ALSA libraries are now dynamically loaded](https://github.com/godotengine/godot/pull/46107), [as well as libudev](https://github.com/godotengine/godot/pull/46117).
  - This allows running official Godot binaries on systems that don't have PulseAudio installed.
- **Linux/macOS:** [Implement the `--no-window` command line argument for parity with Windows](https://github.com/godotengine/godot/pull/42276).
- **macOS:** [macOS ARM64 support](https://github.com/godotengine/godot/pull/39788) in official binaries for Apple M1 chip (only standard build for now).

#### Rendering

- [2D batching for GLES3](https://github.com/godotengine/godot/pull/42119) (it was implemented for GLES2 in 3.2.2), and improvements to GLES2's batching.
- [New *experimental* buffer orphan/stream project settings to improve 2D performance on specific platforms](https://github.com/godotengine/godot/pull/47864).
- [New software skinning for MeshInstance](https://github.com/godotengine/godot/pull/40313) to replace the slow GPU skinning on devices that don't support the fast GPU skinning (especially mobile).
- [Configurable amount of lights per object](https://github.com/godotengine/godot/pull/43606), now defaulting to 32 instead of 8.
- [New CPU lightmapper](https://github.com/godotengine/godot/pull/44628).
  - Uses a raytracing approach with optional denoising.
  - Environment lighting is now supported.
  - Lightmaps are now stored in an atlas when baking if GLES3 is the current backend at the time of baking.
  - Bicubic lightmap sampling is now used to improve the final appearance, both in GLES3 and GLES2. It's enabled by default on desktop platforms.
- [Anisotropic filtering now works when using the GLES2 backend](https://github.com/godotengine/godot/pull/45654).
- [FXAA property in Viewport and associated project setting for GLES3 and GLES2](https://github.com/godotengine/godot/pull/42006).
  - Upsides: Faster than MSAA, smooths out alpha-tested materials and specular aliasing.
  - Downsides: Lower quality than MSAA and makes the 3D viewport blurrier.
- [Debanding property in Viewport and associated project setting for GLES3](https://github.com/godotengine/godot/pull/42942).
  - Significantly reduces visible banding in 3D. The effect is mainly visible on smooth gradients, solid surfaces and in heavy fog.
  - Only effective when HDR is enabled in the Project Settings (which is the default).
- [New `METALLIC` built-in for the `light()` function in the shader language](https://github.com/godotengine/godot/pull/42548).
  - This is also exposed in the visual shader editor.
- [Add setting for shadow cubemap max size](https://github.com/godotengine/godot/pull/48059).

#### XR (Augmented Reality / Virtual Reality)

- [Add a `set_interface()` method](https://github.com/godotengine/godot/pull/46781).
- [Expose the depth buffer to GDNative](https://github.com/godotengine/godot/pull/46781).
- [Allow supplying a depth buffer from an ARVR plugin](https://github.com/godotengine/godot/pull/46781).
- **HTML5:** [WebXR support](https://github.com/godotengine/godot/pull/42397) for VR games.

### Changed

#### Core

- [Deleted object access now raises an error instead of a warning](https://github.com/godotengine/godot/pull/48041).
- [Improved error messages when passing nonexistent node paths to `get_node()`](https://github.com/godotengine/godot/pull/46243).
- [Optimized transform propagation for hidden 3D nodes](https://github.com/godotengine/godot/pull/45583).
- [Modernized multi-threading APIs](https://github.com/godotengine/godot/pull/45618).
  - This might cause regressions in projects that use multiple threads. Please report such issues on GitHub.
- [Suggestions are now printed when attempting to use a nonexistent input action name](https://github.com/godotengine/godot/pull/45902).
- [SVG images can now be used as a project icon](https://github.com/godotengine/godot/pull/43369).
- [Tweaked log file names for consistency between Mono and non-Mono builds](https://github.com/godotengine/godot/pull/44148).
- [Tweaked command line `--print-fps` display to display milliseconds per frame timings in addition to FPS](https://github.com/godotengine/godot/pull/47735).
- [OpenSimplexNoise is now guaranteed to give consistent results across platforms](https://github.com/godotengine/godot/issues/47211).
  - This change breaks compatibility: you get different results even for the same seed.

#### Editor

- [Improved inspector subresource editing visibility](https://github.com/godotengine/godot/pull/45907).
- [Improved the 3D selection box appearance for better visibility](https://github.com/godotengine/godot/pull/43424).
  - The 3D selection box color can be changed in the Editor Settings.
- [Increased default opacity for 3D manipulator gizmos for better visibility](https://github.com/godotengine/godot/pull/44384).
- [Improved visibility for the dashed line in the TextureRegion editor](https://github.com/godotengine/godot/pull/45164).
- [Pressed CheckButtons are now colored in blue for easier recognition](https://github.com/godotengine/godot/pull/44556).
- [The autokeying icon in the animation editor is now red when pressed to emphasize its "recording" status](https://github.com/godotengine/godot/pull/42417).
- [Scroll bars are now thicker and have a subtle background to better hint where they start and stop](https://github.com/godotengine/godot/pull/47633).
- [Undo/redo log messages now give more context when performing actions in the 2D editor](https://github.com/godotengine/godot/pull/42229).
- [The editor now uses 75% scaling by default on small displays (such as 1366×768)](https://github.com/godotengine/godot/pull/43611).
  - This can be reverted by setting the editor scale to 100% in the Editor Settings.
- [The editor now uses 150% scaling by default on 4K monitors, regardless of their DPI](https://github.com/godotengine/godot/pull/45910).
  - This can be adjusted by setting the editor scale to the desired value in the Editor Settings.
- [Rename Node is now bound to <kbd>F2</kbd>](https://github.com/godotengine/godot/pull/38201).
  - To account for this change, switching between editors using keyboard shortcuts now requires holding down <kbd>Ctrl</kbd>.
  - Editor shortcuts can be changed back to the previous values in the Editor Settings' Shortcuts tab.
- [Changed the Search Help shortcut from <kbd>Shift + F1</kbd> to <kbd>F1</kbd>](https://github.com/godotengine/godot/pull/43773).
- [Changed the FileSystem dock Copy Path shortcut from <kbd>Ctrl + C</kbd> to <kbd>Ctrl + Shift + C</kbd>](https://github.com/godotengine/godot/pull/43397).
- [Changed 3D editor's Primary Grid Steps setting from 10 to 8](https://github.com/godotengine/godot/pull/43754).
  - This uses a power-of-two value by default.
- [Increased the default `profiler_frame_max_functions` to 512](https://github.com/godotengine/godot/pull/43697).
  - This fixes many instances where functions didn't appear in the script profiler.
- [The inspector now allows using a comma as a decimal separator](https://github.com/godotengine/godot/pull/42376).
- [Editor plugins are now searched for recursively](https://github.com/godotengine/godot/pull/43734).
- [Increased the page size for array/dictionary editors in the inspector from 10 to 20](https://github.com/godotengine/godot/pull/44864).
  - The value can now be increased further in the Editor Settings at the cost of slower node switching times in the inspector.
- [The "Auto" editor setting hints for editor scale and font hinting now display the value they apply](https://github.com/godotengine/godot/pull/45270).
- [Tweaked property hints for SpatialMaterial depth properties to allow greater control and quality](https://github.com/godotengine/godot/pull/44130).
- [Pause Mode and Script are no longer collapsed in categories in the inspector](https://github.com/godotengine/godot/pull/43566).
- **Asset library:** [The Retry button now only appears if the download has failed](https://github.com/godotengine/godot/pull/46105).
- **Asset library:** [Search now starts automatically after entering text](https://github.com/godotengine/godot/pull/42402).
- **Asset library:** [Moved the Asset Library API URLs to the Editor Settings](https://github.com/godotengine/godot/pull/45202).
- **Project manager:** [Drag-and-dropping a ZIP archive to the project manager window will now prompt for importing it](https://github.com/godotengine/godot/pull/45706).
- **Project manager:** [Display loading text while the project is opening](https://github.com/godotengine/godot/pull/46026).
- **Project manager:** [The Open Project Folder button is now more visible](https://github.com/godotengine/godot/pull/45642).

#### GUI

- [Range now returns a ratio of 1.0 if the minimum and maximum value are equal](https://github.com/godotengine/godot/pull/45220).
- [ColorPicker now keeps the hue value when the saturation or value is set to zero](https://github.com/godotengine/godot/pull/46230).
- [The Control virtual method `_make_custom_tooltip()` should now return a `Control` instead of an `Object`](https://github.com/godotengine/godot/pull/43280).
  - Scripts overriding this method will have to be updated.

#### Import

- [Allow a greater range of characters in glTF bone names](https://github.com/godotengine/godot/pull/47074).
- [glTF import now uses vertex colors by default](https://github.com/godotengine/godot/pull/41007).
- [Fix ETC compressor `lossy_quality` handling](https://github.com/godotengine/godot/pull/44682).
  - High `lossy_quality` values will now incur significantly longer compression times, but will also improve the resulting texture quality.

#### Mono (C#)

- [The `copy_mono_root` SCons build option now defaults to `yes`](https://github.com/godotengine/godot/pull/42332).
- Official builds now use Mono 6.12.0.122.

#### Networking

- [Increase the default HTTPClient download chunk size to 64 KiB](https://github.com/godotengine/godot/pull/42896).
  - This improves download speeds significantly, including for the in-editor export template downloader.
  - This change also affects HTTPRequest.

#### Rendering

- [Shadows now have a proper soft appearance in GLES2 when using the PCF13 shadow filter](https://github.com/godotengine/godot/pull/46301).
- [The Ensure Correct Normals render mode and associated SpatialMaterial property are now supported in GLES2](https://github.com/godotengine/godot/pull/47540).
- [Real-time lights no longer affect objects that receive baked lighting if Bake Mode is set to All](https://github.com/godotengine/godot/pull/41629).
- [YSort rendering order is now more deterministic](https://github.com/godotengine/godot/pull/42375).

#### Physics

- [2D collision shapes are now displayed with outlines when **Debug > Visible Collision Shapes** is enabled](https://github.com/godotengine/godot/pull/46291).

#### Porting

- **Android:** [Update logic to sign prebuilt APKs with `apksigner` instead of `jarsigner`, as required for Android API 30](https://github.com/godotengine/godot/pull/44645).
- **Android:** [Disable the `requestLegacyExternalStorage` attribute when there are no storage permissions](https://github.com/godotengine/godot/pull/47954).
- **HTML5:** [Improved gamepad support by using an internal implementation of the Gamepad API](https://github.com/godotengine/godot/pull/45078).
- **HTML5** [Changed HTML shell templates](https://github.com/godotengine/godot/pull/46201). [See updated documentation](https://docs.godotengine.org/en/3.3/tutorials/platform/customizing_html5_shell.html).
- **Linux:** Binaries are now stripped of string and symbol tables, reducing their size significantly.
  - Editor: 9 MB less (standard) and 35 MB less (Mono).
  - Templates: 5-6 MB less (standard) and 30 MB less (Mono).
- **macOS:** [Add entitlements configuration and export template `.dylib` signing to the export](https://github.com/godotengine/godot/pull/46618).
- **macOS:** [Code signing on export is now enabled by default. If no identity is specified, an *ad hoc* certificate is generated and used automatically](https://github.com/godotengine/godot/pull/46618).
  - This is done because applications must be signed to be run on macOS Big Sur, even for private use.
  - Self-signed certificates work for private use, but they will be considered untrusted on other people's computers.
- **macOS:** [Add a Replace Existing Signature export option to fix signing on export with broken OSXCross ad-hoc linker signatures](https://github.com/godotengine/godot/pull/47141).
  - This is enabled by default.
- **macOS:** [Improve Mono distribution in .app bundle to allow codesigning exported projects](https://github.com/godotengine/godot/pull/43768).
- **macOS:** Binaries are now stripped of string and symbol tables, which reduces their size significantly:
  - Editor: 14 MB less (standard) and 9 MB less (Mono).
  - Templates: 9-10 MB less (standard) and 6 MB less (Mono).
- **macOS:** Official editor binaries are now signed and notarized.

### Removed

#### Import

- [Removed the now-redundant ResourceImporterCSV importer](https://github.com/godotengine/godot/pull/47301).
  - This is superseded by the new ["keep" import mode](https://github.com/godotengine/godot/pull/47268) feature.

### Fixed

#### Audio

- [Use higher-quality resampling for Ogg Vorbis and MP3 sounds](https://github.com/godotengine/godot/pull/46086).
  - This fixes bitrate-like artifacts when playing low-frequency sounds.
- [Fix pops when using `play()` in AudioStreamPlayer2D and AudioStreamPlayer3D](https://github.com/godotengine/godot/pull/46151).

#### Core

- [Fix Editable Children issues with node renaming, moving, duplicating and instancing](https://github.com/godotengine/godot/pull/39533).
- [Freed Objects no longer delay to `null` in debug builds](https://github.com/godotengine/godot/pull/41866).
- [Various fixes to Variant and Reference pointers](https://github.com/godotengine/godot/pull/43049).
- [Optimized octree and fixed a leak](https://github.com/godotengine/godot/pull/41123).
- [Fix crash when opening a ZIP data pack](https://github.com/godotengine/godot/pull/42196).
- [`OS.delay_msec()` and `OS.delay_usec()` with a negative value no longer freeze the engine and return an error message instead](https://github.com/godotengine/godot/pull/46194).
- Various fixes to CSG mesh generation. There should be less instances of holes being created in CSG meshes.

#### Editor

- [Fix exporting if the temporary export directory doesn't exist yet](https://github.com/godotengine/godot/pull/45561).
- [Objects can no longer be added to the Project Settings since they can't be serialized correctly](https://github.com/godotengine/godot/pull/42616).
- [Fix hidden nodes being selectable in the 2D editor in specific cases](https://github.com/godotengine/godot/pull/46261).
- [Implementing clearing of diagnostics in the GDScript language server, fixing issues with errors lingering after files were deleted](https://github.com/godotengine/godot/pull/47553).
- [The Export PCK/ZIP action now obeys the export path configured in the export preset as expected](https://github.com/godotengine/godot/pull/45359).

#### GDScript

- [Fix leaks due to cyclic references](https://github.com/godotengine/godot/pull/41931).

#### GUI

- [Fix artifacts in DynamicFont when scaling with filtering enabled](https://github.com/godotengine/godot/pull/43171).
- [DynamicFonts are now loaded to memory on all platforms to avoid locking files](https://github.com/godotengine/godot/pull/44117).
- [Fix fallback emoji font color](https://github.com/godotengine/godot/pull/44212).
- [ColorPicker button text and tooltips now appear as expected in exported projects](https://github.com/godotengine/godot/pull/47547).
- [Fix RichTextLabel losing its `text` due to being replaced with empty BBCode when custom effects are defined](https://github.com/godotengine/godot/pull/47486).
- [Fix incorrect editor background color when using a transparent editor theme color](https://github.com/godotengine/godot/pull/36004).

#### Import

- [Relax node](https://github.com/godotengine/godot/pull/45545) and [bone naming constraints in glTF](https://github.com/godotengine/godot/pull/47074).
  - To preserve compatibility with models imported in 3.2, [a `use_legacy_names` import setting was added](https://github.com/godotengine/godot/pull/48058).
- [Fix parsing Base64-encoded buffer and image data in glTF](https://github.com/godotengine/godot/pull/42501).
- [Fix handling of normalized accessor property in glTF](https://github.com/godotengine/godot/pull/44746).

#### Mono (C#)

- [Fix targeting .NETFramework with .NET 5](https://github.com/godotengine/godot/pull/44135).
- [Fix System.Collections.Generic.List marshalling](https://github.com/godotengine/godot/pull/45029).
- [Fix support for Unicode identifiers](https://github.com/godotengine/godot/pull/45310).
- [Fixes to Mono on WebAssembly](https://github.com/godotengine/godot/pull/44374).

#### Network

- [Fix UDP ports being silently reused without an error on Linux in PacketPeerUDP](https://github.com/godotengine/godot/pull/43918).

#### Physics

- [Multiple fixes to one-way collisions](https://github.com/godotengine/godot/pull/42574).
- [Fix `test_body_motion` recovery and rest info](https://github.com/godotengine/godot/pull/46148).
- **GodotPhysics:** [Fix incorrect moment of inertia calculations for built-in 3D collision shapes](https://github.com/godotengine/godot/pull/47284).
- [Many physics fixes for both 2D and 3D](https://github.com/godotengine/godot/pulls?q=is%3Apr+milestone%3A3.3+label%3Atopic%3Aphysics+is%3Amerged).

#### Porting

- **Android:** [Fix splash screen loading](https://github.com/godotengine/godot/pull/42389).
- **iOS:** [Fix multiple issues with PVRTC import, disable ETC1](https://github.com/godotengine/godot/pull/38076).
- **iOS:** [Fixes to keyboard input, including better IME support](https://github.com/godotengine/godot/pull/43560).
- **Linux:** [Fix keyboard input lag and clipboard delay issues](https://github.com/godotengine/godot/pull/42341).
- **Linux:** [Fix audio corruption when using the ALSA driver](https://github.com/godotengine/godot/pull/43928).
- **Linux:** [Fix PRIME hybrid graphics detection on Steam](https://github.com/godotengine/godot/pull/46792).
- **macOS:** [Fix mouse position in captured mode](https://github.com/godotengine/godot/pull/42328).
- **macOS:** [Improve `get_screen_dpi()` reliability for non-integer scaling factors](https://github.com/godotengine/godot/pull/42478).
- **Windows:** [Fix debugger not getting focused on break](https://github.com/godotengine/godot/pull/40555).

#### Rendering

- [Various fixes to 3D light culling](https://github.com/godotengine/godot/pull/46694).
  - DirectionalLight's Cull Mask property is now effective.
- [Fix large Sprite3D UV wobbling with low-resolution textures](https://github.com/godotengine/godot/pull/42537).
- [Fix impact of `lifetime_randomness` on properties using a curve](https://github.com/godotengine/godot/pull/45496).
- [Fix 2D normal maps when using batching + NVIDIA workaround](https://github.com/godotengine/godot/pull/41323).
- [Fix PanoramaSky artifacts on Android in GLES2](https://github.com/godotengine/godot/pull/44489).
- [Fix glow on devices with only 8 texture slots in GLES2](https://github.com/godotengine/godot/pull/42446).
- [Use a separate texture unit for `light_texture` in GLES2](https://github.com/godotengine/godot/pull/42538).
- [Fix reflection probes in WebGL 1.0 (GLES2 on HTML5)](https://github.com/godotengine/godot/pull/45465).
- [Fix screen-space reflections tracing the environment in GLES3](https://github.com/godotengine/godot/pull/38954).
- [Fade screen-space reflections towards the inner margin in GLES3](https://github.com/godotengine/godot/pull/41892).
- [Ensure Reinhard tonemapping values are positive in GLES3](https://github.com/godotengine/godot/pull/42056).

## [3.2.3] - 2020-09-17

See the [release announcement](https://godotengine.org/article/maintenance-release-godot-3-2-3) for details.

### Added

- Android: Add option to enable high precision float in GLES2
- C#: Add Visual Studio support
- HTML5: Improvements and bugfixes backported from the `master` branch
  - Note: This PR adds threads support, but as this support is still [disabled in many browsers](https://caniuse.com/#feat=sharedarraybuffer) due to security concerns, the option is not enabled by default. Build HTML5 templates with `threads_enabled=yes` to test it.
- Input: Support SDL2 half axes and inverted axes mappings
- iOS: Add support of iOS's dynamic libraries to GDNative
- iOS: Add methods to embed a framework
- LineEdit: Add option to disable virtual keyboard for LineEdit
- macOS: Implement confined mouse mode
- macOS: Implement seamless display scaling
- Rendering: Allow nearest neighbor lookup when using mipmaps

### Changed

- C#: New `csproj` style with backport of Godot.NET.Sdk
  - This change breaks forward compatibility, C# projects opened in 3.2.3 will no longer work with 3.2.2 or earlier. Backup your project files before upgrading.
- GDScript: Auto completion enhanced for extends and class level identifier
- HTML5: Implement HTML5 cancel/ok button swap on Windows
- Physics: Better damping implementation for Bullet rigid bodies
  - This makes the behavior of the GodotPhysics and Bullet backends consistent, and more user-friendly with Bullet. If you're using damping with the Bullet backend, you may need to adjust some properties to restore the behavior from 3.2.2 or earlier (see [GH-42051](https://github.com/godotengine/godot/issues/42051#issuecomment-692132877)).
- Project Settings: Enable file logging by default on desktops to help with troubleshooting
- Script editor: Don't open dominant script in external editor
- Sprite3D: Use mesh instead of immediate for drawing Sprite3D

### Fixed

- Android: Fix Return key events in LineEdit & TextEdit on Android
- C#: Fix crash when pass null in print array in `GD.Print`
- C#: Fix restore not called when building game projects
- C#: Fix potential crash with nested classes
- C#: Fix endless reload loop if project has unicode chars
- Core: Fix debugger error when Dictionary key is a freed Object
- Core: Fix leaked ObjectRCs on object Variant reassignment
- GLES2: Fixed mesh data access errors in GLES2
- GLES2: Batching - Fix `FORCE_REPEAT` not being set properly on npot hardware
- GLES3: Force depth prepass when using alpha prepass
- GLES3: Fix OpenGL error when generating radiance
- HTML5: More fixes, audio fallback, fixed FPS
- IK: Fixed SkeletonIK not working with scaled skeletons
- Import: Fix custom tracks causing issues on reimport
- Import: Fix upstream stb_vorbis regression causing crashes with some OGG files
- iOS: Fix for iOS touch recognition
- iOS: Fix possible crash on exit when leaking translation remappings
- macOS: Add support for the Apple Silicon (ARM64) build target
  - ARM64 binaries are not included in macOS editor or template builds yet. It's going to take some time before our [dependencies and toolchains](https://github.com/godotengine/godot-build-scripts/pull/10) are updated to support it.
- macOS: Set correct external file attributes, and creation time
- macOS: Refocus last key window after `OS::alert` is closed
- macOS: Fix crash of failed `fork`
- Networking: Fix `UDPServer` and `DTLSServer` on Windows compatibility
- Particles: Fix 2D Particle velocity with directed emission mask
- PathFollow3D: Fix repeated updates of PathFollow3D Transform
- Physics: Trigger broadphase update when changing collision layer/mask
- Physics: Fix laxist collision detection on one way shapes
- Physics: Properly pass safe margin on initialization (fixes jitter in GodotPhysics backend)
- Project Settings: Fix overriding compression related settings
- Rendering: Fixed images in black margins
- Rendering: Properly calculate Polygon2D AABB with skeleton
- RichTextLabel: Fix RichTextLabel fill alignment regression
- RichTextLabel: Fix `center` alignment bug
- Shaders: Fix specular `render_mode` for Visual Shaders
- SkeletonIK: Fix calling `reload_goal()` when starting IK with `start(true)`
- TileSet: Fix potential crash when editing polygons
- Tree: Fix crash when hovering columns after removing a column
- Windows: DirectInput: Use correct joypad ID
- Thirdparty library updates: mbedtls 2.16.8, stb_vorbis 1.20, wslay 1.1.1

## [3.2.2] - 2020-06-26

See the [release announcement](https://godotengine.org/article/maintenance-release-godot-3-2-2) for details.

### Added

- 2D: Expose the `cell_size` affecting `VisibilityNotifier2D` precision
- 2D: Add `MODULATE` builtin to canvas item shaders
- Android: Add signal support to Godot Android plugins
- AStar: Implements `estimate_cost`/`compute_cost` for AStar2D
- C#: Add iOS support
- C#: Allow debugging exported games
- Debug: Add a suffix to the window title when running from a debug build
- Editor: Add rotation widget to 3D viewport
- Editor: Add editor freelook navigation scheme settings
- Editor: Allow duplicating files when holding Control
- GLES2: Add 2D batch rendering across items
- GLES3: Add Nvidia `draw_rect` flickering workaround
- GLES2/GLES3: Add support for OpenGL external textures
- Input: Add keyboard layout enumeration / set / get functions
- macOS: Enable signing of DMG and ZIP'ed exports
- Networking: DTLS support + optional ENet encryption
- Object: Add `has_signal` method
- RichTextLabel: Add option to fit height to contents
- Shaders: Add shader time scaling
- Windows: Add tablet driver selection (WinTab, Windows Ink)

### Changed

- Android: Re-architecture of the plugin system
- Android: The `GodotPayments` plugin was moved to an external first-party plugin using the Google Play Billing library
- Core: Ensure COWData does not always reallocate on resize
- Core: Better handling of `Variant`s pointing to released `Object`s
- Editor: Account for file deletion and renaming in Export Presets
- Editor: Improved go-to definition (Ctrl + Click) in script editor
- Files: Improve UX of drive letters
- HTML5: Switch key detection from `keyCode` to `code`
- HTML5: Use 2-phase setup in JavaScript
- Import: Add support for glTF lights
- Input: Fix joypad GUID conversion to match new SDL format on OSX and Windows
- Language Server: Switch the GDScript LSP from WebSocket to TCP, compatible with more external editors
- Main: Improve the low processor mode sleep precision
- Physics: Normalize up direction vector in `move_and_slide()`
- UWP: Renamed the "Windows Universal" export preset to "UWP", to avoid confusion
- Windows: Make stack size on Windows match Linux and macOS

### Fixed

- Android: Fix `LineEdit` virtual keyboard issues
- AStar: Make `get_closest_point()` deterministic for equidistant points
- Audio: Fix volume interpolation in positional audio nodes
- C#: Sync csproj when files are changed from the FileSystem dock
- C#: Replace uses of old Configuration and update old csprojs
- C#: Revert marshalling of IDictionary/IEnumerable implementing types
- C#: Fix inherited scene not inheriting parent's exported properties
- C#: Fix exported values not updated in the remote inspector
- Core: Fixed false positives in the culling system
- Core: Fix leaks and crashes in `OAHashMap`
- CSG: Various bug fixes
- GDNative: Fix Variant size on 32-bit platforms
- GDScript: Fix leaked objects when game ends with yields in progress
- GDScript: Fix object leaks caused by unfulfilled yields
- GDScript: Various bugs fixed in the parser
- GLES2: Avoid unnecessary material rebind when using skeleton
- GLES2/GLES3: Reset texture flags after radiance map generation
- HTML5: Implement audio buffer size calculation, should fix iOS Safari audio issues
- Image: Fixing wrong blending rect methods
- Image: Fix upscaling image with bilinear interpolation option specified
- Import: Fix changing the import type of multiple files at once
- Import: Respect 'mesh compression' editor import option in Assimp and glTF importers
- Input: Various fixes for touch pen input
- macOS: Ignore process serial number argument passed by macOS Gatekeeper
- macOS: Fix exports losing executable permission when unzipped
- Particles: Fix uninitialized memory in CPUParticles and CPUParticles2D
- Physics: Make soft body completely stiff to attachment point
- Physics: Test collision mask before creating constraint pair in Godot physics broadphase 2D and 3D
- RegEx: Enable Unicode support for RegEx class
- RichTextLabel: Fix alignment bug with `[center]` and `[right]` tags
- Skeleton: Fix IK rotation issue
- VR: Fix aspect ratio on HMD projection matrix
- Windows: Fix certain characters being recognized as special keys when using the US international layout
- Windows: Fix quoting arguments with special characters in `OS.execute()`
- Windows: Do not probe joypads if `DirectInput` cannot be initializer
- Windows: Fix overflow condition with QueryPerformanceCounter

## [3.2.1] - 2020-03-10

See the [release announcement](https://godotengine.org/article/maintenance-release-godot-3-2-1) for details.

### Added

- Skin: Add support for named binds

### Changed

- TileSet: Hide TileSet properties from Inspector, fixing OOM crash on huge tilesets

### Fixed

- Android: Fix double tap pressed event regression
- Android: Fix LineEdit virtual keyboard inputs
- Bullet: Fix detection of concave shape in Area
- Camera2D: Fix inverted use of Camera2D `offset_v`
- Debugger: Fix crash inspecting freed objects
- Expression: Fix parsing integers as 32-bit
- HTML5: Fix `EMWSClient::get_connection_status()`
- HTML5: Fix touch events support with Emscripten 1.39.5+
- macOS: Fix gamepad disconnection callback on macOS Catalina
- Particles: Fix undefined behavior with atan in GPU Particles
- Video: Workaround WebM playback bug after AudioServer latency fixes
- Windows: Fix UPNP regression after upstream update
- Windows: Disable NetSocket address reuse

## [3.2] - 2020-01-29

### Added

- Support for [pseudo-3D depth in 2D](https://godotengine.org/article/godot-32-will-get-pseudo-3d-support-2d-engine).
- Support for importing 3D scenes using Assimp.
  - Many formats are supported, including FBX.
- [Support for generating audio procedurally and analyzing audio spectrums.](https://godotengine.org/article/godot-32-will-get-new-audio-features)
- WebRTC support.
  - Includes support for the high-level multiplayer API.
  - Supports NAT traversal using STUN or TURN.
- Support for automatically building Android templates before exporting.
  - This makes 3rd-party SDK integration easier.
- Support for [texture atlases in 2D](https://godotengine.org/article/atlas-support-returns-godot-3-2).
- Major improvements to the visual shader system. ([News post 1](https://godotengine.org/article/major-update-for-visual-shader-in-godot-3-2), [News post 2](https://godotengine.org/article/major-update-visual-shaders-godot-3-2-part-2))
  - Redesigned visual shader editor with drag-and-drop capability.
    - Textures can be dragged from the FileSystem dock to be added as nodes.
  - Most functions available in GLSL are now exposed.
  - Many constants such as `Pi` or `Tau` can now be used directly.
  - Support for boolean uniforms and sampler inputs.
  - New Sampler port type.
  - New conditional nodes.
  - New Expression node, allowing shader code to be written in visual shaders.
  - Support for plugins (custom nodes).
    - Custom nodes can be drag-and-dropped from the FileSystem dock.
  - Ability to copy and paste nodes.
  - Ability to delete multiple nodes at once by pressing <kbd>Delete</kbd>.
  - The node creation menu is now displayed when dragging a connection to an empty space on the graph.
  - GLES3-only functions are now distinguished from others in the creation dialog.
  - Ability to preview the code generated by the visual shader.
  - Ability to convert visual shaders to text-based shaders.
  - See the [complete list of new functions](https://github.com/godotengine/godot/pull/26164).
- Improved visual scripting.
  - Visual scripting now uses an unified graph where all functions are represented.
  - Nodes can now be edited directly in the graph.
  - Support for fuzzy searching.
  - The `tool` mode can now be enabled in visual scripts.
  - New Deconstruct node to deconstruct a complex value into a scalar value.
  - Miscellaneous UI improvements.
- Support for enabling/disabling parts of the editor or specific nodes.
  - This is helpful for education, or when working with artists to help prevent inadvertent changes.
- Language server for GDScript.
  - This can be used to get better integration with external editors.
- Version control integration in the editor.
  - This integration is VCS-agnostic (GDNative plugins provide specific VCS support).
- Improved GridMap editor.
  - The copied mesh is now displayed during pasting.
  - The duplication/paste indicator is now rotated correctly around the pivot point.
  - Ability to cancel paste and selection by pressing <kbd>Escape</kbd>.
  - Erasing is now done using <kbd>RMB</kbd> instead of <kbd>Shift + RMB</kbd>.
    - Freelook can still be accessed by pressing <kbd>Shift + F</kbd>.
- Improved MeshLibrary generation.
  - When appending to an existing MeshLibrary, previews are now only generated for newly-added or modified meshes.
  - Tweaked the previews' camera angle and light directions for better results.
  - Materials assigned to the MeshInstance instead of the Mesh are now exported to the MeshLibrary.
    - This is useful when exporting meshes from an imported scene (such as glTF), as it allows materials to persist across re-imports.
- [Improved Control anchor and margin workflow.](https://github.com/godotengine/godot/pull/27559)
- [Network profiler.](https://github.com/godotengine/godot/pull/31870)
- Improved NavigationMesh generation.
  - GridMaps can now be used to bake navigation meshes.
  - EditorNavigationMeshGenerator can now be used in `tool` scripts.
  - Support for generating navigation meshes from static colliders.
  - When using static colliders as a geometry source, a layer mask can be specified to ignore certain colliders.
  - The generator no longer relies on the global transform, making it possible to generate navmeshes on nodes that are not in the scene tree.
  - Navigation gizmos are now updated after every new bake.
- Support for skinning in 3D skeletons.
- CameraServer singleton to retrieve images from mobile cameras or webcams as textures.
- A crosshair is now displayed when using freelook in the 3D editor.
- Project camera override button at the top of the 2D and 3D editors.
  - When enabled, the editor viewport's camera will be replicated in the running project.
- RichTextLabel can now be extended with real-time effects and custom BBCodes.
  - Effects are implemented using the ItemFX resource.
- `[img=<width>x<height>]` tag to resize an image displayed in a RichTextLabel.
  - If `<width>` or `<height>` is 0, the image will be adjusted to keep its original aspect.
- Revamped node connection dialog for improved ease of use.
- The Signals dock now displays a signal's description in a tooltip when hovering it.
- Input actions can now be reordered by dragging them.
- Animation frames can now be reordered by dragging them.
- Ruler tool to measure distances and angles in the 2D editor.
- "Clear Guides" menu option in the 2D editor to remove all guides.
- The 2D editor grid now displays a "primary" line every 8 lines for easier measurements.
  - This value can be adjusted in the Configure Snap dialog.
- Projects can now have a description set in the Project Settings.
  - This description is displayed as a tooltip when hovering the project in the Project Manager.
- All Variant types can now be added as project settings using the editor (instead of just `bool`, `int`, `float` and `String`).
- Pressing <kbd>Ctrl + F</kbd> now focuses the search field in the Project Settings and Editor Settings.
- Quick Open dialog (<kbd>Shift + Alt + O</kbd>) to open any resource in the project.
  - Unlike the existing dialogs, it's not limited to scenes or scripts.
- Ability to convert a Sprite to a Mesh2D, Polygon2D, CollisionPolygon2D or LightOccluder2D.
- MultiMeshInstance2D node for using MultiMesh in 2D.
- PointMesh primitive.
  - Drawn as a rectangle with a constant size on screen, which is cheaper compared to using triangle-based billboards.
- 2D polygon boolean operations and Delaunay triangulation are now available in the Geometry singleton.
- [New convex decomposition](https://godotengine.org/article/godot-3-2-adds-support-convex-decomposition) using the [V-HACD](https://github.com/kmammou/v-hacd) library.
  - Can decompose meshes into multiple convex shapes for increased accuracy.
- Support for grouping nodes in the 3D editor.
- "Slow" modifier in freelook (accessed by holding <kbd>Alt</kbd>).
- The 2D editor panning limits can now be disabled in the Editor Settings.
- "Undo Close Tab" option in the scene tabs context menu.
- The editor is now capped to 20 FPS when the window is unfocused.
  - This decreases CPU/GPU usage if something causes the editor to redraw continuously (such as particles).
- The editor's FPS cap can now be adjusted in the Editor Settings (both when focused and unfocused).
- Version information is now displayed at the bottom of the editor.
  - This is intended to make the Godot version easily visible in video tutorials.
- Support for constants in the shader language.
- Support for local and varying arrays in the shader language.
- Support for `switch` statements in the shader language.
- Support for `do {...} while (...)` loops in the shader language.
  - Unlike `while`, the expression in the `do` block will always be run at least once.
- Support for hexadecimal number literals in the shader language.
- Ported several GLES3 shader functions such as `round()` to GLES2.
- `SHADOW_VEC` shader parameter to alter 2D shadow computations in custom shaders.
- Filter search box in the remote scene tree dock.
- Ability to expand/collapse nodes recursively in the scene tree dock by holding <kbd>Shift</kbd> and clicking on a folding arrow.
- Support for depth of field, glow and BCS in the GLES2 renderer.
- MSAA support in the GLES2 renderer.
- Ability to render viewports directly to the screen in the GLES2 renderer.
  - This can be faster on low-end devices, but it comes at a convenience cost.
- Project settings to set the maximum number of lights and reflections in the GLES3 renderer.
  - Decreasing these values can lead to faster shader compilations, resulting in lower loading times.
- Heightmap collision shape for efficient terrain collisions.
- AStar2D class, making A* use easier in 2D.
- Disabled collision shapes can now be added directly, without having to disable them manually after one step.
- Context menu options to close other scene tabs, scene tabs to the right, or all scene tabs.
- The audio bus volumes can now be snapped by holding <kbd>Ctrl</kbd> while dragging the slider.
- Hovering an audio bus' volume slider now displays its volume in a tooltip.
- Values in the Gradient and Curve editors can now be snapped by holding <kbd>Ctrl</kbd>.
  - Precise snapping can be obtained by holding <kbd>Shift</kbd> as well.
- Support for snapping when scaling nodes in the 2D editor.
- Precise snapping in the 3D editor when holding <kbd>Shift</kbd>.
- "Align Rotation with View" in the 3D editor.
  - Unlike "Align Transform with View", only the selected node's rotation will be modified.
  - "Align Selection with View" has been renamed to "Align Transform with View".
- All 3D gizmos now make use of snapping if enabled.
- CSG shapes are now highlighted with a translucent overlay when selected.
  - Shapes in Union mode will use a blue overlay color by default.
  - Shapes in Subtraction mode will use an orange overlay color by default.
  - Shapes in Intersection mode will use a white overlay color.
- Ability to move a vertex along a single axis when holding <kbd>Shift</kbd> in polygon editors.
- Support for binary literals in GDScript (e.g. `0b101010` for `42`).
- AutoLoads can now be used as a type in GDScript.
- Ability to define script templates on a per-project basis.
  - Template files should be placed into a `script_templates/` directory in the project and have an extension that matches the language (`.gd` for GDScript, `.cs` for C#).
  - The path to the script templates directory can be changed in the Project Settings.
- Ability to limit the minimum and maximum window size using `OS.set_min_window_size()` and `OS.set_max_window_size()`.
- `Node.process_priority` property to set or get a node's processing priority.
  - This was previously only available as `Node.set_process_priority()` (without an associated getter).
- `Node.editor_description` property for documentation purposes.
  - When hovering a node with a description in the scene tree dock, the description will be displayed in a tooltip.
- `Button.keep_pressed_outside` property to keep a button pressed when moving the pointer outside while pressed.
- `Button.expand_icon` property to make a button's icon expand/shrink with the button's size.
- `Popup.set_as_minsize()` method to shrink a popup to its minimum size.
- `Tree.get_icon_modulate()` and `Tree.set_icon_modulate()` methods to change an icon's color in a Tree.
- `Tree.call_recursive()` method to call a method on a TreeItem and its children recursively.
- `Light.use_gi_probe` property to exclude specific lights from GIProbe computations.
- TranslationServer method `get_loaded_locales()` to retrieve the list of languages with a translation loaded.
- `FRUSTUM` 3D camera mode to create tilted frustums for mirror or portal effects.
- `CanvasItem.draw_rect()` now has `width` and `antialiased` properties to match `draw_line()`'s functionality.
- `Engine.get_idle_frames()` and `Engine.get_physics_frames()` to get the number of idle and physics frame iterations since the project started.
  - Unlike `Engine.get_frames_drawn()`, `Engine.get_idle_frames()` will be incremented even if the render loop is disabled.
- `Engine.get_physics_interpolation_fraction()` to get the fraction through the current physics tick at the time of the current frame.
  - This can be used to implement fixed timestep interpolation.
- Support for shadow-to-opacity in 3D to render shadows in augmented reality contexts.
- Ability to change a Position2D gizmo's size.
- New Vector2 and Vector3 methods:
  - `move_toward()` to retrieve a vector moved towards another by a specified number of units.
  - `direction_to()` to retrieve a normalized vector pointing from a vector to another.
    - This is a shorter alternative to `(b - a).normalized()`.
- AStar functions `set_point_disabled()` and `is_point_disabled()` to selectively disable points.
- Tween now emits a `tween_all_completed` signal when all tweens are completed.
- `Input.get_current_cursor_shape()` to retrieve the current cursor shape.
- `InputEventAction` now has a `strength` property to simulate analog inputs.
- `String.repeat()` method to repeat a string several times and return it.
- `String.count()` method to count the number of occurrences of a substring in a string.
- `String.humanize_size()` method to display a file size as an human-readable string.
- `String.strip_escapes()` to strip non-printable escape characters from a string, including tabulations and newlines (but not spaces).
- `String.sha1_text()` and `String.sha1_buffer()` methods to return a string's SHA-1 hash.
- Line2D `clear_points()` method to clear all points.
- Line2D now has a "Width Curve" property to make its width vary at different points.
- `assert()` now accepts an optional second parameter to display a custom message when the assertion fails.
- `posmod()` built-in GDScript function that behaves like `fposmod()`, but returns an integer value.
- `smoothstep()` built-in GDScript function for smooth easing of values.
- `lerp_angle()` built-in GDScript function to interpolate between two angles.
- `ord()` built-in GDScript function to return the Unicode code point of a 1-character string.
- `PoolByteArray.hex_encode()` method to get a string of hexadecimal numbers.
- `Font.get_wordwrap_string_size()` method to return the rectangle size needed to draw a word-wrapped text.
- `Camera.get_camera_rid()` method to retrieve a Camera's RID.
- `Array.slice()` method to duplicate a subset of an Array and return it.
- The GraphEdit box selection colors can now be changed by tweaking the `selection_fill` and `selection_stroke` theme items.
- Toggleable HSV mode for ColorPicker.
- ColorPicker properties to toggle the visibility and editability of presets.
- The default ColorPicker mode (RGB, HSV, RAW) can now be changed in the Editor Settings.
- ColorPicker now displays an indicator to denote "overbright" colors (which can't be displayed as-is in the preview).
- Hovering a Color property in the editor inspector now displays a tooltip with the exact values.
- `Color.transparent` constant (equivalent to `Color(1, 1, 1, 0)`).
- `KinematicBody.get_floor_normal()` and `KinematicBody2D.get_floor_normal()` to retrieve the collided floor's normal.
- `VehicleWheel.get_rpm()` method to retrieve a vehicle wheel's rotations per minute.
- Per-wheel throttle, brake and steering in VehicleBody.
- `GeometryInstance.set_custom_aabb()` to set a custom bounding box (used for view frustum culling).
- `FuncRef.call_funcv()` to call a FuncRef with an array containing arguments.
  - In contrast to `FuncRef.call_func()`, only a single array argument is expected.
- `Mesh.get_aabb()` is now exposed to scripting.
- `PhysicalBone.apply_impulse()` and `PhysicalBone.apply_central_impulse()` methods to push ragdolls around.
- `ProjectSettings.load_resource_pack()` now features an optional `replace_files` argument (defaulting to `true`), which controls whether the loaded resource pack can override existing files in the virtual filesystem.
- `SpinBox.apply()` method to evaluate and apply the expression in the SpinBox's value immediately.
- `ConfigFile.erase_section_key()` method to remove a single key from a ConfigFile.
- `OS.execute()` now returns the process' exit code when blocking mode is enabled.
- `OS.is_window_focused()` method that returns `true` if the window is currently focused.
  - Tracking the focus state manually using `NOTIFICATION_WM_FOCUS_IN` and `NOTIFICATION_WM_FOCUS_OUT` is no longer needed to achieve this.
- `OS.low_processor_mode_sleep_usec` is now exposed as a property.
  - This makes it possible to change its value at runtime, rather than just defining it once in the Project Settings.
- `SceneTree.quit()` now accepts an optional argument with an exit code.
  - If set to a value greater than or equal to 0, it will override the `OS.exit_code` property.
- `VisualServer.get_video_adapter_name()` and `VisualServer.get_video_adapter_vendor()` methods to retrieve the user's graphics card model and vendor.
- `VisualServer.multimesh_create()` is now exposed to scripting.
- Ability to override how scripted objects are converted to strings by defining a `_to_string()` method.
- Export hints for 2D and 3D physics/render layers.
- Editor plugins can now add new tabs to the Project Settings.
- Standalone ternary expression warning in GDScript.
- Variable shadowing warning in GDScript.
  - Will be displayed if:
    - a block variable shadows a member variable,
    - a subclass variable shadows a member variable,
    - a function argument shadows a member variable.
- Script reflection methods are now exposed to GDScript.
  - See `Script.get_script_property_list()`, `Script.get_script_method_list()`, `Script.get_script_signal_list()`, `Script.get_script_constant_map()` and `Script.get_property_default_value()`.
- `randfn(mean, deviation)` method to generate random numbers following a normal Gaussian distribution.
- Ability to read the standard error stream when using `OS.execute()` (disabled by default).
- Option to disable boot splash filtering (nearest-neighbor interpolation).
- The GridMap editor now offers a search field and size slider.
- DynamicFont resources now have a thumbnail in the editor.
- Minimap in the script editor.
- Bookmarks in the script editor for easier code navigation.
- Filter search box for the script list and member list.
- Singletons and `class_name`-declared classes are now highlighted with a separate color in the script editor.
- The editor help now displays class properties' default and overridden values.
- The script editor's Find in Files dialog can now search in user-defined file types (`editor/search_in_file_extensions` in the Project Settings).
- The script editor search now displays the number of matches.
- The script editor search now selects the current match for easier replacing.
- "Evaluate Expression" contextual option in the script editor.
  - This option evaluates the selected expression and replaces it (e.g. `2 + 2` becomes `4`).
- Autocompletion support for `change_scene()`.
- Ability to skip breakpoints while debugging.
- Drag-and-drop support in the TileSet editor.
- Ability to attach scripts to nodes by dragging a name from the script list to a node in the scene tree.
- Icons are now displayed next to code completion items, making their type easier to distinguish.
- TileMap property `centered_textures` can be used to center textures on their tile, instead of using the tile's top-left corner as position for the texture.
- "Ignore" flag to ignore specific tiles when autotiling in the TileMap editor.
- Keyboard shortcuts to rotate tiles in the TileMap editor.
  - Default shortcuts are <kbd>A</kbd> (rotate left), <kbd>S</kbd> (rotate right), <kbd>X</kbd> (flip horizontally), <kbd>Y</kbd> (flip vertically).
- Ability to keep a node's local transform when reparenting it by holding <kbd>Shift</kbd>.
- Basis constants `IDENTITY`, `FLIP_X`, `FLIP_Y`, `FLIP_Z`.
- Ability to create sprite frames in AnimatedSprite from a sprite sheet.
- `frame_coords` property in Sprite and Sprite3D to set/get the coordinates of the frame to display from the sprite sheet.
- `billboard` property in Sprite3D.
- Reimplemented support for editing multiple keys at once in the animation editor.
- Support for FPS snapping in the Animation editor.
- Autokeying in the Animation editor.
  - Keyframes will be created automatically when translating, rotating or scaling nodes if a track exists already.
  - Keys must be inserted manually for the first time.
- AnimationNodeBlendTreeEditor improvements.
  - Ability to exclude multiple selected nodes at once.
  - Context menu to add new nodes (activated by right-clicking).
- The AnimationPlayer Call Method mode is now configurable.
  - Method calls can be "deferred" or "immediate", "deferred" being the default.
- OccluderPolygon2D is now draggable in the editor.
- The tooltip position offset is now configurable.
- The default cursor used when hovering RichTextLabels can now be changed.
- "Dialog Autowrap" property in AcceptDialog to wrap the label's text automatically.
- The 2D editor's panning shortcut can now be changed.
- The shortcuts to quit the editor can now be changed.
- Support for emission masks in CPUParticles2D.
- `direction` property in CPUParticles and ParticlesMaterial.
- `lifetime_randomness` property in CPUParticles and ParticlesMaterial.
- CPUParticles now uses a different gizmo icon to distinguish them from Particles.
- "Restart" button to restart particle emission in the editor.
- AnimatedSprites' animations can now be played backwards.
- TextureRects can now have their texture flipped horizontally or vertically.
- StyleBoxFlat shadows can now have an offset.
- StyleBoxFlat now computes UV coordinates for its `canvas_item` vertices, which can be used in custom shaders.
- Profiler data can now be exported to a CSV file.
- The 2D polygon editor now displays vertex numbers when hovering vertices.
- RectangleShapes now have a third handle to drag both axes at once.
- Global class resources are now displayed in the Resource property inspector.
- Double-clicking an easing property in the inspector will now make the editor display a numeric field.
  - This makes it easier to enter precise values for properties such as light attenuation.
- `interface/editor/default_float_step` editor setting to configure floating-point values' default step in the Inspector.
- Audio buses are now stylized to look like boxes that can be dragged.
- The default audio bus layout file path can now be changed in the Project Settings.
- The LineEdit and TextEdit controls now display their contextual menu when pressing the <kbd>Menu</kbd> key.
- `shortcut_keys_enabled` and `selecting_enabled` LineEdit and TextEdit properties to disable keyboard shortcuts and selecting text.
- The LineEdit "disabled" font color can now be changed.
- The TextEdit "readonly" font color can now be changed.
- LineEdit can now have its `right_icon` set in scripts.
- The `nine_patch_stretch` TextureProgress property now enables stretching when using a radial fill mode.
- Support for loading and saving encrypted files in ConfigFile.
- `get_path()` and `get_path_absolute()` are now implemented in FileAccessEncrypted.
- "Disabled" attenuation model for AudioStreamPlayer3D, making the sound not fade with distance while keeping it positional.
- AudioEffectPitchShift's FFT size and oversampling are now adjustable.
- TextEdit's tab drawing and folding is now exposed to GDScript.
- Orphan node monitor in the Performance singleton.
  - Counts the number of nodes that were created but aren't instanced in the scene tree.
- Ability to change eye height in VR.
- CSV files can now be imported as non-translation files.
- Scene resources such as materials can now be imported as `.tres` files.
- Support for importing 1-bit, 4-bit and 8-bit BMP files.
  - Size dimensions must be a multiple of 8 for 1-bit images and 2 for 4-bit images.
- `use_lld=yes` flag to link with [LLD](https://lld.llvm.org/) on Linux when compiling with Clang.
  - This results in faster iteration times when developing Godot itself or modules.
- `use_thinlto=yes` flag to link with [ThinLTO](https://clang.llvm.org/docs/ThinLTO.html) when using Clang.
- Multicast support in PacketPeerUDP.
- `NetworkedMultiplayerEnet.server_relay` property to disable server relaying.
  - This can be used to increase security when building a fully-authoritative server.
- Automatic timeout for TCP connections (defaults to 30 seconds, can be changed in the Project Settings).
- `HTTPRequest.timeout` property (defaults to 0, which is disabled).
- `HTTPRequest.download_chunk_size` property.
  - This value can be adjusted to reduce the allocation overhead and file writes when downloading large files.
  - The default value was increased for faster downloads (4 KB → 64 KB).
- WebSocket improvements.
  - Support for SSL in WebSocketServer.
  - WebSocketClient can now use custom SSL certificates (except on HTML5).
  - WebSocketClient can now define custom headers.
- The editor now features a built-in Web server for testing HTML5 projects.
- Button to remove all missing projects in the Project Manager.
- Reimplemented support for embedding project data in the PCK file.
- Ability to take editor screenshots by pressing <kbd>Ctrl + F12</kbd>.
- Editor plugins can now set the current active editor as well as toggle the distraction-free mode.
- **Android:** [Support for adaptive icons.](https://docs.godotengine.org/en/latest/getting_started/workflow/export/exporting_for_android.html#providing-launcher-icons)
  - All icon densities are now generated automatically by the exporter.
  - Only 3 images now need to be supplied to support all icon formats and densities (legacy icon, adaptive foreground, adaptive background).
- **Android:** Support for the Oculus Mobile SDK.
- **Android:** Support for requesting permissions at runtime.
- **Android:** `NOTIFICATION_APP_PAUSED` and `NOTIFICATION_APP_RESUMED` notifications are now emitted when the app is paused and resumed.
- **Android:** Support for pen input devices.
- **Android/iOS:** Support for vibrating the device.
- [**HTML5:** Partial clipboard support.](https://github.com/godotengine/godot/pull/29298)
- **iOS:** Support for [ARKit](https://developer.apple.com/augmented-reality/).
- **iOS:** `OS.get_model_name()` now returns a value with the device name.
- **iOS:** The Home indicator is now hidden by default to avoid being in the way of the running project.
  - It can be restored in the Project Settings.
- **Windows:** Ability to toggle the console window in the Editor Settings.
- **Windows:** Project setting to enable Vsync using the compositor (DWM), disabled by default.
  - On some hardware, this may fix stuttering issues when running a project in windowed mode.
- **Windows:** Support for code signing using `signtool` on Windows and `osslsigncode` on other platforms.
- **Windows:** Support for using Clang and ThinLTO when compiling using MinGW.
- **Windows/macOS:** `OS.set_native_icon()` method to set an `.ico` or `.icns` window/taskbar icon at runtime.
- **Windows/macOS/X11:** Support for graphic tablet pen pressure and tilt in InputEventMouseMotion.
- **macOS:** LineEdit now supports keyboard shortcuts commonly available on macOS.
- **macOS:** Multiple instances of the editor can now be opened at once.
- **macOS:** Recent and favorite projects are now listed in the project manager dock menu.
- **macOS:** The list of open scenes is now displayed in the editor dock menu.
- **macOS:** Support for modifying global and dock menus.
- **macOS:** Improved support for code signing when exporting projects.
- **macOS:** Support for defining camera and microphone usage descriptions when exporting a project.
- **macOS/X11:** [A zsh completion file for the editor is now available.](https://github.com/godotengine/godot/blob/master/misc/dist/shell/_godot.zsh-completion)
- **X11:** The instance PID is now set as the `_NET_WM_PID` window attribute, so that external programs can easily access it.
- **Mono:** Support for exporting to Android and HTML5.
- **Mono:** Support for using Rider as an external editor.
- **Mono:** Support for attaching external profilers like dotTrace using the `MONO_ENV_OPTIONS` environment variable.
- **Mono:** New DynamicGodotObject class to access dynamic properties from scripts written in GDScript.
- **Mono:** Support for resource type hints in exported arrays.
- **Mono:** New `mono/unhandled_exception_policy` project setting to keep running after an unhandled exception.
- [**Mono:** New Godot constants to conditionally react to system variables at compile-time.](https://github.com/godotengine/godot/pull/28786)
- **Mono:** Support for Visual Studio 2019's MSBuild.

### Changed

- Tween and Timer now display an error message if they are started without being added to the scene tree first.
- Tweaked Timer's wait time property hint to allow values with 3 decimals and above 4096.
- Functions called from a signal can no longer disconnect the node from the signal they're connected to (unless using `call_deferred()`).
- Tabs and space indentation can no longer be mixed in the same GDScript file.
  - Each file must now use only tabs or spaces for indentation (not both).
- `assert()` in GDScript must now always be used with parentheses.
  - `assert(true)` is still valid, but `assert true` isn't valid anymore.
  - This is to account for the optional second parameter that defines a custom message.
- The "Trim" and "Normalize" WAV import options are now disabled by default.
  - This makes the default behavior more consistent with Ogg import.
- Ogg samples now have an icon in the editor, like WAV samples.
- Camera2D drag margins are now disabled by default.
  - If porting a project from Godot 3.1 where drag margins were used, these must be enabled manually again.
- The Camera2D Offset property now ignores the Limit property.
  - To get the old behavior back, move the camera itself instead of changing the offset.
- `Camera.project_position()` now requires a second `depth` argument to determine the distance of the point from the camera.
  - To get the old behavior back, pass the Camera's `near` property value as the second argument.
- `Skeleton.set_bone_global_pose()` was replaced by `Skeleton.set_bone_global_pose_override()`.
- UDP broadcasting is now disabled by default and must be enabled by calling `set_broadcast_enabled(true)` on the PacketPeerUDP instance.
- The editor and project manager now open slightly faster.
- Improved the Project Manager user interface.
  - New, simpler design with more space available for the project list.
  - Improved reporting of missing projects.
  - The search field is now focused when starting the Project Manager if there is at least one project in the list.
  - The search field now searches in both the project name and path.
    - If the search term contains a `/`, the whole path will be used to match the search them. Otherwise, only the last path component will be searched in.
- Refactored the Project Manager to be more efficient, especially with large project lists.
- Images in the Project Manager and Asset Library are now resized with Lanczos filtering for a smoother appearance.
- The editor now uses the font hinting algorithm that best matches the OS' default.
  - Hinting is set to "None" on macOS, and set to "Light" on Windows and Linux.
  - This can be changed in the Editor Settings.
- The editor window dimming when a popup appears is now less intense (60% → 50%).
  - The animation was also removed as it made the editor feel sluggish at lower FPS.
- Several editor menus have been reorganized for consistency and conciseness.
- Undo/Redo now supports more actions throughout the editor.
- Increased the height of the ItemList editor popup.
  - This makes it easier to edit large amounts of items.
- Opening a folder in FileDialog will now scroll back to the top.
- Folder icons in FileDialog can now be displayed with a different color using the `folder_icon_modulate` constant, making them easier to distinguish from files.
  - Folder icons in editor file dialogs are now tinted with the accent color.
- Improved colors in the light editor theme for better readability and consistency.
- Improved A* performance significantly by using a binary heap and OAHashMap.
- Tweaked the AABB transform algorithm to be ~1.2 times faster.
- Optimized the variant reference function, making complex scripts slightly faster.
- Disabled high-quality voxel cone tracing by default.
  - This makes GIProbe much faster out of the box, at the cost of less realistic reflections.
- Lowered the default maximum directional shadow distance (200 → 100).
  - This makes directional shadow rendering consistent between the editor and running project when using the default Camera node settings.
- Tweaked the default depth fog maximum distance to be independent of the Camera's `far` value (0..100).
  - This makes fog display consistent between the editor and a running project.
- Tweaked the default height fog values to be more logical (0..100 → 10..0).
  - This means height fog will be drawn from top-to-bottom, instead of being drawn from bottom-to-top.
- Significantly improved SSAO performance by using a lower sample count.
  - SSAO now uses 3×3 blurring by default, resulting in less visible noise patterns.
- When "Keep 3D Linear" is enabled, colors are no longer clamped to [0, 1] when using Linear tonemapping.
  - This allows rendering HDR values in floating-point texture targets for further processing or saving HDR data into files.
- The lightmap baker now calculates lightmap sizes dynamically based on surface area.
- Improved 3D KinematicBody performance and reliability.
- Orbiting in the 3D editor can now be done while holding <kbd>Alt</kbd>, for better compatibility with graphics tablets.
- Keys and actions are now released when the window loses focus.
- Tweens can now have a duration of 0.
- Particles and CPUParticles' Sphere emission shape now uses an uniform density sphere.
- `Viewport.size_override_stretch` is now exposed as a property (rather than just setter/getter methods).
- One-click deploy to Android now requires just one click if only one device is connected.
- The Project Manager will now infer a project name from the project path if the name was left to the default value.
- The WebSockets implementation now uses the smaller [wslay](https://tatsuhiro-t.github.io/wslay/) library instead of libwebsockets.
- Box selections in the editor now use a subtle outline for better visibility.
- Most 2D lines are now antialiased in the editor.
- CheckButtons now use a simpler design in the editor.
- Messages originating from the editor are now faded in the editor log.
  - This makes messages printed by the project stand out more.
- Folding arrows in the editor inspector are now displayed at the left for consistency with other foldable elements.
- Hovering or dragging guides in the 2D editor will now turn the cursor into a "resizing" shape.
- The editor update spinner is now hidden by default.
  - It can be enabled again in the Editor Settings.
- The "Update Always" option is now editor-wide instead of being project-specific.
- ColorPicker, OptionButton and MenuButton now use toggle mode, making them appear pressed when clicked.
- The ColorPicker preview was moved below the picker area to be closer to the sliders.
- Increased the Light2D height range from -100..100 to -2048..2048.
  - Lower and higher values can be entered manually too.
- Decreased the `rotation_degrees` range in various nodes to -360..360 to be easier to adjust using the slider.
  - Lower and higher values can still be entered manually, which is useful for animation purposes.
- The default RichTextLabel color is now `#ffffff`, matching the default Label color for better consistency.
- SpinBoxes now calculate the entered value using the Expression class.
  - For example, writing `2 + 2` in a SpinBox then pressing Enter will result in `4`.
- Saved resources no longer contain dependency indices and metadata such as node folding, resulting in more VCS-friendly files.
- The script editor's line length guideline is now enabled by default.
- The script editor state (such as breakpoints or the current line) is now preserved across editor sessions.
- The script editor's "Auto Brace Complete" setting is now enabled by default.
- The scripts panel toggle button is now located at the bottom-left of the script editor (instead of the File menu).
- Editor plugins can now be enabled without having an init script defined.
- Custom nodes added by plugins now have a translucent script icon in the scene tree dock.
- `EditorInterface.get_current_path()` to get the full path currently displayed in the FileSystem dock in an editor plugin.
- Copy constructors are now allowed for built-in types in GDScript.
  - This allows constructs such as `Vector2(Vector2(12, 34))`, which may be useful to simplify code in some cases.
- `weakref(null)` is now allowed in GDScript.
  - This makes checking for a valid reference more concise, as `if my_ref.get_ref()` is now sufficient (no need for `if my_ref and my_ref.get_ref()`).
- The number of signal connections and groups is now displayed in a tooltip when hovering the associated buttons in the scene tree dock.
- The right mouse button can now be used to pan in the 2D editor.
  - This is to improve usability when using a touchpad.
  - The middle mouse button can still be used to pan in the 2D editor.
- Zooming is now allowed while panning in the 2D editor.
- When the "Scroll To Pan" editor setting is enabled, the 2D editor can now be zoomed in by holding <kbd>Ctrl</kbd> and scrolling the mouse wheel.
- Zoom percentages in the 2D editor are now relative to the editor scale if the editor scale is higher than 100%.
- The 2D editor now displays the current zoom percentage.
  - The zoom percentage can be clicked to reset the zoom level to 100%.
- Improved sorting options in the Asset Library.
- Images now load faster in the Asset Library.
- A loading placeholder is now displayed while icons are loading in the Asset Library.
- Images failing to load in the Asset Library display a "broken file" icon.
- Improved the Asset Library page loading transitions.
- Tweaked the Asset Library detail page layout for better readability.
- Audio mixer faders now use a non-linear algorithm to better fit human hearing.
- Tooltips now appear faster when hovering elements in the editor (0.7 seconds → 0.5 seconds).
- Increased the low-processor usage mode's default maximum refresh rate (125 FPS → 144 FPS).
  - This makes the editor feel slightly smoother on 144 Hz displays.
- Tree scrolling when dragging now uses a larger drag margin, making drag-and-drop more convenient.
- Holding <kbd>Ctrl</kbd> now toggles snapping in GraphEdit.
- Improved the timeline's appearance in the animation editor.
- Improved snapping in the animation editor.
  - Snapping can be toggled temporarily by holding the <kbd>Ctrl</kbd> key.
  - Snapping can be made more precise by holding the <kbd>Shift</kbd> key.
  - Timeline snapping is now toggled by the Snap setting (like when moving keyframes).
- Keyframes are now easier to select in the animation editor.
- Selected keyframes now appear slightly larger in the animation editor.
- Boolean and color keyframe icons are now aligned to other keyframes in the animation editor.
- The Animation editor's line widths are now resized to match the editor scale.
- BPTC compression is now available for all HDR image formats.
- `Image.save_exr()` to save an image in EXR format, which supports high bit depths.
- Improved path and polygon editors.
  - New handle icons for path and polygon points.
  - Smooth path point and curve tangents now use different icons to be distinguished from sharp points.
  - Tangent lines are now gray in the Path2D and Path editors.
  - Path2D lines are now antialiased.
- Increased the TileSet and polygon UV editor's maximum zoom levels (400% → 1600%).
- Decreased the maximum allowed StyleBoxFlat corner detail (128 → 20).
  - This prevents slowness and glitches caused by using overly detailed corners.
- 3D collision shapes and RayCasts are now drawn in gray when disabled.
- Improved RayCast2D and one-way collision drawing.
  - Disabled RayCast2Ds are now displayed in gray.
  - One-way collision arrows are now orange by default, making them easier to distinguish them from RayCast2Ds.
  - Tweaked RayCast2D and one-way collision line shapes to look more like arrows.
- Improved rendering in the curve editor.
  - The grid is now rendered correctly when using a light theme.
  - The main line and edge line colors have been swapped for better visibility.
  - Tangent line widths are now resized to match the editor scale.
- Improved rendering in the performance monitor.
  - Dark colors are now used on light backgrounds for better visibility.
  - Graph lines are now thinner and opaque.
  - Graph line widths are now resized to match the editor scale.
  - Rounded values now display trailing zeroes to make their precision clearer.
- TileMap support for transform operations on cell textures bigger than the cell size has been reworked to properly support isometric tiles.
  - Breaks compatibility with some TileMaps from previous Godot versions. An opt-in `compatibility_mode` property can be used to restore the previous behavior.
- Some TileMap editor options were moved to the toolbar.
- The TileMap editor now displays coordinate information in the 2D viewport's bottom-left corner.
  - This fixes the TileMap editor width changing when hovering tiles in a small window.
- Brackets are now only inserted when necessary when autocompleting methods in the script editor.
- Improved dialogs when saving or removing an editor layout.
- Whitespace-only selections no longer cause the script editor to highlight all occurrences.
- Saving a script will now add a newline at the end of file if none was present already.
- Reorganized sections in the editor help to be in a more logical order.
- The editor help now uses horizontal margins if the screen is wide enough.
  - This makes sure lines keep a reasonable length for better readability.
- Increased line spacing in the editor help and asset library descriptions.
- The editor help now displays bold text using a bold font (instead of using a monospace font).
- The editor help now displays code using a slightly different color to be easier to distinguish.
- The editor help now displays types after parameter names to follow the GDScript static typing syntax.
- Editor help is now accessed using <kbd>Shift + F1</kbd>, for consistency with other applications.
  - Contextural help is now accessed using <kbd>Alt + F1</kbd> to accommodate for this change.
- The script editor's Find in Files dialog is now always available, even when no script is opened.
- Pressing <kbd>Shift + Enter</kbd> in the script editor Find dialog will now go to the previous match.
- Improved the node deletion confirmation message.
  - If there is only one node to delete, its name is displayed in the message.
  - If there is more than one node to delete, the number of nodes to delete is displayed.
- Improved the "Snap Object to Floor" functionality in the 3D editor.
  - An error message is now displayed if no nodes could be snapped.
  - Increased the maximum snapping height (10 → 20).
  - Increased the maximum snapping tolerance (0.1 → 0.2).
- 2D/3D selections, rotations and selected texts are now highlighted with the editor theme's accent color.
- 3D light gizmos are now tinted using the light's color, making navigation easier while using the unshaded display mode.
- Improved the 3D light and AudioStreamPlayer3D gizmos to better represent their depth in the 3D world.
- Tweaked the 3D manipulator gizmo's colors for better visibility.
- Tweaked the 2D and 3D axis colors for consistency with gizmo colors.
- Increased the default 3D manipulator gizmo opacity (0.2 → 0.4).
- The multiline text editor popup dialog's width is now capped on large displays.
  - This prevents lines from becoming very long, which could hamper text readability.
- Non-printable escape characters are now stripped when pasting text into a LineEdit.
- The TextEdit caret color now matches the default font color, making it easier to see.
- Empty exported NodePath properties now return `null` instead of `self`.
- Built-in scripts are no longer allowed to use `class_name` as it wasn't working properly.
- The second parameter of `substr()` is now optional and defaults to `-1`.
- More editor actions can now have shortcuts assigned (such as Revert Scene or Export).
- The project export path may now be written in a relative path.
  - Directories will be created recursively if the target directory doesn't exist.
- Items in the FileSystem dock can now be deselected by clicking empty space.
- "Set as Main Scene" context option for scenes in the FileSystem dock.
- The unused class variable GDScript warning is now disabled by default due to false positives.
- Warning-ignore comments now allow whitespace after the `#` character.
- Improved error reporting in the Particles emission point creation dialog.
- The number of warnings and errors that can be received in the remote debugger is now capped per second rather than per frame.
  - The default limit is 100 errors and 100 warnings per second, making it possible for the script editor to report up to 100 warnings before having messages hidden.
- UTF-8 characters are now supported in input action names.
- All platforms now use the `custom_template` property in each export preset to store the path to the custom export template (instead of `custom_package` for some platforms).
- Tween methods' `trans_type` and `ease_type` arguments are now optional, defaulting to `TRANS_LINEAR` and `EASE_IN_OUT` respectively.
- `PCKPacker.pck_start()` and `PCKPacker.flush()`'s `alignment` and `verbose` arguments (respectively) are now optional, defaulting to `0` and `false`.
- Exported PCK files now contain the Godot patch version in their header.
  - This can be used by external tools to detect the Godot version more accurately.
- Exporting a project PCK or ZIP from the command line must now be done with the new `--export-pack` command-line argument.
  - This was done to remove the ambiguity when exporting a project to macOS from the command line.
- Updated FreeType to 2.10, which changes how font metrics are calculated.
  - This may affect the appearance of some Controls, see [this issue](https://github.com/godotengine/godot/issues/28335) for details.
- The SCons build system now automatically detects the host platform.
  - `platform=<platform>` is no longer required when compiling for the host platform.
  - `platform=list` can be used to list the supported target platforms.
- **Windows:** Drive letters in file paths are now capitalized.
- **macOS:** <kbd>Control + H</kbd> and <kbd>Control + D</kbd> in TextEdit now delete the character at the left and right of the cursor (respectively).
- **macOS:** <kbd>Command + Left</kbd> in TextEdit now moves the cursor to the first non-whitespace character.
- **macOS:** Non-resizable windows are now allowed to enter fullscreen mode.
- **macOS:** The editor's title bar now uses dark mode on Mojave.
- **X11:** `OS.set_window_postion()` now takes window decorations into account.

### Removed

- Unused Panel `panelf` and `panelnc` styles.
- thekla_atlas dependency, as light baking now relies on [xatlas](https://github.com/jpcy/xatlas) for UV unwrapping.
- Rating icons in the Asset Library, as this feature isn't implemented in the backend.
- Some editor languages are no longer available due to missing support for RTL and text shaping in Godot:
  - Affected languages are Arabic, Bengali, Persian, Hebrew, Hindi, Malayalam, Sinhalese, Tamil, Telugu and Urdu.
  - These languages will be re-added once Godot supports RTL and text shaping.
- **Android:** ARMv6 support.
- **iOS:** ARMv7 support.
  - ARMv7 export templates can still be compiled from source to support the iPhone 5 and older.

### Fixed

- The Project Manager now remembers the sorting option that was previously set.
- The editor and project manager now have a minimum window size defined.
  - This prevents controls from overlapping each other by resizing the window to a very small size.
- Fixed radiance map generation, resulting in improved 3D performance and visual quality.
- Fixed issues with PBR environment mapping.
  - Materials should now look closer to what they look like in Substance Designer/Painter.
- Depth of field now affects transparent objects.
- Radiance is now generated when using a clear color sky.
- Contact shadows no longer display when shadow casting is disabled.
- Larger data types can now be constructed by swizzling in the shader language.
  - For instance, `vec2 test2 = vec2(0.0, 1.0); vec3 test3 = test2.xxx;` now works as in GLSL.
- The `AMBIENT_LIGHT_DISABLED` and `SHADOWS_DISABLED` flags now work when using the GLES2 renderer.
- The Keep background mode now works when using the GLES2 renderer.
- Several fixes to the GLES2 renderer:
  - Fixed transparency order.
  - Fixed vertex lighting being too bright.
  - Fixed occasional light flickering.
  - Fixed shadows cast from transparent materials.
  - Fog is no longer computed on unshaded materials.
    - This matches the GLES3 renderer's behavior.
  - GLES2 shader uniforms now use `highp` precision by default.
    - This prevents linking issues on some Android devices.
  - Negative OmniLights and SpotLights now work as expected.
  - The 3D editor's View Information pane now displays statistics correctly when using the GLES2 renderer.
- Textures compressed with ETC now support transparency by falling back to RGBA4444 or LA8.
- Alternate display modes are now marked as disabled in the editor when using the GLES2 renderer, as these are only supported when using GLES3.
- Fixed several inconsistencies between Particles and CPUParticles.
- Fixed particles scale randomization.
- Particles are now set to emit correctly when restarting.
- CheckBox and CheckButton now use the `check_vadjust` custom constant to adjust the icon Y position as intended.
- Fixed various issues with tab-related icons.
- Fixed issues in WebM colorspace corrections, resulting in better color output.
- CSG is now taken into account when generating navigation meshes.
- Curve2D and Curve3D interpolated values now behave as expected.
- Numeric slider grabbers in the editor inspector now update when scrolling using the mouse wheel.
- Scene modifications are no longer lost when renaming a file in the FileSystem dock.
- "Show in FileSystem" now clears the current search, so that the selected item can be seen immediately.
- LineEdit and TextEdit's context menus no longer display editing options if they are read-only.
- SpinBox mouse events are now correctly triggered by its LineEdit part.
- Per-word navigation in LineEdit and TextEdit now handles UTF-8 characters correctly.
- LineEdit placeholders, Tabs' names and WindowDialog titles now react correctly to translation changes.
- Fixed UI navigation when using gamepad analog sticks.
- Buttons' state is now reset when they exit the scene tree.
  - This prevents them from lingering in a "hovered" or "pressed" state.
- Tooltips now disappear when hiding the node they belong to.
- Encoded packet flags are no longer sent in the ENet multiplayer protocol, as ENet itself already sends that data.
  - This saves 4 bytes per packet.
- Audio trimming is now less aggressive, cutting at -50 dB instead of -30 dB.
- Audio trimming now has a small fade-out period, preventing audible pops.
- Audio mix rate and output latency settings are now consistently applied on all platforms.
- Fixed multichannel panning for AudioStreamPlayer3D.
- Opening a recent built-in script will now load the associated scene automtaically since doing so is required to edit the script.
- Declaring a class with `class_name` that has the same name as a singleton will now display a clearer error message.
- `script` is no longer allowed as a member variable name in GDScript, as that conflicts with the internal `script` property used by Object.
- Assigning a variable with a function index will no longer evaluate the function twice.
  - For instance, doing `a[function()] += 1` will no longer evaluate `function()` twice.
  - If the function has side effects, this may change the resulting program behavior.
- GDScript type checks are now enabled in release export templates.
- The Label font shadow now draws the font outline as well (if the base font has one).
- `Font.draw_char()` now draws the font outline as well (if the base font has one).
- The editor no longer redraws continuously when selecting a Control in a Container.
- Added some missing feature tags to the Project Settings "Override For..." menu.
- The `low_processor_mode_sleep_usec` project setting no longer affects the editor.
- Typed arrays and dictionaries no longer have their values shared across instances.
- `self` and object types can now be indexed as a dictionary again (like in Godot 3.0 and prior).
- Fixed `to_lower()` conversion with Cyrillic characters.
- The Find in Files replace dialog now allows empty replacement texts.
- The bottom panel no longer disappears when opening the theme editor on small displays.
- The script editor's color picker now changes only one color if multiple colors are present on the same line.
- The script editor's line length guideline is now drawn behind text.
- The script editor's line length guideline is now drawn at the correct position when font hinting is disabled.
- The script editor now automatically indents a line if the previous one ends with `[` or `(`.
  - This makes it possible to wrap arrays or function declarations/calls without pressing <kbd>Tab</kbd> every line.
- Fixed autocompletion in the script editor.
  - The script editor can now autocomplete enum values.
  - The script editor can now autocomplete node paths starting with `$"` or `$'`.
- Custom script editor templates can now use type hints.
- Shift operators with a number not between 0 and 63 (inclusive) will now result in a compile-time error in GDScript.
- Warnings no longer count towards the "Too many errors!" message.
- AnimationTrackEdit now displays invalid value keys again (as it did in 3.0).
- Fixed the display of function/audio/animation tracks in the blend tree animation filter.
- The editor shortcuts menu no longer displays all unassigned shortcuts when searching for a substring of "None".
- The editor's performance monitor now displays memory/file sizes larger than 2 GB correctly.
- The editor debugger now displays keyboard shortcuts when hovering the "Step Into", "Step Over", "Break" and "Continue" buttons.
- The editor debugger now always handles connections.
  - Subsequent connections will be dropped immediately to avoid locking.
- Large rotation offset/snap values no longer appear to be cut off in the Configure Snap dialog.
- Documentation tooltips in the editor now wrap to multiple lines correctly.
- Locked 3D nodes are no longer selectable in the 3D viewport, matching the 2D editor's behavior.
- All 3D gizmos now notify changes correctly, which means the inspector now displays up-to-date properties after using them.
- The 3D manipulator gizmo's size is now capped at low viewport heights, preventing it from outgrowing the viewport's bounds.
- The editor filesystem now refreshes on file changes if the project is located on an exFAT filesystem.
- Fixed many cases of colors not changing correctly when switching the editor from a dark theme to a light theme (or vice versa) without restarting.
- The Show in File Manager context menu option now works with files marked as favorite.
- The random number generator's seed is now properly set up.
- Antialiased and rounded StyleBoxFlat corners now handle different border widths correctly.
- The StyleBox preview now accounts for shadows and content margins.
  - This fixes the preview going out of bounds in the inspector.
- Text resources no longer contain an extraneous line break at the end of file.
- Transform's `FLIP_Y` and `FLIP_Z` constants now work as expected.
- Fixed importing BMP images.
- The positional command-line argument is now only considered to be a scene path if it ends with `.scn`, `.tscn` or `.escn`.
  - This makes it possible to parse command-line arguments in a standard fashion (`--foo bar` now works, not just `--foo=bar`).
  - This also makes it possible to use file associations or drag-and-drop and have the positional argument parsed by the project.
- The `--audio-driver` and `--video-driver` command-line arguments are now validated; an error message will be printed if an invalid value is passed.
- The `--check-only` command-line argument now returns a non-zero exit code if an invalid script is passed using `--script`.
- Exporting a project via the command-line now returns a non-zero exit code if an error occurred during exporting.
- Console output is no longer colored when standard output isn't a TTY.
  - This prevents Godot from writing ANSI escape codes when redirecting standard output or standard error to a file.
- **Android:** Gamepads are now correctly detected when the application starts.
- **Android:** Fix some keyboards being detected as gamepads and not working as a result.
- **Android:** The editor now detects if the device is connected using wireless `adb` and will debug using Wi-Fi in this case.
- **HTML5:** Fixed the pointer position on hiDPI displays.
- **HTML5:** `OS.get_system_time_msec()` now returns the correct value like on other platforms.
- **iOS:** On iOS 11 or later, gestures near screen edges are now handled by Godot instead of the OS.
- **Windows:** Line endings are now converted to CRLF when setting clipboard content.
- **Windows:** Getting the path to the Downloads directory using `OS.get_system_dir()` now works correctly.
  - This fixes line endings being invisible when pasting into other applications.
- **macOS:** `OS.get_real_window_size()` and `OS.set_window_size()` are now handled correctly on hiDPI displays.
- **X11:** `OS.get_window_position()` now returns absolute coordinates.
- **X11:** Fixed audio playing on the wrong speakers when using PulseAudio on 5.1 setups.
- **X11:** `OS.set_window_maximized()` now gives up after 0.5 seconds.
  - This makes the editor no longer freeze on startup when using fvwm.

## [3.1] - 2019-03-13

### Added

- OpenGL ES 2.0 renderer.
- [Visual shader editor.](https://godotengine.org/article/visual-shader-editor-back)
  - New PBR output nodes.
  - Conversion between Vector3 and scalar types is now automatic.
  - Ability to create custom nodes via scripting.
  - Ports can now be previewed.
- [3D soft body physics.](https://godotengine.org/article/soft-body)
- [3D ragdoll system.](https://godotengine.org/article/godot-ragdoll-system)
- [Constructive solid geometry in 3D.](https://godotengine.org/article/godot-gets-csg-support)
- [2D meshes and skeletal deformation.](https://godotengine.org/article/godot-gets-2d-skeletal-deform)
- [Various improvements to KinematicBody2D.](https://godotengine.org/article/godot-31-will-get-many-improvements-kinematicbody)
  - Support for snapping the body to the floor.
  - Support for RayCast shapes in kinematic bodies.
  - Support for synchronizing kinematic movement to physics, avoiding an one-frame delay.
- WebSockets support using [libwebsockets](https://libwebsockets.org/).
- UPnP support using [MiniUPnP](http://miniupnp.free.fr).
- [Revamped inspector.](https://godotengine.org/article/godot-gets-new-inspector)
  - Improved visualization and editing of numeric properties.
  - Vector and matrix types can now be edited directly (no pop-ups).
  - Subresources can now be edited directly within the same inspector.
  - Layer names can now be displayed in the inspector.
  - Proper editing of arrays and dictionaries.
  - Ability to reset any property to its default value.
- [Improved animation editor.](https://godotengine.org/article/godot-gets-brand-new-animation-editor-cinematic-support)
  - Simpler, less cluttered layout.
  - New Bezier, Audio and Animation tracks.
  - Several key types can be previewed directly in the track editor.
  - Tracks can now be grouped and filtered on a per-node basis.
  - Copying and pasting tracks between animations is now possible.
  - New Capture mode to blend from a node's current value to the first key in a track.
- [Improved animation tree and new state machine.](https://godotengine.org/article/godot-gets-new-animation-tree-state-machine)
  - More visual feedback in the blend tree editor.
  - 1D and 2D blend spaces are now supported.
  - Ability to write custom blending logic.
  - Support for root motion.
- [New FileSystem dock.](https://godotengine.org/article/godot-gets-new-filesystem-dock-3-1)
  - Unified view of folders and files in the same panel.
  - Files can now be marked as favorites, not only folders.
  - Files now have icons representing their type, or thumbnail previews when relevant.
  - New search field to filter entries in the tree.
- [OpenSimplexNoise and NoiseTexture resources.](https://godotengine.org/article/simplex-noise-lands-godot-31)
- [Optional static typing in GDScript.](https://godotengine.org/article/optional-typing-gdscript)
  - Does not currently improve performance, but helps write more robust code.
- Warning system in GDScript.
  - Reports potential code issues such as:
    - unused variables,
    - standalone expressions,
    - discarded return values from functions,
    - unreachable code after a `return` statement,
    - …
  - Warnings can be disabled in the Project Settings or by writing special comments.
- [GDScript keyword `class_name` to register scripts as classes.](https://docs.godotengine.org/en/latest/getting_started/step_by_step/scripting_continued.html#register-scripts-as-classes)
- Simple expression language independent from GDScript, used by inspector boxes that accept numeric values.
  - Can also be used in projects.
- C# projects can now be exported for Windows, Linux, and macOS targets.
- The `server` platform is back as it was in Godot 2.1.
  - It is now again possible to run a headless Godot instance on Linux.
- Support for BPTC texture compression on desktop platforms.
- New properties for SpatialMaterial.
  - Dithering-based distance fade, for fading materials without making them transparent.
  - Disable ambient light on a per-material basis.
- Option to link Mono statically on Windows.
- Unified class and reference search in the editor.
- Revamped TileSet editor with support for undo/redo operations.
- Various quality-of-life improvements to the Polygon2D and TextureRegion editors.
- RandomNumberGenerator class that allows for multiple instances at once.
- Array methods `min()` and `max()` to return the smallest and largest value respectively.
- Dictionary method `get(key[, default])` where `default` is returned if the key does not exist.
- Node method `print_tree_pretty()` to print a graphical view of the scene tree.
- String methods `trim_prefix()`, `trim_suffix()`, `lstrip()`, `rstrip()`.
- OS methods:
  - `get_system_time_msecs()`: Return the system time with milliseconds.
  - `get_audio_driver_name()` and `get_audio_driver_count()` to query audio driver information.
  - `get_video_driver_count()` and `get_video_driver_name()` to query renderer information.
  - `center_window()`: Center the window on the screen.
  - `move_window_to_foreground()`: Move the window to the foreground.
- StreamPeerTCP method `set_no_delay()` to enable the `TCP_NODELAY` option.
- EditorPlugin method `remove_control_from_container()`.
- Ability to set Godot windows as "always on top".
- Ability to create windows with per-pixel transparency.
- New GLSL built-in functions in the shader language:
  - `radians()`
  - `degrees()`
  - `asinh()`
  - `acosh()`
  - `atanh()`
  - `exp2()`
  - `log2()`
  - `roundEven()`
- New command-line options:
  - `--build-solutions`: Build C# solutions without starting the editor.
  - `--print-fps`: Display frames per second to standard output.
  - `--quit`: Quit the engine after the first main loop iteration.
- Debugger button to copy error messages.
- Support for `.escn` scenes has been added for use with the new Blender exporter.
- It is now possible to scale an OBJ mesh when importing.
- `popup_closed` signal for `ColorPickerButton`.
- Methods that are deprecated can now print warnings.
- Input actions can now provide an analog value.
- Input actions can now be mapped to either a specific device or all devices.
- DNS resolution for high-level networking.
- Servers can now kick/disconnect peers in high-level networking.
- Servers can now access IP and port information of peers in high-level networking.
- High-level multiplayer API decoupled from SceneTree (see `SceneTree.multiplayer_api`/`SceneTree.custom_multiplayer_api`), can now be extended.
- `Input.set_default_cursor_shape()` to change the default shape in the viewport.
- Custom cursors can now be as large as 256×256 (needed to be exactly 32×32 before).
- Support for radio-looking items with icon in `PopupMenu`s.
- Drag and drop to rearrange Editor docks.
- TileSet's `TileMode` is now exposed to GDScript.
- `OS.get_ticks_usec()` is now exposed to GDScript.
- Normals can now be flipped when generated via `SurfaceTool`.
- TextureProgress bars can now be bilinear (extending in both directions).
- The character used for masking secrets in LineEdit can now be changed.
- Improved DynamicFont:
  - DynamicFonts can now use high-quality outlines generated by FreeType.
  - DynamicFonts can now have their anti-aliasing disabled.
  - DynamicFonts can now have their hinting tweaked ("Normal", "Light" or "None").
  - Colored glyphs such as emoji are now supported.
- Universal translation of touch input to mouse input.
- AudioStreamPlayer, AudioStreamPlayer2D, and AudioStreamPlayer3D now have a pitch scale property.
- Support for MIDI input.
- Support for audio capture from microphones.
- `GROW_DIRECTION_BOTH` for Controls.
- Selected tiles can be moved in the tile map editor.
- The editor can now be configured to display the project window on the previous or next monitor (relative to the editor).
  - If either end is reached, then the project will start on the last or first monitor (respectively).
- Signal in VideoPlayer to notify when the video finished playing.
- `Image.bumpmap_to_normalmap()` to convert bump maps to normal maps.
- `File.get_path()` and `File.get_path_absolute()`.
- Unselected tabs in the editor now have a subtle background for easier identification.
- The depth fog's end distance is now configurable independently of the far plane distance.
- The alpha component of the fog color can now be used to control fog density.
- The 3D editor's information panel now displays the camera's coordinates.
- New options to hide the origin and viewport in the 2D editor.
- Improved 3D editor grid:
  - The grid size and number of subdivisions can now be configured.
  - Its primary and secondary colors can now also be changed.
- <kbd>Ctrl</kbd> now toggles snapping in the 3D viewport.
- Find & replace in files (<kbd>Ctrl + Shift + F</kbd> by default).
- Batch node renaming tool (<kbd>Ctrl + F2</kbd> by default).
- More editor scaling options to support HiDPI displays.
- Type icons can now be enabled in the editor again.
- Buttons in the editor to open common directories in the OS file manager:
  - project data directory,
  - user data directory,
  - user settings directory.
- Projects can now be sorted by name or modification date in the project manager.
- Projects can now be imported from ZIP archives in the project manager.
- Improved autocompletion.
  - Keywords are now present in autocompletion results.
- `editor` and `standalone` feature tags to check whether the project is running from an editor or non-editor binary.
- `android_add_asset_dir("...")` method to Android module Gradle build configuration.
- **iOS:** Support for exporting to the iPhone X.
- **iOS:** Re-added support for in-app purchases.

### Changed

- [Built-in vector types now use copy-on-write mode as originally intended](https://godotengine.org/article/why-we-broke-your-pr), resulting in increased engine performance.
- The [mbedtls](https://tls.mbed.org/) library is now used instead of OpenSSL.
- [Renamed several core files](https://github.com/godotengine/godot/pull/25821).
  - Third-party modules may have to be updated to reflect this.
- SSL certificates are now bundled in exported projects unless a custom bundle is specified.
- Improved buffer writing performance on Windows and Linux.
- Removed many debugging prints in the console.
- Export templates now display an error dialog if no project was found when starting.
- DynamicFont oversampling is now enabled by default.
- Nodes' internal logic now consistently use internal physics processing.
- Allow attaching and clearing scripts on multiple nodes at once.
- Default values are no longer saved in scene and resource files.
- The selection rectangle of 2D nodes is now hidden when not pertinent (no more rectangle for collision shapes).
- SSE2 is now enabled in libsquish, resulting in improved S3TC encoding performance.
- Tangent and binormal coordinates are now more consistent across mesh types (primitive/imported), resulting in more predictable normal map and depth map appearance.
- Better defaults for 3D scenes.
  - The default procedural sky now has a more neutral blue tone.
  - The default SpatialMaterial now has a roughness value of 1 and metallic value of 0.
  - The fallback material now uses the same values as the default SpatialMaterial.
- Text editor themes are now sorted alphabetically in the selection dropdown.
- The 3D manipulator gizmo now has a smoother, more detailed appearance.
- The 3D viewport menu button now has a background to make it easier to read.
- QuadMeshes are now built using two triangles (6 vertices) instead of one quad (4 vertices).
  - This was done because quads are deprecated in OpenGL.
- Controls inside containers are no longer movable or resizable but can still be selected.
- The `is` GDScript keyword can now be used to compare a value against built-in types.
- Exported variables with type hints are now always initialized.
  - For example, `export(int) var a` will be initialized to `0`.
- Named enums in GDScript no longer create script constants.
  - This means `enum Name { VALUE }` must now be accessed with `Name.VALUE` instead of `VALUE`.
- Cyclic references to other scripts with `preload()` are no longer allowed.
  - `load()` should be used in at least one of the scripts instead.
- `switch`, `case` and `do` are no longer reserved identifiers in GDScript.
- Shadowing variables from parent scopes is no longer allowed in GDScript.
- Function parameters' default values can no longer depend on other parameters in GDScript.
- Indentation guides are now displayed in a more subtle way in the script editor.
  - Indentation guides are now displayed when indenting using spaces.
- Multi-line strings are now highlighted as strings rather than as comments in the script editor.
  - This is because GDScript does not officially support multiline comments.
- Increased the script editor's line spacing (4 pixels → 6 pixels).
- Increased the caret width in the script editor (1 pixel → 2 pixels).
- The project manager window is now resized to match the editor scale.
- The asset library now makes use of threading, making loading more responsive.
- Line spacing in the script editor, underlines and caret widths are now resized to match the editor scale.
- Replaced editor icons for checkboxes and radio buttons with simpler designs.
- Tweaked the editor's success, error, and warning text colors for better readability and consistency.
- **Android:** Custom permissions are now stored in an array and their amount is no longer limited to 20.
  - Custom permissions will have to be redefined in projects imported from older versions.
- **Android:** Provide error details when an in-app purchase fails.
- **Linux:** `OS.alert()` now uses Zenity or KDialog if available instead of xmessage.
- **Mono:** Display stack traces for inner exceptions.
- **Mono:** Bundle `mscorlib.dll` with Godot to improve portability.

### Removed

- Removed the RtAudio backend on Windows in favor of WASAPI, which is the default since 3.0.
- **macOS:** Support for 32-bit and fat binaries.

### Fixed

- [`move_and_slide()` now behaves differently at low velocities](https://github.com/godotengine/godot/issues/21683), which makes it function as originally intended.
- AnimatedSprite2D's `animation_finished` signal is now triggered at the end of the animation, instead of as soon as the last frame displays.
- Audio buses can now be removed in the editor while they are used by AudioStreamPlayer2D/3D nodes.
- Do not show the project manager unless no project was found at all.
- The animation editor time offset indicator no longer "walks" when resizing the editor.
- Allow creation of a built-in GDScript file even if the filename suggested already exists.
- Show tooltips in the editor when physics object picking is disabled.
- Button shortcuts can now be triggered by gamepad buttons.
- Fix a serialization bug that could cause TSCN files to grow very large.
- Gizmos are now properly hidden on scene load if the object they control is hidden.
- Camera gizmos in the 3D viewport no longer look twice as wide as they actually are.
- Copy/pasting from the editor on X11 will now work more reliably.
- `libgcc_s` and `libstdc++` are now linked statically for better Linux binary portability.
- The FPS cap set by `force_fps` in the Project Settings is no longer applied to the editor.
  - Low FPS caps no longer cause the editor to feel sluggish.
- hiDPI is now detected and used if needed in the project manager.
- The Visual Studio Code external editor option now recognizes more binary names such as `code-oss`, making detection more reliable.
- The `-ffast-math` flag is no longer used when compiling Godot, resulting in increased floating-point determinism.
- Fix spelling of `apply_torque_impulse()` and deprecate the misspelled method.
- Escape sequences like `\n` and `\t` are now recognized in CSV translation files.
- Remove spurious errors when using a PanoramaSky without textures.
- The lightmap baker will now use all available cores on Windows.
- Bullet physics now correctly calculates effective gravity on KinematicBodies.
- Setting the color `v` member now correctly sets the `s` member.
- RichTextLabels now correctly determine the baseline for all fonts.
- SpinBoxes now correctly calculate their initial size.
- OGG streams now correctly signal the end of playback.
- Android exporter no longer writes unnecessary permissions to the exported APK.
- Debugger "focus stealing" now works more reliably.
- Subresources are now always saved when saving a scene.
- Many fixes related to importers (glTF, Collada, audio), physics (Bullet), Mono/C#, GDNative, Android/iOS.
- **Mono:** Many fixes and improvements to C# support (including a `[Signal]` attribute).
- **WebAssembly:** Supply proper CORS headers.

### Security

- Fixed a security issue relating to deserializing Variants.

## [3.0] - 2018-01-29

### Added

- Physically-based renderer using OpenGL ES 3.0.
  - Uses the Disney PBR model, with clearcoat, sheen and anisotropy parameters available.
  - Uses a forward renderer, supporting multi-sample anti-aliasing (MSAA).
  - Parallax occlusion mapping.
  - Reflection probes.
  - Screen-space reflections.
  - Real-time global illumination using voxel cone tracing (GIProbe).
  - Proximity fade and distance fade (useful for creating soft particles and various effects).
  - [Lightmapper](https://godotengine.org/article/introducing-new-last-minute-lightmapper) for lower-end desktop and mobile platforms, as an alternative to GIProbe.
- New SpatialMaterial resource, replacing FixedMaterial.
  - Multiple passes can now be specified (with an optional "grow" property), allowing for effects such as cel shading.
- Brand new 3D post-processing system.
  - Depth of field (near and far).
  - Fog, supporting light transmittance, sun-oriented fog, depth fog and height fog.
  - Tonemapping and Auto-exposure.
  - Screen-space ambient occlusion.
  - Multi-stage glow and bloom, supporting optional bicubic upscaling for better quality.
  - Color grading and various adjustments.
- Rewritten audio engine from scratch.
  - Supports audio routing with arbitrary number of channels, including Area-based audio redirection ([video](https://youtu.be/K2XOBaJ5OQ0)).
  - More than a dozen of audio effects included.
- Rewritten 3D physics using [Bullet](https://bulletphysics.org/).
- UDP-based high-level networking API using [ENet](http://enet.bespin.org/).
- IPv6 support for all of the engine's networking APIs.
- Visual scripting.
- Rewritten import system.
  - Assets are now referenced with their source files, then imported in a transparent manner by the engine.
  - Imported assets are now cached in a `.import` directory, making distribution and versioning easier.
  - Support for ETC2 compression.
  - Support for uncompressed Targa (.tga) textures, allowing for faster importing.
- Rewritten export system.
  - GPU-based texture compression can now be tweaked per-target.
  - Support for exporting resource packs to build DLC / content addons.
- Improved GDScript.
  - Pattern matching using the `match` keyword.
  - `$` shorthand for `get_node()`.
  - Setters and getters for node properties.
  - Underscores in number literals are now allowed for improved readability (for example,`1_000_000`).
  - Improved performance (+20% to +40%, based on various benchmarks).
- [Feature tags](https://docs.godotengine.org/en/latest/getting_started/workflow/export/feature_tags.html) in the Project Settings, for custom per-platform settings.
- Full support for the [glTF 2.0](https://www.khronos.org/gltf/) 3D interchange format.
- Freelook and fly navigation to the 3D editor.
- Built-in editor logging (logging standard output to a file), disabled by default.
- Improved, more intuitive file chooser in the editor.
- Smoothed out 3D editor zooming, panning and movement.
- Toggleable rendering information box in the 3D editor viewport.
  - FPS display can also be enabled in the editor viewport.
- Ability to render the 3D editor viewport at half resolution to achieve better performance.
- GDNative for binding languages like C++ to Godot as dynamic libraries.
  - Community bindings for [D](https://github.com/GodotNativeTools/godot-d), [Nim](https://github.com/pragmagic/godot-nim) and [Python](https://github.com/touilleMan/godot-python) are available.
- Editor settings and export templates are now versioned, making it easier to use several Godot versions on the same system.
- Optional soft shadows for 2D rendering.
- HDR sky support.
- Ability to toggle V-Sync while the project is running.
- Panorama sky support (sphere maps).
- Support for WebM videos (VP8/VP9 with Vorbis/Opus).
- Exporting to HTML5 using WebAssembly.
- C# support using Mono.
  - The Mono module is disabled by default, and needs to be compiled in at build-time.
  - The latest Mono version (5.4) can be used, fully supporting C# 7.0.
- Support for rasterizing SVG to images on-the-fly, using the nanosvg library.
  - Editor icons are now in SVG format, making them better-looking at non-integer scales.
  - Due to the library used, only simpler SVGs are well-supported, more complex SVGs may not render correctly.
- Support for oversampling DynamicFonts, keeping them sharp when scaled to high resolutions.
- Improved StyleBoxFlat.
  - Border widths can now be set per-corner.
  - Support for anti-aliased rounded and beveled corners.
  - Support for soft drop shadows.
- VeryLoDPI (75%) and MiDPI (150%) scaling modes for the editor.
- Improved internationalization support for projects.
  - Language changes are now effective without reloading the current scene.
- Implemented missing features in the HTML5 platform.
  - Cursor style changes.
  - Cursor capturing and hiding.
- Improved styling and presentation of HTML5 exports.
  - A spinner is now displayed during loading.
- Rewritten the 2D and 3D particle systems.
  - Particles are now GPU-based, allowing their use in much higher quantities than before.
  - Meshes can now be used as particles.
  - Particles can now be emitted from a mesh's shape.
  - Properties can now be modified over time using an editable curve.
  - Custom particle shaders can now be used.
- New editor theme, with customizable base color, highlight color and contrast.
  - A light editor theme option is now available, with icons suited to light backgrounds.
  - Alternative dark gray and Arc colors are available out of the box.
- New adaptive text editor theme, adjusting automatically based on the editor colors.
- Support for macOS trackpad gestures in the editor.
- Exporting to macOS now creates a `.dmg` disk image if exporting from an editor running on macOS.
  - Signing the macOS export now is possible if running macOS (requires a valid code signing certificate).
- Exporting to Windows now changes the exported project's icon using `rcedit` (requires WINE if exporting from Linux or macOS).
- Improved build system.
  - Support for compiling using Visual Studio 2017.
  - [SCons](https://scons.org/) 3.0 and Python 3 are now supported (SCons 2.5 and Python 2.7 still work).
  - Link-time optimization can now be enabled by passing `use_lto=yes` to the SCons command line.
    - Produces faster and sometimes smaller binaries.
    - Currently only supported with GCC and MSVC.
  - Added a progress percentage when compiling Godot.
  - `.zip` archives are automatically created when compiling HTML5 export templates.
- Easier and more powerful way to create editor plugins with EditorPlugin and related APIs.

### Changed

- Increased the default low-processor-usage mode FPS limit (60 → 125).
  - This makes the editor smoother and more responsive.
- Increased the default 3D editor camera's field of view (55 → 70).
- Increased the default 3D Camera node's field of view (65 → 70).
- Changed the default editor font (Droid Sans → [Noto Sans](https://www.google.com/get/noto/)).
- Changed the default script editor font (Source Code Pro → [Hack](https://sourcefoundry.org/hack/))
- Renamed `engine.cfg` to `project.godot`.
  - This allows users to open a project by double-clicking the file if Godot is associated to `.godot` files.
- Some methods from the `OS` singleton were moved to the new `Engine` singleton.
- Switched from [GLEW](http://glew.sourceforge.net/) to [GLAD](https://glad.dav1d.de/) for OpenGL wrapping.
- Changed the SCons build flag for simple logs (`colored=yes` → `verbose=no`).
- The HTML5 platform now uses WebGL 2.0 (instead of 1.0).
- Redesigned the Godot logo to be more legible at small sizes.

### Deprecated

- `opacity` and `self_opacity` are replaced by `modulate` and `self_modulate` in all 2D nodes, allowing for full color changes in addition to opacity changes.

### Removed

- Skybox support.
  - Replaced with panorama skies, which are easier to import.
- Opus audio codec support.
  - This is due to the way the new audio engine is designed.
- HTML5 export using asm.js.
  - Only WebAssembly is supported now, since all browsers supporting WebGL 2.0 also support WebAssembly.

[3.4]: https://downloads.tuxfamily.org/godotengine/3.4/Godot_v3.4-stable_changelog_chrono.txt
[3.3]: https://downloads.tuxfamily.org/godotengine/3.3/Godot_v3.3-stable_changelog_chrono.txt
[3.2.3]: https://downloads.tuxfamily.org/godotengine/3.2.3/Godot_v3.2.3-stable_changelog_chrono.txt
[3.2.2]: https://downloads.tuxfamily.org/godotengine/3.2.2/Godot_v3.2.2-stable_changelog_chrono.txt
[3.2.1]: https://downloads.tuxfamily.org/godotengine/3.2.1/Godot_v3.2.1-stable_changelog_chrono.txt
[3.2]: https://downloads.tuxfamily.org/godotengine/3.2/Godot_v3.2-stable_changelog_chrono.txt
[3.1]: https://downloads.tuxfamily.org/godotengine/3.1/Godot_v3.1-stable_changelog_chrono.txt
[3.0]: https://downloads.tuxfamily.org/godotengine/3.0/Godot_v3.0-stable_changelog_chrono.txt
