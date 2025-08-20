/**************************************************************************/
/*  gdextension_spx_ext.h                                                 */
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

#ifndef GDEXTENSION_SPX_EXT_H
#define GDEXTENSION_SPX_EXT_H

#include "gdextension_interface.h"
#ifndef NOT_GODOT_ENGINE
#include "core/variant/variant.h"
extern void gdextension_spx_setup_interface();
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef GDExtensionConstStringPtr GdString;
typedef GDExtensionInt GdInt;
typedef GDExtensionInt GdObj;
typedef GDExtensionBool GdBool;
typedef real_t GdFloat;
typedef Vector4 GdVec4;
typedef Vector3 GdVec3;
typedef Vector2 GdVec2;
typedef Color GdColor;
typedef Rect2 GdRect2;


typedef struct {
	// 0 is return value
	// 1-7 are arguments
	GdVec4 Ret;
	GdVec4 Arg0;
	GdVec4 Arg1;
	GdVec4 Arg2;
	GdVec4 Arg3;
	GdVec4 Arg4;
	GdVec4 Arg5;
	GdVec4 Arg6;
	GdVec4 Arg7;
} CallFrameArgs;

typedef void *GDExtensionSpxCallbackInfoPtr;
typedef void (*GDExtensionSpxGlobalRegisterCallbacks)(GDExtensionSpxCallbackInfoPtr callback_ptr);

// string
typedef void (*GDExtensionSpxStringNewWithLatin1Chars)(GDExtensionUninitializedStringPtr r_dest, const char *p_contents);
typedef void (*GDExtensionSpxStringNewWithUtf8Chars)(GDExtensionUninitializedStringPtr r_dest, const char *p_contents);
typedef void (*GDExtensionSpxStringNewWithLatin1CharsAndLen)(GDExtensionUninitializedStringPtr r_dest, const char *p_contents, GdInt p_size);
typedef void (*GDExtensionSpxStringNewWithUtf8CharsAndLen)(GDExtensionUninitializedStringPtr r_dest, const char *p_contents, GdInt p_size);
typedef GdInt (*GDExtensionSpxStringToLatin1Chars)(GDExtensionConstStringPtr p_self, char *r_text, GdInt p_max_write_length);
typedef GdInt (*GDExtensionSpxStringToUtf8Chars)(GDExtensionConstStringPtr p_self, char *r_text, GdInt p_max_write_length);
// variant
typedef GDExtensionPtrConstructor (*GDExtensionSpxVariantGetPtrConstructor)(GDExtensionVariantType p_type, int32_t p_constructor);
typedef GDExtensionPtrDestructor (*GDExtensionSpxVariantGetPtrDestructor)(GDExtensionVariantType p_type);

// callback
typedef void (*GDExtensionSpxCallbackOnEngineStart)();
typedef void (*GDExtensionSpxCallbackOnEngineUpdate)(GdFloat delta);
typedef void (*GDExtensionSpxCallbackOnEngineFixedUpdate)(GdFloat delta);
typedef void (*GDExtensionSpxCallbackOnEngineDestroy)();
typedef void (*GDExtensionSpxCallbackOnEnginePause)(GdBool is_paused);

typedef void (*GDExtensionSpxCallbackOnSceneSpriteInstantiated)(GdObj obj,GdString type_name);

typedef void (*GDExtensionSpxCallbackOnSpriteReady)(GdObj obj);
typedef void (*GDExtensionSpxCallbackOnSpriteUpdated)(GdFloat delta);
typedef void (*GDExtensionSpxCallbackOnSpriteFixedUpdated)(GdFloat delta);
typedef void (*GDExtensionSpxCallbackOnSpriteDestroyed)(GdObj obj);

typedef void (*GDExtensionSpxCallbackOnSpriteFramesSetChanged)(GdObj obj);
typedef void (*GDExtensionSpxCallbackOnSpriteAnimationChanged)(GdObj obj);
typedef void (*GDExtensionSpxCallbackOnSpriteFrameChanged)(GdObj obj);
typedef void (*GDExtensionSpxCallbackOnSpriteAnimationLooped)(GdObj obj);
typedef void (*GDExtensionSpxCallbackOnSpriteAnimationFinished)(GdObj obj);

typedef void (*GDExtensionSpxCallbackOnSpriteVfxFinished)(GdObj obj);

typedef void (*GDExtensionSpxCallbackOnSpriteScreenExited)(GdObj obj);
typedef void (*GDExtensionSpxCallbackOnSpriteScreenEntered)(GdObj obj);

typedef void (*GDExtensionSpxCallbackOnMousePressed)(GdInt keyid);
typedef void (*GDExtensionSpxCallbackOnMouseReleased)(GdInt keyid);
typedef void (*GDExtensionSpxCallbackOnKeyPressed)(GdInt keyid);
typedef void (*GDExtensionSpxCallbackOnKeyReleased)(GdInt keyid);
typedef void (*GDExtensionSpxCallbackOnActionPressed)(GdString action_name);
typedef void (*GDExtensionSpxCallbackOnActionJustPressed)(GdString action_name);
typedef void (*GDExtensionSpxCallbackOnActionJustReleased)(GdString action_name);
typedef void (*GDExtensionSpxCallbackOnAxisChanged)(GdString action_name, GdFloat value);

typedef void (*GDExtensionSpxCallbackOnCollisionEnter)(GdInt self_id, GdInt other_id);
typedef void (*GDExtensionSpxCallbackOnCollisionStay)(GdInt self_id, GdInt other_id);
typedef void (*GDExtensionSpxCallbackOnCollisionExit)(GdInt self_id, GdInt other_id);
typedef void (*GDExtensionSpxCallbackOnTriggerEnter)(GdInt self_id, GdInt other_id);
typedef void (*GDExtensionSpxCallbackOnTriggerStay)(GdInt self_id, GdInt other_id);
typedef void (*GDExtensionSpxCallbackOnTriggerExit)(GdInt self_id, GdInt other_id);

typedef void (*GDExtensionSpxCallbackOnUiReady)(GdObj obj);
typedef void (*GDExtensionSpxCallbackOnUiUpdated)(GdObj obj);
typedef void (*GDExtensionSpxCallbackOnUiDestroyed)(GdObj obj);

typedef void (*GDExtensionSpxCallbackOnUiPressed)(GdObj obj);
typedef void (*GDExtensionSpxCallbackOnUiReleased)(GdObj obj);
typedef void (*GDExtensionSpxCallbackOnUiHovered)(GdObj obj);
typedef void (*GDExtensionSpxCallbackOnUiClicked)(GdObj obj);
typedef void (*GDExtensionSpxCallbackOnUiToggle)(GdObj obj, GdBool is_on);
typedef void (*GDExtensionSpxCallbackOnUiTextChanged)(GdObj obj, GdString text);


typedef struct {
	// engine
	GDExtensionSpxCallbackOnEngineStart func_on_engine_start;
	GDExtensionSpxCallbackOnEngineUpdate func_on_engine_update;
	GDExtensionSpxCallbackOnEngineFixedUpdate func_on_engine_fixed_update;
	GDExtensionSpxCallbackOnEngineDestroy func_on_engine_destroy;
	GDExtensionSpxCallbackOnEnginePause func_on_engine_pause;

	// scene
	GDExtensionSpxCallbackOnSceneSpriteInstantiated func_on_scene_sprite_instantiated;
	// sprite
	GDExtensionSpxCallbackOnSpriteReady func_on_sprite_ready;
	GDExtensionSpxCallbackOnSpriteUpdated func_on_sprite_updated;
	GDExtensionSpxCallbackOnSpriteFixedUpdated func_on_sprite_fixed_updated;
	GDExtensionSpxCallbackOnSpriteDestroyed func_on_sprite_destroyed;

	// animation
	GDExtensionSpxCallbackOnSpriteFramesSetChanged func_on_sprite_frames_set_changed;
	GDExtensionSpxCallbackOnSpriteAnimationChanged func_on_sprite_animation_changed;
	GDExtensionSpxCallbackOnSpriteFrameChanged func_on_sprite_frame_changed;
	GDExtensionSpxCallbackOnSpriteAnimationLooped func_on_sprite_animation_looped;
	GDExtensionSpxCallbackOnSpriteAnimationFinished func_on_sprite_animation_finished;
	// vfx
	GDExtensionSpxCallbackOnSpriteVfxFinished func_on_sprite_vfx_finished;
	// visibility
	GDExtensionSpxCallbackOnSpriteScreenExited func_on_sprite_screen_exited;
	GDExtensionSpxCallbackOnSpriteScreenEntered func_on_sprite_screen_entered;

	// input
	GDExtensionSpxCallbackOnMousePressed func_on_mouse_pressed;
	GDExtensionSpxCallbackOnMouseReleased func_on_mouse_released;
	GDExtensionSpxCallbackOnKeyPressed func_on_key_pressed;
	GDExtensionSpxCallbackOnKeyReleased func_on_key_released;
	GDExtensionSpxCallbackOnActionPressed func_on_action_pressed;
	GDExtensionSpxCallbackOnActionJustPressed func_on_action_just_pressed;
	GDExtensionSpxCallbackOnActionJustReleased func_on_action_just_released;
	GDExtensionSpxCallbackOnAxisChanged func_on_axis_changed;

	// physic
	GDExtensionSpxCallbackOnCollisionEnter func_on_collision_enter;
	GDExtensionSpxCallbackOnCollisionStay func_on_collision_stay;
	GDExtensionSpxCallbackOnCollisionExit func_on_collision_exit;
	GDExtensionSpxCallbackOnTriggerEnter func_on_trigger_enter;
	GDExtensionSpxCallbackOnTriggerStay func_on_trigger_stay;
	GDExtensionSpxCallbackOnTriggerExit func_on_trigger_exit;

	// ui
	GDExtensionSpxCallbackOnUiReady func_on_ui_ready;
	GDExtensionSpxCallbackOnUiUpdated func_on_ui_updated;
	GDExtensionSpxCallbackOnUiDestroyed func_on_ui_destroyed;

	GDExtensionSpxCallbackOnUiPressed func_on_ui_pressed;
	GDExtensionSpxCallbackOnUiReleased func_on_ui_released;
	GDExtensionSpxCallbackOnUiHovered func_on_ui_hovered;
	GDExtensionSpxCallbackOnUiClicked func_on_ui_clicked;
	GDExtensionSpxCallbackOnUiToggle func_on_ui_toggle;
	GDExtensionSpxCallbackOnUiTextChanged func_on_ui_text_changed;

	
} SpxCallbackInfo;



// SpxAudio
typedef void (*GDExtensionSpxAudioStopAll)();
typedef void (*GDExtensionSpxAudioCreateAudio)(GdObj* ret_value);
typedef void (*GDExtensionSpxAudioDestroyAudio)(GdObj obj);
typedef void (*GDExtensionSpxAudioSetPitch)(GdObj obj, GdFloat pitch);
typedef void (*GDExtensionSpxAudioGetPitch)(GdObj obj, GdFloat* ret_value);
typedef void (*GDExtensionSpxAudioSetPan)(GdObj obj, GdFloat pan);
typedef void (*GDExtensionSpxAudioGetPan)(GdObj obj, GdFloat* ret_value);
typedef void (*GDExtensionSpxAudioSetVolume)(GdObj obj, GdFloat volume);
typedef void (*GDExtensionSpxAudioGetVolume)(GdObj obj, GdFloat* ret_value);
typedef void (*GDExtensionSpxAudioPlay)(GdObj obj, GdString path, GdInt* ret_value);
typedef void (*GDExtensionSpxAudioPause)(GdInt aid);
typedef void (*GDExtensionSpxAudioResume)(GdInt aid);
typedef void (*GDExtensionSpxAudioStop)(GdInt aid);
typedef void (*GDExtensionSpxAudioSetLoop)(GdInt aid, GdBool loop);
typedef void (*GDExtensionSpxAudioGetLoop)(GdInt aid, GdBool* ret_value);
typedef void (*GDExtensionSpxAudioGetTimer)(GdInt aid, GdFloat* ret_value);
typedef void (*GDExtensionSpxAudioSetTimer)(GdInt aid, GdFloat time);
typedef void (*GDExtensionSpxAudioIsPlaying)(GdInt aid, GdBool* ret_value);
// SpxCamera
typedef void (*GDExtensionSpxCameraGetCameraPosition)(GdVec2* ret_value);
typedef void (*GDExtensionSpxCameraSetCameraPosition)(GdVec2 position);
typedef void (*GDExtensionSpxCameraGetCameraZoom)(GdVec2* ret_value);
typedef void (*GDExtensionSpxCameraSetCameraZoom)(GdVec2 size);
typedef void (*GDExtensionSpxCameraGetViewportRect)(GdRect2* ret_value);
// SpxExt
typedef void (*GDExtensionSpxExtRequestExit)(GdInt exit_code);
typedef void (*GDExtensionSpxExtOnRuntimePanic)(GdString msg);
typedef void (*GDExtensionSpxExtPause)();
typedef void (*GDExtensionSpxExtResume)();
typedef void (*GDExtensionSpxExtIsPaused)(GdBool* ret_value);
typedef void (*GDExtensionSpxExtDestroyAllPens)();
typedef void (*GDExtensionSpxExtCreatePen)(GdObj* ret_value);
typedef void (*GDExtensionSpxExtDestroyPen)(GdObj obj);
typedef void (*GDExtensionSpxExtPenStamp)(GdObj obj);
typedef void (*GDExtensionSpxExtMovePenTo)(GdObj obj, GdVec2 position);
typedef void (*GDExtensionSpxExtPenDown)(GdObj obj, GdBool move_by_mouse);
typedef void (*GDExtensionSpxExtPenUp)(GdObj obj);
typedef void (*GDExtensionSpxExtSetPenColorTo)(GdObj obj, GdColor color);
typedef void (*GDExtensionSpxExtChangePenBy)(GdObj obj, GdInt property, GdFloat amount);
typedef void (*GDExtensionSpxExtSetPenTo)(GdObj obj, GdInt property, GdFloat value);
typedef void (*GDExtensionSpxExtChangePenSizeBy)(GdObj obj, GdFloat amount);
typedef void (*GDExtensionSpxExtSetPenSizeTo)(GdObj obj, GdFloat size);
typedef void (*GDExtensionSpxExtSetPenStampTexture)(GdObj obj, GdString texture_path);
// SpxInput
typedef void (*GDExtensionSpxInputGetMousePos)(GdVec2* ret_value);
typedef void (*GDExtensionSpxInputGetKey)(GdInt key, GdBool* ret_value);
typedef void (*GDExtensionSpxInputGetMouseState)(GdInt mouse_id, GdBool* ret_value);
typedef void (*GDExtensionSpxInputGetKeyState)(GdInt key, GdInt* ret_value);
typedef void (*GDExtensionSpxInputGetAxis)(GdString neg_action,GdString pos_action, GdFloat* ret_value);
typedef void (*GDExtensionSpxInputIsActionPressed)(GdString action, GdBool* ret_value);
typedef void (*GDExtensionSpxInputIsActionJustPressed)(GdString action, GdBool* ret_value);
typedef void (*GDExtensionSpxInputIsActionJustReleased)(GdString action, GdBool* ret_value);
// SpxPhysic
typedef void (*GDExtensionSpxPhysicRaycast)(GdVec2 from, GdVec2 to, GdInt collision_mask, GdObj* ret_value);
typedef void (*GDExtensionSpxPhysicCheckCollision)(GdVec2 from, GdVec2 to, GdInt collision_mask, GdBool collide_with_areas, GdBool collide_with_bodies, GdBool* ret_value);
typedef void (*GDExtensionSpxPhysicCheckTouchedCameraBoundaries)(GdObj obj, GdInt* ret_value);
typedef void (*GDExtensionSpxPhysicCheckTouchedCameraBoundary)(GdObj obj,GdInt board_type, GdBool* ret_value);
typedef void (*GDExtensionSpxPhysicSetCollisionSystemType)(GdBool is_collision_by_alpha);
// SpxPlatform
typedef void (*GDExtensionSpxPlatformSetWindowPosition)(GdVec2 pos);
typedef void (*GDExtensionSpxPlatformGetWindowPosition)(GdVec2* ret_value);
typedef void (*GDExtensionSpxPlatformSetWindowSize)(GdInt width, GdInt height);
typedef void (*GDExtensionSpxPlatformGetWindowSize)(GdVec2* ret_value);
typedef void (*GDExtensionSpxPlatformSetWindowTitle)(GdString title);
typedef void (*GDExtensionSpxPlatformGetWindowTitle)(GdString* ret_value);
typedef void (*GDExtensionSpxPlatformSetWindowFullscreen)(GdBool enable);
typedef void (*GDExtensionSpxPlatformIsWindowFullscreen)(GdBool* ret_value);
typedef void (*GDExtensionSpxPlatformSetDebugMode)(GdBool enable);
typedef void (*GDExtensionSpxPlatformIsDebugMode)(GdBool* ret_value);
typedef void (*GDExtensionSpxPlatformGetTimeScale)(GdFloat* ret_value);
typedef void (*GDExtensionSpxPlatformSetTimeScale)(GdFloat time_scale);
typedef void (*GDExtensionSpxPlatformGetPersistantDataDir)(GdString* ret_value);
typedef void (*GDExtensionSpxPlatformSetPersistantDataDir)(GdString path);
typedef void (*GDExtensionSpxPlatformIsInPersistantDataDir)(GdString path, GdBool* ret_value);
// SpxRes
typedef void (*GDExtensionSpxResCreateAnimation)(GdString sprite_type_name,GdString anim_name, GdString context, GdInt fps, GdBool is_altas);
typedef void (*GDExtensionSpxResSetLoadMode)(GdBool is_direct_mode);
typedef void (*GDExtensionSpxResGetLoadMode)(GdBool* ret_value);
typedef void (*GDExtensionSpxResGetBoundFromAlpha)(GdString p_path, GdRect2* ret_value);
typedef void (*GDExtensionSpxResGetImageSize)(GdString p_path, GdVec2* ret_value);
typedef void (*GDExtensionSpxResReadAllText)(GdString p_path, GdString* ret_value);
typedef void (*GDExtensionSpxResHasFile)(GdString p_path, GdBool* ret_value);
typedef void (*GDExtensionSpxResReloadTexture)(GdString path);
typedef void (*GDExtensionSpxResFreeStr)(GdString str);
typedef void (*GDExtensionSpxResSetDefaultFont)(GdString font_path);
// SpxScene
typedef void (*GDExtensionSpxSceneChangeSceneToFile)(GdString path);
typedef void (*GDExtensionSpxSceneDestroyAllSprites)();
typedef void (*GDExtensionSpxSceneReloadCurrentScene)(GdInt* ret_value);
typedef void (*GDExtensionSpxSceneUnloadCurrentScene)();
// SpxSprite
typedef void (*GDExtensionSpxSpriteSetDontDestroyOnLoad)(GdObj obj);
typedef void (*GDExtensionSpxSpriteSetProcess)(GdObj obj, GdBool is_on);
typedef void (*GDExtensionSpxSpriteSetPhysicProcess)(GdObj obj, GdBool is_on);
typedef void (*GDExtensionSpxSpriteSetTypeName)(GdObj obj,GdString type_name);
typedef void (*GDExtensionSpxSpriteSetChildPosition)(GdObj obj, GdString path, GdVec2 pos);
typedef void (*GDExtensionSpxSpriteGetChildPosition)(GdObj obj, GdString path, GdVec2* ret_value);
typedef void (*GDExtensionSpxSpriteSetChildRotation)(GdObj obj, GdString path, GdFloat rot);
typedef void (*GDExtensionSpxSpriteGetChildRotation)(GdObj obj, GdString path, GdFloat* ret_value);
typedef void (*GDExtensionSpxSpriteSetChildScale)(GdObj obj, GdString path, GdVec2 scale);
typedef void (*GDExtensionSpxSpriteGetChildScale)(GdObj obj, GdString path, GdVec2* ret_value);
typedef void (*GDExtensionSpxSpriteCheckCollision)(GdObj obj,GdObj target, GdBool is_src_trigger,GdBool is_dst_trigger, GdBool* ret_value);
typedef void (*GDExtensionSpxSpriteCheckCollisionWithPoint)(GdObj obj,GdVec2 point, GdBool is_trigger, GdBool* ret_value);
typedef void (*GDExtensionSpxSpriteCreateBackdrop)(GdString path, GdObj* ret_value);
typedef void (*GDExtensionSpxSpriteCreateSprite)(GdString path, GdObj* ret_value);
typedef void (*GDExtensionSpxSpriteCloneSprite)(GdObj obj, GdObj* ret_value);
typedef void (*GDExtensionSpxSpriteDestroySprite)(GdObj obj, GdBool* ret_value);
typedef void (*GDExtensionSpxSpriteIsSpriteAlive)(GdObj obj, GdBool* ret_value);
typedef void (*GDExtensionSpxSpriteSetPosition)(GdObj obj, GdVec2 pos);
typedef void (*GDExtensionSpxSpriteGetPosition)(GdObj obj, GdVec2* ret_value);
typedef void (*GDExtensionSpxSpriteSetRotation)(GdObj obj, GdFloat rot);
typedef void (*GDExtensionSpxSpriteGetRotation)(GdObj obj, GdFloat* ret_value);
typedef void (*GDExtensionSpxSpriteSetScale)(GdObj obj, GdVec2 scale);
typedef void (*GDExtensionSpxSpriteGetScale)(GdObj obj, GdVec2* ret_value);
typedef void (*GDExtensionSpxSpriteSetRenderScale)(GdObj obj, GdVec2 scale);
typedef void (*GDExtensionSpxSpriteGetRenderScale)(GdObj obj, GdVec2* ret_value);
typedef void (*GDExtensionSpxSpriteSetColor)(GdObj obj, GdColor color);
typedef void (*GDExtensionSpxSpriteGetColor)(GdObj obj, GdColor* ret_value);
typedef void (*GDExtensionSpxSpriteSetMaterialShader)(GdObj obj, GdString path);
typedef void (*GDExtensionSpxSpriteGetMaterialShader)(GdObj obj, GdString* ret_value);
typedef void (*GDExtensionSpxSpriteSetMaterialParams)(GdObj obj, GdString effect, GdFloat amount);
typedef void (*GDExtensionSpxSpriteGetMaterialParams)(GdObj obj, GdString effect, GdFloat* ret_value);
typedef void (*GDExtensionSpxSpriteSetMaterialParamsVec)(GdObj obj, GdString effect, GdFloat x, GdFloat y, GdFloat z, GdFloat w);
typedef void (*GDExtensionSpxSpriteSetMaterialParamsVec4)(GdObj obj, GdString effect, GdVec4 vec4);
typedef void (*GDExtensionSpxSpriteGetMaterialParamsVec4)(GdObj obj, GdString effect, GdVec4* ret_value);
typedef void (*GDExtensionSpxSpriteSetMaterialParamsColor)(GdObj obj, GdString effect, GdColor color);
typedef void (*GDExtensionSpxSpriteGetMaterialParamsColor)(GdObj obj, GdString effect, GdColor* ret_value);
typedef void (*GDExtensionSpxSpriteSetTextureAltas)(GdObj obj, GdString path, GdRect2 rect2);
typedef void (*GDExtensionSpxSpriteSetTexture)(GdObj obj, GdString path);
typedef void (*GDExtensionSpxSpriteSetTextureAltasDirect)(GdObj obj, GdString path, GdRect2 rect2);
typedef void (*GDExtensionSpxSpriteSetTextureDirect)(GdObj obj, GdString path);
typedef void (*GDExtensionSpxSpriteGetTexture)(GdObj obj, GdString* ret_value);
typedef void (*GDExtensionSpxSpriteSetVisible)(GdObj obj, GdBool visible);
typedef void (*GDExtensionSpxSpriteGetVisible)(GdObj obj, GdBool* ret_value);
typedef void (*GDExtensionSpxSpriteGetZIndex)(GdObj obj, GdInt* ret_value);
typedef void (*GDExtensionSpxSpriteSetZIndex)(GdObj obj, GdInt z);
typedef void (*GDExtensionSpxSpritePlayAnim)(GdObj obj, GdString p_name , GdFloat p_speed, GdBool isLoop, GdBool p_revert );
typedef void (*GDExtensionSpxSpritePlayBackwardsAnim)(GdObj obj,  GdString p_name );
typedef void (*GDExtensionSpxSpritePauseAnim)(GdObj obj);
typedef void (*GDExtensionSpxSpriteStopAnim)(GdObj obj);
typedef void (*GDExtensionSpxSpriteIsPlayingAnim)(GdObj obj, GdBool* ret_value);
typedef void (*GDExtensionSpxSpriteSetAnim)(GdObj obj, GdString p_name);
typedef void (*GDExtensionSpxSpriteGetAnim)(GdObj obj, GdString* ret_value);
typedef void (*GDExtensionSpxSpriteSetAnimFrame)(GdObj obj, GdInt p_frame);
typedef void (*GDExtensionSpxSpriteGetAnimFrame)(GdObj obj, GdInt* ret_value);
typedef void (*GDExtensionSpxSpriteSetAnimSpeedScale)(GdObj obj, GdFloat p_speed_scale);
typedef void (*GDExtensionSpxSpriteGetAnimSpeedScale)(GdObj obj, GdFloat* ret_value);
typedef void (*GDExtensionSpxSpriteGetAnimPlayingSpeed)(GdObj obj, GdFloat* ret_value);
typedef void (*GDExtensionSpxSpriteSetAnimCentered)(GdObj obj, GdBool p_center);
typedef void (*GDExtensionSpxSpriteIsAnimCentered)(GdObj obj, GdBool* ret_value);
typedef void (*GDExtensionSpxSpriteSetAnimOffset)(GdObj obj, GdVec2 p_offset);
typedef void (*GDExtensionSpxSpriteGetAnimOffset)(GdObj obj, GdVec2* ret_value);
typedef void (*GDExtensionSpxSpriteSetAnimFlipH)(GdObj obj, GdBool p_flip);
typedef void (*GDExtensionSpxSpriteIsAnimFlippedH)(GdObj obj, GdBool* ret_value);
typedef void (*GDExtensionSpxSpriteSetAnimFlipV)(GdObj obj, GdBool p_flip);
typedef void (*GDExtensionSpxSpriteIsAnimFlippedV)(GdObj obj, GdBool* ret_value);
typedef void (*GDExtensionSpxSpriteSetVelocity)(GdObj obj, GdVec2 velocity);
typedef void (*GDExtensionSpxSpriteGetVelocity)(GdObj obj, GdVec2* ret_value);
typedef void (*GDExtensionSpxSpriteIsOnFloor)(GdObj obj, GdBool* ret_value);
typedef void (*GDExtensionSpxSpriteIsOnFloorOnly)(GdObj obj, GdBool* ret_value);
typedef void (*GDExtensionSpxSpriteIsOnWall)(GdObj obj, GdBool* ret_value);
typedef void (*GDExtensionSpxSpriteIsOnWallOnly)(GdObj obj, GdBool* ret_value);
typedef void (*GDExtensionSpxSpriteIsOnCeiling)(GdObj obj, GdBool* ret_value);
typedef void (*GDExtensionSpxSpriteIsOnCeilingOnly)(GdObj obj, GdBool* ret_value);
typedef void (*GDExtensionSpxSpriteGetLastMotion)(GdObj obj, GdVec2* ret_value);
typedef void (*GDExtensionSpxSpriteGetPositionDelta)(GdObj obj, GdVec2* ret_value);
typedef void (*GDExtensionSpxSpriteGetFloorNormal)(GdObj obj, GdVec2* ret_value);
typedef void (*GDExtensionSpxSpriteGetWallNormal)(GdObj obj, GdVec2* ret_value);
typedef void (*GDExtensionSpxSpriteGetRealVelocity)(GdObj obj, GdVec2* ret_value);
typedef void (*GDExtensionSpxSpriteMoveAndSlide)(GdObj obj);
typedef void (*GDExtensionSpxSpriteSetGravity)(GdObj obj, GdFloat gravity);
typedef void (*GDExtensionSpxSpriteGetGravity)(GdObj obj, GdFloat* ret_value);
typedef void (*GDExtensionSpxSpriteSetMass)(GdObj obj, GdFloat mass);
typedef void (*GDExtensionSpxSpriteGetMass)(GdObj obj, GdFloat* ret_value);
typedef void (*GDExtensionSpxSpriteAddForce)(GdObj obj, GdVec2 force);
typedef void (*GDExtensionSpxSpriteAddImpulse)(GdObj obj, GdVec2 impulse);
typedef void (*GDExtensionSpxSpriteSetCollisionLayer)(GdObj obj, GdInt layer);
typedef void (*GDExtensionSpxSpriteGetCollisionLayer)(GdObj obj, GdInt* ret_value);
typedef void (*GDExtensionSpxSpriteSetCollisionMask)(GdObj obj, GdInt mask);
typedef void (*GDExtensionSpxSpriteGetCollisionMask)(GdObj obj, GdInt* ret_value);
typedef void (*GDExtensionSpxSpriteSetTriggerLayer)(GdObj obj, GdInt layer);
typedef void (*GDExtensionSpxSpriteGetTriggerLayer)(GdObj obj, GdInt* ret_value);
typedef void (*GDExtensionSpxSpriteSetTriggerMask)(GdObj obj, GdInt mask);
typedef void (*GDExtensionSpxSpriteGetTriggerMask)(GdObj obj, GdInt* ret_value);
typedef void (*GDExtensionSpxSpriteSetColliderRect)(GdObj obj, GdVec2 center, GdVec2 size);
typedef void (*GDExtensionSpxSpriteSetColliderCircle)(GdObj obj, GdVec2 center, GdFloat radius);
typedef void (*GDExtensionSpxSpriteSetColliderCapsule)(GdObj obj, GdVec2 center, GdVec2 size);
typedef void (*GDExtensionSpxSpriteSetCollisionEnabled)(GdObj obj, GdBool enabled);
typedef void (*GDExtensionSpxSpriteIsCollisionEnabled)(GdObj obj, GdBool* ret_value);
typedef void (*GDExtensionSpxSpriteSetTriggerRect)(GdObj obj, GdVec2 center, GdVec2 size);
typedef void (*GDExtensionSpxSpriteSetTriggerCircle)(GdObj obj, GdVec2 center, GdFloat radius);
typedef void (*GDExtensionSpxSpriteSetTriggerCapsule)(GdObj obj, GdVec2 center, GdVec2 size);
typedef void (*GDExtensionSpxSpriteSetTriggerEnabled)(GdObj obj, GdBool trigger);
typedef void (*GDExtensionSpxSpriteIsTriggerEnabled)(GdObj obj, GdBool* ret_value);
typedef void (*GDExtensionSpxSpriteCheckCollisionByColor)(GdObj obj, GdColor color,GdFloat color_threshold, GdFloat alpha_threshold, GdBool* ret_value);
typedef void (*GDExtensionSpxSpriteCheckCollisionByAlpha)(GdObj obj, GdFloat alpha_threshold, GdBool* ret_value);
typedef void (*GDExtensionSpxSpriteCheckCollisionWithSpriteByAlpha)(GdObj obj, GdObj obj_b, GdFloat alpha_threshold, GdBool* ret_value);
// SpxUi
typedef void (*GDExtensionSpxUiBindNode)(GdObj obj, GdString rel_path, GdObj* ret_value);
typedef void (*GDExtensionSpxUiCreateNode)(GdString path, GdObj* ret_value);
typedef void (*GDExtensionSpxUiCreateButton)(GdString path,GdString text, GdObj* ret_value);
typedef void (*GDExtensionSpxUiCreateLabel)(GdString path, GdString text, GdObj* ret_value);
typedef void (*GDExtensionSpxUiCreateImage)(GdString path, GdObj* ret_value);
typedef void (*GDExtensionSpxUiCreateToggle)(GdString path, GdBool value, GdObj* ret_value);
typedef void (*GDExtensionSpxUiCreateSlider)(GdString path, GdFloat value, GdObj* ret_value);
typedef void (*GDExtensionSpxUiCreateInput)(GdString path, GdString text, GdObj* ret_value);
typedef void (*GDExtensionSpxUiDestroyNode)(GdObj obj, GdBool* ret_value);
typedef void (*GDExtensionSpxUiGetType)(GdObj obj, GdInt* ret_value);
typedef void (*GDExtensionSpxUiSetText)(GdObj obj, GdString text);
typedef void (*GDExtensionSpxUiGetText)(GdObj obj, GdString* ret_value);
typedef void (*GDExtensionSpxUiSetTexture)(GdObj obj, GdString path);
typedef void (*GDExtensionSpxUiGetTexture)(GdObj obj, GdString* ret_value);
typedef void (*GDExtensionSpxUiSetColor)(GdObj obj, GdColor color);
typedef void (*GDExtensionSpxUiGetColor)(GdObj obj, GdColor* ret_value);
typedef void (*GDExtensionSpxUiSetFontSize)(GdObj obj, GdInt size);
typedef void (*GDExtensionSpxUiGetFontSize)(GdObj obj, GdInt* ret_value);
typedef void (*GDExtensionSpxUiSetVisible)(GdObj obj, GdBool visible);
typedef void (*GDExtensionSpxUiGetVisible)(GdObj obj, GdBool* ret_value);
typedef void (*GDExtensionSpxUiSetInteractable)(GdObj obj, GdBool interactable);
typedef void (*GDExtensionSpxUiGetInteractable)(GdObj obj, GdBool* ret_value);
typedef void (*GDExtensionSpxUiSetRect)(GdObj obj, GdRect2 rect);
typedef void (*GDExtensionSpxUiGetRect)(GdObj obj, GdRect2* ret_value);
typedef void (*GDExtensionSpxUiGetLayoutDirection)(GdObj obj, GdInt* ret_value);
typedef void (*GDExtensionSpxUiSetLayoutDirection)(GdObj obj,GdInt value);
typedef void (*GDExtensionSpxUiGetLayoutMode)(GdObj obj, GdInt* ret_value);
typedef void (*GDExtensionSpxUiSetLayoutMode)(GdObj obj,GdInt value);
typedef void (*GDExtensionSpxUiGetAnchorsPreset)(GdObj obj, GdInt* ret_value);
typedef void (*GDExtensionSpxUiSetAnchorsPreset)(GdObj obj,GdInt value);
typedef void (*GDExtensionSpxUiGetScale)(GdObj obj, GdVec2* ret_value);
typedef void (*GDExtensionSpxUiSetScale)(GdObj obj,GdVec2 value);
typedef void (*GDExtensionSpxUiGetPosition)(GdObj obj, GdVec2* ret_value);
typedef void (*GDExtensionSpxUiSetPosition)(GdObj obj,GdVec2 value);
typedef void (*GDExtensionSpxUiGetSize)(GdObj obj, GdVec2* ret_value);
typedef void (*GDExtensionSpxUiSetSize)(GdObj obj,GdVec2 value);
typedef void (*GDExtensionSpxUiGetGlobalPosition)(GdObj obj, GdVec2* ret_value);
typedef void (*GDExtensionSpxUiSetGlobalPosition)(GdObj obj,GdVec2 value);
typedef void (*GDExtensionSpxUiGetRotation)(GdObj obj, GdFloat* ret_value);
typedef void (*GDExtensionSpxUiSetRotation)(GdObj obj,GdFloat value);
typedef void (*GDExtensionSpxUiGetFlip)(GdObj obj,GdBool horizontal, GdBool* ret_value);
typedef void (*GDExtensionSpxUiSetFlip)(GdObj obj,GdBool horizontal, GdBool is_flip);


#ifdef __cplusplus
}
#endif

#endif // GDEXTENSION_SPX_EXT_H
