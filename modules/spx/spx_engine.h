/**************************************************************************/
/*  spx_engine.h                                                          */
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

#ifndef SPX_ENGINE_H
#define SPX_ENGINE_H

#include "gdextension_spx_ext.h"
#include "spx_base_mgr.h"

#include <cstdint>
class SceneTree;
class Window;
class Node;
class SpxInputMgr;
class SpxAudioMgr;
class SpxPhysicMgr;
class SpxSpriteMgr;
class SpxUiMgr;
class SpxSceneMgr;
class SpxCameraMgr;
class SpxPlatformMgr;
class SpxResMgr;
class SpxExtMgr;

typedef void (*GDExtensionSpxGlobalRuntimePanicCallback)(GdString msg);
typedef void (*GDExtensionSpxGlobalRuntimeExitCallback)(GdInt code);

class SpxEngine : SpxBaseMgr {
	static SpxEngine *singleton;

public:
	static SpxEngine *get_singleton() { return singleton; }
	static bool has_initialed() { return singleton != nullptr; }
	static void register_callbacks(GDExtensionSpxCallbackInfoPtr callback);
	static void register_runtime_panic_callbacks(GDExtensionSpxGlobalRuntimePanicCallback callback);
	static void register_runtime_exit_callbacks(GDExtensionSpxGlobalRuntimeExitCallback callback);
	virtual ~SpxEngine() = default;

private:
	Vector<SpxBaseMgr *> mgrs;
	SpxInputMgr *input;
	SpxAudioMgr *audio;
	SpxPhysicMgr *physic;
	SpxSpriteMgr *sprite;
	SpxUiMgr *ui;
	SpxSceneMgr *scene;
	SpxCameraMgr *camera;
	SpxPlatformMgr *platform;
	SpxResMgr *res;
	SpxExtMgr *ext;

public:
	SpxInputMgr *get_input() { return input; }
	SpxAudioMgr *get_audio() { return audio; }
	SpxPhysicMgr *get_physic() { return physic; }
	SpxSpriteMgr *get_sprite() { return sprite; }
	SpxUiMgr *get_ui() { return ui; }
	SpxSceneMgr *get_scene() { return scene; }
	SpxCameraMgr *get_camera() { return camera; }
	SpxPlatformMgr *get_platform() { return platform; }
	SpxResMgr *get_res() { return res; }
	SpxExtMgr *get_ext() { return ext; }

private:
	SceneTree *tree;
	Node *spx_root;
	GdInt global_id;
	SpxCallbackInfo callbacks;
	GDExtensionSpxGlobalRuntimePanicCallback on_runtime_panic;
	GDExtensionSpxGlobalRuntimeExitCallback on_runtime_exit;
	bool has_exit;
	bool is_spx_paused;
	bool is_defer_call_pause;
	bool defer_pause_value;
public:
	SpxCallbackInfo *get_callbacks() ;
	GDExtensionSpxGlobalRuntimePanicCallback get_on_runtime_panic() { return on_runtime_panic; }
	GDExtensionSpxGlobalRuntimeExitCallback get_on_runtime_exit() { return on_runtime_exit; }

public:
	GdInt get_unique_id() override;
	Node *get_spx_root() override;
	SceneTree *get_tree() override;
	Window *get_root() override;
	void set_root_node(SceneTree *p_tree, Node *p_node);

	void on_awake() override;
	void on_fixed_update(float delta) override;
	void on_update(float delta) override;
	void on_destroy() override;
	void on_exit(int exit_code) override;

	// SPX Pause functionality - simplified interface
	void pause();
	void resume();
	bool is_paused() const;
	
	// Internal methods for Godot pause synchronization
	void _on_godot_pause_changed(bool is_godot_paused);
};

#endif // SPX_ENGINE_H
