/**************************************************************************/
/*  GameMenuFragment.kt                                                   */
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

package org.godotengine.editor.embed

import android.content.Context
import android.os.Build
import android.os.Bundle
import android.preference.PreferenceManager
import android.view.LayoutInflater
import android.view.MenuItem
import android.view.View
import android.view.ViewGroup
import android.widget.PopupMenu
import android.widget.RadioButton
import androidx.core.content.edit
import androidx.core.view.isVisible
import androidx.fragment.app.Fragment
import org.godotengine.editor.BaseGodotEditor
import org.godotengine.editor.BaseGodotEditor.Companion.SNACKBAR_SHOW_DURATION_MS
import org.godotengine.editor.R
import org.godotengine.godot.utils.DialogUtils

/**
 * Implements the game menu interface for the Android editor.
 */
class GameMenuFragment : Fragment(), PopupMenu.OnMenuItemClickListener {

	companion object {
		val TAG = GameMenuFragment::class.java.simpleName

		private const val PREF_KEY_ALWAYS_ON_TOP = "pref_key_always_on_top"
		private const val PREF_KEY_DONT_SHOW_RESTART_GAME_HINT = "pref_key_dont_show_restart_game_hint"
		private const val PREF_KEY_GAME_MENU_BAR_COLLAPSED = "pref_key_game_menu_bar_collapsed"
	}

	/**
	 * Used to be notified of events fired when interacting with the game menu.
	 */
	interface GameMenuListener {

		/**
		 * Kotlin representation of the RuntimeNodeSelect::SelectMode enum in 'scene/debugger/scene_debugger.h'.
		 */
		enum class SelectMode {
			SINGLE,
			LIST
		}

		/**
		 * Kotlin representation of the RuntimeNodeSelect::NodeType enum in 'scene/debugger/scene_debugger.h'.
		 */
		enum class NodeType {
			NONE,
			TYPE_2D,
			TYPE_3D
		}

		/**
		 * Kotlin representation of the EditorDebuggerNode::CameraOverride in 'editor/debugger/editor_debugger_node.h'.
		 */
		enum class CameraMode {
			NONE,
			IN_GAME,
			EDITORS
		}

		fun suspendGame(suspended: Boolean)
		fun dispatchNextFrame()
		fun toggleSelectionVisibility(enabled: Boolean)
		fun overrideCamera(enabled: Boolean)
		fun selectRuntimeNode(nodeType: NodeType)
		fun selectRuntimeNodeSelectMode(selectMode: SelectMode)
		fun reset2DCamera()
		fun reset3DCamera()
		fun manipulateCamera(mode: CameraMode)

		fun isGameEmbeddingSupported(): Boolean
		fun embedGameOnPlay(embedded: Boolean)

		fun enterPiPMode() {}
		fun minimizeGameWindow() {}
		fun closeGameWindow() {}

		fun isMinimizedButtonEnabled() = false
		fun isFullScreenButtonEnabled() = false
		fun isCloseButtonEnabled() = false
		fun isPiPButtonEnabled() = false
		fun isMenuBarCollapsable() = false

		fun isAlwaysOnTopSupported() = false

		fun onFullScreenUpdated(enabled: Boolean) {}
		fun onGameMenuCollapsed(collapsed: Boolean) {}
	}

	private val collapseMenuButton: View? by lazy {
		view?.findViewById(R.id.game_menu_collapse_button)
	}
	private val pauseButton: View? by lazy {
		view?.findViewById(R.id.game_menu_pause_button)
	}
	private val nextFrameButton: View? by lazy {
		view?.findViewById(R.id.game_menu_next_frame_button)
	}
	private val unselectNodesButton: RadioButton? by lazy {
		view?.findViewById(R.id.game_menu_unselect_nodes_button)
	}
	private val select2DNodesButton: RadioButton? by lazy {
		view?.findViewById(R.id.game_menu_select_2d_nodes_button)
	}
	private val select3DNodesButton: RadioButton? by lazy {
		view?.findViewById(R.id.game_menu_select_3d_nodes_button)
	}
	private val guiVisibilityButton: View? by lazy {
		view?.findViewById(R.id.game_menu_gui_visibility_button)
	}
	private val toolSelectButton: RadioButton? by lazy {
		view?.findViewById(R.id.game_menu_tool_select_button)
	}
	private val listSelectButton: RadioButton? by lazy {
		view?.findViewById(R.id.game_menu_list_select_button)
	}
	private val optionsButton: View? by lazy {
		view?.findViewById(R.id.game_menu_options_button)
	}
	private val minimizeButton: View? by lazy {
		view?.findViewById(R.id.game_menu_minimize_button)
	}
	private val pipButton: View? by lazy {
		view?.findViewById(R.id.game_menu_pip_button)
	}
	private val fullscreenButton: View? by lazy {
		view?.findViewById(R.id.game_menu_fullscreen_button)
	}
	private val closeButton: View? by lazy {
		view?.findViewById(R.id.game_menu_close_button)
	}

	private val popupMenu: PopupMenu by lazy {
		PopupMenu(context, optionsButton).apply {
			setOnMenuItemClickListener(this@GameMenuFragment)
			inflate(R.menu.options_menu)

			if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
				menu.setGroupDividerEnabled(true)
			}
		}
	}

	private val menuItemActionView: View by lazy {
		View(context)
	}
	private val menuItemActionExpandListener = object: MenuItem.OnActionExpandListener {
		override fun onMenuItemActionExpand(item: MenuItem): Boolean {
			return false
		}

		override fun onMenuItemActionCollapse(item: MenuItem): Boolean {
			return false
		}
	}

	private var menuListener: GameMenuListener? = null
	private var alwaysOnTopChecked = false
	private var isGameEmbedded = false
	private var isGameRunning = false

	override fun onAttach(context: Context) {
		super.onAttach(context)
		val parentActivity = activity
		if (parentActivity is GameMenuListener) {
			menuListener = parentActivity
		} else {
			val parentFragment = parentFragment
			if (parentFragment is GameMenuListener) {
				menuListener = parentFragment
			}
		}
	}

	override fun onDetach() {
		super.onDetach()
		menuListener = null
	}

	override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, bundle: Bundle?): View? {
		return inflater.inflate(R.layout.game_menu_fragment_layout, container, false)
	}

	override fun onViewCreated(view: View, bundle: Bundle?) {
		super.onViewCreated(view, bundle)

		val isMinimizeButtonEnabled = menuListener?.isMinimizedButtonEnabled() == true
		val isFullScreenButtonEnabled = menuListener?.isFullScreenButtonEnabled() == true
		val isCloseButtonEnabled = menuListener?.isCloseButtonEnabled() == true
		val isPiPButtonEnabled = menuListener?.isPiPButtonEnabled() == true
		val isMenuBarCollapsable = menuListener?.isMenuBarCollapsable() == true

		// Show the divider if any of the window controls is visible
		view.findViewById<View>(R.id.game_menu_window_controls_divider)?.isVisible =
			isMinimizeButtonEnabled ||
				isFullScreenButtonEnabled ||
				isCloseButtonEnabled ||
				isPiPButtonEnabled ||
				isMenuBarCollapsable

		collapseMenuButton?.apply {
			isVisible = isMenuBarCollapsable
			setOnClickListener {
				collapseGameMenu()
			}
		}
		fullscreenButton?.apply{
			isVisible = isFullScreenButtonEnabled
			setOnClickListener {
				it.isActivated = !it.isActivated
				menuListener?.onFullScreenUpdated(it.isActivated)
			}
		}
		pipButton?.apply {
			isVisible = isPiPButtonEnabled
			setOnClickListener {
				menuListener?.enterPiPMode()
			}
		}
		minimizeButton?.apply {
			isVisible = isMinimizeButtonEnabled
			setOnClickListener {
				menuListener?.minimizeGameWindow()
			}
		}
		closeButton?.apply{
			isVisible = isCloseButtonEnabled
			setOnClickListener {
				menuListener?.closeGameWindow()
			}
		}
		pauseButton?.apply {
			setOnClickListener {
				val isActivated = !it.isActivated
				menuListener?.suspendGame(isActivated)
				it.isActivated = isActivated
			}
		}
		nextFrameButton?.apply {
			setOnClickListener {
				menuListener?.dispatchNextFrame()
			}
		}

		unselectNodesButton?.apply{
			setOnCheckedChangeListener { buttonView, isChecked ->
				if (isChecked) {
					menuListener?.selectRuntimeNode(GameMenuListener.NodeType.NONE)
				}
			}
		}
		select2DNodesButton?.apply{
			setOnCheckedChangeListener { buttonView, isChecked ->
				if (isChecked) {
					menuListener?.selectRuntimeNode(GameMenuListener.NodeType.TYPE_2D)
				}
			}
		}
		select3DNodesButton?.apply{
			setOnCheckedChangeListener { buttonView, isChecked ->
				if (isChecked) {
					menuListener?.selectRuntimeNode(GameMenuListener.NodeType.TYPE_3D)
				}
			}
		}
		guiVisibilityButton?.apply{
			setOnClickListener {
				val isActivated = !it.isActivated
				menuListener?.toggleSelectionVisibility(!isActivated)
				it.isActivated = isActivated
			}
		}

		toolSelectButton?.apply{
			setOnCheckedChangeListener { buttonView, isChecked ->
				if (isChecked) {
					menuListener?.selectRuntimeNodeSelectMode(GameMenuListener.SelectMode.SINGLE)
				}
			}
		}
		listSelectButton?.apply{
			setOnCheckedChangeListener { buttonView, isChecked ->
				if (isChecked) {
					menuListener?.selectRuntimeNodeSelectMode(GameMenuListener.SelectMode.LIST)
				}
			}
		}
		optionsButton?.setOnClickListener {
			popupMenu.show()
		}

		refreshGameMenu(arguments?.getBundle(BaseGodotEditor.EXTRA_GAME_MENU_STATE) ?: Bundle())
	}

	internal fun refreshGameMenu(gameMenuState: Bundle) {
		val sharedPrefs = PreferenceManager.getDefaultSharedPreferences(context)
		if (menuListener?.isMenuBarCollapsable() == true) {
			val collapsed = sharedPrefs.getBoolean(PREF_KEY_GAME_MENU_BAR_COLLAPSED, false)
			view?.isVisible = !collapsed
			menuListener?.onGameMenuCollapsed(collapsed)
		}
		alwaysOnTopChecked = sharedPrefs.getBoolean(PREF_KEY_ALWAYS_ON_TOP, false)
		isGameEmbedded = gameMenuState.getBoolean(BaseGodotEditor.EXTRA_IS_GAME_EMBEDDED, false)
		isGameRunning = gameMenuState.getBoolean(BaseGodotEditor.EXTRA_IS_GAME_RUNNING, false)

		pauseButton?.isEnabled = isGameRunning
		nextFrameButton?.isEnabled = isGameRunning

		val nodeType = gameMenuState.getSerializable(BaseGodotEditor.GAME_MENU_ACTION_SET_NODE_TYPE) as GameMenuListener.NodeType? ?: GameMenuListener.NodeType.NONE
		unselectNodesButton?.isChecked = nodeType == GameMenuListener.NodeType.NONE
		select2DNodesButton?.isChecked = nodeType == GameMenuListener.NodeType.TYPE_2D
		select3DNodesButton?.isChecked = nodeType == GameMenuListener.NodeType.TYPE_3D

		guiVisibilityButton?.isActivated = !gameMenuState.getBoolean(BaseGodotEditor.GAME_MENU_ACTION_SET_SELECTION_VISIBLE, true)

		val selectMode = gameMenuState.getSerializable(BaseGodotEditor.GAME_MENU_ACTION_SET_SELECT_MODE) as GameMenuListener.SelectMode? ?: GameMenuListener.SelectMode.SINGLE
		toolSelectButton?.isChecked = selectMode == GameMenuListener.SelectMode.SINGLE
		listSelectButton?.isChecked = selectMode == GameMenuListener.SelectMode.LIST

		popupMenu.menu.apply {
			if (menuListener?.isGameEmbeddingSupported() == false) {
				setGroupEnabled(R.id.group_menu_embed_options, false)
				setGroupVisible(R.id.group_menu_embed_options, false)
			} else {
				findItem(R.id.menu_embed_game_on_play)?.isChecked = isGameEmbedded

				val keepOnTopMenuItem = findItem(R.id.menu_embed_game_keep_on_top)
				if (menuListener?.isAlwaysOnTopSupported() == false) {
					keepOnTopMenuItem?.isVisible = false
				} else {
					keepOnTopMenuItem?.isEnabled = isGameEmbedded
				}
			}

			setGroupEnabled(R.id.group_menu_camera_options, isGameRunning)
			setGroupVisible(R.id.group_menu_camera_options, isGameRunning)
			findItem(R.id.menu_camera_options)?.isEnabled = false

			findItem(R.id.menu_embed_game_keep_on_top)?.isChecked = alwaysOnTopChecked

			val cameraMode = gameMenuState.getSerializable(BaseGodotEditor.GAME_MENU_ACTION_SET_CAMERA_MANIPULATE_MODE) as GameMenuListener.CameraMode? ?: GameMenuListener.CameraMode.NONE
			if (cameraMode == GameMenuListener.CameraMode.IN_GAME || cameraMode == GameMenuListener.CameraMode.NONE) {
				findItem(R.id.menu_manipulate_camera_in_game)?.isChecked = true
			} else {
				findItem(R.id.menu_manipulate_camera_from_editors)?.isChecked = true
			}
		}
	}

	internal fun isAlwaysOnTop() = isGameEmbedded && alwaysOnTopChecked

	private fun collapseGameMenu() {
		view?.isVisible = false
		PreferenceManager.getDefaultSharedPreferences(context).edit {
			putBoolean(PREF_KEY_GAME_MENU_BAR_COLLAPSED, true)
		}
		menuListener?.onGameMenuCollapsed(true)
	}

	internal fun expandGameMenu() {
		view?.isVisible = true
		PreferenceManager.getDefaultSharedPreferences(context).edit {
			putBoolean(PREF_KEY_GAME_MENU_BAR_COLLAPSED, false)
		}
		menuListener?.onGameMenuCollapsed(false)
	}

	private fun updateAlwaysOnTop(enabled: Boolean) {
		alwaysOnTopChecked = enabled
		PreferenceManager.getDefaultSharedPreferences(context).edit {
			putBoolean(PREF_KEY_ALWAYS_ON_TOP, enabled)
		}
	}

	private fun preventMenuItemCollapse(item: MenuItem) {
		item.setShowAsAction(MenuItem.SHOW_AS_ACTION_COLLAPSE_ACTION_VIEW)
		item.setActionView(menuItemActionView)
		item.setOnActionExpandListener(menuItemActionExpandListener)
	}

	override fun onMenuItemClick(item: MenuItem): Boolean {
		if (!item.hasSubMenu()) {
			preventMenuItemCollapse(item)
		}

		when(item.itemId) {
			R.id.menu_embed_game_on_play -> {
				item.isChecked = !item.isChecked
				menuListener?.embedGameOnPlay(item.isChecked)

				if (item.isChecked != isGameEmbedded && isGameRunning) {
					activity?.let {
						val sharedPrefs = PreferenceManager.getDefaultSharedPreferences(context)
						if (!sharedPrefs.getBoolean(PREF_KEY_DONT_SHOW_RESTART_GAME_HINT, false)) {
							DialogUtils.showSnackbar(
								it,
								if (item.isChecked) getString(R.string.restart_embed_game_hint) else getString(R.string.restart_non_embedded_game_hint),
								SNACKBAR_SHOW_DURATION_MS,
								getString(R.string.dont_show_again_message)
							) {
								sharedPrefs.edit {
									putBoolean(PREF_KEY_DONT_SHOW_RESTART_GAME_HINT, true)
								}
							}
						}
					}
				}
			}

			R.id.menu_embed_game_keep_on_top -> {
				item.isChecked = !item.isChecked
				updateAlwaysOnTop(item.isChecked)
			}

			R.id.menu_camera_override -> {
				item.isChecked = !item.isChecked
				menuListener?.overrideCamera(item.isChecked)

				popupMenu.menu.findItem(R.id.menu_camera_options)?.isEnabled = item.isChecked
			}

			R.id.menu_reset_2d_camera -> {
				menuListener?.reset2DCamera()
			}

			R.id.menu_reset_3d_camera -> {
				menuListener?.reset3DCamera()
			}

			R.id.menu_manipulate_camera_in_game -> {
				if (!item.isChecked) {
					item.isChecked = true
					menuListener?.manipulateCamera(GameMenuListener.CameraMode.IN_GAME)
				}
			}

			R.id.menu_manipulate_camera_from_editors -> {
				if (!item.isChecked) {
					item.isChecked = true
					menuListener?.manipulateCamera(GameMenuListener.CameraMode.EDITORS)
				}
			}
		}
		return false
	}
}
