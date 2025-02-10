/**************************************************************************/
/*  DialogUtils.kt                                                        */
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

package org.godotengine.godot.utils

import android.app.Activity
import android.app.AlertDialog
import android.content.DialogInterface
import android.os.Handler
import android.os.Looper
import android.view.Gravity
import android.view.LayoutInflater
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.PopupWindow
import android.widget.TextView

import org.godotengine.godot.R
import kotlin.math.abs

/**
 * Utility class for managing dialogs.
 */
internal class DialogUtils {
	companion object {
		private val TAG = DialogUtils::class.java.simpleName

		/**
		 * Invoked on dialog button press.
		 */
		@JvmStatic
		private external fun dialogCallback(buttonIndex: Int)

		/**
		 * Invoked on the input dialog submitted.
		 */
		@JvmStatic
		private external fun inputDialogCallback(text: String)

		/**
		 * Displays a dialog with dynamically arranged buttons based on their text length.
		 *
		 * The buttons are laid out in rows, with a maximum of 2 buttons per row. If a button's text
		 * is too long to fit within half the screen width, it occupies the entire row.
		 *
		 * @param activity The activity where the dialog will be displayed.
		 * @param title The title of the dialog.
		 * @param message The message displayed in the dialog.
		 * @param buttons An array of button labels to display.
		 */
		fun showDialog(activity: Activity, title: String, message: String, buttons: Array<String>) {
			var dismissDialog: () -> Unit = {} // Helper to dismiss the Dialog when a button is clicked.
			activity.runOnUiThread {
				val builder = AlertDialog.Builder(activity)
				builder.setTitle(title)
				builder.setMessage(message)

				val buttonHeight = activity.resources.getDimensionPixelSize(R.dimen.button_height)
				val paddingHorizontal = activity.resources.getDimensionPixelSize(R.dimen.dialog_padding_horizontal)
				val paddingVertical = activity.resources.getDimensionPixelSize(R.dimen.dialog_padding_vertical)
				val buttonPadding = activity.resources.getDimensionPixelSize(R.dimen.button_padding)

				// Create a vertical parent layout to hold all rows of buttons.
				val parentLayout = LinearLayout(activity)
				parentLayout.orientation = LinearLayout.VERTICAL
				parentLayout.setPadding(paddingHorizontal, paddingVertical, paddingHorizontal, paddingVertical)

				// Horizontal row layout for arranging buttons.
				var rowLayout = LinearLayout(activity)
				rowLayout.orientation = LinearLayout.HORIZONTAL
				rowLayout.layoutParams = LinearLayout.LayoutParams(
					LinearLayout.LayoutParams.MATCH_PARENT,
					LinearLayout.LayoutParams.WRAP_CONTENT
				)

				// Calculate the maximum width for a button to allow two buttons per row.
				val screenWidth = activity.resources.displayMetrics.widthPixels
				val availableWidth = screenWidth - (2 * paddingHorizontal)
				val maxButtonWidth = availableWidth / 2

				buttons.forEachIndexed { index, buttonLabel ->
					val button = Button(activity)
					button.text = buttonLabel
					button.isSingleLine = true
					button.setPadding(buttonPadding, buttonPadding, buttonPadding, buttonPadding)

					// Measure the button to determine its width.
					button.measure(0, 0)
					val buttonWidth = button.measuredWidth

					val params = LinearLayout.LayoutParams(
						if (buttonWidth > maxButtonWidth) LinearLayout.LayoutParams.MATCH_PARENT else 0,
						buttonHeight
					)
					params.weight = if (buttonWidth > maxButtonWidth) 0f else 1f
					button.layoutParams = params

					// Handle full-width buttons by finalizing the current row, if needed.
					if (buttonWidth > maxButtonWidth) {
						if (rowLayout.childCount > 0) {
							parentLayout.addView(rowLayout)
							rowLayout = LinearLayout(activity)
							rowLayout.orientation = LinearLayout.HORIZONTAL
						}
						// Add the full-width button directly to the parent layout.
						parentLayout.addView(button)
					} else {
						rowLayout.addView(button)

						// Finalize the row if it reaches 2 buttons.
						if (rowLayout.childCount == 2) {
							parentLayout.addView(rowLayout)
							rowLayout = LinearLayout(activity)
							rowLayout.orientation = LinearLayout.HORIZONTAL
						}

						// Handle the last button with incomplete row.
						if (index == buttons.size - 1 && rowLayout.childCount > 0) {
							parentLayout.addView(rowLayout)
						}
					}

					button.setOnClickListener {
						dialogCallback(index)
						dismissDialog()
					}
				}

				// Attach the parent layout to the dialog.
				builder.setView(parentLayout)
				val dialog = builder.create()
				dismissDialog = {dialog.dismiss()}
				dialog.setCancelable(false)
				dialog.show()
			}
		}

		/**
		 * This method shows a dialog with a text input field, allowing the user to input text.
		 *
		 * @param activity The activity where the input dialog will be displayed.
		 * @param title The title of the input dialog.
		 * @param message The message displayed in the input dialog.
		 * @param existingText The existing text that will be pre-filled in the input field.
		 */
		fun showInputDialog(activity: Activity, title: String, message: String, existingText: String) {
			val inputField = EditText(activity)
			val paddingHorizontal = activity.resources.getDimensionPixelSize(R.dimen.dialog_padding_horizontal)
			val paddingVertical = activity.resources.getDimensionPixelSize(R.dimen.dialog_padding_vertical)
			inputField.setPadding(paddingHorizontal, paddingVertical, paddingHorizontal, paddingVertical)
			inputField.setText(existingText)
			activity.runOnUiThread {
				val builder = AlertDialog.Builder(activity)
				builder.setMessage(message).setTitle(title).setView(inputField)
				builder.setPositiveButton(R.string.dialog_ok) {
						dialog: DialogInterface, id: Int ->
					inputDialogCallback(inputField.text.toString())
					dialog.dismiss()
				}
				val dialog = builder.create()
				dialog.setCancelable(false)
				dialog.show()
			}
		}

		/**
		 * Displays a Snackbar with an optional action button.
		 *
		 * @param context The Context in which the Snackbar should be displayed.
		 * @param message The message to display in the Snackbar.
		 * @param duration The duration for which the Snackbar should be visible (in milliseconds).
		 * @param actionText (Optional) The text for the action button. If `null`, the button is hidden.
		 * @param actionCallback (Optional) A callback function to execute when the action button is clicked. If `null`, no action is performed.
		 */
		fun showSnackbar(activity: Activity, message: String, duration: Long = 3000, actionText: String? = null, action: (() -> Unit)? = null) {
			activity.runOnUiThread {
				val bottomMargin = activity.resources.getDimensionPixelSize(R.dimen.snackbar_bottom_margin)
				val inflater = LayoutInflater.from(activity)
				val customView = inflater.inflate(R.layout.snackbar, null)

				val popupWindow = PopupWindow(
					customView,
					ViewGroup.LayoutParams.MATCH_PARENT,
					ViewGroup.LayoutParams.WRAP_CONTENT,
				)

				val messageView = customView.findViewById<TextView>(R.id.snackbar_text)
				messageView.text = message

				val actionButton = customView.findViewById<Button>(R.id.snackbar_action)

				if (actionText != null && action != null) {
					actionButton.text = actionText
					actionButton.visibility = View.VISIBLE
					actionButton.setOnClickListener {
						action.invoke()
						popupWindow.dismiss()
					}
				} else {
					actionButton.visibility = View.GONE
				}

				addSwipeToDismiss(customView, popupWindow)
				popupWindow.showAtLocation(customView, Gravity.BOTTOM, 0, bottomMargin)

				Handler(Looper.getMainLooper()).postDelayed({ popupWindow.dismiss() }, duration)
			}
		}

		private fun addSwipeToDismiss(view: View, popupWindow: PopupWindow) {
			view.setOnTouchListener(object : View.OnTouchListener {
				private var initialX = 0f
				private var dX = 0f
				private val threshold = 300f  // Swipe distance threshold.

				override fun onTouch(v: View?, event: MotionEvent): Boolean {
					when (event.action) {
						MotionEvent.ACTION_DOWN -> {
							initialX = event.rawX
							dX = view.translationX
						}

						MotionEvent.ACTION_MOVE -> {
							val deltaX = event.rawX - initialX
							view.translationX = dX + deltaX
						}

						MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
							val finalX = event.rawX - initialX

							if (abs(finalX) > threshold) {
								// If swipe exceeds threshold, dismiss smoothly.
								view.animate()
									.translationX(if (finalX > 0) view.width.toFloat() else -view.width.toFloat())
									.setDuration(200)
									.withEndAction { popupWindow.dismiss() }
									.start()
							} else {
								// If swipe is canceled, return smoothly.
								view.animate()
									.translationX(0f)
									.setDuration(200)
									.start()
							}
						}
					}
					return true
				}
			})
		}
	}
}
