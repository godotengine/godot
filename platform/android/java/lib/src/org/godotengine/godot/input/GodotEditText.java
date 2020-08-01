/*************************************************************************/
/*  GodotEditText.java                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

package org.godotengine.godot.input;

import org.godotengine.godot.*;

import android.content.Context;
import android.graphics.Point;
import android.graphics.Rect;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.os.ResultReceiver;
import android.text.InputFilter;
import android.text.InputType;
import android.util.AttributeSet;
import android.view.Gravity;
import android.view.KeyEvent;
import android.view.View;
import android.view.ViewGroup;
import android.view.ViewGroup.LayoutParams;
import android.view.ViewTreeObserver;
import android.view.ViewTreeObserver.OnGlobalLayoutListener;
import android.view.WindowManager;
import android.view.inputmethod.EditorInfo;
import android.view.inputmethod.InputMethodManager;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.PopupWindow;

import java.lang.ref.WeakReference;

public class GodotEditText extends EditText {
	// ===========================================================
	// Constants
	// ===========================================================
	private final static int HANDLER_OPEN_IME_KEYBOARD = 2;
	private final static int HANDLER_CLOSE_IME_KEYBOARD = 3;

	// ===========================================================
	// Fields
	// ===========================================================
	private GodotView mView;
	private KeyboardPopup mKeyboardPopup;
	private boolean mHideKeyboardRequested = false;
	private GodotTextInputWrapper mInputWrapper;
	private EditHandler sHandler = new EditHandler(this);
	private String mOriginText;
	private int mMaxInputLength = Integer.MAX_VALUE;
	private boolean mMultiline = false;

	private static class EditHandler extends Handler {
		private final WeakReference<GodotEditText> mEdit;
		public EditHandler(GodotEditText edit) {
			mEdit = new WeakReference<>(edit);
		}

		@Override
		public void handleMessage(Message msg) {
			GodotEditText edit = mEdit.get();
			if (edit != null) {
				edit.handleMessage(msg);
			}
		}
	}

	private class KeyboardPopup extends PopupWindow implements OnGlobalLayoutListener {
		private GodotEditText mEditText;
		private ViewGroup mLayout;
		private boolean mShowingKeyboard = false;
		private boolean mIsPortrait = false;
		private int mKeyboardHeight = 0;

		public KeyboardPopup(final Godot context, final GodotEditText editText) {
			super(context);

			mEditText = editText;

			setWidth(0);
			setHeight(LayoutParams.MATCH_PARENT);

			setSoftInputMode(WindowManager.LayoutParams.SOFT_INPUT_ADJUST_RESIZE);
			setInputMethodMode(PopupWindow.INPUT_METHOD_NEEDED);
			setFocusable(true); // for the text edit to work
			setTouchable(false); // inputs need to go through
		}

		public void show(final View parentView) {
			// Create a popup window with an invisible layout for the virtual keyboard,
			// so the view can be resized to get the vk height without resizing the main godot view.
			final FrameLayout keyboardLayout = new FrameLayout(getContext());
			keyboardLayout.setLayoutParams(new LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT));
			keyboardLayout.setVisibility(View.INVISIBLE);
			mLayout = keyboardLayout;

			setContentView(mLayout);

			mLayout.addView(mEditText);
			getContentView().getViewTreeObserver().addOnGlobalLayoutListener(this);
			showAtLocation(parentView, Gravity.NO_GRAVITY, 0, 0);
		}

		public void hide() {
			mLayout.removeView(mEditText);
			getContentView().getViewTreeObserver().removeOnGlobalLayoutListener(this);
			dismiss();
			mEditText.onKeyboardPopupClosed();

			GodotLib.setVirtualKeyboardHeight(0);
		}

		@Override
		public void onGlobalLayout() {
			Godot godotActivity = (Godot)getContext();

			// Use size first to calculate the exact height of the keyboard alone.
			Point screenSize = new Point();
			godotActivity.getWindowManager().getDefaultDisplay().getSize(screenSize);

			View keyboardView = getContentView();
			Rect gameSize = new Rect();
			keyboardView.getWindowVisibleDisplayFrame(gameSize);

			int keyboardHeight = screenSize.y - gameSize.bottom;

			final int previousHeight = mKeyboardHeight;
			mKeyboardHeight = keyboardHeight;

			// Adjust final height from real size to take other decorations into account
			// like the navigation bar in portrait mode.
			if (godotActivity.isImmersiveUsed()) {
				Point realScreenSize = new Point();
				godotActivity.getWindowManager().getDefaultDisplay().getRealSize(realScreenSize);
				keyboardHeight = realScreenSize.y - gameSize.bottom;
			}

			GodotLib.setVirtualKeyboardHeight(keyboardHeight);

			// Detect orientation changes to avoid closing the keyboard.
			boolean wasPortrait = mIsPortrait;
			//mIsPortrait = gameSize.right < gameSize.bottom;
			mIsPortrait = screenSize.x < screenSize.y;
			if (mIsPortrait != wasPortrait) {
				mShowingKeyboard = false;
				mKeyboardHeight = 0;
			}

			if (mShowingKeyboard) {
				// Wait for the keyboard to be fully initialized
				// to avoid false positive detection during transition from closing state to re-open.
				if ((mKeyboardHeight == 0) && (previousHeight > 0)) {
					// Keyboard has been hidden by user, close popup.
					hide();
				}
			} else {
				// Popup has started, initialize keyboard
				final InputMethodManager imm = (InputMethodManager)getContext().getSystemService(Context.INPUT_METHOD_SERVICE);
				imm.showSoftInput(mEditText, 0, new ResultReceiver(null) {
					@Override
					protected void onReceiveResult(int resultCode, Bundle resultData) {
						if (resultCode == InputMethodManager.RESULT_SHOWN || resultCode == InputMethodManager.RESULT_UNCHANGED_SHOWN) {
							// Keyboard showing success, can be hidden by user.
							mShowingKeyboard = true;
						}
					}
				});
			}
		}
	}

	// ===========================================================
	// Constructors
	// ===========================================================
	public GodotEditText(final Godot context, final GodotView view) {
		super(context);

		setPadding(0, 0, 0, 0);
		setImeOptions(EditorInfo.IME_FLAG_NO_EXTRACT_UI | EditorInfo.IME_ACTION_DONE);
		setLayoutParams(new LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT));

		mView = view;
		mInputWrapper = new GodotTextInputWrapper(mView, this);
		setOnEditorActionListener(mInputWrapper);
		view.requestFocus();
	}

	public void showKeyboardPopup() {
		mKeyboardPopup = new KeyboardPopup((Godot)getContext(), this);
		mKeyboardPopup.show(mView);
	}

	public void hideKeyboardPopup() {
		if (mKeyboardPopup != null) {
			mKeyboardPopup.hide();
		}
	}

	public void onKeyboardPopupClosed() {
		mKeyboardPopup = null;
		mView.requestFocus();

		mHideKeyboardRequested = false;
	}

	public boolean isMultiline() {
		return mMultiline;
	}

	private void handleMessage(final Message msg) {
		switch (msg.what) {
			case HANDLER_OPEN_IME_KEYBOARD: {
				if (!mHideKeyboardRequested) {
					GodotEditText edit = (GodotEditText)msg.obj;
					String text = edit.mOriginText;

					// Start keyboard popup if not already showing.
					if (mKeyboardPopup == null) {
						showKeyboardPopup();
					}

					if (edit.requestFocus()) {
						edit.removeTextChangedListener(edit.mInputWrapper);
						setMaxInputLength(edit);
						edit.setText("");
						edit.append(text);
						if (msg.arg2 != -1) {
							edit.setSelection(msg.arg1, msg.arg2);
							edit.mInputWrapper.setSelection(true);
						} else {
							edit.mInputWrapper.setSelection(false);
						}

						int inputType = InputType.TYPE_CLASS_TEXT;
						if (edit.isMultiline()) {
							inputType |= InputType.TYPE_TEXT_FLAG_MULTI_LINE;
						}
						edit.setInputType(inputType);

						edit.mInputWrapper.setOriginText(text);
						edit.addTextChangedListener(edit.mInputWrapper);
					}
				}
			} break;

			case HANDLER_CLOSE_IME_KEYBOARD: {
				GodotEditText edit = (GodotEditText)msg.obj;

				edit.removeTextChangedListener(mInputWrapper);

				edit.mView.requestFocus();

				// Force close keyboard popup right away.
				hideKeyboardPopup();

				mHideKeyboardRequested = false;
			} break;
		}
	}

	private void setMaxInputLength(EditText p_edit_text) {
		InputFilter[] filters = new InputFilter[1];
		filters[0] = new InputFilter.LengthFilter(this.mMaxInputLength);
		p_edit_text.setFilters(filters);
	}

	// ===========================================================
	// Methods for/from SuperClass/Interfaces
	// ===========================================================
	@Override
	public boolean onKeyDown(final int keyCode, final KeyEvent keyEvent) {
		super.onKeyDown(keyCode, keyEvent);

		/* Let GlSurfaceView get focus if back key is input. */
		if (keyCode == KeyEvent.KEYCODE_BACK) {
			this.mView.requestFocus();
		}

		return true;
	}

	// ===========================================================
	// Methods
	// ===========================================================
	public void showKeyboard(String p_existing_text, boolean p_multiline, int p_max_input_length, int p_cursor_start, int p_cursor_end) {
		int maxInputLength = (p_max_input_length <= 0) ? Integer.MAX_VALUE : p_max_input_length;
		if (p_cursor_start == -1) { // cursor position not given
			this.mOriginText = p_existing_text;
			this.mMaxInputLength = maxInputLength;
		} else if (p_cursor_end == -1) { // not text selection
			this.mOriginText = p_existing_text.substring(0, p_cursor_start);
			this.mMaxInputLength = maxInputLength - (p_existing_text.length() - p_cursor_start);
		} else {
			this.mOriginText = p_existing_text.substring(0, p_cursor_end);
			this.mMaxInputLength = maxInputLength - (p_existing_text.length() - p_cursor_end);
		}

		this.mMultiline = p_multiline;

		final Message msg = new Message();
		msg.what = HANDLER_OPEN_IME_KEYBOARD;
		msg.obj = this;
		msg.arg1 = p_cursor_start;
		msg.arg2 = p_cursor_end;
		sHandler.sendMessage(msg);

		mHideKeyboardRequested = false;
	}

	public void hideKeyboard() {
		mHideKeyboardRequested = true;
	}

	public void onGodotMainLoopStep() {
		if (mHideKeyboardRequested) {
			final Message msg = new Message();
			msg.what = HANDLER_CLOSE_IME_KEYBOARD;
			msg.obj = this;
			sHandler.sendMessage(msg);
		}
	}

	// ===========================================================
	// Inner and Anonymous Classes
	// ===========================================================
}
