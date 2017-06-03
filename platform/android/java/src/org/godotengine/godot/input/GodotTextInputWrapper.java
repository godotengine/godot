/*************************************************************************/
/*  GodotTextInputWrapper.java                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
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
package org.godotengine.godot.input;
import android.content.Context;
import android.text.Editable;
import android.text.TextWatcher;
import android.util.Log;
import android.view.KeyEvent;
import android.view.inputmethod.EditorInfo;
import android.view.inputmethod.InputMethodManager;
import android.widget.TextView;
import android.widget.TextView.OnEditorActionListener;
import org.godotengine.godot.*;

public class GodotTextInputWrapper implements TextWatcher, OnEditorActionListener {
	// ===========================================================
	// Constants
	// ===========================================================
	private static final String TAG = GodotTextInputWrapper.class.getSimpleName();

	// ===========================================================
	// Fields
	// ===========================================================
	private final GodotView mView;
	private final GodotEditText mEdit;
	private String mOriginText;

	// ===========================================================
	// Constructors
	// ===========================================================

	public GodotTextInputWrapper(final GodotView view, final GodotEditText edit) {
		this.mView = view;
		this.mEdit = edit;
	}

	// ===========================================================
	// Getter & Setter
	// ===========================================================

	private boolean isFullScreenEdit() {
		final TextView textField = this.mEdit;
		final InputMethodManager imm = (InputMethodManager) textField.getContext().getSystemService(Context.INPUT_METHOD_SERVICE);
		return imm.isFullscreenMode();
	}

	public void setOriginText(final String originText) {
		this.mOriginText = originText;
	}

	// ===========================================================
	// Methods for/from SuperClass/Interfaces
	// ===========================================================

	@Override
	public void afterTextChanged(final Editable s) {

	}

	@Override
	public void beforeTextChanged(final CharSequence pCharSequence, final int start, final int count, final int after) {
		//Log.d(TAG, "beforeTextChanged(" + pCharSequence + ")start: " + start + ",count: " + count + ",after: " + after);

		for (int i=0;i<count;i++){
			GodotLib.key(KeyEvent.KEYCODE_DEL, 0, true);
			GodotLib.key(KeyEvent.KEYCODE_DEL, 0, false);
		}
	}

	@Override
	public void onTextChanged(final CharSequence pCharSequence, final int start, final int before, final int count) {
		//Log.d(TAG, "onTextChanged(" + pCharSequence + ")start: " + start + ",count: " + count + ",before: " + before);

		for (int i=start;i<start+count;i++){
			int ch = pCharSequence.charAt(i);
			GodotLib.key(0, ch, true);
			GodotLib.key(0, ch, false);
		}

	}

	@Override
	public boolean onEditorAction(final TextView pTextView, final int pActionID, final KeyEvent pKeyEvent) {
		if (this.mEdit == pTextView && this.isFullScreenEdit()) {
			// user press the action button, delete all old text and insert new text
			for (int i = this.mOriginText.length(); i > 0; i--) {
				GodotLib.key(KeyEvent.KEYCODE_DEL, 0, true);
				GodotLib.key(KeyEvent.KEYCODE_DEL, 0, false);
				/*
				if (BuildConfig.DEBUG) {
					Log.d(TAG, "deleteBackward");
				}
				*/
			}
			String text = pTextView.getText().toString();

			/* If user input nothing, translate "\n" to engine. */
			if (text.compareTo("") == 0) {
				text = "\n";
			}

			if ('\n' != text.charAt(text.length() - 1)) {
				text += '\n';
			}

			for(int i = 0; i < text.length(); i++) {
				int ch = text.codePointAt(i);
				GodotLib.key(0, ch, true);
				GodotLib.key(0, ch, false);
			}
			/*
			if (BuildConfig.DEBUG) {
				Log.d(TAG, "insertText(" + insertText + ")");
			}
			*/
		}
		
		if (pActionID == EditorInfo.IME_ACTION_DONE) {
			this.mView.requestFocus();
		}
		return false;
	}

	// ===========================================================
	// Methods
	// ===========================================================

	// ===========================================================
	// Inner and Anonymous Classes
	// ===========================================================
}
