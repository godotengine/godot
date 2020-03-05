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
import android.content.Context;
import android.os.Handler;
import android.os.Message;
import android.text.InputFilter;
import android.util.AttributeSet;
import android.view.KeyEvent;
import android.view.inputmethod.EditorInfo;
import android.view.inputmethod.InputMethodManager;
import android.widget.EditText;
import java.lang.ref.WeakReference;
import org.godotengine.godot.*;

public class GodotEditText extends EditText {
	// ===========================================================
	// Constants
	// ===========================================================
	private final static int HANDLER_OPEN_IME_KEYBOARD = 2;
	private final static int HANDLER_CLOSE_IME_KEYBOARD = 3;

	// ===========================================================
	// Fields
	// ===========================================================
	private GodotRenderView mRenderView;
	private GodotTextInputWrapper mInputWrapper;
	private EditHandler sHandler = new EditHandler(this);
	private String mOriginText;

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

	// ===========================================================
	// Constructors
	// ===========================================================
	public GodotEditText(final Context context) {
		super(context);
		initView();
	}

	public GodotEditText(final Context context, final AttributeSet attrs) {
		super(context, attrs);
		initView();
	}

	public GodotEditText(final Context context, final AttributeSet attrs, final int defStyle) {
		super(context, attrs, defStyle);
		initView();
	}

	protected void initView() {
		setPadding(0, 0, 0, 0);
		setImeOptions(EditorInfo.IME_FLAG_NO_EXTRACT_UI);
	}

	private void handleMessage(final Message msg) {
		switch (msg.what) {
			case HANDLER_OPEN_IME_KEYBOARD: {
				GodotEditText edit = (GodotEditText)msg.obj;
				String text = edit.mOriginText;
				if (edit.requestFocus()) {
					edit.removeTextChangedListener(edit.mInputWrapper);
					edit.setText("");
					edit.append(text);
					edit.mInputWrapper.setOriginText(text);
					edit.addTextChangedListener(edit.mInputWrapper);
					setMaxInputLength(edit, msg.arg1);
					final InputMethodManager imm = (InputMethodManager)mRenderView.getView().getContext().getSystemService(Context.INPUT_METHOD_SERVICE);
					imm.showSoftInput(edit, 0);
				}
			} break;

			case HANDLER_CLOSE_IME_KEYBOARD: {
				GodotEditText edit = (GodotEditText)msg.obj;

				edit.removeTextChangedListener(mInputWrapper);
				final InputMethodManager imm = (InputMethodManager)mRenderView.getView().getContext().getSystemService(Context.INPUT_METHOD_SERVICE);
				imm.hideSoftInputFromWindow(edit.getWindowToken(), 0);
				edit.mRenderView.getView().requestFocus();
			} break;
		}
	}

	private void setMaxInputLength(EditText p_edit_text, int p_max_input_length) {
		if (p_max_input_length > 0) {
			InputFilter[] filters = new InputFilter[1];
			filters[0] = new InputFilter.LengthFilter(p_max_input_length);
			p_edit_text.setFilters(filters);
		} else {
			p_edit_text.setFilters(new InputFilter[] {});
		}
	}

	// ===========================================================
	// Getter & Setter
	// ===========================================================
	public void setView(final GodotRenderView view) {
		mRenderView = view;
		if (mInputWrapper == null)
			mInputWrapper = new GodotTextInputWrapper(mRenderView, this);
		setOnEditorActionListener(mInputWrapper);
		view.getView().requestFocus();
	}

	// ===========================================================
	// Methods for/from SuperClass/Interfaces
	// ===========================================================
	@Override
	public boolean onKeyDown(final int keyCode, final KeyEvent keyEvent) {
		super.onKeyDown(keyCode, keyEvent);

		/* Let GlSurfaceView get focus if back key is input. */
		if (keyCode == KeyEvent.KEYCODE_BACK) {
			mRenderView.getView().requestFocus();
		}

		return true;
	}

	// ===========================================================
	// Methods
	// ===========================================================
	public void showKeyboard(String p_existing_text, int p_max_input_length) {
		mOriginText = p_existing_text;

		final Message msg = new Message();
		msg.what = HANDLER_OPEN_IME_KEYBOARD;
		msg.obj = this;
		msg.arg1 = p_max_input_length;
		sHandler.sendMessage(msg);
	}

	public void hideKeyboard() {
		final Message msg = new Message();
		msg.what = HANDLER_CLOSE_IME_KEYBOARD;
		msg.obj = this;
		sHandler.sendMessage(msg);
	}

	// ===========================================================
	// Inner and Anonymous Classes
	// ===========================================================
}
