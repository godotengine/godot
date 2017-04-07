/*************************************************************************/
/*  GodotEditText.java                                                   */
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
import android.util.AttributeSet;
import android.view.KeyEvent;
import android.widget.EditText;
import org.godotengine.godot.*;
import android.os.Handler;
import android.os.Message;
import android.view.inputmethod.InputMethodManager;
import android.view.inputmethod.EditorInfo;

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
	private GodotTextInputWrapper mInputWrapper;
	private static Handler sHandler;
	private String mOriginText;

	// ===========================================================
	// Constructors
	// ===========================================================
	public GodotEditText(final Context context) {
		super(context);
		this.initView();
	}

	public GodotEditText(final Context context, final AttributeSet attrs) {
		super(context, attrs);
		this.initView();
	}

	public GodotEditText(final Context context, final AttributeSet attrs, final int defStyle) {
		super(context, attrs, defStyle);
		this.initView();
	}
	
	protected void initView() {
		this.setPadding(0,  0, 0, 0);
		this.setImeOptions(EditorInfo.IME_FLAG_NO_EXTRACT_UI);
		
		sHandler = new Handler() {
			@Override
			public void handleMessage(final Message msg) {
				switch (msg.what) {
					case HANDLER_OPEN_IME_KEYBOARD:
						{
							GodotEditText edit = (GodotEditText) msg.obj;
							String text = edit.mOriginText;
							if (edit.requestFocus())
							{
								edit.removeTextChangedListener(edit.mInputWrapper);
								edit.setText("");
								edit.append(text);
								edit.mInputWrapper.setOriginText(text);
								edit.addTextChangedListener(edit.mInputWrapper);
								final InputMethodManager imm = (InputMethodManager) mView.getContext().getSystemService(Context.INPUT_METHOD_SERVICE);
								imm.showSoftInput(edit, 0);
							}
						}
						break;

					case HANDLER_CLOSE_IME_KEYBOARD:
						{
							GodotEditText edit = (GodotEditText) msg.obj;
							
							edit.removeTextChangedListener(mInputWrapper);
							final InputMethodManager imm = (InputMethodManager) mView.getContext().getSystemService(Context.INPUT_METHOD_SERVICE);
							imm.hideSoftInputFromWindow(edit.getWindowToken(), 0);
							edit.mView.requestFocus();
						}
						break;
				}
			}
		};
	}

	// ===========================================================
	// Getter & Setter
	// ===========================================================
	public void setView(final GodotView view) {
		this.mView = view;
		if(mInputWrapper == null)
			mInputWrapper = new GodotTextInputWrapper(mView, this);
		this.setOnEditorActionListener(mInputWrapper);
		view.requestFocus();
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
	public void showKeyboard(String p_existing_text) {
		this.mOriginText = p_existing_text;
		
		final Message msg = new Message();
		msg.what = HANDLER_OPEN_IME_KEYBOARD;
		msg.obj = this;
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
