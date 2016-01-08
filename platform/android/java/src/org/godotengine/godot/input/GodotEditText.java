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
