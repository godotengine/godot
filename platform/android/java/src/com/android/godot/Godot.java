/*************************************************************************/
/*  Godot.java                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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
package com.android.godot;

import android.app.Activity;
import android.os.Bundle;
import android.view.MotionEvent;
import android.widget.RelativeLayout;
import android.widget.LinearLayout;
import android.view.ViewGroup.LayoutParams;


import android.app.*;
import android.content.*;
import android.view.*;
import android.view.inputmethod.InputMethodManager;
import android.os.*;
import android.util.Log;
import android.graphics.*;
import android.text.method.*;
import android.text.*;
import android.media.*;
import android.hardware.*;
import android.content.*;

import android.net.Uri;
import android.media.MediaPlayer;

import java.lang.reflect.Method;
import java.util.List;
import java.util.ArrayList;
import com.android.godot.payments.PaymentsManager;
import java.io.IOException;
import android.provider.Settings.Secure;
import android.widget.FrameLayout;
import com.android.godot.input.*;


public class Godot extends Activity implements SensorEventListener
{	
	static public class SingletonBase {

		protected void registerClass(String p_name, String[] p_methods) {

			GodotLib.singleton(p_name,this);

			Class clazz = getClass();
			Method[] methods = clazz.getDeclaredMethods();
			for (Method method : methods) {
				boolean found=false;
				System.out.printf("METHOD: %s\n",method.getName());

				for (String s : p_methods) {
				System.out.printf("METHOD CMP WITH: %s\n",s);
					if (s.equals(method.getName())) {
						found=true;
						System.out.printf("METHOD CMP VALID");
						break;
					}
				}
				if (!found)
					continue;

				System.out.printf("METHOD FOUND: %s\n",method.getName());

				List<String> ptr = new ArrayList<String>();

				Class[] paramTypes = method.getParameterTypes();
				for (Class c : paramTypes) {
					ptr.add(c.getName());
				}

				String[] pt = new String[ptr.size()];
				ptr.toArray(pt);

				GodotLib.method(p_name,method.getName(),method.getReturnType().getName(),pt);


			}
		}

		public void registerMethods() {}
	}

/*
	protected List<SingletonBase> singletons = new ArrayList<SingletonBase>();
	protected void instanceSingleton(SingletonBase s) {

		s.registerMethods();
		singletons.add(s);
	}

*/

	public GodotView mView;

	private SensorManager mSensorManager;
	private Sensor mAccelerometer;

	public FrameLayout layout;


	static public GodotIO io;

	public static void setWindowTitle(String title) {
		//setTitle(title);
	}

	public interface ResultCallback {
		public void callback(int requestCode, int resultCode, Intent data);
	};
	public ResultCallback result_callback;

	private PaymentsManager mPaymentsManager = null;

	@Override protected void onActivityResult (int requestCode, int resultCode, Intent data) {
		if(requestCode == PaymentsManager.REQUEST_CODE_FOR_PURCHASE){
			mPaymentsManager.processPurchaseResponse(resultCode, data);
		}else if (result_callback != null) {
			result_callback.callback(requestCode, resultCode, data);
			result_callback = null;
		};
	};

	public void onVideoInit(boolean use_gl2) {

//		mView = new GodotView(getApplication(),io,use_gl2);
//		setContentView(mView);

		layout = new FrameLayout(this);
		layout.setLayoutParams(new LayoutParams(LayoutParams.FILL_PARENT,LayoutParams.FILL_PARENT));
		setContentView(layout);
		
		// GodotEditText layout
		GodotEditText edittext = new GodotEditText(this); 
        edittext.setLayoutParams(new ViewGroup.LayoutParams(LayoutParams.FILL_PARENT,LayoutParams.WRAP_CONTENT));
        // ...add to FrameLayout
        layout.addView(edittext);
		
		mView = new GodotView(getApplication(),io,use_gl2, this);
		layout.addView(mView,new LayoutParams(LayoutParams.FILL_PARENT,LayoutParams.FILL_PARENT));
		mView.setKeepScreenOn(true);
		
        edittext.setView(mView);
        io.setEdit(edittext);
	}

	private static Godot _self;
	
	public static Godot getInstance(){
		return Godot._self;
	}
	
	@Override protected void onCreate(Bundle icicle) {

		System.out.printf("** GODOT ACTIVITY CREATED HERE ***\n");

		super.onCreate(icicle);
		_self = this;
		Window window = getWindow();
		window.addFlags(WindowManager.LayoutParams.FLAG_TURN_SCREEN_ON
			| WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

		io = new GodotIO(this);
		io.unique_id = Secure.getString(getContentResolver(), Secure.ANDROID_ID);
		GodotLib.io=io;
		GodotLib.initialize(this,io.needsReloadHooks());
		mSensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
		mAccelerometer = mSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
		mSensorManager.registerListener(this, mAccelerometer, SensorManager.SENSOR_DELAY_NORMAL);

		result_callback = null;
		
		mPaymentsManager = PaymentsManager.createManager(this).initService();
		
	//	instanceSingleton( new GodotFacebook(this) );


	}

	@Override protected void onDestroy(){
		
		if(mPaymentsManager != null ) mPaymentsManager.destroy();
		super.onDestroy();
	}
	
	@Override protected void onPause() {
		super.onPause();
		mView.onPause();
		mSensorManager.unregisterListener(this);
		GodotLib.focusout();

	}

	@Override protected void onResume() {
		super.onResume();
		mView.onResume();
		mSensorManager.registerListener(this, mAccelerometer, SensorManager.SENSOR_DELAY_NORMAL);
		GodotLib.focusin();
	}

	@Override public void onSensorChanged(SensorEvent event) {
		float x = event.values[0];
		float y = event.values[1];
		float z = event.values[2];
		GodotLib.accelerometer(x,y,z);
	}

	@Override public final void onAccuracyChanged(Sensor sensor, int accuracy) {
		// Do something here if sensor accuracy changes.
	}

/*
	@Override public boolean dispatchKeyEvent(KeyEvent event) {

		if (event.getKeyCode()==KeyEvent.KEYCODE_BACK) {

			System.out.printf("** BACK REQUEST!\n");

			GodotLib.quit();
			return true;
		}
		System.out.printf("** OTHER KEY!\n");

		return false;
	}
*/

	@Override public void onBackPressed() {

		GodotLib.quit();
	}

	public void forceQuit() {

		System.exit(0);
	}


	//@Override public boolean dispatchTouchEvent (MotionEvent event) {
	public boolean gotTouchEvent(MotionEvent event) {

		super.onTouchEvent(event);
		int evcount=event.getPointerCount();
		if (evcount==0)
			return true;

		int[] arr = new int[event.getPointerCount()*3];

		for(int i=0;i<event.getPointerCount();i++) {

			arr[i*3+0]=(int)event.getPointerId(i);
			arr[i*3+1]=(int)event.getX(i);
			arr[i*3+2]=(int)event.getY(i);
		}

		//System.out.printf("gaction: %d\n",event.getAction());
		switch(event.getAction()&MotionEvent.ACTION_MASK) {

			case MotionEvent.ACTION_DOWN: {
				GodotLib.touch(0,0,evcount,arr);
				//System.out.printf("action down at: %f,%f\n", event.getX(),event.getY());
			} break;
			case MotionEvent.ACTION_MOVE: {
				GodotLib.touch(1,0,evcount,arr);
				//for(int i=0;i<event.getPointerCount();i++) {
				//	System.out.printf("%d - moved to: %f,%f\n",i, event.getX(i),event.getY(i));
				//}
			} break;
			case MotionEvent.ACTION_POINTER_UP: {
				int pointer_idx = event.getActionIndex();
				GodotLib.touch(4,pointer_idx,evcount,arr);
				//System.out.printf("%d - s.up at: %f,%f\n",pointer_idx, event.getX(pointer_idx),event.getY(pointer_idx));
			} break;
			case MotionEvent.ACTION_POINTER_DOWN: {
				int pointer_idx = event.getActionIndex();
				GodotLib.touch(3,pointer_idx,evcount,arr);
				//System.out.printf("%d - s.down at: %f,%f\n",pointer_idx, event.getX(pointer_idx),event.getY(pointer_idx));
			} break;
			case MotionEvent.ACTION_CANCEL:
			case MotionEvent.ACTION_UP: {
				GodotLib.touch(2,0,evcount,arr);
				//for(int i=0;i<event.getPointerCount();i++) {
				//	System.out.printf("%d - up! %f,%f\n",i, event.getX(i),event.getY(i));
				//}
			} break;

		}
		return true;
	}

    @Override public boolean onKeyMultiple(final int inKeyCode, int repeatCount, KeyEvent event) {
        String s = event.getCharacters();
        if (s == null || s.length() == 0)
        	return super.onKeyMultiple(inKeyCode, repeatCount, event);
        
        final char[] cc = s.toCharArray();
        int cnt = 0;
        for (int i = cc.length; --i >= 0; cnt += cc[i] != 0 ? 1 : 0);
        if (cnt == 0) return super.onKeyMultiple(inKeyCode, repeatCount, event);
        final Activity me = this;
        queueEvent(new Runnable() {
            // This method will be called on the rendering thread:
            public void run() {
                for (int i = 0, n = cc.length; i < n; i++) {
                    int keyCode;
                    if ((keyCode = cc[i]) != 0) {
                        // Simulate key down and up...
                		GodotLib.key(0, keyCode, true);
                		GodotLib.key(0, keyCode, false);
                    }
                }
            }
        });
        return true;
    }	

	private void queueEvent(Runnable runnable) {
		// TODO Auto-generated method stub
		
	}

	public PaymentsManager getPaymentsManager() {
		return mPaymentsManager;
	}

//	public void setPaymentsManager(PaymentsManager mPaymentsManager) {
//		this.mPaymentsManager = mPaymentsManager;
//	};


	// Audio


}
