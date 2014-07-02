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
import java.io.InputStream;

import javax.microedition.khronos.opengles.GL10;
import java.security.MessageDigest;
import java.io.File;
import java.io.FileInputStream;
import java.util.LinkedList;

public class Godot extends Activity implements SensorEventListener
{	

	static final int MAX_SINGLETONS = 64;

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

			Godot.singletons[Godot.singleton_count++]=this;
		}

		protected void onMainActivityResult(int requestCode, int resultCode, Intent data) {


		}

		protected void onMainPause() {}
		protected void onMainResume() {}
		protected void onMainDestroy() {}

		protected void onGLDrawFrame(GL10 gl) {}
		protected void onGLSurfaceChanged(GL10 gl, int width, int height) {} // singletons will always miss first onGLSurfaceChanged call
		//protected void onGLSurfaceCreated(GL10 gl, EGLConfig config) {} // singletons won't be ready until first GodotLib.step()

		public void registerMethods() {}
	}

/*
	protected List<SingletonBase> singletons = new ArrayList<SingletonBase>();
	protected void instanceSingleton(SingletonBase s) {

		s.registerMethods();
		singletons.add(s);
	}

*/

	private String[] command_line;

	public GodotView mView;
	private boolean godot_initialized=false;


	private SensorManager mSensorManager;
	private Sensor mAccelerometer;

	public FrameLayout layout;
	public RelativeLayout adLayout;


	static public GodotIO io;

	public static void setWindowTitle(String title) {
		//setTitle(title);
	}


	static SingletonBase singletons[] = new SingletonBase[MAX_SINGLETONS];
	static int singleton_count=0;


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

		for(int i=0;i<singleton_count;i++) {

			singletons[i].onMainActivityResult(requestCode,resultCode,data);
		}
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
		
		// Ad layout
		adLayout = new RelativeLayout(this);
		adLayout.setLayoutParams(new LayoutParams(LayoutParams.FILL_PARENT,LayoutParams.FILL_PARENT));
		layout.addView(adLayout);
		
	}

	private static Godot _self;
	
	public static Godot getInstance(){
		return Godot._self;
	}
	

	private String[] getCommandLine() {
            InputStream is;
            try {
		is = getAssets().open("_cl_");
                byte[] len = new byte[4];
                int r = is.read(len);
		if (r<4) {
                    System.out.printf("**ERROR** Wrong cmdline length.\n");
		    Log.d("GODOT", "**ERROR** Wrong cmdline length.\n");
                    return new String[0];
                }
		int argc=((int)(len[3]&0xFF)<<24) | ((int)(len[2]&0xFF)<<16) | ((int)(len[1]&0xFF)<<8) | ((int)(len[0]&0xFF));
                String[] cmdline = new String[argc];

                for(int i=0;i<argc;i++) {
                    r = is.read(len);
                    if (r<4) {

			Log.d("GODOT", "**ERROR** Wrong cmdline param lenght.\n");
                        return new String[0];
                    }
		    int strlen=((int)(len[3]&0xFF)<<24) | ((int)(len[2]&0xFF)<<16) | ((int)(len[1]&0xFF)<<8) | ((int)(len[0]&0xFF));
                    if (strlen>65535) {
			Log.d("GODOT", "**ERROR** Wrong command len\n");
                        return new String[0];
                    }
		    byte[] arg = new byte[strlen];
                    r = is.read(arg);
		    if (r==strlen) {
                        cmdline[i]=new String(arg,"UTF-8");
		    }
			}
			return cmdline;
		} catch (Exception e) {
		e.printStackTrace();
		System.out.printf("**ERROR** No commandline.\n");
		Log.d("GODOT", "**ERROR** Exception " + e.getClass().getName() + ":" + e.getMessage());
			return new String[0];
		}


	}


	String expansion_pack_path;


	private void initializeGodot() {

		if (expansion_pack_path!=null) {

			String[] new_cmdline;
			int cll=0;
			if (command_line!=null) {
				new_cmdline = new String[ command_line.length + 2 ];
				cll=command_line.length;
				for(int i=0;i<command_line.length;i++) {
					new_cmdline[i]=command_line[i];
				}
			} else {
				new_cmdline = new String[ 2 ];
			}

			new_cmdline[cll]="-main_pack";
			new_cmdline[cll+1]=expansion_pack_path;
			command_line=new_cmdline;
		}

		io = new GodotIO(this);
		io.unique_id = Secure.getString(getContentResolver(), Secure.ANDROID_ID);
		GodotLib.io=io;
		GodotLib.initialize(this,io.needsReloadHooks(),command_line);
		mSensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
		mAccelerometer = mSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
		mSensorManager.registerListener(this, mAccelerometer, SensorManager.SENSOR_DELAY_NORMAL);

		result_callback = null;

		mPaymentsManager = PaymentsManager.createManager(this).initService();
		godot_initialized=true;

	}


	@Override protected void onCreate(Bundle icicle) {

		System.out.printf("** GODOT ACTIVITY CREATED HERE ***\n");

		super.onCreate(icicle);
		_self = this;
		Window window = getWindow();
		window.addFlags(WindowManager.LayoutParams.FLAG_TURN_SCREEN_ON
			| WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);


		//check for apk expansion API
		if (true) {
			command_line = getCommandLine();
			boolean use_apk_expansion=false;
			String main_pack_md5=null;
			String main_pack_key=null;

			List<String> new_args = new LinkedList<String>();


			for(int i=0;i<command_line.length;i++) {

				boolean has_extra = i< command_line.length -1;
				if (command_line[i].equals("-use_apk_expansion")) {
					use_apk_expansion=true;
				} else if (has_extra && command_line[i].equals("-apk_expansion_md5")) {
					main_pack_md5=command_line[i+1];
					i++;
				} else if (has_extra && command_line[i].equals("-apk_expansion_key")) {
					main_pack_key=command_line[i+1];
					i++;
				} else if (command_line[i].trim().length()!=0){
					new_args.add(command_line[i]);
				}
			}

			if (new_args.isEmpty())
				command_line=null;
			else
				command_line = new_args.toArray(new String[new_args.size()]);

			if (use_apk_expansion && main_pack_md5!=null && main_pack_key!=null) {
				//check that environment is ok!
				if (!Environment.getExternalStorageState().equals( Environment.MEDIA_MOUNTED )) {
					Log.d("GODOT", "**ERROR! No media mounted!");
					//show popup and die
				}

				// Build the full path to the app's expansion files
				try {
					expansion_pack_path = Environment.getExternalStorageDirectory().toString() + "/Android/obb/"+this.getPackageName();
					expansion_pack_path+="/"+"main."+getPackageManager().getPackageInfo(getPackageName(), 0).versionCode+"."+this.getPackageName()+".obb";
				} catch (Exception e) {
					e.printStackTrace();
				}

				File f = new File(expansion_pack_path);

				boolean pack_valid = true;
				Log.d("GODOT","**PACK** - Path "+expansion_pack_path);

				if (!f.exists()) {

					pack_valid=false;
					Log.d("GODOT","**PACK** - File does not exist");

				} else {
					try {

						InputStream fis =  new FileInputStream(expansion_pack_path);

						// Create MD5 Hash
						byte[] buffer = new byte[16384];

						MessageDigest complete = MessageDigest.getInstance("MD5");
						int numRead;
						do {
							numRead = fis.read(buffer);
							if (numRead > 0) {
								complete.update(buffer, 0, numRead);
							}
					       } while (numRead != -1);


						fis.close();
						byte[] messageDigest = complete.digest();

						// Create Hex String
						StringBuffer hexString = new StringBuffer();
						for (int i=0; i<messageDigest.length; i++)
							hexString.append(Integer.toHexString(0xFF & messageDigest[i]));
						String md5str =  hexString.toString();

						Log.d("GODOT","**PACK** - My MD5: "+hexString+" - APK md5: "+main_pack_md5);
						if (!hexString.equals(main_pack_md5)) {
							pack_valid=false;
						}
					} catch (Exception e) {
						e.printStackTrace();
						Log.d("GODOT","**PACK FAIL**");
						pack_valid=false;
					}


				}

				if (!pack_valid) {



				}

			}
		}

		initializeGodot();

		
	//	instanceSingleton( new GodotFacebook(this) );


	}

	@Override protected void onDestroy(){
		
		if(mPaymentsManager != null ) mPaymentsManager.destroy();
		for(int i=0;i<singleton_count;i++) {
			singletons[i].onMainDestroy();
		}
		super.onDestroy();
	}
	
	@Override protected void onPause() {
		super.onPause();
		if (!godot_initialized)
			return;
		mView.onPause();
		mSensorManager.unregisterListener(this);
		GodotLib.focusout();

		for(int i=0;i<singleton_count;i++) {
			singletons[i].onMainPause();
		}
	}

	@Override protected void onResume() {
		super.onResume();
		if (!godot_initialized)
			return;

		mView.onResume();
		mSensorManager.registerListener(this, mAccelerometer, SensorManager.SENSOR_DELAY_NORMAL);
		GodotLib.focusin();

		for(int i=0;i<singleton_count;i++) {

			singletons[i].onMainResume();
		}

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

		System.out.printf("** BACK REQUEST!\n");
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
