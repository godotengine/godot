/*************************************************************************/
/*  Godot.java                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
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
package org.godotengine.godot;

import android.R;
import android.app.Activity;
import android.os.Bundle;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Button;
import android.widget.ProgressBar;
import android.widget.RelativeLayout;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.view.ViewGroup.LayoutParams;
import android.app.*;
import android.content.*;
import android.content.SharedPreferences.Editor;
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
import android.content.pm.PackageManager.NameNotFoundException;
import android.net.Uri;
import android.media.MediaPlayer;

import java.lang.reflect.Method;
import java.util.List;
import java.util.ArrayList;

import org.godotengine.godot.payments.PaymentsManager;

import java.io.IOException;

import android.provider.Settings.Secure;
import android.widget.FrameLayout;

import org.godotengine.godot.input.*;

import java.io.InputStream;
import javax.microedition.khronos.opengles.GL10;
import java.security.MessageDigest;
import java.io.File;
import java.io.FileInputStream;
import java.util.LinkedList;

import com.google.android.vending.expansion.downloader.Constants;
import com.google.android.vending.expansion.downloader.DownloadProgressInfo;
import com.google.android.vending.expansion.downloader.DownloaderClientMarshaller;
import com.google.android.vending.expansion.downloader.DownloaderServiceMarshaller;
import com.google.android.vending.expansion.downloader.Helpers;
import com.google.android.vending.expansion.downloader.IDownloaderClient;
import com.google.android.vending.expansion.downloader.IDownloaderService;
import com.google.android.vending.expansion.downloader.IStub;

import android.os.Bundle;
import android.os.Messenger;
import android.os.SystemClock;


public class Godot extends Activity implements SensorEventListener, IDownloaderClient
{

	static final int MAX_SINGLETONS = 64;
	private IStub mDownloaderClientStub;
    private IDownloaderService mRemoteService;
    private TextView mStatusText;
    private TextView mProgressFraction;
    private TextView mProgressPercent;
    private TextView mAverageSpeed;
    private TextView mTimeRemaining;
    private ProgressBar mPB;

    private View mDashboard;
    private View mCellMessage;

    private Button mPauseButton;
    private Button mWiFiSettingsButton;

    private boolean use_32_bits=false;
    private boolean use_immersive=false;
    private boolean mStatePaused;
    private int mState;
	private boolean keep_screen_on=true;

	static private Intent mCurrentIntent;

	@Override public void onNewIntent(Intent intent) {
		mCurrentIntent = intent;
	}

	static public Intent getCurrentIntent() {
		return mCurrentIntent;
	}

	private void setState(int newState) {
        if (mState != newState) {
            mState = newState;
            mStatusText.setText(Helpers.getDownloaderStringResourceIDFromState(newState));
        }
    }

    private void setButtonPausedState(boolean paused) {
        mStatePaused = paused;
        int stringResourceID = paused ? com.godot.game.R.string.text_button_resume :
        	com.godot.game.R.string.text_button_pause;
        mPauseButton.setText(stringResourceID);
    }

	static public class SingletonBase {

		protected void registerClass(String p_name, String[] p_methods) {

			GodotLib.singleton(p_name,this);

			Class clazz = getClass();
			Method[] methods = clazz.getDeclaredMethods();
			for (Method method : methods) {
				boolean found=false;
				Log.d("XXX","METHOD: %s\n" + method.getName());

				for (String s : p_methods) {
				Log.d("XXX", "METHOD CMP WITH: %s\n" + s);
					if (s.equals(method.getName())) {
						found=true;
						Log.d("XXX","METHOD CMP VALID");
						break;
					}
				}
				if (!found)
					continue;

				Log.d("XXX","METHOD FOUND: %s\n" + method.getName());

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
		protected boolean onMainBackPressed() { return false; }

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
	private boolean use_apk_expansion;

	public GodotView mView;
	private boolean godot_initialized=false;


	private SensorManager mSensorManager;
	private Sensor mAccelerometer;
	private Sensor mMagnetometer;
	private Sensor mGyroscope;

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

		//mView = new GodotView(getApplication(),io,use_gl2);
		//setContentView(mView);

		layout = new FrameLayout(this);
		layout.setLayoutParams(new LayoutParams(LayoutParams.FILL_PARENT,LayoutParams.FILL_PARENT));
		setContentView(layout);

		// GodotEditText layout
		GodotEditText edittext = new GodotEditText(this);
		   edittext.setLayoutParams(new ViewGroup.LayoutParams(LayoutParams.FILL_PARENT,LayoutParams.WRAP_CONTENT));
        // ...add to FrameLayout
		   layout.addView(edittext);

		mView = new GodotView(getApplication(),io,use_gl2,use_32_bits, this);
		layout.addView(mView,new LayoutParams(LayoutParams.FILL_PARENT,LayoutParams.FILL_PARENT));
		edittext.setView(mView);
		io.setEdit(edittext);

		final Godot godot = this;
		mView.getViewTreeObserver().addOnGlobalLayoutListener(new ViewTreeObserver.OnGlobalLayoutListener() {
				@Override
				public void onGlobalLayout() {
					Point fullSize = new Point();
					godot.getWindowManager().getDefaultDisplay().getSize(fullSize);
					Rect gameSize = new Rect();
					godot.mView.getWindowVisibleDisplayFrame(gameSize);

					final int keyboardHeight = fullSize.y - gameSize.bottom;
					Log.d("GODOT", "setVirtualKeyboardHeight: " + keyboardHeight);
					GodotLib.setVirtualKeyboardHeight(keyboardHeight);
				}
		});

		// Ad layout
		adLayout = new RelativeLayout(this);
		adLayout.setLayoutParams(new LayoutParams(LayoutParams.FILL_PARENT,LayoutParams.FILL_PARENT));
		layout.addView(adLayout);

		final String[] current_command_line = command_line;
		final GodotView view = mView;
		mView.queueEvent(new Runnable() {
			@Override
			public void run() {
				GodotLib.setup(current_command_line);
				runOnUiThread(new Runnable() {
					@Override
					public void run() {
						view.setKeepScreenOn("True".equals(GodotLib.getGlobal("display/driver/keep_screen_on")));
					}
				});
			}
		});

	}

	public void setKeepScreenOn(final boolean p_enabled) {
		keep_screen_on = p_enabled;
		if (mView != null){
			runOnUiThread(new Runnable() {
				@Override
				public void run() {
					mView.setKeepScreenOn(p_enabled);
				}
			});
		}
	}

	public void alert(final String message, final String title) {
		runOnUiThread(new Runnable() {
			@Override
			public void run() {
				AlertDialog.Builder builder = new AlertDialog.Builder(getInstance());
				builder.setMessage(message).setTitle(title);
				builder.setPositiveButton(
					"OK",
					new DialogInterface.OnClickListener() {
						public void onClick(DialogInterface dialog, int id) {
							dialog.cancel();
						}
					});
				AlertDialog dialog = builder.create();
				dialog.show();
			}
		});
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
                    Log.d("XXX","**ERROR** Wrong cmdline length.\n");
		    Log.d("GODOT", "**ERROR** Wrong cmdline length.\n");
                    return new String[0];
                }
		int argc=((int)(len[3]&0xFF)<<24) | ((int)(len[2]&0xFF)<<16) | ((int)(len[1]&0xFF)<<8) | ((int)(len[0]&0xFF));
                String[] cmdline = new String[argc];

                for(int i=0;i<argc;i++) {
                    r = is.read(len);
                    if (r<4) {

			Log.d("GODOT", "**ERROR** Wrong cmdline param length.\n");
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
			        Log.d("GODOT", "initializeGodot: command_line: is not null" );
				new_cmdline = new String[ command_line.length + 2 ];
				cll=command_line.length;
				for(int i=0;i<command_line.length;i++) {
					new_cmdline[i]=command_line[i];
				}
			} else {
			        Log.d("GODOT", "initializeGodot: command_line: is null" );
				new_cmdline = new String[ 2 ];
			}

			new_cmdline[cll]="--main_pack";
			new_cmdline[cll+1]=expansion_pack_path;
			command_line=new_cmdline;
		}

		io = new GodotIO(this);
		io.unique_id = Secure.getString(getContentResolver(), Secure.ANDROID_ID);
		GodotLib.io=io;
		Log.d("GODOT", "command_line is null? " + ((command_line == null)?"yes":"no"));
		/*if(command_line != null){
		    Log.d("GODOT", "Command Line:");
		    for(int w=0;w <command_line.length;w++){
		        Log.d("GODOT","   " + command_line[w]);
		    }
		}*/
		mSensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
		mAccelerometer = mSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
		mSensorManager.registerListener(this, mAccelerometer, SensorManager.SENSOR_DELAY_GAME);
		mMagnetometer = mSensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD);
		mSensorManager.registerListener(this, mMagnetometer, SensorManager.SENSOR_DELAY_GAME);
		mGyroscope = mSensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
		mSensorManager.registerListener(this, mGyroscope, SensorManager.SENSOR_DELAY_GAME);

		GodotLib.initialize(this, io.needsReloadHooks(), getAssets(), use_apk_expansion);

		result_callback = null;

		mPaymentsManager = PaymentsManager.createManager(this).initService();

		godot_initialized=true;

	}

	@Override
	public void onServiceConnected(Messenger m) {
	    mRemoteService = DownloaderServiceMarshaller.CreateProxy(m);
	    mRemoteService.onClientUpdated(mDownloaderClientStub.getMessenger());
	}



	@Override
	protected void onCreate(Bundle icicle) {

		Log.d("GODOT", "** GODOT ACTIVITY CREATED HERE ***\n");

		super.onCreate(icicle);
		_self = this;
		Window window = getWindow();
		//window.addFlags(WindowManager.LayoutParams.FLAG_TURN_SCREEN_ON | WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
		window.addFlags(WindowManager.LayoutParams.FLAG_TURN_SCREEN_ON);

		//check for apk expansion API
		if (true) {
		        boolean md5mismatch = false;
			command_line = getCommandLine();
			String main_pack_md5=null;
			String main_pack_key=null;

			List<String> new_args = new LinkedList<String>();


			for(int i=0;i<command_line.length;i++) {

				boolean has_extra = i< command_line.length -1;
				if (command_line[i].equals("--use_depth_32")) {
					use_32_bits=true;
				} else if (command_line[i].equals("--use_immersive")) {
					use_immersive=true;
					if(Build.VERSION.SDK_INT >= 19.0){ // check if the application runs on an android 4.4+
						window.getDecorView().setSystemUiVisibility(
								    View.SYSTEM_UI_FLAG_LAYOUT_STABLE
									    | View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
									    | View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
									    | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION // hide nav bar
									    | View.SYSTEM_UI_FLAG_FULLSCREEN // hide status bar
									    | View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY);

						UiChangeListener();
					}
				} else if (command_line[i].equals("--use_apk_expansion")) {
					use_apk_expansion=true;
				} else if (has_extra && command_line[i].equals("--apk_expansion_md5")) {
					main_pack_md5=command_line[i+1];
					i++;
				} else if (has_extra && command_line[i].equals("--apk_expansion_key")) {
					main_pack_key=command_line[i+1];
					SharedPreferences prefs = getSharedPreferences("app_data_keys", MODE_PRIVATE);
					Editor editor = prefs.edit();
					editor.putString("store_public_key", main_pack_key);

					editor.commit();
					i++;
				} else if (command_line[i].trim().length()!=0){
					new_args.add(command_line[i]);
				}
			}

			if (new_args.isEmpty()){
				command_line=null;
			}else{

				command_line = new_args.toArray(new String[new_args.size()]);
                        }
			if (use_apk_expansion && main_pack_md5!=null && main_pack_key!=null) {
				//check that environment is ok!
				if (!Environment.getExternalStorageState().equals( Environment.MEDIA_MOUNTED )) {
					Log.d("GODOT", "**ERROR! No media mounted!");
					//show popup and die
				}

				// Build the full path to the app's expansion files
				try {
					expansion_pack_path = Helpers.getSaveFilePath(getApplicationContext());
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

				} else if(  obbIsCorrupted(expansion_pack_path, main_pack_md5)){
					Log.d("GODOT", "**PACK** - Expansion pack (obb) is corrupted");
					pack_valid = false;
					try{
					    f.delete();
					}catch(Exception e){
					    Log.d("GODOT", "**PACK** - Error deleting corrupted expansion pack (obb)");
					}
				}

				if (!pack_valid) {
					Log.d("GODOT", "Pack Invalid, try re-downloading.");

					Intent notifierIntent = new Intent(this, this.getClass());
					notifierIntent.setFlags(Intent.FLAG_ACTIVITY_NEW_TASK |
			                                Intent.FLAG_ACTIVITY_CLEAR_TOP);

                                                PendingIntent pendingIntent = PendingIntent.getActivity(this, 0,
			                notifierIntent, PendingIntent.FLAG_UPDATE_CURRENT);

			        int startResult;
					try {
						Log.d("GODOT", "INITIALIZING DOWNLOAD");
						startResult = DownloaderClientMarshaller.startDownloadServiceIfRequired(
								getApplicationContext(),
								pendingIntent,
								GodotDownloaderService.class);
						Log.d("GODOT", "DOWNLOAD SERVICE FINISHED:" + startResult);

			        if (startResult != DownloaderClientMarshaller.NO_DOWNLOAD_REQUIRED) {
						Log.d("GODOT", "DOWNLOAD REQUIRED");
			            // This is where you do set up to display the download
			            // progress (next step)
			        	mDownloaderClientStub = DownloaderClientMarshaller.CreateStub(this,
			        			GodotDownloaderService.class);

			        	setContentView(com.godot.game.R.layout.downloading_expansion);
			        	mPB = (ProgressBar) findViewById(com.godot.game.R.id.progressBar);
			            mStatusText = (TextView) findViewById(com.godot.game.R.id.statusText);
			            mProgressFraction = (TextView) findViewById(com.godot.game.R.id.progressAsFraction);
			            mProgressPercent = (TextView) findViewById(com.godot.game.R.id.progressAsPercentage);
			            mAverageSpeed = (TextView) findViewById(com.godot.game.R.id.progressAverageSpeed);
			            mTimeRemaining = (TextView) findViewById(com.godot.game.R.id.progressTimeRemaining);
			            mDashboard = findViewById(com.godot.game.R.id.downloaderDashboard);
			            mCellMessage = findViewById(com.godot.game.R.id.approveCellular);
			            mPauseButton = (Button) findViewById(com.godot.game.R.id.pauseButton);
			            mWiFiSettingsButton = (Button) findViewById(com.godot.game.R.id.wifiSettingsButton);

			            return;
			        } else{
			        	Log.d("GODOT", "NO DOWNLOAD REQUIRED");
			        }
					} catch (NameNotFoundException e) {
						// TODO Auto-generated catch block
						Log.d("GODOT", "Error downloading expansion package:" + e.getMessage());
					}

				}

			}
		}

		mCurrentIntent = getIntent();

		initializeGodot();


		//instanceSingleton( new GodotFacebook(this) );


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
		if (!godot_initialized){
			if (null != mDownloaderClientStub) {
		        mDownloaderClientStub.disconnect(this);
		    }
			return;
		}
		mView.onPause();
		mView.queueEvent(new Runnable() {
			@Override
			public void run() {
				GodotLib.focusout();
			}
		});
		mSensorManager.unregisterListener(this);

		for(int i=0;i<singleton_count;i++) {
			singletons[i].onMainPause();
		}
	}

	@Override protected void onResume() {
		super.onResume();
		if (!godot_initialized){
			if (null != mDownloaderClientStub) {
		        mDownloaderClientStub.connect(this);
		    }
			return;
		}

		mView.onResume();
		mView.queueEvent(new Runnable() {
			@Override
			public void run() {
				GodotLib.focusin();
			}
		});
		mSensorManager.registerListener(this, mAccelerometer, SensorManager.SENSOR_DELAY_GAME);
		mSensorManager.registerListener(this, mMagnetometer, SensorManager.SENSOR_DELAY_GAME);
		mSensorManager.registerListener(this, mGyroscope, SensorManager.SENSOR_DELAY_GAME);

		if(use_immersive && Build.VERSION.SDK_INT >= 19.0){ // check if the application runs on an android 4.4+
			Window window = getWindow();
			window.getDecorView().setSystemUiVisibility(
					    View.SYSTEM_UI_FLAG_LAYOUT_STABLE
						    | View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
						    | View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
						    | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION // hide nav bar
						    | View.SYSTEM_UI_FLAG_FULLSCREEN // hide status bar
						    | View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY);
		}

		for(int i=0;i<singleton_count;i++) {

			singletons[i].onMainResume();
		}



	}

	public void UiChangeListener() {
		final View decorView = getWindow().getDecorView();
		decorView.setOnSystemUiVisibilityChangeListener (new View.OnSystemUiVisibilityChangeListener() {
			@Override
			public void onSystemUiVisibilityChange(int visibility) {
				if ((visibility & View.SYSTEM_UI_FLAG_FULLSCREEN) == 0) {
					decorView.setSystemUiVisibility(
					View.SYSTEM_UI_FLAG_LAYOUT_STABLE
					| View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
					| View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
					| View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
					| View.SYSTEM_UI_FLAG_FULLSCREEN
					| View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY);
				}
			}
		});
	}

	@Override public void onSensorChanged(SensorEvent event) {
		Display display = ((WindowManager) getSystemService(WINDOW_SERVICE)).getDefaultDisplay();
		int displayRotation = display.getRotation();

		float[] adjustedValues = new float[3];
		final int axisSwap[][] = {
		{  1,  -1,  0,  1  },     // ROTATION_0
		{-1,  -1,  1,  0  },     // ROTATION_90
		{-1,    1,  0,  1  },     // ROTATION_180
		{  1,    1,  1,  0  }  }; // ROTATION_270

		final int[] as = axisSwap[displayRotation];
		adjustedValues[0]  =  (float)as[0] * event.values[ as[2] ];
		adjustedValues[1]  =  (float)as[1] * event.values[ as[3] ];
		adjustedValues[2]  =  event.values[2];

		final float x = adjustedValues[0];
		final float y = adjustedValues[1];
		final float z = adjustedValues[2];

		final int typeOfSensor = event.sensor.getType();
		if (mView != null) {
			mView.queueEvent(new Runnable() {
				@Override
				public void run() {
					if (typeOfSensor == Sensor.TYPE_ACCELEROMETER) {
						GodotLib.accelerometer(x,y,z);
					}
					if (typeOfSensor == Sensor.TYPE_MAGNETIC_FIELD) {
						GodotLib.magnetometer(x,y,z);
					}
					if (typeOfSensor == Sensor.TYPE_GYROSCOPE) {
						GodotLib.gyroscope(x,y,z);
					}
				}
			});
		}
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
		boolean shouldQuit = true;

		for(int i=0;i<singleton_count;i++) {
			if (singletons[i].onMainBackPressed()) {
				shouldQuit = false;
			}
		}

		System.out.printf("** BACK REQUEST!\n");
		if (shouldQuit && mView != null) {
			mView.queueEvent(new Runnable() {
				@Override
				public void run() {
					GodotLib.back();
				}
			});
		}
	}

	public void forceQuit() {

		System.exit(0);
	}



	private boolean obbIsCorrupted(String f, String main_pack_md5){

		    try {

			    InputStream fis =  new FileInputStream(f);

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
			    for (int i=0; i<messageDigest.length; i++) {
				    String s = Integer.toHexString(0xFF & messageDigest[i]);

				    if (s.length()==1) {
					s="0"+s;
				    }
				    hexString.append(s);
			    }
			    String md5str =  hexString.toString();

			    //Log.d("GODOT","**PACK** - My MD5: "+hexString+" - APK md5: "+main_pack_md5);
			    if (!md5str.equals(main_pack_md5)) {
				    Log.d("GODOT","**PACK MD5 MISMATCH???** - MD5 Found: "+md5str+" "+Integer.toString(md5str.length())+" - MD5 Expected: "+main_pack_md5+" "+Integer.toString(main_pack_md5.length()));
				    return true;
			    }
			    return false;
		    } catch (Exception e) {
			    e.printStackTrace();
			    Log.d("GODOT","**PACK FAIL**");
			    return true;
		    }
	}

	//@Override public boolean dispatchTouchEvent (MotionEvent event) {
	public boolean gotTouchEvent(final MotionEvent event) {

		super.onTouchEvent(event);
		final int evcount=event.getPointerCount();
		if (evcount==0)
			return true;

		if (mView != null) {
			final int[] arr = new int[event.getPointerCount()*3];

			for(int i=0;i<event.getPointerCount();i++) {

				arr[i*3+0]=(int)event.getPointerId(i);
				arr[i*3+1]=(int)event.getX(i);
				arr[i*3+2]=(int)event.getY(i);
			}

			//System.out.printf("gaction: %d\n",event.getAction());
			final int action = event.getAction() & MotionEvent.ACTION_MASK;
			mView.queueEvent(new Runnable() {
				@Override
				public void run() {
					switch(action) {
						case MotionEvent.ACTION_DOWN: {
							GodotLib.touch(0,0,evcount,arr);
							//System.out.printf("action down at: %f,%f\n", event.getX(),event.getY());
						} break;
						case MotionEvent.ACTION_MOVE: {
							GodotLib.touch(1,0,evcount,arr);
							/*
							for(int i=0;i<event.getPointerCount();i++) {
								System.out.printf("%d - moved to: %f,%f\n",i, event.getX(i),event.getY(i));
							}
							*/
						} break;
						case MotionEvent.ACTION_POINTER_UP: {
							final int indexPointUp = event.getActionIndex();
							final int pointer_idx = event.getPointerId(indexPointUp);
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
							/*
							for(int i=0;i<event.getPointerCount();i++) {
								System.out.printf("%d - up! %f,%f\n",i, event.getX(i),event.getY(i));
							}
							*/
						} break;
					}
				}
			});
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

	/*
	public void setPaymentsManager(PaymentsManager mPaymentsManager) {
		this.mPaymentsManager = mPaymentsManager;
	}
	*/


	// Audio

	/**
     * The download state should trigger changes in the UI --- it may be useful
     * to show the state as being indeterminate at times. This sample can be
     * considered a guideline.
     */
    @Override
    public void onDownloadStateChanged(int newState) {
    	Log.d("GODOT", "onDownloadStateChanged:" + newState);
        setState(newState);
        boolean showDashboard = true;
        boolean showCellMessage = false;
        boolean paused;
        boolean indeterminate;
        switch (newState) {
            case IDownloaderClient.STATE_IDLE:
            	Log.d("GODOT", "STATE IDLE");
                // STATE_IDLE means the service is listening, so it's
                // safe to start making calls via mRemoteService.
                paused = false;
                indeterminate = true;
                break;
            case IDownloaderClient.STATE_CONNECTING:
            case IDownloaderClient.STATE_FETCHING_URL:
            	Log.d("GODOT", "STATE CONNECTION / FETCHING URL");
                showDashboard = true;
                paused = false;
                indeterminate = true;
                break;
            case IDownloaderClient.STATE_DOWNLOADING:
            	Log.d("GODOT", "STATE DOWNLOADING");
                paused = false;
                showDashboard = true;
                indeterminate = false;
                break;

            case IDownloaderClient.STATE_FAILED_CANCELED:
            case IDownloaderClient.STATE_FAILED:
            case IDownloaderClient.STATE_FAILED_FETCHING_URL:
            case IDownloaderClient.STATE_FAILED_UNLICENSED:
            	Log.d("GODOT", "MANY TYPES OF FAILING");
                paused = true;
                showDashboard = false;
                indeterminate = false;
                break;
            case IDownloaderClient.STATE_PAUSED_NEED_CELLULAR_PERMISSION:
            case IDownloaderClient.STATE_PAUSED_WIFI_DISABLED_NEED_CELLULAR_PERMISSION:
            	Log.d("GODOT", "PAUSED FOR SOME STUPID REASON");
                showDashboard = false;
                paused = true;
                indeterminate = false;
                showCellMessage = true;
                break;

            case IDownloaderClient.STATE_PAUSED_BY_REQUEST:
            	Log.d("GODOT", "PAUSED BY STUPID USER");
                paused = true;
                indeterminate = false;
                break;
            case IDownloaderClient.STATE_PAUSED_ROAMING:
            case IDownloaderClient.STATE_PAUSED_SDCARD_UNAVAILABLE:
            	Log.d("GODOT", "PAUSED BY ROAMING      WTF!?");
                paused = true;
                indeterminate = false;
                break;
            case IDownloaderClient.STATE_COMPLETED:
            	Log.d("GODOT", "COMPLETED");
                showDashboard = false;
                paused = false;
                indeterminate = false;
//                validateXAPKZipFiles();
                initializeGodot();
                return;
            default:
            	Log.d("GODOT", "DEFAULT ????");
                paused = true;
                indeterminate = true;
                showDashboard = true;
        }
        int newDashboardVisibility = showDashboard ? View.VISIBLE : View.GONE;
        if (mDashboard.getVisibility() != newDashboardVisibility) {
            mDashboard.setVisibility(newDashboardVisibility);
        }
        int cellMessageVisibility = showCellMessage ? View.VISIBLE : View.GONE;
        if (mCellMessage.getVisibility() != cellMessageVisibility) {
            mCellMessage.setVisibility(cellMessageVisibility);
        }

        mPB.setIndeterminate(indeterminate);
        setButtonPausedState(paused);
    }


	@Override
	public void onDownloadProgress(DownloadProgressInfo progress) {
		mAverageSpeed.setText(getString(com.godot.game.R.string.kilobytes_per_second,
                Helpers.getSpeedString(progress.mCurrentSpeed)));
        mTimeRemaining.setText(getString(com.godot.game.R.string.time_remaining,
                Helpers.getTimeRemaining(progress.mTimeRemaining)));

        progress.mOverallTotal = progress.mOverallTotal;
        mPB.setMax((int) (progress.mOverallTotal >> 8));
        mPB.setProgress((int) (progress.mOverallProgress >> 8));
        mProgressPercent.setText(Long.toString(progress.mOverallProgress
                * 100 /
                progress.mOverallTotal) + "%");
        mProgressFraction.setText(Helpers.getDownloadProgressString
                (progress.mOverallProgress,
                        progress.mOverallTotal));

	}

}
