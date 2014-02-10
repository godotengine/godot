package com.android.godot;

import android.app.Activity;
import android.os.Bundle;
import android.view.MotionEvent;

public class Godot extends Activity
{


	GodotView mView;

	static public GodotIO io;


	@Override protected void onCreate(Bundle icicle) {
		super.onCreate(icicle);
		io = new GodotIO(getAssets());
		GodotLib.io=io;
		mView = new GodotView(getApplication(),io);
		setContentView(mView);
	}

	@Override protected void onPause() {
		super.onPause();
		mView.onPause();
	}

	@Override protected void onResume() {
		super.onResume();
		mView.onResume();
	}

	@Override public boolean dispatchTouchEvent (MotionEvent event) {

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

}
