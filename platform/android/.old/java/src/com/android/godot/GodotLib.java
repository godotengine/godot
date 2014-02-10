package com.android.godot;

// Wrapper for native library

public class GodotLib {


     public static GodotIO io;

     static {
	 System.loadLibrary("godot_android");
     }

    /**
     * @param width the current view width
     * @param height the current view height
     */

     public static native void init(int width, int height);
     public static native void step();
     public static native void touch(int what,int pointer,int howmany, int[] arr);
}
