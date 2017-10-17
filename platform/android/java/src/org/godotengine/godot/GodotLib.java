/*************************************************************************/
/*  GodotLib.java                                                        */
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

     public static native void initialize(Godot p_instance,boolean need_reload_hook,Object p_asset_manager, boolean use_apk_expansion);
		 public static native void setup(String[] p_cmdline);
     public static native void resize(int width, int height,boolean reload);
     public static native void newcontext(boolean p_32_bits);
     public static native void back();
     public static native void step();
     public static native void touch(int what,int pointer,int howmany, int[] arr);
     public static native void accelerometer(float x, float y, float z);
     public static native void magnetometer(float x, float y, float z);
     public static native void gyroscope(float x, float y, float z);
	 public static native void key(int p_scancode, int p_unicode_char, boolean p_pressed);
	 public static native void joybutton(int p_device, int p_but, boolean p_pressed);
	 public static native void joyaxis(int p_device, int p_axis, float p_value);
	 public static native void joyhat(int p_device, int p_hat_x, int p_hat_y);
	 public static native void joyconnectionchanged(int p_device, boolean p_connected, String p_name);
     public static native void focusin();
     public static native void focusout();
     public static native void audio();
     public static native void singleton(String p_name,Object p_object);
     public static native void method(String p_sname,String p_name,String p_ret,String[] p_params);
     public static native String getGlobal(String p_key);
	public static native void callobject(int p_ID, String p_method, Object[] p_params);
	public static native void calldeferred(int p_ID, String p_method, Object[] p_params);

	public static native void setVirtualKeyboardHeight(int p_height);

}
