/*************************************************************************/
/*  PaymentsCache.java                                                   */
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
package org.godotengine.godot.payments;

import android.content.Context;
import android.content.SharedPreferences;
import android.util.Log;

public class PaymentsCache {
	
	public Context context;

	public PaymentsCache(Context context){
		this.context = context;
	}
	
	
	public void setConsumableFlag(String set, String sku, Boolean flag){
		SharedPreferences sharedPref = context.getSharedPreferences("consumables_" + set, Context.MODE_PRIVATE); 
	    SharedPreferences.Editor editor = sharedPref.edit();
	    editor.putBoolean(sku, flag);
	    editor.commit();
}

	public boolean getConsumableFlag(String set, String sku){
	    SharedPreferences sharedPref = context.getSharedPreferences(
	    		"consumables_" + set, Context.MODE_PRIVATE);
	    return sharedPref.getBoolean(sku, false);
	}


	public void setConsumableValue(String set, String sku, String value){
		SharedPreferences sharedPref = context.getSharedPreferences("consumables_" + set, Context.MODE_PRIVATE); 
	    SharedPreferences.Editor editor = sharedPref.edit();
	    editor.putString(sku, value);
		//Log.d("XXX", "Setting asset: consumables_" + set + ":" + sku);
	    editor.commit();
	}

	public String getConsumableValue(String set, String sku){
	    SharedPreferences sharedPref = context.getSharedPreferences(
	    		"consumables_" + set, Context.MODE_PRIVATE);
		//Log.d("XXX", "Getting asset: consumables_" + set + ":" + sku);
	    return sharedPref.getString(sku, null);
	}

}
