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
//	    Log.d("XXX", "Setting asset: consumables_" + set + ":" + sku);
	    editor.commit();
	}

	public String getConsumableValue(String set, String sku){
	    SharedPreferences sharedPref = context.getSharedPreferences(
	    		"consumables_" + set, Context.MODE_PRIVATE);
//	    Log.d("XXX", "Getting asset: consumables_" + set + ":" + sku);
	    return sharedPref.getString(sku, null);
	}

}
