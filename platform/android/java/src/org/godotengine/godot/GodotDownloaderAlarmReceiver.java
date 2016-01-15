package org.godotengine.godot;

import com.google.android.vending.expansion.downloader.DownloaderClientMarshaller;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager.NameNotFoundException;
import android.util.Log;

/**
 * You should start your derived downloader class when this receiver gets the message
 * from the alarm service using the provided service helper function within the
 * DownloaderClientMarshaller. This class must be then registered in your AndroidManifest.xml
 * file with a section like this:
 *         <receiver android:name=".GodotDownloaderAlarmReceiver"/>
 */
public class GodotDownloaderAlarmReceiver extends BroadcastReceiver {

    @Override
    public void onReceive(Context context, Intent intent) {
    	Log.d("GODOT", "Alarma recivida");
	try {
	    DownloaderClientMarshaller.startDownloadServiceIfRequired(context, intent, GodotDownloaderService.class);
	} catch (NameNotFoundException e) {
	    e.printStackTrace();
	    Log.d("GODOT", "Exception: " + e.getClass().getName() + ":" + e.getMessage());
	}
    }
}
