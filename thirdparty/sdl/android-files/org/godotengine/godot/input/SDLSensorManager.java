package org.godotengine.godot.input;

import android.hardware.Sensor;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.util.Log;

// This class coordinates synchronized access to sensor manager registration
//
// This prevents a java.util.ConcurrentModificationException exception on
// Android 16, specifically on the Samsung Tab S9 Ultra.

class SDLSensorManager
{
    static private SDLSensorManager mManager = new SDLSensorManager();

    static final int RETRY_COUNT = 3;

    public static void registerListener(SensorManager manager, SensorEventListener listener, Sensor sensor, int samplingPeriodUs) {
        mManager.RegisterListener(manager, listener, sensor, samplingPeriodUs);
    }

    public static void unregisterListener(SensorManager manager, SensorEventListener listener, Sensor sensor) {
        mManager.UnregisterListener(manager, listener, sensor);
    }

    private synchronized void RegisterListener(SensorManager manager, SensorEventListener listener, Sensor sensor, int samplingPeriodUs) {
        int retries = 0;
        boolean complete = false;
        while (!complete) {
            try {
                manager.registerListener(listener, sensor, samplingPeriodUs, null);
                complete = true;
            } catch (java.util.ConcurrentModificationException e) {
                ++retries;
                if (retries <= RETRY_COUNT) {
                    // Sleep a bit and try again
                    try {
                        Thread.sleep(1);
                    } catch (Exception e2) {
                    }
                } else {
                    Log.v("SDL", "Multiple ConcurrentModificationException caught while registering sensor listener, canceling operation");
                    complete = true;
                }
            }
        }
    }

    private synchronized void UnregisterListener(SensorManager manager, SensorEventListener listener, Sensor sensor) {
        int retries = 0;
        boolean complete = false;
        while (!complete) {
            try {
                manager.unregisterListener(listener, sensor);
                complete = true;
            } catch (java.util.ConcurrentModificationException e) {
                ++retries;
                if (retries <= RETRY_COUNT) {
                    // Sleep a bit and try again
                    try {
                        Thread.sleep(1);
                    } catch (Exception e2) {
                    }
                } else {
                    Log.v("SDL", "Multiple ConcurrentModificationException caught while unregistering sensor listener, canceling operation");
                    complete = true;
                }
            }
        }
    }
}

