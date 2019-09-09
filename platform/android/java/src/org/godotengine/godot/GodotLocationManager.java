package org.godotengine.godot;

import android.Manifest;
import android.content.pm.PackageManager;
import android.location.Location;
import android.util.Log;

import androidx.core.content.ContextCompat;

import com.google.android.gms.location.FusedLocationProviderClient;
import com.google.android.gms.location.LocationCallback;
import com.google.android.gms.location.LocationRequest;
import com.google.android.gms.location.LocationResult;
import com.google.android.gms.location.LocationServices;

public class GodotLocationManager {

    public static final String ACCESS_COARSE_LOCATION_PERMISSION = "ACCESS_COARSE_LOCATION";
    public static final String ACCESS_FINE_LOCATION_PERMISSION = "ACCESS_FINE_LOCATION";

    private static GodotLocationManager singleton;
    private Godot mActivity;
    private LocationCallback mLocationCallback;
    private FusedLocationProviderClient mFusedLocationClient;
    private boolean locationUpdatesStart = false;

    private GodotLocationManager(Godot activity) {
        this.mActivity = activity;
        mFusedLocationClient = LocationServices.getFusedLocationProviderClient(activity);
    }

    static GodotLocationManager getInstance(Godot activity) {
        if (singleton == null)
            singleton = new GodotLocationManager(activity);
        return singleton;
    }

    private void initializeLocationCallback() {
        if (mActivity == null)
            return;

        locationUpdatesStart = true;
        mLocationCallback = new LocationCallback() {
            @Override
            public void onLocationResult(LocationResult locationResult) {
                if (locationResult == null || locationResult.getLastLocation() == null) {
                    Log.d("GODOT", "Location object is null.");
                    return;
                }

                Location location = locationResult.getLastLocation();
                double longitude = location.getLongitude();
                double latitude = location.getLatitude();
                float accuracy = location.getAccuracy();
                float verticalAccuracyMeters = 0.0f;
                double altitude = location.getAltitude();
                float speed = location.getSpeed();
                long time = location.getTime();

                if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
                    verticalAccuracyMeters = location.getVerticalAccuracyMeters();
                }

                GodotLib.updateLocation((float) longitude, (float) latitude, accuracy, verticalAccuracyMeters,
                        (float) altitude, speed, time);

            }
        };

    }

    void startLocationUpdates(long interval, long maxWaitTime, int priority) {
        if (locationUpdatesStart)
            return;

        if (ContextCompat.checkSelfPermission(mActivity,
                Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
            Log.d("GODOT", "If you want to location requests, you need to give permission \"ACCESS_FINE_LOCATION\".");
            return;
        }

        initializeLocationCallback();

        LocationRequest request = new LocationRequest();
        request.setInterval(interval);
        request.setMaxWaitTime(maxWaitTime);

        switch (priority) {
            case LocationRequest.PRIORITY_BALANCED_POWER_ACCURACY:
                request.setPriority(LocationRequest.PRIORITY_BALANCED_POWER_ACCURACY);
                break;
            case LocationRequest.PRIORITY_LOW_POWER:
                request.setPriority(LocationRequest.PRIORITY_LOW_POWER);
                break;
            case LocationRequest.PRIORITY_NO_POWER:
                request.setPriority(LocationRequest.PRIORITY_NO_POWER);
                break;
            default:
                request.setPriority(LocationRequest.PRIORITY_HIGH_ACCURACY);
        }

        mFusedLocationClient.requestLocationUpdates(request,
                mLocationCallback, null);
        Log.d("GODOT", "Location update started.");
    }

    void stopLocationUpdates() {
        if (locationUpdatesStart && mLocationCallback != null) {
            locationUpdatesStart = false;
            mFusedLocationClient.removeLocationUpdates(mLocationCallback);
            Log.d("GODOT", "Location update stopped.");
        }
    }

}
