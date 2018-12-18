# Third party libraries


## Google's vending library

- Upstream: https://github.com/google/play-licensing/tree/master/lvl_library/src/main/java/com/google/android/vending
- Version: git (eb57657, 2018) with modifications
- License: Apache 2.0

Overwrite all files under `com/google/android/vending`

### Modify some files to avoid compile error and lint warning

#### com/google/android/vending/licensing/util/Base64.java
```
@@ -338,7 +338,8 @@ public class Base64 {
                        e += 4;
                }
 
-               assert (e == outBuff.length);
+               if (BuildConfig.DEBUG && e != outBuff.length)
+                       throw new RuntimeException();
                return outBuff;
        }
```

#### com/google/android/vending/licensing/LicenseChecker.java
```
@@ -29,8 +29,8 @@ import android.os.RemoteException;
 import android.provider.Settings.Secure;
 import android.util.Log;
 
-import com.android.vending.licensing.ILicenseResultListener;
-import com.android.vending.licensing.ILicensingService;
+import com.google.android.vending.licensing.ILicenseResultListener;
+import com.google.android.vending.licensing.ILicensingService;
 import com.google.android.vending.licensing.util.Base64;
 import com.google.android.vending.licensing.util.Base64DecoderException;
```
```
@@ -287,13 +287,15 @@ public class LicenseChecker implements ServiceConnection {
     if (logResponse) {
-        String android_id = Secure.getString(mContext.getContentResolver(),
-                            Secure.ANDROID_ID);
+        String android_id = Secure.ANDROID_ID;
         Date date = new Date();
```
