# Third-party libraries

This file list third-party libraries used in the Android source folder,
with their provenance and, when relevant, modifications made to those files.

## Google's vending library

- Upstream: https://github.com/google/play-licensing/tree/master/lvl_library/src/main/java/com/google/android/vending
- Version: git (eb57657, 2018) with modifications
- License: Apache 2.0

Overwrite all files under `com/google/android/vending`.

Modify those files to avoid compile error and lint warning:

- `com/google/android/vending/licensing/util/Base64.java`

```diff
@@ -338,7 +338,8 @@ public class Base64 {
                        e += 4;
                }
 
-               assert (e == outBuff.length);
+               if (BuildConfig.DEBUG && e != outBuff.length)
+                       throw new RuntimeException();
                return outBuff;
        }
```

- `com/google/android/vending/licensing/LicenseChecker.java`

```diff
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
