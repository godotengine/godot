# apksig ([commit ac5cbb07d87cc342fcf07715857a812305d69888](https://android.googlesource.com/platform/tools/apksig/+/ac5cbb07d87cc342fcf07715857a812305d69888))

apksig is a project which aims to simplify APK signing and checking whether APK signatures are
expected to verify on Android. apksig supports
[JAR signing](https://docs.oracle.com/javase/8/docs/technotes/guides/jar/jar.html#Signed_JAR_File)
(used by Android since day one) and
[APK Signature Scheme v2](https://source.android.com/security/apksigning/v2.html) (supported since
Android Nougat, API Level 24). apksig is meant to be used outside of Android devices.

The key feature of apksig is that it knows about differences in APK signature verification logic
between different versions of the Android platform. apksig thus thoroughly checks whether an APK's
signature is expected to verify on all Android platform versions supported by the APK. When signing
an APK, apksig chooses the most appropriate cryptographic algorithms based on the Android platform
versions supported by the APK being signed.

## apksig library

apksig library offers three primitives:

* `ApkSigner` which signs the provided APK so that it verifies on all Android platform versions
  supported by the APK. The range of platform versions can be customized.
* `ApkVerifier` which checks whether the provided APK is expected to verify on all Android
  platform versions supported by the APK. The range of platform versions can be customized.
* `(Default)ApkSignerEngine` which abstracts away signing APKs from parsing and building APKs.
  This is useful in optimized APK building pipelines, such as in Android Plugin for Gradle,
  which need to perform signing while building an APK, instead of after. For simpler use cases
  where the APK to be signed is available upfront, the `ApkSigner` above is easier to use.

_NOTE: Some public classes of the library are in packages having the word "internal" in their name.
These are not public API of the library. Do not use \*.internal.\* classes directly because these
classes may change any time without regard to existing clients outside of `apksig` and `apksigner`._

