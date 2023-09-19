REM Windows Templates
scons profile=TemplatesCustom.py platform=windows target=template_release arch=x86_32
scons platform=windows target=template_release arch=x86_64

REM Linux Templates
scons platform=linux target=template_release arch=x86_32
scons platform=linux target=template_release arch=x86_64

REM Android Templates
scons platform=android target=template_release arch=armv7
scons platform=android target=template_release arch=arm64v8
cd platform/android/java
.\gradlew cleanGodotTemplates
.\gradlew generateGodotTemplates
cd ../../..

REM Web Templates
scons platform=web target=template_release javascript_eval=no
scons platform=web target=template_debug javascript_eval=no