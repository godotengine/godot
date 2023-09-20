REM Windows Templates
REM scons profile=TemplatesCustom.py platform=windows target=template_release arch=x86_32
scons profile=TemplatesCustom.py platform=windows target=template_release arch=x86_64

REM Linux Templates
REM scons platform=linux target=template_release arch=x86_32
REM scons profile=TemplatesCustom.py platform=linux target=template_release arch=x86_64

REM Android Templates
scons profile=TemplatesCustom.py platform=android target=template_release arch=armv7
scons profile=TemplatesCustom.py platform=android target=template_release arch=arm64v8
cd platform/android/java
.\gradlew cleanGodotTemplates
.\gradlew generateGodotTemplates
cd ../../..

REM Web Templates
scons profile=TemplatesWeb.py platform=web target=template_release javascript_eval=no