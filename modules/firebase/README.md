## Firebase SDK wrapper

official guide doc : https://firebase.google.com/docs/guides

### Enabling module

#### Downloads

This module working on specific version of firebase cpp sdk. 
[5.4.4](https://dl.google.com/firebase/sdk/cpp/firebase_cpp_sdk_5.4.4.zip).   
You can try bump up lastest version of firebase sdk.
[latest](https://firebase.google.com/download/cpp)

#### Compiling

Extract firebase cpp sdk into 
[thirdparty/firebase](../../thirdparty/firebase)

Tested platforms
 - osx
 - windows(MSVC only)
 - android

##### Android

Firebase SDK and Federated identity providers initialized with application resources. modify values in 
[android/res/values](../../modules/firebase/android/res/values).
googleservices.xml can simply replace by firebase cpp sdk's generate_xml_from_google_services_json tool

### Demo

pretty simple demo 
[here](https://github.com/paper-vessel/godot-demo)
