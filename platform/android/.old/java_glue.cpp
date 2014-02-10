#include "java_glue.h"
#include "os_android.h"
#include "main/main.h"
#include <unistd.h>
#include "file_access_jandroid.h"
#include "dir_access_jandroid.h"

static OS_Android *os_android=NULL;


struct TST {

	int a;
	TST() {

		a=5;
	}
};

TST tst;

struct JAndroidPointerEvent {

	Vector<OS_Android::TouchPos> points;
	int pointer;
	int what;
};

static List<JAndroidPointerEvent> pointer_events;
static bool initialized=false;
static Mutex *input_mutex=NULL;

JNIEXPORT void JNICALL Java_com_android_godot_GodotLib_init(JNIEnv * env, jobject obj,  jint width, jint height)
{


	if (initialized) // wtf
		return;

	__android_log_print(ANDROID_LOG_INFO,"godot","**INIT EVENT! - %p\n",env);


	initialized=true;

	__android_log_print(ANDROID_LOG_INFO,"godot","***************** HELLO FROM JNI!!!!!!!!");

	{
		//setup IO Object

		jclass cls = env->FindClass("com/android/godot/Godot");
		if (cls) {

			cls=(jclass)env->NewGlobalRef(cls);
			__android_log_print(ANDROID_LOG_INFO,"godot","*******CLASS FOUND!!!");
		}
		__android_log_print(ANDROID_LOG_INFO,"godot","STEP2, %p",cls);
		jfieldID fid = env->GetStaticFieldID(cls, "io", "Lcom/android/godot/GodotIO;");
		__android_log_print(ANDROID_LOG_INFO,"godot","STEP3 %i",fid);
		jobject ob = env->GetStaticObjectField(cls,fid);
		__android_log_print(ANDROID_LOG_INFO,"godot","STEP4, %p",ob);
		jobject gob = env->NewGlobalRef(ob);


		FileAccessJAndroid::setup(env,gob);
		DirAccessJAndroid::setup(env,gob);
	}



	os_android = new OS_Android(width,height);

	char wd[500];
	getcwd(wd,500);

	__android_log_print(ANDROID_LOG_INFO,"godot","test construction %i\n",tst.a);
	__android_log_print(ANDROID_LOG_INFO,"godot","running from dir %s\n",wd);

	__android_log_print(ANDROID_LOG_INFO,"godot","**SETUP");



#if 0
	char *args[]={"-test","render",NULL};
	__android_log_print(ANDROID_LOG_INFO,"godot","pre asdasd setup...");
	Error err  = Main::setup("apk",2,args);
#else
	Error err  = Main::setup("apk",0,NULL);
#endif
	if (err!=OK) {
		__android_log_print(ANDROID_LOG_INFO,"godot","*****UNABLE TO SETUP");

		return; //should exit instead and print the error
	}

	__android_log_print(ANDROID_LOG_INFO,"godot","**START");


	if (!Main::start()) {

		return; //should exit instead and print the error
	}
	input_mutex=Mutex::create();

	os_android->main_loop_begin();


}

JNIEXPORT void JNICALL Java_com_android_godot_GodotLib_step(JNIEnv * env, jobject obj)
{

	__android_log_print(ANDROID_LOG_INFO,"godot","**STEP EVENT! - %p-%i\n",env,Thread::get_caller_ID());




	{

		FileAccessJAndroid::update_env(env);
		DirAccessJAndroid::update_env(env);
	}

	input_mutex->lock();

	while(pointer_events.size()) {

		JAndroidPointerEvent jpe=pointer_events.front()->get();
		os_android->process_touch(jpe.what,jpe.pointer,jpe.points);
		pointer_events.pop_front();
	}

	input_mutex->unlock();


	if (os_android->main_loop_iterate()==true) {

		return; //should exit instead
	}


}

JNIEXPORT void JNICALL Java_com_android_godot_GodotLib_touch(JNIEnv * env, jobject obj, jint ev,jint pointer, jint count, jintArray positions) {



	__android_log_print(ANDROID_LOG_INFO,"godot","**TOUCH EVENT! - %p-%i\n",env,Thread::get_caller_ID());





	Vector<OS_Android::TouchPos> points;
	for(int i=0;i<count;i++) {

		jint p[3];
		env->GetIntArrayRegion(positions,i*3,3,p);
		OS_Android::TouchPos tp;
		tp.pos=Point2(p[1],p[2]);
		tp.id=p[0];
		points.push_back(tp);
	}

	JAndroidPointerEvent jpe;
	jpe.pointer=pointer;
	jpe.points=points;
	jpe.what=ev;

	input_mutex->lock();

	pointer_events.push_back(jpe);

	input_mutex->unlock();
	//if (os_android)
//		os_android->process_touch(ev,pointer,points);

}

//Main::cleanup();

//return os.get_exit_code();
