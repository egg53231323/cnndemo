#include <jni.h>
#include <string>
#include "../libs/ncnn-android-lib/include/net.h"

extern "C"
JNIEXPORT jstring

JNICALL
Java_com_example_android_tflitecamerademo_CameraActivity_stringFromJNI(
        JNIEnv *env,
        jobject /* this */) {
    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}
