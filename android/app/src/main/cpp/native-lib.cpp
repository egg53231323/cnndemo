#include <jni.h>
#include <string>
#include "NCNNNet.h"
#include "Utility.h"

extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_android_tflitecamerademo_CameraActivity_stringFromJNI(
        JNIEnv *env,
        jobject /* this */) {
    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}

extern "C"
JNIEXPORT jlong JNICALL
Java_com_h3d_NCNNNet_nativeCreativeInstance(JNIEnv *env, jobject instance) {
    return (jlong)new NCNNNet();
}

extern "C"
JNIEXPORT void JNICALL
Java_com_h3d_NCNNNet_nativeDestroyInstance(JNIEnv *env, jobject instance, jlong obj) {
    NCNNNet *net = (NCNNNet*)obj;
    if (NULL != net) {
        delete net;
        net = NULL;
    }
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_h3d_NCNNNet_nativeLoad(JNIEnv *env, jobject instance, jlong obj, jstring paramPath_,
                                jstring modelPath_) {
    NCNNNet *net = (NCNNNet*)obj;
    if (NULL == net) {
        return false;
    }

    const char *paramPath = env->GetStringUTFChars(paramPath_, 0);
    const char *modelPath = env->GetStringUTFChars(modelPath_, 0);

    bool res = net->load(paramPath, modelPath);

    env->ReleaseStringUTFChars(paramPath_, paramPath);
    env->ReleaseStringUTFChars(modelPath_, modelPath);
    return true;
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_h3d_NCNNNet_nativePredict(JNIEnv *env, jobject instance, jlong obj, jbyteArray data_,
                                   jint width, jint height, jint format, jfloatArray score_) {
    NCNNNet *net = (NCNNNet*)obj;
    if (NULL == net || NULL == data_ || NULL == score_) {
        return false;
    }

    jbyte *data = env->GetByteArrayElements(data_, NULL);
    jfloat *score = env->GetFloatArrayElements(score_, NULL);

    bool res = net->predict((unsigned char*)data, width, height, format);
    if (res) {
        int size = net->getResSize();
        LOGI("ncnn predict size %d", size);
        if (size > 0) {
            float *resData = net->getRes();
            int scoreArrayLength = env->GetArrayLength(score_);
            if (scoreArrayLength >= size) {
                env->SetFloatArrayRegion(score_, 0, size, resData);
            }
            else {
                LOGE("scoreArrayLength %d < ncnn res length %d", scoreArrayLength, res);
            }
        }
    }

    env->ReleaseByteArrayElements(data_, data, 0);
    env->ReleaseFloatArrayElements(score_, score, 0);
    return res;
}