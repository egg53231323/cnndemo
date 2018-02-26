#ifndef UTILITY_H
#define UTILITY_H

#define ENABLE_ANDROID_LOG

#ifdef ENABLE_ANDROID_LOG
#include  <android/log.h>
#define LOGTAG "CNNDemoNative"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,LOGTAG,__VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOGTAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR,LOGTAG,__VA_ARGS__)
#else
#define LOGI(...)
#define LOGD(...)
#define LOGE(...)
#endif

#endif
