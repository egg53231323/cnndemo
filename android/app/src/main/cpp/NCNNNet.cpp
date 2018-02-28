//
// Created by c on 2018/2/23.
//

#include "NCNNNet.h"
#include "../libs/ncnn-android-lib/include/net.h"
#include "Utility.h"

NCNNNet::NCNNNet(): mNet(NULL), mRes(NULL), mMeanVals(NULL), mNormVals(NULL) {
    mNet = new ncnn::Net();
    mRes = new ncnn::Mat();
}

NCNNNet::~NCNNNet() {
    if (NULL != mNet) {
        delete mNet;
        mNet = NULL;
    }
    if (NULL != mRes) {
        delete mRes;
        mRes = NULL;
    }
    resetMeanNorm();
}

bool NCNNNet::load(const char *param, const char *model) {
    int res = mNet->load_param(param);
    if (0 != res) {
        LOGE("ncnn net load_param failed. res: %d", res);
        return false;
    }
    res = mNet->load_model(model);
    if (0 != res) {
        LOGE("ncnn net load_model failed. res: %d", res);
        return false;
    }
    LOGI("ncnn net load succeed");
    return true;
}

void NCNNNet::clear() {
    mNet->clear();
}

void NCNNNet::setMeanAndNorm(float *mean, int meanSize, float *norm, int normSize) {
    resetMeanNorm();
    if (NULL != mean && meanSize > 0) {
        mMeanVals = new float[meanSize];
        memcpy(mMeanVals, mean, sizeof(float) * meanSize);
    }
    if (NULL != norm && normSize > 0) {
        mNormVals = new float[normSize];
        memcpy(mNormVals, norm, sizeof(float) * normSize);
    }
}

void NCNNNet::resetMeanNorm() {
    if (NULL != mMeanVals) {
        delete []mMeanVals;
        mMeanVals = NULL;
    }
    if (NULL != mNormVals) {
        delete []mNormVals;
        mNormVals = NULL;
    }
}

bool NCNNNet::predict(unsigned char *data, int width, int height, int format, int threadNum /* = 0*/) {
    ncnn::Mat in = ncnn::Mat::from_pixels(data, format, width, height);
    
    in.substract_mean_normalize(mMeanVals, mNormVals);

    ncnn::Extractor ex = mNet->create_extractor();
    if (threadNum > 0) {
        ex.set_num_threads(threadNum);
    }
    ex.set_light_mode(true);

    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("prob", out);

    *mRes = out.reshape(out.w * out.h * out.c);
    return true;
}

int NCNNNet::getResSize() {
    return mRes->w;
}

float *NCNNNet::getRes() {
    return (float*)mRes->data;
}

