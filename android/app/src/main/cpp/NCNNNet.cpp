//
// Created by c on 2018/2/23.
//

#include "NCNNNet.h"
#include "../libs/ncnn-android-lib/include/net.h"
#include "Utility.h"

NCNNNet::NCNNNet(): mNet(NULL), mRes(NULL) {
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

bool NCNNNet::predict(unsigned char *data, int width, int height, int format, int threadNum /* = 0*/) {
    ncnn::Mat in = ncnn::Mat::from_pixels(data, format, width, height);

    const float mean_vals[3] = {103.94f, 116.78f, 123.68f};
    const float norm_vals[3] = {0.017f,0.017f,0.017f};
    in.substract_mean_normalize(mean_vals, norm_vals);

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

