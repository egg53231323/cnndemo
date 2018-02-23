//
// Created by c on 2018/2/23.
//

#include "NCNNNet.h"
#include "../libs/ncnn-android-lib/include/net.h"

NCNNNet::NCNNNet(): mNet(NULL) {
    mNet = new ncnn::Net();
}

NCNNNet::~NCNNNet() {
    if (NULL != mNet) {
        delete mNet;
        mNet = NULL;
    }
}

bool NCNNNet::load(const char *param, const char *model) {
    mNet->load_param(param);
    mNet->load_model(model);
    return true;
}

void NCNNNet::clear() {
    mNet->clear();
}

bool NCNNNet::predict(unsigned char *data, int width, int height, std::vector<float> &scores) {
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(data, ncnn::Mat::PIXEL_BGR, width, height, 224, 224);

    const float mean_vals[3] = {103.94f, 116.78f, 123.68f};
    const float norm_vals[3] = {0.017f,0.017f,0.017f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = mNet->create_extractor();
    ex.set_light_mode(true);

    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("fc7", out);  //此处与squeezenet不同

    scores.resize(out.c);
    for (int j=0; j<out.c; j++)
    {
        //const float *prob = (const float *)(out.data + out.cstep * j);
        //scores[j] = prob[0];
    }
    return true;
}

