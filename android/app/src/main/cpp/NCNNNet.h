//
// Created by c on 2018/2/23.
//

#ifndef CNNDEMO_NCNNNET_H
#define CNNDEMO_NCNNNET_H

#include <vector>

namespace ncnn {
    class Net;
    class Mat;
}

class NCNNNet {
public:
    NCNNNet();
    virtual ~NCNNNet();

    bool load(const char *param, const char *model);
    void clear();
    void setMeanAndNorm(float *mean, int meanSize, float *norm, int normSize);
    bool predict(unsigned char *data, int width, int height, int format, int threadNum = 0);
    int getResSize();
    float *getRes();
protected:
    void resetMeanNorm();
    ncnn::Net *mNet;
    ncnn::Mat *mRes;
    float *mMeanVals;
    float *mNormVals;
};


#endif //CNNDEMO_NCNNNET_H
