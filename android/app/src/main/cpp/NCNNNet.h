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
    bool predict(unsigned char *data, int width, int height, int format, int threadNum = 0);
    int getResSize();
    float *getRes();
protected:
    ncnn::Net *mNet;
    ncnn::Mat *mRes;
};


#endif //CNNDEMO_NCNNNET_H
