if not exist model (
    md model
)
if not exist model\ncnn (
    md model\ncnn
)
set caffe2ncnn_model_name=mobilenet
ncnn\caffe2ncnn.exe %caffe2ncnn_model_name%.prototxt %caffe2ncnn_model_name%.caffemodel model\ncnn\%caffe2ncnn_model_name%.param model\ncnn\%caffe2ncnn_model_name%.bin