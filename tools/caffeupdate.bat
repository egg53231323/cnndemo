set caffe_bin_dir=..\..\caffe\bin
set update_prototxt=..\..\MobileNet-Caffe\mobilenet_deploy.prototxt
set update_model=..\..\MobileNet-Caffe\mobilenet.caffemodel
set out_name=mobilenet

%caffe_bin_dir%\upgrade_net_proto_text.exe %update_prototxt% %out_name%.prototxt
%caffe_bin_dir%\upgrade_net_proto_binary.exe %update_model% %out_name%.caffemodel