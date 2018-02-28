/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package com.example.android.tflitecamerademo;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.os.Environment;
import android.os.SystemClock;
import android.util.Log;

import com.baidu.mdl.demo.MDL;
import com.baidu.mdl.demo.MDLException;
import com.h3d.NCNNNet;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import org.tensorflow.lite.Interpreter;

/** Classifies images with Tensorflow Lite. */
public class ImageClassifier {

  public static final int TYPE_TF_LITE = 0;
  public static final int TYPE_NCNN = 1;
  public static final int TYPE_MDL = 2;

  /** Tag for the {@link Log}. */
  private static final String TAG = "TfLiteCameraDemo";

  /** Name of the model file stored in Assets. */
  private static final String MODEL_PATH = "mobilenet_quant_v1_224.tflite";

  /** Name of the label file stored in Assets. */
  private static final String LABEL_PATH = "labels.txt";

  /** Number of results to show in the UI. */
  private static final int RESULTS_TO_SHOW = 3;

  /** Dimensions of inputs. */
  private static final int DIM_BATCH_SIZE = 1;

  private static final int DIM_PIXEL_SIZE = 3;

  static final int DIM_IMG_SIZE_X = 224;
  static final int DIM_IMG_SIZE_Y = 224;

  /* Preallocated buffers for storing image data in. */
  private int[] intValues = new int[DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y];

  /** An instance of the driver class to run model inference with Tensorflow Lite. */
  private Interpreter tflite;

  /** Labels corresponding to the output of the vision model. */
  private List<String> labelList;

  /** A ByteBuffer to hold image data, to be feed into Tensorflow Lite as inputs. */
  private ByteBuffer imgData = null;

  /** An array to hold inference results, to be feed into Tensorflow Lite as outputs. */
  private byte[][] labelProbArray = null;
  /** multi-stage low pass filter * */
  private float[][] filterLabelProbArray = null;

  private static final int FILTER_STAGES = 3;
  private static final float FILTER_FACTOR = 0.4f;

  private String[] mTypeName;
  private int mType = TYPE_TF_LITE;
  private String mFileDir = "";

  private static final String NCNN_MOBILE_NET_PARAM_FILE_NAME = "mobilenet_v2.param";
  private static final String NCNN_MOBILE_NET_MODEL_FILE_NAME = "mobilenet_v2.bin";
  private static final String NCNN_GOOGLE_NET_PARAM_FILE_NAME = "ncnn_bvlc_googlenet.param";
  private static final String NCNN_GOOGLE_NET_MODEL_FILE_NAME = "ncnn_bvlc_googlenet.bin";
  private static final String NCNN_MOBILE_NET_LABEL_FILE_NAME = "mobilenet_v2.txt";
  private byte[] mImageData;
  private NCNNNet mNCNNNet;
  private List<String> mNCNNLabelList;

  private static final String MDL_MOBILE_NET_MODEL_FILE_NAME = "mdl_model.min.json";
  private static final String MDL_MOBILE_NET_PARAM_FILE_NAME = "mdl_data.min.bin";
  private static final String MDL_GOOGLE_NET_MODEL_FILE_NAME = "mdl_bvlc_googlenet.min.json";
  private static final String MDL_GOOGLE_NET_PARAM_FILE_NAME = "mdl_bvlc_googlenet.min.bin";
  private float[] mMDLImageData;
  private List<String> mMDLLabelList;
  private MDL mMDL;


  private float[] mGoogleNetRGBMean;
  private float[] mMobileNetRGBMean;
  private float[] mMobileNetRGBScale;

  private boolean mUseMobileNet = true;

  private PriorityQueue<Map.Entry<String, Float>> sortedLabels =
      new PriorityQueue<>(
          RESULTS_TO_SHOW,
          new Comparator<Map.Entry<String, Float>>() {
            @Override
            public int compare(Map.Entry<String, Float> o1, Map.Entry<String, Float> o2) {
              return (o1.getValue()).compareTo(o2.getValue());
            }
          });

  /** Initializes an {@code ImageClassifier}. */
  ImageClassifier(Activity activity) throws IOException {
    //const float mean_vals[3] = {103.94f, 116.78f, 123.68f};
    //const float norm_vals[3] = {0.017f,0.017f,0.017f};

    mMobileNetRGBMean = new float[3];
    mMobileNetRGBMean[0] = 103.94f;
    mMobileNetRGBMean[1] = 116.78f;
    mMobileNetRGBMean[2] = 123.68f;

    mMobileNetRGBScale = new float[3];
    mMobileNetRGBScale[0] = 0.017f;
    mMobileNetRGBScale[1] = 0.017f;
    mMobileNetRGBScale[2] = 0.017f;

    mGoogleNetRGBMean = new float[3];
    mGoogleNetRGBMean[0] = 103.94f;
    mGoogleNetRGBMean[1] = 116.78f;
    mGoogleNetRGBMean[2] = 123.68f;
    //mGoogleNetRGBMean[0] = 148;
    //mGoogleNetRGBMean[1] = 148;
    //mGoogleNetRGBMean[2] = 148;


    tflite = new Interpreter(loadModelFile(activity));
    labelList = loadLabelList(activity, LABEL_PATH);
    imgData =
        ByteBuffer.allocateDirect(
            DIM_BATCH_SIZE * DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE);
    imgData.order(ByteOrder.nativeOrder());
    labelProbArray = new byte[1][labelList.size()];
    filterLabelProbArray = new float[FILTER_STAGES][labelList.size()];
    Log.d(TAG, "Created a Tensorflow Lite Image Classifier.");

    mFileDir = activity.getFilesDir().getAbsolutePath()+ "/";
    mImageData = new byte[DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE];
    InitNCNN(activity);
    mNCNNLabelList = loadLabelList(activity, NCNN_MOBILE_NET_LABEL_FILE_NAME);
    for (int i = 0; i < mNCNNLabelList.size(); i++) {
      String label = mNCNNLabelList.get(i);
      if (null != label && label.length() > 12) {
        label = label.substring(11, label.length() - 1);
      }
      mNCNNLabelList.set(i, label);
    }

    InitMDL(activity);
    mMDLImageData = new float[DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE];
    mMDLLabelList = mNCNNLabelList;

    mTypeName = new String[3];
    mTypeName[TYPE_TF_LITE] = "TF_Lite";
    mTypeName[TYPE_NCNN] = "NCNN";
    mTypeName[TYPE_MDL] = "MDL";
  }

  String[] getTypeName() {
    return mTypeName;
  }

  /** Classifies a frame from the preview stream. */
  String classifyFrame(Bitmap bitmap) {
    convertBitmapToByteBuffer(bitmap);
    String textToShow = "";
    String textType = mTypeName[mType];
    switch (mType) {
      case TYPE_TF_LITE:
        if (tflite == null) {
          Log.e(TAG, "Image classifier has not been initialized; Skipped.");
          return "Uninitialized Classifier.";
        }
        textToShow = classifyFrame_TF_Lite();
        break;
      case TYPE_NCNN:
        textToShow = classifyFrame_NCNN();
        break;
      case TYPE_MDL:
        textToShow = classifyFrame_MDL();
        break;
      default:
        textToShow = classifyFrame_TF_Lite();
        break;
    }
    return textType + ": " + textToShow;
  }

  public void setType(int type) {
    mType = type;
  }

  String classifyFrame_TF_Lite() {
    // Here's where the magic happens!!!
    long startTime = SystemClock.uptimeMillis();
    tflite.run(imgData, labelProbArray);
    long endTime = SystemClock.uptimeMillis();
    //Log.d(TAG, "Timecost to run model inference: " + Long.toString(endTime - startTime));

    // Smooth the results across frames.
    //applyFilter();

    // Print the results.
    String textToShow = printTopKLabels();
    textToShow = Long.toString(endTime - startTime) + "ms" + textToShow;
    return textToShow;
  }

  String classifyFrame_NCNN() {
    long startTime = SystemClock.uptimeMillis();
    mNCNNNet.predict(mImageData, DIM_IMG_SIZE_X, DIM_IMG_SIZE_Y, NCNNNet.PIXEL_RGB);
    long endTime = SystemClock.uptimeMillis();
    float[] score = mNCNNNet.getScore();
    String textToShow = printTopKLabels(mNCNNLabelList, score);
    textToShow = Long.toString(endTime - startTime) + "ms" + textToShow;
    return textToShow;
  }

  String classifyFrame_MDL() {

    int pixel = 0;
    int offset = DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y;
    if (mUseMobileNet) {
      for (int i = 0; i < DIM_IMG_SIZE_X; ++i) {
        for (int j = 0; j < DIM_IMG_SIZE_Y; ++j) {
          mMDLImageData[pixel] = (float)((mImageData[pixel * DIM_PIXEL_SIZE] - mMobileNetRGBMean[0]) * mMobileNetRGBScale[0]);
          mMDLImageData[pixel + offset] = (float)((mImageData[pixel * DIM_PIXEL_SIZE + 1] - mMobileNetRGBMean[1]) * mMobileNetRGBScale[1]);
          mMDLImageData[pixel + offset * 2] = (float)((mImageData[pixel * DIM_PIXEL_SIZE + 2] - mMobileNetRGBMean[2]) * mMobileNetRGBScale[2]);
          //mMDLImageData[pixel * DIM_PIXEL_SIZE + 0] = (float)((mImageData[pixel * DIM_PIXEL_SIZE] - 123.68) * 0.017);
          //mMDLImageData[pixel * DIM_PIXEL_SIZE + 1] = (float)((mImageData[pixel * DIM_PIXEL_SIZE + 1] - 116.78) * 0.017);
          //mMDLImageData[pixel * DIM_PIXEL_SIZE + 2] = (float)((mImageData[pixel * DIM_PIXEL_SIZE + 2] - 103.94) * 0.017);
          pixel++;
        }
      }
    }
    else {
      for (int i = 0; i < DIM_IMG_SIZE_X; ++i) {
        for (int j = 0; j < DIM_IMG_SIZE_Y; ++j) {
          //mMDLImageData[pixel] = (float)((mImageData[pixel * DIM_PIXEL_SIZE] - mGoogleNetRGBMean[0]));
          //mMDLImageData[pixel + offset] = (float)((mImageData[pixel * DIM_PIXEL_SIZE + 1] - mGoogleNetRGBMean[1]));
          //mMDLImageData[pixel + offset * 2] = (float)((mImageData[pixel * DIM_PIXEL_SIZE + 2] - mGoogleNetRGBMean[2]));
          mMDLImageData[pixel * DIM_PIXEL_SIZE + 0] = (float)((mImageData[pixel * DIM_PIXEL_SIZE] - mGoogleNetRGBMean[0]));
          mMDLImageData[pixel * DIM_PIXEL_SIZE + 1] = (float)((mImageData[pixel * DIM_PIXEL_SIZE + 1] - mGoogleNetRGBMean[1]));
          mMDLImageData[pixel * DIM_PIXEL_SIZE + 2] = (float)((mImageData[pixel * DIM_PIXEL_SIZE + 2] - mGoogleNetRGBMean[2]));
          pixel++;
        }
      }
    }

    long startTime = SystemClock.uptimeMillis();
    float[] score = null;
    try {
      score = mMDL.predictImage(mMDLImageData);
    }
    catch (MDLException e) {
      e.printStackTrace();
    }
    long endTime = SystemClock.uptimeMillis();

    String textToShow = printTopKLabels(mMDLLabelList, score);
    textToShow = Long.toString(endTime - startTime) + "ms" + textToShow;
    return textToShow;
  }

  void applyFilter() {
    int numLabels = labelList.size();

    // Low pass filter `labelProbArray` into the first stage of the filter.
    for (int j = 0; j < numLabels; ++j) {
      filterLabelProbArray[0][j] +=
          FILTER_FACTOR * (labelProbArray[0][j] - filterLabelProbArray[0][j]);
    }
    // Low pass filter each stage into the next.
    for (int i = 1; i < FILTER_STAGES; ++i) {
      for (int j = 0; j < numLabels; ++j) {
        filterLabelProbArray[i][j] +=
            FILTER_FACTOR * (filterLabelProbArray[i - 1][j] - filterLabelProbArray[i][j]);
      }
    }

    // Copy the last stage filter output back to `labelProbArray`.
    for (int j = 0; j < numLabels; ++j) {
      labelProbArray[0][j] = (byte)filterLabelProbArray[FILTER_STAGES - 1][j];
    }
  }

  /** Closes tflite to release resources. */
  public void close() {
    tflite.close();
    tflite = null;
  }

  /** Reads label list from Assets. */
  private List<String> loadLabelList(Activity activity, String fileName) throws IOException {
    List<String> list = new ArrayList<String>();
    BufferedReader reader =
        new BufferedReader(new InputStreamReader(activity.getAssets().open(fileName)));
    String line;
    while ((line = reader.readLine()) != null) {
      list.add(line);
    }
    reader.close();
    return list;
  }

  /** Memory-map the model file in Assets. */
  private MappedByteBuffer loadModelFile(Activity activity) throws IOException {
    AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(MODEL_PATH);
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
  }

  /** Writes Image data into a {@code ByteBuffer}. */
  private void convertBitmapToByteBuffer(Bitmap bitmap) {
    if (imgData == null) {
      return;
    }
    imgData.rewind();
    // ARGB_8888
    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
    // Convert the image to floating point.
    int pixel = 0;
    long startTime = SystemClock.uptimeMillis();
    for (int i = 0; i < DIM_IMG_SIZE_X; ++i) {
      for (int j = 0; j < DIM_IMG_SIZE_Y; ++j) {
        final int val = intValues[pixel];
        imgData.put((byte) ((val >> 16) & 0xFF));
        imgData.put((byte) ((val >> 8) & 0xFF));
        imgData.put((byte) (val & 0xFF));

        mImageData[pixel * DIM_PIXEL_SIZE] = (byte) ((val >> 16) & 0xFF);
        mImageData[pixel * DIM_PIXEL_SIZE + 1] =(byte) ((val >> 8) & 0xFF);
        mImageData[pixel * DIM_PIXEL_SIZE + 2] =(byte) (val & 0xFF);
        pixel++;
      }
    }
    long endTime = SystemClock.uptimeMillis();
    //Log.d(TAG, "Timecost to put values into ByteBuffer: " + Long.toString(endTime - startTime));
  }

  /** Prints top-K labels, to be shown in UI as the results. */
  private String printTopKLabels() {
    for (int i = 0; i < labelList.size(); ++i) {
      sortedLabels.add(
          new AbstractMap.SimpleEntry<>(labelList.get(i), (labelProbArray[0][i] & 0xff) / 255.0f));
      if (sortedLabels.size() > RESULTS_TO_SHOW) {
        sortedLabels.poll();
      }
    }
    String textToShow = "";
    final int size = sortedLabels.size();
    for (int i = 0; i < size; ++i) {
      Map.Entry<String, Float> label = sortedLabels.poll();
      textToShow = String.format("\n%s: %4.2f", label.getKey(), label.getValue()) + textToShow;
    }
    return textToShow;
  }


  private void InitNCNN(Activity activity) {
    String paramFile = NCNN_MOBILE_NET_PARAM_FILE_NAME;
    String modelFile = NCNN_MOBILE_NET_MODEL_FILE_NAME;
    if (!mUseMobileNet) {
      paramFile = NCNN_GOOGLE_NET_PARAM_FILE_NAME;
      modelFile = NCNN_GOOGLE_NET_MODEL_FILE_NAME;
    }
    copyAssetToSDCard(activity, paramFile);
    copyAssetToSDCard(activity, modelFile);
    mNCNNNet = new NCNNNet();
    if (mUseMobileNet) {
      mNCNNNet.setMeanAndNorm(mMobileNetRGBMean, mMobileNetRGBScale);
    }
    else {
      mNCNNNet.setMeanAndNorm(mGoogleNetRGBMean, null);
    }

    String fileDir = mFileDir;
    boolean res = mNCNNNet.load(fileDir + paramFile, fileDir + modelFile);
    Log.i(TAG, "NCNN res " + res);
  }

  private void InitMDL(Activity activity) {
    String paramFile = MDL_MOBILE_NET_PARAM_FILE_NAME;
    String modelFile = MDL_MOBILE_NET_MODEL_FILE_NAME;
    if (!mUseMobileNet) {
      paramFile = MDL_GOOGLE_NET_PARAM_FILE_NAME;
      modelFile = MDL_GOOGLE_NET_MODEL_FILE_NAME;
    }
    copyAssetToSDCard(activity, paramFile);
    copyAssetToSDCard(activity, modelFile);
    String fileDir = mFileDir;
    try {
      mMDL = new MDL();
      boolean res = mMDL.load(mFileDir + modelFile, mFileDir + paramFile);
      Log.i(TAG, "MDL model load " + res);
      mMDL.setThreadNum(1);
    }
    catch (MDLException e) {
      e.printStackTrace();
    }

  }

  private String printTopKLabels(List<String> list, float[] score) {
    sortedLabels.clear();
    for (int i = 0; i < list.size(); ++i) {
      sortedLabels.add(new AbstractMap.SimpleEntry<>(list.get(i), score[i]));
      if (sortedLabels.size() > RESULTS_TO_SHOW) {
        sortedLabels.poll();
      }
    }
    String textToShow = "";
    final int size = sortedLabels.size();
    for (int i = 0; i < size; ++i) {
      Map.Entry<String, Float> label = sortedLabels.poll();
      String key = label.getKey();
      if (key.length() > 20) {
        key = key.substring(0, 20);
      }
      textToShow = String.format("\n%s: %4.2f", key, label.getValue()) + textToShow;
    }
    return textToShow;
  }

  private void copyAssetToSDCard(Activity activity, String fileName) {
    String fileDir = mFileDir;
    File dir = new File(fileDir);
    if (!dir.exists()) {
      dir.mkdir();
    }
    String filePath = fileDir + fileName;
    File file = new File(filePath);
    if (file.exists()) {
      file.delete();
    }
    try {
      InputStream inputStream = activity.getAssets().open(fileName);
      FileOutputStream fileOutputStream = new FileOutputStream(file);
      byte[] buffer = new byte[1024];
      int count = 0;
      while ((count = inputStream.read(buffer)) > 0) {
        fileOutputStream.write(buffer, 0, count);
      }
      fileOutputStream.flush();
      fileOutputStream.close();
      inputStream.close();
    }
    catch (IOException e) {
      e.printStackTrace();
    }
  }
}
