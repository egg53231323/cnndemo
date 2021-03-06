package com.h3d;

/**
 * Created by c on 2018/2/23.
 */

public class NCNNNet {
    private long mObj = 0;
    private float[] mScore = new float[1000];
    public final static int PIXEL_RGB = 1;
    public final static int PIXEL_BGR = (1 << 1);
    public final static int PIXEL_GRAY = (1 << 2);
    public final static int PIXEL_RGBA = (1 << 3);
    public NCNNNet() {
        mObj = nativeCreativeInstance();
    }

    public void ReleaseResource() {
        nativeDestroyInstance(mObj);
    }

    public boolean load(String paramPath, String modelPath) {
        return nativeLoad(mObj, paramPath, modelPath);
    }

    public boolean predict(byte[] data, int width, int height, int format) {
        return nativePredict(mObj, data, width, height, format, 4, mScore);
    }

    public void setMeanAndNorm(float[] mean, float[] norm) {
        nativeSetMeanAndNorm(mObj, mean, norm);
    }

    public float[] getScore() {
        return mScore;
    }

    private native long nativeCreativeInstance();
    private native void nativeDestroyInstance(long obj);
    private native boolean nativeLoad(long obj, String paramPath, String modelPath);
    private native boolean nativePredict(long obj, byte[] data, int width, int height, int format, int threadNum, float[] score);
    private native void nativeSetMeanAndNorm(long obj, float[] mean, float[] norm);
}
