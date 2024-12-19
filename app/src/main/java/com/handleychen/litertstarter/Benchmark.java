package com.handleychen.litertstarter;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.os.Trace;
import android.util.Log;
import android.util.Pair;
import java.io.Closeable;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.security.DigestInputStream;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Random;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

public class Benchmark implements Closeable {

    private static final String TAG = "Benchmark";

    private final Context context;
    private Interpreter tfLiteInterpreter;
    private TensorBuffer inputTensorBuffer, outputTensorBuffer;
    private int width, height;

    public Benchmark(Context c) {
        context = c;
    }

    /**
     * Load a TF Lite model from disk.
     *
     * @param assets Android app asset manager.
     * @param modelFilename File name of the resource to load.
     * @return The loaded model in MappedByteBuffer format, and a unique model identifier hash string.
     * @throws IOException If the model file does not exist or cannot be read.
     */
    private static Pair<MappedByteBuffer, String> loadModelFile(AssetManager assets, String modelFilename)
            throws IOException, NoSuchAlgorithmException {
        AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
        MappedByteBuffer buffer;
        String hash;

        try (FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor())) {
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();

            // Map the file to a buffer
            buffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);

            // Compute the hash
            MessageDigest hashDigest = MessageDigest.getInstance("MD5");
            inputStream.skip(startOffset);
            try (DigestInputStream dis = new DigestInputStream(inputStream, hashDigest)) {
                byte[] data = new byte[8192];
                int numRead = 0;
                while (numRead < declaredLength) {
                    numRead += dis.read(data, 0, Math.min(8192, (int) declaredLength - numRead));
                }
                // Computing MD5 hash
            }

            // Convert hash to string
            StringBuilder hex = new StringBuilder();
            for (byte b : hashDigest.digest()) {
                hex.append(String.format("%02x", b));
            }
            hash = hex.toString();
        }

        return new Pair<>(buffer, hash);
    }

    private static String constructShapeString(int[] shape) {
        if (shape == null || shape.length == 0) {
            return "[]";
        }

        StringBuilder sb = new StringBuilder();
        sb.append("[");

        for (int i = 0; i < shape.length; i++) {
            sb.append(shape[i]);
            if (i < shape.length - 1) {
                sb.append(", ");
            }
        }

        sb.append("]");
        return sb.toString();
    }

    public void createInterpreter() {
        String modelPath = "quicksrnetsmall.tflite"; // mobilenetv1, quicksrnetsmall，quicksrnetsmall_quantized,
        Pair<MappedByteBuffer, String> modelAndHash;
        try {
            modelAndHash = loadModelFile(context.getAssets(), modelPath);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        Interpreter.Options tfLiteOptions = new Interpreter.Options();
        tfLiteOptions.setRuntime(Interpreter.Options.TfLiteRuntime.FROM_APPLICATION_ONLY);
        tfLiteOptions.setAllowBufferHandleOutput(true);
        tfLiteOptions.setUseNNAPI(false);
        tfLiteOptions.setNumThreads(Runtime.getRuntime().availableProcessors() / 2);
        tfLiteOptions.setUseXNNPACK(true);

        tfLiteInterpreter = new Interpreter(modelAndHash.first, tfLiteOptions);
        tfLiteInterpreter.allocateTensors();
    }

    public void createInputOutput() {
        // create input & output buffer
        Tensor inputTensor = tfLiteInterpreter.getInputTensor(0);
        int[] inputShape = inputTensor.shape();
        DataType inputType = inputTensor.dataType();
        Tensor outputTensor = tfLiteInterpreter.getOutputTensor(0);
        int[] outputShape = outputTensor.shape();
        DataType outputType = outputTensor.dataType();
        Log.i(TAG, " inputShape=" + constructShapeString(inputShape) + " inputType=" + inputType.toString()
                + " outputShape=" + constructShapeString(outputShape) + " outputType=" + outputType.toString());
        inputTensorBuffer = TensorBuffer.createFixedSize(inputShape, inputType);
        outputTensorBuffer = TensorBuffer.createFixedSize(outputShape, outputType);
        if (inputShape.length == 4) {
            width = inputShape[2];
            height = inputShape[1];
        }

        // generate random data
        Random random = new Random();
        ByteBuffer inputByteBuffer = inputTensorBuffer.getBuffer();
        while (inputByteBuffer.hasRemaining()) {
            inputByteBuffer.put((byte) random.nextInt(256));
        }
        inputByteBuffer.rewind();
    }

    public void warmup() {
        for (int i = 0; i < 3; i++) {
            outputTensorBuffer.getBuffer().clear();
            tfLiteInterpreter.run(inputTensorBuffer.getBuffer(), outputTensorBuffer.getBuffer());
            outputTensorBuffer.getBuffer().rewind();
        }
    }

    public void inference() {
        outputTensorBuffer.getBuffer().clear();
        long startT = System.currentTimeMillis();
        Trace.beginSection("runInference");
        tfLiteInterpreter.run(inputTensorBuffer.getBuffer(), outputTensorBuffer.getBuffer());
        Trace.endSection();
        Log.i(TAG, "inferT= " + (System.currentTimeMillis() - startT) + " ms");
        outputTensorBuffer.getBuffer().rewind();
    }

    @Override
    public void close() {
        tfLiteInterpreter.close();
    }
}
