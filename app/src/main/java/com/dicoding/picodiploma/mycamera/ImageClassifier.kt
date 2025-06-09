package com.dicoding.picodiploma.mycamera

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class ImageClassifier(private val context: Context) {

    private val labels = listOf("Sehat", "Antrachnose", "Fruit Flies") // atau baca dari assets
    private val inputSize = 224
    private val interpreter: Interpreter

    init {
        val assetFileDescriptor = context.assets.openFd("GuavaDisease_CNN_Model.tflite")
        val fileInputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = fileInputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        val modelByteBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
        interpreter = Interpreter(modelByteBuffer)
    }

    fun classify(bitmap: Bitmap): Pair<String, Float> {
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        val inputBuffer = ByteBuffer.allocateDirect(4 * inputSize * inputSize * 3)
        inputBuffer.order(ByteOrder.nativeOrder())

        for (y in 0 until inputSize) {
            for (x in 0 until inputSize) {
                val px = resizedBitmap.getPixel(x, y)
                inputBuffer.putFloat(Color.red(px) / 255f)
                inputBuffer.putFloat(Color.green(px) / 255f)
                inputBuffer.putFloat(Color.blue(px) / 255f)
            }
        }

        val output = Array(1) { FloatArray(labels.size) }
        interpreter.run(inputBuffer, output)

        val maxIdx = output[0].indices.maxByOrNull { output[0][it] } ?: -1
        val label = labels[maxIdx]
        val confidence = output[0][maxIdx]

        return Pair(label, confidence)
    }
}
