package com.handleychen.litertstarter;

import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;
import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

public class BenchmarkActivity extends AppCompatActivity {

    private static final String TAG = "BenchmarkActivity";
    private HandlerThread backgroundThread;
    private Benchmark benchmark;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });

        benchmark = new Benchmark(this);
        backgroundThread = new HandlerThread("inferenceThread");
        backgroundThread.start();
        Handler backgroundHandler = new Handler(backgroundThread.getLooper());
        backgroundHandler.post(() -> {
            benchmark.createInterpreter();
            benchmark.createInputOutput();
            benchmark.warmup();
            for (int i = 0; i < 10; i++) {
                benchmark.inference();
            }
        });
    }


    @Override
    protected void onDestroy() {
        super.onDestroy();
        Log.v(TAG, "onDestroy");
        benchmark.close();
        backgroundThread.quitSafely();
    }
}
