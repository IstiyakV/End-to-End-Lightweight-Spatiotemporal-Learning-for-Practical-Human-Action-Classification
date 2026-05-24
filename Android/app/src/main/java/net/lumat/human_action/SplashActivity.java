package net.lumat.human_action;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.animation.Animation;
import android.view.animation.AnimationUtils;
import android.widget.ImageView;
import android.widget.TextView;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.concurrent.TimeUnit;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;

public class SplashActivity extends AppCompatActivity {

    private static final String TAG = "SplashActivity";
    private static final String SERVER_URL = "https://har.lumat.net/download-ptl-model?filename=class_names_v2.json";
    
    private Context ctx;
    private ImageView imgLogo;
    private TextView statusText;
    private OkHttpClient client;
    private Handler mainHandler = new Handler(Looper.getMainLooper());

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_splash);
        ctx = SplashActivity.this;
        imgLogo = findViewById(R.id.splash);
        statusText = findViewById(R.id.statusText);

        // Initialize OkHttpClient with short timeouts for Splash Screen check
        client = new OkHttpClient.Builder()
                .connectTimeout(8, TimeUnit.SECONDS)
                .readTimeout(8, TimeUnit.SECONDS)
                .build();

        try {
            // Load and start logo fade-in animation
            Animation animFadein = AnimationUtils.loadAnimation(getApplicationContext(), R.anim.fade_in);
            imgLogo.startAnimation(animFadein);
        } catch (Exception e) {
            Log.e(TAG, "Animation failed: " + e.getMessage());
        }

        // Start server connection and asset check
        startSyncAndCheckConnection();
    }

    private void startSyncAndCheckConnection() {
        updateStatus("Connecting to Lumat.net...");

        final File jsonFile = new File(getFilesDir(), "class_names_v2.json");

        Request request = new Request.Builder()
                .url(SERVER_URL)
                .get()
                .build();

        client.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(Call call, IOException e) {
                Log.e(TAG, "Connection failed: " + e.getMessage());
                handleSyncFailure(jsonFile, "Server unreachable. " + e.getLocalizedMessage());
            }

            @Override
            public void onResponse(Call call, Response response) throws IOException {
                if (response.isSuccessful()) {
                    try {
                        updateStatus("Syncing action classes metadata...");
                        
                        InputStream is = response.body().byteStream();
                        FileOutputStream fos = new FileOutputStream(jsonFile);
                        byte[] buffer = new byte[2048];
                        int bytesRead;
                        while ((bytesRead = is.read(buffer)) != -1) {
                            fos.write(buffer, 0, bytesRead);
                        }
                        fos.flush();
                        fos.close();
                        is.close();

                        updateStatus("Server connection verified! Lumat.net synced.");
                        
                        mainHandler.postDelayed(() -> navigateToMainScreen(), 1000);
                    } catch (Exception ex) {
                        Log.e(TAG, "Failed to write class_names.json: " + ex.getMessage());
                        handleSyncFailure(jsonFile, "Write error: " + ex.getMessage());
                    }
                } else {
                    Log.e(TAG, "Server error code: " + response.code());
                    handleSyncFailure(jsonFile, "Server returned status code " + response.code());
                }
            }
        });
    }

    private void handleSyncFailure(final File jsonFile, final String errorReason) {
        if (jsonFile.exists()) {
            updateStatus("Offline cache loaded! Syncing offline...");
            mainHandler.postDelayed(() -> navigateToMainScreen(), 1200);
        } else {
            updateStatus("Sync failed! Retrying in 4 seconds...\n(" + errorReason + ")");
            mainHandler.postDelayed(() -> startSyncAndCheckConnection(), 4000);
        }
    }

    private void updateStatus(final String status) {
        runOnUiThread(() -> {
            if (statusText != null) {
                statusText.setText(status);
            }
        });
    }

    private void navigateToMainScreen() {
        Intent intent = new Intent(SplashActivity.this, HumanActionActivity.class);
        startActivity(intent);
        finish();
    }
}