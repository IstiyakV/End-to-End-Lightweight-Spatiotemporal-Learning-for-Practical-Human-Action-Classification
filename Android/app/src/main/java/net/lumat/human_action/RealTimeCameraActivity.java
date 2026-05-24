package net.lumat.human_action;

import android.app.AlertDialog;
import android.graphics.SurfaceTexture;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.TextureView;
import android.view.View;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import com.google.android.material.button.MaterialButton;

import java.io.File;
import java.util.List;
import java.util.Random;

public class RealTimeCameraActivity extends AppCompatActivity {

    private static final String TAG = "RealTimeCameraActivity";

    // View bindings
    private TextureView cameraTextureView;
    private ImageView simulatedCameraBg;
    private ImageButton btnBackHeader;
    private MaterialButton btnSwitchCamera;
    private MaterialButton btnChangeModel;
    private TextView tvActiveModelLabel;
    private TextView liveActionClass;
    private ProgressBar liveActionProgress;
    private MaterialButton btnStopCamera;

    // Camera variables
    private android.hardware.Camera mCamera;
    private boolean isFrontCamera = false;
    private int currentCameraId = -1;

    // Active model variables
    private String activeModelName = "ucf101_run_best.ptl";

    // Simulator variables
    private Handler simulatorHandler = new Handler(Looper.getMainLooper());
    private Runnable simulatorRunnable;
    private String[] mockActions = {
            "Billiards (94.2%)", "Boxing/Punching (88.7%)", "Bench Press (91.0%)", 
            "Biking/Cycling (96.5%)", "Breast Stroke (87.4%)", "Basketball Dunk (92.1%)", 
            "Jumping Jacks (95.0%)", "Golf Swing (89.1%)", "Playing Piano (93.3%)"
    };

    private TextureView.SurfaceTextureListener surfaceTextureListener = new TextureView.SurfaceTextureListener() {
        @Override
        public void onSurfaceTextureAvailable(@NonNull SurfaceTexture surface, int width, int height) {
            startCameraPreview(surface);
        }

        @Override
        public void onSurfaceTextureSizeChanged(@NonNull SurfaceTexture surface, int width, int height) {
        }

        @Override
        public boolean onSurfaceTextureDestroyed(@NonNull SurfaceTexture surface) {
            stopCamera();
            return true;
        }

        @Override
        public void onSurfaceTextureUpdated(@NonNull SurfaceTexture surface) {
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_real_time_camera);

        // Bind views
        cameraTextureView = findViewById(R.id.cameraTextureView);
        simulatedCameraBg = findViewById(R.id.simulatedCameraBg);
        btnBackHeader = findViewById(R.id.btnBackHeader);
        btnSwitchCamera = findViewById(R.id.btnSwitchCamera);
        btnChangeModel = findViewById(R.id.btnChangeModel);
        tvActiveModelLabel = findViewById(R.id.tvActiveModelLabel);
        liveActionClass = findViewById(R.id.liveActionClass);
        liveActionProgress = findViewById(R.id.liveActionProgress);
        btnStopCamera = findViewById(R.id.btnStopCamera);

        // Click listeners
        btnBackHeader.setOnClickListener(v -> finish());
        btnStopCamera.setOnClickListener(v -> finish());
        btnSwitchCamera.setOnClickListener(v -> switchCamera());
        btnChangeModel.setOnClickListener(v -> showModelSelectionDialog());

        // Attach surface listener
        cameraTextureView.setSurfaceTextureListener(surfaceTextureListener);

        // Resolve initial offline model
        File filesDir = getFilesDir();
        File[] ptlFiles = filesDir.listFiles((dir, name) -> name.endsWith(".ptl"));
        if (ptlFiles != null && ptlFiles.length > 0) {
            activeModelName = ptlFiles[0].getName();
        }
        tvActiveModelLabel.setText("Model: " + activeModelName);

        // Start mock classifier
        startSimulatorLoop();
    }

    private void startCameraPreview(SurfaceTexture surface) {
        try {
            if (mCamera != null) {
                stopCamera();
            }

            int facing = isFrontCamera ? android.hardware.Camera.CameraInfo.CAMERA_FACING_FRONT : android.hardware.Camera.CameraInfo.CAMERA_FACING_BACK;
            currentCameraId = findCameraId(facing);
            if (currentCameraId == -1) {
                currentCameraId = 0;
            }

            mCamera = android.hardware.Camera.open(currentCameraId);
            mCamera.setDisplayOrientation(90);

            android.hardware.Camera.Parameters parameters = mCamera.getParameters();
            List<String> focusModes = parameters.getSupportedFocusModes();
            if (focusModes != null && focusModes.contains(android.hardware.Camera.Parameters.FOCUS_MODE_CONTINUOUS_VIDEO)) {
                parameters.setFocusMode(android.hardware.Camera.Parameters.FOCUS_MODE_CONTINUOUS_VIDEO);
            }
            mCamera.setParameters(parameters);

            mCamera.setPreviewTexture(surface);
            mCamera.startPreview();

            if (simulatedCameraBg != null) {
                simulatedCameraBg.setVisibility(View.GONE);
            }
        } catch (Exception e) {
            Log.e(TAG, "Failed to start camera preview: " + e.getMessage());
            Toast.makeText(this, "Failed to load camera preview", Toast.LENGTH_SHORT).show();
        }
    }

    private void stopCamera() {
        try {
            if (mCamera != null) {
                mCamera.stopPreview();
                mCamera.release();
                mCamera = null;
            }
        } catch (Exception e) {
            Log.e(TAG, "Error stopping camera: " + e.getMessage());
        }
    }

    private int findCameraId(int facing) {
        int numberOfCameras = android.hardware.Camera.getNumberOfCameras();
        for (int i = 0; i < numberOfCameras; i++) {
            android.hardware.Camera.CameraInfo info = new android.hardware.Camera.CameraInfo();
            android.hardware.Camera.getCameraInfo(i, info);
            if (info.facing == facing) {
                return i;
            }
        }
        return -1;
    }

    private void switchCamera() {
        isFrontCamera = !isFrontCamera;
        if (cameraTextureView != null && cameraTextureView.isAvailable()) {
            startCameraPreview(cameraTextureView.getSurfaceTexture());
        }
    }

    private void showModelSelectionDialog() {
        File filesDir = getFilesDir();
        File[] ptlFiles = filesDir.listFiles((dir, name) -> name.endsWith(".ptl"));

        if (ptlFiles == null || ptlFiles.length == 0) {
            AlertDialog.Builder builder = new AlertDialog.Builder(this, AlertDialog.THEME_HOLO_DARK);
            builder.setTitle("Select 3D CNN Model")
                   .setMessage("No offline PyTorch models are cached on this device yet.\n\nPlease synchronize offline models from the main conversation panel first.")
                   .setPositiveButton("OK", null)
                   .show();
            return;
        }

        String[] filenames = new String[ptlFiles.length];
        for (int i = 0; i < ptlFiles.length; i++) {
            filenames[i] = ptlFiles[i].getName();
        }

        AlertDialog.Builder builder = new AlertDialog.Builder(this, AlertDialog.THEME_HOLO_DARK);
        builder.setTitle("Select Local Offline Model")
               .setItems(filenames, (dialog, which) -> {
                   File chosenFile = ptlFiles[which];
                   loadOfflineModel(chosenFile);
               })
               .setNegativeButton("Cancel", null)
               .show();
    }

    private void loadOfflineModel(File modelFile) {
        tvActiveModelLabel.setText("Model: Loading " + modelFile.getName() + "...");
        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    // Preload the PyTorch Mobile Lite model
                    org.pytorch.LiteModuleLoader.load(modelFile.getAbsolutePath());
                    runOnUiThread(() -> {
                        activeModelName = modelFile.getName();
                        tvActiveModelLabel.setText("Model: " + activeModelName);
                        Toast.makeText(RealTimeCameraActivity.this, "Successfully loaded offline model: " + activeModelName, Toast.LENGTH_SHORT).show();
                    });
                } catch (Exception e) {
                    e.printStackTrace();
                    runOnUiThread(() -> {
                        tvActiveModelLabel.setText("Model: " + activeModelName + " (Load failed)");
                        Toast.makeText(RealTimeCameraActivity.this, "Failed to load model: " + e.getMessage(), Toast.LENGTH_LONG).show();
                    });
                }
            }
        }).start();
    }

    private void startSimulatorLoop() {
        final Random random = new Random();
        simulatorRunnable = new Runnable() {
            @Override
            public void run() {
                String randomAction = mockActions[random.nextInt(mockActions.length)];
                liveActionClass.setText("Detected Action: " + randomAction);
                liveActionProgress.setProgress(60 + random.nextInt(40));
                simulatorHandler.postDelayed(this, 1500);
            }
        };
        simulatorHandler.post(simulatorRunnable);
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (cameraTextureView != null && cameraTextureView.isAvailable()) {
            startCameraPreview(cameraTextureView.getSurfaceTexture());
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        stopCamera();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (simulatorRunnable != null) {
            simulatorHandler.removeCallbacks(simulatorRunnable);
        }
    }
}
