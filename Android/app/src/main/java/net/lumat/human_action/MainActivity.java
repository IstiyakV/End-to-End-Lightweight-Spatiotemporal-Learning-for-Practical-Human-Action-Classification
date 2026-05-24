package net.lumat.human_action;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.ImageView;

import com.bumptech.glide.Glide;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        SetupGifBG();
        SetupHumanAction();

    }

    private void SetupHumanAction() {
        ImageView human_action_btn = findViewById(R.id.human_action_btn);
        human_action_btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(getApplicationContext(), HumanActionActivity.class);
                startActivity(intent);
            }
        });
    }

    private void SetupGifBG() {
        ImageView backgroundImageView = findViewById(R.id.backgroundImageView);
        Glide.with(this)
                .asGif()
                .load(R.raw.bg) // Replace with the name of your GIF resource
                .into(backgroundImageView);
    }
}