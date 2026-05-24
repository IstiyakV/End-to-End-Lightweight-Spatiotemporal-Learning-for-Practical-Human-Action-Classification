"""
HAR Control Center — Main Application Window.
Premium dark-theme native GUI using CustomTkinter.
Launch: python gui.py
"""

import sys
import customtkinter as ctk
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gui.frames.sidebar import Sidebar
from gui.frames.dashboard import DashboardFrame
from gui.frames.training_config import TrainingConfigFrame
from gui.frames.training_monitor import TrainingMonitorFrame
from gui.frames.dataset_manager import DatasetManagerFrame
from gui.frames.model_tester import ModelTesterFrame
from gui.frames.results_viewer import ResultsViewerFrame
from gui.frames.network_architect import NetworkArchitectFrame
from gui.frames.retrain_model import RetrainModelFrame
from gui.frames.sota_benchmark import SOTABenchmarkFrame
from gui.frames.transfer_learning import TransferLearningFrame
from gui.services.trainer_service import TrainerService, find_paused_experiments
from gui.settings import load_settings, save_setting
from gui.theme import COLORS

# Global theme
settings = load_settings()
mode = settings.get("appearance_mode", "dark")
ctk.set_appearance_mode(mode)
ctk.set_default_color_theme("blue")


class HARControlCenter(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("HAR Control Center  -  Human Action Recognition")
        self.geometry("1420x870")
        self.minsize(1200, 700)
        self.configure(fg_color=COLORS["bg"])

        # Try to set window icon
        try:
            self.iconbitmap(default="")
        except:
            pass

        # Services
        self.trainer_service = TrainerService()

        # Layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar
        self.sidebar = Sidebar(self, self._switch_frame)
        self.sidebar.grid(row=0, column=0, sticky="nsw")

        # Thin separator line
        sep = ctk.CTkFrame(self, width=1, fg_color=COLORS["border"])
        sep.grid(row=0, column=0, sticky="nse")

        # Content
        self.content = ctk.CTkFrame(self, fg_color=COLORS["bg"], corner_radius=0)
        self.content.grid(row=0, column=1, sticky="nsew")
        self.content.grid_columnconfigure(0, weight=1)
        self.content.grid_rowconfigure(0, weight=1)

        # Create all frames
        self.frames = {}
        self._create_frames()
        self._switch_frame("dashboard")

        # Check for paused experiments
        self.after(500, self._check_paused_experiments)

    def _check_paused_experiments(self):
        paused = find_paused_experiments()
        if paused:
            state = paused[-1]
            exp_name = state.get("config", {}).get("experiment_name", "Unknown")
            epoch = state.get("current_epoch", 0)
            
            dialog = ctk.CTkToplevel(self)
            dialog.title("Resume Training?")
            dialog.geometry("400x200")
            dialog.resizable(False, False)
            dialog.transient(self)
            dialog.grab_set()
            
            # Center the dialog on the physical screen
            dialog.update_idletasks()
            screen_width = dialog.winfo_screenwidth()
            screen_height = dialog.winfo_screenheight()
            x = (screen_width - 400) // 2
            y = (screen_height - 200) // 2
            dialog.geometry(f"+{x}+{y}")
            
            ctk.CTkLabel(dialog, text="Incomplete Session Detected", font=("Segoe UI", 18, "bold"), text_color=COLORS["accent"]).pack(pady=(20, 10))
            ctk.CTkLabel(dialog, text=f"An interrupted experiment '{exp_name}' was found at epoch {epoch}.\nWould you like to resume it now?", font=("Segoe UI", 12), justify="center").pack(pady=10)
            
            btn_frame = ctk.CTkFrame(dialog, fg_color="transparent")
            btn_frame.pack(pady=20)
            
            def on_yes():
                dialog.destroy()
                self.trainer_service.load_paused_state(state)
                self._switch_frame("monitor")
                
            def on_no():
                dialog.destroy()
                
            ctk.CTkButton(btn_frame, text="No, start fresh", width=120, fg_color=COLORS["border"], hover_color=COLORS["card"], text_color=COLORS["text"], command=on_no).pack(side="left", padx=10)
            ctk.CTkButton(btn_frame, text="Yes, Resume", width=120, fg_color=COLORS["success"], hover_color=COLORS["success"], text_color="#000000", command=on_yes).pack(side="left", padx=10)


    def _create_frames(self):
        frame_map = {
            "dashboard": DashboardFrame,
            "datasets": DatasetManagerFrame,
            "network": NetworkArchitectFrame,
            "transfer": TransferLearningFrame,
            "training": TrainingConfigFrame,
            "retrain": RetrainModelFrame,
            "monitor": TrainingMonitorFrame,
            "tester": ModelTesterFrame,
            "benchmark": SOTABenchmarkFrame,
            "results": ResultsViewerFrame,
        }
        for name, cls in frame_map.items():
            frame = cls(self.content, self)
            # Use place to map all frames at startup, avoiding expensive Tkinter layout recalculations (reflows)
            frame.place(x=0, y=0, relwidth=1, relheight=1)
            self.frames[name] = frame

    def _switch_frame(self, name: str):
        self.update_idletasks()  # Clear pending visual updates
        if name in self.frames:
            # Swapping Z-order stack via tkraise is processed at C-level in under 1ms!
            self.frames[name].tkraise()
            self.sidebar.set_active(name)
        self.update_idletasks()  # Allow immediate render of the new frame

    def change_theme(self, mode):
        mode = mode.lower()
        if "light" in mode:
            mode = "light"
        elif "dark" in mode:
            mode = "dark"
        ctk.set_appearance_mode(mode)
        save_setting("appearance_mode", mode)
        self.configure(fg_color=COLORS["bg"])
        
        # Propagate custom listener event to all frames
        for frame in self.frames.values():
            if hasattr(frame, "on_theme_changed"):
                frame.on_theme_changed(mode)


def main():
    app = HARControlCenter()
    app.mainloop()


if __name__ == "__main__":
    main()
