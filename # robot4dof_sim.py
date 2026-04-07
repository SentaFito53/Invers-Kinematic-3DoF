#      _ _  ___   ___  ___   ___ _  __  ___ ___ __  __ 
#     | | ||   \ / _ \| __| |_ _| |/ / / __|_ _|  \/  |
#     |_  _| |) | (_) | _|   | || ' <  \__ \| || |\/| |
#       |_||___/ \___/|_|   |___|_|\_\ |___/___|_|  |_|
#                          3-DOF VERSION

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class RobotArm3DOF_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("3 DOF Robot Arm Simulator - Analytical IK")
        self.root.state('zoomed')

        # Robot parameters
        self.U1 = tk.DoubleVar(value=125)   # Link 1 (shoulder → elbow)
        self.U2 = tk.DoubleVar(value=125)   # Link 2 (elbow → end-effector)
        self.base_height = tk.DoubleVar(value=0)

        # Target position
        self.target_x = tk.DoubleVar(value=150)
        self.target_y = tk.DoubleVar(value=100)
        self.target_z = tk.DoubleVar(value=80)

        # Forward kinematics joint angles
        self.theta1_fk = tk.DoubleVar(value=0)
        self.theta2_fk = tk.DoubleVar(value=0)
        self.theta3_fk = tk.DoubleVar(value=0)

        # Current joint angles (radians)
        self.angles = [0.0, 0.0, 0.0]

        self.setup_gui()

    # ─────────────────────────────────────────────────────────────────────────
    def setup_gui(self):
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)   # plot column expands

        # ── Left column: controls ──────────────────────────────────────────
        control_frame = ttk.LabelFrame(main_frame, text="Robot Control", padding="10")
        control_frame.grid(row=0, column=0, rowspan=2,
                           sticky="nsew", padx=5, pady=5)
        control_frame.grid_columnconfigure(1, weight=1)

        row = 0

        # Robot Parameters
        ttk.Label(control_frame, text="Robot Parameters",
                  font=('Arial', 10, 'bold')).grid(
            row=row, column=0, columnspan=2, pady=(0, 8), sticky="w")
        row += 1

        for label, var in [("L1 (cm):", self.U1),
                            ("L2 (cm):", self.U2),
                            ("Base (cm):", self.base_height)]:
            ttk.Label(control_frame, text=label,
                      font=('Arial', 9)).grid(row=row, column=0,
                                              sticky="w", pady=2)
            e = ttk.Entry(control_frame, textvariable=var,
                          width=8, font=('Arial', 9))
            e.grid(row=row, column=1, padx=(5, 0), pady=2, sticky="w")
            e.bind('<Return>', self.on_entry_return)
            row += 1

        ttk.Button(control_frame, text="Update Parameters",
                   command=self.calculate_ik).grid(
            row=row, column=0, columnspan=2, pady=(4, 0), sticky="ew")
        row += 1

        ttk.Separator(control_frame, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=10)
        row += 1

        # Target Position (IK input)
        ttk.Label(control_frame, text="Target Position (IK)",
                  font=('Arial', 10, 'bold')).grid(
            row=row, column=0, columnspan=2, pady=(0, 8), sticky="w")
        row += 1

        self._ik_entries = []
        for label, var in [("X (cm):", self.target_x),
                            ("Y (cm):", self.target_y),
                            ("Z (cm):", self.target_z)]:
            ttk.Label(control_frame, text=label,
                      font=('Arial', 9)).grid(row=row, column=0,
                                              sticky="w", pady=2)
            e = ttk.Entry(control_frame, textvariable=var,
                          width=8, font=('Arial', 9))
            e.grid(row=row, column=1, padx=(5, 0), pady=2, sticky="w")
            e.bind('<Return>', self.on_entry_return)
            self._ik_entries.append(e)
            row += 1

        ttk.Button(control_frame, text="Calculate Inverse Kinematics",
                   command=self.calculate_ik,
                   style='Accent.TButton').grid(
            row=row, column=0, columnspan=2, pady=8, sticky="ew")
        row += 1

        ttk.Separator(control_frame, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=8)
        row += 1

        # Forward Kinematics
        ttk.Label(control_frame, text="Forward Kinematics",
                  font=('Arial', 10, 'bold')).grid(
            row=row, column=0, columnspan=2, pady=(0, 8), sticky="w")
        row += 1

        self._fk_entries = []
        for label, var in [("θ1 (°):", self.theta1_fk),
                            ("θ2 (°):", self.theta2_fk),
                            ("θ3 (°):", self.theta3_fk)]:
            ttk.Label(control_frame, text=label,
                      font=('Arial', 9)).grid(row=row, column=0,
                                              sticky="w", pady=2)
            e = ttk.Entry(control_frame, textvariable=var,
                          width=8, font=('Arial', 9))
            e.grid(row=row, column=1, padx=(5, 0), pady=2, sticky="w")
            e.bind('<Return>', self.on_entry_return)
            self._fk_entries.append(e)
            row += 1

        ttk.Button(control_frame, text="Calculate Forward Kinematics",
                   command=self.calculate_fk,
                   style='Accent.TButton').grid(
            row=row, column=0, columnspan=2, pady=8, sticky="ew")
        row += 1

        # ── Results text box (below controls) ────────────────────────────
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        results_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        results_frame.grid_columnconfigure(0, weight=1)
        results_frame.grid_rowconfigure(0, weight=1)

        self.results_text = tk.Text(results_frame, height=20, width=42,
                                    font=('Consolas', 8))
        sb = ttk.Scrollbar(results_frame, orient="vertical",
                           command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=sb.set)
        self.results_text.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")

        # ── Right column: 3-D plot ─────────────────────────────────────────
        plot_frame = ttk.Frame(main_frame)
        plot_frame.grid(row=0, column=1, rowspan=3,
                        sticky="nsew", padx=5, pady=5)
        plot_frame.grid_rowconfigure(0, weight=1)
        plot_frame.grid_columnconfigure(0, weight=1)

        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Mouse interaction
        self._drag_start = None
        self.canvas.mpl_connect('scroll_event',       self.on_scroll)
        self.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.canvas.mpl_connect('button_release_event', self.on_button_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)

        # Style
        style = ttk.Style()
        style.configure('Accent.TButton', font=('Arial', 9, 'bold'))

        # Initial render
        self.calculate_ik()

    # ─── Kinematics ──────────────────────────────────────────────────────────
    def inverse_kinematics(self, x, y, z):
        """
        Analytical IK for 3-DOF planar arm (base yaw + 2-link planar).

        θ1 : base yaw  (atan2 in XY plane)
        θ2 : shoulder pitch
        θ3 : elbow pitch
        """
        U1 = self.U1.get()
        U2 = self.U2.get()
        base_h = self.base_height.get()

        # Base rotation
        theta1 = math.atan2(y, x)

        # Reach in the arm plane
        r = math.sqrt(x**2 + y**2)
        h = z - base_h

        d = math.sqrt(r**2 + h**2)

        # Reachability check
        if d > (U1 + U2) or d < abs(U1 - U2):
            return None

        # Elbow angle (law of cosines)
        cos3 = (d**2 - U1**2 - U2**2) / (2 * U1 * U2)
        cos3 = max(-1.0, min(1.0, cos3))
        theta3 = math.acos(cos3)          # elbow-up solution

        # Shoulder angle
        alpha = math.atan2(h, r)
        beta  = math.atan2(U2 * math.sin(theta3),
                            U1 + U2 * math.cos(theta3))
        theta2 = alpha - beta

        return [theta1, theta2, theta3]

    def forward_kinematics(self, angles):
        """Return end-effector (x, y, z) from joint angles (radians)."""
        theta1, theta2, theta3 = angles
        U1 = self.U1.get()
        U2 = self.U2.get()
        base_h = self.base_height.get()

        reach = (U1 * math.cos(theta2) +
                 U2 * math.cos(theta2 + theta3))
        x = reach * math.cos(theta1)
        y = reach * math.sin(theta1)
        z = base_h + U1 * math.sin(theta2) + U2 * math.sin(theta2 + theta3)

        return [x, y, z]

    def joint_positions(self, angles):
        """Return numpy array of all joint XYZ positions."""
        theta1, theta2, theta3 = angles
        U1 = self.U1.get()
        U2 = self.U2.get()
        base_h = self.base_height.get()

        pts = []

        # Origin / floor
        pts.append([0, 0, 0])
        # Base top
        pts.append([0, 0, base_h])

        # Shoulder → elbow
        x1 = U1 * math.cos(theta1) * math.cos(theta2)
        y1 = U1 * math.sin(theta1) * math.cos(theta2)
        z1 = base_h + U1 * math.sin(theta2)
        pts.append([x1, y1, z1])

        # Elbow → end-effector
        x2 = x1 + U2 * math.cos(theta1) * math.cos(theta2 + theta3)
        y2 = y1 + U2 * math.sin(theta1) * math.cos(theta2 + theta3)
        z2 = z1 + U2 * math.sin(theta2 + theta3)
        pts.append([x2, y2, z2])

        return np.array(pts)

    # ─── Actions ─────────────────────────────────────────────────────────────
    def calculate_ik(self):
        x = self.target_x.get()
        y = self.target_y.get()
        z = self.target_z.get()

        angles = self.inverse_kinematics(x, y, z)
        if angles is None:
            self._show("ERROR: Target position unreachable!\n"
                       "Adjust coordinates or link lengths.")
            return

        self.angles = angles
        fk = self.forward_kinematics(angles)
        err = math.sqrt(sum((a - b)**2 for a, b in zip(fk, [x, y, z])))

        t1, t2, t3 = (math.degrees(a) for a in angles)
        text = (
            "=== INVERSE KINEMATICS RESULTS ===\n\n"
            f"Target Position:\n"
            f"  X = {x:.2f} cm\n"
            f"  Y = {y:.2f} cm\n"
            f"  Z = {z:.2f} cm\n\n"
            "Joint Angles:\n"
            f"  θ1 (Base Yaw) = {t1:+.3f}°\n"
            f"  θ2 (Shoulder) = {t2:+.3f}°\n"
            f"  θ3 (Elbow)    = {t3:+.3f}°\n\n"
            "FK Verification:\n"
            f"  X = {fk[0]:.4f} cm\n"
            f"  Y = {fk[1]:.4f} cm\n"
            f"  Z = {fk[2]:.4f} cm\n\n"
            f"Position Error: {err:.6f} cm\n\n"
            "Robot Parameters:\n"
            f"  L1 = {self.U1.get():.1f} cm\n"
            f"  L2 = {self.U2.get():.1f} cm\n"
            f"  Base = {self.base_height.get():.1f} cm"
        )
        self._show(text)
        self.plot_robot()

    def calculate_fk(self):
        try:
            angles_deg = [self.theta1_fk.get(),
                          self.theta2_fk.get(),
                          self.theta3_fk.get()]
            angles = [math.radians(a) for a in angles_deg]
        except Exception as e:
            self._show(f"ERROR: {e}")
            return

        self.angles = angles
        fk = self.forward_kinematics(angles)
        t1, t2, t3 = angles_deg

        text = (
            "=== FORWARD KINEMATICS RESULTS ===\n\n"
            "Input Joint Angles:\n"
            f"  θ1 (Base Yaw) = {t1:+.3f}°\n"
            f"  θ2 (Shoulder) = {t2:+.3f}°\n"
            f"  θ3 (Elbow)    = {t3:+.3f}°\n\n"
            "End-Effector Position:\n"
            f"  X = {fk[0]:.4f} cm\n"
            f"  Y = {fk[1]:.4f} cm\n"
            f"  Z = {fk[2]:.4f} cm\n\n"
            "Robot Parameters:\n"
            f"  L1 = {self.U1.get():.1f} cm\n"
            f"  L2 = {self.U2.get():.1f} cm\n"
            f"  Base = {self.base_height.get():.1f} cm"
        )
        self._show(text)
        self.plot_robot()

    def _show(self, text):
        self.results_text.delete(1.0, "end")
        self.results_text.insert("end", text)

    def on_entry_return(self, _event=None):
        self.calculate_ik()

    # ─── Plot ────────────────────────────────────────────────────────────────
    def plot_robot(self):
        self.ax.clear()
        pts = self.joint_positions(self.angles)

        # Arm links
        self.ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                     'o-', lw=4, ms=8, color='royalblue', label='Robot Arm')

        # Target marker (IK mode)
        tx, ty, tz = self.target_x.get(), self.target_y.get(), self.target_z.get()
        self.ax.scatter(tx, ty, tz, color='red', s=120,
                        zorder=5, label='Target')

        # Workspace boundary
        U1, U2 = self.U1.get(), self.U2.get()
        base_h = self.base_height.get()
        radius = U1 + U2
        theta = np.linspace(0, 2 * math.pi, 60)
        cx, cy = radius * np.cos(theta), radius * np.sin(theta)
        self.ax.plot(cx, cy, np.zeros_like(cx) + base_h,
                     'g--', alpha=0.25, label='Workspace')
        self.ax.plot(cx, cy, np.full_like(cx, base_h + radius),
                     'g--', alpha=0.25)

        # Labels
        self.ax.set_xlabel('X (cm)', fontsize=10)
        self.ax.set_ylabel('Y (cm)', fontsize=10)
        self.ax.set_zlabel('Z (cm)', fontsize=10)
        self.ax.set_title('3-DOF Robot Arm Simulation', fontsize=12, fontweight='bold')

        lim = radius * 1.1
        self.ax.set_xlim([-lim, lim])
        self.ax.set_ylim([-lim, lim])
        self.ax.set_zlim([0, lim])

        self.ax.legend(fontsize=9)
        self.canvas.draw()

    # ─── Mouse interaction ────────────────────────────────────────────────────
    def on_scroll(self, event):
        factor = 1.1 if event.button == 'up' else 0.9
        for getter, setter in [(self.ax.get_xlim, self.ax.set_xlim),
                               (self.ax.get_ylim, self.ax.set_ylim),
                               (self.ax.get_zlim, self.ax.set_zlim)]:
            lo, hi = getter()
            mid = (lo + hi) / 2
            setter([mid + (lo - mid) * factor,
                    mid + (hi - mid) * factor])
        self.canvas.draw()

    def on_button_press(self, event):
        if event.button == 1:
            self._drag_start = (event.x, event.y)

    def on_button_release(self, _event):
        self._drag_start = None

    def on_motion(self, event):
        if self._drag_start is None or event.button != 1:
            return
        dx = event.x - self._drag_start[0]
        dy = event.y - self._drag_start[1]
        self.ax.view_init(elev=self.ax.elev - dy * 0.5,
                          azim=self.ax.azim - dx * 0.5)
        self.canvas.draw()
        self._drag_start = (event.x, event.y)


# ─── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app = RobotArm3DOF_GUI(root)
    root.mainloop()