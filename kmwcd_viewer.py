"""
/*--------------------------------------------------------------------
 *    The MB-system:  kmwcd_viewer.py  13/8/2025
 *
 *    Copyright (c) 2025 by
 *    Christian dos Santos Ferreira
 *      MARUM
 *      University of Bremen
 *      Bremen Germany
 *    David W. Caress (caress@mbari.org)
 *      Monterey Bay Aquarium Research Institute
 *      Moss Landing, California, USA
 *    Dale N. Chayes 
 *      Center for Coastal and Ocean Mapping
 *      University of New Hampshire
 *      Durham, New Hampshire, USA
 *     
 *    MB-System was created by Caress and Chayes in 1992 at the
 *      Lamont-Doherty Earth Observatory
 *      Columbia University
 *      Palisades, NY 10964
 *
 *    See README.md file for copying and redistribution conditions.
 *--------------------------------------------------------------------*/
/*
"""
import os
# Import modules / symbols required by the application.
import sys
import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timezone
#

# suppress matplotlib warning
# Suppress specific Matplotlib warning to keep the UI output clean.
warnings.filterwarnings(
    "ignore",
    message="The input coordinates to pcolormesh are interpreted as cell centers"
)
#

# ensure KMALL is importable from local kmall.py
# Determine repository root so we can import local modules (kmall.py) as KMALL.
repo_root = os.path.abspath(os.path.dirname(__file__))
if repo_root not in sys.path:
# Prepend repository root to Python path, ensuring KMALL can be imported.
    sys.path.insert(0, repo_root)
# Import modules / symbols required by the application.
import KMALL
#

# Import modules / symbols required by the application.
from PySide6.QtWidgets import (
    QCheckBox,
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QDoubleSpinBox, QSizePolicy, QSlider
)
# Import modules / symbols required by the application.
from PySide6.QtCore import Qt, QTimer
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
#

# Custom slider: override wheel to step exactly one ping per notch.
class OneStepWheelSlider(QSlider):
# Execute statement (class:OneStepWheelSlider)
    # moves exactly one ping per wheel notch
    # Handle mouse wheel events and move the slider by one step per notch.
    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta > 0 and self.value() < self.maximum():
            self.setValue(self.value() + 1)
        elif delta < 0 and self.value() > self.minimum():
            self.setValue(self.value() - 1)
        event.accept()
#

# Compute raw amplitude matrix and geometry (across-track Y, depth Z) for a ping.
def compute_amplitude_geometry(beamdata, dg):
# Convert per-beam amplitude lists into a 2D DataFrame (beams × samples).
    amp = pd.DataFrame.from_dict(dg['beamData']['sampleAmplitude05dB_p'])
# Read sound speed (m/s) from rxInfo to convert time samples to range.
    sound_speed = dg['rxInfo']['soundVelocity_mPerSec']
# Read ADC sample frequency (Hz) from rxInfo.
    sf = dg['rxInfo']['sampleFreq_Hz']
# Take beam steering angles (deg) relative to vertical for each beam.
    angles = np.array(beamdata.get('beamPointAngReVertical_deg', []), dtype=float)
# Execute statement (def:compute_amplitude_geometry)
    angles = np.where(np.isfinite(angles), angles, 0.0) if angles.size else np.zeros(amp.shape[0])
# Convert sample index to slant range in meters: range = i * 0.5 * c / fs.
    ranges = np.arange(amp.shape[1]) * 0.5 * sound_speed / sf
# Compute depth grid Z (positive downward): Z = -cos(angle) * range.
    za = -(np.cos(np.deg2rad(angles))[:, None] * ranges[None, :])
# Compute across-track grid Y: Y = sin(angle) * range.
    ya = (np.sin(np.deg2rad(angles))[:, None] * ranges[None, :])
# Execute statement (def:compute_amplitude_geometry)
    return amp, ya, za
#

# Clean geometry arrays column-wise (ffill/bfill) to avoid NaNs when plotting.
def fill_nonfinite_cols(arr):
    arr = np.array(arr, dtype=float, copy=True)
    for j in range(arr.shape[1]):
        col = arr[:, j]
        if not np.any(np.isfinite(col)):
            arr[:, j] = 0.0
        else:
            s = pd.Series(col).ffill().bfill().to_numpy()
            mask = ~np.isfinite(s)
            if mask.any():
                s[mask] = np.nanmean(s)
            arr[:, j] = s
    return arr
#

# Main application window: UI layout, data loading, plotting, and export.
class MainWindow(QMainWindow):
# Initialize widgets, signals/slots, figure/canvas, and default state.
    def __init__(self):
        super().__init__()
        self.setWindowTitle("KMWCD Viewer – Amplitude + Geo-Metadata (SPO)")
        self.loader = None
        self.pings = None
        self.nav_df = None
        self.mrz = None
        self.mrz_loader = None
        self.current_ping = 0
        # Replay state
        self.replay_timer = QTimer(self)
        self.replay_timer.timeout.connect(self._replay_tick)
        self._is_replaying = False
        # enable keyboard focus
        # Accept keyboard focus so keyPressEvent receives arrow keys.
        self.setFocusPolicy(Qt.StrongFocus)
#
        w = QWidget()
        self.setCentralWidget(w)
        layout = QVBoxLayout(w)
#

        # header
        hh = QHBoxLayout()
        # Create the 'Open' button to select a .kmwcd/.kmall file.
        btn_open = QPushButton("Open .kmwcd/.kmall")
        btn_open.clicked.connect(self.open_file)
        hh.addWidget(btn_open)
        # Label showing the currently opened file name.
        self.lbl_file = QLabel("No file loaded")
        self.lbl_file.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        hh.addWidget(self.lbl_file)
        # Create the 'Save Image' button with PNG/JPEG export options.
        btn_save = QPushButton("Save Image")
        btn_save.clicked.connect(self.save_image)
        hh.addWidget(btn_save)
        # Toggle MRZ bottom overlay
        self.chk_bottom = QCheckBox("Bottom (MRZ)")
        self.chk_bottom.setChecked(False)
        self.chk_bottom.stateChanged.connect(lambda _ : self.redraw())
        hh.addWidget(self.chk_bottom)
        # Label with dynamic metadata (UTC time, SOG, lat/lon).
        self.meta_label = QLabel("")
        self.meta_label.setWordWrap(True)
        hh.addWidget(self.meta_label)
        layout.addLayout(hh)
#

        # depth controls
        dh = QHBoxLayout()
        dh.addWidget(QLabel("Depth Min (m):"))
        # Depth minimum spinbox (m).
        self.dmin = QDoubleSpinBox(); self.dmin.setRange(0, 1e5); self.dmin.valueChanged.connect(self.redraw)
        dh.addWidget(self.dmin)
        dh.addWidget(QLabel("Depth Max (m):"))
        # Depth maximum spinbox (m).
        self.dmax = QDoubleSpinBox(); self.dmax.setRange(0, 1e5); self.dmax.valueChanged.connect(self.redraw)
        dh.addWidget(self.dmax)
        layout.addLayout(dh)
#

        # amplitude controls
        ah = QHBoxLayout()
        ah.addWidget(QLabel("Amp Min:"))
        # Amplitude minimum spinbox; controls colorbar lower bound.
        self.amin = QDoubleSpinBox(); self.amin.setRange(0, 1e6); self.amin.valueChanged.connect(self.redraw)
        ah.addWidget(self.amin)
        ah.addWidget(QLabel("Amp Max:"))
        # Amplitude maximum spinbox; controls colorbar upper bound.
        self.amax = QDoubleSpinBox(); self.amax.setRange(0, 1e6); self.amax.valueChanged.connect(self.redraw)
        ah.addWidget(self.amax)
        layout.addLayout(ah)
#

        # across-track X scale control
        xh = QHBoxLayout()
        xh.addWidget(QLabel("Across-track Max (m):"))
        self.xmax = QDoubleSpinBox()
        self.xmax.setRange(0, 1e6)
        self.xmax.valueChanged.connect(self.redraw)
        xh.addWidget(self.xmax)
        layout.addLayout(xh)
        # Replay controls
        xh.addWidget(QLabel("Replay (pings/s):"))
        self.replay_rate = QDoubleSpinBox()
        self.replay_rate.setRange(0.1, 10.0)
        self.replay_rate.setSingleStep(0.1)
        self.replay_rate.setDecimals(1)
        self.replay_rate.setValue(2.0)
        xh.addWidget(self.replay_rate)
        self.btn_replay = QPushButton("Replay")
        self.btn_replay.setCheckable(True)
        self.btn_replay.toggled.connect(self._toggle_replay)
        xh.addWidget(self.btn_replay)
        self.replay_rate.valueChanged.connect(self._update_replay_interval)
        
#

        # canvas + colorbar
        # Create Matplotlib figure/axes pair used for the water column image.
        self.fig, self.ax = plt.subplots()
        self.fig.subplots_adjust(right=0.85)
        # Fixed colorbar axes to prevent plot area from shifting on redraw.
        self.cax = self.fig.add_axes([0.87, 0.15, 0.03, 0.7])
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas, stretch=1)
#

        # ping slider
        ph = QHBoxLayout()
        ph.addWidget(QLabel("Ping:"))
        # Ping slider with custom 1-step wheel behavior.
        self.slider = OneStepWheelSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.on_slide)
        self.slider.valueChanged.connect(self._set_ping_from_slider)
        ph.addWidget(self.slider)
        self.lbl_ping = QLabel("0")
        ph.addWidget(self.lbl_ping)
        layout.addLayout(ph)
#

# Initialize status bar to Ready.
        self.statusBar().showMessage("Ready")
#

    # File open handler: index datagrams, collect SPO, compute global ranges with progress.
    def open_file(self):
        # Pop up native file chooser for .kmwcd/.kmall files.
        fname, _ = QFileDialog.getOpenFileName(self, "Open .kmwcd/.kmall", "", "*.kmwcd *.kmall")
        if not fname:
            return
        self.lbl_file.setText(os.path.basename(fname))
        self.statusBar().showMessage("Indexing…"); QApplication.processEvents()
        # Instantiate KMALL reader for the selected file.
        self.loader = KMALL.kmall(fname); self.loader.index_file()
        idx = self.loader.Index.copy()
        # MWC filter
        # Filter index for #MWC ping datagrams.
        mwc = idx[idx.get('MessageType','')=='#MWC']
        if mwc.empty:
        # Fallback: handle MessageType objects stored as bytes.
            tmp = idx.copy(); tmp['msg_bytes']=tmp['MessageType'].apply(lambda x: x if isinstance(x,bytes) else (x.encode() if isinstance(x,str) else x)); mwc=tmp[tmp['msg_bytes']==b'#MWC']
        if mwc.empty:
            # Fallback: case-insensitive substring match on MessageType.
            mwc = idx[idx['MessageType'].astype(str).str.contains('MWC',case=False)]
        self.pings = mwc.reset_index(drop=True)
        if self.pings.empty:
            # Abort early if the file has no water-column data.
            self.statusBar().showMessage("No #MWC data found."); return
        # Initialize ping slider range and reset position
        try:
            if hasattr(self, 'slider'):
                self.slider.setMinimum(0)
                self.slider.setMaximum(max(0, len(self.pings) - 1))
                self.slider.setSingleStep(1)
                self.slider.setValue(0)
            if hasattr(self, 'lbl_ping'):
                self.lbl_ping.setText('0')
            self.current_ping = 0
        except Exception:
            pass
        
        # Initialize ping slider range and reset position
        try:
            if hasattr(self, 'slider'):
                self.slider.setMaximum(max(0, len(self.pings) - 1))
                self.slider.setMinimum(0)
                self.slider.setSingleStep(1)
                self.slider.setValue(0)
            if hasattr(self, 'lbl_ping'):
                self.lbl_ping.setText('0')
            self.current_ping = 0
        except Exception:
            pass
        # Build MRZ index from current file; if not present and ext is .kmwcd, try companion .kmall
        try:
            _idx = getattr(self, 'Index', None) or idx  # tolerate both self.Index and local idx
            _mrz = _idx[_idx['MessageType'] == b'#MRZ'] if 'MessageType' in _idx else None
            if _mrz is None or _mrz.empty:
                _mrz = _idx[_idx['MessageType'].astype(str).str.contains('MRZ', case=False, na=False)]
            if _mrz is None or _mrz.empty:
                base, ext = os.path.splitext(fname)
                if ext.lower() == '.kmwcd':
                    cand = base + '.kmall'
                    if os.path.exists(cand):
                        self.mrz_loader = KMALL.kmall(cand); self.mrz_loader.index_file()
                        _mrz = self.mrz_loader.Index[self.mrz_loader.Index['MessageType'].astype(str).str.contains('MRZ', case=False, na=False)]
                        if _mrz is not None and not _mrz.empty:
                            self.mrz = _mrz.reset_index().reset_index(drop=True)
            else:
                self.mrz = _mrz.reset_index().reset_index(drop=True); self.mrz_loader = self.loader
        except Exception:
            self.mrz = None
# SPO filter
        # Collect #SPO navigation datagrams (position and speed).
        spo = idx[idx.get('MessageType','')=='#SPO']
        if spo.empty:
            tmp2 = idx.copy(); tmp2['msg_bytes']=tmp2['MessageType'].apply(lambda x: x if isinstance(x,bytes) else (x.encode() if isinstance(x,str) else x)); spo=tmp2[tmp2['msg_bytes']==b'#SPO']
        if spo.empty:
            # Fallback: case-insensitive substring match on MessageType.
            spo = idx[idx['MessageType'].astype(str).str.contains('SPO',case=False)]
        spo = spo.reset_index(drop=True)
        recs = []
        for _, row in spo.iterrows():
            # Extract the navigation payload from the SPO datagram.
            dg = self.loader.read_index_row(row); sd = dg.get('sensorData', {}); dt = sd.get('datetime') or dg.get('header', {}).get('dgdatetime')
            if not isinstance(dt, datetime): continue
            t = dt.astimezone(timezone.utc); lat = sd.get('correctedLat_deg', np.nan); lon = sd.get('correctedLong_deg', np.nan); sog = sd.get('speedOverGround_mPerSec', np.nan)
            recs.append((t, lat, lon, sog))
            # Build a time-indexed DataFrame with (lat, lon, SOG) for nearest-time lookup.
        self.nav_df = pd.DataFrame(recs, columns=['time','lat','lon','sog']).set_index('time') if recs else None
        # Initialize slider bounds and reset to the first ping.
        total = len(self.pings); self.slider.setMaximum(total - 1); self.current_ping = 0
        # Replay state
        self.replay_timer = QTimer(self)
        self.replay_timer.timeout.connect(self._replay_tick)
        self._is_replaying = False; self.slider.setValue(0)
#

        # compute global ranges with percentage in status bar
        total = len(self.pings)
        amp_min, amp_max = float('inf'), float('-inf')
        depth_max = 0.0
        # Track the maximum absolute across-track distance observed.
        across_max = 0.0
        last_pct = -1
        # Pre-scan each ping once to compute global ranges and update progress %.
        for i in range(total):
            pct = int((i + 1) / total * 100) if total > 0 else 100
            if pct != last_pct:
                self.statusBar().showMessage(f"Loading pings… {pct}%")
                QApplication.processEvents()
                last_pct = pct
            # Random-access read of the i-th #MWC datagram.
            dg = self.loader.read_index_row(self.pings.iloc[i])
            # Compute amplitude and coordinate grids for this ping.
            amp, ya, za = compute_amplitude_geometry(pd.DataFrame.from_dict(dg['beamData']), dg)
            arr = amp.to_numpy(dtype=float)
            # Consider only finite amplitudes when tracking global min/max.
            fin = arr[np.isfinite(arr)]
            if fin.size:
                amp_min = min(amp_min, fin.min())
                amp_max = max(amp_max, fin.max())
            # Depths (positive down) from the geometry array.
            depths = -za
            if depths.size:
                depth_max = max(depth_max, depths.max())
        self.statusBar().showMessage("Load complete")
        QApplication.processEvents()
        if amp_min < amp_max:
            self.amin.setRange(amp_min, amp_max)
            self.amax.setRange(amp_min, amp_max)
            self.amin.setValue(amp_min)
            self.amax.setValue(amp_max)
        self.dmin.setRange(0, depth_max)
        self.dmax.setRange(0, depth_max)
        self.dmin.setValue(0)
        self.dmax.setValue(depth_max)
#

        # set X max spinbox based on global across-track extent
        if across_max > 0:
            self.xmax.setRange(0, across_max * 2)  # allow user to widen further
            self.xmax.setValue(across_max)
        else:
            self.xmax.setRange(0, 1e5)
            self.xmax.setValue(0)
#

        self.redraw()

    # Slider changed: record new ping index and trigger redraw.
    def on_slide(self, val):
        self.current_ping = val; self.lbl_ping.setText(str(val)); self.redraw()
#

    # Keyboard handler: Left/Right arrows move by ±1 ping.
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Left and self.current_ping > 0:
            self.slider.setValue(self.current_ping - 1)
        elif event.key() == Qt.Key_Right and self.current_ping < self.slider.maximum():
            self.slider.setValue(self.current_ping + 1)
        else:
            super().keyPressEvent(event)
#

    # Export current figure as PNG/JPEG at double DPI, with metadata caption.
    def save_image(self):
        # overlay metadata text
        filename = self.lbl_file.text() or ''
        meta = self.meta_label.text()
        # Temporarily overlay filename + meta at the bottom-left of the figure.
        overlay = self.fig.text(0.01, 0.01, f"{filename} | {meta}", va='bottom', ha='left', fontsize=8,
                                 color='white', bbox=dict(facecolor='black', alpha=0.5, pad=2))
        # file dialog with both formats
        # Ask user for destination path and format (PNG or JPEG).
        path, filt = QFileDialog.getSaveFileName(
            self, "Save Image", filename, "PNG (*.png);;JPEG (*.jpg)"
        )
        if path:
            fmt = 'png' if path.lower().endswith('.png') else 'jpg'
            # double the resolution
            # Double output resolution by doubling Matplotlib DPI.
            dpi = self.fig.get_dpi() * 2
            self.fig.savefig(path, format=fmt, dpi=dpi, bbox_inches='tight')
        overlay.remove()
#

    # Render the currently selected ping using the latest controls.
    def redraw(self):
        # keep slider synced with current ping
        self._sync_slider()
        dg = self.loader.read_index_row(self.pings.iloc[self.current_ping])
        # Retrieve the ping's UTC timestamp from the header.
        hdr = dg.get('header', {}); dt = hdr.get('dgdatetime'); ping_time = dt.astimezone(timezone.utc) if isinstance(dt, datetime) else None
        lat = lon = sog = np.nan
        if self.nav_df is not None and ping_time:
            # Find the closest SPO record by time for nav metadata.
            idxer = self.nav_df.index.get_indexer([ping_time], method='nearest'); pos = idxer[0] if idxer.size else -1
            if pos >= 0:
                navrec = self.nav_df.iloc[pos]; lat, lon, sog = navrec['lat'], navrec['lon'], navrec['sog']
        sog_str = f"{sog:.2f}" if np.isfinite(sog) else 'N/A'
        lat_str = f"{lat:.6f}" if np.isfinite(lat) else 'N/A'
        lon_str = f"{lon:.6f}" if np.isfinite(lon) else 'N/A'
        time_str = ping_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " UTC" if ping_time else 'N/A'
        # Display the ping metadata (UTC time, SOG, latitude, longitude).
        self.meta_label.setText(f"Ping {self.current_ping}: DateTime={time_str}, SOG={sog_str} m/s, Pos={lat_str},{lon_str}")
        # Compute amplitude and coordinate grids for this ping.
        amp, ya, za = compute_amplitude_geometry(pd.DataFrame.from_dict(dg['beamData']), dg)
        # Execute statement (def:redraw)
        ya = -ya # <-- flip left/right on the plot
        # Sanitize Y/Z grids to avoid non-finite artifacts in pcolormesh.
        ya_arr = fill_nonfinite_cols(ya); za_arr = fill_nonfinite_cols(za); depth_arr = -za_arr
        # Build a boolean mask for the configured depth window.
        mask = (depth_arr >= self.dmin.value()) & (depth_arr <= self.dmax.value())
        # Convert amplitude DataFrame to a NumPy array (samples × beams).
        data = amp.to_numpy().T; data = np.where(mask.T, data, np.nan)
        # Clear the axes before drawing the new frame.
        self.ax.clear(); mesh = self.ax.pcolormesh(ya_arr.T, depth_arr.T, data, shading='nearest', cmap='viridis', vmin=self.amin.value(), vmax=self.amax.value())

        # Overlay MRZ bottom detections as dots, if available
        try:
            if getattr(self, 'chk_bottom', None) and self.chk_bottom.isChecked() and getattr(self, 'mrz', None) is not None and not self.mrz.empty:
                hdr = dg.get('header', {})
                ping_sec = hdr.get('dgtime')
                if ping_sec is not None:
                    import numpy as _np
                    # nearest MRZ row by absolute time
                    times = _np.array(self.mrz['Time'], dtype=float)
                    pos = int(_np.argmin(_np.abs(times - float(ping_sec))))
                    mrz_row = self.mrz.iloc[pos]
                    loader = getattr(self, 'mrz_loader', None) or self.loader
                    mrz_dg = loader.read_index_row(mrz_row)
                    s = mrz_dg.get('sounding', {})
                    y = _np.asarray(s.get('y_reRefPoint_m', []), dtype=float)
                    z = _np.asarray(s.get('z_reRefPoint_m', []), dtype=float)  # positive down
                    dt = _np.asarray(s.get('detectionType', []), dtype=int)
                    dc = _np.asarray(s.get('detectionClass', []), dtype=int)
                    m = _np.isfinite(y) & _np.isfinite(z)
                    if m.any():
                        y, z = y[m], z[m]
                        dt = dt[m] if dt.size == m.size else _np.zeros_like(y, dtype=int)
                        dc = dc[m] if dc.size == m.size else _np.zeros_like(y, dtype=int)

                        # Auto-align MRZ across-track to match image X direction
                        try:
                            col = ya_arr.shape[1] // 2
                            delta_plot = float(ya_arr[-1, col] - ya_arr[0, col])
                            delta_mrz = float(y[-1] - y[0])
                            if _np.sign(delta_plot) != _np.sign(delta_mrz):
                                y = -y
                        except Exception:
                            pass

                        # Color & symbol
                        color_map = {0: 'red', 2: 'darkblue'}
                        is_extra = (dc != 0)
                        for t in (0, 2, -1):
                            sel = (dt == t) if t >= 0 else ~((dt == 0) | (dt == 2))
                            if not sel.any(): continue
                            c = color_map.get(t, 'gray')
                            ns = sel & (~is_extra)
                            if ns.any():
                                self.ax.scatter(y[ns], z[ns], s=12, marker='o', edgecolors='none', c=c)
                            xs = sel & is_extra
                            if xs.any():
                                self.ax.scatter(y[xs], z[xs], s=12, marker='o', facecolors='none', edgecolors=c, linewidths=0.8)

                        try:
                            self.statusBar().showMessage(f"MRZ points: {y.size}")
                        except Exception:
                            pass
        except Exception:
            # Non-fatal: continue without crashing on overlay errors
            pass
        # Refresh the fixed colorbar axis with the new mesh.
        self.cax.clear(); self.fig.colorbar(mesh, cax=self.cax, label='Amplitude')
        # Execute statement (def:redraw)
        d0, d1 = self.dmin.value(), self.dmax.value(); d1 = d1 if abs(d1 - d0) > 1e-6 else d0 + 1e-3
        # Apply the chosen depth limits to the Y axis.
        self.ax.set_ylim(d0, d1)
        # apply static X scale if provided
        try:
            # Read user-set X max (across-track) for static X-limits.
            xmax = float(self.xmax.value()) if hasattr(self, 'xmax') else 0.0
        except Exception:
            xmax = 0.0
        if xmax > 0:
            # Enforce symmetric X-limits to prevent view jitter between pings.
            self.ax.set_xlim(-xmax, xmax)
        else:
            # fallback to auto-scaling
            # If X max is 0, fall back to Matplotlib autoscale on X.
            self.ax.autoscale(enable=True, axis='x', tight=True)
            # Invert Y axis so depth increases downward.
        self.ax.invert_yaxis()
        self.ax.set_xlabel('Across-track (m)')
        self.ax.set_ylabel('Depth (m)')
        self.canvas.draw()
#

# Program entry point: build QApplication, create MainWindow, and start event loop.
    def _sync_slider(self):
        try:
            if hasattr(self, 'slider') and self.pings is not None:
                self.slider.blockSignals(True)
                self.slider.setMinimum(0)
                self.slider.setMaximum(max(0, len(self.pings) - 1))
                self.slider.setSingleStep(1)
                self.slider.setValue(int(self.current_ping))
                self.slider.blockSignals(False)
                if hasattr(self, 'lbl_ping'):
                    self.lbl_ping.setText(str(int(self.current_ping)))
        except Exception:
            pass

    def _set_ping_from_slider(self, v: int):
        try:
            self.current_ping = int(v)
            self.redraw()
        except Exception:
            pass


    def _toggle_replay(self, checked: bool):
        if checked and (self.pings is not None) and (len(self.pings) > 0):
            self._is_replaying = True
            self._update_replay_interval()
            self.replay_timer.start()
            try: self.btn_replay.setText(f"Stop ({self.replay_rate.value():.1f} p/s)")
            except Exception: pass
        else:
            self._is_replaying = False
            self.replay_timer.stop()
            try: self.btn_replay.setText("Replay")
            except Exception: pass

    def _update_replay_interval(self):
        try:
            pps = float(self.replay_rate.value())
            if pps <= 0: pps = 1.0
            self.replay_timer.setInterval(max(1, int(1000.0 / pps)))
            if self._is_replaying:
                try: self.btn_replay.setText(f"Stop ({pps:.1f} p/s)")
                except Exception: pass
        except Exception:
            pass

    def _replay_tick(self):
        if self.pings is None or len(self.pings) == 0:
            self._toggle_replay(False)
            return
        self.current_ping = (self.current_ping + 1) % len(self.pings)
        self.redraw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(1000, 800)
    win.show()
    sys.exit(app.exec())