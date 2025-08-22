BUILD_ID='v17e_diag10_mrzflip_FORCE_1'
print(f"[KMWCD Viewer] BUILD_ID={BUILD_ID}")

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
from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QTableWidgetItem,
    QTableWidget,
    QDialog,
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
    def _fmt_sog_kn(self, sog_kn):
        try:
            import math
            return f"{sog_kn:.2f} kn" if sog_kn is not None and not math.isnan(float(sog_kn)) else "N/A"
        except Exception:
            return "N/A"


    def _late_kmall_scan(self):

        print("[diag] _late_kmall_scan: start")
        try:
            import pandas as pd, numpy as np, os
            path = getattr(self, 'current_path', None)
            print(f"[diag] _late_kmall_scan: current_path={path}")
            if not path:
                print("[diag] late scan: no current_path"); return
            def _pair_files(p):
                base, ext = os.path.splitext(p)
                kmwcd = base + '.kmwcd'; kmall = base + '.kmall'
                if ext.lower()=='.kmwcd': return p, (kmall if os.path.exists(kmall) else None)
                if ext.lower()=='.kmall': return (base + '.kmwcd' if os.path.exists(kmwcd) else p), p
                return p, (kmall if os.path.exists(kmall) else None)
            wc_path, nav_path = _pair_files(path)
            print(f"[diag] _late_kmall_scan: pair wc={wc_path} nav={nav_path}")
            if not nav_path:
                print("[diag] no KMALL sibling"); return
            try:
                self.nav_loader = KMALL.kmall(nav_path); self.nav_loader.index_file()
                print('[diag] KMALL Index rows', len(self.nav_loader.Index))
            except Exception as ex:
                print('[diag] nav_loader error', ex); return
            def _norm(x):
                try:
                    if isinstance(x,(bytes,bytearray)):
                        try: x=x.decode('ascii','ignore')
                        except Exception: x=str(x)
                    s=str(x).strip()
                    if s.startswith("b'") and s.endswith("'"): s=s[2:-1]
                    if s.startswith('b"') and s.endswith('"'): s=s[2:-1]
                    if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')): s=s[1:-1]
                    if s.startswith('EMdgm'): s=s[5:]
                    if s.startswith('#'): s=s[1:]
                    return s.upper()
                except Exception: return str(x).upper()
            col=None; ser=None
            for c in ['MessageType','messageType','MsgType','DgmType','DgType','Message','Datagram','dgid','Identifier','Id','id']:
                if c in self.nav_loader.Index.columns:
                    try: ser=self.nav_loader.Index[c].map(_norm); col=c; break
                    except Exception: pass
            if ser is None:
                for c in self.nav_loader.Index.columns:
                    s=self.nav_loader.Index[c]
                    if getattr(s,'dtype',None)==object:
                        try: ser=s.map(_norm); col=c; break
                        except Exception: pass
            if ser is None: print('[diag] no message column'); return
            try:
                import pandas as _pd
                print('[diag] KMALL tags raw sample:', list(_pd.unique(self.nav_loader.Index[col].dropna()))[:30])
                print('[diag] KMALL tags norm sample:', list(_pd.unique(ser.dropna()))[:30])
            except Exception: pass
            mrz = int((ser=='MRZ').sum()); skm = int((ser=='SKM').sum()); spo = int((ser=='SPO').sum())
            # nav_df
            self.nav_df = pd.DataFrame(columns=['lat','lon','cog_deg','sog_mps'])
            try:
                recs=[]
                mask=(ser=='SPO')
                if getattr(mask,'any',lambda:False)():
                    rows=self.nav_loader.Index[mask]
                    for _,r in rows.reset_index().iterrows():
                        try:
                            dg=self.nav_loader.read_index_row(r)
                            sd=dg.get('sensorData',{})
                            t=dg.get('datetime') or dg.get('header',{}).get('dgdatetime')
                            recs.append((t, sd.get('correctedLat_deg',np.nan), sd.get('correctedLong_deg',np.nan),
                                         sd.get('courseOverGround_deg',np.nan), sd.get('speedOverGround_mps',np.nan)))
                        except Exception: pass
                if recs:
                    df=pd.DataFrame(recs, columns=['time','lat','lon','cog_deg','sog_mps'])
                    df['time']=pd.to_datetime(df['time'], utc=True, errors='coerce')
                    try: df['time']=df['time'].dt.tz_convert(None)
                    except Exception: df['time']=df['time'].dt.tz_localize(None)
                    self.nav_df=df.set_index('time').sort_index()
                    try:
                        self.nav_df['sog']=self.nav_df.get('sog_mps')
                        self.nav_df['cog']=self.nav_df.get('cog_deg')
                        self.nav_df['sog_kn']=self.nav_df['sog']*1.9438445
                    except Exception:
                        pass
                    try:
                        self.nav_df['sog']=self.nav_df.get('sog_mps')
                        self.nav_df['cog']=self.nav_df.get('cog_deg')
                        self.nav_df['sog_kn']=self.nav_df['sog']*1.9438445
                    except Exception: pass
            except Exception as ex: print('[diag] nav_df error', ex)
            # hdg_df
            self.hdg_df = pd.DataFrame(columns=['heading_deg'])
            try:
                recs=[]
                mask=(ser=='SKM')
                if getattr(mask,'any',lambda:False)():
                    rows=self.nav_loader.Index[mask]
                    for _,r in rows.reset_index().iterrows():
                        try:
                            dg=self.nav_loader.read_index_row(r)
                            samp=dg.get('sample',{}).get('KMdefault',{})
                            heads=samp.get('heading_deg',[])
                            if isinstance(heads,(list,tuple)) and len(heads):
                                import numpy as _np
                                hdg=float(_np.nanmean(_np.array(heads,dtype='float64')))
                            else: hdg=float('nan')
                            t=dg.get('datetime') or dg.get('header',{}).get('dgdatetime')
                            recs.append((t, hdg))
                        except Exception: pass
                if recs:
                    df=pd.DataFrame(recs, columns=['time','heading_deg'])
                    df['time']=pd.to_datetime(df['time'], utc=True, errors='coerce')
                    try: df['time']=df['time'].dt.tz_convert(None)
                    except Exception: df['time']=df['time'].dt.tz_localize(None)
                    self.hdg_df=df.set_index('time').sort_index()
            except Exception as ex: print('[diag] hdg_df error', ex)
            print(f"[diag] counts (KMALL, late) mrz={mrz} skm={skm} spo={spo} | tables nav={len(self.nav_df)} hdg={len(self.hdg_df)}")
            try:
                self.statusBar().showMessage(f"Nav summary: MRZ={mrz}, SKM={skm}, SPO={spo} | tables nav={len(self.nav_df)}, hdg={len(self.hdg_df)}", 1500)
                QTimer.singleShot(900, lambda: self.statusBar().showMessage(
                    f"Nav summary: MRZ={mrz}, SKM={skm}, SPO={spo} | tables nav={len(self.nav_df)}, hdg={len(self.hdg_df)}", 6000))
            except Exception as ex:
                print('[diag] status blip error', ex)
        except Exception as ex:
            print("[diag] late scan fatal", ex)

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
        self.chk_bottom = QCheckBox("Bottom")
        self.chk_bottom.setChecked(False)
        self.chk_bottom.stateChanged.connect(lambda _ : self.redraw())
        hh.addWidget(self.chk_bottom)
        # Pick switch (toggle picking mode)
        self.chk_pick = QCheckBox("Pick")
        self.chk_pick.setChecked(False)
        self.chk_pick.clicked.connect(self._show_pick_dialog)
        self.chk_pick.toggled.connect(self._toggle_pick)
        hh.addWidget(self.chk_pick)
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
        try:
            self.canvas.mpl_connect('button_press_event', self._on_plot_click)
        except Exception:
            pass
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
        # Picking state
        self._pick_dialog = None
        self._ping_dialog = None  # alias for compatibility
        self._pick_dialog_shown_once = False
        self._pick_active = False

#

    # File open handler: index datagrams, collect SPO, compute global ranges with progress.
    def open_file(self):
        # Pop up native file chooser for .kmwcd/.kmall files.
        fname, _ = QFileDialog.getOpenFileName(self, "Open .kmwcd/.kmall", "", "*.kmwcd *.kmall")
        if not fname:
            return
        self.lbl_file.setText(os.path.basename(fname)); self.current_path=fname; print('[diag] open_file: current_path', fname)
        self.statusBar().showMessage("Indexing…"); QApplication.processEvents()
        # Instantiate KMALL reader for the selected file.
        self.loader = KMALL.kmall(fname); self.loader.index_file()
        idx = self.loader.Index.copy()
        # Build navigation (SPO) and heading (SKM) tables for azimuth/picking
        self.nav_df = None
        self.hdg_df = None
        try:
            import pandas as _pd, numpy as _np
            msg = idx['MessageType'].astype(str) if 'MessageType' in idx.columns else None
            spo_idx = idx[msg.str.contains("b'#SPO'", regex=False, na=False)] if msg is not None else _pd.DataFrame()
            skm_idx = idx[msg.str.contains("b'#SKM'", regex=False, na=False)] if msg is not None else _pd.DataFrame()
            # SPO -> lat, lon, COG, SOG
            recs = []
            for t, row in spo_idx.iterrows():
                try:
                    self.loader.FID.seek(int(row['ByteOffset']), 0)
                    spo = self.loader.read_EMdgmSPO()
                    sd = spo.get('sensorData', {})
                    recs.append({
                        'time': t,
                        'lat': sd.get('correctedLat_deg', _np.nan),
                        'lon': sd.get('correctedLong_deg', _np.nan),
                        'cog_deg': sd.get('courseOverGround_deg', _np.nan),
                        'sog_mps': sd.get('speedOverGround_mPerSec', _np.nan),
                    })
                except Exception:
                    pass
            if recs:
                self.nav_df = _pd.DataFrame(recs).dropna(subset=['time']).drop_duplicates(subset=['time']).set_index('time').sort_index()
            # SKM -> mean heading per block at block time
            hrec = []
            for t, row in skm_idx.iterrows():
                try:
                    self.loader.FID.seek(int(row['ByteOffset']), 0)
                    skm = self.loader.read_EMdgmSKM()
                    samp = skm.get('sample', {}).get('KMdefault', {})
                    heads = _np.array(samp.get('heading_deg', []), dtype=float)
                    if heads.size:
                        hrec.append({'time': t, 'heading_deg': float(_np.nanmean(heads))})
                except Exception:
                    pass
            if hrec:
                self.hdg_df = _pd.DataFrame(hrec).dropna(subset=['time']).drop_duplicates(subset=['time']).set_index('time').sort_index()
        except Exception:
            self.nav_df = None; self.hdg_df = None

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

    # Slider changed: record new ping index and trigger redraw.        # --- Normalize nav/heading indices to pandas datetime (UTC naive) ---
        try:
            import pandas as _pd
            for _name in ('nav_df','hdg_df'):
                _df = getattr(self, _name, None)
                if _df is not None and len(_df):
                    _t = _pd.to_datetime(_df.index, utc=True, errors='coerce')
                    _t = _t.tz_convert('UTC').tz_localize(None)
                    _df = _df.assign(__t__=_t).dropna(subset=['__t__']).set_index('__t__').sort_index()
                    setattr(self, _name, _df)
        except Exception:
            pass
        print('[diag] open_file: schedule _late_kmall_scan')
        QTimer.singleShot(0, self._late_kmall_scan)

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
    def _toggle_pick(self, on: bool):
        self._pick_active = bool(on)
        try:
            self.canvas.setCursor(Qt.CrossCursor if self._pick_active else Qt.ArrowCursor)
        except Exception:
            pass

    def _show_pick_dialog(self, checked=False):
        try:
            if self._pick_dialog is None:
                self._pick_dialog = _PickDialog(self)
                try:
                    self._pick_dialog.finished.connect(lambda _=None: (self.chk_pick.setChecked(False) if hasattr(self,'chk_pick') else None))
                except Exception:
                    pass
                self._ping_dialog = self._pick_dialog
                try:
                    w = max(600, int(self.width()))
                    h = max(240, int(self.height() // 3))
                    self._pick_dialog.resize(w, h)
                    self._pick_dialog.setMinimumSize(600, 240)
                except Exception:
                    pass
            self._pick_dialog.show()
            if not self._pick_dialog_shown_once:
                try:
                    self._pick_dialog.raise_(); self._pick_dialog.activateWindow()
                except Exception:
                    pass
                self._pick_dialog_shown_once = True
        except Exception as e:
            try:
                self.statusBar().showMessage(f"Pick Error Dialog: {e}", 6000)
            except Exception:
                pass

    
    def _get_nav_for_time(self, when):
        """Return (lat, lon, cog_deg, sog_mps) nearest to time `when` (tz-robust)."""
        import numpy as _np
        import pandas as _pd
        df = getattr(self, 'nav_df', None)
        if when is None or df is None or len(df) == 0:
            return float('nan'), float('nan'), float('nan'), float('nan')
        try:
            wt = _pd.to_datetime(when, utc=True, errors='coerce')
            try: wt = wt.tz_convert(None)
            except Exception: wt = wt.tz_localize(None)
            idxer = df.index.get_indexer([wt], method='nearest')
            pos = int(idxer[0]) if getattr(idxer, 'size', 0) else -1
            if pos >= 0:
                rec = df.iloc[pos]
                lat = float(rec.get('lat', _np.nan)); lon = float(rec.get('lon', _np.nan))
                cog = float(rec.get('cog_deg', rec.get('cog', _np.nan)))
                sog = float(rec.get('sog_mps', rec.get('sog', _np.nan)))
                return lat, lon, cog, sog
            return float('nan'), float('nan'), float('nan'), float('nan')
        except Exception:
            return float('nan'), float('nan'), float('nan'), float('nan')

    def _get_nav_heading(self, when):
        """Return 6-tuple: (lat, lon, azimuth_deg, heading_deg, cog_deg, sog_mps)."""

        import numpy as _np, math
        lat, lon, cog, sog = self._get_nav_for_time(when)
        heading = float('nan')
        # nearest mean heading from SKM block
        try:
            hdf = getattr(self, 'hdg_df', None)
            if when is not None and hdf is not None and len(hdf) > 0:
                idxer = hdf.index.get_indexer([when], method='nearest')
                pos = int(idxer[0]) if getattr(idxer, 'size', 0) else -1
                if pos >= 0:
                    heading = float(hdf.iloc[pos].get('heading_deg', _np.nan))
        except Exception:
            pass
        if not _np.isfinite(heading):
            heading = cog
        az = heading if _np.isfinite(heading) else cog
        return (
            float(lat) if lat==lat else float('nan'),
            float(lon) if lon==lon else float('nan'),
            float(az) if az==az else float('nan'),
            float(heading) if heading==heading else float('nan'),
            float(cog) if cog==cog else float('nan'),
            float(sog) if sog==sog else float('nan'),
        )

    def _on_plot_click(self, event):
        """Left-click to pick nearest cell; requires Pick ON."""
        try:
            if (event.button != 1 or event.inaxes is None or event.inaxes is not getattr(self, 'ax', None) or not getattr(self, '_pick_active', False)):
                return

            lg = getattr(self, '_last_grid', None)
            if not lg:
                try: self.statusBar().showMessage('Nothing to pick (no grid).', 3000)
                except Exception: pass
                return

            import numpy as _np
            x = float(event.xdata); y = float(event.ydata)
            ya = lg.get('ya_arr'); da = lg.get('depth_arr'); A = lg.get('amp_arr'); ping_time = lg.get('ping_time')
            if ya is None or da is None:
                return
            d2 = (ya - x)**2 + (da - y)**2
            i, j = _np.unravel_index(_np.nanargmin(d2), d2.shape)
            across = float(ya[i, j]); depth = float(da[i, j])
            amp = _np.nan
            try: amp = float(A[i, j])
            except Exception:
                try: amp = float(A[j, i])
                except Exception: pass

            # Navigation & azimuth
            lat = lon = az = heading = cog = float('nan')
            if hasattr(self, '_get_nav_heading'):
                nav_tuple = self._get_nav_heading(ping_time)
                try:
                    vals = list(nav_tuple) + [float('nan')]*6
                    lat, lon, az, heading, cog, sog = vals[:6]
                except Exception:
                    lat = lon = az = heading = cog = sog = float('nan')
                nav_tuple = self._get_nav_heading(ping_time)
                try:
                    _vals = list(nav_tuple) if isinstance(nav_tuple, (list, tuple)) else []
                    _vals += [float('nan')]*6
                    lat, lon, az, heading, cog, sog = _vals[:6]
                except Exception:
                    lat = lon = az = heading = cog = sog = float('nan')
            elif hasattr(self, '_get_nav_for_time'):
                nav4 = self._get_nav_for_time(ping_time)
                try:
                    _v4 = list(nav4) if isinstance(nav4, (list, tuple)) else []
                    _v4 += [float('nan')]*4
                    lat, lon, cog, sog = _v4[:4]
                except Exception:
                    lat = lon = cog = sog = float('nan')

            
            # Determine azimuth source for status display
            az_status = 'None'
            try:
                import numpy as _np
                if _np.isfinite(heading):
                    az_status = 'Heading'
                elif _np.isfinite(cog):
                    az_status = 'COG'
            except Exception:
                az_status = 'None'
            import math
            if az == az:
                rad = math.radians(az)
                east_m  =  across * math.cos(rad)
                north_m = -across * math.sin(rad)
            else:
                east_m, north_m = across, 0.0

            def _meters_to_latlon(lat_deg, east_m, north_m):
                lat_rad = math.radians(lat_deg if lat_deg == lat_deg else 0.0)
                dlat = north_m / 111320.0
                dlon = east_m / (111320.0 * max(1e-6, math.cos(lat_rad)))
                return dlat, dlon

            dlat, dlon = _meters_to_latlon(lat, east_m, north_m) if (lat == lat and lon == lon) else (float('nan'), float('nan'))
            pt_lat = lat + dlat if (lat == lat and dlat == dlat) else float('nan')
            pt_lon = lon + dlon if (lon == lon and dlon == dlon) else float('nan')

            ping_idx = int(getattr(self, 'current_ping', -1))
            utc_str = ''
            # Azimuth status from values
            az_status = 'None'
            try:
                import numpy as _np
                if _np.isfinite(heading): az_status = 'Heading'
                elif _np.isfinite(cog): az_status = 'COG'
            except Exception:
                az_status = 'None'
            try:
                if ping_time is not None:
                    utc_str = ping_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] + ' UTC'
            except Exception:
                pass

            
            # --- Determine Azimuth status from actual values used ---
            try:
                import numpy as _np, math as _math
                def _ang_diff(a, b):
                    if not (_np.isfinite(a) and _np.isfinite(b)):
                        return float('inf')
                    d = (a - b + 540.0) % 360.0 - 180.0
                    return abs(d)
                _d_h = _ang_diff(az, heading)
                _d_c = _ang_diff(az, cog)
                if _np.isfinite(az) and _d_h <= 0.01:
                    az_status = 'Heading'
                elif _np.isfinite(az) and _d_c <= 0.01:
                    az_status = 'COG'
                elif _np.isfinite(heading):
                    az_status = 'Heading'
                elif _np.isfinite(cog):
                    az_status = 'COG'
                else:
                    az_status = 'None'
            except Exception:
                az_status = 'None'
            # --- Fallback: if az_status is None, probe KMALL index for SPO/SKM near ping_time ---
            try:
                if az_status == 'None' and ping_time is not None and hasattr(self, 'loader') and hasattr(self.loader, 'Index'):
                    import numpy as _np
                    import pandas as _pd
                    idx = self.loader.Index
                    if isinstance(idx, _pd.DataFrame) and 'MessageType' in idx.columns and 'ByteOffset' in idx.columns:
                        msg = idx['MessageType'].astype(str)
                        # Nearest SPO -> COG
                        try:
                            spo_idx = idx[msg.str.contains("b'#SPO'", regex=False, na=False)]
                            if len(spo_idx) > 0:
                                times = spo_idx.index.values.astype(float)
                                spi = int(_np.argmin(_np.abs(times - float(ping_time))))
                                self.loader.FID.seek(int(spo_idx.iloc[spi]['ByteOffset']), 0)
                                spo = self.loader.read_EMdgmSPO()
                                sd = spo.get('sensorData', {})
                                _cog = float(sd.get('courseOverGround_deg', float('nan')))
                                if _np.isfinite(_cog):
                                    az_status = 'COG'
                        except Exception:
                            pass
                        # Nearest SKM -> Heading (mean within block)
                        if az_status == 'None':
                            try:
                                skm_idx = idx[msg.str.contains("b'#SKM'", regex=False, na=False)]
                                if len(skm_idx) > 0:
                                    times = skm_idx.index.values.astype(float)
                                    ski = int(_np.argmin(_np.abs(times - float(ping_time))))
                                    self.loader.FID.seek(int(skm_idx.iloc[ski]['ByteOffset']), 0)
                                    skm = self.loader.read_EMdgmSKM()
                                    samp = skm.get('sample', {}).get('KMdefault', {})
                                    heads = _np.array(samp.get('heading_deg', []), dtype=float)
                                    if heads.size and _np.isfinite(_np.nanmean(heads)):
                                        az_status = 'Heading'
                            except Exception:
                                pass
            except Exception:
                pass
            row = [ping_idx, utc_str,
                f"{across:.2f}", f"{depth:.2f}",
                '' if lat != lat else f"{lat:.6f}", '' if lon != lon else f"{lon:.6f}",
                '' if pt_lat != pt_lat else f"{pt_lat:.6f}", '' if pt_lon != pt_lon else f"{pt_lon:.6f}"
            , az_status]

            if getattr(self, '_pick_dialog', None) is None:
                self._show_pick_dialog()
            if self._pick_dialog is not None:
                try: self._pick_dialog.add_row(row)
                except Exception as e:
                    try: self.statusBar().showMessage(f'Pick add-row error: {e}', 5000)
                    except Exception: pass

        except Exception as e:
            try: self.statusBar().showMessage(f'Pick click error: {e}', 5000)
            except Exception: pass

    def redraw(self):
        # Guard: nothing to draw before a file is open
        if getattr(self, 'loader', None) is None or getattr(self, 'pings', None) is None or self.pings is None or len(self.pings) == 0:
            self.ax.clear(); self.canvas.draw_idle(); return
        # keep slider synced with current ping
        self._sync_slider()
        dg = self.loader.read_index_row(self.pings.iloc[self.current_ping])
        # Retrieve the ping's UTC timestamp from the header.
        hdr = dg.get('header', {}); dt = hdr.get('dgdatetime'); ping_time = dt.astimezone(timezone.utc) if isinstance(dt, datetime) else None
        # Fetch nav for label (SOG knots, position)
        try:
            lat, lon, cog, sog_mps = self._get_nav_for_time(ping_time)
        except Exception:
            lat = lon = cog = sog_mps = float('nan')
        sog_kn = (sog_mps * 1.9438445) if (isinstance(sog_mps,(int,float)) and sog_mps==sog_mps) else float('nan')
        lat_str = f"{lat:.6f}" if (isinstance(lat,(int,float)) and lat==lat) else 'N/A'
        lon_str = f"{lon:.6f}" if (isinstance(lon,(int,float)) and lon==lon) else 'N/A'
        time_str = (ping_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] + ' UTC') if ping_time else 'N/A'
        sog_text = (f"{sog_kn:.2f} kn" if (isinstance(sog_kn,(int,float)) and sog_kn==sog_kn) else 'N/A')
        lat = lon = sog = np.nan
        if self.nav_df is not None and ping_time:
            # Find the closest SPO record by time for nav metadata.
            import pandas as _pd, numpy as _np
            _idx = self.nav_df.index
            try:
                _sec = (_pd.to_datetime(_idx, utc=True, errors='coerce').astype('int64')/1e9).values
            except Exception:
                try:
                    _sec = _idx.values.astype('int64')/1e9
                except Exception:
                    _sec = _idx.values.astype(float)
            _pt = ping_time.timestamp() if hasattr(ping_time, 'timestamp') else float(ping_time)
            pos = int(_np.argmin(_np.abs(_sec - _pt)))
            if pos >= 0:
                navrec = self.nav_df.iloc[pos]
                lat = navrec.get('lat')
                lon = navrec.get('lon')
                sog = navrec.get('sog', navrec.get('sog_mps'))
                try:
                    sog_kn = float(sog) * 1.9438445 if sog is not None else float('nan')
                except Exception:
                    sog_kn = float('nan')
        sog_str = f"{sog:.2f}" if np.isfinite(sog) else 'N/A'
        lat_str = f"{lat:.6f}" if np.isfinite(lat) else 'N/A'
        lon_str = f"{lon:.6f}" if np.isfinite(lon) else 'N/A'
        time_str = ping_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + " UTC" if ping_time else 'N/A'
        # Display the ping metadata (UTC time, SOG, latitude, longitude).
        self.meta_label.setText(f"Ping {self.current_ping}: DateTime={time_str}, SOG={sog_text}, Pos={lat_str},{lon_str}")
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
        try:
            self._last_grid = {
                'ya_arr': ya_arr.copy() if 'ya_arr' in locals() else None,
                'depth_arr': (depth_arr.copy() if 'depth_arr' in locals() else ((-za_arr).copy() if 'za_arr' in locals() else None)),
                'amp_arr': data.copy() if 'data' in locals() else None,
                'ping_time': ping_time if 'ping_time' in locals() else None
            }
        except Exception:
            self._last_grid = None

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
                    dt = _np.asarray(s.get('detectionType', []), dtype=int)    # 0=normal,1=extra,2=rejected
                    dc = _np.asarray(s.get('detectionClass', []), dtype=int)   # 0=normal,1=extra
                    dm = _np.asarray(s.get('detectionMethod', []), dtype=int)  # 1=Amplitude, 2=Phase
                    m = _np.isfinite(y) & _np.isfinite(z)
                    if m.any():
                        y, z = y[m], z[m]
                        # Always flip MRZ across-track to match WCD orientation
                        y = -y
# Align aux arrays
                        dt = dt[m] if dt.size == m.size else _np.zeros_like(y, dtype=int)
                        dc = dc[m] if dc.size == m.size else _np.zeros_like(y, dtype=int)
                        dm = dm[m] if dm.size == m.size else _np.zeros_like(y, dtype=int)

                        # Color by detectionMethod; override to green for rejected detectionType==2
                        base_color = _np.where(dm==1, 'red', _np.where(dm==2, 'darkblue', 'gray'))
                        color = base_color.copy()
                        color = _np.where(dt==2, 'green', color)

                        # Symbol by detectionClass (solid normal, hollow extra)
                        is_extra = (dc != 0)
                        # normal (filled)
                        sel = ~is_extra
                        if sel.any():
                            y = -y  # hard flip MRZ across-track
                            self.ax.scatter(y[sel], z[sel], s=12, marker='o', edgecolors='none', c=color[sel])
                        # extra (hollow)
                        sel = is_extra
                        if sel.any():
                            self.ax.scatter(y[sel], z[sel], s=12, marker='o', facecolors='none', edgecolors=color[sel], linewidths=0.8)

                        try:
                            self.statusBar().showMessage(f"MRZ flip=ON, points: {y.size}")
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




class _PickDialog(QDialog):
    COLS = ["Ping","UTC","AcrossTrack_m","Depth_m","ShipLat","ShipLon","PointLat","PointLon","Azimuth"]
    closed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Picked Positions")
        from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout
        self.table = QTableWidget(0, len(self.COLS), self)
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.btn_save = QPushButton("Save as TXT")
        self.btn_remove = QPushButton("Remove selected")
        self.btn_close = QPushButton("Close")
        self.btn_save.clicked.connect(self._save_txt)
        self.btn_remove.clicked.connect(self._remove_selected)
        self.btn_close.clicked.connect(self._clear_and_close)
        hl = QHBoxLayout()
        hl.addWidget(self.btn_save)
        hl.addWidget(self.btn_remove)
        hl.addStretch(1)
        hl.addWidget(self.btn_close)
        vl = QVBoxLayout(self)
        vl.addWidget(self.table)
        vl.addLayout(hl)

    def add_row(self, values):
        r = self.table.rowCount()
        self.table.insertRow(r)
        for c, v in enumerate(values):
            self.table.setItem(r, c, QTableWidgetItem("" if v is None else str(v)))

    def clear_rows(self):
        self.table.setRowCount(0)

    def _remove_selected(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)

    def _save_txt(self):
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(self, "Save Picks", "picks.txt", "Text Files (*.txt);;All Files (*)")
        if not path: return
        with open(path, "w") as f:
            f.write("\t".join(self.COLS) + "\n")
            for r in range(self.table.rowCount()):
                vals = [self.table.item(r, c).text() if self.table.item(r,c) else "" for c in range(self.table.columnCount())]
                f.write("\t".join(vals) + "\n")

    def _clear_and_close(self):
        self.clear_rows()
        try: self.closed.emit()
        except Exception: pass
        self.close()


def _bootstrap():
    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec()

if __name__ == "__main__":
    import sys
    print("[KMWCD Viewer] Entering Qt event loop...")
    sys.exit(_bootstrap())