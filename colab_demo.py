# ==============================================================
# nmrmetaproc v1.0.0 -- Google Colab Demo
# Folorunsho Bright Omage, Toyin Bright Omage, Ljubica Tasic | DOI: 10.5281/zenodo.19194107
# No external data download needed -- generates its own FIDs
# ==============================================================

# ---- CELL 1: Install ----------------------------------------
import subprocess
subprocess.run(["pip", "install", "nmrmetaproc", "-q"], check=True)
import nmrmetaproc
print(f"nmrmetaproc {nmrmetaproc.__version__}")

# ---- CELL 2: Generate synthetic Bruker FID data -------------
import numpy as np, os

def make_bruker_fid(out_dir, metabolites, noise=0.001,
                    sfo1=600.17, sw_hz=9090.91, td=65536, o1_hz=2820.8):
    os.makedirs(os.path.join(out_dir, "pdata", "1"), exist_ok=True)
    dt = 1.0 / sw_hz
    t = np.arange(td) * dt
    fid = np.zeros(td, dtype=complex)
    for _n, ppm, amp, t2, mult in metabolites:
        freq = (ppm - o1_hz / sfo1) * sfo1
        for m in range(mult):
            off = (m - (mult - 1) / 2) * 7.0
            fid += amp * np.exp(2j*np.pi*(freq+off)*t) * np.exp(-t/t2)
    tsp_f = (0.0 - o1_hz / sfo1) * sfo1
    fid += 0.5 * np.exp(2j*np.pi*tsp_f*t) * np.exp(-t/1.5)
    fid += noise * (np.random.randn(td) + 1j*np.random.randn(td))
    sc = fid * 1e6
    raw = np.empty(2*td, dtype=np.int32)
    raw[0::2] = sc.real.clip(-2**31, 2**31-1).astype(np.int32)
    raw[1::2] = sc.imag.clip(-2**31, 2**31-1).astype(np.int32)
    open(os.path.join(out_dir,"fid"),"wb").write(raw.tobytes())
    acqus = (
        "##TITLE= Synthetic\n##JCAMPDX= 5.0\n##DATATYPE= Parameter Values\n"
        f"##$SFO1= {sfo1}\n##$BF1= {sfo1}\n##$O1= {o1_hz}\n"
        f"##$SW_h= {sw_hz}\n##$SW= {sw_hz/sfo1:.6f}\n##$TD= {td*2}\n"
        "##$NS= 128\n##$DS= 4\n##$RG= 101\n"
        f"##$AQ= {td*dt:.6f}\n##$DW= {dt*1e6:.3f}\n"
        "##$NUC1= <1H>\n##$PULPROG= <noesygppr1d>\n##$SOLVENT= <H2O>\n"
        "##$BYTORDA= 0\n##$DTYPA= 0\n##$DECIM= 2\n##$DSPFVS= 0\n##$GRPDLY= 0\n"
        "##$D= (0..63)\n4.0 1.5 " + "0 "*62 + "\n##END=\n"
    )
    open(os.path.join(out_dir,"acqus"),"w").write(acqus)
    procs = (
        "##TITLE= procs\n##JCAMPDX= 5.0\n##DATATYPE= Parameter Values\n"
        f"##$SI= 32768\n##$SF= {sfo1:.4f}\n##$OFFSET= 14.0\n"
        f"##$SW_p= {sw_hz:.2f}\n##$LB= 0.30\n##$WDW= 1\n"
        "##$PHC0= 0.0\n##$PHC1= 0.0\n##END=\n"
    )
    open(os.path.join(out_dir,"pdata","1","procs"),"w").write(procs)

CTRL = [
    ("VLDL",       0.86, 1.5,  0.30, 1),
    ("Isoleucine", 0.94, 0.20, 0.70, 1),
    ("Leucine",    0.96, 0.25, 0.70, 2),
    ("Valine",     0.99, 0.35, 0.70, 2),
    ("3-HB",       1.20, 0.30, 0.70, 2),
    ("LDL",        1.26, 1.20, 0.30, 1),
    ("Lactate",    1.33, 1.0,  0.80, 2),
    ("Alanine",    1.48, 0.40, 0.70, 2),
    ("Acetate",    1.92, 0.30, 0.90, 1),
    ("Glutamate",  2.35, 0.35, 0.60, 1),
    ("Citrate",    2.54, 0.50, 0.70, 2),
    ("Creatine",   3.04, 0.45, 0.80, 1),
    ("Choline",    3.20, 0.25, 0.70, 1),
    ("Glucose-a",  3.42, 0.80, 0.60, 1),
    ("Glucose-b",  3.73, 0.70, 0.60, 1),
    ("Glucose-an", 5.24, 0.40, 0.50, 2),
    ("Tyrosine",   6.90, 0.15, 0.50, 2),
    ("Formate",    8.46, 0.10, 0.90, 1),
]
DIS = [
    ("VLDL",       0.86, 2.2,  0.30, 1),
    ("Isoleucine", 0.94, 0.13, 0.70, 1),
    ("Leucine",    0.96, 0.16, 0.70, 2),
    ("Valine",     0.99, 0.22, 0.70, 2),
    ("3-HB",       1.20, 0.70, 0.70, 2),
    ("LDL",        1.26, 1.90, 0.30, 1),
    ("Lactate",    1.33, 1.90, 0.80, 2),
    ("Alanine",    1.48, 0.50, 0.70, 2),
    ("Acetate",    1.92, 0.50, 0.90, 1),
    ("Glutamate",  2.35, 0.55, 0.60, 1),
    ("Citrate",    2.54, 0.20, 0.70, 2),
    ("Creatine",   3.04, 0.28, 0.80, 1),
    ("Choline",    3.20, 0.40, 0.70, 1),
    ("Glucose-a",  3.42, 1.30, 0.60, 1),
    ("Glucose-b",  3.73, 1.10, 0.60, 1),
    ("Glucose-an", 5.24, 0.65, 0.50, 2),
    ("Tyrosine",   6.90, 0.22, 0.50, 2),
    ("Formate",    8.46, 0.17, 0.90, 1),
]

np.random.seed(42)
for i in range(1, 7):
    m = [(n,p,a*(1+0.12*np.random.randn()),max(0.1,t2*(1+0.08*np.random.randn())),mu) for n,p,a,t2,mu in CTRL]
    make_bruker_fid(f"demo_nmr/Control/{i}", m)
for i in range(1, 7):
    m = [(n,p,a*(1+0.12*np.random.randn()),max(0.1,t2*(1+0.08*np.random.randn())),mu) for n,p,a,t2,mu in DIS]
    make_bruker_fid(f"demo_nmr/Disease/{i}", m)
print("Generated 6 Control + 6 Disease Bruker FID samples in demo_nmr/")

# ---- CELL 3: Process with nmrmetaproc -----------------------
from nmrmetaproc import NMRProcessor
proc = NMRProcessor(lb=0.3, bin_width=0.01, ppm_range=(0.5, 9.5),
                    normalization="pqn", snr_threshold=10, linewidth_threshold=5.0)
results = proc.process("demo_nmr")
results.save("demo_output")
print(f"Passed QC: {results.n_passed}/{results.n_total}")
print(f"Matrix:    {results.spectral_matrix.shape}")
print(f"Negatives: {(results.spectral_matrix < 0).any().any()}")

# ---- CELL 4: QC report --------------------------------------
import pandas as pd
matrix = pd.read_csv("demo_output/spectral_matrix.csv", index_col=0)
qc     = pd.read_csv("demo_output/qc_report.csv")
print(qc[["sample_id","snr","linewidth_hz","passed"]].to_string(index=False))

# ---- CELL 5: Full spectra plot ------------------------------
import matplotlib.pyplot as plt, matplotlib.patches as mpatches
ppm = [float(c) for c in matrix.columns]
clr = {"Control":"#2196F3", "Disease":"#F44336"}
fig, ax = plt.subplots(figsize=(16,5))
for s in matrix.index:
    g = "Control" if s.startswith("Control") else "Disease"
    ax.plot(ppm, matrix.loc[s], color=clr[g], alpha=0.6, lw=0.7)
ax.invert_xaxis()
ax.set_xlabel("Chemical Shift (ppm)", fontsize=13)
ax.set_ylabel("PQN-normalised Intensity", fontsize=13)
ax.set_title("nmrmetaproc -- Processed 1H NMR Spectra", fontsize=14)
ax.legend(handles=[mpatches.Patch(color=c,label=g) for g,c in clr.items()], fontsize=11)
plt.tight_layout(); plt.savefig("full_spectra.png", dpi=300); plt.show()

# ---- CELL 6: Zoom metabolite regions ------------------------
ppm_arr = np.array(ppm)
fig, axes = plt.subplots(2, 3, figsize=(18,10))
regions = [
    (0.82, 1.05, "BCAAs + VLDL"),
    (1.22, 1.55, "Lactate + LDL"),
    (1.88, 2.00, "Acetate"),
    (2.48, 2.62, "Citrate"),
    (2.98, 3.30, "Creatine + Choline"),
    (3.38, 3.85, "Glucose"),
]
for ax, (lo, hi, title) in zip(axes.flat, regions):
    mask = (ppm_arr >= lo) & (ppm_arr <= hi)
    for s in matrix.index:
        g = "Control" if s.startswith("Control") else "Disease"
        ax.plot(ppm_arr[mask], matrix.loc[s].values[mask], color=clr[g], alpha=0.7, lw=1.0)
    ax.invert_xaxis(); ax.set_xlabel("ppm"); ax.set_title(title, fontsize=11)
axes[0,0].legend(handles=[mpatches.Patch(color=c,label=g) for g,c in clr.items()], fontsize=9)
plt.suptitle("nmrmetaproc -- Key Metabolite Regions", fontsize=14)
plt.tight_layout(); plt.savefig("regions.png", dpi=300); plt.show()

# ---- CELL 7: Group comparison -------------------------------
peaks = {
    "Lactate (1.33)":  (1.28, 1.38),
    "LDL (1.26)":      (1.22, 1.30),
    "Acetate (1.92)":  (1.88, 1.96),
    "Citrate (2.54)":  (2.50, 2.58),
    "Creatine (3.04)": (3.00, 3.08),
    "Glucose (3.42)":  (3.38, 3.46),
}
ctrl = matrix[matrix.index.str.startswith("Control")]
dis  = matrix[matrix.index.str.startswith("Disease")]
expected = {"Lactate (1.33)":"UP","LDL (1.26)":"UP","Acetate (1.92)":"UP",
            "Citrate (2.54)":"DOWN","Creatine (3.04)":"DOWN","Glucose (3.42)":"UP"}
labels, mc_list, md_list = [], [], []
print(f"{'Metabolite':<22} {'Control':>12} {'Disease':>12} {'Ratio':>8}  Result  Expected")
print("-"*75)
for name, (lo, hi) in peaks.items():
    mask = (ppm_arr >= lo) & (ppm_arr <= hi)
    c = ctrl.values[:, mask].sum(1).mean()
    d = dis.values[:, mask].sum(1).mean()
    r = d / c if c > 0 else float("nan")
    obs = "UP" if r > 1.1 else ("DOWN" if r < 0.9 else "SAME")
    ok = "OK" if obs == expected[name] else "FAIL"
    print(f"{name:<22} {c:>12.3e} {d:>12.3e} {r:>8.2f}  {obs:<5}   {expected[name]} [{ok}]")
    labels.append(name.split("(")[0].strip()); mc_list.append(c); md_list.append(d)

x = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(12,5))
ax.bar(x-0.18, mc_list, 0.35, label="Control", color="#2196F3", alpha=0.85)
ax.bar(x+0.18, md_list, 0.35, label="Disease",  color="#F44336", alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=11)
ax.set_ylabel("Mean Integrated Intensity", fontsize=12)
ax.set_title("Group Comparison -- nmrmetaproc Output", fontsize=13)
ax.legend(fontsize=11)
plt.tight_layout(); plt.savefig("group_comparison.png", dpi=300); plt.show()

# ---- Citation -----------------------------------------------
print("""
Cite nmrmetaproc:
  Omage, F. B., Omage, T. B., & Tasic, L. (2026). nmrmetaproc: NMR Metabolomics Spectral Processor (v1.0.0).
  Zenodo. https://doi.org/10.5281/zenodo.19194107
""")
