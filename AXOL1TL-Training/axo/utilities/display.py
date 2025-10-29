import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
import math
import io
import base64
import math
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
from matplotlib.lines import Line2D

def dict_to_html_table(config_dict):
    html = "<table border='1' cellpadding='5' cellspacing='0' style='border-collapse: collapse;'>"
    for key, value in config_dict.items():
        if isinstance(value, dict):
            html += f"<tr><td><strong>{key}</strong></td><td>{dict_to_html_table(value)}</td></tr>"
        else:
            html += f"<tr><td><strong>{key}</strong></td><td>{value}</td></tr>"
    html += "</table>"
    return html


import numpy as np
import matplotlib.pyplot as plt
import io
import base64

def plot_embedding_pca(axo_embed,axo_man):
    pca = PCA(n_components=2)

    # Training the PCA
    embedding_pca = pca.fit_transform(axo_embed.test_bkg_embedding_vicreg)
    # Using on MC
    signal_pca_dict = {}
    for signal_name in axo_man.data_file["Signal_data"].keys():
        signal_pca_dict[signal_name] = pca.transform(axo_embed.signal_embedding_dict_vicreg[signal_name])



    signal_names = list(axo_man.data_file["Signal_data"].keys())
    n_signals = len(signal_names)
    ncols = 3
    nrows = (n_signals + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 5))
    axes = axes.flatten()
    
    Nsamples = 10000
    percentile_levels = np.linspace(90, 99, 15)
    
    for idx, signal_name in enumerate(signal_names):
        ax = axes[idx]
    
        # Extract signal and background data
        bg_data = embedding_pca[:Nsamples, :2].T
        sg_data = signal_pca_dict[signal_name][:Nsamples, :2].T
    
        # Determine individual ranges for plotting
        all_data = np.hstack([bg_data, sg_data])
        xmin, xmax = all_data[0].min(), all_data[0].max()
        ymin, ymax = all_data[1].min(), all_data[1].max()
    
        xx, yy = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
        grid_coords = np.vstack([xx.ravel(), yy.ravel()])
    
        # KDE for background
        bg_kde = gaussian_kde(bg_data)
        bg_density = bg_kde(grid_coords).reshape(xx.shape)
        bg_levels = np.percentile(bg_density.flatten(), percentile_levels)
        ax.contour(xx, yy, bg_density, levels=bg_levels, colors='red', linewidths=1.0)
    
        # KDE for signal
        sg_kde = gaussian_kde(sg_data)
        sg_density = sg_kde(grid_coords).reshape(xx.shape)
        sg_levels = np.percentile(sg_density.flatten(), percentile_levels)
        ax.contour(xx, yy, sg_density, levels=sg_levels, colors='blue', linewidths=1.0)
    
        # Labels
        ax.set_title(signal_name, fontsize=12)
        ax.set_xlabel("PC 1", fontsize=10)
        ax.set_ylabel("PC 2", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.3)
    
    # Remove unused axes
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])
    
    # Custom legend (shared)
    legend_lines = [
        Line2D([0], [0], color="red", lw=2, label="Zerobias Background"),
        Line2D([0], [0], color="blue", lw=2, label="Signal")
    ]
    fig.legend(handles=legend_lines, loc="upper center", ncol=2, frameon=True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode("utf-8")
    return img_str


def plot_vae_reconstruction(axo_embed,axo_man):
    No_embeddings = axo_embed.test_bkg_embedding_vicreg.shape[-1]
    ncols = 3
    nrows = (No_embeddings + ncols - 1) // ncols  # Ceiling division
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes[:No_embeddings]):
        ax.hist(axo_embed.test_bkg_embedding_vicreg[:, i], bins=100, alpha=0.8, 
                label="VICReg Embedding", color='C0', edgecolor='black', linewidth=0.3)
        ax.hist(axo_embed.test_bkg_embedding_vae_reco[:, i], bins=100, alpha=0.8, 
                label="VAE Reconstruction", color='C1', edgecolor='black', linewidth=0.3)
        
        ax.set_title(f"Feature {i+1}", fontsize=11)
        ax.set_yscale("log")
        ax.set_xlabel("Value", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.legend(fontsize=9, loc='upper right', frameon=True)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    
    # Remove unused subplots if any
    for j in range(No_embeddings, len(axes)):
        fig.delaxes(axes[j])
    
    fig.suptitle("Distribution of Embedding Features and Reconstructions", fontsize=16, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode("utf-8")
    return img_str


def plot_signal_pu_histograms(dist_plots):
    signals = list(dist_plots.pu_hist.keys())
    num_signals = len(signals)
    num_cols = 3
    num_rows = (num_signals + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 24))
    axes = axes.flatten()

    for i, signal in enumerate(signals):
        ax = axes[i]
        H, score_edges, pu_edges = dist_plots.pu_hist[signal]

        # H.shape = (len(score_edges)-1, len(pu_edges)-1)
        # We'll place PU on the x-axis, Score on the y-axis.
        # extent = [x_min, x_max, y_min, y_max]
        extent = [
            pu_edges[0],
            pu_edges[-1],
            score_edges[0],
            score_edges[-1],
        ]

        # imshow wants the array with shape (Ny, Nx). 
        # By default, H is (score_bins, pu_bins). That puts "score" in the first axis.
        # This lines up with the 'extent' so that x=pu_edges, y=score_edges.
        img = ax.imshow(
            H, 
            origin="lower",
            extent=extent,
            aspect="auto",
            interpolation="nearest"
        )

        fig.colorbar(img, ax=ax)

        ax.set_xlabel("Pile-up")
        ax.set_ylabel("Score")
        ax.set_yscale("log")  # optional; caution: imshow doesn't truly transform the axis scale
        ax.set_title(signal)
        ax.grid(False)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(pad=2.0)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode("utf-8")

    return img_str

def plot_signal_ht_histograms(dist_plots):
    signals = list(dist_plots.ht_hist.keys())
    num_signals = len(signals)
    num_cols = 3
    num_rows = (num_signals + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 24))
    axes = axes.flatten()

    for i, signal in enumerate(signals):
        ax = axes[i]
        H, score_edges, ht_edges = dist_plots.ht_hist[signal]
        # H is shape (len(score_edges)-1, len(ht_edges)-1)

        extent = [
            ht_edges[0],
            ht_edges[-1],
            score_edges[0],
            score_edges[-1],
        ]

        img = ax.imshow(
            H,
            origin="lower",
            extent=extent,
            aspect="auto",
            interpolation="nearest"
        )
        fig.colorbar(img, ax=ax)

        ax.set_xlabel("H$_T$")
        ax.set_ylabel("Score")
        ax.set_yscale("log")  # same note about "true" log scale
        ax.set_title(signal)
        ax.grid(False)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(pad=2.0)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode("utf-8")

    return img_str

def plot_object_pt_histograms(dist_plots, object_type):
    signals = list(dist_plots.object_pt_hist.keys())
    num_signals = len(signals)
    num_cols = 3
    num_rows = (num_signals + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 24))
    axes = axes.flatten()

    for i, signal in enumerate(signals):
        ax = axes[i]
        if object_type not in dist_plots.object_pt_hist[signal]:
            # skip missing object type for that signal
            continue

        H, score_edges, pt_edges = dist_plots.object_pt_hist[signal][object_type]
        # H.shape = (len(score_edges)-1, len(pt_edges)-1)

        extent = [
            pt_edges[0],
            pt_edges[-1],
            score_edges[0],
            score_edges[-1],
        ]

        img = ax.imshow(
            H,
            origin="lower",
            extent=extent,
            aspect="auto",
            interpolation="nearest"
        )
        fig.colorbar(img, ax=ax)

        ax.set_xlabel("pT (GeV)")
        ax.set_ylabel("Score")
        ax.set_yscale("log") 
        ax.set_title(f"{signal}")
        ax.grid(False)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(pad=2.0)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode("utf-8")

    return img_str


def plot_pu_vs_rate(all_axo):

    # These are dictionaries keyed by thresholds
    pu_dict = all_axo.pu_values          # dict: { threshold -> 1D array of PU }
    rate_dict = all_axo.total_rates_khz  # dict: { threshold -> 1D array of rates }
    unc_dict = all_axo.rate_uncertainties # dict: { threshold -> 1D array of uncertainties }

    num_thresholds = len(pu_dict)
    num_cols = 3
    num_rows = (num_thresholds + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 24))
    axes = axes.flatten()

    # Each item in pu_dict is (threshold_key, array_of_pileups)
    for i, (thres_key, pu_vals) in enumerate(pu_dict.items()):
        ax = axes[i]

        # Get the corresponding rate values and uncertainties
        r_vals = rate_dict[thres_key]
        unc_vals = unc_dict[thres_key]

        # Plot with error bars
        ax.errorbar(
            pu_vals, r_vals, yerr=unc_vals, 
            fmt='o-', capsize=3, markersize=4, label=f'Rate'
        )

        ax.set_xlabel('Pile-up')
        ax.set_ylabel('Rate (kHz)')
        ax.grid(True)
        ax.legend()
        
        # The title might be the "target rate" or threshold key you used in your code
        ax.set_title(f'Target Rate: {thres_key} kHz')

        # Optionally start y-axis from 0
        ax.set_ylim(bottom=0)

    # Hide any leftover empty subplots if num_thresholds isn't a multiple of num_cols
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(pad=2.0)

    # Convert figure to base64
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode('utf-8')

    return img_str


def plot_scores_zb_vs_sn_2d(all_axo, bins=20, cmap="plasma"):
    """
    Plots a 2D histogram comparing Zero Bias scores vs. Single Neutrino scores.
    Returns a base64 PNG string.
    """
    # 1) Convert to NumPy arrays
    zb_scores = np.array(all_axo.zb_score)
    sn_scores = np.array(all_axo.sn_score)

    # 2) Flatten (ravel) to ensure 1D
    zb_scores = zb_scores.ravel()
    sn_scores = sn_scores.ravel()

    # 3) Match lengths
    n = min(len(zb_scores), len(sn_scores))
    zb_scores = zb_scores[:n]
    sn_scores = sn_scores[:n]

    # Create a figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # 2D histogram
    hist = ax.hist2d(
        zb_scores,
        sn_scores,
        bins=bins,
        cmap=cmap,
        density=True,
        range=[[0, 5], [0, 5]]  # adjust or remove if your scores exceed 0-20
    )

    # Colorbar
    cbar = plt.colorbar(hist[3], ax=ax)
    cbar.set_label("Density")

    ax.set_xlabel("Zero Bias Score")
    ax.set_ylabel("Single Neutrino Score")
    ax.grid(True)

    # Convert to base64
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)

    return img_str


def plot_scores_zb_vs_sn(all_axo, bins=20, score_range=(0, 50)):
    """
    Plots two 1D histograms (Zero Bias vs. Single Neutrino), side by side in the same axes.
    Returns a base64 PNG string.
    """
    # 1) Convert to NumPy arrays
    zb_scores = np.array(all_axo.zb_score)
    sn_scores = np.array(all_axo.sn_score)

    # 2) Flatten to 1D
    zb_scores = zb_scores.ravel()
    sn_scores = sn_scores.ravel()

    # 3) Match lengths
    n = min(len(zb_scores), len(sn_scores))
    zb_scores = zb_scores[:n]
    sn_scores = sn_scores[:n]

    # Create a figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # 1D histograms
    ax.hist(
        zb_scores,
        bins=bins,
        range=score_range,
        histtype='step',
        label='Zero Bias',
        density=True,
        linewidth=2
    )
    ax.hist(
        sn_scores,
        bins=bins,
        range=score_range,
        histtype='step',
        label='Single Neutrino',
        density=True,
        linewidth=2
    )
    ax.set_xlabel("Score")
    ax.set_ylabel("Normalized Counts")
    ax.set_yscale('log')
    ax.grid(True)
    ax.legend()


    # Convert to base64
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)

    return img_str


def plot_data_distribution(axo_man):
    fig, axes = plt.subplots(nrows=19, ncols=3, figsize=(30, 50))
    fig.suptitle("Histograms of Train and Test Data", fontsize=16)
    
    Objects = ["MET"]+["EGAMMA"]*4+["MUON"]*4+["JET"]*10
    Feature = [r"$p_T$",r"$\eta$",r"$\phi$"]
    for i in range(19):
        for j in range(3):
            ax = axes[i, j]
            stat = ax.hist(axo_man.data_file["Background_data"]["Train"]["DATA"][:, i, j], bins=100, label="Train", color='blue', density = True)
            ax.hist(axo_man.data_file["Background_data"]["Test"]["DATA"][:, i, j], bins=stat[1], label="Test",histtype ="step",color='green', density = True)
            ax.set_title(f"{Objects[i]} {Feature[j]}")
            ax.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)
    
    html_return = f"""
    <div>
        <h1>ZeroBias Data Distribution</h1>
        <img src="data:image/png;base64,{img_str}" alt="Data_distribution">
    </div>
    """
    
    
    for s in axo_man.data_file["Signal_data"].keys():
        
        fig, axes = plt.subplots(nrows=19, ncols=3, figsize=(30, 50))
        fig.suptitle(f"Histograms of Test Data and MC {s} Data", fontsize=16)
        
        Objects = ["MET"]+["EGAMMA"]*4+["MUON"]*4+["JET"]*10
        Feature = [r"$p_T$",r"$\eta$",r"$\phi$"]
        for i in range(19):
            for j in range(3):
                ax = axes[i, j]
                stat = ax.hist(axo_man.data_file["Background_data"]["Test"]["DATA"][:, i, j], bins=100, label="ZB Test", color='blue', density = True)
                ax.hist(axo_man.data_file["Signal_data"][s]["DATA"][:, i, j], bins=stat[1], label="MC",histtype ="step",color='green', density = True)
                ax.set_title(f"{s} {Objects[i]} {Feature[j]}")
                ax.legend()
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close(fig)
    
        html_return += f"""
    <div>
        <h1>{s} Data Distribution</h1>
        <img src="data:image/png;base64,{img_str}" alt="Data_distribution">
    </div>
    """
    return html_return

def generate_axolotl_html_report(config, all_axo, axo_embed, dist_plots, dict_axo, histogram_dict, threshold_dict, history_dict, raw_wrt_pure, output_file):

    FIGSIZE_HIST = (36, 18)
    FIGSIZE_LINE = (30, 20)
    FIGSIZE_HISTORY = (30, 24)
    SUPTITLE_SIZE = 20
    AXIS_TITLE_SIZE = 12
    LABEL_SIZE = 10
    TICK_SIZE = 10
    LINEWIDTH = 2
    AXVLINE_WIDTH = 2
    
    plt.style.use('default')
    # Sort the dict_axo keys numerically
    sorted_dict_axo = dict(sorted(dict_axo.items(), key=lambda item: float(item[0])))

    # Generate HTML for configuration sections
    data_config_html = dict_to_html_table(config['data_config'])
    data_config_rem = config['data_config'].copy()
    data_config_rem.pop("Read_configs")
    data_config_html_read_bkg_html = dict_to_html_table(config['data_config']["Read_configs"]["BACKGROUND"])
    data_config_html_read_sig_html = dict_to_html_table(config['data_config']["Read_configs"]["SIGNAL"])
    data_config_html_read_rem_html = dict_to_html_table(data_config_rem)
    train_config_html = dict_to_html_table(config['train'])
    determinism_config_html = dict_to_html_table(config['determinism'])
    model_config_html = dict_to_html_table(config['model'])
    threshold_config_html = dict_to_html_table(config['threshold'])
    store_config_html = dict_to_html_table(config['store'])
    lr_schedule_config = dict_to_html_table(config["lr_schedule"])
    train_recipe = config["train_recipe"]

    # Unpack threshold_dict (raw, pure)
    threshold_dict_raw, threshold_dict_pure = threshold_dict

    # Create separate tables for raw and pure thresholds
    threshold_config_html_raw = dict_to_html_table(threshold_dict_raw)
    threshold_config_html_pure = dict_to_html_table(threshold_dict_pure)

    raw_wrt_pure_html = dict_to_html_table(raw_wrt_pure)

    # Build HTML header and configuration tables
    html_output = f"""
    <html>
    <head>
        <title>AXOLOTL1 Configuration</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                line-height: 1.6;
            }}
            h1, h2, h3 {{
                color: #333;
                margin-top: 20px;
            }}
            table {{
                width: 100%;
                margin-bottom: 20px;
                border: 1px solid #ccc;
                border-collapse: collapse;
                page-break-inside: auto;
            }}
            tr {{
                page-break-inside: avoid;
                page-break-after: auto;
            }}
            th, td {{
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            td {{
                vertical-align: top;
            }}
            img {{
                max-width: 100%;
                height: auto;
            }}
            @page {{
                size: A4 landscape;
                margin: 1cm;
            }}
            .page-break {{
                page-break-before: always;
            }}
        </style>
    </head>
    <body>
        <div class="page-break">
            <h1>Configuration for AXOLOTL1</h1>
            <h2>Read Configuration - ZEROBIAS</h2>
            {data_config_html_read_bkg_html}
        </div>
        <div class="page-break">
            <h2>Read Configuration - MonteCarlo</h2>
            {data_config_html_read_sig_html}
        </div>
        <div class="page-break">
            <h2>Preprocessing Configs</h2>
            {data_config_html_read_rem_html}
        </div>
        <div class="page-break">
            <h2>Training Configuration</h2>
            {train_config_html}
        </div>
        <div class="page-break">
            <h2>Determinism Configuration</h2>
            {determinism_config_html}
        </div>
        <div class="page-break">
            <h2>Model Configuration</h2>
            {model_config_html}
        </div>
        <div class="page-break">
            <h2>Learning Rate Scheduler Configuration</h2>
            {lr_schedule_config}
        </div>
        <div class="page-break">
            <h2>Threshold Configuration (from config)</h2>
            {threshold_config_html}
        </div>
        <div class="page-break">
            <h2>Thresholds</h2>
            <h3>Raw Thresholds</h3>
            {threshold_config_html_raw}
            <br/>
            <h3>Pure Thresholds</h3>
            {threshold_config_html_pure}
            <br/>
            <h3>Raw Rates (right) for pure rates left</h3>
            {raw_wrt_pure_html}
            <br/>
        </div>
        <div class="page-break">
            <h2>Store Configuration</h2>
            {store_config_html}
        </div>
    """

    # Loop through each threshold DataFrame and add it to the HTML
    for threshold, df in sorted_dict_axo.items():
        html_output += f"""
        <div class="page-break">
            <h1>AXO Score Table for Threshold: {threshold} kHz</h1>
            {df.to_html(index=False, escape=False)}
        </div>
        """

    # ------------------------ Figure 1a: Signal Histograms ------------------------
    linestyles = ['-', '--', '-.', ':', (0,(3,1,1,1)), (0,(3,5,1,5,1,5))]
    
    # Put background first; then signals
    all_signals = ["background"] + [s for s in histogram_dict if s != "background"]
    num_signals = len(all_signals)
    num_cols = 3
    num_rows = math.ceil(num_signals / num_cols)
    
    fig_width = 7 * num_cols
    fig_height = 5 * num_rows
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))
    axes = axes.flatten()
    
    for idx, signal_name in enumerate(all_signals):
        ax = axes[idx]
        
        # Title
        if signal_name == "background":
            ax.set_title("ZeroBias (Background)", fontsize=AXIS_TITLE_SIZE)
        else:
            ax.set_title(signal_name, fontsize=AXIS_TITLE_SIZE)
        
        # Plot histogram
        color = "blue" if signal_name == "background" else None
        hep.histplot(histogram_dict[signal_name], ax=ax, color=color)
        ax.set_yscale("log")
        ax.grid(True)
        
        # Lists for the two legends
        lines_raw, labels_raw = [], []
        lines_pure, labels_pure = [], []
        
        # --- PURE lines in black ---
        sorted_pure = sorted(threshold_dict_pure.items(), key=lambda x: float(x[0]))
        for i, (thres, value) in enumerate(sorted_pure):
            style = linestyles[i % len(linestyles)]
            line = ax.axvline(x=value, color='black', linestyle=style, linewidth=AXVLINE_WIDTH)
            lines_pure.append(line)
            labels_pure.append(f"{thres} kHz Pure")
        
        # --- RAW lines in red ---
        # Offset slightly (e.g. +5) so they don't overlap
        sorted_raw = sorted(threshold_dict_raw.items(), key=lambda x: float(x[0]))
        for i, (thres, value) in enumerate(sorted_raw):
            style = linestyles[i % len(linestyles)]
            line = ax.axvline(x=value + 5, color='red', linestyle=style, linewidth=AXVLINE_WIDTH)
            lines_raw.append(line)
            labels_raw.append(f"{thres} kHz Raw")
        
        # Two separate legends
        leg_raw = ax.legend(
            lines_raw, 
            labels_raw, 
            loc='upper right', 
            bbox_to_anchor=(1.0, 1.0),
            fontsize=TICK_SIZE, 
            title='Raw'
        )
        ax.add_artist(leg_raw)
        
        leg_pure = ax.legend(
            lines_pure, 
            labels_pure, 
            loc='upper right', 
            bbox_to_anchor=(0.75, 1.0),
            fontsize=TICK_SIZE, 
            title='Pure'
        )
        ax.add_artist(leg_pure)
    # Remove any unused axes
    for ax in axes[len(all_signals):]:
        fig.delaxes(ax)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode('utf-8')
    
    html_output += f"""
    <div class="page-break">
        <h1>Signal Histograms</h1>
        <img src="data:image/png;base64,{img_str}" alt="Signal Histograms">
    </div>
    """

    # ------------------------ Figure 1b: Signal Histograms in log scale ------------------------
    linestyles = ['-', '--', '-.', ':', (0,(3,1,1,1)), (0,(3,5,1,5,1,5))]
    
    # Put background first; then signals
    all_signals = ["background"] + [s for s in histogram_dict if s != "background"]
    num_signals = len(all_signals)
    num_cols = 3
    num_rows = math.ceil(num_signals / num_cols)
    
    fig_width = 7 * num_cols
    fig_height = 5 * num_rows
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))
    axes = axes.flatten()
    
    for idx, signal_name in enumerate(all_signals):
        ax = axes[idx]
        
        # Title
        if signal_name == "background":
            ax.set_title("ZeroBias (Background)", fontsize=AXIS_TITLE_SIZE)
        else:
            ax.set_title(signal_name, fontsize=AXIS_TITLE_SIZE)
        
        color = "blue" if signal_name == "background" else None
        hep.histplot(histogram_dict[signal_name], ax=ax, color=color)
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.grid(True)
        
        # Lists for the two legends
        lines_raw, labels_raw = [], []
        lines_pure, labels_pure = [], []
        
        # --- PURE lines in black ---
        sorted_pure = sorted(threshold_dict_pure.items(), key=lambda x: float(x[0]))
        for i, (thres, value) in enumerate(sorted_pure):
            style = linestyles[i % len(linestyles)]
            line = ax.axvline(x=value, color='black', linestyle=style, linewidth=AXVLINE_WIDTH)
            lines_pure.append(line)
            labels_pure.append(f"{thres} kHz Pure")
        
        # --- RAW lines in red ---
        sorted_raw = sorted(threshold_dict_raw.items(), key=lambda x: float(x[0]))
        for i, (thres, value) in enumerate(sorted_raw):
            style = linestyles[i % len(linestyles)]
            line = ax.axvline(x=value + 5, color='red', linestyle=style, linewidth=AXVLINE_WIDTH)
            lines_raw.append(line)
            labels_raw.append(f"{thres} kHz Raw")
        
        leg_raw = ax.legend(
            lines_raw, 
            labels_raw, 
            loc='upper right', 
            bbox_to_anchor=(1.0, 1.0),
            fontsize=TICK_SIZE,
            title='Raw'
        )
        ax.add_artist(leg_raw)
    
        leg_pure = ax.legend(
            lines_pure, 
            labels_pure, 
            loc='upper right', 
            bbox_to_anchor=(0.75, 1.0),
            fontsize=TICK_SIZE,
            title='Pure'
        )
        ax.add_artist(leg_pure)
    
    for ax in axes[len(all_signals):]:
        fig.delaxes(ax)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode('utf-8')
    
    html_output += f"""
    <div class="page-break">
        <h1>Signal Histograms Log Scale</h1>
        <img src="data:image/png;base64,{img_str}" alt="Signal Histograms">
    </div>
    """

    # ------------------------ Figure 2: AXO Scores vs. Thresholds ------------------------
    for eff_type in ["raw-raw", "raw-pure", "pure-raw", "pure-pure"]:
        sorted_thresholds = sorted(dict_axo.keys(), key=lambda x: float(x))
        signal_names = dict_axo[sorted_thresholds[0]]['Signal Name'].tolist()
        num_signals = len(signal_names)
        grid_size = math.ceil(math.sqrt(num_signals))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=FIGSIZE_LINE)
        axes = axes.flatten()

        for i, signal_name in enumerate(signal_names):
            ax = axes[i]
            x_data = [float(threshold) for threshold in sorted_thresholds]
            y_data = [
                dict_axo[threshold].loc[dict_axo[threshold]['Signal Name'] == signal_name, f'AXO {eff_type}'].values[0]
                for threshold in sorted_thresholds
            ]
            ax.plot(x_data, y_data, marker='o', label=signal_name, linewidth=LINEWIDTH)
            ax.set_title(signal_name, fontsize=AXIS_TITLE_SIZE)
            ax.set_xlabel('Threshold (kHz)', fontsize=LABEL_SIZE)
            ax.set_ylabel('AXO SCORE', fontsize=LABEL_SIZE)
            ax.set_yscale('log')
            ax.grid(True)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        fig.suptitle('AXO for different Signals', fontsize=SUPTITLE_SIZE)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode('utf-8')
        html_output += f"""
        <div class="page-break">
            <h1>{eff_type} Efficiencies for different rates</h1>
            <img src="data:image/png;base64,{img_str}" alt="{eff_type} Efficiencies Signals">
        </div>
        """

    # ------------------------ Figure 3: Training History ------------------------
    train_history = history_dict.copy()
    train_history.pop("pure-pure", None)
    train_history.pop("raw-pure", None)
    num_history_items = len(train_history)
    grid_size = math.ceil(math.sqrt(num_history_items))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=FIGSIZE_HISTORY)
    axes = axes.flatten()

    for i, (key, values) in enumerate(train_history.items()):
        ax = axes[i]
        ax.plot(range(1, len(values) + 1), values, label=key, linewidth=LINEWIDTH)
        ax.set_title(key, fontsize=AXIS_TITLE_SIZE)
        ax.set_xlabel('Epoch', fontsize=LABEL_SIZE)
        ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
        ax.grid(True, linestyle='--', linewidth=0.7)
        ax.xaxis.set_label_coords(0.5, -0.1)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle('History for AXO Training', fontsize=SUPTITLE_SIZE)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode('utf-8')
    html_output += f"""
    <div style="page-break-inside: avoid;">
        <img src="data:image/png;base64,{img_str}" alt="Training History">
    </div>
    """

    # ------------------------ Figure 4: Efficiency Evolution ------------------------
    for eff_type in ["raw-pure", "pure-pure"]:
        eff_history_dict = history_dict[eff_type]
        num_history_items = len(eff_history_dict)
        grid_size = math.ceil(math.sqrt(num_history_items))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=FIGSIZE_HISTORY)
        axes = axes.flatten()

        for i, (key, values) in enumerate(eff_history_dict.items()):
            ax = axes[i]
            ax.plot(range(1, len(values) + 1), values, label=key, linewidth=LINEWIDTH)
            ax.set_title(key, fontsize=AXIS_TITLE_SIZE)
            ax.set_xlabel('Epoch', fontsize=LABEL_SIZE)
            ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
            ax.grid(True, linestyle='--', linewidth=0.7)
            ax.xaxis.set_label_coords(0.5, -0.1)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        fig.suptitle(f'Efficiency ({eff_type}) History for AXO Training', fontsize=SUPTITLE_SIZE)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode('utf-8')
        html_output += f"""
        <div style="page-break-inside: avoid;">
            <img src="data:image/png;base64,{img_str}" alt="Efficiency ({eff_type}) History for AXO Training">
        </div>
        """

    html_output += plot_data_distribution(all_axo)

    html_output += f"""
    <div>
        <h1>VICReg Embedding PCA</h1>
        <img src="data:image/png;base64,{plot_embedding_pca(axo_embed = axo_embed,axo_man=all_axo)}" alt="Embedding">
    </div>
    """
    
    pu_vs_rate_img_str = plot_pu_vs_rate(all_axo)

    # Embed the PU vs Rate plot in the HTML
    html_output += f"""
    <div>
        <h1>Single Neutrino nPV vs Rate</h1>
        <img src="data:image/png;base64,{pu_vs_rate_img_str}" alt="PU vs Rate">
    </div>
    """

    scores_pu_img_str = plot_scores_zb_vs_sn_2d(all_axo)

    # Embed the PU vs Rate plot in the HTML
    html_output += f"""
    <div>
        <h1>Single Neutrino vs ZB scores</h1>
        <img src="data:image/png;base64,{scores_pu_img_str}" alt="PU vs Rate">
    </div>
    """

    scores_pu_img_str = plot_scores_zb_vs_sn(all_axo)

    # Embed the PU vs Rate plot in the HTML
    html_output += f"""
    <div>
        <h1>Single Neutrino vs ZB scores</h1>
        <img src="data:image/png;base64,{scores_pu_img_str}" alt="PU vs Rate">
    </div>
    """

    scores_pu_img_str = plot_signal_pu_histograms(dist_plots)

    # Embed the PU vs Rate plot in the HTML
    html_output += f"""
    <div>
        <h1>Score vs PU</h1>
        <img src="data:image/png;base64,{scores_pu_img_str}" alt="PU vs Rate">
    </div>
    """

    scores_pu_img_str = plot_signal_ht_histograms(dist_plots)

    # Embed the PU vs Rate plot in the HTML
    html_output += f"""
    <div>
        <h1>Score vs HT</h1>
        <img src="data:image/png;base64,{scores_pu_img_str}" alt="PU vs Rate">
    </div>
    """

    
    objects = [
        ("eg_leading", "Score vs egamma leading pT"),
        ("eg_subleading", "Score vs egamma subleading pT"),
        ("mu_leading", "Score vs muon leading pT"),
        ("mu_subleading", "Score vs muon subleading pT"),
        ("jet_leading", "Score vs jet leading pT"),
        ("jet_subleading", "Score vs jet subleading pT")
    ]

    for obj, title in objects:
        scores_pu_img_str = plot_object_pt_histograms(dist_plots, obj)

        html_output += f"""
        <div>
            <h1>{title}</h1>
            <img src="data:image/png;base64,{scores_pu_img_str}" alt="{title}">
        </div>
        """

    html_output += f"""
    <div>
        <h1>VAE Reconstruction Histogram</h1>
        <img src="data:image/png;base64,{plot_vae_reconstruction(axo_embed = axo_embed,axo_man=all_axo)}" alt="Embedding">
    </div>
    """

    html_output += "</body></html>"

    if output_file is not None:
        with open(output_file, "w") as file:
            file.write(html_output)
        print(f"HTML file generated: {output_file}")