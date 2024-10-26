import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import json

def plot_df(df, x, y, group, kind, **kwargs):
    fig, ax = plt.subplots(figsize=(4.5, 2.7))
    if group is not None:
        for v, g in df.groupby(group):
            g.plot(x = x, y = y, kind = kind, ax=ax, label=f"{group} = {v}", marker='o', grid=True, **kwargs)
        ax.legend(["engine 0", "engine 1"])
    else:
        df.plot(x = x, y = y, kind = kind, ax=ax, marker='o', grid=True, **kwargs)
    if "ylabel" not in kwargs:
        ax.set_ylabel(y)
    return fig, ax

def process_result_file(filename):
    # Pre-processing
    outfile = filename.replace("csv", "pdf")
    pdf_pages = PdfPages(outfile)

    df = pd.read_csv(filename)
    df["gpu_memory"] = df["gpu_memory"].apply(json.loads)
    json_df = pd.json_normalize(df['gpu_memory'])
    df = pd.concat([df.drop(columns=['gpu_memory']), json_df], axis=1)

    # For TTFT
    for expr_id, g in df.groupby("expr_id"):
        context_len = list(g["context_len"])[0]
        fig, ax = plot_df(g, x = "request_id", y = "TTFT", group="engine_id", kind = "line", ylabel="TTFT (s)", ylim = (0, None))
        ax.set_title(f"Context len = {context_len}")
        pdf_pages.savefig(fig, bbox_inches="tight")

    # For THP
    for expr_id, g in df.groupby("expr_id"):
        context_len = list(g["context_len"])[0]
        max_throughput = g["throughput"].max()  # Get the maximum throughput value
        ylim_max = max_throughput + 10  # Set the upper limit to max value + 10
        fig, ax = plot_df(g, x = "request_id", y = "throughput", group="engine_id", kind = "line", ylabel="Tokens / sec", ylim = (0, ylim_max))
        ax.set_title(f"Context len = {context_len}")
        pdf_pages.savefig(fig, bbox_inches="tight")

    # For GPU mem util
    gpu_columns = list(filter(lambda s: "gpu" in s, df.columns))
    tmp_df = df[["expr_id"] + gpu_columns].drop_duplicates()
    max_gpu_usage = tmp_df[gpu_columns].max().max() / 1000  # Divide by 1000 to convert to GB
    ylim_max = max_gpu_usage + 10
    fig, ax = plt.subplots(figsize = (4.5, 2.7))
    for col in gpu_columns:
        tmp_df[col] /= 1000
        tmp_df.plot(x = "expr_id", y = col, marker = "o", ylabel = "GPU memory usage (GB)", ax=ax, grid = True, ylim = (0, ylim_max))
    pdf_pages.savefig(fig, bbox_inches="tight")
    pdf_pages.close()


if __name__ == "__main__":
    process_result_file("test_lmcache_local_cpu.csv")
