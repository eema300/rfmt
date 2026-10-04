import matplotlib.pyplot as plt
 
 
def plot_mse_with_best_dict(
    mse_dict,
    xlabel="Epoch",
    title="MSE over training",
    save_path=None,
):
    
    sorted_keys = sorted(mse_dict.keys())
    mse_values = [mse_dict[k][0] for k in sorted_keys]
 
    new_best_keys = [k for k in sorted_keys if mse_dict[k][1]]
    new_best_vals = [mse_dict[k][0] for k in new_best_keys]
 
    overall_best_key = min(mse_dict, key=lambda k: mse_dict[k][0])
    overall_best_val = mse_dict[overall_best_key][0]
 
    fig, ax = plt.subplots(figsize=(9, 5))
 
    ax.plot(sorted_keys, mse_values, color="tab:blue", linewidth=1.2, label="MSE", zorder=1)
 
    ax.scatter(
        new_best_keys,
        new_best_vals,
        color="tab:orange",
        s=30,
        zorder=2,
        label="New best observed",
    )
 
    ax.scatter(
        [overall_best_key],
        [overall_best_val],
        color="tab:red",
        s=100,
        marker="*",
        zorder=3,
        label="Overall best",
    )
    ax.annotate(
        f"best: {overall_best_val:.6f}\n({xlabel.lower()} {overall_best_key})",
        xy=(overall_best_key, overall_best_val),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=9,
        color="tab:red",
        arrowprops=dict(arrowstyle="->", color="tab:red", lw=0.8),
    )
 
    ax.set_xlabel(xlabel)
    ax.set_ylabel("MSE")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
 
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f">> saved plot to {save_path}")
    else:
        plt.show()
 
    return fig, ax
 
 
if __name__ == "__main__":

    # testing (using real mse data)
    interpolate_mse = {
        1: (0.010431766937221957, True),
        2: (0.002623275434872963, True),
        3: (0.0021100820658293946, True),
        4: (0.0018755007165375496, True),
        5: (0.001621783132448812, True),
        6: (0.0015065101517588975, True),
        7: (0.0014590132328667, True),
        8: (0.0013942424225319436, True),
        9: (0.0013284885572015673, True),
    }

    sample_mse = {
        1: (0.0017857218626886606, True),
        2: (0.00097010558238253, True),
        3: (0.0007852850249037147, True),
        4: (0.0006208212580531836, True),
        5: (0.0005418909713625908, True),
        6: (0.0004590752942021936, True),
        7: (0.00042529701022431254, True),
        8: (0.00041915677138604224, True),
        9: (0.0003788627509493381, True)
    }

    p = plot_mse_with_best_dict(mse_dict=sample_mse, xlabel='Batch', title='Training MSE')