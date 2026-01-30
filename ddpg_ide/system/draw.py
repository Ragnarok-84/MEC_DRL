import numpy as np
import glob, os
import matplotlib.pyplot as plt
import pandas as pd



def read_all_runs(dir_path):
    files = sorted(glob.glob(os.path.join(dir_path, "*.npz")))
    all_rs, all_ps, all_bs = [], [], []

    print(f"Found {len(files)} files in {dir_path}")

    for f in files:
        data = np.load(f)
        
        
        if 'rewards' in data:
            res_r = data['rewards']
            res_p = data['powers']
            res_b = data['buffers'] 
        else:
            
            res_r = data["arr_0"]
            res_p = data["arr_1"]
            res_b = data["arr_2"]

        all_rs.append(res_r)
        all_ps.append(res_p)
        all_bs.append(res_b)

    return np.array(all_rs), np.array(all_ps), np.array(all_bs)



def moving_average(data, window):
    return np.convolve(data, np.ones(window), 'valid') / window


def plot_stats_table(data_2d, FIG_DIR,  label):
    
    means = np.mean(data_2d, axis=1)
    overall_mean = np.mean(means)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis('off')

    table_data = [[f"{m:.3f}"] for m in means] + [[f"{overall_mean:.3f}"]]
    row_labels = [f"Run {i+1}" for i in range(len(means))] + ["Average"]

    ax.table(
        cellText=table_data,
        rowLabels=row_labels,
        colLabels=[f"Mean {label}"],
        loc='center',
        cellLoc='center'
    )

    plt.title(f"Statistics of {label}")
    plt.tight_layout()

    
    safe_label = label.replace(" ", "_").replace("(", "").replace(")", "")
    table_path = os.path.join(FIG_DIR, f"table_{safe_label}.png")
    plt.savefig(table_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved table: {table_path}")
    print(f"Aggregate {label}: {overall_mean:.3f}")


def plot_all_users_curve(all_rs, all_ps, all_bs, FIG_DIR, win=10):

   
    rs_run_ep = np.mean(all_rs, axis=2)
    ps_run_ep = np.mean(all_ps, axis=2)
    bs_run_ep = np.mean(all_bs, axis=2)

    data_sets = {
        "Reward": (rs_run_ep, "Average Reward"),
        "Power (W)": (ps_run_ep, "Average Power"),
        "Delay (ms)": (bs_run_ep, "Average Delay"),
    }

    for label, (data_2d, ylabel) in data_sets.items():

        mean_curve = np.mean(data_2d, axis=0)
        sm_curve = moving_average(mean_curve, win)

        plt.figure(figsize=(10, 5))
        plt.plot(mean_curve, alpha=0.4, label="Mean")
        plt.plot(
            range(win - 1, len(mean_curve)),
            sm_curve,
            linewidth=2,
            label=f"Smoothed (win={win})"
        )

        plt.xlabel("Episode")
        plt.ylabel(ylabel)
        plt.title(f"{label} over Episodes (All Runs & Users)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # ==== SAVE CURVE FIGURE ====
        safe_label = label.replace(" ", "_").replace("(", "").replace(")", "")
        curve_path = os.path.join(FIG_DIR, f"curve_{safe_label}.png")
        plt.savefig(curve_path, dpi=300, bbox_inches="tight")
        plt.show()

        print(f"Saved curve: {curve_path}")

        # ==== TABLE SEPARATE ====
        plot_stats_table(data_2d, FIG_DIR, label)
        


def plot_all_runs_together(all_rs, all_ps, all_bs, FIG_DIR, win=10):
   
  
    rs_run_ep = np.mean(all_rs, axis=2) 
    ps_run_ep = np.mean(all_ps, axis=2)
    bs_run_ep = np.mean(all_bs, axis=2)

    data_sets = {
        "Reward": (rs_run_ep, "Average Reward"),
        "Power_W": (ps_run_ep, "Average Power (W)"),
        "Delay_ms": (bs_run_ep, "Average Delay (ms)"),
    }

    num_runs = rs_run_ep.shape[0]
    colors = plt.cm.tab10(np.linspace(0, 1, num_runs))

    for label, (data_2d, ylabel) in data_sets.items():
        
      
        plt.figure(figsize=(12, 6))
        for run_idx in range(num_runs):
            plt.plot(data_2d[run_idx], alpha=0.6, label=f"Run {run_idx+1}", color=colors[run_idx])
        
        
        mean_all = np.mean(data_2d, axis=0)
        plt.plot(mean_all, 'k--', linewidth=2.5, label="Mean (All Runs)")
        
        plt.xlabel("Episode", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(f"All Runs - {label} (Raw)", fontsize=14)
        plt.legend(loc='best', fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        fig_path = os.path.join(FIG_DIR, f"all_runs_{label}_raw.png")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved: {fig_path}")

        
        plt.figure(figsize=(12, 6))
        for run_idx in range(num_runs):
            sm_curve = moving_average(data_2d[run_idx], win)
            plt.plot(
                range(win - 1, len(data_2d[run_idx])),
                sm_curve,
                alpha=0.7,
                linewidth=1.5,
                label=f"Run {run_idx+1}",
                color=colors[run_idx]
            )
        
        
        sm_mean = moving_average(mean_all, win)
        plt.plot(
            range(win - 1, len(mean_all)),
            sm_mean,
            'k-',
            linewidth=3,
            label="Mean (All Runs)"
        )
        
        plt.xlabel("Episode", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(f"All Runs - {label} (Smoothed, win={win})", fontsize=14)
        plt.legend(loc='best', fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        fig_path = os.path.join(FIG_DIR, f"all_runs_{label}_smoothed.png")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved: {fig_path}")


def plot_individual_run(run_idx, run_data_r, run_data_p, run_data_b, FIG_DIR, win=10):
    
   
    r_mean = np.mean(run_data_r, axis=1)
    p_mean = np.mean(run_data_p, axis=1)
    b_mean = np.mean(run_data_b, axis=1)

    data_sets = {
        "Reward": (r_mean, "Average Reward"),
        "Power_W": (p_mean, "Average Power (W)"),
        "Delay_ms": (b_mean, "Average Delay (ms)"),
    }

   
    run_dir = os.path.join(FIG_DIR, f"run_{run_idx+1}")
    os.makedirs(run_dir, exist_ok=True)

    for label, (data_1d, ylabel) in data_sets.items():
        sm_curve = moving_average(data_1d, win)

        plt.figure(figsize=(10, 5))
        plt.plot(data_1d, alpha=0.4, label="Raw")
        plt.plot(
            range(win - 1, len(data_1d)),
            sm_curve,
            linewidth=2,
            label=f"Smoothed (win={win})"
        )

        plt.xlabel("Episode")
        plt.ylabel(ylabel)
        plt.title(f"Run {run_idx+1} - {label}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save
        fig_path = os.path.join(run_dir, f"{label}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()

    print(f"✓ Saved Run {run_idx+1} figures in: {run_dir}")


def plot_all_runs_separately(all_rs, all_ps, all_bs, FIG_DIR, win=10):
    
    num_runs = all_rs.shape[0]
    
    for run_idx in range(num_runs):
        plot_individual_run(
            run_idx,
            all_rs[run_idx],
            all_ps[run_idx],
            all_bs[run_idx],
            FIG_DIR,
            win
        )

def summarize_per_user(all_rs, all_ps, all_bs):
    

    num_runs, num_episodes, num_users = all_rs.shape

    
    avg_r = np.mean(all_rs, axis=(0, 1))
    avg_p = np.mean(all_ps, axis=(0, 1))
    avg_b = np.mean(all_bs, axis=(0, 1))

    df = pd.DataFrame({
        "User": [f"User {i+1}" for i in range(num_users)],
        "Reward (avg)": avg_r,
        "Power (avg, W)": avg_p,
        "Delay (avg, ms)": avg_b
    })

   
    df.loc[len(df)] = [
        "Average (all users)",
        np.mean(avg_r),
        np.mean(avg_p),
        np.mean(avg_b)
    ]

    return df


def summarize_per_user(all_rs, all_ps, all_bs):
    

    num_runs, num_episodes, num_users = all_rs.shape

    
    avg_r = np.mean(all_rs, axis=(0, 1))
    avg_p = np.mean(all_ps, axis=(0, 1))
    avg_b = np.mean(all_bs, axis=(0, 1))

    df = pd.DataFrame({
        "User": [f"User {i+1}" for i in range(num_users)],
        "Reward (avg)": avg_r,
        "Power (avg, W)": avg_p,
        "Delay (avg, ms)": avg_b
    })

   
    df.loc[len(df)] = [
        "Average (all users)",
        np.mean(avg_r),
        np.mean(avg_p),
        np.mean(avg_b)
    ]

    return df


def plot_training_curves_multi_user(all_rs, FIG_DIR, arrival_rates=None, win=10):
    
    num_runs, num_episodes, num_users = all_rs.shape
    
    
    rewards_per_episode = np.mean(all_rs, axis=0)
    
    colors = plt.cm.tab10(np.linspace(0, 1, num_users))
  
    plt.figure(figsize=(10, 6))
    
    for user_idx in range(num_users):
        if arrival_rates is not None:
            label = f"λ = {arrival_rates[user_idx]:.1f} Mbps"
        else:
            label = f"User {user_idx + 1}"
        
        plt.plot(
            rewards_per_episode[:, user_idx],
            alpha=0.6,
            linewidth=1.5,
            color=colors[user_idx],
            label=label
        )
    
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Average Reward", fontsize=12)
    plt.title("Training Process - Multi-User (Raw)", fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    
 
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.2)
    
    plt.tight_layout()
    
    fig_path = os.path.join(FIG_DIR, "training_multi_user_raw.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {fig_path}")
    
    
    plt.figure(figsize=(10, 6))
    
    for user_idx in range(num_users):
        if arrival_rates is not None:
            label = f"λ = {arrival_rates[user_idx]:.1f} Mbps"
        else:
            label = f"User {user_idx + 1}"
        
      
        sm_curve = moving_average(rewards_per_episode[:, user_idx], win)
        
        plt.plot(
            range(win - 1, num_episodes),
            sm_curve,
            linewidth=2,
            color=colors[user_idx],
            label=label
        )
    
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Average Reward", fontsize=12)
    plt.title(f"Training Process - Multi-User (Smoothed, win={win})", fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.2)
    
    plt.tight_layout()
    
    fig_path = os.path.join(FIG_DIR, "training_multi_user_smoothed.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {fig_path}")
    
   
    plt.figure(figsize=(12, 6))
    
    for user_idx in range(num_users):
        if arrival_rates is not None:
            label = f"λ = {arrival_rates[user_idx]:.1f} Mbps"
        else:
            label = f"User {user_idx + 1}"
        
       
        plt.plot(
            rewards_per_episode[:, user_idx],
            alpha=0.3,
            linewidth=1,
            color=colors[user_idx]
        )
        
       
        sm_curve = moving_average(rewards_per_episode[:, user_idx], win)
        plt.plot(
            range(win - 1, num_episodes),
            sm_curve,
            linewidth=2.5,
            color=colors[user_idx],
            label=label
        )
    
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Average Reward", fontsize=12)
    plt.title(f"Training Process - Multi-User (win={win})", fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.2)
    
    plt.tight_layout()
    
    fig_path = os.path.join(FIG_DIR, "training_multi_user_combined.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {fig_path}")


def plot_training_curves_per_run(all_rs, FIG_DIR, arrival_rates=None, win=10):
    
    num_runs, num_episodes, num_users = all_rs.shape
    colors = plt.cm.tab10(np.linspace(0, 1, num_users))
    
    for run_idx in range(num_runs):
        
        run_dir = os.path.join(FIG_DIR, f"training_run_{run_idx+1}")
        os.makedirs(run_dir, exist_ok=True)
        
        
        rewards_this_run = all_rs[run_idx]
        
        
        plt.figure(figsize=(10, 6))
        
        for user_idx in range(num_users):
            if arrival_rates is not None:
                label = f"λ = {arrival_rates[user_idx]:.1f} Mbps"
            else:
                label = f"User {user_idx + 1}"
            
            plt.plot(
                rewards_this_run[:, user_idx],
                alpha=0.6,
                linewidth=1.5,
                color=colors[user_idx],
                label=label
            )
        
        plt.xlabel("Episode", fontsize=12)
        plt.ylabel("Average Reward", fontsize=12)
        plt.title(f"Run {run_idx+1} - Training Process (Raw)", fontsize=14)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.2)
        
        plt.tight_layout()
        
        fig_path = os.path.join(run_dir, "raw.png")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        plt.figure(figsize=(10, 6))
        
        for user_idx in range(num_users):
            if arrival_rates is not None:
                label = f"λ = {arrival_rates[user_idx]:.1f} Mbps"
            else:
                label = f"User {user_idx + 1}"
            
            sm_curve = moving_average(rewards_this_run[:, user_idx], win)
            
            plt.plot(
                range(win - 1, num_episodes),
                sm_curve,
                linewidth=2,
                color=colors[user_idx],
                label=label
            )
        
        plt.xlabel("Episode", fontsize=12)
        plt.ylabel("Average Reward", fontsize=12)
        plt.title(f"Run {run_idx+1} - Training Process (Smoothed, win={win})", fontsize=14)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.2)
        
        plt.tight_layout()
        
        fig_path = os.path.join(run_dir, "smoothed.png")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        plt.figure(figsize=(12, 6))
        
        for user_idx in range(num_users):
            if arrival_rates is not None:
                label = f"λ = {arrival_rates[user_idx]:.1f} Mbps"
            else:
                label = f"User {user_idx + 1}"
            
            plt.plot(
                rewards_this_run[:, user_idx],
                alpha=0.3,
                linewidth=1,
                color=colors[user_idx]
            )
            
            
            sm_curve = moving_average(rewards_this_run[:, user_idx], win)
            plt.plot(
                range(win - 1, num_episodes),
                sm_curve,
                linewidth=2.5,
                color=colors[user_idx],
                label=label
            )
        
        plt.xlabel("Episode", fontsize=12)
        plt.ylabel("Average Reward", fontsize=12)
        plt.title(f"Run {run_idx+1} - Training Process (win={win})", fontsize=14)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.2)
        
        plt.tight_layout()
        
        fig_path = os.path.join(run_dir, "combined.png")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"✓ Saved Run {run_idx+1} training curves in: {run_dir}")
