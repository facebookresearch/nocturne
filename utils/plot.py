from matplotlib import pyplot as plt
import numpy as np

def plot_agent_trajectory(agent_df, act_space_dim):
    """Make figure of agent position, speed and action trajectories."""

    # Get accuracy for fig
    nonnan_ids = np.logical_not(
        np.logical_or(
            np.isnan(agent_df.policy_act),
            np.isnan(agent_df.expert_act),
        ),
    )
    acc = (agent_df.policy_act.values[nonnan_ids] == agent_df.expert_act.values[nonnan_ids]).sum() / nonnan_ids.shape[0]

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot expert and agent positions
    axs[0].plot(agent_df.expert_pos_x, agent_df.expert_pos_y, '.-', color='g', label='Expert')
    axs[0].plot(agent_df.policy_pos_x, agent_df.policy_pos_y, '.-', label='Policy')
    axs[0].scatter(agent_df.expert_pos_x.iloc[0], agent_df.expert_pos_y.iloc[0], marker='x', color='darkred', s=110, zorder=5, label=r'$(x_0, y_0)$')
    axs[0].legend(facecolor='white', framealpha=1)
    axs[0].set_xlabel(r'$x$-axis')
    axs[0].set_ylabel(r'$y$-axis')
    axs[0].set_title('Vehicle trajectory')

    axs[1].plot(agent_df['timestep'].values, agent_df.expert_speed, '.-', color='g', label="Expert")
    axs[1].plot(agent_df['timestep'].values, agent_df.policy_speed, '.-', label="Policy")
    axs[1].legend(facecolor='white', framealpha=1)
    axs[1].set_xlabel(r'$t$')
    axs[1].set_ylabel(r'mph')
    axs[1].set_title('Vehicle speed')
    
    markerline, stemlines, baseline = axs[2].stem(agent_df['timestep'].values, np.abs(agent_df.expert_act - agent_df.policy_act), linefmt='grey', bottom=1.1, label="$|a_t^{expert} - a_t^{π}|$")
    plt.setp(markerline, marker='o', markersize=3, color='black')  # Adjust marker size here for axs[0]
    plt.setp(stemlines, linewidth=1)
    axs[2].legend(facecolor='white', framealpha=1)
    axs[2].set_xlabel(r'$t$')
    axs[2].set_ylabel('Joint action index')
    axs[2].set_title(f'Action accuracy: {acc*100} % ($D^A$ = {act_space_dim})')

    # Adding grids with a specific alpha value to both subplots
    axs[0].grid(alpha=0.5)  # Grid for axs[0] with alpha value
    axs[1].grid(alpha=0.5)  # Grid for axs[1] with alpha value

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    plt.show()