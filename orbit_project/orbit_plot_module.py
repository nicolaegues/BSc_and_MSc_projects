import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import animation


current_dir = os.path.dirname(os.path.abspath(__file__)) 

def plot_general_2d_animation(nr_frames, positions_np, body_names,  zoomed_body_index1, zoomed_body_index2, dt, colors = None):

    """
    Function that plots an animated 2D plot of the orbits of multiple bodies over time.
    displays two subplots: one showing the full system and another showing a zoomed-in view of two specific bodies.

    Args: 
    - nr_frames: Number of frames for the animation.
    - positions_np: An array containing the positions (x, y, z) of the bodies over time., ina flattened manner.
    - body_names: A list of names for the bodies.
    - zoomed_body_index1: Index of the first body to zoom in on.
    - zoomed_body_index2: Index of the second body to zoom in on.
    - dt: the time step size used for the integration.
    - colors: List of colors for the bodies -  if None, a default color map is used.

    """

    n_bodies = positions_np.shape[0]

    if n_bodies ==3: 
        colors = ["orange", "red", "green"]

    elif n_bodies != 3 or colors == None:
        colormap = plt.cm.get_cmap("tab10", n_bodies)  
        colors = [colormap(i) for i in range(n_bodies)]
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax1 = ax[0]
    ax2 = ax[1]

    ax1.set_aspect('equal', adjustable='box')     
    ax2.set_aspect('equal', adjustable='box')    

    frame_factor = int(positions_np.shape[1]/nr_frames)


    x_positions = positions_np[:, :, 0]
    y_positions = positions_np[:, :, 1]

    all_x = x_positions.flatten()
    all_y = y_positions.flatten()
    x_min, x_max = all_x.min()*1.2, all_x.max()*1.2
    y_min, y_max = all_y.min()*1.2, all_y.max()*1.2, 
    ax1.set_xlim(x_min, x_max) 
    ax1.set_ylim(y_min, y_max)


    lines = []
    circles = []
    for i in range(n_bodies):
        line, = ax1.plot([], [], color= colors[i], label=body_names[i])
        circle, = ax1.plot([], [], marker='o', markersize=6, color=colors[i])
        lines.append(line)
        circles.append(circle)

    time_text= ax1.text(0.95*x_min, 0.9*y_max, "", fontsize = "small")

    zoomed_lines = []
    zoomed_circles = []
    for i in [zoomed_body_index1, zoomed_body_index2]:
        zoomed_line, = ax2.plot([], [], color = colors[i]) 
        zoomed_circle, = ax2.plot([], [], marker='o', markersize=6, color= colors[i], label = body_names[i])
        zoomed_lines.append(zoomed_line)
        zoomed_circles.append(zoomed_circle)

    ax1.legend(loc = "lower right")


    def animate(frame):

        step = frame * frame_factor

        if dt < 1:
            seconds = frame*frame_factor/dt
        else: 
            seconds = frame*frame_factor*dt

        #hours =seconds/3600
        days = seconds/86400
        years = days/365

        for i in range(n_bodies):
            lines[i].set_data(x_positions[i, :step], y_positions[i, :step])
            circles[i].set_data([x_positions[i, step - 1]], [y_positions[i, step - 1]])

        time_text.set_text(f'Time: {(days):.2f} Days {(years):.2f} Years ')


        ############################################

        x_curr = x_positions[zoomed_body_index1, step]
        y_curr = y_positions[zoomed_body_index1, step]

        ax[1].set_xlim(x_curr - 1e9, x_curr + 1e9) 
        ax[1].set_ylim(y_curr - 1e9, y_curr + 1e9) 

        for i, index in enumerate([zoomed_body_index1, zoomed_body_index2]):
            zoomed_lines[i].set_data(x_positions[index, :step], y_positions[index, :step])
            zoomed_circles[i].set_data([x_positions[index, step - 1]], [y_positions[index, step - 1]])

        return (*lines, *circles, time_text, *zoomed_lines, *zoomed_circles)
    
    anim = animation.FuncAnimation(fig, animate, frames = nr_frames, interval = 1, blit = True )
    anim.save(current_dir + "/my_orbits.gif")

    
def plot_energy(te, ke, pe):
    """
    Plots the total, kinetic, and potential energy of the system. 
    Generates 4 subplots - one for each type of energy, and the last showing all three together.

    Args: 
    - te: the total emergy of the system over time
    - ke: the total kinetic energy of the system over time
    - te: the total potential energy of the system over time

    """
    t = np.arange(len(te))

    fig, axes = plt.subplots(2, 2,figsize = (16, 10))
    axes[0, 0].plot(ke)
    axes[0, 0].set_title(f"Kinetic energy of the system")
    axes[1, 0].plot(pe)
    axes[1, 0].set_title(f"Potential energy of the system")

    axes[0, 1].plot(te)
    axes[0, 1].set_title(f"Total energy of the system")
   
    axes[1, 1].plot(t, te, c = "r", label = "Total Energy")
    axes[1, 1].plot(t, ke, c = "g", label = "Kinetic Energy")
    axes[1, 1].plot(t, pe, c = "b", label = "Potential Energy")
    axes[1, 1].legend()

    for ax in axes: 
        for ax2 in ax: 
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Energy (J)")
    
    plt.savefig(current_dir + "/individual_energies.png")


def plot_errors(te_error, dts, global_errors):
    """
    Function that generate two subplots: one showing the Relative energy conservation error over time for the system, 
    and one showing the Global relative error versus size of time step. 

    Args: 
    - te_error: the local relative total enrgy errors over time
    - dts: the time step sizes used for analysis of hte global error
    - tglobal_errors: the global relative total energy errors
    
    """

    fig, axes = plt.subplots(1, 2, figsize = (14, 4))
    axes[0].plot(te_error)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Relative error (%)")
    axes[0].set_title(f"Relative energy conservation error for the system")

    axes[1].plot(dts, global_errors)
    axes[1].set_xlabel("Time step size (s)")
    axes[1].set_ylabel("Global Relative Error (%)")
    axes[1].set_title("Global relative error vs time step")

    plt.savefig(current_dir + "/errors.png")


def plot_vars_vs_masses(solar_masses, eccentricities, moon_masses, periods): 

    """
    Function that produces one subplot for the eccentricity of Earth's orbit versus solar mass, and 
    one subplot for the Moon's synodic period versus lunar mass.

    Args: 
    - solar_masses: the solar mass-factors used for the analysis
    - eccentricities: the resulting eccentricities of Earth's orbit
    - moon_masses: the lunar mass-factors used forthe analysis
    - periods: the resulting lunar synodic periods.
    
    """

    fig, ax = plt.subplots(1, 2, figsize = (14, 4))
    
    ax[0].plot(solar_masses, eccentricities)
    ax[0].set_title(f"Earth's orbital eccentricity vs Sun-mass")
    ax[0].set_xlabel("Solar masses")
    ax[0].set_ylabel("Eccentricity of Earth's orbit ")

    ax[1].plot(moon_masses, periods)
    ax[1].set_title(f"Moon's synodic period vs Moon-mass")
    ax[1].set_xlabel("Lunar masses")
    ax[1].set_ylabel("Moon's synodic period")

    plt.savefig(current_dir + "/masses_vs_params.png")

