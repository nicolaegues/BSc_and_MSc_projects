import numpy as np
import matplotlib.pyplot as plt
import time


m = 120 #kg, approx. of Baumgartner's weight (body+equipment)
Cd = 1.15 #drag coefficient(approx 1.0-1.3 for a skydiver)
p0 = 1.2 #kg*m^-3, air density (approx. 1.2 at ambient temp and pressure)
h = 7640 #m, scale height for the atmosphere (for c)
A = 0.8 #m^2, cross sectional area of the sky diver. Approximate the person as a box with area 0.08m
k_0 = 0.5*(Cd*p0*A) #, kg*m^-2, constant k 
g = 9.81 #m/s^2, acceleration due to gravity



t_min = 0.0
t_max= 1000.0 #s, 
dt = 0.01
nr_steps =int(t_max/dt)

y0 =39000 #m, inital height 
v0 = 0 #m/s, initial velocity

#########################################################################################################################################################
# Functions

def y_analytical(t):
    """
    Function for an analytical calculation of the height of a falling object with a constant drag factor as a function of time.

    Parameters
    ----------
    t : Argument for the height function.
    
    Returns
    -------
    y(t): Height of the object as a function of time
    """
    return y0-m/k_0*np.log(np.cosh(np.sqrt(k_0*g/m)*t))

def v_analytical(t):
    """
    Does an analytical calculation of the vertical speed of a falling object uwith a constant drag factor as a function of time.

    Parameters
    ----------
    t : Argument for the speed function.
    
    Returns
    -------
    v(t): Speed of the object as a function of time
    """
    return -np.sqrt(m*g/k_0)*np.tanh(np.sqrt(k_0*g/m)*t)

def plot_analytical_freefall(Time, Height, Speed, suptitle):
    """
    Function to plot the analytical Height and Speed of the object as a function of Time

    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.set(xlabel="Time (s)", ylabel="Height (m)", title="Altitude")
    ax2.set(xlabel="Time (s)", ylabel="Speed (m/s)", title="Vertical speed")
    ax1.plot(Time, Height, "tab:red")
    ax2.plot(Time, Speed, "tab:green")
    fig.suptitle(suptitle)
    plt.show()

def plot_euler_freefall(time, height_vals, speed_vals, acc_vals, suptitle):  
    """
    Function to plot Euler's Height, Speed, and Acceleration of the object as a function of Time

    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    ax1.set(xlabel="Time (s)", ylabel="Height (m)", title="Altitude")
    ax2.set(xlabel="Time (s)", ylabel="Speed (m/s)", title="Vertical speed")
    ax3.set(xlabel="Time (s)", ylabel="Acceleration (m/s^2)", title="Vertical acceleration")
    ax1.plot(time, height_vals, "tab:red")
    ax2.plot(time, speed_vals, "tab:green")
    ax3.plot(time, acc_vals, "tab:blue")
    fig.suptitle(suptitle)
    plt.show()

def plot_combined_freefall(time_euler, height_vals_euler, speed_vals_euler, time_analytical, height_vals_analytical, speed_vals_analytical, suptitle):
    """
    Function to plot the analytical and the Euler results (Height and Speed as a function of Time) in the same figure

    """
    fig, axes= plt.subplots(2, 2, figsize=(10, 8), sharex = True)
    

    # Plot Euler method results for height and speed
    #upper left
    axes[0, 0].plot(time_euler, height_vals_euler, color="tab:red")
    axes[0, 0].set_title('Euler Method - Altitude')
    axes[0, 0].set(ylabel="Height (m)")

    #upper right
    axes[0, 1].plot(time_euler, speed_vals_euler, color="tab:green")
    axes[0, 1].set_title('Euler Method - Vertical speed')
    axes[0, 1].set(ylabel = 'Speed (m/s)')

    # Plot Analytical method results for height and speed
    #lower left
    axes[1, 0].plot(time_analytical, height_vals_analytical, color="tab:orange")
    axes[1, 0].set_title('Analytical Method - Altitude')
    axes[1, 0].set(xlabel = "Time (s)", ylabel = 'Height (m)')

    #lower right
    axes[1, 1].plot(time_analytical, speed_vals_analytical, color="tab:blue")
    axes[1, 1].set_title('Analytical Method - Vertical speed')
    axes[1, 1].set(xlabel = "Time (s)", ylabel = 'Speed (m/s)')


    # Additional layout settings
    plt.suptitle(suptitle, fontsize=14)
    plt.show()

def plot_dt_effect(stepsize, height_error, speed_error, elapsedT):
    """
    Function to plot how the RSME between the analytical and the Euler method, and the time to compute the Euler method vary with the step size.
    For part b)

    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    ax1.set(xlabel="Step size dt (s)", ylabel="Height RSME")
    ax2.set(xlabel="Step size dt (s)", ylabel="Speed RSME")
    ax3.set(xlabel="Step size dt (s)", ylabel="Euler computing time (s)")
    ax1.plot(stepsize, height_error, "tab:red")
    ax2.plot(stepsize, speed_error, "tab:green")
    ax3.plot(stepsize, elapsedT, "tab:blue")
    fig.suptitle("RMSE and computing time as a function of step-size")
    plt.show()

def plot_dependency(varying_par_list, duration_list, vel_list, my_xlabel, title):
    """
    Function to plot the Duration of fall and Maximum velocity as a function of height and as a function of the factor Cd*A/m. 
    For part c)

    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.set(xlabel=my_xlabel, ylabel="Duration of fall (m)", title="Duration of Fall")
    ax2.set(xlabel=my_xlabel, ylabel="Maximum velocity (m/s)", title="Maximum Velocity")
    ax1.plot(varying_par_list, duration_list, "tab:red")
    ax2.plot(varying_par_list, vel_list, "tab:green")
    fig.suptitle(title)
    plt.show()

def Analytical_Method():
    """
    Function to calculate Height and Speed during the fall analytically. 
    
    Returns
    -------
    t_a[:i+1]: array of time values
    y_a[:i+1]: array of height values
    v_a[:i+1]: array of velocity values
    t_ground_a: duration of fall
    v_max_a: maximum velocity
    v_ground_a: velocity at ground
    t_vmax_a: time of maximum velocity


    """

    y_a = np.zeros(nr_steps)
    v_a = np.zeros(nr_steps)
    t_a= np.linspace(t_min, t_max, nr_steps)
    y_a[0] = y0 
    v_a[0] = v0 
    
    
    for i in range(nr_steps):
        y_a[i] = y_analytical(t_a[i])
        v_a[i] = v_analytical(t_a[i])

        if y_a[i] <= 0:
            break

    v_max_a = np.min(v_a)
    t_vmax_index_a = np.argmin(v_a)
    t_vmax_a = t_a[t_vmax_index_a]
    t_ground_a = t_a[i]
    v_ground_a = v_a[i]
    
    return t_a[:i+1], y_a[:i+1], v_a[:i+1], t_ground_a, v_max_a, v_ground_a, t_vmax_a

def Euler_Method(drag_type):
    """
    Function to calculate Height, Speed and Acceleration during the fall with the Euler MEthod, 
    either using a constant drag factor or a varying drag factor depending on user input.

    Parameters:
    ----------
    drag_type: either "with constant k" or "with varying k"
    
    Returns
    -------
    t_e[:i+1]: array of time values
    y_e[:i+1]: array of height values
    v_e[:i+1]: array of velocity values
    -a_e[:i+1]: array of acceleration values
    t_ground_e: duration of fall
    v_max_e: maximum velocity
    v_ground_e: velocity at ground
    t_vmax_e: time of maximum velocity
    elapsed_time: time taken to compute the Euler method

    """
    try: 
        start_time = time.time()
        k_e= np.zeros(nr_steps)
        y_e = np.zeros(nr_steps)
        v_e = np.zeros(nr_steps)
        a_e = np.zeros(nr_steps)
        t_e= np.linspace(t_min, t_max, nr_steps)
        y_e[0] = y0 
        v_e[0] = v0 
        a_e[0] = g 

        if drag_type == "with constant k":
            for i in range(1, nr_steps): 
                a_e[i] = g-k_0*(v_e[i-1]**2)/m
                #v_e[i] = v_e[i-1]-dt*(g-k_0*(v_e[i-1]**2)/m)
                v_e[i] = v_e[i-1]-dt*(a_e[i-1])
                y_e[i] = y_e[i-1] + dt*v_e[i-1]
                if y_e[i] <= 0:
                    break
            

        elif drag_type == "with varying k":   
            for i in range(1, nr_steps): 
                k_e[i] = k_0*np.exp(-y_e[i-1]/h)
                a_e[i] = g-k_e[i]*(v_e[i-1]**2)/m
                #v_e[i] = v_e[i-1]-dt*(g-k_e[i]*(v_e[i-1]**2)/m)
                v_e[i] = v_e[i-1]-dt*(a_e[i-1])
                y_e[i] = y_e[i-1] + dt*v_e[i-1]
                if y_e[i] <= 0:
                    break

        if y_e[-1] > 0:
            raise ValueError("Euler method: Object did not reach the ground within the specified time range.")
        
    except ValueError as ve:
        print(f"Error: {ve}")

    t_ground_e = t_e[i]
    v_term_e= v_e[i]
    v_max_e = np.min(v_e)
    t_vmax_index_e = np.argmin(v_e)
    t_vmax_e = t_e[t_vmax_index_e]

    end_time = time.time()
    elapsed_time = end_time- start_time
    
    return t_e[:i+1], y_e[:i+1], v_e[:i+1], -a_e[:i+1],t_ground_e, v_max_e, v_term_e, t_vmax_e, elapsed_time

def modified_Euler_Method(drag_type):
    """
    Function to calculate Height, Speed and Acceleration during the fall with the modified Euler Method, 
    either using a constant drag factor or a varying drag factor depending on user input.

    Parameters:
    ----------
    drag_type: either "with constant k" or "with varying k"
    
    Returns
    -------
    t_e[:i+1]: array of time values
    y_e[:i+1]: array of height values
    v_e[:i+1]: array of velocity values
    -a_e[:i+1]: array of acceleration values
    t_ground_e: duration of fall
    v_max_e: maximum velocity
    v_ground_e: velocity at ground
    t_vmax_e: time of maximum velocity
    elapsed_time: time taken to compute the Euler method
    
    """
    try: 

        start_time = time.time()
        k_me= np.zeros(nr_steps)
        y_me = np.zeros(nr_steps)
        v_me = np.zeros(nr_steps)
        a_me = np.zeros(nr_steps)
        t_me= np.linspace(t_min, t_max, nr_steps)
        y_me[0] = y0 
        v_me[0] = v0 
        a_me[0] = g 

        if drag_type == "with constant k":
            for i in range(1, nr_steps): 
                a_me[i] = g-k_0*(v_me[i-1]**2)/m
            
                # Modified Euler method
                v_midpoint = v_me[i - 1] - dt * (a_me[i - 1]) / 2
                a_midpoint = g - k_0 * (v_midpoint ** 2) / m
                v_me[i] = v_me[i - 1] - dt * a_midpoint
                y_me[i] = y_me[i - 1] + dt * v_midpoint

                if y_me[i] <= 0:
                    break

        elif drag_type == "with varying k":   
            for i in range(1, nr_steps): 
                k_me[i] = k_0*np.exp(-y_me[i-1]/h)
                a_me[i] = g-k_me[i]*(v_me[i-1]**2)/m

                # Modified Euler method
                v_midpoint = v_me[i - 1] - dt * (a_me[i - 1]) / 2
                a_midpoint = g - k_me[i] * (v_midpoint ** 2) / m
                v_me[i] = v_me[i - 1] - dt * a_midpoint
                y_me[i] = y_me[i - 1] + dt * v_midpoint


                if y_me[i] <= 0:
                    break

        if y_me[-1] > 0:
            raise ValueError("Euler method: Object did not reach the ground within the specified time range.")
        
    except ValueError as ve:
        print(f"Error: {ve}")

    t_ground_me = t_me[i]
    v_term_me= v_me[i]
    v_max_me = np.min(v_me)
    t_vmax_index_me = np.argmin(v_me)
    t_vmax_me = t_me[t_vmax_index_me]

    end_time = time.time()
    elapsed_time = end_time- start_time
    
    return t_me[:i+1], y_me[:i+1], v_me[:i+1], -a_me[:i+1],t_ground_me, v_max_me, v_term_me, t_vmax_me, elapsed_time

def calculate_rmse(analytical_val, euler_val):
    """
    Function to calculate the RSME between the analytical and the Euler results. 

    Returns
    -------
    RSME between Euler and analytical values.

    """
    min_length = min(len(analytical_val), len(euler_val))
    return np.sqrt(np.mean((analytical_val[:min_length] - euler_val[:min_length])**2))

#########################################################################################################################################################
#Main program

MyInput = '0'
while MyInput != 'q':
    try:
        MyInput = input('Enter a choice, "a", "b", "c", "d", or "q" to quit: ')
        if MyInput not in ['a', 'b', 'c', 'd', 'q']:
            raise ValueError("Invalid choice. Please enter a valid option.")
    except ValueError as e:
        print(f"Error: {e}")
        continue

##################################################################################################
#Part a
    if MyInput == 'a':
        print('You have chosen part (a)')
        a_t, a_y, a_v, a_t_ground, a_v_max, a_v_ground, a_t_vmax = Analytical_Method()
       
        #prints analytical results in table format
        print(f"{'Method':<25}{'Initial height (m)':<25}{'Duration of fall (s)':<25}{'Velocity at ground (m/s)':<25}{'Maximum velocity (m/s)':<25}{'Time of max velocity (s)':<25}")
        print("-" * 140)
        print(f"{'Analytical Method':<25}{y0:<25}{a_t_ground:<25.4f}{a_v_ground:<25.4f}{a_v_max:<25.4f}{a_t_vmax:<25.4f}")
        
        #option to plot results
        while True: 
            p_input = input('Type "plotA" if you want to plot the results or press Enter to skip.')
            if p_input == "plotA":
                plot_analytical_freefall(a_t, a_y, a_v, "Newtonian freefall with constant drag factor - Analytical method")
            elif p_input == "":
                break
            else:
                print("That's not a valid option, try again")

##################################################################################################
#Part b
    elif MyInput == 'b':
        print('You have chosen part (b)')


        #option to use Euler or modified Euler method
        while True:   
            e_input = input("Type 'E' If you want to use the Euler method, or 'ME' if you want to use the modified Euler method: ")
            if e_input == "E":
                e_t, e_y, e_v, e_a, e_t_ground, e_v_max, e_v_ground, e_t_vmax, e_comp_t = Euler_Method("with constant k")
                break
            elif e_input == "ME":
                e_t, e_y, e_v, e_a, e_t_ground, e_v_max, e_v_ground, e_t_vmax, e_comp_t = modified_Euler_Method("with constant k")
                break
            else: 
                print("That's not a valid option, try again")
        a_t, a_y, a_v, a_t_ground, a_v_max, a_v_ground, a_t_vmax  = Analytical_Method()
        
        #error analysis between both methods
        rmse_height = calculate_rmse(a_y, e_y)
        rmse_speed = calculate_rmse(a_v, e_v)

        diff_t_ground = abs(a_t_ground -e_t_ground)
        diff_v_term = abs(a_v_ground - e_v_ground)
        diff_v_max = abs(a_v_max - e_v_max)
        diff_t_vmax = abs(a_t_vmax - e_t_vmax)

        print("Error between the analytical and Euler methods:")
        print("\tRMSE (Height): {:.4f}".format(rmse_height))
        print("\tRMSE (Speed): {:.4f}".format(rmse_speed))

        print("Time to compute the Euler Method: \n\t{:.6f} seconds".format(e_comp_t))
        
        #prints Euler and anlytical results, and their difference in table format
        print(f"{'Method':<25}{'Duration of fall (s)':<25}{'Velocity at ground (m/s)':<25}{'Maximum velocity (m/s)':<25}{'Time of max velocity (s)':<25}")
        print("-" * 140)
        print(f"{'Euler Method':<25}{e_t_ground:<25.4f}{e_v_ground:<25.4f}{e_v_max:<25.4f}{e_t_vmax:<25.4f}")
        print(f"{'Analytical Method':<25}{a_t_ground:<25.4f}{a_v_ground:<25.4f}{a_v_max:<25.4f}{a_t_vmax:<25.4f}")
        print(f"{'Difference':<25}{diff_t_ground:<25.6f}{diff_v_term:<25.6f}{diff_v_max:<25.6f}{diff_t_vmax:<25.6f}")
        
        #plotting options
        p_input = 0
        while p_input != "":
            p_input = input('Type "plotE" if you only want to plot the Euler results, "plotAE" if you want to plot both methods, "plotR" if you want to plot RMSE of the two methods as a function of step size, or press Enter to skip: ')
            if p_input == "plotE":
                plot_euler_freefall(e_t, e_y, e_v, e_a, "Newtonian freefall with constant drag factor - Euler Method")
                
            elif p_input == "plotAE":
                plot_combined_freefall(e_t, e_y, e_v, a_t, a_y, a_v, "Newtonian freefall with constant drag factor")
            elif p_input == "plotR":
                rmse_list_h = []
                rmse_list_v = []
                elapsed_t_list = []
                dt_list = list(np.arange(0.001, 20, 0.01))
                for i in dt_list : 
                    dt = i
                    nr_steps =int(t_max/dt)
                    e__t, e__y, e__v, e__a, e__t_ground, e__v_max, e__v_ground, e__t_vmax, e__comp_t = Euler_Method("with constant k")
                    a__t, a__y, a__v, a__t_ground, a__v_max, a__v_ground, a__t_vmax  = Analytical_Method()
                    rmse_h = calculate_rmse(a__y, e__y)
                    rmse_v = calculate_rmse(a__v, e__v)
                    rmse_list_h.append(rmse_h)
                    rmse_list_v.append(rmse_v) 
                    elapsed_t_list.append(e__comp_t)    
                plot_dt_effect(dt_list, rmse_list_h, rmse_list_v, elapsed_t_list)
            elif p_input == "":
                break
            else:
                print("That's not a valid option, try again")

##################################################################################################
#Part c
    elif MyInput == "c": 
        print('You have chosen part (c)')
    
        e_t, e_y, e_v, e_a, e_t_ground, e_v_max, e_v_ground, e_t_vmax, e_comp_t = Euler_Method("with varying k")
        
        #prints Euler results in table format
        print(f"{'Method':<25}{'Duration of fall (s)':<25}{'Velocity at ground (m/s)':<25}{'Maximum velocity (m/s)':<25}{'Time of max velocity (s)':<25}")
        print("-" * 140)
        print(f"{'Euler Method':<25}{e_t_ground:<25.4f}{e_v_ground:<25.4f}{e_v_max:<25.4f}{e_t_vmax:<25.4f}")
        
        #plotting options
        p_input = 0
        while p_input != "":
            p_input = input('Type "plotE" to plot Euler results, "plotH" to vary initial height, "plotC" to vary the factor Cd*A/m, or press Enter to skip: ')
            if p_input == "plotE":
                plot_euler_freefall(e_t, e_y, e_v, e_a, "Newtonian freefall with varying drag factor- Euler Method")

            elif p_input == "plotH":
                y0_list = []
                v_max_y0_list = []
                t_ground_y0_list= []
                for y0 in range(1000, 39000, 1000):
                    euler_vals = Euler_Method("with varying k")
                    t_ground_y0_list.append(euler_vals[4])
                    v_max_y0_list.append(euler_vals[5])
                    y0_list.append(y0)
                plot_dependency(y0_list, t_ground_y0_list, v_max_y0_list, "Initial height (m)", "Duration of fall and Maximum velocity as a function of height")
            
            
            elif p_input == "plotC":
                m_list = []
                v_max_m_list = []
                t_ground_m_list= []
                for m in range(20, 250):
                    euler_vals = Euler_Method("with varying k")
                    t_ground_m_list.append(euler_vals[4])
                    v_max_m_list.append(euler_vals[5])
                    m_list.append(Cd*A/m)  
                
                plot_dependency(m_list, t_ground_m_list, v_max_m_list, "Cd*A/m (m^2/kg)", "Duration of fall and Maximum velocity as a function of factor Cd*A/m")
            
            elif p_input == "":
                break

            else: 
                print("That's not a valid option, try again")
        

##################################################################################################
#Part d
    elif MyInput == "d": 

        e_t, e_y, e_v, e_a, e_t_ground, e_v_max, e_v_term, e_t_vmax, e_comp_t= Euler_Method("with varying k")
        temp_list = []
        gamma = 1.4
        r = 8.3144626
        m_mol = 0.0289645
        v_sound = np.zeros(len(e_y))

        #calculate temperature as a function of height
        for y_el in e_y: 
            if y_el > 25100: 
                temp = 141.3 +0.0030*y_el
                temp_list.append(temp)
            elif 11000 < y_el <= 25100:
                temp = 216.5
                temp_list.append(temp)
            elif y_el <= 11000: 
                temp = 288.0-0.0065*y_el
                temp_list.append(temp)
        
        #calculate speed of sound as a function of height
        for i in range(len(temp_list)):
            v_sound[i] = np.sqrt((gamma*r*temp_list[i])/m_mol)

        #determine his maximum Mach number 
        t_vmax_index = np.argmin(e_v)
        v_sound_at_v_max = v_sound[t_vmax_index]
        max_mach = abs(e_v_max)/v_sound_at_v_max
        print("Baumgartner's maximum Mach number as he falls is: \n\t{:.6f}".format(max_mach))

        #determine from when to when he breaks the sound barrier
        v_diff = np.subtract(abs(e_v), v_sound)
        
        break_indices = np.where(v_diff > 0)[0]
        if break_indices.size == 0:
            print("Baumgartner did not break the sound barrier during the fall.")
        
        else: 
            first_break_index = break_indices[0]
            last_break_index = break_indices[-1]
            print("The sound barrier is broken between: \n\t{:.4f} s and {:.4f} s, \nbetween the heights: \n\t{:.4f} m and {:.4f} m".format(e_t[first_break_index], e_t[last_break_index], e_y[first_break_index], e_y[last_break_index]))

        #plotting options
        p_input = 0
        while p_input != "":
            p_input = input("Type 'plotV' to plot Temperature as a function of Height, or 'plotS' to plot the Speed of sound and his Vertical speed as a function of Time, or press Enter to skip: ")
            if p_input == "plotV":
                fig, (ax1) = plt.subplots(1, figsize = (12,4))
                ax1.set(xlabel="Height(m)", ylabel = "Temperature (K)", title="Temperature(K) vs Altitude(m)")
                ax1.plot(e_y, temp_list, "r")
                plt.show()
            elif p_input == "plotS":
                fig, (ax1) = plt.subplots(1, figsize = (12,4))
                ax1.set(xlabel="Time (s)", title="Speed of sound and Vertical speed vs Time")
                ax1.plot(e_t, v_sound, "r", label = "Speed of sound (m/s)")
                ax1.plot(e_t, -e_v, "g", label = "(absolute) Vertical speed (m/s)")
                plt.legend()
                plt.show()

            elif p_input == "":
                break

            else: 
                print("That's not a valid option, try again")
        
        
