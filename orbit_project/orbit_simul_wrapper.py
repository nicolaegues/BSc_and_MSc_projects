
"""

Compilation Commands:

To compile the C++ code into a shared library (so the python wrapper can use it):
g++ -fPIC -shared orbit_simul.cpp -o orbit_simul.so

Running the python script:
python3 orbit_simul_wrapper.py

"""

import ctypes
import os
import numpy as np
from scipy.signal import find_peaks
from orbit_plot_module import plot_general_2d_animation, plot_energy, plot_errors, plot_vars_vs_masses
import pandas as pd

print("\nRunning Python wrapper!\n")

current_dir = os.path.dirname(os.path.abspath(__file__)) 

#load the shared library (precompiled C++ program) that contains the c++ functions I'll be wanting to access
cpp_orbit_simul = ctypes.CDLL(current_dir + "/orbit_simul.so")

#Set the initial parameters for the chosen celestial bodies within a system
solar_system = {
    "Sun": {"mass": 1.98847e30, "initial_position": [0.0, 0.0, 0.0], "initial_velocity": [0.0, 0.0, 0.0]},
    "Moon": {"mass": 7.3476e22, "initial_position": [1.495e11, 3.84e8, 0], "initial_velocity": [-1.022e3, 2.978e4, 0]},
    "Earth": {"mass": 5.9772e24, "initial_position": [1.495e11, 0, 0], "initial_velocity": [0, 2.978e4, 0]},
}


AU = 1.496e11
G = 6.67e-11
kepler_system = {
    "Kepler-16A": {"mass": 0.69 * solar_system["Sun"]["mass"], "initial_position": [0.0, 0.0, 0.0], "initial_velocity": [0.0, 0.0, 0.0]},
    "Kepler-16B": {"mass": 0.202 * solar_system["Sun"]["mass"],"initial_position": [0.22 * AU, 0, 0],"initial_velocity": [0, (G * (0.69 * solar_system["Sun"]["mass"]) / (0.22 * AU))**0.5, 0]},
    "Kepler-16b": {"mass": 0.33 * 1.898e27, "initial_position": [0.7 * AU, 0, 0],"initial_velocity": [0, (G * (0.69 * solar_system["Sun"]["mass"]) / (0.7 * AU))**0.5, 0]
    }
}


system = solar_system #choose a system to simulate

#collect the initial parameters into lists to pass for later use in the C++ functions.
masses_l = []
initial_positions = []
initial_velocities = []
for name, body in system.items(): 
    masses_l.append(body["mass"])
    initial_positions.extend(body["initial_position"])
    initial_velocities.extend(body["initial_velocity"])


nr_bodies = len(masses_l) #number of bodies to simulate
t_max = 94608000 # the time to run the simulation form in seconds (for 3 years, set to 94608000)
dt = 3600 #the size of the timestep in seconds (set 3600 for 1 hour)
nr_steps = int(t_max/dt) #The number of steps the Velocity Verlet algorithm will perform


sun_index = list(solar_system.keys()).index("Sun") 
earth_index = list(solar_system.keys()).index("Earth")
moon_index = list(solar_system.keys()).index("Moon")

def moon_synodic_period(r_sun, r_earth, r_moon, dt): 
    """
    Function to calculate the synodic period of the Moon by tracking the peaks in the difference between the SUn-Moon distances and the Earth-Moon distaneces

    Args: 
    - r_sun: array containing the 3D positions over time of the Sun
    - r_earth: array containing the 3D positions over time of the Earth
    - r_moon: array containing the 3D positions over time of the Moon
    - dt: size of the time-step

    Returns: 
    - The average synodic period in days.
    
    """

    sun_moon_dist = np.linalg.norm(r_sun - r_moon, axis = 1)
    earth_moon_dist = np.linalg.norm(r_earth - r_moon, axis = 1)

    diff = sun_moon_dist - earth_moon_dist

    peak_indices, _ = find_peaks(diff)

    #Get the mean period
    all_periods = [(peak_indices[i+1]-peak_indices[i])*dt for i in range(len(peak_indices)-1)]
    avg_period = np.mean(all_periods)

    avg_period_days = avg_period/86400

    return avg_period_days

def sidereal_period(r1, r2, dt):

    """
    Function to calculate the sidereal period of a planet by tracking the angular position of the body over time and identifying full cycles.


    Args: 
    - r_1: array containing the 3D positions over time of the host star
    - r_2: array containing the 3D positions over time of the orbiting planet.
    - dt: the size of the time-step.
 
    Returns: 
    - The average sidereal period in days.
    
    """


    # Relative position of body 1 (eg Earth) with respect to body 2 (eg the Sun)
    r_rel = r1 - r2
    
    x_rel, y_rel = r_rel[:, 0], r_rel[:, 1]
    
    # Compute angular position over time
    angles = np.arctan2(y_rel, x_rel)

    #The angles jump from pi to -pi (and viceverse), so need to remove the discontinuties to get a steady increase
    angles = np.unwrap(angles) 

    #normalise to be in full rotations, where one full rotation is 2pi.
    angles = angles / (2 * np.pi)

    #This rounds down each angle, turning fractions of rotations into full integer rotations (so that we can then track complete orbits)
    angles = np.floor(angles)
   
   #CHanges in integer roation value mean completions of full cycles.
    diff_angles = np.diff(angles)

    #Find where angle crosses multiples of 2π
    crossings = np.where(diff_angles != 0)[0]
    
    #Compute time intervals between successive crossings
    periods = np.diff(crossings) * dt  
    
    avg_period = np.mean(periods)
    avg_period_days = avg_period/86400  
    
    return avg_period_days

def moon_period_vs_mass(nr_evals):
    """
    Function to calculate the Moon's synodic period against a the mass of the Moon, which is varied from 1 to 100 lunar masses.

    Args: 
    - nr_evals: How many different lunar masses to test.

    Returns: 
    - moon_masses_frac: the factor of lunar mass used to retrieve each synodic period.
    - periods: the collected synodic periods.
    
    """

    og_moon_mass = masses_l[moon_index]
    #to modify the lunar mass inside the loop with the other masses untouched:
    masses_f = masses_l.copy()

    periods = []
    moon_masses_frac = []

    for frac in np.linspace(1, 100, nr_evals): 

        #update the masses ctypes array to pass into the velocity verlet function.
        new_mass = frac*og_moon_mass
        masses_f[moon_index] = new_mass
        masses_arr = (ctypes.c_double * nr_bodies)(*masses_f)

        all_positions_1= (ctypes.c_double * (nr_bodies* nr_steps * 3))()
        all_velocities_1 = (ctypes.c_double * (nr_bodies* nr_steps * 3))()

        all_positions_1_np = np.ctypeslib.as_array(all_positions).reshape(nr_bodies, nr_steps, 3 )

        cpp_orbit_simul.velocity_verlet(nr_bodies, masses_arr, initial_positions, initial_velocities, nr_steps, dt, all_positions_1, all_velocities_1)

        #positions of the bodies with this new lunar mass
        r_sun_np = all_positions_1_np[0]
        r_earth_np = all_positions_1_np[1]
        r_moon_np = all_positions_1_np[2]

        period = moon_synodic_period(r_sun_np, r_earth_np, r_moon_np, dt)

        moon_masses_frac.append(frac)
        periods.append(period)
       
    return moon_masses_frac, periods

def error_vs_dt(nr_evals):

    """
    Function to calculate the the global relative total energy error against size of the timestep, which is varied from 1 to 8000 seconds.

    Args: 
    - nr_evals: How many different time step sizes to test.
    
    Returns: 
    - dts: The time step sizes.
    - errors: The global relative total energy errors.
    
    """

    dts = []
    errors = []

    #Create cytpes arrays to store the positions and velocities of the bodies (in a flattened manner)
    all_positions_2 = (ctypes.c_double * (nr_bodies* nr_steps * 3))()
    all_velocities_2 = (ctypes.c_double * (nr_bodies* nr_steps * 3))()

    TE_2 =  (ctypes.c_double * nr_steps)()
    KE_2 =  (ctypes.c_double * nr_steps)()
    PE_2 =  (ctypes.c_double * nr_steps)()
    TE_error_2 =  (ctypes.c_double * nr_steps)()
    TE_error_2_np =  np.ctypeslib.as_array(TE_error_2)


    for dt in np.linspace(1, 8000, nr_evals): 

        cpp_orbit_simul.velocity_verlet(nr_bodies, masses, initial_positions, initial_velocities, nr_steps, dt, all_positions_2, all_velocities_2)
        cpp_orbit_simul.energy_calc( nr_bodies, nr_steps, all_positions_2, all_velocities_2, masses, KE_2, PE_2, TE_2, TE_error_2)
        
        dts.append(dt)
        errors.append(TE_error_2_np[-1]) #global relative error
       

    return dts, errors
    

############################################################################

#the lists storing the body's parameters are converted to cytpes arrays to make them compatible with the C++ functions. 
masses = (ctypes.c_double * nr_bodies)(*masses_l)
initial_positions = (ctypes.c_double * (nr_bodies*3))(*initial_positions)
initial_velocities = (ctypes.c_double * (nr_bodies*3))(*initial_velocities)

#Create cytpes arrays to store the positions and velocities of the bodies (in a flattened manner)
all_positions = (ctypes.c_double * (nr_bodies* nr_steps * 3))()
all_velocities = (ctypes.c_double * (nr_bodies* nr_steps * 3))()

#Convert the ctypes arrays to numpy arrays for later use
all_positions_np = np.ctypeslib.as_array(all_positions).reshape(nr_bodies, nr_steps, 3 )
all_velocities_np = np.ctypeslib.as_array(all_velocities).reshape(nr_bodies, nr_steps, 3 )

#Specifiy the argument and return type(s) of the function
cpp_orbit_simul.velocity_verlet.argtypes = [ctypes.c_int, 
                                            ctypes.POINTER(ctypes.c_double), 
                                             ctypes.POINTER(ctypes.c_double), 
                                             ctypes.POINTER(ctypes.c_double), 
                                                 ctypes.c_int,
                                                 ctypes.c_double, 
                                             ctypes.POINTER(ctypes.c_double), 
                                             ctypes.POINTER(ctypes.c_double), 
                                             ]

cpp_orbit_simul.velocity_verlet.restype = None

#call the function
cpp_orbit_simul.velocity_verlet(nr_bodies, masses, initial_positions, initial_velocities, nr_steps, dt, all_positions, all_velocities)


print("Simulation done, plotting GIF: ")
#Plot ()
plot_general_2d_animation(250, all_positions_np, list(system.keys()), earth_index, moon_index, dt)

#Note: The folloring is specific to the 3-body solar system case.
#Extract the 3D position's for each body, to save them as a CSV. This is to optinionally use the csv as input to a different plotting function (i.e. the one given in project)

r_sun_np = all_positions_np[sun_index]
r_earth_np = all_positions_np[earth_index]
r_moon_np = all_positions_np[moon_index]

all_pos = np.hstack([r_sun_np[:, :2], r_earth_np[:, :2], r_moon_np[:, :2]])
all_pos_df = pd.DataFrame(all_pos, columns = ["x1", "y1", "x2", "y2", "x3", "y3"])
all_pos_df.to_csv("solar_sys_xy_pos.csv", index = False, header=True)

############################################################################
# ENERGY CALCULATIONS

print("Calculating and plotting the energies...")

#Create cytpes arrays to store the  of the bodies 
TE =  (ctypes.c_double * nr_steps)()
KE =  (ctypes.c_double * nr_steps)()
PE =  (ctypes.c_double * nr_steps)()
TE_error =  (ctypes.c_double * nr_steps)()

#Convert the ctypes arrays to numpy arrays for later use
TE_np =  np.ctypeslib.as_array(TE)
KE_np =  np.ctypeslib.as_array(KE)
PE_np =  np.ctypeslib.as_array(PE)
TE_error_np =  np.ctypeslib.as_array(TE_error)

#Specifiy the argument and return type(s) of the function
cpp_orbit_simul.energy_calc.argtypes = [ctypes.c_int,
                                        ctypes.c_int,
                                        ctypes.POINTER(ctypes.c_double),
                                        ctypes.POINTER(ctypes.c_double),
                                        ctypes.POINTER(ctypes.c_double),
                                        ctypes.POINTER(ctypes.c_double),
                                        ctypes.POINTER(ctypes.c_double),
                                        ctypes.POINTER(ctypes.c_double),
                                        ctypes.POINTER(ctypes.c_double)]


cpp_orbit_simul.energy_calc.restype = None

#call the function
cpp_orbit_simul.energy_calc(nr_bodies, nr_steps, all_positions, all_velocities, masses, KE, PE, TE, TE_error)

#Plot the total-, kinetic- and potential energies of the system.
plot_energy(TE_np, KE_np, PE_np)

#Plot the local energy conservation error over time and
#how the global energy conservation error varies with the time-step
dts, global_errors = error_vs_dt(50)
plot_errors(TE_error_np, dts, global_errors)


############################################################################
# ECCENTRICTY AND ORBITAL PERIODS CALCULATIONS

print("Calculating eccentricities and orbital periods: ")
cpp_orbit_simul.eccentricity.argtypes = [ctypes.POINTER(ctypes.c_double), 
                                        ctypes.c_int,
                                        ctypes.c_int,
                                        ctypes.c_int,]

cpp_orbit_simul.eccentricity.restype = ctypes.c_double


#Get Earth's orbital eccentricity
body_1 = sun_index  #Sun
body_2 = earth_index #Earth
ecc = cpp_orbit_simul.eccentricity(all_positions, body_1, body_2, nr_steps)

earth_sidereal = sidereal_period(r_sun_np, r_earth_np, dt)
moon_sidereal = sidereal_period(r_earth_np, r_moon_np, dt)
moon_synodic = moon_synodic_period(r_sun_np, r_earth_np, r_moon_np, dt)


print(f"Eccentricity of Earth's orbit: {ecc:.10f}")
print(f"Earth's sidereal period: {earth_sidereal:.4f}")
print(f"Moon's sidereal period: {moon_sidereal:.4f}", )
print(f"Moon's synodic period: {moon_synodic:.4f} ")


############################################################################
# ANALYSIS OF ECCENTRICTY VS SOLAR MASS

nr_evals = 10
eccentricities= (ctypes.c_double * (nr_evals))()
solar_masses= (ctypes.c_double * (nr_evals))()

eccentricities_np =  np.ctypeslib.as_array(eccentricities)
solar_masses_np =  np.ctypeslib.as_array(solar_masses)

cpp_orbit_simul.eccentricity_vs_sun_mass.argtypes = [ctypes.c_int, 
                                                    ctypes.POINTER(ctypes.c_double), 
                                                    ctypes.POINTER(ctypes.c_double), 
                                                    ctypes.POINTER(ctypes.c_double), 
                                                    ctypes.c_int,
                                                    ctypes.c_double, 
                                                    ctypes.POINTER(ctypes.c_double), 
                                                    ctypes.POINTER(ctypes.c_double),  
                                                    ctypes.c_int,
                                                    ctypes.c_int,
                                                    ctypes.c_int,
                                                    ctypes.POINTER(ctypes.c_double), 
                                                    ctypes.POINTER(ctypes.c_double), 
                                                    ]

cpp_orbit_simul.eccentricity_vs_sun_mass.restype = None
#call the function
cpp_orbit_simul.eccentricity_vs_sun_mass( nr_bodies, masses, initial_positions, initial_velocities, nr_steps, dt, all_positions, all_velocities,
                                     body_1, body_2 , nr_evals, eccentricities, solar_masses)

moon_masses_frac, periods = moon_period_vs_mass(20)

plot_vars_vs_masses(solar_masses_np, eccentricities_np, moon_masses_frac, periods)






