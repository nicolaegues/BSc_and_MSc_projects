import ctypes
import numpy as np
import matplotlib.pyplot as plt
import os


"""
https://docs.python.org/3/library/ctypes.html
"""

print("\nRunning Python wrapper!\n")

current_dir = os.path.dirname(os.path.abspath(__file__)) 

#load the shared library (precompiled C++ program) that contains the c++ functions I'll be wanting to access
mini2 = ctypes.CDLL(current_dir + "/mini2_nicola.so")

lower_b = 0.0 #the lower bound of the integration interval
upper_b = 1.0 #the upper bound of the integration interval
correct_integral = 22/7 - np.pi #correct value of the integral
iterations = 30 # Number of iterations to testing various spacings (and resulting absolute error)

#Create ctypes arrays to store the results for interval spacings and errors (for each method), with a defined size equal to the nr of iterations
interval_spacings = (ctypes.c_double * iterations)()
mid_abs_errors = (ctypes.c_double * iterations)()
left_abs_errors = (ctypes.c_double * iterations)()
trap_abs_errors = (ctypes.c_double * iterations)()
simp_abs_errors = (ctypes.c_double * iterations)()

#Convert the ctypes arrays to numpy arrays for later
interval_spacings_np = np.ctypeslib.as_array(interval_spacings)
mid_abs_errors_np = np.ctypeslib.as_array(mid_abs_errors)
left_abs_errors_np = np.ctypeslib.as_array(left_abs_errors)
trap_abs_errors_np = np.ctypeslib.as_array(trap_abs_errors)
simp_abs_errors_np = np.ctypeslib.as_array(simp_abs_errors)

"""
the cytpes library does not support returning c++ vectors into python.
To work around this, the ctype arrays are passed into the function instead - when passed to the cpp function, these will be treated as pointers to the first elements of the arrays. 
Based on these pointers to the first elements (first element's memory address), the memory locations of the arrays can then be iteratively filled with the calculated values.
"""

#Specifiy the argument and return type(s) of the function
mini2.error_vs_spacing_pointers.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
mini2.error_vs_spacing_pointers.restype = None
#call the function
mini2.error_vs_spacing_pointers(lower_b, upper_b, correct_integral, iterations, interval_spacings, mid_abs_errors, left_abs_errors, trap_abs_errors, simp_abs_errors)


interval_spacings_formatted = [f'{val:.1e}' for val in interval_spacings_np.flatten()] #for neater xtick labels

#Plot absolute error vs spacing
fig, ax = plt.subplots(figsize = (10, 6))
ax.plot(mid_abs_errors_np, label = "Midpoint method")
ax.plot(left_abs_errors_np, label = "Leftpoint method", linewidth = 2)
ax.plot(trap_abs_errors_np, label = "Trapezium method")
ax.plot(simp_abs_errors_np, label = "Simpon's method")

ax.set_yscale('log')
ax.set_xticks(np.arange(len(interval_spacings_np)))  # Set tick positions based on the number of spacings
ax.set_xticklabels(interval_spacings_formatted, rotation=45) 
ax.set_ylabel("Absolute error (log-scale)")
ax.set_xlabel("Interval spacing")
ax.set_title("Absolute error vs spacing")
ax.legend()
plt.savefig(current_dir + "/error_vs_spacing_nicola")

############################################################################################################################

max_threads = 16 #Maximum number of threads for the parallel computation
spacing = 1e-8 #Subinterval spacing to use for this parallel testing

#Create ctypes arrays to store the results for the number of threads and computation times
all_nr_threads = (ctypes.c_double * max_threads)()
times = (ctypes.c_double * max_threads)()
all_nr_threads_np = np.ctypeslib.as_array(all_nr_threads)
times_np = np.ctypeslib.as_array(times)

#Specifiy the argument and return type(s) of the function
mini2.time_vs_threads_pointers.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
mini2.time_vs_threads_pointers.restype = None
#call the function
mini2.time_vs_threads_pointers(lower_b, upper_b, spacing, max_threads, all_nr_threads, times)

#calculate the speedup
speedup = times_np[0]/times_np

#Plot both the computation time and the Speedup vs number of threads 
fig, ax = plt.subplots(2,figsize = (10, 8))
ax[0].plot(all_nr_threads_np, times_np)
ax[0].set_ylabel("Wall time (s)")
ax[0].set_xlabel("Nr of threads")

ax[1].plot(all_nr_threads_np, speedup)
ax[1].set_ylabel("Speedup")
ax[1].set_xlabel("Nr of threads")
fig.suptitle("Time and Speedup - Midpoint method")
plt.savefig(current_dir + "/time_vs_threads_nicola")
