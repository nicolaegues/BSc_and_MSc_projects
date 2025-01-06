import numpy as np
from scipy.integrate import quad, dblquad
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import sys



λ = 1*10**(-6) #m, wavelength of incident light
k = (2*np.pi)/λ #1/m, wavenumber of incident light
z = 0.0002 #m, distnace from aperture to screen
aperture_width =0.00002#m, width/diameter of aperture

E_0 = 0.01 # V/m, value of E-field of incident light in aperture
perm = 8.854*10**(-12) #F/m permittivity of free space
c = 3*10**8 #m/s, speed of light

N_1d = 200 #number of data points for 1d integral
N_2d = 50 #number of data points for 2d integral
MC_samples = 50 #number of samples for Monte Carlo integration


#########################################################################################################################################################
# Functions

##################################################################################################
#GENERAL FUNCTIONS
def get_screen_width(fres_nr): 
    """
    Finds the range of screen coordinates that depict the diffraction pattern in an appropiately "zoomed-in" level.

    Parameters
    ----------
    fres_nr : Fresnel number
    
    Returns
    -------
    Screen width
    """
    if 0.02 < z or z < 0.02 or (z == 0.02 and aperture_width == 2*10**(-5)):
        if fres_nr >= 0.5:
            screen_width = 0.00005
        elif 0.05 <= fres_nr < 0.5:
            screen_width = 0.0005
        elif 0.005 <= fres_nr < 0.05:
            screen_width = 0.005
        elif 0.0005 <= fres_nr < 0.005: 
            screen_width = 0.05
        elif fres_nr < 0.0005:
            screen_width = 0.5
        return screen_width  
    
    else : 
        if fres_nr >= 0.5: 
            screen_width = 0.005
        elif 0.05 <= fres_nr < 0.5:
            screen_width = 0.003
        elif 0.005 <= fres_nr < 0.05:
            screen_width = 0.005
        elif 0.0005 <= fres_nr < 0.005: 
            screen_width = 0.01
        elif fres_nr < 0.0005:
            screen_width = 0.1
        return screen_width  

def fresnel_nr_func(z_, ap_w): 
    """
    Calculates of the Fresnel number

    Parameters
    ----------
    z_ : Screen distance
    ap_w: Aperture width
    
    Returns
    -------
    Fresnel number as a function of the aperture width and screen distance.
    """
    return (ap_w/2)**2/(z_*λ)

def arrays(screen_wid, N):
    """
    Produces the screen-coordinate arrays in x and y as a function of the screen width and N.

    """
    sx_array = np.linspace(-screen_wid/2, screen_wid/2, N)
    sy_array = np.linspace(-screen_wid/2, screen_wid/2, N)
    return sx_array, sy_array

def Printer(data): 
    """
    Function to print constantly changing data (due to loop) in the same place. Used to print percentage of completion of integral. 
    """
    sys.stdout.write("\r\x1b[K" + str(data))
    sys.stdout.flush()

##################################################################################################
#PLOTTING FUNCTIONS
def plot_1d_fresnel(z_, ap_w, intensity_vals, cen_wid_h = 0, cen_wid = 0):
    """
    Function to plot the 1D Fresnel intensity diffraction. Also plots the width of the central maximum in the pattern. 
    For part a)

    """
    fres_nr = fresnel_nr_func(z_, ap_w)
    screen_width = get_screen_width(fres_nr)
    sx_arr, sy_arr = arrays(screen_width, N_1d)
    
    fig, ax = plt.subplots()
    ax.set(xlabel = "Screen coordinate (m)", ylabel = "Relative Intensity")
    ax.plot(sx_arr, intensity_vals)
    ax.set_title("1D diffraction (quad)")

    xmin = 0.5 - (cen_wid/screen_width)/2
    xmax = 0.5 + (cen_wid/screen_width)/2
    ax.axhline(y = cen_wid_h, xmin = xmin, xmax = xmax, color='red', linestyle='--', label='Central Width')
    ax.legend()
    plt.show()

def plot_2d_fresnel(i_2darr, shape = ""): 
    """
    Function to plot the 2D Fresnel intensity diffraction. 
    Used in part b), c), and d).

    """
    plt.imshow(i_2darr)
    plt.xlabel("screen coordinate (m)")
    plt.colorbar(label="Intensity")
    if shape == "square": 
            plt.title("2D diffraction from square ap- dblquad")
    if shape == "rectangular": 
            plt.title("2D diffraction from rectangle- dblquad")
    if shape == "circle": 
            plt.title("2D diffraction from circle- dblquad")
    if shape == "MCcircle": 
            plt.title("2D diffraction from circle- Monte Carlo")
    plt.show()

def plot_1varyingpar(varying_par, y_vals, x_label, y_label, title):
    """
    Function for a single plot of x  vs y. 
    Used in part d). 

    """
    fig, ax = plt.subplots()
    ax.set(xlabel = x_label, ylabel = y_label)
    ax.plot(varying_par, y_vals)
    plt.title(title)
    plt.show()
    
def plot_3varyingpar(varying_par1, varying_par2, varying_par3, y1_list, y2_list, y3_list, x1_label, x2_label, x3_label):
    """
    Function to plot three different paramters against the RMSE. 
    Used in part a). 
    """

    fig, axes= plt.subplots(3, 1, figsize=(10, 8), sharex = False)

    #up
    axes[0].plot(varying_par1, y1_list, color="tab:red")
    axes[0].set(xlabel=x1_label, ylabel="RMSE)")

    #umiddle
    axes[1].plot(varying_par2, y2_list, color="tab:green")
    axes[1].set(xlabel=x2_label, ylabel ="RMSE")


    #bottom
    axes[2].plot(varying_par3, y3_list, color="tab:orange")
    axes[2].set(xlabel = x3_label, ylabel = "RMSE")

    plt.suptitle("RMSE vs quad parameters")
    plt.show()

def plot_4varyingpar(varying_par1, varying_par2, y1_list, y2_list, y1_i_list, y2_i_list, x1_label, x2_label):
    """
    Function to plot two different parameters against both the Central peak width in the intensity diffraction pattern and the mean diffracted intensity. 
    Used in part a). 
    """
    fig, axes= plt.subplots(2, 2, figsize=(10, 8), sharex = "col")

    #upper left
    axes[0, 0].plot(varying_par1, y1_list, color="tab:red")
    axes[0, 0].set_title("Central peak width (m)")
    axes[0, 0].set(xlabel=x1_label, ylabel="Central peak width (m)")

    #upper right
    axes[0, 1].plot(varying_par2, y2_list, color="tab:green")
    axes[0, 1].set_title("Central peak width (m)")
    axes[0, 1].set(xlabel=x2_label, ylabel ="Central peak width (m)")


    #lower left
    axes[1, 0].plot(varying_par1, y1_i_list, color="tab:orange")
    axes[1, 0].set_title("mean intensity")
    axes[1, 0].set(xlabel = x1_label, ylabel = "Mean intensity")

    #lower right
    axes[1, 1].plot(varying_par2, y2_i_list, color="tab:blue")
    axes[1, 1].set_title("Mean intensity")
    axes[1, 1].set(xlabel = x2_label, ylabel = "Mean intensity")

    plt.show()


##################################################################################################
#PART A FUNCTIONS
def real_1dkernel(ax, sx, k_, z_):
    """
    Functino for the real part of the kernel of the 1D- Fresnel integral

    Parameters
    ----------
    ax: aperture x-coordinate
    sx: screen x - coordinate
    k_: Wave number
    z_: Screen distance
    
    Returns
    -------
    The real part as a function of the aperture- and screen coordinate, screen distance, and wavenumber. 
    """
    return np.cos((k_/(2*z_))*(sx-ax)**2)

def imag_1dkernel(ax, sx, k_, z_):
    """
    Function for the the imaginary part of the kernel of the 1D- Fresnel integral

    Parameters
    ----------
    ax: aperture x-coordinate
    sx: screen x - coordinate
    k_: Wave number
    z_: Screen distance
    
    Returns
    -------
    The imaginary part as a function of the aperture- and screen coordinate, screen distance, and wavenumber. 
    """
    return np.sin((k_/(2*z_))*(sx-ax)**2)
def quad_integration( ap_w, z_, eps_abs = 1.49*10**(-8), eps_rel = 1.49*10**(-8), limit_ = 50):
    """
    1D integration of the  real and imaginary parts of the Fresnel integral (evaluation at each point of the 1D screen)
    Does so using scipy's integrate.quad() function.

    Parameters
    ----------
    ap_w: aperture width
    z_: Screen distance
    eps_abs: the quad epsabs parameter
    eps_rel: the quad epsrel parameter
    limit_: the quad limit parameter
    
    Returns
    -------
    intensity_arr: The intensities at each screen coordinate 
    rmse: the root mean squared absolute error of the integration
    """
    
    #adjusts screen width according to fresnel number
    fres_nr = fresnel_nr_func(z_, ap_w)
    screen_width = get_screen_width(fres_nr)
    sx_arr, sy_arr = arrays(screen_width, N_1d)

    const = k*E_0/(2*np.pi*z_) #constant in front of Fresnel integral
    ax_min = -ap_w/2 #lower aperture x-limit integral
    ax_max = ap_w/2 #lower aperture x-limit integral

    intensity_arr = np.zeros(N_1d) #intensity of the diffracted light
    abserror_arr = np.zeros(N_1d) 

    #quad integrate at each screen coordinate 
    for i in range(N_1d):
        realpart, realerror= quad(real_1dkernel, ax_min, ax_max, args=(sx_arr[i], k, z_), epsabs = eps_abs, epsrel = eps_rel, limit = limit_)
        imagpart, imagerror= quad(imag_1dkernel, ax_min, ax_max, args=(sx_arr[i], k, z_), epsabs = eps_abs, epsrel = eps_rel, limit = limit_)
        mod_E_squared = const**2*(realpart**2 + imagpart**2)
        intensity_arr[i] = perm*c*mod_E_squared
        combined_abserror = np.sqrt(realerror ** 2 + imagerror ** 2)
        abserror_arr[i] = combined_abserror
        

    rmse = np.sqrt(np.mean((abserror_arr)**2))

    return intensity_arr, rmse

def central_peak_width(z_, ap_w, intensity_seq):
    """
    1D: Provides an estimate of the central peak width (FWHM) and the height at which it was evaluated in the intensity diffraction pattern, by fitting a gaussian functin to the pattern. 

    Parameters
    ----------
    z_: Screen distance
    ap_w: aperture width
    intensity_seq: 1D sequence of intensity values
    
    Returns
    -------
    central_width: The full width half maximum (FWHM) of the central maximum 
    central_width_height: Height at which the FWHM was evaluated
    """
    fres_nr = fresnel_nr_func(z_, ap_w)
    screen_width = get_screen_width(fres_nr)
    # Function to fit a Gaussian curve
    def gaussian(x, A, mu, sigma):
        return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

    # Find central peak location
    peak_index = int(len(intensity_seq) / 2) - 1

    # Fit the Gaussian curve to the data
    x = np.arange(len(intensity_seq))
    params, _ = curve_fit(gaussian, x, intensity_seq, p0=(1, peak_index, 1))

    gaussian_peak = find_peaks(gaussian(x, params[0], params[1], params[2]))
    peak = gaussian_peak[0]
    if len(gaussian_peak[0]) == 0:
        peak = [peak_index]
    central_width_s, central_width_height, _, _= peak_widths(gaussian(x, params[0], params[1], params[2]), peaks = peak) #central widths of peaks in samples
    central_width = (central_width_s/N_1d)*(screen_width) #width in m
        

    return central_width[0], central_width_height[0] # You might also return other parameters as needed

##################################################################################################
#PART B AND C FUNCTIONS
def real_2dkernel(ay, ax, sy, sx, k_, z_):
    """
    Function for the real part of the kernel of the 2D- Fresnel integral

    Parameters
    ----------
    ay: aperture y-coordinate
    ax: aperture x-coordinate
    sy_ screen y-coordinate
    sx: screen x - coordinate
    k_: Wave number
    z_: Screen distance
    
    Returns
    -------
    The real part as a function of the aperture- and screen coordinate, screen distance, and wavenumber. 
    """
    return np.cos((k_/(2*z_))*((sx-ax)**2 + (sy-ay)**2))

def imag_2dkernel(ay, ax, sy, sx, k_, z_):
    """
    Function for the imaginary part of the kernel of the 2D- Fresnel integral

    Parameters
    ----------
    ay: aperture y-coordinate
    ax: aperture x-coordinate
    sy_ screen y-coordinate
    sx: screen x - coordinate
    k_: Wave number
    z_: Screen distance
    
    Returns
    -------
    The iamginary part as a function of the aperture- and screen coordinate, screen distance, and wavenumber. 
    """
    return np.sin((k_/(2*z_))*((sx-ax)**2 + (sy-ay)**2))

def ax_min_func(ay):
    """
    Lower x-limit of the integration as a function of aperture y-coordinate, 
    here to integrate over a circle of radius aperture_width/2.

    """
    return -np.sqrt((aperture_width/2)**2 - ay**2)

def ax_max_func(ay): 
    """
    Upper x-limit of the integration as a function of aperture y-coordinate, 
    here to integrate over a circle of radius aperture_width/2.
    """

    return np.sqrt((aperture_width/2)**2 - ay**2)

def dblquad_integration(ax_min, ax_max, shape = "",): 
    """
    2D integration of the  real and imaginary parts of the Fresnel integral (evaluation at each point of the 2D screen)
    Does so using scipy's integrate.dblquad() function.

    Parameters
    ----------
    ap_w: aperture width
    z_: Screen distance
    eps_abs: the quad epsabs parameter
    eps_rel: the quad epsrel parameter
    limit_: the quad limit parameter
    
    Returns
    -------
    intensity_2darr: The intensities at each screen coordinate 
    """
    fres_nr = fresnel_nr_func(z, aperture_width)
    screen_width = get_screen_width(fres_nr)
    sx_arr, sy_arr = arrays(screen_width, N_2d)
    const = k*E_0/(2*np.pi*z) #constant in front of simplified Fresnel integral

    ay_min = -aperture_width/2 #lower aperture y-limit integral
    ay_max = aperture_width/2 #lower aperture y-limit integral
    
    intensity_2darr = np.zeros((N_2d, N_2d))

    for j in range(N_2d): 
        for i in range(N_2d): 
            real2dpart, real2derror = dblquad(real_2dkernel, ay_min, ay_max, ax_min, ax_max, args = (sy_arr[j], sx_arr[i], k, z))
            imag2dpart, imag2derror = dblquad(imag_2dkernel, ay_min, ay_max, ax_min, ax_max, args = (sy_arr[j], sx_arr[i], k, z))
            mod_E_squared = const**2*(real2dpart**2 + imag2dpart**2)
            intensity_2darr[i, j] = perm*c*mod_E_squared
            Printer(str(j/N_2d*100) + " %")

    return intensity_2darr
   
##################################################################################################
#PART D FUNCTIONS
def MonteCarlo_integration(MCsamples): 
    """
    2D integration of the  real and imaginary parts of the Fresnel integral (evaluation at each point of the 2D screen)
    Does so using the Monte Carlo integration method. 

    Parameters
    ----------
    MCsamples: Number of random aperture coordinates to generate at each point in the screen
    
    Returns
    -------
    intensity_2darr: The intensities at each screen coordinate 
    mean_sterror: Mean standard error of the mean for the integration
    mean_abserror: Mean absolute error relative to the dblquad integration method 

    """
    intensity_2darr = np.zeros((N_2d, N_2d))
    intensity_2darr_quad = np.zeros((N_2d, N_2d))
    error_2darr = np.zeros((N_2d, N_2d))

    radius = aperture_width/2
    fres_nr = fresnel_nr_func(z, aperture_width)
    screen_width = get_screen_width(fres_nr)
    sx_arr, sy_arr = arrays(screen_width, N_2d)
    const = k*E_0/(2*np.pi*z)
    
                    
    for j in range(N_2d): 
        for i in range(N_2d): 
            real_integral_vals = np.zeros(MCsamples)
            imag_integral_vals = np.zeros(MCsamples)

            ax_vals = np.random.uniform(-radius, radius, MCsamples)
            ay_vals = np.random.uniform(-radius, radius, MCsamples)
            for s in range(MCsamples):
                ax_rand = ax_vals[s]
                ay_rand = ay_vals[s]
                if np.sqrt((ax_rand**2) + (ay_rand**2)) <= radius:
                #only evaluates the integrals for those aperture-coordinates within the radius
                    real_integral_vals[s] = real_2dkernel(ay_rand, ax_rand, sy_arr[j], sx_arr[i], k, z)
                    imag_integral_vals[s] = imag_2dkernel(ay_rand, ax_rand, sy_arr[j], sx_arr[i], k, z)

                
            #MC integral
            real_f_avg = real_integral_vals.sum()/MCsamples
            imag_f_avg = imag_integral_vals.sum()/MCsamples
            A = (2*radius)**2
            mod_E_squared = A**2*const**2*(real_f_avg**2 + imag_f_avg**2)
            intensity_2darr[i, j] = perm*c*mod_E_squared
            

            #standard error in the mean computation
            mean_fsq_real = (real_integral_vals**2).sum()/MCsamples
            sq_mean_real = real_f_avg**2
            real_f_avg_error = np.sqrt((mean_fsq_real - sq_mean_real)/MCsamples)

            mean_fsq_imag = (imag_integral_vals**2).sum()/MCsamples
            sq_mean_imag = imag_f_avg**2
            imag_f_avg_error = np.sqrt((mean_fsq_imag - sq_mean_imag)/MCsamples)

            combined_error= np.sqrt(real_f_avg_error ** 2 + imag_f_avg_error ** 2)
            error_2darr[i, j] = combined_error

            #computing the dblquad integral to compare
            real2dpart, real2derror = dblquad(real_2dkernel, -radius, radius, -radius, radius, args = (sy_arr[j], sx_arr[i], k, z))
            imag2dpart, imag2derror = dblquad(imag_2dkernel, -radius, radius, -radius, radius, args = (sy_arr[j], sx_arr[i], k, z))
            mod_E_squared_quad = const**2*(real2dpart**2 + imag2dpart**2)
            intensity_2darr_quad[i, j] = perm*c*mod_E_squared_quad

            #absolute error with dblquad computation
            abserror_2darr= abs(intensity_2darr_quad - intensity_2darr)

            Printer(str(j/N_2d*100) + " %")

    mean_abserror = np.mean(abserror_2darr)
    mean_sterror = np.mean(error_2darr)

        
    return intensity_2darr, mean_sterror, mean_abserror


#########################################################################################################################################################
# Main program

MyInput = '0'
while MyInput != 'q':
    try:
        MyInput = input('Enter a choice, "a", "b", "c", "d", or "q" to quit: ')
        if MyInput not in ['a', 'b', 'c', 'd', 'q']:
            raise ValueError("Invalid choice. Please enter a valid option.")
    except ValueError as e:
        print(f"Error: {e}")
        continue

########################################################################################################################
# Part a
    if MyInput == 'a':
        print('You have chosen part (a)')
        intensity_arr, rmse = quad_integration(aperture_width, z)

        #table for overview of used values
        fres_nr = fresnel_nr_func(z, aperture_width)
        screen_width = get_screen_width(fres_nr)
        print("Calculating the 1D diffraction for following values:")
        print(f"{'N':<20}{'Aperture width (m)':<20}{'Distance to screen (m)':<20}{'Fresnel number':<20}{'Screen width (m)':<20}")
        print("-" * 140)
        print(f"{N_1d:<20}{aperture_width:<20}{z:<20}{fres_nr:<20.5f}{screen_width:<20}")

        #calculates approximate transition points from far-field to near-field effect 
        #for the default values of z and aperture width when the other is sweeped. 
        z_arr = np.linspace(0.00002, 0.02, 1*10**(5))
        for el in z_arr: 
            z_el = el 
            fresnelnr = fresnel_nr_func(z_el, 2*10**(-5))
            if fresnelnr >= 0.5: 
                trans_z = el 
                break

        ap_arr = np.linspace(0.00001, 0.0008, 1*10**(5))
        for el in ap_arr: 
            ap_el = el 
            fresnelnr = fresnel_nr_func(0.02, ap_el)
            if fresnelnr >= 0.5: 
                trans_ap = el 
                break

        print("With the given values, at fix λ and aperture width, near-field effects (Fresnel diffraction) occur approximately at or below a screen distance of {:.6f} m.".format(trans_z))
        print("With the given values, at fix λ and screen distance, near-field effects (Fresnel diffraction) occur approximately at or above an aperture width of {:6f} m.".format(trans_ap))

        #options to plot results
        while True: 
            p_input = input('Type "plot1" if you want to plot the 1d diffraction pattern, "plot2" to see how the central peak width varies with aperture width and screen distance, "plot3" to see how the RMSE of the integration varies with quad parameters, or press Enter to skip.')
            if p_input == "plot1":
                central_width, central_width_height = central_peak_width(z, aperture_width, intensity_arr)
                plot_1d_fresnel(z, aperture_width, intensity_arr, central_width_height, central_width)
        

            elif p_input == "plot2":

                #Sweeps values for aperture width and screen distance with the other held fixed. 
                central_width_aplist = []
                mean_i_aplist = []
                ap_arr = np.linspace(0.00001, 0.0008, 100)
                for el in ap_arr: 
                    ap_el = el
                    intensity_arr_ap, rmse = quad_integration(ap_el, 0.02)
                    central_width, _ = central_peak_width(0.02, ap_el, intensity_arr_ap)
                    central_width_aplist.append(central_width)
                    mean_intensity_ap= np.mean(intensity_arr_ap)
                    mean_i_aplist.append(mean_intensity_ap)

                central_width_zlist = []
                mean_i_zlist = []
                z_arr = np.linspace(0.00002, 0.02,  100)
                for i in z_arr: 
                    z_el = i
                    intensity_arr_z, rmse = quad_integration(2*10**(-5), z_el)
                    central_width, _= central_peak_width(z_el, 2*10**(-5), intensity_arr_z)
                    central_width_zlist.append(central_width)
                    mean_intensity_z = np.mean(intensity_arr_z)
                    mean_i_zlist.append(mean_intensity_z)

                plot_4varyingpar(ap_arr, z_arr, central_width_aplist, central_width_zlist, mean_i_aplist, mean_i_zlist, "Aperture width (m)", "Screen distance (m)")

            
            elif p_input == "plot3":
                rmse_list = []
                
                epsrel_arr = np.logspace(-10, 2, 30)
                epsabs_arr = np.logspace(-10, -4.5, 30)
                lim_arr = np.linspace(1, 30, 30)
                
                #Analyses the integration RMSE as a functin of epsrel, apsabs and limit.
                rmse_rellist = []
                for el in epsrel_arr: 
                    epsrel = el
                    intensity_arr, d_rmse = quad_integration(2*10**(-5), 0.00002, eps_rel = epsrel)
                    rmse_rellist.append(d_rmse)

                
                rmse_abslist = []
                for el in epsabs_arr: 
                    epsabs = el
                    intensity_arr, d_rmse = quad_integration( 2*10**(-5), 0.00002, eps_abs = epsabs)
                    rmse_abslist.append(d_rmse)
                
                rmse_limlist = []
                for el in lim_arr: 
                    limit = int(el)
                    intensity_arr, d_rmse = quad_integration( 2*10**(-5), 0.00002, limit_ = limit)
                    rmse_limlist.append(d_rmse)

                plot_3varyingpar(epsrel_arr, epsabs_arr, lim_arr, rmse_rellist, rmse_abslist, rmse_limlist, "quad - epsrel", "quad - epsabs","quad -limit")
                
                pass
            
            elif p_input == "":
                break
            else:
                print("That's not a valid option, try again")
        
##################################################################################################
#Part b
    elif MyInput == 'b':
        print('You have chosen part (b)')

        #table for overview of used values
        fres_nr = fresnel_nr_func(z, aperture_width)
        screen_width = get_screen_width(fres_nr)
        print("Calculating the 1D diffraction for following values:")
        print(f"{'N':<20}{'Aperture width (m)':<20}{'Distance to screen (m)':<20}{'Fresnel number':<20}{'Screen width (m)':<20}")
        print("-" * 140)
        print(f"{N_2d:<20}{aperture_width:<20}{z:<20}{fres_nr:<20.5f}{screen_width:<20}")

        while True: 
            p_input = input('Type "plot1" if you want to plot 2d diffractionfrom a square aperture, "plot2" from a rectangular aperture, or press Enter to skip:')
            x_wid = aperture_width
            if p_input == "plot1":
                intensity_2darr= dblquad_integration(-x_wid/2, x_wid/2)
                plot_2d_fresnel(intensity_2darr, "square")
            elif p_input == "plot2":
                intensity_2darr= dblquad_integration( -x_wid, x_wid)
                plot_2d_fresnel(intensity_2darr, "rectangular")
            elif p_input == "":
                break
            else:
                print("That's not a valid option, try again")

##################################################################################################
#Part c
    elif MyInput == 'c':
        print('You have chosen part (c)')

        #table for overview of used values
        fres_nr = fresnel_nr_func(z, aperture_width)
        screen_width = get_screen_width(fres_nr)
        print("Calculating the 1D diffraction for following values:")
        print(f"{'N':<20}{'Aperture width (m)':<20}{'Distance to screen (m)':<20}{'Fresnel number':<20}{'Screen width (m)':<20}")
        print("-" * 140)
        print(f"{N_2d:<20}{aperture_width:<20}{z:<20}{fres_nr:<20.5f}{screen_width:<20}")

        while True: 
            p_input = input('Type "plot1" if you want to plot 2d diffraction from a circular aperture, or press Enter to skip.')
            if p_input == "plot1":
                intensity_2darr= dblquad_integration( ax_min_func, ax_max_func)
                plot_2d_fresnel(intensity_2darr, "circle")

            elif p_input == "":
                break
            else:
                print("That's not a valid option, try again")

##################################################################################################
#Part d
    elif MyInput == 'd':
        #table for overview of used values
        fres_nr = fresnel_nr_func(z, aperture_width)
        screen_width = get_screen_width(fres_nr)
        print("Calculating the 1D diffraction for following values:")
        print(f"{'MC_samples':<20}{'Aperture width (m)':<20}{'Distance to screen (m)':<20}{'Fresnel number':<20}{'Screen width (m)':<20}")
        print("-" * 140)
        print(f"{MC_samples:<20}{aperture_width:<20}{z:<20}{fres_nr:<20.5f}{screen_width:<20}")

        print('You have chosen part (d)')
        while True: 
            p_input = input('Type "plot1" if you want to plot the 2d diffraction via the Monte Carlo integration, "plot2" to see how the Error(s) vary with the number of samples (takes a few mins), or press Enter to skip.')
            if p_input == "plot1":
                intensity_2darr, mean_st_error,  mean_abs_error = MonteCarlo_integration(MC_samples)
                plot_2d_fresnel(intensity_2darr, "MCcircle")
            elif p_input == "plot2":

                #Sweeps values for the number of MC samples 
                #and analyses the mean standard error and mean absolute error (with dblquad)
                MC_samples_arr = np.linspace(10, 1000, 30)
                mean_sterror_list = []
                mean_abserror_list = []
                for el in MC_samples_arr: 
                    sample_nr = int(el)
                    Printer(str(sample_nr))
                    intensity_2darr, mean_st_error, mean_abs_error= MonteCarlo_integration(sample_nr)
                    mean_sterror_list.append(mean_st_error)
                    mean_abserror_list.append(mean_abs_error)

                plot_1varyingpar(MC_samples_arr, mean_sterror_list, "MC_samples", "Mean MSE", "Mean Standard Error in the Mean")
                plot_1varyingpar(MC_samples_arr, mean_abserror_list, "MC_samples", "Mean AE" , "Mean Absolute Error relative to dblquad method")

            elif p_input == "":
                break
            else:
                print("That's not a valid option, try again")
