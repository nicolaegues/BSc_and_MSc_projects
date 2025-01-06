

#include <iostream>
#include <cmath>
#include <vector>
#include <tuple>


double G = 6.67e-11; // m^3*kg^-1*s^-2, Gravitational constant


extern "C"{ ////this extern keyword is for the python wrapper - (as python is written in C)


/**
 * @brief Calculates the gravitational acceleration acting on a body due to all other bodies in the system.
 * 
 * The gravitational acceleration is calculated using Newton's law of gravitation based on the positions of all bodies and their masses.
 * 
 * @param i The index of the body for which the acceleration is calculated.
 * @param positions A reference to a 2D vector that contains the positions of all bodies in the system (for a given time-step).
 *                  Each inner vector holds the 3D coordinates of a body. The size of the outer vector is the number of bodies.
 * @param masses A pointer to an array that contains the masses of the bodies.
 * @return A 1D vector of size 3 that contains the x, y, z components of the acceleration acting on body i. 
 * 
 */
std::vector<double> grav_acceleration(int i, std::vector<std::vector<double>>& positions, double* masses){

    int nr_bodies = positions.size(); //the number of bodies in the system

    std::vector<double> acceleration(3, 0.0); //initialise 3D acceleration vector with 0's, to add the contributions from each body later.

    for (int j = 0; j < nr_bodies; j++){ //Iterate through the bodies (to retrieve their positions), excluding the body (i) whose acceleration we're calculating.

        if (i != j){

            std::vector<double> r_diff(3); //vector to hold body i's and body j's 3D difference.
 
            for (int k = 0; k <3; k++){ //loop through the x, y, z, components of the body's positions
                r_diff[k] = positions[i][k] - positions[j][k];
            }

            //get the distance between the two bodies
            double r_diff_norm = std::sqrt((r_diff[0]*r_diff[0] )+ (r_diff[1]*r_diff[1]) + (r_diff[2] *r_diff[2]));

            //Apply Newton's law of gravitation, summing the accelerations for each vector component.
            for (int k = 0; k <3; k++){
                acceleration[k] += - G*masses[j]*r_diff[k]/(r_diff_norm*r_diff_norm*r_diff_norm);
            }
        }
    }

    return acceleration;
}


/**
 * @brief Helper functino to calculate the updated position of a body as part of the Velocity Verlet integration method. 
 * 
 * @param r0 A vector of size 3 that holds the current 3D position of the body. 
 * @param v0 A vector of size 3 that holds the current 3D velocity of the body. 
 * @param a0 A vector of size 3 that holds the current 3D acceleration of the body. 
 * @param dt The timestep size for the integration.
 * 
 * @return A vector of size 3 that holds the new 3D position vector of the body.
 */
std::vector<double> new_verlet_positions(std::vector<double> r0, std::vector<double> v0, std::vector<double> a0, double dt){

    std::vector<double> new_pos(3);


    for (int i = 0; i< 3; i++){
        new_pos[i] = r0[i] + v0[i]*dt + 0.5*a0[i]*(dt*dt);
    }

    return new_pos;

}


/**
 * @brief Helper functino to calculate the updated velocity of a body as part of the Velocity Verlet integration method. 
 * 
 * @param v0 A vector of size 3 that holds the current 3D velocity of the body. 
 * @param a0 A vector of size 3 that holds the current 3D acceleration of the body. 
 * @param a_new A vector of size 3 that holds the updated 3D acceleration of the body. 
 * @param dt The timestep size for the integration.
 * 
 * @return A vector of size 3 that holds the new 3D velocity vector of the body.
 */
std::vector<double> new_verlet_velocities(std::vector<double> v0, std::vector<double> a0, std::vector<double> a_new, double dt){

    std::vector<double> new_vel(3); //initialise vector of size 3 to hold the new volicity

    for (int i = 0; i< 3; i++){
        new_vel[i] = v0[i] + 0.5*dt*(a0[i] + a_new[i]);
    }

    return new_vel;

}

/**
 * @brief Calculates the kinetic, potential, and total energy of a given body N-body system 
 *        from stored positions and velocities. Also calculates the percentage error in the total energy.
 * 
 * @param nr_bodies The number of bodies in the system.
 * @param nr_steps The number of timesteps in the simulation.
 * @param positions A pointer to an array containing the positions of all bodies over time.
 *                   (with positions stored sequentially for each body and timestep)
 * @param velocities A pointer to an array containing the velocities of all bodies over time.
 * @param masses A pointer to an array containing the masses of all bodies. 
 * @param KE A pointer to an array where the kinetic energy values of system will be storef
 * @param PE A pointer to an array where the potential energy values of the system will be stored.
 * @param TE A pointer to an array where the total energy values of the system will be stored.
 * @param TE_error A pointer to an array where the percentage error in total energy (relative to the initial total energy) at each timestep will be stored. 
 *              
 */
void energy_calc(int nr_bodies, int nr_steps, double* positions, double* velocities, double* masses, double* KE, double* PE, double* TE, double* TE_error){
    
    for (int step = 0; step < nr_steps; step++){

        double total_ke = 0.0;
        double total_pe = 0.0;
    
        for (int i = 0; i < nr_bodies; i++){
            
            double vel_vec_sq = 0.0;

            for (int j = 0; j<3; j++){

                double vel = velocities[(i*nr_steps + step)*3 + j];
                vel_vec_sq += vel*vel;
            }

            double ke =  0.5*masses[i]*vel_vec_sq;
            total_ke += ke;

        }

        for (int i = 0; i < nr_bodies; i++) {
            for (int j = i + 1; j < nr_bodies; j++) { //to make sure we get unique pairs
                
                double pos_vec_sq = 0.0;

                for (int k = 0; k < 3; k++) {
                    double pos_i = positions[(i * nr_steps + step) * 3 + k];
                    double pos_j = positions[(j * nr_steps + step) * 3 + k];
                    double diff = pos_i - pos_j;
                    pos_vec_sq += diff * diff;
                }

                double distance = std::sqrt(pos_vec_sq);
                double pe = (-G* masses[i]*masses[j])/distance;

                total_pe += pe;
            }
        }

        double total_te = total_ke + total_pe;

        KE[step]= total_ke;
        PE[step] = total_pe;
        TE[step] = total_te;

        //std::cout<<ke<<", "<<pe<<", "<<te<<std::endl;
        double perc_error = std::abs(total_te - TE[0])/std::abs(TE[0]);
        //std::cout<<perc_error<<std::endl;
        TE_error[step] = perc_error;
    }
   
}

/**
 * @brief Computes the orbital eccentricity of a body's orbit around a second body.
 * 
 * @param positions A pointer to an array containing the positions of all bodies over time (stored sequentially)
 *                
 * @param body_1_nr The index of the first body - the host star
 * @param body_2_nr The index of the second body - the orbiting body
 * @param nr_steps The number of timesteps in the simulation.
 * 
 * @return The orbital eccentricity of the body. 
 */
double eccentricity(double* positions, int body_1_nr, int body_2_nr, int nr_steps){

    double r_a = 0.0;
    double r_p = std::numeric_limits<double>::infinity(); 

    for (int step = 0; step < nr_steps; step++){

        double distance_sq = 0.0;
    
        for (int i = 0; i<3; i++){

            auto body_1_pos = positions[(body_1_nr*nr_steps + step)*3 + i];
            auto body_2_pos = positions[(body_2_nr*nr_steps + step)*3 + i];

            double dr = body_1_pos - body_2_pos;
            distance_sq += dr*dr;
        }
        
        double distance = std::sqrt(distance_sq);
       
    
        r_a = std::max(r_a, distance);
        r_p = std::min(r_p, distance);

    }

    double e = (r_a - r_p) / (r_a + r_p);

    return e;

}


/**
 * @brief Simulates the motion of a system of N bodies using the Velocity Verlet integration method.
 * 
 * @param nr_bodies The number of bodies in the system.
 * @param masses A pointer to an array that contains the masses of the bodies
 * @param initial_positions A pointer to an array that contains the initial 3D positions 
 *                          of the bodies. They're stored such that the body's (x, y, z) components are arranged sequentially.
 * @param initial_velocities A pointer to an array that contains the initial velocities 
 *                          of the bodies - vx, vy, and vz components of the bodies arranged sequentially.
 * @param nr_steps The number of steps the Velocity Verlet algorithm will perform
 * @param dt The size of each time-step
 * @param positions_vec A pointer to an array where the computed positions of the 
 *                      bodies will be stored. Stored in flattned manner such that body 1's positions (of length nr_steps, with flattened x, y, z components) appear first, 
 *                      then body 2's etc... 
 * @param velocities_vec A pointer to an array where the computed velocities of the 
 *                       bodies will be stored in the same manner as in the positions_vec.
 * 
 */
void velocity_verlet(int nr_bodies, double* masses, double* initial_positions, double* initial_velocities, int nr_steps, double dt, 
                     double* positions_vec, double* velocities_vec){

    //Initialise vectors for the 3D positions, velocities and accelerations of all bodies. Are updated at each time-step.
    std::vector<std::vector<double>> positions(nr_bodies, std::vector<double>(3));
    std::vector<std::vector<double>> velocities(nr_bodies, std::vector<double>(3));
    std::vector<std::vector<double>> accelerations(nr_bodies, std::vector<double>(3));

    //Fill the position and velocity vectors with the initial parameters of each body.
    for (int i = 0; i < nr_bodies; i++) {
        for (int j = 0; j < 3; j++){
            positions[i][j] = initial_positions[i*3 + j]; 
            velocities[i][j] = initial_velocities[i*3 + j];  
        }
    }

    //For every body, fill the vector with the 3D acceleration vector resulting from the grav. forces of the remaining bodies in the system. 
    for (int i = 0; i < nr_bodies; i++) {
        accelerations[i] = grav_acceleration(i, positions, masses);
    }


    //Store positions, velocities and accelerations 
    for (int i = 0; i < nr_bodies; i++) {
        for (int j = 0; j < 3; j++) {
            positions_vec[(i*nr_steps)*3 +j] = positions[i][j];
            velocities_vec[(i*nr_steps)*3 +j] = velocities[i][j];
        }
    }

    
    for (int step=1; step< nr_steps; step++) {

        //Store the accelerations from the last time-step to use for the velocity-update
        std::vector<std::vector<double>> old_accelerations = accelerations;

        //Update positions
        for (int i=0; i< nr_bodies; i++) {
       
            positions[i] = new_verlet_positions(positions[i], velocities[i], accelerations[i], dt);
        }

        //Update accelerations
        for (int i=0; i< nr_bodies; i++) {
            accelerations[i] = grav_acceleration(i, positions, masses);
        }


        //Update velocities
        for (int i= 0; i< nr_bodies; i++) {
            velocities[i] = new_verlet_velocities(velocities[i], old_accelerations[i], accelerations[i], dt);
        }

        //Store positions, velocities and accelerations
        for (int i= 0; i< nr_bodies; i++) {
            for (int j = 0; j < 3; j++) {
                positions_vec[(i*nr_steps+step)*3 + j] = positions[i][j];
                velocities_vec[(i*nr_steps+step)*3 + j] = velocities[i][j];
            }
        }
    }
}


/**
 * @brief Computes the eccentricity of a planet's orbit as a function of the Sun's mass.
 * 
 * @param n The total number of bodies in the system.
 * @param masses A pointer to an array containing the masses of the bodies in the system. 
 * @param initial_positions A pointer to an array containing the initial positions of the bodies. 
 * @param initial_velocities A pointer to an array containing the initial velocities of the bodies.
 * @param nr_steps The number of timesteps in the simulation.
 * @param dt The time step for the simulation in seconds.
 * @param positions_vec A pointer to an array where the positions of all bodies at each timestep will be stored.
 * @param velocities_vec A pointer to an array where the velocities of all bodies at each timestep will be stored.
 * @param body_1_nr The index of the first body  (host star)
 * @param body_2_nr The index of the second body  (orbiting planet)
 * @param nr_evals The number of different Sun mass fractions to evaluate.
 * @param eccentricities A pointer to an array where the computed eccentricities for each Sun mass fraction 
 *                       will be stored
 * @param sun_masses A pointer to an array where the corresponding Sun mass fractions will be stored. 
 */
void eccentricity_vs_sun_mass(int n, double* masses, double* initial_positions, double* initial_velocities, int nr_steps, double dt, 
                     double* positions_vec, double* velocities_vec, 
                            int body_1_nr, int body_2_nr, int nr_evals, double* eccentricities, double* sun_masses){

    double stepsize = (1.3 - 0.7)/(nr_evals-1);

    double og_sun_mass = masses[0];
    for (int i =  0; i <nr_evals; i++){

        double Ms_frac = (0.7 + i*stepsize);
        masses[0] = og_sun_mass*Ms_frac;
        velocity_verlet(n, masses, initial_positions, initial_velocities, nr_steps, dt, 
                     positions_vec, velocities_vec);

        double ecc = eccentricity(positions_vec, body_1_nr, body_2_nr, nr_steps);

        eccentricities[i] = ecc;
        sun_masses[i] = Ms_frac;
    }

    masses[0] = og_sun_mass;

}

}


int main(){

    return 0;
}

