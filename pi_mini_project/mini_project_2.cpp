#define _USE_MATH_DEFINES //without including this M_PI is not recognized
#include <cmath>
#include <iostream>
#include <iomanip>
#include <tuple>
#include <vector>
#include <ctime>
#include <omp.h>



// The function we wish to integrate
double f(double x){

    return pow(x, 4)*pow((1-x), 4)/(1 + pow(x, 2));
    
}


/** 
 * @brief Calculates the numerical integral of a function using the Midpoint (or the "Leftopoint") rule. Is parallelisable.
 * 
 * Midpoint rule: The given interval is divided into subintervals of a certain width (spacing). 
 * At the midpoint of each subinterval the function value is calculated and multiplied by the spacing to give the area of a rectangle. 
 * All of these areas are summed to give an estimate of the integral.
 * Leftpoint rule: Same logic, but instead of midpoints, the left bounds of each subinterval are used.
 * If a thread number >1 is chosen, this computation is parallelised with OpenMP.

 * 
 * @param lower_b lower bound of the integration interval
 * @param upper_b upper bound of the integration interval
 * @param spacing the spacing of each subinterval (so spaing between evaluated functino points)
 * @param start where the first function point will be evaluated - if spacing = lower_bound, then this turns into the "leftpoint" method. 
 *              alternatively, if start = lower_bound + spacing/2, this turns into the midpoint method.
 * @param nr_of_threads the number of threads to use for parallelisation
    
 * @return The estimated integral
 */
double midpoint_rule(double lower_b, double upper_b, double spacing, double start, int nr_of_threads){

    double integral = 0.0;
    int nr_intervals = (upper_b - lower_b)/spacing; 

    #pragma omp parallel for num_threads(nr_of_threads) reduction(+:integral)
    //Iterate through all the midpoints (or "leftpoints"), making sure to stop before the upper bound is reached.
    
    for (int i = 0; i < nr_intervals; i ++){ 
        double x = start + i*spacing;
        integral += f(x);
    }

    integral *= spacing; //this is equivalent to multiplying f(i) with the spacing each time in the loop

    return integral;

}


/** 
 * @brief Calculates the numerical integral of a function using the trapezoidal rule. Is parallelisable.
 * 
 * Trapezoidal rule: The integration interval is divided into subintervals. 
 * For each subinterval, the area of a trapzeoid under the function is calculated - all of these are summed to give the integral estimate.
 * The corresponding equation is: integral = spacing/2* (first point + 2*(sum of rest of points)+ last point)
 * 
 * @param lower_b lower bound of the integration interval
 * @param upper_b upper bound of the integration interval
 * @param spacing the spacing of each subinterval (so spaing between evaluated functino points)
 * @param nr_of_threads the number of threads to use for parallelisation
    
 * @return The estimated integral
 */
double trapezoidal_rule(double lower_b, double upper_b, double spacing, int nr_of_threads){

    double integral = 0.0;
    int nr_intervals = (upper_b - lower_b)/spacing;

    integral += f(lower_b) + f(upper_b); //the bounds are included, but not multiplied by 2 (as is done below)

    #pragma omp parallel for num_threads(nr_of_threads) reduction(+:integral)
    for (int i = 1; i < nr_intervals; i++ ){ // Starts with i = 1 given that the interval bounds are included.
        double x = lower_b + i*spacing; 
        integral += f(x)*2;
    }

    integral *= spacing/2; 

    return integral;

}

/** 
 * @brief Calculates the numerical integral of a function using Simpson's rule. Is parallelisable.
 * 
 * Simpson's rule: The integration interval is divided into subintervals (which must be an even number!)
 * Each pair of subintervals gives 3 points (the endpoints, and the midpoint), allowing for a parabola to be fitted for each pair.
 * Using this method, it can be proven that calculating the area amounts to calucalating following equation: 
 * integral = spacing/3 * (first point + 4(sum of odds) + 2*(sum of evens) + last point)
 * 
 * @param lower_b lower bound of the integration interval
 * @param upper_b upper bound of the integration interval
 * @param spacing the spacing of each subinterval (so spaing between evaluated functino points)
 * @param nr_of_threads the number of threads to use for parallelisation
    
 * @return The estimated integral
 */
double simpsons_rule(double lower_b, double upper_b, double spacing, int nr_of_threads){

    double integral = 0.0;
    int nr_intervals = (upper_b - lower_b)/spacing;

    //the number of intervals needs to be even: With our bounds (0 and 1) this is true, but to handle
    //cases where they might be odd, the last interval is removed.
    if (nr_intervals % 2 != 0){
        nr_intervals--;
    }

    integral += f(lower_b) + f(upper_b); //the function values at the bounds are weighted by 1 

    //the odd intervals are weighted by 4
    #pragma omp parallel for num_threads(nr_of_threads) reduction(+:integral)
    for (int i = 1; i < nr_intervals; i += 2){ 
        auto x = lower_b + i*spacing;
        integral += f(x)*4;
    }

    //the even intervals are weighted by 2
    #pragma omp parallel for num_threads(nr_of_threads) reduction(+:integral)
    for (int i = 2; i < nr_intervals; i+= 2){ 
        auto x = lower_b + i*spacing;
        integral += f(x)*2;
    }

    integral *= spacing/3; 

    return integral;

}


/**
 * @brief Determines the required subinterval spacing for the Midpoint rule to estimate pi to 10 decimal places.
 * 
 * It is known that the integral of this particular function between 0 and 1 equals 22/7-pi. We can therefore estimate pi by estimating the integral.
 * 
 * An initial spacing is reduced iteratively (by 10% in each iteration), calculating the integral using each spacing.
 * The loop continues until the absolute difference between the estimated Pi and the correct value of pi is less than or equal to the amount specified by "diff_thresh".
 * After the loop finishes (i.e. the spacing is found), the estimated Pi value, the correct value and the absolute error are printed, 
 * and the spacing returned.
 * 
 * @param lower_b lower bound of the integration interval
 * @param upper_b upper bound of the integration interval
 * @return the spacing required to estimate pi to 10 decimal places.
 */
double get_spacing_for_10decimals(double lower_b, double upper_b){

    double diff_thresh = 1e-10; //the loop below only stops once the absolute difference between the estimated value of pi and pi is at least this small.
    double spacing = 0.5; //The initial spacing between the midpoints. 
    double pi_estimate; //initialise variable for our pi estimate.
    double abs_diff = 1.0; //Initial absolute difference - start with high number to start the loop.

    while (abs_diff > diff_thresh){ //this loop will break once the absolute difference is equal or lower to the specified threshold.

        spacing *= 0.9; //reduce the spacing in each iteration
        double mid_start  = lower_b + spacing/2;
        double integral = midpoint_rule( lower_b, upper_b, spacing, mid_start, 1);
        pi_estimate = 22.0/7.0 - integral;
        abs_diff = std::abs(M_PI-pi_estimate);
        }
    
    double nr_intervals = (upper_b-lower_b)/spacing;
    std::cout<<"A spacing of: " << spacing << " or equivalently "<< nr_intervals<<" x-values are required to estimate pi to 10 decimal places. "<<std::endl; 
    std::cout<<"The estimate is: " << std::setprecision(10)<<pi_estimate<< ", the correct value is: "<<std::setprecision(10)<<M_PI <<", and the absolute difference is: "<<abs_diff<<std::endl;

    return spacing;
}

/**
 * @brief Calculates the mean integrals and executions times for all numerical integration methods. 
 * @param lower_b The lower bound of the integration interval.
 * @param upper_b The upper bound of the integration interval.
 * @param spacing The spacing between evaluation points used in the integration methods.
 * @return A tuple containing the following values:
 * - mean_mid_integral: The mean integral calculated using the Midpoint Rule.
 * - mean_mid_time: The mean execution time for the Midpoint Rule.
 * - mean_left_integral: The mean integral calculated using the Leftpoint Rule.
 * - mean_left_time: The mean execution time for the Leftpoint Rule.
 * - mean_trap_integral: The mean integral calculated using the Trapezoidal Rule.
 * - mean_trap_time: The mean execution time for the Trapezoidal Rule.
 * - mean_simp_integral: The mean integral calculated using Simpson's Rule.
 * - mean_simp_time: The mean execution time for Simpson's Rule.
*/
std::tuple<double, double, double, double, double, double, double, double> mean_integrals(double lower_b, double upper_b, double spacing){

    auto mid_start  = lower_b + spacing/2; //first point for midpoint rule
    auto left_start  = lower_b; //first point for leftpoint rule
    int max_iter = 10; //how many times to run the integrations for to calculate the mean values.

    double mean_mid_integral = 0.0;
    double mean_left_integral = 0.0;
    double mean_trap_integral = 0.0;
    double mean_simp_integral = 0.0;

    double mean_mid_time = 0.0;
    double mean_left_time = 0.0;
    double mean_trap_time = 0.0;
    double mean_simp_time = 0.0;

    for (int i = 0; i<max_iter; i++){
        auto mid_t1 = omp_get_wtime();
        mean_mid_integral += midpoint_rule( lower_b, upper_b, spacing, mid_start, 1);
        auto mid_t2 = omp_get_wtime();
        mean_mid_time += mid_t2 - mid_t1;

        auto left_t1 = omp_get_wtime();
        mean_left_integral += midpoint_rule( lower_b, upper_b, spacing, left_start, 1);
        auto left_t2 = omp_get_wtime();
        mean_left_time += left_t2 - left_t1;

        auto trap_t1 = omp_get_wtime();
        mean_trap_integral += trapezoidal_rule( lower_b, upper_b, spacing, 1);
        auto trap_t2 = omp_get_wtime();
        mean_trap_time += trap_t2 - trap_t1;

        auto simp_t1 = omp_get_wtime();
        mean_simp_integral += simpsons_rule( lower_b, upper_b, spacing, 1);
        auto simp_t2 = omp_get_wtime();
        mean_simp_time += simp_t2 - simp_t1;

    }
    mean_mid_integral /= max_iter;
    mean_left_integral /= max_iter;
    mean_trap_integral /= max_iter;
    mean_simp_integral /= max_iter;

    mean_mid_time /= max_iter;
    mean_left_time /= max_iter;
    mean_trap_time /= max_iter;
    mean_simp_time /= max_iter;

return {mean_mid_integral, mean_mid_time, mean_left_integral, mean_left_time, mean_trap_integral, mean_trap_time, mean_simp_integral, mean_simp_time};

}

/** 
 * @brief Computes the absolute errors of all numerical integration methods for a range of different spacings, and stores them in vectors. 
 * @param lower_b The lower bound of the integration interval.
 * @param upper_b The upper bound of the integration interval.
 * @param correct_integral The correct value of the integral
 * @return A tuple of five vectors:
 * - spacings: A vector of the spacing values used in the calculations.
 * - mid_abs_errors: A vector of the absolute errors for the Midpoint Rule.
 * - left_abs_errors: A vector of the absolute errors for the Leftpoint Rule.
 * - trap_abs_errors: A vector of the absolute errors for the Trapezoidal Rule.
 * - simp_abs_errors: A vector of the absolute errors for Simpson's Rule.
*/
std::tuple<std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>> error_vs_spacing_vectors(double lower_b, double upper_b, double correct_integral){

    std::vector<double> mid_abs_errors;
    std::vector<double> left_abs_errors;
    std::vector<double> trap_abs_errors;
    std::vector<double> simp_abs_errors;
    std::vector<double> spacings;

    //The spacing starts at 0.1 and decreases by a factor of 0.8 until reaching a value of 0.0001. For each spacing,
    //the function calculates the integral using each method and computes the absolute error relative to the known correct integral.
    for (double test_spacing = 0.1; test_spacing > 0.0001; test_spacing *= 0.8){

        auto mid_start  = lower_b + test_spacing/2;
        auto left_start = lower_b;

        auto mid_integral = midpoint_rule(lower_b, upper_b, test_spacing, mid_start, 1 );
        auto left_integral = midpoint_rule(lower_b, upper_b, test_spacing, left_start, 1);
        auto trap_integral = trapezoidal_rule(lower_b, upper_b, test_spacing, 1 );
        auto simp_integral = simpsons_rule(lower_b, upper_b, test_spacing, 1 );

        auto mid_abs_error = std::abs(correct_integral - mid_integral);
        auto left_abs_error = std::abs(correct_integral - left_integral);
        auto trap_abs_error = std::abs(correct_integral - trap_integral);
        auto simp_abs_error = std::abs(correct_integral - simp_integral);

        spacings.push_back(test_spacing);
        mid_abs_errors.push_back(mid_abs_error);
        left_abs_errors.push_back(left_abs_error);
        trap_abs_errors.push_back(trap_abs_error);
        simp_abs_errors.push_back(simp_abs_error);

    }

    return {spacings, mid_abs_errors, left_abs_errors, trap_abs_errors, simp_abs_errors};

}

//this extern keyword is for the python wrapper - (as python is written in C)
extern "C" {

/**
 * @brief Computes the absolute errors of all numerical integration methods for a range of different spacings. Takes pointers as input to then store these values.
 * 
 * Based on these pointers to the first elements (first element's memory address), the memory locations of the arrays can be iteratively filled with the calculated values.

 * @param lower_b The lower bound of the integration interval.
 * @param upper_b The upper bound of the integration interval.
 * @param correct_integral The correct value of the integral
 * @param size The number of iterations (spacing values) to compute the errors for.
 * @param spacings A pointer to an array where the spacing values used in each iteration will be stored.
 * @param mid_abs_errors A pointer to an array where the absolute errors for the Midpoint rule will be stored.
 * @param left_abs_errors A pointer to an array where the absolute errors for the Leftpoint rule will be stored.
 * @param trap_abs_errors A pointer to an array where the absolute errors for the Trapezoidal rule will be stored.
 * @param simp_abs_errors A pointer to an array where the absolute errors for the Simpson's rule will be stored.
 */
void error_vs_spacing_pointers( double lower_b, double upper_b, double correct_integral, int size, double* spacings, double* mid_abs_errors, double* left_abs_errors, double* trap_abs_errors, double* simp_abs_errors){

    double first_spacing = 0.1;
    double factor = 0.8;

    for (int i = 0; i < size; i++){
        double test_spacing = first_spacing * std::pow(factor, i);

        auto mid_start  = lower_b + test_spacing/2;
        auto left_start = lower_b;

        auto mid_integral = midpoint_rule( lower_b, upper_b, test_spacing, mid_start , 1);
        auto left_integral = midpoint_rule(lower_b, upper_b, test_spacing, left_start , 1);
        auto trap_integral = trapezoidal_rule(lower_b, upper_b, test_spacing, 1);
        auto simp_integral = simpsons_rule(lower_b, upper_b, test_spacing, 1 );

        auto mid_abs_error = correct_integral - mid_integral;
        auto left_abs_error = correct_integral - left_integral;
        auto trap_abs_error = correct_integral - trap_integral;
        auto simp_abs_error = correct_integral - simp_integral;

        spacings[i] = test_spacing;
        mid_abs_errors[i] = std::abs(mid_abs_error);
        left_abs_errors[i] = std::abs(left_abs_error);
        trap_abs_errors[i] = std::abs(trap_abs_error);
        simp_abs_errors[i] = std::abs(simp_abs_error);

    }

}


/**
 * @brief Calculates the execution time of the midpoint rule for different thread counts parallelising the computation. Takes pointers as input to then store these values in arrays.
 * @param lower_b The lower bound of the integration interval.
 * @param upper_b The upper bound of the integration interval.
 * @param spacing The spacing used in the midpoint method.
 * @param max_threads The maximum number of threads to test.
 * @param all_nr_threads A pointer to an array where the number of threads used in each iteration will be stored.
 * @param times A pointer to an array where the wall time (execution time) for each thread count will be stored.
 */
void time_vs_threads_pointers( double lower_b, double upper_b, double spacing, int max_threads, double* all_nr_threads, double* times){

    auto start  = lower_b + spacing/2;
    for (int i = 0; i < max_threads; i++){

        auto nr_of_threads = i + 1;
        auto start_time_wall = omp_get_wtime();
        auto integral = midpoint_rule( lower_b, upper_b, spacing, start, nr_of_threads);
        auto end_time_wall = omp_get_wtime();;

        auto wall_time = end_time_wall - start_time_wall;

        all_nr_threads[i] = nr_of_threads;
        times[i] = wall_time;
    }

}

}


int main(){

    double lower_b = 0.0; //the lower bound of the integration interval
    double upper_b = 1.0; //the upper bound of the integration interval
    double correct_integral = 22.0/7.0 - M_PI; //the correct value of the integral (RHS of the equation)

    // find spacing required to estimate pi to 10 decimal points (i.e.to get an accurate enough integral)
    double acc_spacing = get_spacing_for_10decimals(lower_b, upper_b);

    //Using this spacing, get the mean integral values and the mean wall times for each method.
    auto [mean_mid_integral, mean_mid_time, mean_left_integral, mean_left_time, mean_trap_integral, mean_trap_time, mean_simp_integral, mean_simp_time] = mean_integrals(lower_b, upper_b, acc_spacing);
    //From the mean integral values get the absolute errors for each method. 
    double mid_abs_error = std::abs(mean_mid_integral-correct_integral);
    double left_abs_error = std::abs(mean_left_integral-correct_integral);
    double trap_abs_error = std::abs(mean_trap_integral-correct_integral);
    double simp_abs_error = std::abs(mean_simp_integral-correct_integral);

    //output formatting
    std::cout<<" "<<std::endl;
    std::cout<<"INTEGRATION ESTIMATES ACROSS METHODS (SAME SPACING, 1 THREAD):"<<std::endl;
    std::cout << std::setw(20) << " " << std::setw(20) << "Midpoint rule"<< std::setw(20) << "Leftpoint rule" << std::setw(20) << "trapezoidal rule"<< std::setw(20) << "Simpson's rule" << std::endl;
    std::cout << std::string(100, '-') << std::endl;
    std::cout << std::setw(20) << "Mean integral" << std::setw(20) << std::setprecision(10) << mean_mid_integral<< std::setw(20) << mean_left_integral<< std::setw(20) << mean_trap_integral<< std::setw(20) << mean_simp_integral<< std::endl;
    std::cout << std::setw(20) << "Abs error" << std::setw(20) << mid_abs_error<< std::setw(20) << left_abs_error<< std::setw(20) << trap_abs_error<< std::setw(20) << simp_abs_error<< std::endl;
    std::cout << std::setw(20) << "Mean wall time (s)" << std::setw(20) << mean_mid_time<< std::setw(20) << mean_left_time<< std::setw(20) << mean_trap_time<< std::setw(20) << mean_simp_time<< std::endl;
    std::cout << std::string(100, '-') << std::endl;

    std::cout << "-----> The Midpoint rule is therefore the most accurate. The Leftpoint rule is slightly less accurate," <<std::endl
     <<"given that it uses the left edge of the interval, slightly over/underestimating the integral (instead of balancing the two.)"<<std::endl
     << " Surprisingly, despite being theoretically the most accurate method, Simpson's rule is also slighlty less accurat (as well as the Trapzeoid rule)."<<std::endl
     <<" Simpson's rule is hereby the fastest and the Midpoint rule the slowest (this is most likely due to the extra overhead resulting from having to calculate the midpoints at each iteration.)"<<std::endl;
     

    //---------------------------------------------------------------------------------------------------------------------

    //Get the absolute errors for a range of different spacings. 
    auto [spacings, mid_abs_errors, left_abs_errors, trap_abs_errors, simp_abs_errors] = error_vs_spacing_vectors(lower_b,  upper_b, correct_integral);

    //output formatting
    std::cout<<std::endl;
    std::cout<<"ABSOLUTE ERROR VS SPACING (1 THREAD):"<<std::endl;
    std::cout << std::setw(20) << "Spacing"<< std::setw(20) << "Midpoint error"<< std::setw(20) << "Leftpoint error"<< std::setw(20) << "trapezoidal error"<< std::setw(20) << "Simpson's error" << std::endl;
    std::cout << std::string(100, '-') << std::endl;

    for (int i = 0; i < spacings.size(); ++i) {
        std::cout << std::setw(20) << spacings[i]<< std::setw(20) << mid_abs_errors[i]<< std::setw(20) << left_abs_errors[i]<< std::setw(20) << trap_abs_errors[i]<< std::setw(20) << simp_abs_errors[i] << std::endl;
    }
    std::cout << std::string(100, '-') << std::endl;

    std::cout << "(Results are also plotted via the python wrapper.)"<<std::endl;

    std::cout << "-----> The plot for the absolute error vs spacing shows that for all methods, as the subinterval spacing decreases, the error reduces and converges at a final value (about 5e-17)."<<std::endl
    <<" This plataeu can be explained by the error mostly arisising from floating-point precision errors at spacings that small (instead of the method's accuracy)."<<std::endl
    << "In a non-log scale, the midpoint, leftpoint and Trapezoidal errors are all expected to be proportional to O(h^2) (where h is the spacing), "<<std::endl
    << "while for Simpon's method is supposed to have an error term proportional to O(h^4), and to thus converge the fastest."<<std::endl
    << "In a log scale, this would thus be equivalent to the first three having a slope of -2 and the latter a slope of -4. Up until the end, this linearity is observed - "<<std::endl
    <<"however, Simpson's rule converges the slowest, with the midpoint rule converging the fastest. The Leftpoint method sows the same pattern as the Trapezoidal method."<<std::endl
    <<"For all methods, the error oscillates as it decreases."<< std::endl;

    //---------------------------------------------------------------------------------------------------------------------

    std::cout<<" "<<std::endl;
    std::cout<<"MIDPOINT INTEGRATION TIME VS NR OF THREADS:"<<std::endl;

    //Define the spacing to test the parallelisation with and the maximum number of threads to iterate through.
    double spacing = 1e-8;
    int max_threads = 14;

    std::cout<<"Spacing used: "<<spacing << std::endl;
    std::cout << std::string(100, '-') << std::endl;


    double start  = lower_b + spacing/2;
    for (int nr_of_threads = 1; nr_of_threads <= max_threads; nr_of_threads++){
        auto start_time_wall = omp_get_wtime();
        double integral = midpoint_rule( lower_b, upper_b, spacing, start, nr_of_threads);
        auto end_time_wall = omp_get_wtime();;

        auto wall_time = end_time_wall - start_time_wall;
        std::cout<< std::setw(5)<<nr_of_threads<< " threads --> " <<std::setw(10)<< wall_time<< " seconds"<< std::endl;
    }

    std::cout << "(Results are also plotted via the python wrapper.)"<<std::endl;

    std::cout << "-----> The plots show that the wall time drops sharply as the number of threads increases from 1 to around 4. " <<std::endl
    <<"After a certain number of threads (about 8), the wall time decreases more slowly, reaching a plateau at about 16 threads." <<std::endl
    <<"The wall time is here most likely limited by other factors (eg communication overhead)."<<std::endl
    <<"The speedup plot shows a linear increase, meaning that each additional thread is providing an approx. proportional reduction in wall time."<<std::endl;

    return 0;

}
