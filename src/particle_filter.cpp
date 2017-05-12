/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"

using namespace std;

default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	num_particles = 100;

	particles = std::vector<Particle>(num_particles);

	int id=0;

	for(auto &p: particles){
		p.id = id++;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1.0/num_particles;
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);

	for(auto &p: particles){
		double newtheta = p.theta + delta_t*yaw_rate;
		p.x += velocity*(sin(newtheta) - sin(p.theta))/yaw_rate + dist_x(gen);
		p.y += velocity*(cos(p.theta) - cos(newtheta))/yaw_rate + dist_y(gen);
		p.theta = newtheta + dist_theta(gen);
	}
}

std::vector<LandmarkObs> transform(const std::vector<LandmarkObs> &landmarks, const Particle &particle) {
	std::vector<LandmarkObs> transformed_obs;
	for(auto &landmark: landmarks){
		double sin_theta = sin(particle.theta);
		double cos_theta = cos(particle.theta);
		double transformed_x = landmark.x*cos_theta - landmark.y*sin_theta + particle.x;
		double transformed_y = landmark.x*sin_theta + landmark.y*cos_theta + particle.y;
		LandmarkObs obs = {landmark.id, transformed_x, transformed_y};
		transformed_obs.push_back(obs);
	}

	return transformed_obs;
}


void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for(auto &obs: observations) {
		double min = 1e+300, d=-1.0;
		for(auto const& landmark: predicted) {
			if(min > (d=dist(landmark.x, landmark.y, obs.x, obs.y))) {
				min = d;
				obs.id = landmark.id;
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html


	for(auto &p: particles){
		std::vector<LandmarkObs> landmarks_in_range;
		for(auto &landmark: map_landmarks.landmark_list){
			if(dist(landmark.x_f, landmark.y_f, p.x, p.y) <= sensor_range){
				LandmarkObs landmark_obs = {int(landmarks_in_range.size()), double(landmark.x_f), double(landmark.y_f)};
				landmarks_in_range.push_back(landmark_obs);
			}
		}

		std::vector<LandmarkObs> transformed_obs = transform(observations, p);
		dataAssociation(landmarks_in_range, transformed_obs);

		double sum_sqr_x_diff=0.0, sum_sqr_y_diff=0.0;
		for(auto const& obs: transformed_obs){
			double x_diff = obs.x - landmarks_in_range[obs.id].x;
			double y_diff = obs.y - landmarks_in_range[obs.id].y;
			sum_sqr_x_diff += x_diff*x_diff;
			sum_sqr_y_diff += y_diff*y_diff; 
		}

		double std_x = std_landmark[0], std_y = std_landmark[1];

		p.weight = exp(-sum_sqr_x_diff/(2*std_x*std_x)-sum_sqr_y_diff/(2*std_y*std_y));
	}

	if(weights.size() != num_particles){
		weights = std::vector<double>(num_particles);
	}

	for(int i=0; i<num_particles; i++){
		weights[i] = particles[i].weight;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	std::discrete_distribution<int> dist_particle(weights.begin(), weights.end());
	std::vector<Particle> resampled_particles;
	// default_random_engine gen;

	for(int i=0; i<num_particles; i++){
		resampled_particles.push_back(particles[dist_particle(gen)]);
	}

	particles = resampled_particles;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
