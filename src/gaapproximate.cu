/**
 * GA Approximate: Try to approximate a simple function using Genetic Algorithm
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cfloat>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

/**
 * Macros to configure experiment
 */
#define POPULATION_SIZE 10
#define GENOTYPE_SIZE 2
#define MAX_NUMBER_OF_GENERATIONS 100
#define CROSSOVER_PROB 0.8
#define CROSSING_PROB 0.5
#define MUTATION_PROB 0.1
#define SAMPLING_POINTS 10
#define RAND_01 ((double)rand()/RAND_MAX)

void mutate(float* chromosome)
{
	for (int i = 0; i < GENOTYPE_SIZE; i++)
	{
		if (RAND_01 < MUTATION_PROB)
			chromosome[i] += 2 * RAND_01 - 1;
	}
}

void crossOver(float* chrom1, float* chrom2, float* newChrom1, float* newChrom2)
{
	for (int i = 0; i < GENOTYPE_SIZE; i++)
	{
		if (RAND_01 < CROSSING_PROB)
		{
			newChrom1[i] = chrom1[i];
			newChrom2[i] = chrom2[i];
		}
		else
		{
			newChrom1[i] = chrom2[i];
			newChrom2[i] = chrom1[i];
		}
	}
}

int rouletteWheel(float* fitness_values, double sum_fitness_values)
{
	double offsetSelection = 0;

	double chooseValue = RAND_01;

	for (int i = 0; i < POPULATION_SIZE; i++)
	{
		offsetSelection += (fitness_values[i] / sum_fitness_values);

		if (offsetSelection >= chooseValue)
		{
			return i;
		}
	}

	return POPULATION_SIZE-1;
}

__host__ __device__ float targetFunction(float value)
{
	// 10 * x^2 + 3 * x + 10
	return 10 * value * value + 3 * value - 10;
}

/**
 * CUDA Kernel Device code
 *
 * Run Genetic Algorithm experiments
 */
__global__ void RunGAIteration(float *population, float *fitness_values)
{
	// Get chromosome id
	int c_id = threadIdx.x;

	float *chromosome = population + (c_id * GENOTYPE_SIZE);
	float output, desiredOutput, testPoint;

	float fitness_value_of_instance = 0;

	for (int i = 0; i < SAMPLING_POINTS; i++)
	{
		testPoint = i * (20 / SAMPLING_POINTS) - 10;
		desiredOutput = targetFunction(testPoint);

		output = chromosome[0];

		for (int g = 1; g < GENOTYPE_SIZE; g++)
		{
			output += chromosome[g] * testPoint;
			testPoint *= testPoint;
		}

		fitness_value_of_instance += (desiredOutput - output) * (desiredOutput - output);
	}

	fitness_value_of_instance /= SAMPLING_POINTS;

	fitness_values[c_id] = fitness_value_of_instance;
}

/**
 * Host main routine
 */
int
main(void)
{
	// Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;

	float *population, *fitness_values;
	int POPULATION_MATRIX_SIZE = POPULATION_SIZE * GENOTYPE_SIZE;
	float FITNESS_VALUES_SIZE = POPULATION_SIZE;

	float *newGeneration = (float*) malloc(POPULATION_MATRIX_SIZE * sizeof(float));

	cudaMallocManaged(&population, POPULATION_MATRIX_SIZE * sizeof(float));
	cudaMallocManaged(&fitness_values, FITNESS_VALUES_SIZE * sizeof(float));

	// Generate a first random population
	for (int i = 0; i < POPULATION_SIZE; i++) {
		for (int g = 0; g < GENOTYPE_SIZE; g++) {
			population[i*GENOTYPE_SIZE + g] = (rand() % 100 - 50) / 10.0;
		}
	}

	// Run GA
	double fitness_values_sum;

	int c_id1, c_id2;

	float *chromosome1, *chromosome2, *newChromosome1, *newChromosome2;

	float bestFitness = FLT_MAX, bestFitnessOnGeneration = FLT_MAX;

	for (int gen = 0; gen < MAX_NUMBER_OF_GENERATIONS; gen++)
	{
		RunGAIteration<<<1, POPULATION_SIZE>>>(population, fitness_values);
		cudaDeviceSynchronize();

		err = cudaGetLastError();

		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to launch RunGAIteration kernel (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		// Get best fitness value
		bestFitnessOnGeneration = FLT_MAX;
		for (int i = 0; i < POPULATION_SIZE; i++)
		{
			if (fitness_values[i] < bestFitness)
			{
				bestFitness = fitness_values[i];
			}

			if (fitness_values[i] < bestFitnessOnGeneration)
			{
				bestFitnessOnGeneration = fitness_values[i];
			}
		}
		printf("The best fitness value on generation %d is %.2f.\n", gen, bestFitnessOnGeneration);
		printf("The best fitness value until now is %.2f.\n", gen, bestFitness);

		/** Get new generation **/

		// Verify sum of fitness values
		fitness_values_sum = 0;

		for (int i = 0; i < POPULATION_SIZE; i++)
			fitness_values_sum += fitness_values[i];

		for (int i = 0; i < POPULATION_SIZE; i+=2)
		{
			// Select a pair chromosomes using RouletteWheel
			c_id1 = rouletteWheel(fitness_values, fitness_values_sum);
			c_id2 = rouletteWheel(fitness_values, fitness_values_sum);

			chromosome1 = population + c_id1;
			chromosome2 = population + c_id2;

			newChromosome1 = newGeneration + (i*GENOTYPE_SIZE);
			newChromosome2 = newGeneration + ((i+1)*GENOTYPE_SIZE);

			// Crossover
			if (RAND_01 <= CROSSOVER_PROB)
			{
				// Do Crossover
				crossOver(chromosome1, chromosome2, newChromosome1, newChromosome2);
			}
			else
			{
				// If not crossover, just copy the 2 chromosomes taken with Roulette Wheel method
				memcpy(newChromosome1, chromosome1, GENOTYPE_SIZE);
				memcpy(newChromosome2, chromosome2, GENOTYPE_SIZE);
			}

			// Mutate genes of generated pair of chromosomes
			mutate(newChromosome1);
			mutate(newChromosome2);
		}

		for (int i = 0; i < POPULATION_MATRIX_SIZE; i++)
		{
			population[i] = newGeneration[i];
		}
	}

	// Verify each chromosome
	for (int i = 0; i < POPULATION_SIZE; i++)
		printf("Chromosome %d got fitness value of %.2f.\n", i, fitness_values[i]);

	free(newGeneration);
	cudaFree(population);
	cudaFree(fitness_values);

	return 0;
}

