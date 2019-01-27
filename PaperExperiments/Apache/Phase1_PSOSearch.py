import os
import pandas as pd
import numpy as np
import itertools
import datetime
from scipy import special
import sys

import ProgramToInfer

# calculate how many combinations are there for choosing no_of_select elements from no_of_elements
def get_size_of_comb(no_of_elements, no_of_select):
    size_of_comb_with_degree = np.int(special.comb(no_of_elements, no_of_select, repetition=True))
    return size_of_comb_with_degree


# given no_of_elements elements, evaluate the number of combinations with a highest_degree
def get_size_of_comb_with_highest_degree(no_of_elements, highest_degree):
    size_of_comb = 0
    degree = 0
    while degree < highest_degree + 1:
        size_of_comb_with_degree = get_size_of_comb(no_of_elements, degree)
        size_of_comb += size_of_comb_with_degree
        degree += 1
    return np.int(size_of_comb)


# given a vecor, return all the combinations that its elements can form
def comb(vector, highest_degree):
    comb_vector = np.array([1])
    degree = 1
    while degree < highest_degree + 1:
        comb_vector_with_degree = np.array(
            [np.prod(element) for element in itertools.combinations_with_replacement(vector, degree)])
        comb_vector = np.concatenate((comb_vector, comb_vector_with_degree), axis=0)
        degree += 1
    return comb_vector


# generate all the i0 (i0_all) with the given population
def generate_i0_all(inputcases_range, population_of_inputcases, no_of_elements_input):
    # inputcases_lower = inputcases_range[:, 0]
    # inputcases_upper = inputcases_range[:, 1]
    # i0_all = np.tile([inputcases_lower], (population_of_inputcases, 1)) + np.tile([inputcases_upper - inputcases_lower],
    #                                                                               (population_of_inputcases,
    #                                                                                1)) * np.random.rand(
    #     population_of_inputcases, no_of_elements_input)

    inputcases_lower = inputcases_range[:, 0:1]
    inputcases_upper = inputcases_range[:, 1:]
    i0_all = np.empty((population_of_inputcases, no_of_elements_input))
    for index_i0 in range(population_of_inputcases):
        i0_all[index_i0] = np.random.uniform(low=inputcases_lower, high=inputcases_upper,
                                             size=(no_of_elements_input, 1)).reshape(no_of_elements_input, )

    return i0_all


# given i0 and A, return all the inputs for the relation
def generate_i(i0, comb_i0, A, mode_input_relation):
    # print(A)
    i_1_to_end = np.dot(A, comb_i0.reshape(-1, 1)).reshape(A.shape[0], -1)

    # i_1_to_end = np.ones((A.shape[0], A.shape[1]))
    # for index_run in range(A.shape[0]):
    #     i_1_to_end[index_run] = np.dot(A[index_run], comb_i0)

    # equal input relation
    if mode_input_relation == 1:
        i = np.concatenate((i0.reshape(1, -1), i_1_to_end), axis=0)
    # largerthan input relation. Only applicable to 2 inputs now.
    elif mode_input_relation == 2:
        i_1_to_end += np.random.uniform(low=0.1, high=1.0, size=i_1_to_end.shape)
        i = np.concatenate((i0.reshape(1, -1), i_1_to_end), axis=0)
    # lessthan input relation. Only applicable to 2 inputs now.
    elif mode_input_relation == 3:
        i_1_to_end -= np.random.uniform(low=0.1, high=1.0, size=i_1_to_end.shape)
        i = np.concatenate((i0.reshape(1, -1), i_1_to_end), axis=0)
    return i


# calculate output(o)
def get_o(program, func_index, i, no_of_elements_output):
    o = np.zeros((i.shape[0], no_of_elements_output))
    for index_i in range(i.shape[0]):
        o[index_i] = program(i[index_i], func_index)
    return o


# the cost function for one particle (a pair of A and B)
def get_cost_of_AB(program, func_index, A, B, i0_all, mode_input_relation, mode_output_relation,
                   degree_of_input_relation,
                   degree_of_output_relation, no_of_elements_output):
    # print(A)
    if mode_output_relation == 1:
        cost_of_AB = 0.0

        for index_i0 in range(i0_all.shape[0]):
            i0 = i0_all[index_i0]
            comb_i0 = comb(i0, degree_of_input_relation)
            i = generate_i(i0, comb_i0, A, mode_input_relation)
            o = get_o(program, func_index, i, no_of_elements_output)
            o_flatten = np.ravel(o)
            comb_o = comb(o_flatten, degree_of_output_relation)

            distance = np.dot(B, comb_o)
            if np.isreal(distance) and not np.isnan(distance):
                cost_of_AB += np.abs(distance)
            else:
                cost_of_AB = (cost_of_AB + 1.0) * 10.0


    elif mode_output_relation == 2:
        cost_of_AB = 1.0

        for index_i0 in range(i0_all.shape[0]):
            i0 = i0_all[index_i0]
            comb_i0 = comb(i0, degree_of_input_relation)
            i = generate_i(i0, comb_i0, A, mode_input_relation)
            o = get_o(program, func_index, i, no_of_elements_output)
            o_flatten = np.ravel(o)
            comb_o = comb(o_flatten, degree_of_output_relation)

            distance = np.dot(B, comb_o)
            if np.isreal(distance) and not np.isnan(distance):
                if distance > 0:
                    cost_of_AB -= 1.0 / i0_all.shape[0]

    elif mode_output_relation == 3:
        cost_of_AB = 1.0

        for index_i0 in range(i0_all.shape[0]):
            i0 = i0_all[index_i0]
            comb_i0 = comb(i0, degree_of_input_relation)
            i = generate_i(i0, comb_i0, A, mode_input_relation)
            o = get_o(program, func_index, i, no_of_elements_output)
            o_flatten = np.ravel(o)
            comb_o = comb(o_flatten, degree_of_output_relation)

            distance = np.dot(B, comb_o)
            if np.isreal(distance) and not np.isnan(distance):
                if distance < 0:
                    cost_of_AB -= 1.0 / i0_all.shape[0]

    return cost_of_AB


# get the cost for all the particles
def get_cost_of_AB_all(program, func_index, A_all, B_all, i0_all, mode_input_relation, mode_output_relation,
                       degree_of_input_relation, degree_of_output_relation, no_of_elements_output):
    cost_of_AB_all = np.empty(A_all.shape[0])

    for index_particle in range(A_all.shape[0]):
        cost_of_AB_all[index_particle] = get_cost_of_AB(program, func_index, A_all[index_particle],
                                                        B_all[index_particle], i0_all,
                                                        mode_input_relation, mode_output_relation,
                                                        degree_of_input_relation, degree_of_output_relation,
                                                        no_of_elements_output)
    return cost_of_AB_all


# check if all the elements of a matrix is less than the thre. If yes, change them to thre or -thre accordingly
def anti_degrade(a, thre):
    compared_matrix_a_thre = np.less(np.absolute(a), thre)
    if np.prod(compared_matrix_a_thre) == 1:
        compared_matrix_a_zero = np.less_equal(a, 0)
        a_anti_degrade = np.zeros(a.shape) + compared_matrix_a_zero * (-thre) + (1 - compared_matrix_a_zero) * thre
        return a_anti_degrade
    else:
        return a


class PSO:
    def __init__(self, program, func_index, no_of_inputs, mode_input_relation, mode_output_relation,
                 degree_input_relation, degree_output_relation, no_of_elements_input, no_of_elements_output,
                 no_of_particles, no_of_inputcases, inputcases_range, const_range, coeff_range):
        self.program = program
        self.func_index = func_index
        self.no_of_inputs = no_of_inputs
        self.mode_input_relation = mode_input_relation
        self.mode_output_relation = mode_output_relation
        self.degree_input_relation = degree_input_relation
        self.degree_output_relation = degree_output_relation
        self.no_of_elements_input = no_of_elements_input
        self.no_of_elements_output = no_of_elements_output
        self.no_of_particles = no_of_particles
        self.no_of_inputcases = no_of_inputcases
        self.inputcases_range = inputcases_range
        self.const_range = const_range
        self.coeff_range = coeff_range

        self.size_of_comb_i0 = get_size_of_comb_with_highest_degree(self.no_of_elements_input,
                                                                    self.degree_input_relation)
        self.shape_of_A_all = (
            self.no_of_particles, (self.no_of_inputs - 1), self.no_of_elements_input, self.size_of_comb_i0)
        self.size_of_A_all = np.prod(self.shape_of_A_all)
        self.shape_of_B_all = (self.no_of_particles,
                               get_size_of_comb_with_highest_degree((self.no_of_inputs * self.no_of_elements_output),
                                                                    self.degree_output_relation))
        self.size_of_B_all = np.prod(self.shape_of_B_all)
        self.indices_A_highest_degree = get_size_of_comb(self.no_of_elements_input, self.degree_input_relation)
        self.indices_B_highest_degree = get_size_of_comb((self.no_of_elements_output * self.no_of_inputs),
                                                         self.degree_output_relation)

        # i = [i0, i1, i2 ...], i1 = np.dot(A[0], i0), i2 = np.dot(A[1], i0)
        self.Ashape0 = self.no_of_inputs - 1

    def generate_initial_A_all(self):
        A_all_cons = np.round(np.random.uniform(low=self.const_range[0], high=self.const_range[1],
                                       size=(self.no_of_particles, self.Ashape0, self.no_of_elements_input, 1)), decimals=2)
        # print(A_all_cons)
        A_all_coeff = np.round(np.random.uniform(low=self.coeff_range[0], high=self.coeff_range[1], size=(
            self.no_of_particles, self.Ashape0, self.no_of_elements_input, (self.size_of_comb_i0 - 1))), decimals=1)

        # prevent A from degradation
        for index_particle in range(self.no_of_particles):
            for index_input in range(self.no_of_inputs - 1):
                A_all_coeff[index_particle:index_particle + 1, index_input:index_input + 1, :, -self.indices_A_highest_degree:] = anti_degrade(A_all_coeff[index_particle:index_particle + 1, index_input:index_input + 1, :, -self.indices_A_highest_degree:], 0.5)

        A_all = np.concatenate((A_all_cons, A_all_coeff), axis=3)
        # print(A_all)
        return A_all

    def generate_initial_B_all(self):
        B_all_cons = np.random.uniform(low=self.const_range[0], high=self.const_range[1],
                                       size=(self.shape_of_B_all[0], 1))
        B_all_coeff = np.random.uniform(low=self.coeff_range[0], high=self.coeff_range[1],
                                        size=(self.shape_of_B_all[0], (self.shape_of_B_all[1] - 1)))

        # keep B from degradation
        for index_particle in range(self.no_of_particles):
            B_all_coeff[index_particle:index_particle + 1, -self.indices_B_highest_degree:] = anti_degrade(
                B_all_coeff[index_particle:index_particle + 1, -self.indices_B_highest_degree:], 0.5)

        B_all = np.concatenate((B_all_cons, B_all_coeff), axis=1)
        return B_all

    # def trans_AB_all_to_p_all(self, A_all, B_all):
    #     A_all_flatten = A_all.flatten()
    #     B_all_flatten = B_all.flatten()
    #     p_all = np.concatenate((A_all_flatten, B_all_flatten))

    # def trans_p_all_to_AB_all(self, p_all):
    #     A_all = p_all[0:self.size_of_A_all,].reshape(self.shape_of_A_all)
    #     B_all = p_all[self.size_of_A_all:,].reshape(self.shape_of_B_all)

    def update_AB_all(self, i0_all, A_all, A_v_all, B_all, B_v_all, A_all_p_best, B_all_p_best, index_g_best,
                      omega, cost_of_AB_all_p_best):
        epsilon1 = np.float(1.49445)
        epsilon2 = np.float(1.49445)
        r1_A_all = np.random.uniform(low=0.0, high=1.0, size=A_all.shape)
        r2_A_all = np.random.uniform(low=0.0, high=1.0, size=A_all.shape)
        r1_B_all = np.random.uniform(low=0.0, high=1.0, size=B_all.shape)
        r2_B_all = np.random.uniform(low=0.0, high=1.0, size=B_all.shape)

        A_all_g_best = A_all_p_best[index_g_best]
        B_all_g_best = B_all_p_best[index_g_best]
        # tile_A_all_g_best = np.tile(A_all_g_best, (A_all.shape[0], 1, 1, 1))
        # tile_B_all_g_best = np.tile(B_all_g_best, (B_all.shape[0], 1))

        A_v_all_next = omega * A_v_all + epsilon1 * r1_A_all * (A_all_p_best - A_all) + epsilon2 * r2_A_all * (
                A_all_g_best - A_all)
        B_v_all_next = omega * B_v_all + epsilon1 * r1_B_all * (B_all_p_best - B_all) + epsilon2 * r2_B_all * (
                B_all_g_best - B_all)

        A_all_next = A_all + A_v_all_next
        B_all_next = B_all + B_v_all_next

        A_all_next_cons = A_all_next[:, :, :, 0:1]
        A_all_next_coeff = A_all_next[:, :, :, 1:]
        B_all_next_cons = B_all_next[:, 0:1]
        B_all_next_coeff = B_all_next[:, 1:]

        # keep A_all_cons in boundary
        compared_matrix_A_all_next_cons_low = np.greater_equal(A_all_next_cons, self.const_range[0])
        if np.prod(compared_matrix_A_all_next_cons_low) != 1:
            A_all_next_cons = A_all_next_cons * compared_matrix_A_all_next_cons_low + (
                    1 - compared_matrix_A_all_next_cons_low) * self.const_range[0]
        compared_matrix_A_all_next_cons_high = np.less_equal(A_all_next_cons, self.const_range[1])
        if np.prod(compared_matrix_A_all_next_cons_high) != 1:
            A_all_next_cons = A_all_next_cons * compared_matrix_A_all_next_cons_high + (
                    1 - compared_matrix_A_all_next_cons_high) * self.const_range[1]

        # keep A_all_coeff in boundary
        compared_matrix_A_all_next_coeff_low = np.greater_equal(A_all_next_coeff, self.coeff_range[0])
        if np.prod(compared_matrix_A_all_next_coeff_low) != 1:
            A_all_next_coeff = A_all_next_coeff * compared_matrix_A_all_next_coeff_low + (
                    1 - compared_matrix_A_all_next_coeff_low) * self.coeff_range[0]
        compared_matrix_A_all_next_coeff_high = np.less_equal(A_all_next_coeff, self.coeff_range[1])
        if np.prod(compared_matrix_A_all_next_coeff_high) != 1:
            A_all_next_coeff = A_all_next_coeff * compared_matrix_A_all_next_coeff_high + (
                    1 - compared_matrix_A_all_next_coeff_high) * self.coeff_range[1]

        # prevent A from degradation
        for index_particle in range(self.no_of_particles):
            for index_input in range(self.no_of_inputs - 1):
                A_all_next_coeff[index_particle:index_particle + 1, index_input:index_input + 1, :, -self.indices_A_highest_degree:] = anti_degrade(A_all_next_coeff[index_particle:index_particle + 1, index_input:index_input + 1, :, -self.indices_A_highest_degree:], 0.5)

        A_all_next = np.concatenate((A_all_next_cons, A_all_next_coeff), axis=3)

        # keep B_all_cons in boundary
        compared_matrix_B_all_next_cons_low = np.greater_equal(B_all_next_cons, self.const_range[0])
        if np.prod(compared_matrix_B_all_next_cons_low) != 1:
            B_all_next_cons = B_all_next_cons * compared_matrix_B_all_next_cons_low + (
                    1 - compared_matrix_B_all_next_cons_low) * self.const_range[0]
        compared_matrix_B_all_next_cons_high = np.less_equal(B_all_next_cons, self.const_range[1])
        if np.prod(compared_matrix_B_all_next_cons_high) != 1:
            B_all_next_cons = B_all_next_cons * compared_matrix_B_all_next_cons_high + (
                    1 - compared_matrix_B_all_next_cons_high) * self.const_range[1]

        # keep B_all_coeff in boundary
        compared_matrix_B_all_next_coeff_low = np.greater_equal(B_all_next_coeff, self.coeff_range[0])
        if np.prod(compared_matrix_B_all_next_coeff_low) != 1:
            B_all_next_coeff = B_all_next_coeff * compared_matrix_B_all_next_coeff_low + (
                    1 - compared_matrix_B_all_next_coeff_low) * self.coeff_range[0]
        compared_matrix_B_all_next_coeff_high = np.less_equal(B_all_next_coeff, self.coeff_range[1])
        if np.prod(compared_matrix_B_all_next_coeff_high) != 1:
            B_all_next_coeff = B_all_next_coeff * compared_matrix_B_all_next_coeff_high + (
                    1 - compared_matrix_B_all_next_coeff_high) * self.coeff_range[1]

        # keep B from degradation
        for index_particle in range(self.no_of_particles):
            B_all_next_coeff[index_particle: index_particle + 1, -self.indices_B_highest_degree:] = anti_degrade(
                B_all_next_coeff[index_particle: index_particle + 1, -self.indices_B_highest_degree:], 0.5)

        B_all_next = np.concatenate((B_all_next_cons, B_all_next_coeff), axis=1)

        A_v_all_next = A_all_next - A_all
        B_v_all_next = B_all_next - B_all

        # update p_best
        cost_of_AB_all_next = get_cost_of_AB_all(self.program, self.func_index, A_all_next, B_all_next, i0_all,
                                                 self.mode_input_relation,
                                                 self.mode_output_relation, self.degree_input_relation,
                                                 self.degree_output_relation, self.no_of_elements_output)
        compared_matrix_cost_p_all = np.less_equal(cost_of_AB_all_p_best, cost_of_AB_all_next)
        A_all_p_best = A_all_p_best * compared_matrix_cost_p_all.reshape(self.no_of_particles, 1, 1, 1) + A_all_next * (
                1 - compared_matrix_cost_p_all.reshape(self.no_of_particles, 1, 1, 1))
        B_all_p_best = B_all_p_best * compared_matrix_cost_p_all.reshape(self.no_of_particles, 1) + B_all_next * (
                1 - compared_matrix_cost_p_all.reshape(self.no_of_particles, 1))
        cost_of_AB_all_p_best = cost_of_AB_all_p_best * compared_matrix_cost_p_all + cost_of_AB_all_next * (
                1 - compared_matrix_cost_p_all)

        # for index_particle in range(NO_OF_PARTICLES):
        #     cost_of_AB_next = get_cost_of_AB(program, A_all_next[index_particle], B_all_next[index_particle], i0_all, MODE_INPUT_RELATION, MODE_OUTPUT_RELATION, DEGREE_OF_INPUT_RELATION, DEGREE_OF_OUTPUT_RELATION, NO_OF_ELEMENTS_OUTPUT)
        #     cost_of_AB_p_best = cost_of_AB_all_p_best[index_particle]

        #     if cost_of_AB_next < cost_of_AB_p_best:
        #         A_all_p_best[index_particle] = A_all_next[index_particle]
        #         B_all_p_best[index_particle] = B_all_next[index_particle]
        #         cost_of_AB_all_p_best[index_particle] = cost_of_AB_next

        index_g_best = np.argmin(cost_of_AB_all_p_best)

        return A_all_next, B_all_next, A_v_all_next, B_v_all_next, A_all_p_best, B_all_p_best, index_g_best, cost_of_AB_all_p_best

    # def update_p_best_g_best(self, program, i0_all, mode_input_relation, mode_output_relation, degree_of_input_relation, degree_of_output_relation, A_all_p_best, B_all_p_best, A_all_next, B_all_next):
    #     cost_of_p_all_p_best = self.get_cost_of_AB_all(program, A_all_p_best, B_all_p_best, i0_all, mode_input_relation, mode_output_relation, degree_of_input_relation, degree_of_output_relation)
    #     cost_of_p_all_next = self.get_cost_of_AB_all(program, A_all_next, B_all_next, i0_all, mode_input_relation, mode_output_relation, degree_of_input_relation, degree_of_output_relation)
    #     print(cost_of_p_all_next.shape)
    #     print(cost_of_p_all_p_best.shape)
    #     compared_cost = cost_of_p_all_p_best < cost_of_p_all_next
    #     A_all_p_best_next = np.zeros(A_all_p_best.shape)
    #     B_all_p_best_next = np.zeros(B_all_p_best.shape)

    #     for index_particle in range(A_all_next.shape[0]):
    #         A_all_p_best_next[index_particle] = A_all_p_best[index_particle] * compared_cost[index_particle] + A_all_next[index_particle] * (1 - compared_cost[index_particle])
    #         B_all_p_best_next[index_particle] = B_all_p_best[index_particle] * compared_cost[index_particle] + B_all_next[index_particle] * (1 - compared_cost[index_particle])
    #     # print(A_all_p_best_next.shape)
    #     # print(B_all_p_best_next.shape)
    #     cost_of_p_all_next = self.get_cost_of_AB_all(program, A_all_p_best_next, B_all_p_best_next, i0_all, mode_input_relation, mode_output_relation, degree_of_input_relation, degree_of_output_relation)
    #     index_g_best = np.argmin(cost_of_p_all_next)

    #     return A_all_p_best_next, B_all_p_best_next, index_g_best

    def run(self):
        i0_all = generate_i0_all(self.inputcases_range, self.no_of_inputcases, self.no_of_elements_input)
        A_all = self.generate_initial_A_all()
        B_all = self.generate_initial_B_all()

        A_v_all = np.zeros(A_all.shape)
        B_v_all = np.zeros(B_all.shape)
        A_all_p_best = np.copy(A_all)
        B_all_p_best = np.copy(B_all)
        cost_of_AB_all_p_best = get_cost_of_AB_all(self.program, self.func_index, A_all, B_all, i0_all,
                                                   self.mode_input_relation,
                                                   self.mode_output_relation, self.degree_input_relation,
                                                   self.degree_output_relation, self.no_of_elements_output)

        index_g_best = np.argmin(cost_of_AB_all_p_best)

        omega_s = 0.9
        omega_e = 0.4

        iterations = pso_iterations
        iteration = 0
        while iteration < iterations:
            omega = omega_s - (omega_s - omega_e) * ((iteration / iterations) ** 2)
            A_all, B_all, A_v_all, B_v_all, A_all_p_best, B_all_p_best, index_g_best, cost_of_AB_all_p_best = self.update_AB_all(
                i0_all, A_all, A_v_all, B_all, B_v_all, A_all_p_best, B_all_p_best, index_g_best, omega,
                cost_of_AB_all_p_best)
            min_cost = cost_of_AB_all_p_best[index_g_best]
            # print(f'func_index is {func_index}, iteration is {iteration}, min_cost is {np.round(min_cost, decimals=3)}')
            ## break the iteration in advance if solution is found
            # if self.mode_output_relation == 1:
            #     if min_cost < 1:
            #         break
            # else:
            #     if min_cost < 0.05:
            #         break
            iteration += 1

        A = A_all_p_best[index_g_best]
        B = B_all_p_best[index_g_best]

        return min_cost, A, B


def main():
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    if not os.path.isdir(f"{output_path}/phase1"):
        os.mkdir(f"{output_path}/phase1")

    no_of_elements_input = ProgramToInfer.getNEI(func_index)
    no_of_elements_output = ProgramToInfer.getNEO(func_index)


    no_of_particles = 30
    no_of_inputcases = 100

    min_cost_candidates = []
    A_candidates = []
    B_candidates = []

    pso_run = 0
    while True:
        AutoMR = PSO(ProgramToInfer.program, func_index, no_of_inputs, mode_input_relation, mode_output_relation,
                     degree_of_input_relation, degree_of_output_relation, no_of_elements_input,
                     no_of_elements_output, no_of_particles, no_of_inputcases, inputcases_range,
                     const_range, coeff_range)
        min_cost, A, B = AutoMR.run()

        min_cost_candidates.append(np.round(min_cost, decimals=3))
        A_candidates.append(A)
        B_candidates.append(B)
        np.savez('{}/phase1/{}_{}_{}_{}_{}_{}.npz'.format(output_path, func_index,
                                                                  no_of_inputs,
                                                                  mode_input_relation,
                                                                  mode_output_relation,
                                                                  degree_of_input_relation,
                                                                  degree_of_output_relation),
                 min_cost_candidates=min_cost_candidates, A_candidates=np.array(A_candidates),
                 B_candidates=np.array(B_candidates))
        print(f"func_index is {func_index}, parameters is {parameters}, pso_run is {pso_run}")
        print(A)
        print(B)
        print("----------\n")

        pso_run += 1
        if not pso_run < pso_runs:
            break

if __name__ == "__main__":
    ## get parameters from command
    # func_indices = [int(x) for x in sys.argv[1].strip(":").split(' ')]
    # NOI = [int(x) for x in sys.argv[2].split(' ')]
    # MIR = [int(x) for x in sys.argv[3].split(' ')]
    # MOR = [int(x) for x in sys.argv[4].split(' ')]
    # DIR = [int(x) for x in sys.argv[5].split(' ')]
    # DOR = [int(x) for x in sys.argv[6].split(' ')]

    # type in parameters
    func_indices = [int(sys.argv[1])]
    parameters_collection = [sys.argv[2]]

    output_path = ProgramToInfer.output_path
    pso_runs = ProgramToInfer.pso_runs
    pso_iterations = ProgramToInfer.pso_iterations

    coeff_range = ProgramToInfer.coeff_range
    const_range = ProgramToInfer.const_range

    for func_index in func_indices:
        times = pd.DataFrame()
        inputcases_range = np.tile(ProgramToInfer.get_input_range(func_index), (ProgramToInfer.getNEI(func_index), 1))

        for parameters in parameters_collection:
            parameters_int = [int(e) for e in parameters.split("_")]
            no_of_inputs = parameters_int[0]
            mode_input_relation = parameters_int[1]
            mode_output_relation = parameters_int[2]
            degree_of_input_relation = parameters_int[3]
            degree_of_output_relation = parameters_int[4]
            t1 = datetime.datetime.now()
            main()
            t2 = datetime.datetime.now()
            cost_time = np.round((t2-t1).total_seconds(), decimals=2)

            times.loc[f"{func_index}_{parameters}", "pso_iterations"] = pso_iterations
            times.loc[f"{func_index}_{parameters}", "search"] = cost_time
            times.to_csv(f"{output_path}/{func_index}_{parameters}_times.csv")

