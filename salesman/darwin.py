import csv
import time

import numpy as np

from salesman.city import CityFactory
from salesman.route import (
    tour_swap_2_random, Route, generate_rand_routes,
    sort_routes, display_tours_stats, apply_local_optimizations,
    calculate_capacity, get_random_part_from_father, divide_route_before_after,
    plot_route, generate_nearest_neighbour_route,
    plot_shortest_route_len_vs_epochs)


class NaturalSelection:
    @staticmethod
    def keep_n_best(routes_sorted, routes_length, n_keep):
        routes_sorted = routes_sorted[0:n_keep]
        routes_length = routes_length[0:n_keep]
        return routes_sorted, routes_length


def generate_parent_list(capacity_normalized, n_new, routes):
    parent_idx = np.random.choice(
        np.arange(0, capacity_normalized.shape[0]),
        p=capacity_normalized,
        size=n_new,
        replace=True
    )
    parents = [routes[x] for x in parent_idx]
    return parents


def pair_matching(routes_list, n_new=100):
    """

    :param routes_list:
    :param n_new:
    :param cut_len: length of the route fragment that is injected from
    father to mother
    :return:
    """
    # sort routes by the len
    routes, routes_len = sort_routes(routes_list)

    # display_tours_stats(routes_len)
    capacity = calculate_capacity(routes_sorted=routes)
    capacity_normalized = capacity / np.sum(capacity)

    fathers = generate_parent_list(capacity_normalized, n_new, routes)
    mothers = generate_parent_list(capacity_normalized, n_new, routes)

    return fathers, mothers


def mutate_child(child, n_mutations):
    # introduce random mutation (swap two cities)
    child_mutated = child.copy()
    for i in range(n_mutations):
        child_mutated = tour_swap_2_random(idx=None, tour=child)
    return child_mutated


def add_child(children_routes, child, cities):
    child_route = Route(cities=cities)
    child_route.set_tour(child, validation=True)
    children_routes.append(child_route)
    return children_routes


def breed(mothers, fathers, cut_len=5, n_mutations=1):
    children_routes = []

    cities = mothers[0].cities

    for f_route, m_route in zip(fathers, mothers):
        f_tour = f_route.tour.copy()
        m_tour = m_route.tour.copy()
        tour_len = len(f_tour)

        injection = get_random_part_from_father(f_tour, tour_len, cut_len)

        # find anchor in mother, divide to `before` and `after`
        start_idx = m_tour.index(injection[0])

        before, after = divide_route_before_after(m_tour, start_idx)

        # clean-up before and after from elements that are in injection
        before_clean = [city for city in before if city not in injection]
        after_clean = [city for city in after if city not in injection]

        # glue 'before', 'injection', 'after'
        child = before_clean + injection + after_clean

        child_mutated = mutate_child(child, n_mutations)

        children_routes = add_child(children_routes, child, cities)
        children_routes = add_child(children_routes, child_mutated, cities)

    return children_routes


def save_to_file(points_list, img_dir, f_name):
    with open(img_dir + f_name + '.csv', 'w') as out:
        csv_out = csv.writer(out, delimiter=',',
                             lineterminator='\n')
        # csv_out.writerow(['name', 'num'])
        for row in points_list:
            csv_out.writerow(row)


def display_epoch_stats(tl, epoch_idx, avg_adapt_gain, epoch_gain,
                        elapsed_time):
    msg = (f"epoch: {epoch_idx:03d}, " +
           f"min: {min(tl):.2f}, " +
           f"avg: {sum(tl) / len(tl):.2f}, " +
           f"avg adapt gain {avg_adapt_gain:.3f}%, " +
           f"epoch_gain {epoch_gain:.3f}%, " +
           f"took: {elapsed_time:.3f}"
           )
    print(msg)


# DUplicate with natural selection
def keep_n_best_routes(routes_sorted, n_best):
    elite_routes = routes_sorted[0:n_best]
    return elite_routes


def evolve(routes, params):
    dist_log = []

    # sort routes by the len
    routes, routes_len = sort_routes(routes)

    # disp and save status at epoch start
    display_tours_stats(routes_len, epoch_idx=0)
    dist_log.append((0, min(routes_len)))

    n_keep = int(np.round(params['survival_ratio'] * params[
        'initial_population']))

    for epoch in range(1, params['n_epochs']):
        t = time.time()
        min_prev = min(routes_len)
        # apply natural selection
        routes, routes_len = NaturalSelection.keep_n_best(routes_sorted=routes,
                                                          routes_length=routes_len,
                                                          n_keep=n_keep)

        # apply local optimizations
        routes, gains = apply_local_optimizations(routes,
                                                  params['n_three'],
                                                  params['n_two'],
                                                  verbose_lvl=0)
        avg_adapt_gain = sum(gains) / len(gains)

        # create pairs of fathers and mothers
        fathers_routes, mothers_routes = pair_matching(routes,
                                                       n_new=100)

        # prolong life for some of the best routes found so far
        routes = keep_n_best_routes(routes, params['n_best_prolong_life'])

        # reproduction - new generation of routes
        children_routes = breed(fathers_routes, mothers_routes,
                                cut_len=params['cut_length'],
                                n_mutations=params['n_mut'])
        routes.extend(children_routes)

        # sort routes by the len
        routes, routes_len = sort_routes(routes)

        # store epoch info in logs
        dist_log.append((epoch, min(routes_len)))

        save_to_file(dist_log, img_dir='res/', f_name=params['f_name'])

        plot_shortest_route_len_vs_epochs(dist_log,
                                          save_plot=True,
                                          show_plot=True,
                                          img_dir='res/',
                                          f_name=params['f_name'],
                                          epoch=epoch)

        plot_route(routes[0],
                   epoch=epoch,
                   save_plot=True,
                   show_plot=True,
                   img_dir='res/',
                   f_name=params['f_name'],
                   show_labels=False,
                   labels=None)

        min_curr = min(routes_len)
        epoch_gain = 100.0 * (min_prev - min_curr) / min_prev
        display_epoch_stats(
            routes_len,
            epoch_idx=epoch,
            avg_adapt_gain=avg_adapt_gain,
            epoch_gain=epoch_gain,
            elapsed_time=time.time() - t)


def run():
    # config cities and initial population count
    params = {
        'num_cities': 80,
        'n_three': 1,
        'n_two': 1,
        'n_epochs': 300,
        'seed_city': 0,
        'seed_routes': 0,
        'n_best_prolong_life': 4,
    }

    # TODO: make dictionary with parameter sets
    for initial_population in [500]:
        for survival_ratio in [0.1]:
            for cut_length in [5]:
                for n_mut in [1]:
                    params['initial_population'] = initial_population
                    name = f"tsp_len-epoch" + \
                           f"_cit={params['num_cities']}" + \
                           f"_pop={initial_population}" + \
                           f"_trip={params['n_three']}" + \
                           f"_two={params['n_two']}" + \
                           f"_epochs={params['n_epochs']}" + \
                           f"_surv={survival_ratio}" + \
                           f"_cut={cut_length}" + \
                           f"_mut={n_mut}"

                    # generate cities and tours
                    np.random.seed(params['seed_city'])

                    # City - circle
                    # cities_init = CityFactory.circle(r=1,
                    #                                  n_cities=params[
                    #                                      'num_cities'])
                    # City - random
                    cities_init = CityFactory.random(
                        n_cities=params['num_cities'],
                        seed=params['seed_city'])

                    # TODO: save cities
                    routes_init = generate_rand_routes(cities_init,
                                                       n_routes=initial_population,
                                                       seed=params[
                                                           'seed_routes'])
                    params['survival_ratio'] = survival_ratio
                    params['cut_length'] = cut_length
                    params['n_mut'] = n_mut
                    params['f_name'] = name

                    evolve(routes=routes_init, params=params)

                    # Nearest neighbour as reference
                    tour = generate_nearest_neighbour_route(cities_init)
                    route_nn = Route(cities=cities_init)
                    route_nn.set_tour(tour)
                    plot_route(route_nn, show_labels=False, labels=None)
                    a = 0


if __name__ == "__main__":
    run()
