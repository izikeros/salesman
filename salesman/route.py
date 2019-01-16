import random
import matplotlib.pyplot as plt
import numpy as np


def tour_swap_2_next(idx, tour):
    tour[idx - 1], tour[idx] = tour[idx], tour[idx - 1]
    return tour


def tour_swap_2_random(idx=None, tour=None):
    idx_1, idx_2 = random.sample(range(len(tour)), 2)
    tour[idx_2], tour[idx_1] = tour[idx_1], tour[idx_2]
    return tour


def tour_reverse_triple(idx, tour):
    prev = idx - 1
    nxt = int((idx + 1) % len(tour))
    try:
        tour[prev], tour[nxt] = tour[nxt], tour[prev]
    except:
        a = 0
    return tour


def tour_compare(tour_1, tour_2):
    for i, c1 in enumerate(tour_1):
        if c1 != tour_2[i]:
            print(f'i: {i}, {c1}-{tour_2[i]}')


def generate_rand_routes(cities, n_routes, seed=0):
    new_routes = []
    random.seed(seed)
    for i in range(n_routes):
        r = Route(cities)
        r.generate()
        new_routes.append(r)
    return new_routes


def generate_nearest_neighbour_route(cities):
    n_cities = cities['xy'].shape[0]
    dist = cities['dist']

    # correct diagonal
    d_val = 10000
    for i in range(n_cities):
        dist[i][i] = d_val

    idx = 0
    tour = [idx]
    cities_left = n_cities - 1
    while cities_left > 0:
        distances_from_this_citi = list(dist[idx].values())
        idx_new = np.argmin(distances_from_this_citi)

        # add closest city to the tour list
        tour.append(idx_new)

        # prevent comming back to this city
        dist[idx_new][idx] = d_val

        idx = idx_new
        cities_left -= 1
    return tour


def disp_tour_len_gain(len_best, len_tmp, len_0, prefix=''):
    gain = 100 * (len_best - len_tmp) / len_best
    print(f'{prefix} gain: {gain:.2f} %, '
          f'this: {len_tmp:.3f} '
          f'prev: {len_best:.3f}, '
          f'orig: {len_0:.3f}'
          )


class Route:
    def __init__(self, cities=None):
        self.tour = None
        self.cities = cities  # dict with keys: xy (n_cit x 2 np array), dist
        self.len = None
        self.len_up_to_date = None
        self.fitness = None
        self.fitness_up_to_date = None

    def get_num_cities(self):
        n_cities = None
        if self.cities is not None:
            n_cities = self.cities['xy'].shape[0]
        return n_cities

    def set_tour(self, tour_updated, verbose=False, validation=False):
        if verbose:
            l1 = self.get_length()
            l2 = self.calc_length(tour_updated)
            print(f'replacing route with '
                  f'len: {l1:.3f} by route with '
                  f'len: {l2:.3f}')

        validated = True
        if validation:
            validated = False
            tour_sort = tour_updated.copy()
            tour_sort.sort()
            cities_list = list(range(self.get_num_cities()))
            if tour_sort == cities_list:
                validated = True
            else:
                print(tour_sort)
                print(cities_list)
                raise ValueError(f"List of cities is not complete")

        if validated:
            self.tour = tour_updated
        self.len_up_to_date = False

    def generate(self):
        n_cities = self.cities['xy'].shape[0]
        t = random.sample(range(n_cities), n_cities)
        self.tour = t

    def calc_length(self, city_list=None):
        """calculate length of the tour

        - if no specific tour provided as input - calculate len of the tour
            in the object
        - if tour provided - calculate len of provided tour
        """
        length = 0.0
        dist = self.cities['dist']
        if not city_list:
            city_list = self.tour

        n_cities = len(city_list)

        for i in range(1, n_cities):
            c1 = city_list[i - 1]
            c2 = city_list[i]
            length += dist[c1][c2]

        # add length of the segment for returning from last city to start point
        c1 = city_list[n_cities - 1]
        c2 = city_list[0]
        length += dist[c1][c2]
        return length

    def get_length(self):
        """return route len. Recalculate if needed, store in object"""
        if self.len_up_to_date:
            return self.len
        else:
            self.len = self.calc_length()
            self.len_up_to_date = True
        return self.len

    def improve(self, method, prefix='', verbose=False):
        """swap neighbouring cities in the tour for each pair in the tour

                If the change increases fitness keep the change,
                if the change is not improving fitness don't apply the change

                return tour(list of cities) and tour len
                """
        # experimental tour to be compared with original
        tour_tmp = self.tour.copy()
        tour_best = self.tour.copy()
        len_best = self.get_length()
        len_0 = self.get_length()
        if verbose:
            print('')
        for idx, city in enumerate(tour_best):
            # swap elements with provided improvement method
            tour_tmp = method(idx, tour_tmp)
            len_tmp = self.calc_length(tour_tmp)
            if len_tmp < len_best:
                if verbose:
                    disp_tour_len_gain(len_best, len_best, len_0, prefix)
                tour_best = tour_tmp.copy()
                len_best = len_tmp
        return tour_best, len_best


def plot_tours_len_histogram(tl):
    plt.hist(tl, 50)
    plt.show()


def plot_shortest_route_len_vs_epochs(points_list,
                                      epoch,
                                      save_plot=True,
                                      show_plot=False,
                                      img_dir=None,
                                      f_name=None
                                      ):
    x, y = zip(*points_list)
    if save_plot or show_plot:
        plt.clf()
        plt.step(x, y)
        plt.xlabel('epoch')
        plt.ylabel('length of shortest route found')
        plt.ylim(bottom=0)

        # TODO: better path composition
        if save_plot:
            plt.savefig(img_dir + "len-epoch_" + f_name + f"_{epoch:04d}" +
                        '.png', dpi=90)

        if show_plot:
            plt.draw()
            plt.pause(0.1)


def plot_route(route,
               epoch=0,
               save_plot=False,
               show_plot=True,
               img_dir='res/',
               f_name=None,
               show_labels=False,
               labels=None):
    x = route.cities['xy'][route.tour, 0]
    y = route.cities['xy'][route.tour, 1]

    colors = (0, 0, 0)
    area = 100
    plt.clf()
    plt.scatter(x, y, s=area, c=colors, alpha=0.5)
    plt.plot(x, y)

    if show_labels:
        for label, x, y in zip(labels, x, y):
            plt.annotate(
                label,
                xy=(x, y), xytext=(-20, 20),
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow',
                          alpha=0.5),
                arrowprops=dict(arrowstyle='->',
                                connectionstyle='arc3,rad=0'))

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Tour len: {route.get_length():.2f}, epoch:{epoch:03d}')
    if save_plot:
        plt.savefig(img_dir + "route_" + f_name + f"_{epoch:04d}" +
                    '.png', dpi=90)
    if show_plot:
        plt.draw()
        plt.pause(0.1)


def get_random_part_from_father(f_tour, tour_len, cut_len):
    # get random part from father
    start_idx = int(np.random.randint(0, tour_len, 1))
    injection = []
    for offset in range(start_idx, start_idx + cut_len):
        idx = offset % tour_len
        injection.append(f_tour[idx])
    return injection


def calculate_capacity(routes_sorted):
    """calculate strength/capacity to face challenge

     capacity is in range 0-1
     1 - normalized rank

     :returns capacity for items in order as in routes
     """

    n_routes = len(routes_sorted)
    capacity = np.arange(n_routes)
    capacity = 1 - (capacity / n_routes)

    return capacity


def sort_routes(routes_in, routes_length=None):
    """sort random routes by length in ascending order"""

    # calculate tours len if not provided
    if not routes_length:
        routes_length = get_routes_len(routes_in)

    # sort routes
    routes_sorted_idx = np.argsort(routes_length)

    routes_in = [routes_in[idx] for idx in routes_sorted_idx]
    routes_len = [routes_length[idx] for idx in routes_sorted_idx]

    return routes_in, routes_len


def display_tours_stats(tl, epoch_idx):
    print(f"epoch: {epoch_idx}, num: {len(tl)}, min: {min(tl):.2f}, ",
          f"avg: {sum(tl) / len(tl):.2f}")


def divide_route_before_after(tour, start_idx):
    # divide mother on 'before' and 'after' anchor parts)
    before = []
    if start_idx > 0:
        before = tour[0:start_idx]

    after = []
    if start_idx < len(tour) - 1:
        for offset in range(start_idx + 1, len(tour)):
            idx = offset % len(tour)
            after.append(tour[idx])
    return before, after


def apply_local_optimizations(tours_list, n_3_passes, n_2_passes,
                              verbose_lvl=0):
    # verbosity levels
    lvl_overall_change = 1
    lvl_replace = 2
    lvl_improve = 3

    # apply local optimizations
    gains_percent = []
    for tour in tours_list:
        len_orig = tour.get_length()
        len_new = len_orig
        for i3 in range(n_3_passes):
            tour_new, len_new = tour.improve(
                method=tour_reverse_triple,
                prefix=f'3-{i3}',
                verbose=verbose_lvl >= lvl_improve,
            )
            tour.set_tour(tour_new, verbose=verbose_lvl >= lvl_replace)

            for i2 in range(n_2_passes):
                tour_new, len_new = tour.improve(
                    method=tour_swap_2_next,
                    prefix=f'2-{i2}',
                    verbose=verbose_lvl >= lvl_improve,
                )
                tour.set_tour(tour_new, verbose=verbose_lvl >= lvl_replace)

        # display results
        gain = 100 * (len_orig - len_new) / len_orig
        gains_percent.append(gain)

        if verbose_lvl >= lvl_overall_change:
            print(f'{len_orig:.3f}, -> {len_new:.3f}')

    return tours_list, gains_percent


def get_routes_len(routes_list):
    """return list of length of routes"""
    routes_length = []
    for i, this_route in enumerate(routes_list):
        routes_length.append(this_route.get_length())
    return routes_length
