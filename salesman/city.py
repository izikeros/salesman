import numpy as np
import matplotlib.pyplot as plt


def cities_on_circle(r: float, n_cities: int):
    pi = np.pi
    xy = np.zeros((n_cities, 2))
    x = np.linspace(0, 2 * pi, n_cities + 1)
    x = x[0:n_cities]

    xy[:, 0] = np.cos(x) * r
    xy[:, 1] = np.sin(x) * r
    return xy


def cities_on_line(n_cities: int) -> np.ndarray:
    xy = np.zeros((n_cities, 2))
    xy[:, 0] = np.linspace(0, 1, n_cities)
    return xy


def cities_random(n_cities: int) -> np.ndarray:
    xy = np.random.rand(n_cities, 2)
    return xy


def gen_distance_lookup(xy: np.ndarray) -> dict:
    dist = {}
    for idx1 in range(xy.shape[0]):
        x1 = xy[idx1][0]
        y1 = xy[idx1][1]
        dist[idx1] = {}
        for idx2 in range(xy.shape[0]):
            x2 = xy[idx2][0]
            y2 = xy[idx2][1]
            d = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            dist[idx1][idx2] = d
    return dist


class CityFactory:
    @staticmethod
    def custom(xy: np.ndarray) -> dict:
        dist = gen_distance_lookup(xy)
        cities = {'xy': xy, 'dist': dist}
        return cities

    @staticmethod
    def line(n_cities: int) -> dict:
        xy = cities_on_line(n_cities)
        dist = gen_distance_lookup(xy)
        cities = {'xy': xy, 'dist': dist}
        return cities

    @staticmethod
    def circle(r: float, n_cities: int) -> dict:
        xy = cities_on_circle(r=r, n_cities=n_cities)
        dist = gen_distance_lookup(xy)
        cities = {'xy': xy, 'dist': dist}
        return cities

    @staticmethod
    def random(n_cities: int, seed=0) -> dict:
        np.random.seed(seed)
        xy = cities_random(n_cities)
        dist = gen_distance_lookup(xy)
        cities = {'xy': xy, 'dist': dist}
        return cities


def get_cities_labels(n_cities) -> list:
    """generate labels to differentiate cities on the plot"""
    city_labels = ['{0}'.format(i) for i in range(n_cities)]
    return city_labels


def plot_cities(xy, route_labels=None, show_labels=False):
    colors = (0, 0, 0)
    area = 10
    plt.scatter(xy[:, 0], xy[:, 1], s=area, c=colors, alpha=0.5)
    plt.title('Cities')

    if show_labels:
        for label, x, y in zip(route_labels, xy[:, 0], xy[:, 1]):
            plt.annotate(
                label,
                xy=(x, y), xytext=(-20, 20),
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
