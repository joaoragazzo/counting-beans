class Component:
    def __init__(self, label: int, pixels: list[tuple[int, int]], distances: list[int]):
        self.pixels: list[tuple[int, int]] = pixels
        self.distances: list[int] = distances
        self.label: int = label

    def get_possible_center(self):
        return self.pixels[round((len(self.pixels) / 2) - 1)]

    def get_n_higher_value(self, n: int):
        max_distances_sorted: list = list(set(sorted(self.distances)))
        return max_distances_sorted[-n:][0]
