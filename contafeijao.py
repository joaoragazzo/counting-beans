"""
    UNIFAL - Universidade Federal de Alfenas
    Bacharelado em Ciência da Computação
Trabalho..: Contagem de feijões
Professor.: Luiz Eduardo da Silva
Aluno.....: João Paulo Martyr Ragazzo
Data......: 20/05/2024
"""
import multiprocessing

from colors import Colors
from image import Image
from component import Component
from numbers import Numbers
import sys
from math import sqrt, ceil

"""
    Basic operations
"""


def load_image(path: str) -> Image:
    """
    Load a PGM - P2 image.

    :param path: The path of the image file.
    :return:
    """

    if path[-3:] != "pgm":
        print(Colors.FAIL + "You can use only PGM format." + Colors.ENDC)
        exit(1)

    with open(path) as image:
        if image.readline() != "P2\n":
            print(Colors.FAIL + "Please, insert a valid image file." + Colors.ENDC)
            exit(1)

        metadata: list[str] = []

        line = image.readline()
        while line.startswith("#"):
            metadata.append(line)
            line = image.readline()

        width, height = map(int, line.split())
        max_color = int(image.readline())
        temp_image = image.read().split()
        try:
            image_loaded = [list(map(int, temp_image[i:i + width])) for i in range(0, len(temp_image), width)]
        except TypeError or IndexError:
            print(Colors.FAIL + "Please, insert a valid image file." + Colors.ENDC)
            exit(1)

        for y in range(height):
            for x in range(width):
                if image_loaded[y][x] > max_color:
                    print(Colors.FAIL + "There is a pixel with a invalid value in " + str(x) + ":" + str(y) +
                          " position." + Colors.ENDC)
                    exit(1)

        return Image(image_loaded, max_color, width, height, metadata)


def alloc_blank_matrix(width: int, height: int, background_color: int = 0) -> list[list]:
    """
    Alloc a blank image matrix

    :param width: The width of the image
    :param height: The height of the image
    :param background_color: The background color of the image (default 0)
    :return:
    """

    return [[background_color for _ in range(width)] for _ in range(height)]


def save_image(path: str, image: Image) -> None:
    """
    Save the image PGM P2 image file.

    :param path: The path to be saved.
    :param image: The image to be saved.
    :return:
    """
    with open(path, "w") as image_file:

        image_file.write("P2\n")
        image_file.write(f"{image.width} {image.height}\n")
        image_file.write(f"{image.max_color}\n")
        for y in range(image.height):
            for x in range(image.width):
                pixel = image.loaded[y][x]
                image_file.write(f"{pixel} ")


"""
    Math operations
"""


def cartesian_distance(x, y, x1, y1) -> int:
    """
    Calculate the distance between two points.
    :param x: The x coordinate of the first point.
    :param y: The y coordinate of the first point.
    :param x1: The  x coordinate of the second point.
    :param y1: The y coordinate of the second point.
    :return: The distance between the two points.
    """
    return ceil(sqrt((x1 - x) ** 2 + (y1 - y) ** 2))


"""
    Image manipulation
"""


def thresholding(image: Image, sensitivity: int) -> None:
    """
    Create a black and white image with a white background with black components

    :param image: The image that will be processed
    :param sensitivity: The sensitivity when selecting the components. Higher values means that more shades of black will
    be considered part of components, resulting in bigger components.
    :return:
    """

    threshold = (image.max_color * sensitivity) / 100

    for y in range(image.height):
        for x in range(image.width):
            pixel = image.loaded[y][x]
            if pixel < threshold:
                image.loaded[y][x] = 0
            else:
                image.loaded[y][x] = 255


def erosion(image: Image, interactions: int) -> None:
    """
    This function decrease the border of the beans to increase the
    precision of further bean count.

    :param interactions: The number of interactions
    :param image: The image object to be processed.
    :return:
    """

    for _ in range(interactions):
        image_out = [row[:] for row in image.loaded]
        for y in range(1, image.height - 1):
            for x in range(1, image.width - 1):
                px1, px2, px3 = image.loaded[y - 1][x - 1], image.loaded[y - 1][x], image.loaded[y - 1][x + 1]
                px4, px, px5 = image.loaded[y][x - 1], image.loaded[y][x], image.loaded[y][x + 1]
                px6, px7, px8 = image.loaded[y + 1][x - 1], image.loaded[y + 1][x], image.loaded[y + 1][x + 1]

                if any([px2, px4, px5, px7]):
                    image_out[y][x] = 255
                else:
                    image_out[y][x] = 0
        image.loaded = image_out


def dilation(image: Image, interactions: int) -> None:
    """
    This function increases the border of the beans (objects) in the image.

    :param interactions: The number of interactions
    :param image: The image object to be processed.
    :return:
    """

    for _ in range(interactions):
        image_out = [row[:] for row in image.loaded]
        for y in range(1, image.height - 1):
            for x in range(1, image.width - 1):
                px1, px2, px3 = image.loaded[y - 1][x - 1], image.loaded[y - 1][x], image.loaded[y - 1][x + 1]
                px4, px, px5 = image.loaded[y][x - 1], image.loaded[y][x], image.loaded[y][x + 1]
                px6, px7, px8 = image.loaded[y + 1][x - 1], image.loaded[y + 1][x], image.loaded[y + 1][x + 1]

                if any([px2 == 0, px4 == 0, px5 == 0, px7 == 0]):
                    image_out[y][x] = 0
                else:
                    image_out[y][x] = image.loaded[y][x]

        image.loaded = image_out


def opening(image: Image, erosion_interactions: int, dilation_interactions: int) -> None:
    """
    Applies erosion followed by dilation to improve image quality.

    :param image: The image object to be processed.
    :param erosion_interactions: The number of iterations for the erosion process.
    :param dilation_interactions: The number of iterations for the dilation process.
    :return:
    """

    erosion(image, erosion_interactions)
    dilation(image, dilation_interactions)


def flood_fill(matrix: list[list[int]], x: int, y: int, target: int = 255, replace: int = 0) -> list[tuple[int, int]]:
    """
    Use 4-neighborhood for fill a segment

    :param matrix: The matrix to be filled
    :param x: The x coordinate of the pixel to start the fill
    :param y: The y coordinate of the pixel to start the fill
    :param target: The pixel that will be filled
    :param replace: The value that will replace target
    :return: A list containing all coordinates filled
    """
    if target == replace:
        return []

    rows, cols = len(matrix), len(matrix[0])
    stack = [(x, y)]
    filled = []

    while stack:
        cx, cy = stack.pop()
        if 0 <= cx < cols and 0 <= cy < rows and matrix[cy][cx] == target:
            matrix[cy][cx] = replace
            filled.append((cx, cy))
            stack.append((cx + 1, cy))
            stack.append((cx - 1, cy))
            stack.append((cx, cy + 1))
            stack.append((cx, cy - 1))

    return filled

"""
    Fill all components' white spaces 
"""


def mark_external_area(matrix: list[list[int]], x: int, y: int, visited: list[list[bool]], target: int = 255, replace: int = 0) -> None:
    """
    This function is complementary for filling all components' white spaces. It works by flooding the external
    area from the matrix to another color.

    :param matrix: The matrix that will have the external area marked
    :param x: The x coordinate of the pixel to start the fill
    :param y: The y coordinate of the pixel to start the fill
    :param visited: The visited matrix, used to avoid infinite or unnecessary recursion
    :param target: The pixel value that will be filled
    :param replace: The value that will replace target
    :return: None
    """
    rows, cols = len(matrix), len(matrix[0])
    stack = [(x, y)]

    while stack:
        cx, cy = stack.pop()
        if 0 <= cx < cols and 0 <= cy < rows and matrix[cy][cx] == target and not visited[cy][cx]:
            matrix[cy][cx] = replace
            visited[cy][cx] = True
            # Push neighbors onto stack
            if cx + 1 < cols:
                stack.append((cx + 1, cy))
            if cx - 1 >= 0:
                stack.append((cx - 1, cy))
            if cy + 1 < rows:
                stack.append((cx, cy + 1))
            if cy - 1 >= 0:
                stack.append((cx, cy - 1))


def find_and_fill_components(image: Image):
    """
    Paint all components whitespaces to black

    :param image: The image to have the components filled
    :return:
    """

    visited: list[list[bool]] = [[False for _ in range(image.width)] for _ in range(image.height)]
    mark_external_area(image.loaded, 0, 0, visited, replace=100)

    white_spaces: list[(int, int)] = []

    for y in range(image.height):
        for x in range(image.width):
            if image.loaded[y][x] == 255:
                white_spaces.append((x, y))

    for white_space in white_spaces:
        flood_fill(image.loaded, white_space[0], white_space[1])

    for y in range(image.height):
        for x in range(image.width):
            if image.loaded[y][x] == 100:
                image.loaded[y][x] = 255


"""
    Making the distance matrix
"""


def find_pixel_distance_from_background(matrix: list[list[int]], x: int, y: int, background_color: int) -> float:
    """
    Find a single pixel's cartesian distance from the nearst background pixel.

    :param matrix: The image matrix.
    :param x: The x coordinates of the pixel.
    :param y: The y coordinates of the pixel.
    :param background_color:  The pixel value of the background color.
    :return: The pixel distance from the nearst background pixel.
    """
    if matrix[y][x] == background_color:
        return 0

    background_found = False
    coordinates: list[tuple[int, int]] = []
    des = 1
    while not background_found:
        px1, px2, px3 = matrix[max(0, y - des)][max(0, x - des)], matrix[max(0, y - des)][x], matrix[max(0, y - des)][
            min(len(matrix[0]) - 1, x + des)]
        px4, px5 = matrix[y][max(0, x - des)], matrix[y][min(len(matrix[0]) - 1, x + des)]
        px6, px7, px8 = matrix[min(len(matrix) - 1, y + des)][max(0, x - des)], matrix[min(len(matrix) - 1, y + des)][
            x], matrix[min(len(matrix) - 1, y + des)][min(len(matrix[0]) - 1, x + des)]

        if px1 == background_color:
            coordinates.append((x - des, y - des))

        if px2 == background_color:
            coordinates.append((x, y - des))

        if px3 == background_color:
            coordinates.append((x + des, y - des))

        if px4 == background_color:
            coordinates.append((x - des, y))

        if px5 == background_color:
            coordinates.append((x + des, y))

        if px6 == background_color:
            coordinates.append((x - des, y + des))

        if px7 == background_color:
            coordinates.append((x, y + des))

        if px8 == background_color:
            coordinates.append((x + des, y + des))

        if background_color in [px1, px2, px3, px4, px5, px6, px7, px8]:
            background_found = True

        des += 1

    distances: list[float] = []

    for coordinate in coordinates:
        distance = cartesian_distance(x, y, coordinate[0], coordinate[1])
        distances.append(distance)

    return min(distances)


def find_distance_matrix(image: Image, background_color: int = 255) -> list[list[int]]:
    """
    Find the pixels distance between the background and the target pixel.

    :param image: The image
    :param background_color: The pixel value that represents the background color
    :return: A matrix containing the pixels distance between the background and the target pixel
    """
    matrix = image.loaded
    distance_matrix = alloc_blank_matrix(len(matrix[0]), len(matrix))

    for y in range(image.height):
        for x in range(image.width):
            distance = find_pixel_distance_from_background(matrix, x, y, background_color)
            distance_matrix[y][x] = distance

    return distance_matrix


"""
    Applying selective erosion
"""


def apply_selective_erosion(image: Image, components: list[Component], tolerance: int) -> None:
    """
    Apply selective erosion in a single component based on their distance from the background.

    :param image: The image that will be eroded
    :param components: A list of dict, having all components and their distance from the background
    :param tolerance: The tolerance of the erosion. Higher values means that more pixels will be preserved
    :return:
    """

    new_image: list[list[int]] = alloc_blank_matrix(image.width, image.height, background_color=255)

    for component in components:
        minimum = component.get_n_higher_value(tolerance)
        for (x, y), distance in zip(component.pixels, component.distances):
            if distance >= minimum:
                new_image[y][x] = 0

    image.loaded = new_image


def find_and_separate_components(image: Image, distance_matrix: list[list[int]], background_color: int = 255,
                                 default: int = 0) -> list[Component]:
    """
    Find and label all components

    :param image: The image to label components
    :param distance_matrix: The distance matrix
    :param sensitivity: The sensitivity to selective erosion
    :param background_color: The default background color (default 255)
    :param default: The default pixel value (default 0)
    :return:
    """

    matrix: list[list[int]] = image.loaded
    label = 1

    components: list[Component] = []

    for y in range(image.height):
        for x in range(image.width):
            if matrix[y][x] != background_color and matrix[y][x] == default:
                pixels = flood_fill(matrix, x, y, target=default, replace=label)
                distances: list[int] = []
                for pixel in pixels:
                    distances.append(distance_matrix[pixel[1]][pixel[0]])

                components.append(
                    Component(label, pixels, distances)
                )

                label += 1

    return components


"""
    Marking all components
"""


def place_number_in_image(image: Image, number_matrix: list[list[int]], top_left_y: int, top_left_x: int):
    for y in range(7):
        for x in range(7):
            if number_matrix[y][x] == 1:
                if 0 <= top_left_y + y < image.height and 0 <= top_left_x + x < image.width:
                    image.loaded[top_left_y + y][top_left_x + x] = 0


def mark_components(image: Image, components: list[Component], radius: int) -> None:
    numbers_dict = {
        "0": Numbers.ZERO.value,
        "1": Numbers.ONE.value,
        "2": Numbers.TWO.value,
        "3": Numbers.THREE.value,
        "4": Numbers.FOUR.value,
        "5": Numbers.FIVE.value,
        "6": Numbers.SIX.value,
        "7": Numbers.SEVEN.value,
        "8": Numbers.EIGHT.value,
        "9": Numbers.NINE.value
    }

    for component in components:
        possible_center = component.get_possible_center()

        for y in range(possible_center[1] - radius, possible_center[1] + radius + 1):
            for x in range(possible_center[0] - radius, possible_center[0] + radius + 1):
                if 0 <= y < image.height and 0 <= x < image.width:
                    if (y == possible_center[1] - radius or y == possible_center[1] + radius or
                            x == possible_center[0] - radius or x == possible_center[0] + radius):
                        image.loaded[y][x] = 1

        number_size: int = radius - 9
        for number in str(component.label):
            matrix: list[list[int]] = numbers_dict[number]

            place_number_in_image(image, matrix, possible_center[1] - radius - 8,
                                  possible_center[0] - radius + number_size)
            number_size += 8


"""
    Calibrate the variables
"""


def test_new_settings(path_to_image: str, process_image_var: int, erosion_var: int, selective_erosion_var: int,
                      opening_var_1: int, opening_var_2: int) -> int:
    image = load_image(path_to_image)
    thresholding(image, process_image_var)
    find_and_fill_components(image)
    erosion(image, erosion_var)

    distance_matrix = find_distance_matrix(image)
    components = find_and_separate_components(image, distance_matrix)

    apply_selective_erosion(image, components, selective_erosion_var)
    opening(image, opening_var_1, opening_var_2)

    components = find_and_separate_components(image, distance_matrix)
    try:
        number_of_components = components[len(components) - 1].label
    except IndexError:
        return 0

    return number_of_components


def evaluate_settings(config):
    paths = [
        "./Images/beans1.pgm", "./Images/beans2.pgm", "./Images/beans3.pgm",
        "./Images/beans4.pgm", "./Images/beans5.pgm", "./Images/beans6.pgm",
        "./Images/beans7.pgm"
    ]

    process_image_var, erosion_var, selective_erosion_var, opening_var_1, opening_var_2 = config

    beans_numbers = [
        test_new_settings(path, process_image_var, erosion_var, selective_erosion_var, opening_var_1, opening_var_2)
        for path in paths
    ]

    estimed_results = [41, 68, 96, 123, 10, 20, 30]

    error = sum(abs(estimed_results[i] - beans_numbers[i]) for i in range(7))

    return {
        "config": f"""
        Process Image Var: {process_image_var} 
        Erosion Var: {erosion_var} 
        Selective Erosion Var: {selective_erosion_var} 
        Opening Var (1): {opening_var_1} 
        Opening Var (2): {opening_var_2} 
        """,
        "error": error,
        "found": beans_numbers == estimed_results,
        "results": beans_numbers
    }


def find_best_settings():
    configs = [
        (process_image_var, erosion_var, selective_erosion_var, opening_var_1, opening_var_2)
        for process_image_var in range(15, 40)
        for erosion_var in range(1, 5)
        for selective_erosion_var in range(1, 8)
        for opening_var_1 in range(0, 5)
        for opening_var_2 in range(0, 5)
    ]

    settings = []

    def callback(result):
        settings.append(result)
        print(f"Config: {result['config']}")
        print(f"Results so far -> {result['results']}")
        print(f"Error -> {result['error']}")
        print(f"=============================")

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        for config in configs:
            pool.apply_async(evaluate_settings, args=(config,), callback=callback)

        pool.close()
        pool.join()

    with open('./results.txt', 'w') as f:
        settings_sorted = sorted(settings, key=lambda x: x['error'])
        for setting in settings_sorted:
            f.write(setting["config"])
            f.write(f"Error: ( {setting['error']} )\n")
            f.write("=============================\n")

    print("Process completed. Best settings saved.")


"""
    Main program
"""


def main():
    # find_best_settings() Used just for find the best settings.

    try:
        path_to_image = sys.argv[1]
    except IndexError:
        print(Colors.FAIL.value + "Please, insert the path to the image." + Colors.ENDC.value)
        exit(0)

    image = load_image(path_to_image)
    thresholding(image, 22)
    find_and_fill_components(image)
    erosion(image, 4)

    distance_matrix = find_distance_matrix(image)
    components = find_and_separate_components(image, distance_matrix)

    apply_selective_erosion(image, components, 4)
    opening(image, 1, 2)

    # Final labeling
    components = find_and_separate_components(image, distance_matrix)
    number_of_components = components[len(components) - 1].label

    print("#componentes= " + str(number_of_components))

    # Making final image
    original_image = load_image(path_to_image)

    mark_components(original_image, components, 10)

    save_image("output.pgm", original_image)


if __name__ == "__main__":
    main()
