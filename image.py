class Image:
    def __init__(self, image, max_color, width, height, metadata):
        self.max_color: int = max_color
        self.loaded: list[list[int]] = image
        self.width: int = width
        self.height: int = height
        self.metadata: list[str] = metadata
