from torchvision import transforms


class ToFloat(object):
    def __init__(self):
        pass

    def __call__(self, image):
        return image.float()


def build_image_transformer(image_size):
    image_transformer = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        ToFloat()
    ])
    return image_transformer