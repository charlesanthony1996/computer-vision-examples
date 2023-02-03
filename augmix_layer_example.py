import cv2 as cv
import albumentations as A

transform = A.Compose([
    A.RandomCrop(width= 240, height=240),
    A.HorizontalFlip(p =0.5),
    A.RandomBrightnessContrast(p =0.2)
])

print(transform)
# read an image with opencv and convert it to the RGB colorspace

image = cv.imread("/users/charles/desktop/apple.jpg")
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)


# print(image)

# augment an image
transformed = transform(image= image)
transformed_image = transformed["image"]

# print(transformed)


