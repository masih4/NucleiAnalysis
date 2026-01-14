import cv2
import torchstain
import os
from torchvision import transforms
import matplotlib.pyplot as plt

def multitarget_macenko(multi_normalizer):
    size = 1024

    # setup preprocessing and preprocess image to be normalized
    T = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255)
    ])

    # initialize normalizers for each backend and fit to target image
    # single_normalizer = torchstain.normalizers.MacenkoNormalizer(backend="torch")
    # single_normalizer.fit(target)
    pth = '/home/ntorbati/STORAGE/NucleiAnalysis/tif/stain_normalization/'
    ims = [im for im in os.listdir(pth)]
    imgs = []


    for im in ims:
        imgs.append(T(cv2.resize(cv2.cvtColor(cv2.imread(os.path.join(pth, im),), cv2.COLOR_BGR2RGB), (size, size))))
    multi_normalizer.fit(imgs)



    return multi_normalizer

if __name__ == '__main__':
    T = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255)
    ])
    multi_normalizer = torchstain.normalizers.MultiMacenkoNormalizer(backend="torch", norm_mode="avg-post")
    multi_normalizer = multitarget_macenko(multi_normalizer)
    im = '/home/ntorbati/STORAGE/NucleiAnalysis/tif/10. nuinsseg/train/human_bladder_01.tif'

    im1 = T(cv2.cvtColor(cv2.imread(im), cv2.COLOR_BGR2RGB))
    # transform
    result_multi, _, _ = multi_normalizer.normalize(I=im1, stains=True)

    # convert to numpy and set dtype
    result_multi = result_multi.numpy().astype("float32") / 255.
    plt.imshow(result_multi)
    plt.show()