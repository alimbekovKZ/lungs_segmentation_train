import albumentations as A


def transformA(img_size):
    train_transform = A.Compose(
        [
            A.Resize(img_size, img_size, interpolation=1, p=1),
            A.ShiftScaleRotate(
                always_apply=False,
                p=0.1,
                shift_limit=(-0.059, 0.050),
                scale_limit=(-0.1, 0.07),
                rotate_limit=(-29, 29),
                interpolation=0,
                border_mode=1,
                value=(0, 0, 0),
                mask_value=None,
            ),
            A.Blur(always_apply=False, p=0.07, blur_limit=(3, 5)),
            A.GridDistortion(
                always_apply=False,
                p=0.07,
                num_steps=5,
                distort_limit=(-0.3, 0.3),
                interpolation=0,
                border_mode=0,
                value=(0, 0, 0),
                mask_value=None,
            ),
            A.GridDropout(ratio=0.01, p=0.1),
            A.GaussNoise(var_limit=(0, 0.001), p=0.1),
            A.Downscale(p=0.13, scale_min=0.4, scale_max=0.75, interpolation=0),
        ],
        p=1,
    )
    test_transform = A.Compose(
        [A.Resize(img_size, img_size, interpolation=1, p=1)], p=1
    )
    return train_transform, test_transform
