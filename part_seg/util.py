import numpy as np
import torch

def parameter_number(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def normal2unit(vertices: "(vertice_num, 3)"):
    """
    Return: (vertice_num, 3) => normalized into unit sphere
    """
    center = vertices.mean(dim= 0)
    vertices -= center
    distance = vertices.norm(dim= 1)
    vertices /= distance.max()
    return vertices

def rotate(points, degree: float, axis: int):
    """Rotate along upward direction"""
    rotate_matrix = torch.eye(3)
    theta = (degree/360)*2*np.pi
    cos = np.cos(theta)
    sin = np.sin(theta)
    
    axises = [0, 1, 2]
    assert  axis in axises
    axises.remove(axis)

    rotate_matrix[axises[0], axises[0]] = cos
    rotate_matrix[axises[0], axises[1]] = -sin
    rotate_matrix[axises[1], axises[0]] = sin
    rotate_matrix[axises[1], axises[1]] = cos
    points = points @ rotate_matrix
    return points


def augmentation_transform(points, config, normals=None, verbose=False):
        """Implementation of an augmentation transform for point clouds."""

        ##########
        # Rotation
        ##########

        # Initialize rotation matrix
        R = np.eye(points.shape[1])

        if points.shape[1] == 3:
            if config.augment_rotation == 'vertical':

                # Create random rotations
                theta = np.random.rand() * 2 * np.pi
                c, s = np.cos(theta), np.sin(theta)
                R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

            elif config.augment_rotation == 'all':

                # Choose two random angles for the first vector in polar coordinates
                theta = np.random.rand() * 2 * np.pi
                phi = (np.random.rand() - 0.5) * np.pi

                # Create the first vector in carthesian coordinates
                u = np.array([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])

                # Choose a random rotation angle
                alpha = np.random.rand() * 2 * np.pi

                # Create the rotation matrix with this vector and angle
                R = create_3D_rotations(np.reshape(u, (1, -1)), np.reshape(alpha, (1, -1)))[0]

        R = R.astype(np.float32)

        #######
        # Scale
        #######

        # Choose random scales for each example
        min_s = config.augment_scale_min
        max_s = config.augment_scale_max
        if config.augment_scale_anisotropic:
            scale = np.random.uniform(min_s, max_s, points.shape[1])
        else:
            scale = np.random.uniform(min_s, max_s)

        # Add random symmetries to the scale factor
        symmetries = np.array(config.augment_symmetries).astype(np.int32)
        symmetries *= np.random.randint(2, size=points.shape[1])
        scale = (scale * (1 - symmetries * 2)).astype(np.float32)

        #######
        # Noise
        #######

        noise = (np.random.randn(points.shape[0], points.shape[1]) * config.augment_noise).astype(np.float32)
        noise = np.clip(noise, -1*config.augment_noise_clip, config.augment_noise_clip)

        #######
        # Shift
        #######

        if config.augment_shift:
            shift = np.random.uniform(low=-config.augment_shift, high=config.augment_shift, size=[3]).astype(np.float32)

        ##################
        # Apply transforms
        ##################

        # Do not use np.dot because it is multi-threaded
        #augmented_points = np.dot(points, R) * scale + noise
        augmented_points = np.sum(np.expand_dims(points, 2) * R, axis=1) * scale + noise
        if config.augment_shift:
            augmented_points = np.add(augmented_points, shift)


        if normals is None:
            return augmented_points
        else:
            # Anisotropic scale of the normals thanks to cross product formula
            if config.normal_scale:
                normal_scale = scale[[1, 2, 0]] * scale[[2, 0, 1]]
            else:
                normal_scale = np.ones(points.shape[1])
            augmented_normals = np.dot(normals, R) * normal_scale
            # Renormalise
            augmented_normals *= 1 / (np.linalg.norm(augmented_normals, axis=1, keepdims=True) + 1e-6)

            if verbose:
                test_p = [np.vstack([points, augmented_points])]
                test_n = [np.vstack([normals, augmented_normals])]
                test_l = [np.hstack([points[:, 2]*0, augmented_points[:, 2]*0+1])]
                show_ModelNet_examples(test_p, test_n, test_l)

            return augmented_points, augmented_normals

class PartSegConfig():

    ####################
    # Dataset parameters
    ####################

    # Augmentations (PartSeg)
    augment_scale_anisotropic = True
    augment_symmetries = [False, False, False]
    augment_rotation = 'none'
    augment_scale_min = 0.9
    augment_scale_max = 1.1
    augment_noise = 0.001
    augment_occlusion = 'none'
    augment_shift = 0.2

def test():
    points = np.random.rand(1024,3)
    config = PartSegConfig()
    points = augmentation_transform(points, config)
    print(points.shape)

if __name__ == '__main__':
    test()
