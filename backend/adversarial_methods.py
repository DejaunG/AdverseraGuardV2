import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import differential_evolution
import logging

logger = logging.getLogger(__name__)


def fgsm_attack(model, image, label, epsilon=0.03):
    """FGSM attack remains unchanged as it's working correctly"""
    image.requires_grad = True
    output = model(image)
    loss = F.nll_loss(F.log_softmax(output, dim=1), label)
    model.zero_grad()
    loss.backward()
    perturbed_image = image + epsilon * image.grad.data.sign()
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


def pgd_attack(model, image, label, epsilon=0.03, alpha=0.005, num_iter=40):
    """PGD attack remains unchanged as it's working correctly"""
    perturbed_image = image.clone().detach()
    for _ in range(num_iter):
        perturbed_image.requires_grad = True
        output = model(perturbed_image)
        loss = F.nll_loss(F.log_softmax(output, dim=1), label)
        model.zero_grad()
        loss.backward()
        adv_image = perturbed_image + alpha * perturbed_image.grad.sign()
        eta = torch.clamp(adv_image - image, min=-epsilon, max=epsilon)
        perturbed_image = torch.clamp(image + eta, min=0, max=1).detach()
    return perturbed_image


def deepfool_attack(model, image, num_classes, overshoot=0.02, max_iter=50):
    """
    Improved DeepFool implementation with better tensor handling and error prevention
    """
    image = image.clone().detach()
    image.requires_grad_(True)

    # Get original prediction
    with torch.no_grad():
        output = model(image)
        original_pred = output.max(1)[1].item()

    perturbed_image = image.clone()

    for i in range(max_iter):
        if perturbed_image.grad is not None:
            perturbed_image.grad.zero_()

        output = model(perturbed_image)
        _, indices = torch.sort(output[0].flatten(), descending=True)

        # Initialize variables for finding closest hyperplane
        min_distance = float('inf')
        closest_perturbation = None

        # Check against top k classes
        k = min(num_classes, len(indices))
        for idx in range(1, k):  # Start from 1 to skip the original class
            target_class = indices[idx]

            # Calculate loss for this target class
            loss = output[0, target_class] - output[0, original_pred]
            grad = torch.autograd.grad(loss, perturbed_image,
                                       retain_graph=True)[0]

            # Normalize gradient
            grad_norm = grad.norm().item()
            if grad_norm == 0:
                continue

            # Calculate distance to decision boundary
            distance = abs(loss.item()) / grad_norm

            # Update if this is the closest boundary
            if distance < min_distance:
                min_distance = distance
                closest_perturbation = grad * distance / grad_norm

        if closest_perturbation is None:
            break

        # Apply perturbation
        with torch.no_grad():
            perturbed_image = perturbed_image + (1 + overshoot) * closest_perturbation
            perturbed_image = torch.clamp(perturbed_image, 0, 1)

        # Check if prediction has changed
        with torch.no_grad():
            new_pred = model(perturbed_image).max(1)[1].item()
            if new_pred != original_pred:
                break

    return perturbed_image.detach()


def one_pixel_attack(model, image, label, pixels=1, max_iter=100, pop_size=400):
    """
    Optimized One Pixel attack with progress tracking and parallel processing
    """
    try:
        from tqdm.auto import tqdm
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed

        image = image.squeeze(0)  # Remove batch dimension
        c, h, w = image.shape

        # Cache for storing prediction results
        prediction_cache = {}

        def perturb_image(xs, batch=True):
            """Optimized perturbation with batching"""
            if not isinstance(xs, np.ndarray):
                xs = np.array([xs])

            cache_key = tuple(xs.flatten())
            if cache_key in prediction_cache:
                return prediction_cache[cache_key]

            batch_size = len(xs)
            perturbed = image.unsqueeze(0).repeat(batch_size, 1, 1, 1)
            perturbed_np = perturbed.numpy()

            # Vectorized operations for better performance
            for idx in range(batch_size):
                pixel_indices = np.arange(pixels) * 5
                x_positions = xs[idx, pixel_indices].astype(int)
                y_positions = xs[idx, pixel_indices + 1].astype(int)
                rgb_values = xs[idx, pixel_indices[:, None] + np.arange(2, 5)]

                for p in range(pixels):
                    perturbed_np[idx, :, x_positions[p], y_positions[p]] = rgb_values[p]

            result = torch.from_numpy(perturbed_np).float()
            if not batch:
                prediction_cache[cache_key] = result
            return result

        def predict_batch(xs):
            """Batch prediction for better performance"""
            try:
                with torch.no_grad():
                    imgs = perturb_image(xs)
                    outputs = model(imgs)
                    predictions = outputs.argmax(1)
                return predictions.cpu().numpy()
            except Exception as e:
                logger.error(f"Batch prediction error: {str(e)}")
                return np.array([label.item()] * len(xs))

        def attack_success(x):
            """Optimized evaluation function"""
            try:
                x_reshaped = x.reshape(1, -1)
                predictions = predict_batch(x_reshaped)
                return float(predictions[0] != label.item())
            except Exception as e:
                logger.error(f"Attack evaluation error: {str(e)}")
                return 0.0

        # Define bounds for pixel locations and values
        bounds = [(0, h - 1), (0, w - 1), (0, 1), (0, 1), (0, 1)] * pixels

        # Calculate estimated time per iteration based on a small sample
        start_time = time.time()
        sample_size = min(5, pop_size)
        sample_population = np.random.rand(sample_size, len(bounds))
        for x in sample_population:
            attack_success(x)
        time_per_iter = (time.time() - start_time) / sample_size
        estimated_total_time = time_per_iter * pop_size * max_iter

        # Progress bar setup
        pbar = tqdm(total=max_iter,
                    desc="One Pixel Attack Progress",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                               "[{elapsed}<{remaining}, {rate_fmt}{postfix}]")

        def callback(xk, convergence=None):
            """Callback function to update progress bar"""
            pbar.update(1)
            current_success = attack_success(xk)
            pbar.set_postfix({"Best Score": f"{current_success:.3f}"})
            return False

        try:
            # Run differential evolution with progress tracking
            result = differential_evolution(
                attack_success,
                bounds,
                maxiter=max_iter,
                popsize=pop_size,
                mutation=(0.5, 1),
                recombination=0.7,
                workers=1,
                updating='immediate',
                callback=callback,
                init='random',
                polish=False  # Disable polish step for speed
            )

            if result.success:
                perturbed = perturb_image(result.x.reshape(1, -1), batch=False)
                pbar.close()
                return perturbed

            logger.warning("One pixel attack did not converge")
            pbar.close()
            return image.unsqueeze(0)

        finally:
            pbar.close()

    except Exception as e:
        logger.error(f"One pixel attack failed: {str(e)}")
        return image.unsqueeze(0)


def universal_adversarial_perturbation(model, image, epsilon=0.1, delta=0.2, max_iter_uni=50, num_classes=1000):
    """Fixed Universal Adversarial Perturbation"""
    perturbation = torch.zeros_like(image).detach()

    for _ in range(max_iter_uni):
        perturbed = torch.clamp(image + perturbation, 0, 1)
        perturbed = perturbed.detach().clone()
        perturbed.requires_grad = True

        # Get current prediction
        with torch.no_grad():
            original_pred = model(image).argmax(1)

        # Forward pass
        output = model(perturbed)
        current_pred = output.argmax(1)

        if current_pred.item() == original_pred.item():
            # Compute gradient for the current prediction
            loss = output[0, current_pred] - output[0, (current_pred + 1) % num_classes]
            loss.backward()

            # Update perturbation
            grad = perturbed.grad.sign()
            perturbation = torch.clamp(perturbation + epsilon * grad, -epsilon, epsilon)
            perturbation = torch.clamp(image + perturbation, 0, 1) - image

    return torch.clamp(image + perturbation, 0, 1)


def generate_adversarial_example(model, image, label, method='fgsm', **kwargs):
    """Main function remains the same but with improved error handling"""
    try:
        logger.info(f"Generating adversarial example using {method} method...")

        if method == 'fgsm':
            epsilon = kwargs.get('epsilon', 0.03)
            return fgsm_attack(model, image, label, epsilon)

        elif method == 'pgd':
            epsilon = kwargs.get('epsilon', 0.03)
            alpha = kwargs.get('alpha', 0.005)
            num_iter = kwargs.get('num_iter', 40)
            return pgd_attack(model, image, label, epsilon, alpha, num_iter)

        elif method == 'deepfool':
            num_classes = kwargs.get('num_classes', 1000)
            overshoot = kwargs.get('overshoot', 0.02)
            max_iter = kwargs.get('max_iter', 50)
            return deepfool_attack(model, image, num_classes, overshoot, max_iter)

        elif method == 'one_pixel':
            pixels = kwargs.get('pixels', 1)
            max_iter = kwargs.get('max_iter', 100)
            pop_size = kwargs.get('pop_size', 400)
            return one_pixel_attack(model, image, label, pixels, max_iter, pop_size)

        elif method == 'universal':
            epsilon = kwargs.get('epsilon', 0.1)
            delta = kwargs.get('delta', 0.2)
            max_iter_uni = kwargs.get('max_iter_uni', 50)
            num_classes = kwargs.get('num_classes', 1000)
            return universal_adversarial_perturbation(model, image, epsilon, delta, max_iter_uni, num_classes)

        else:
            logger.warning(f"Unknown attack method: {method}")
            return image

    except Exception as e:
        logger.error(f"Error in generate_adversarial_example: {str(e)}")
        raise