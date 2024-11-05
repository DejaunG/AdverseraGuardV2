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
    """Fixed DeepFool attack"""
    image = image.detach().clone()
    perturbed_image = image.detach().clone()
    perturbed_image.requires_grad = True

    output = model(perturbed_image)
    original_pred = output.max(1)[1].item()

    for _ in range(max_iter):
        if perturbed_image.grad is not None:
            perturbed_image.grad.zero_()

        output = model(perturbed_image)
        _, indices = torch.sort(output[0].flatten(), descending=True)

        # Initialize variables for finding closest hyperplane
        min_distance = float('inf')
        closest_perturbation = None

        for k in indices[1:num_classes]:
            # Zero gradients
            if perturbed_image.grad is not None:
                perturbed_image.grad.zero_()

            # Compute loss for current class
            loss = output[0, k] - output[0, original_pred]
            loss.backward(retain_graph=True)

            # Get current gradient
            current_grad = perturbed_image.grad.clone()

            # Compute distance to decision boundary
            w_norm = current_grad.norm().item()
            if w_norm == 0:
                continue
            distance = abs(loss.item()) / w_norm

            # Update if this is the closest hyperplane
            if distance < min_distance:
                min_distance = distance
                closest_perturbation = current_grad * (distance / w_norm)

        if closest_perturbation is None:
            break

        # Apply perturbation
        with torch.no_grad():
            perturbed_image += (1 + overshoot) * closest_perturbation
            perturbed_image = torch.clamp(perturbed_image, 0, 1)

        perturbed_image.requires_grad = True

        # Check if prediction changed
        output = model(perturbed_image)
        if output.max(1)[1].item() != original_pred:
            break

    return perturbed_image.detach()


def one_pixel_attack(model, image, label, pixels=1, max_iter=100, pop_size=400):
    """Fixed One Pixel attack"""
    c, h, w = image.squeeze().shape

    def perturb_image(xs):
        xs = np.array(xs).reshape(-1, pixels * 5)
        batch = len(xs)
        images = image.repeat(batch, 1, 1, 1)

        for i in range(batch):
            for p in range(pixels):
                x_pos = int(xs[i, p * 5])
                y_pos = int(xs[i, p * 5 + 1])
                for c_idx in range(3):
                    images[i, c_idx, x_pos, y_pos] = xs[i, p * 5 + 2 + c_idx]

        return images

    def predict_classes(xs):
        imgs = perturb_image(xs)
        with torch.no_grad():
            output = model(imgs)
        return output.argmax(1).cpu().numpy()

    def attack_success(x):
        pred = predict_classes(x)
        return (pred != label.item()).any()

    bounds = [(0, h - 1), (0, w - 1), (0, 1), (0, 1), (0, 1)] * pixels

    result = differential_evolution(
        attack_success, bounds, maxiter=max_iter, popsize=pop_size,
        mutation=(0.5, 1), recombination=0.7
    )

    if result.success:
        return perturb_image(result.x)[0]
    return image


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