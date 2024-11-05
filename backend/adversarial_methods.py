import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import differential_evolution

def fgsm_attack(model, image, label, epsilon=0.03):
    image.requires_grad = True
    output = model(image)
    loss = F.nll_loss(F.log_softmax(output, dim=1), label)
    model.zero_grad()
    loss.backward()
    perturbed_image = image + epsilon * image.grad.data.sign()
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def pgd_attack(model, image, label, epsilon=0.03, alpha=0.005, num_iter=40):
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
    image.requires_grad = True
    output = model(image)
    label = output.max(1)[1].item()

    perturbed_image = image.clone()
    for _ in range(max_iter):
        output = model(perturbed_image)
        _, indices = torch.sort(output.data.flatten(), descending=True)

        loss = None
        for k in indices[1:num_classes]:
            zero_loss = (output[0, k] - output[0, label])
            if loss is None:
                loss = zero_loss
            else:
                loss = torch.min(loss, zero_loss)

        model.zero_grad()
        loss.backward(retain_graph=True)

        grad = perturbed_image.grad.data
        perturbation = (abs(loss) + 1e-4) * grad / (grad.norm(p=2) + 1e-4)

        perturbed_image = perturbed_image + (1 + overshoot) * perturbation
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        perturbed_image = perturbed_image.detach()
        perturbed_image.requires_grad = True

        if model(perturbed_image).max(1)[1].item() != label:
            break

    return perturbed_image

def one_pixel_attack(model, image, label, pixels=1, max_iter=100, pop_size=400):
    c, h, w = image.shape

    def perturb_image(xs):
        if xs.ndim < 2:
            xs = np.array([xs])
        batch = len(xs)
        imgs = image.repeat(batch, 1, 1, 1)
        for i in range(batch):
            for j in range(pixels):
                x, y = int(xs[i, j * 5]), int(xs[i, j * 5 + 1])
                imgs[i, 0, x, y] = xs[i, j * 5 + 2]
                imgs[i, 1, x, y] = xs[i, j * 5 + 3]
                imgs[i, 2, x, y] = xs[i, j * 5 + 4]
        return imgs

    def predict_classes(xs):
        imgs = perturb_image(xs)
        with torch.no_grad():
            output = model(imgs)
        return output.argmax(1).cpu().numpy()

    def attack_success(x):
        pred = predict_classes(x)
        return pred != label.item()

    bounds = [(0, h - 1), (0, w - 1), (0, 1), (0, 1), (0, 1)] * pixels
    result = differential_evolution(
        attack_success, bounds, maxiter=max_iter, popsize=pop_size,
        recombination=1, atol=-1, callback=lambda x, convergence: attack_success(x)
    )

    perturbed_image = perturb_image(result.x)
    return perturbed_image[0]

def universal_adversarial_perturbation(model, dataset, epsilon=0.1, delta=0.2, max_iter_uni=50, max_iter_df=100, num_classes=1000):
    n_samples = len(dataset)
    v = torch.zeros_like(next(iter(dataset))[0])
    fooling_rate = 0.0

    for _ in range(max_iter_uni):
        np.random.shuffle(dataset)
        for sample, _ in dataset:
            if fooling_rate >= 1 - delta:
                break

            perturbed_sample = torch.clamp(sample + v, 0, 1)
            with torch.no_grad():
                original_pred = model(sample.unsqueeze(0)).argmax().item()
                perturbed_pred = model(perturbed_sample.unsqueeze(0)).argmax().item()

            if original_pred == perturbed_pred:
                dr = deepfool_attack(model, perturbed_sample.unsqueeze(0), num_classes, max_iter=max_iter_df).squeeze(0) - sample
                v = torch.clamp(v + dr, -epsilon, epsilon)

        fooling_rate = compute_fooling_rate(model, dataset, v)

    return v

def compute_fooling_rate(model, dataset, perturbation):
    fooled = 0
    for sample, _ in dataset:
        perturbed_sample = torch.clamp(sample + perturbation, 0, 1)
        with torch.no_grad():
            original_pred = model(sample.unsqueeze(0)).argmax().item()
            perturbed_pred = model(perturbed_sample.unsqueeze(0)).argmax().item()
        if original_pred != perturbed_pred:
            fooled += 1
    return fooled / len(dataset)

def generate_adversarial_example(model, image, label, method='fgsm', **kwargs):
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
        return one_pixel_attack(model, image.squeeze(0), label, pixels, max_iter, pop_size)
    elif method == 'universal':
        dataset = kwargs.get('dataset')
        epsilon = kwargs.get('epsilon', 0.1)
        delta = kwargs.get('delta', 0.2)
        max_iter_uni = kwargs.get('max_iter_uni', 50)
        max_iter_df = kwargs.get('max_iter_df', 100)
        num_classes = kwargs.get('num_classes', 1000)
        perturbation = universal_adversarial_perturbation(model, dataset, epsilon, delta, max_iter_uni, max_iter_df, num_classes)
        return torch.clamp(image + perturbation, 0, 1)
    else:
        return image  # No attack