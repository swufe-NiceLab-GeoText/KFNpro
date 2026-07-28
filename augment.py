"""Online text augmentation for adversarial robustness training.

Provides lightweight perturbations (no external model needed) to generate
pseudo-adversarial examples during training. The model learns consistency
between clean and perturbed representations.
"""

import random
import re


def random_word_delete(text, p=0.1):
    """Randomly delete words with probability p."""
    words = text.split()
    if len(words) <= 1:
        return text
    remaining = [w for w in words if random.random() > p]
    if len(remaining) == 0:
        return random.choice(words)
    return ' '.join(remaining)


def random_word_swap(text, n=1):
    """Randomly swap n pairs of adjacent words."""
    words = text.split()
    if len(words) <= 1:
        return text
    for _ in range(n):
        idx = random.randint(0, len(words) - 2)
        words[idx], words[idx + 1] = words[idx + 1], words[idx]
    return ' '.join(words)


def random_char_insert(text, p=0.05):
    """Randomly insert characters (typo simulation)."""
    chars = list(text)
    result = []
    for c in chars:
        result.append(c)
        if random.random() < p and c.isalpha():
            result.append(random.choice('abcdefghijklmnopqrstuvwxyz'))
    return ''.join(result)


def random_char_swap(text, p=0.05):
    """Randomly swap adjacent characters (typo simulation)."""
    chars = list(text)
    for i in range(len(chars) - 1):
        if random.random() < p:
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return ''.join(chars)


def augment_text(text, strategy='mixed', intensity=0.15):
    """Apply random augmentation to a single text.
    
    Args:
        text: input string
        strategy: 'delete' | 'swap' | 'typo' | 'mixed'
        intensity: augmentation intensity (0.0 - 1.0)
    Returns:
        augmented text string
    """
    if strategy == 'delete':
        return random_word_delete(text, p=intensity)
    elif strategy == 'swap':
        n_swaps = max(1, int(len(text.split()) * intensity))
        return random_word_swap(text, n=n_swaps)
    elif strategy == 'typo':
        return random_char_insert(text, p=intensity * 0.3)
    elif strategy == 'mixed':
        # Randomly pick one augmentation type
        aug_type = random.choice(['delete', 'swap', 'typo', 'char_swap'])
        if aug_type == 'delete':
            return random_word_delete(text, p=intensity)
        elif aug_type == 'swap':
            n_swaps = max(1, int(len(text.split()) * intensity))
            return random_word_swap(text, n=n_swaps)
        elif aug_type == 'typo':
            return random_char_insert(text, p=intensity * 0.3)
        else:
            return random_char_swap(text, p=intensity * 0.3)
    return text


def augment_batch(texts, strategy='mixed', intensity=0.15):
    """Augment a batch of texts.
    
    Args:
        texts: list of strings
        strategy: augmentation strategy
        intensity: augmentation intensity
    Returns:
        list of augmented strings
    """
    return [augment_text(t, strategy=strategy, intensity=intensity) for t in texts]
