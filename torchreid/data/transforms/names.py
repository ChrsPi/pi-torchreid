"""Transform-name canonicalization helpers."""

from collections.abc import Sequence

LEGACY_TRANSFORM_ALIASES = {
    "random_flip": "RandomHorizontalFlip",
    "color_jitter": "ColorJitter",
}


def canonicalize_transform_name(name: str) -> str:
    """Map legacy names to the canonical transform identifiers."""
    return LEGACY_TRANSFORM_ALIASES.get(name, name)


def canonicalize_transform_list(transforms: str | Sequence[str] | None) -> list[str]:
    """Normalize user-provided transform names into canonical identifiers."""
    if transforms is None:
        return []
    if isinstance(transforms, str):
        transforms = [transforms]
    return [canonicalize_transform_name(str(name)) for name in transforms]
