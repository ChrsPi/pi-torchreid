"""Transform-name canonicalization helpers."""

from collections.abc import Sequence

LEGACY_TRANSFORM_ALIASES = {
    "random_flip": "RandomHorizontalFlip",
    "color_jitter": "ColorJitter",
}

_CANONICAL_TO_LEGACY_ALIASES: dict[str, tuple[str, ...]] = {}
for legacy_name, canonical_name in LEGACY_TRANSFORM_ALIASES.items():
    aliases = _CANONICAL_TO_LEGACY_ALIASES.setdefault(canonical_name, ())
    _CANONICAL_TO_LEGACY_ALIASES[canonical_name] = (*aliases, legacy_name)


def canonicalize_transform_name(name: str) -> str:
    """Map legacy names to the canonical transform identifiers."""
    return LEGACY_TRANSFORM_ALIASES.get(name, name)


def get_transform_config_keys(name: str) -> tuple[str, ...]:
    """Return config subtree keys to try for a transform, canonical first."""
    canonical_name = canonicalize_transform_name(name)
    legacy_aliases = _CANONICAL_TO_LEGACY_ALIASES.get(canonical_name, ())
    return (canonical_name, *legacy_aliases)


def canonicalize_transform_list(transforms: str | Sequence[str] | None) -> list[str]:
    """Normalize user-provided transform names into canonical identifiers."""
    if transforms is None:
        return []
    if isinstance(transforms, str):
        transforms = [transforms]
    return [canonicalize_transform_name(str(name)) for name in transforms]
